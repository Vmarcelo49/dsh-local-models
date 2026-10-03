/**
 * dsh-local-models — node half.
 *
 * Owns one llama-server child process spawned from the Local Models settings
 * tab, polled until /health on 127.0.0.1:$PORT comes up, stopped from the
 * same tab, and killed when this fiber disposes (graceful dsh shutdown).
 *
 * HTTP API mounted on the dsh webserver under /local-models:
 *   GET   /local-models/browse?dir=PATH → dirs + .gguf files
 *   POST  /local-models/gguf-meta       → {path} → GGUF header parse (cached)
 *   GET   /local-models/status          → state + fresh /health probe
 *   GET   /local-models/logs?offset=&max= → tail of llama-server.log (terminal)
 *   POST  /local-models/run             → {path, ctx, mtp, kv} → spawn llama-server
 *   POST  /local-models/stop            → stop the child (or reap the port)
 *   POST  /local-models/settings        → {autostartRouter, autoUnloadMins,
 *                                          binPath, shortcuts} → settings.json
 *   POST  /local-models/runtime/check   → {binPath} → resolve + `<bin> --version`
 *   POST  /local-models/register        → add the ready server as an llm-pi-ai
 *                                          provider route via the settings service
 *
 * The llama-server binary is resolved per launch from an ordered candidate
 * chain (Runtime setting → LOCAL_MODELS_BIN → PATH → usual dirs), so pointing
 * the tab at a new build needs no dsh restart. See binCandidates().
 *
 * GGUF parsing is a port of the header-only parser from the user's
 * vram-calculator project (src/gguf.ts) over Node fs instead of Blob slices.
 */
import { spawn, execSync, execFile } from "node:child_process";
import {
	accessSync,
	appendFileSync,
	constants as fsConstants,
	existsSync,
	mkdirSync,
	openSync,
	readFileSync,
	readSync,
	readdirSync,
	renameSync,
	statSync,
	writeFileSync,
	closeSync,
} from "node:fs";
import { basename, join, dirname, resolve, sep, delimiter, isAbsolute } from "node:path";
import { fileURLToPath } from "node:url";
import { homedir } from "node:os";
import process from "node:process";

export const inject = ["settings", "credentials", "webServer"];

const PORT_RAW = Number(process.env.LOCAL_MODELS_PORT ?? 8080);
const PORT = Number.isFinite(PORT_RAW) && PORT_RAW > 0 && PORT_RAW < 65536
	? Math.floor(PORT_RAW)
	: 8080;
const BIN_NAME = process.platform === "win32" ? "llama-server.exe" : "llama-server";
/** LOCAL_MODELS_BIN — the pre-UI way to point at the binary. Still honored,
 * but the Runtime setting (settings.json `binPath`) wins over it. */
const BIN_ENV = process.env.LOCAL_MODELS_BIN ?? "";
/** File-browser shortcut dirs from the environment (": "-separated, "name=path"
 * entries allowed). The Runtime card's folder list overrides it; this stays as
 * the fallback for installs that never opened the tab. */
const SHORTCUTS_ENV = parseShortcutSpec(process.env.LOCAL_MODELS_SHORTCUTS ?? "");
const HEALTH_TIMEOUT_MS = 1000;
const START_TIMEOUT_S = 300;
/** Reasoning-effort levels selectable in the Local Models tab.
 * Probe-validated against this model's chat template: template accepts only
 * low / medium / xhigh as reasoning_effort values ("none" disables thinking,
 * launch --reasoning off disables at startup); high / minimal / max raise a
 * Jinja error (HTTP 500) and are therefore excluded. */
const THINKING_LEVELS = ["off", "low", "medium", "xhigh"];
/** Image-payload guards declared on vision registrations (env-overridable). */
const IMAGE_MAX_BYTES = Number(process.env.LOCAL_MODELS_MAX_IMAGE_BYTES ?? 10 * 1024 * 1024);
const IMAGE_PIXEL_BUDGET = Number(process.env.LOCAL_MODELS_IMAGE_PIXEL_BUDGET ?? 4_194_304);
/** Router-mode cap (env-overridable, LOCAL_MODELS_ROUTER_MAX). Default 1:
 * the VRAM policy is "always unload the active model before the requested one
 * loads" — the router queues the request and unloads the idle LRU model first
 * (sequential, never two resident models). */
const ROUTER_MAX = Number(process.env.LOCAL_MODELS_ROUTER_MAX ?? 1);
const ROUTER_PRESETS_FILE = join(dataDir(), "router-presets.ini");
/** llama-server stdout+stderr sink; the terminal tail route serves this file. */
const LOG_PATH = join(dataDir(), "llama-server.log");
/** Run the mmproj (vision projector) on the CPU - weights in system RAM -
 * freeing ~0.87 GiB VRAM from the daily envelope. Encode on CPU takes
 * ~15-22 s per image. Set LOCAL_MODELS_MMPROJ_CPU=0 to keep vision on the GPU. */
const MMPROJ_CPU = process.env.LOCAL_MODELS_MMPROJ_CPU !== "0";
/**
 * Max output advertised when a server is registered in dsh: 32K tokens,
 * matching the llm-pi-ai adapter default. This matters beyond truncation:
 * the advertised maxTokens becomes the adapter's defaultMaxTokens, which
 * dsh-compaction-basic uses as its reserved output `O` in `W - O - headroom`.
 * Advertising the whole window (the old 131K) leaves no message budget at
 * `W = O` — proactive compaction warns once and never fires.
 *
 * NOTE — the 32K cap alone does NOT buy a late threshold under the core
 * defaults (headroomTokens 65536, thresholdRatio 0.8, retainRatio 0.16; see
 * compactionBudgetFor() below). For a 131072 window with O = 32768 the
 * effective threshold is min(0.8 * W, W - O - headroom) = 32768, i.e. the
 * session starts compacting at ~32-38K pressure tokens (the meter prices
 * tools + system on top of the surface, so the visible trigger lands a few K
 * above the raw threshold) — and with xhigh + preserveThinking every turn
 * adds 10-20K thinking tokens, so compaction fires again on the next step:
 * the "compaction loop". The honest fix for mid-size windows is a per-model
 * headroom override on the agent-preset row that mounts the running engine
 * (`preset-standard` → `plugins` → `compaction` group → `compaction-basic`;
 * headroom ~6-16K moves the 131K threshold to ~82-92K) — a patch on the
 * top-level `compaction-basic` row is dead config (that row is disabled),
 * and a smaller O would truncate the xhigh thinking blocks this cap
 * exists to protect. Heavy xhigh thinking that needs more must raise
 * per-request maxTokens explicitly (which honestly moves the compaction
 * threshold earlier instead of silently disabling it).
 */
const MAX_OUTPUT_TOKENS = 32768;
/** Default output share of the window: never offer more than half as max
 * output, so a message budget always survives (`W - O > 0`). */
function defaultMaxOutput(contextWindow) {
	return Math.max(1, Math.min(MAX_OUTPUT_TOKENS, Math.floor(contextWindow / 2)));
}
/**
 * dsh-compaction-basic defaults this plugin's advertisement is priced
 * against. Duplicated here (not imported) because the core package is not a
 * dependency of this plugin — keep in sync with upstream's resolveConfig().
 */
export const COMPACTION_DEFAULTS = {
	thresholdRatio: 0.8,
	headroomTokens: 65536,
	retainRatio: 0.16,
};
/**
 * Price one advertised window the way dsh-compaction-basic's
 * resolveCompactSpec() does, so the tab/skill can tell the user WHEN
 * proactive compaction will actually fire — before they hit the loop.
 *
 *   messageBudget = W - O            (history available to messages)
 *   pressure      = messageBudget - headroom
 *   threshold     = min(floor(W * ratio), pressure)   (fire at/above this)
 *   retain        = floor(messageBudget * retainRatio) (tail kept per compact)
 *
 * `viable` is false when the pressure budget is <= 0: proactive compaction
 * is then disabled entirely (core warns once and only recovers on overflow).
 * That is the 96K-window cliff: W = 98304, O = 32768 leaves exactly 0.
 *
 * `recommendedHeadroom` is the headroom override that lands the threshold at
 * ~70% of the window (clamped to [4096, 65536] so a summary + one retry
 * always fit): the value to put in a `modelPolicies` entry for this route.
 * Pure; `maxTokens` defaults to what buildProviderProfile would advertise.
 */
export function compactionBudgetFor(contextWindow, opts = {}) {
	const W = Number.isInteger(contextWindow) && contextWindow > 0 ? contextWindow : 8192;
	const O = Number.isInteger(opts.maxTokens) && opts.maxTokens > 0
		? opts.maxTokens
		: defaultMaxOutput(W);
	const ratio = typeof opts.thresholdRatio === "number" && opts.thresholdRatio > 0 && opts.thresholdRatio <= 1
		? opts.thresholdRatio
		: COMPACTION_DEFAULTS.thresholdRatio;
	const headroom = Number.isInteger(opts.headroomTokens) && opts.headroomTokens >= 0
		? opts.headroomTokens
		: COMPACTION_DEFAULTS.headroomTokens;
	const retainRatio = typeof opts.retainRatio === "number" && opts.retainRatio > 0 && opts.retainRatio < ratio
		? opts.retainRatio
		: COMPACTION_DEFAULTS.retainRatio;
	const messageBudget = W - O;
	const pressureBudget = messageBudget - headroom;
	const thresholdTokens = Math.floor(Math.min(W * ratio, pressureBudget));
	const retainTokens = Math.floor(messageBudget * retainRatio);
	// Headroom that would put the threshold at ~70% of the window: the
	// pressure budget must cover 0.7 * W, i.e. headroom <= message - 0.7W.
	const recommendedHeadroom = Math.max(
		4096,
		Math.min(COMPACTION_DEFAULTS.headroomTokens, messageBudget - Math.floor(W * 0.7)),
	);
	return {
		contextWindow: W,
		maxTokens: O,
		headroomTokens: headroom,
		messageBudget,
		pressureBudget,
		thresholdTokens,
		retainTokens,
		viable: pressureBudget > 0 && retainTokens < thresholdTokens,
		recommendedHeadroom,
	};
}
/** Fixed-MTP ceiling. The tab offers 0-7 and the API clamps here: upstream
 * accepts any `--spec-draft-n-max` and clamps the effective depth to the
 * model's own nextn depth at load. Depth 3 is the measured 16 GiB sweet spot
 * (deeper drafts cost VRAM and collapse decode at large ctx — see
 * bench/mtp_tuning.md), but the whole range stays selectable. */
const MTP_MAX = 7;
/** Draft acceptance floor for fixed MTP. Upstream's own default is 0.00
 * (ungated). Gating is workload-dependent, not a free win: it *raises*
 * acceptance while lowering throughput whenever the drafter is unsure, and
 * the penalty is large (63.5% acc / 124 drafts vs 91.1% / 263 over the same
 * 256 generated = -32% decode on short-context real text), while on
 * deep-context, high-acceptance text it is worth ~+2%. Ungated is the config
 * with the better worst case; n-max 3 is the depth that was never bad in
 * either regime. Measured in bench/mtp_tuning.md. */
const MTP_P_MIN = 0;
/**
 * KV cache quantization types accepted by upstream llama-server
 * (common/arg.cpp `kv_cache_types`; anything else is rejected at arg-parse
 * time). Highest precision first. The K/V defaults are the tuned 16 GiB pair
 * this tab has always shipped, so an untouched tab spawns the same args.
 */
const KV_CACHE_TYPES = ["f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "iq4_nl", "q4_0"];
const KV_CACHE_DEFAULT_K = "q5_0";
const KV_CACHE_DEFAULT_V = "q4_1";

function dshHome() {
	return process.env.DSH_HOME ?? join(process.env.HOME ?? "/tmp", ".dsh");
}

function dataDir() {
	return join(dshHome(), "local-models");
}

// ---------------------------------------------------------------------------
// Runtime: where the llama.cpp binaries live.
//
// The path used to be hardcoded to one machine's build dir. It is now resolved
// per launch (not at import time) from an ordered candidate list, so pointing
// the tab at a fresh build takes effect on the next Load — no dsh restart:
//
//   Runtime setting (settings.json `binPath`)  ← what the tab writes
//   → LOCAL_MODELS_BIN                         ← the pre-UI env knob
//   → every directory on PATH
//   → the usual per-user build dirs (~/llama.cpp/build/bin,
//     ~/Projetos/llama.cpp/build/bin, ~/.local/bin)
//   → /usr/local/bin, /usr/bin                 ← distro packages
//
// Every layer is either configured or home-relative: no absolute path of any
// particular machine is baked in. Both a directory ("…/build/bin") and the
// binary itself are accepted.
// ---------------------------------------------------------------------------

/** "name=path" / plain-path entries from LOCAL_MODELS_SHORTCUTS. Label
 * defaults to the dir's basename. Mirrors what the tab sends for its own list
 * (the client renders `label` + `path` chips either way). */
export function parseShortcutSpec(spec) {
	return String(spec ?? "").split(":").map((s) => s.trim()).filter(Boolean).map((s) => {
		const eq = s.indexOf("=");
		return eq > 0
			? { label: s.slice(0, eq).trim() || "dir", path: s.slice(eq + 1).trim() }
			: { label: basename(s) || s, path: s };
	}).filter((s) => s.path);
}

/** Expand a leading "~/" against `home` (empty home = leave as typed). */
export function expandHome(p, home) {
	const s = typeof p === "string" ? p.trim() : "";
	if (!s || !home) return s;
	if (s === "~") return home;
	return s.startsWith("~/") ? join(home, s.slice(2)) : s;
}

function defaultIsDir(p) {
	try {
		return statSync(p).isDirectory();
	} catch {
		return false;
	}
}

function defaultIsFile(p) {
	try {
		return statSync(p).isFile();
	} catch {
		return false;
	}
}

/** Is this path runnable? A file that exists but lost +x is the classic
 * "spawn EACCES" report, so it is called out separately from "not found". */
export function isExecutable(p) {
	try {
		accessSync(p, fsConstants.X_OK);
		return statSync(p).isFile();
	} catch {
		return false;
	}
}

/** Human label for a candidate's provenance, shown by the Runtime card. */
export const BIN_SOURCE_LABEL = {
	setting: "Runtime setting",
	env: "LOCAL_MODELS_BIN",
	path: "PATH",
	home: "~ build dir",
	system: "system dir",
};

/** Is this a binary we are willing to launch as llama-server? The name check
 * is what keeps the Runtime card from becoming a general-purpose "run any
 * binary" button, and what stops a stray file from widening the file-browser
 * roots to whatever directory it lives in. Exported for tests. */
export function isLlamaServerPath(p) {
	return typeof p === "string" && p !== "" && basename(p) === BIN_NAME && isExecutable(p);
}

/** Why a path is not a llama-server, or null when it is. Shared by the
 * /settings validation, the /runtime/check verdict and requireBin() so all
 * three tell the user the same thing. Exported for tests. */
export function binIdentityError(p) {
	if (typeof p !== "string" || p === "") return "no " + BIN_NAME + " path";
	if (basename(p) !== BIN_NAME) return "not a " + BIN_NAME + " binary: " + p;
	if (!isExecutable(p)) return "not executable: " + p + " (chmod +x)";
	return null;
}

/** Ordered llama-server candidates: every entry carries where it came from.
 * `search: false` keeps only the explicit entries (setting + env) — that is
 * what "test this one path" means for the Runtime card's Check button, so a
 * miss there can never look like a hit on some other layer. Pure but for the
 * injected `isDir`, so tests can drive it without depending on what happens to
 * be installed on the machine running them. */
export function binCandidates(opts = {}) {
	const {
		configured = "",
		env = "",
		pathEnv = "",
		home = "",
		name = BIN_NAME,
		search = true,
		systemDirs = ["/usr/local/bin", "/usr/bin"],
		isDir = defaultIsDir,
	} = opts;
	const out = [];
	const seen = new Set();
	// A candidate may name the binary or the directory holding it; a directory
	// (existing, or written with a trailing slash) gets the binary appended.
	const push = (candidate, source, asDir = false) => {
		let p = expandHome(candidate, home);
		if (!p) return;
		if (asDir || isDir(p) || p.endsWith("/") || p.endsWith(sep)) p = join(p, name);
		if (seen.has(p)) return;
		seen.add(p);
		out.push({ path: p, source, sourceLabel: BIN_SOURCE_LABEL[source] ?? source });
	};
	push(configured, "setting");
	if (search) {
		push(env, "env");
		for (const dir of String(pathEnv ?? "").split(delimiter)) {
			if (dir.trim()) push(resolve(dir.trim()), "path", true);
		}
		if (home) {
			// Per-user build locations, most common first. Home-relative on
			// purpose: an absolute path of one machine must never ship here.
			push(join(home, "llama.cpp", "build", "bin"), "home", true);
			push(join(home, "Projetos", "llama.cpp", "build", "bin"), "home", true);
			push(join(home, ".local", "bin"), "home", true);
		}
		for (const dir of systemDirs) push(dir, "system", true);
	}
	return out;
}

/** First candidate that exists on disk, plus the full search list so a miss
 * can tell the user everywhere we looked. `resolved` is null when nothing
 * exists — callers refuse the launch instead of spawning a phantom path. */
export function resolveBin(opts = {}) {
	const candidates = binCandidates(opts);
	const isFile = opts.isFile ?? defaultIsFile;
	const hit = candidates.find((c) => isFile(c.path)) ?? null;
	return {
		configured: String(opts.configured ?? "").trim(),
		resolved: hit ? hit.path : null,
		source: hit ? hit.source : null,
		sourceLabel: hit ? hit.sourceLabel : null,
		executable: hit ? isExecutable(hit.path) : false,
		candidates,
	};
}

/** The runtime as it stands right now: settings + env + this machine. */
export function currentBin() {
	return resolveBin({
		configured: readSettings().binPath,
		env: BIN_ENV,
		pathEnv: process.env.PATH ?? "",
		home: homedir(),
	});
}

/** Promise wrapper around child_process.execFile: resolves with both streams
 * instead of rejecting, so the caller can read a version banner off a binary
 * that exits non-zero. */
function execFileAsync(file, args, opts) {
	return new Promise((resolvePromise) => {
		execFile(file, args, { encoding: "utf8", ...opts }, (err, stdout, stderr) => {
			resolvePromise({ err: err ?? null, stdout: String(stdout ?? ""), stderr: String(stderr ?? "") });
		});
	});
}

/** `<bin> --version`: proves the binary actually starts and names the build the
 * tab would launch. Async — a hung binary must not stall the dsh event loop —
 * and cached by path+mtime+size+mode, because `chmod +x` changes neither mtime
 * nor size and a stale EACCES would otherwise stick forever. Only successful
 * probes are cached (a failure is cheap to retry and must never be sticky) and
 * the cache is capped, so cycling through paths cannot grow it without bound.
 * `runner` is injectable so tests need no real build. */
const versionCache = new Map();
const VERSION_CACHE_MAX = 32;
export async function probeBin(p, runner = execFileAsync) {
	if (typeof p !== "string" || !p) return { ok: false, error: "no binary path" };
	let key;
	try {
		const st = statSync(p);
		key = st.mtimeMs + ":" + st.size + ":" + st.mode;
	} catch (err) {
		return { ok: false, error: err?.message ?? String(err) };
	}
	const cached = versionCache.get(p);
	if (cached && cached.key === key) return cached.value;
	const { err, stdout, stderr } = await runner(p, ["--version"], { timeout: 5000 });
	// Some builds print the banner and still exit non-zero: keep the banner.
	const out = String(stdout ?? "") + String(stderr ?? "");
	const line = out.split("\n").map((l) => l.trim()).filter(Boolean)[0] ?? "";
	const value = line
		? { ok: true, version: line }
		: { ok: false, error: err ? (err.message ?? String(err)) : "no --version output" };
	if (value.ok) {
		// Refresh the insertion order so the cap evicts the least recently used.
		versionCache.delete(p);
		versionCache.set(p, { key, value });
		while (versionCache.size > VERSION_CACHE_MAX) versionCache.delete(versionCache.keys().next().value);
	} else {
		versionCache.delete(p);
	}
	return value;
}

/** Already-probed version for this path, or null. Never spawns anything, so
 * the 2 s status poll can report it for free. */
export function cachedBinVersion(p) {
	const hit = versionCache.get(p);
	return hit ? (hit.value.version ?? null) : null;
}

/** Browser shortcut dirs in effect: the Runtime card's list once the user has
 * saved one (even an empty one), else LOCAL_MODELS_SHORTCUTS, else none (home
 * is always reachable). */
export function shortcutDirs() {
	const configured = readSettings().shortcuts;
	if (configured !== null) return configured.map((p) => ({ label: basename(p) || p, path: p }));
	return SHORTCUTS_ENV;
}

// ---------------------------------------------------------------------------
// GPU inventory: what the tab's VRAM budget and its multi-GPU controls are
// based on. NVIDIA is asked through nvidia-smi, AMD is read from sysfs
// (amdgpu's mem_info_vram_total), anything else stays unknown and the tab
// falls back to the 16 GiB assumption it always shipped. Detection never
// blocks the status poll (see peekGpuInventory) and never blocks a launch: a
// failed probe only means "unknown".
// ---------------------------------------------------------------------------
const VRAM_ASSUMED_BYTES = 16 * 1024 * 1024 * 1024;
const GPU_CACHE_MS = 60_000;
const GPU_PROBE_TIMEOUT_MS = 4000;
/** Best-effort PCI id → product name (e.g. "Navi 48 [Radeon RX 9070 XT]").
 * Optional file: absent on many systems, and absence just leaves the PCI id. */
const PCI_IDS_PATHS = ["/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids", "/usr/share/pci.ids"];

/** Parse `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`
 * (one "name, MiB" per GPU). Pure, so the parse is testable without a GPU. */
export function parseNvidiaSmi(out) {
	const gpus = [];
	for (const raw of String(out ?? "").split("\n")) {
		const line = raw.trim();
		if (!line || /^name\s*,/i.test(line)) continue; // skip a --format=csv header
		const comma = line.lastIndexOf(",");
		if (comma <= 0) continue;
		const name = line.slice(0, comma).trim();
		const mib = Number(line.slice(comma + 1).trim());
		if (!name || !Number.isFinite(mib) || mib <= 0) continue;
		gpus.push({ vendor: "nvidia", name, vramBytes: Math.round(mib * 1024 * 1024) });
	}
	return gpus;
}

/** Resolve "1002:7550" against a pci.ids body ("<vendor>  <name>" lines with
 * tab-indented devices). Pure: the caller reads the file. */
export function lookupPciName(text, vendorId, deviceId) {
	const vendor = String(vendorId ?? "").toLowerCase();
	const device = String(deviceId ?? "").toLowerCase();
	if (!vendor || !device) return null;
	let inVendor = false;
	for (const line of String(text ?? "").split("\n")) {
		if (!line || line.startsWith("#")) continue;
		if (!line.startsWith("\t")) {
			// "<id>  <name>" (a vendor). Stop at the next vendor once inside.
			const id = line.slice(0, 4).trim().toLowerCase();
			if (inVendor) break;
			inVendor = id === vendor && /^\S+\s{2}/.test(line);
			continue;
		}
		if (!inVendor) continue;
		// "\t<id>  <name>" is a device; "\t\t<id>  <name>" is a subsystem.
		if (line.startsWith("\t\t")) continue;
		if (line.slice(1, 5).trim().toLowerCase() !== device) continue;
		const name = line.slice(5).trim();
		if (name) return name;
	}
	return null;
}

/** First readable pci.ids, or null. Read once and kept: the file is static. */
let pciIdsText;
function pciIds() {
	if (pciIdsText !== undefined) return pciIdsText;
	pciIdsText = null;
	for (const p of PCI_IDS_PATHS) {
		try {
			pciIdsText = readFileSync(p, "utf8");
			break;
		} catch { /* try the next location */ }
	}
	return pciIdsText;
}

/** AMD GPUs from sysfs. `listDir`/`read` are injected so tests need no /sys. */
export function readAmdGpus(opts = {}) {
	const {
		drmDir = "/sys/class/drm",
		listDir = (d) => readdirSync(d),
		read = (p) => readFileSync(p, "utf8").trim(),
	} = opts;
	let cards;
	try {
		cards = listDir(drmDir);
	} catch {
		return [];
	}
	const gpus = [];
	for (const card of cards.filter((c) => /^card\d+$/.test(c)).sort()) {
		const dev = join(drmDir, card, "device");
		try {
			// 0x1002 = AMD; everything else here (i915/nouveau/virtio…) is left
			// to the other probes.
			if (!read(join(dev, "vendor")).toLowerCase().includes("0x1002")) continue;
			const bytes = Number(read(join(dev, "mem_info_vram_total")));
			if (!Number.isFinite(bytes) || bytes <= 0) continue;
			// PCI_ID=1002:7550 → a real product name when pci.ids is installed.
			const pciId = /PCI_ID=([0-9a-f]{4}):([0-9a-f]{4})/i.exec(read(join(dev, "uevent")) ?? "");
			const product = pciId ? lookupPciName(pciIds(), pciId[1], pciId[2]) : null;
			gpus.push({
				vendor: "amd",
				name: product ?? "AMD GPU" + (pciId ? " " + pciId[1] + ":" + pciId[2] : ""),
				vramBytes: bytes,
				card,
			});
		} catch { /* not a readable amdgpu card */ }
	}
	return gpus;
}

/** Honor CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES: llama.cpp only sees the
 * listed GPUs, so the budget must not count the hidden ones. Numeric lists are
 * mapped by index; UUID/MIG entries (and "all") cannot be mapped from here and
 * keep the full list rather than guessing. */
export function filterVisibleGpus(gpus, spec) {
	if (typeof spec !== "string") return gpus;
	const s = spec.trim();
	if (s.toLowerCase() === "all") return gpus;
	if (s === "" || s === "-1" || s.toLowerCase() === "none") return [];
	const parts = s.split(",").map((p) => p.trim()).filter(Boolean);
	if (parts.some((p) => !/^\d+$/.test(p))) return gpus;
	const want = new Set(parts.map(Number));
	return gpus.filter((_, i) => want.has(i));
}

/** Detect the GPUs llama.cpp would see. Never throws: unknown hardware is a
 * normal answer (`gpus: []`, `assumed: true`). */
export async function detectGpus(opts = {}) {
	const { runner = execFileAsync, env = process.env, ...sysOpts } = opts;
	let gpus = [];
	let source = "assumed";
	try {
		const { err, stdout } = await runner(
			"nvidia-smi",
			["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
			{ timeout: GPU_PROBE_TIMEOUT_MS },
		);
		if (!err) {
			const found = filterVisibleGpus(parseNvidiaSmi(stdout), env.CUDA_VISIBLE_DEVICES);
			if (found.length > 0) {
				gpus = found;
				source = "nvidia-smi";
			}
		}
	} catch { /* no nvidia-smi here */ }
	if (gpus.length === 0) {
		const amd = filterVisibleGpus(readAmdGpus(sysOpts), env.HIP_VISIBLE_DEVICES ?? env.ROCR_VISIBLE_DEVICES);
		if (amd.length > 0) {
			gpus = amd;
			source = "sysfs";
		}
	}
	return { gpus, source, totalBytes: gpus.reduce((sum, g) => sum + g.vramBytes, 0), assumed: gpus.length === 0 };
}

let gpuCache = { at: 0, value: null };
let gpuProbe = null;
/** Cached inventory for the status route. Deliberately synchronous: it never
 * waits for nvidia-smi (a hung probe must not stall the 2 s poll). An expired
 * cache returns the previous answer and refreshes in the background, so the
 * next poll has the new one. */
export function peekGpuInventory() {
	if (gpuCache.value && Date.now() - gpuCache.at < GPU_CACHE_MS) return gpuCache.value;
	if (gpuProbe === null) {
		gpuProbe = detectGpus()
			.then((value) => { gpuCache = { at: Date.now(), value }; })
			.catch(() => { gpuCache = { at: Date.now(), value: null }; })
			.finally(() => { gpuProbe = null; });
	}
	return gpuCache.value ?? { gpus: [], source: "probing", pending: true, totalBytes: VRAM_ASSUMED_BYTES, assumed: true };
}

async function portUp() {
	try {
		const res = await fetch("http://127.0.0.1:" + PORT + "/health", {
			signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS),
		});
		return res.ok;
	} catch {
		return false;
	}
}

/** SIGTERM, then SIGKILL a couple of seconds later if the child is still up.
 * The escalation timer is tracked per child so exit/error handlers and
 * dispose() can cancel it — otherwise it fires after the fact holding a
 * stale reference. The timer is unref'd so it never keeps the host alive. */
const killTimers = new WeakMap();
function killChild(child) {
	if (child === null) return;
	try {
		child.kill("SIGTERM");
	} catch {
		/* already gone */
	}
	const timer = setTimeout(() => {
		killTimers.delete(child);
		try {
			if (child.exitCode === null && child.signalCode === null) {
				child.kill("SIGKILL");
			}
		} catch {
			/* already gone */
		}
	}, 2000);
	if (typeof timer.unref === "function") timer.unref();
	killTimers.set(child, timer);
}

/** Cancel a pending SIGKILL escalation (child already reaped). */
function clearKillTimer(child) {
	if (child === null) return;
	const timer = killTimers.get(child);
	if (timer !== undefined) {
		clearTimeout(timer);
		killTimers.delete(child);
	}
}

/** Best-effort kill of whatever pid is listening on $PORT.
 * Orphan detection only: callers invoke this when no tracked child exists
 * (stop() already confirmed the port is up and run() refuses a busy port),
 * so this never replaces killing the recorded child.pid. Our own pid is
 * excluded so a busy default port can't turn into self-SIGTERM. */
function killPortListener() {
	let pids;
	try {
		pids = execSync("lsof -ti :" + PORT, { encoding: "utf8" }).trim();
	} catch {
		return false;
	}
	let killed = false;
	for (const raw of pids.split("\n")) {
		const pid = Number(raw);
		if (Number.isInteger(pid) && pid > 0 && pid !== process.pid) {
			try {
				process.kill(pid, "SIGTERM");
				killed = true;
			} catch {
				/* gone */
			}
		}
	}
	return killed;
}
// ---------------------------------------------------------------------------
// GGUF header parsing (ported from vram-calculator/src/gguf.ts).
// ---------------------------------------------------------------------------
const GV_UINT8 = 0, GV_INT8 = 1, GV_UINT16 = 2, GV_INT16 = 3, GV_UINT32 = 4,
	GV_INT32 = 5, GV_FLOAT32 = 6, GV_BOOL = 7, GV_STRING = 8, GV_ARRAY = 9,
	GV_UINT64 = 10, GV_INT64 = 11, GV_FLOAT64 = 12;
const CHUNK = 4 * 1024 * 1024;
// Tokenizer arrays are counted, never materialised: every sentencepiece/BPE
// GGUF writes a quarter of a million entries here, and the only thing the
// estimator asks of them is a length — nVocab, which is exactly the
// tokenizer.ggml.tokens count when <arch>.vocab_size is absent (it is absent
// from every Qwen3.5/3.8-family file). The payload is still walked, so the KV
// pairs that follow the array stay reachable.
const COUNT_ONLY_ARRAY_KEYS = new Set([
	"tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges",
]);

class FileReader {
	constructor(path) {
		this.path = path;
		this.fd = openSync(path, "r");
		this.size = statSync(path).size;
		this.buffer = new Uint8Array(0);
		this.pos = 0;
		this.view = null;
	}
	ensure(n) {
		while (this.pos + n > this.buffer.length) {
			const start = this.buffer.length;
			if (start >= this.size) {
				throw new Error("Unexpected end of file while parsing GGUF header");
			}
			const want = Math.min(Math.max(n, CHUNK), this.size - start);
			const chunk = new Uint8Array(want);
			const got = readSync(this.fd, chunk, 0, want, start);
			if (got <= 0) {
				throw new Error("Unexpected end of file while parsing GGUF header");
			}
			const merged = new Uint8Array(this.buffer.length + got);
			merged.set(this.buffer, 0);
			merged.set(chunk.subarray(0, got), this.buffer.length);
			this.buffer = merged;
			this.view = new DataView(this.buffer.buffer, this.buffer.byteOffset, this.buffer.byteLength);
		}
	}
	readRaw(n) {
		this.ensure(n);
		const out = this.buffer.subarray(this.pos, this.pos + n);
		this.pos += n;
		return out;
	}
	u8() { this.ensure(1); const v = this.view.getUint8(this.pos); this.pos += 1; return v; }
	i8() { this.ensure(1); const v = this.view.getInt8(this.pos); this.pos += 1; return v; }
	u16() { this.ensure(2); const v = this.view.getUint16(this.pos, true); this.pos += 2; return v; }
	i16() { this.ensure(2); const v = this.view.getInt16(this.pos, true); this.pos += 2; return v; }
	u32() { this.ensure(4); const v = this.view.getUint32(this.pos, true); this.pos += 4; return v; }
	i32() { this.ensure(4); const v = this.view.getInt32(this.pos, true); this.pos += 4; return v; }
	u64() {
		this.ensure(8);
		const lo = this.view.getUint32(this.pos, true);
		const hi = this.view.getUint32(this.pos + 4, true);
		this.pos += 8;
		return Number((BigInt(hi) << 32n) | BigInt(lo));
	}
	i64() {
		this.ensure(8);
		const lo = this.view.getUint32(this.pos, true);
		const hi = this.view.getInt32(this.pos + 4, true);
		this.pos += 8;
		return Number((BigInt(hi) << 32n) | BigInt(lo));
	}
	f32() { this.ensure(4); const v = this.view.getFloat32(this.pos, true); this.pos += 4; return v; }
	f64() { this.ensure(8); const v = this.view.getFloat64(this.pos, true); this.pos += 8; return v; }
	readString() {
		const len = this.u64();
		const bytes = this.readRaw(len);
		return new TextDecoder().decode(bytes);
	}
	/** Step over a string without decoding it; returns its byte length. */
	skipString() {
		const len = this.u64();
		this.readRaw(len);
		return len;
	}
	/** Read a metadata value. keep=false walks it without keeping it: strings
	 * are measured but never decoded, arrays are counted but never built (see
	 * COUNT_ONLY_ARRAY_KEYS). An array yields its element count either way. */
	readValue(type, keep = true) {
		switch (type) {
			case GV_UINT8: return this.u8();
			case GV_INT8: return this.i8();
			case GV_UINT16: return this.u16();
			case GV_INT16: return this.i16();
			case GV_UINT32: return this.u32();
			case GV_INT32: return this.i32();
			case GV_FLOAT32: return this.f32();
			case GV_BOOL: return this.u8() !== 0;
			case GV_STRING: return keep ? this.readString() : this.skipString();
			case GV_ARRAY: {
				const elemType = this.u32();
				const count = this.u64();
				if (!Number.isSafeInteger(count) || count > this.size) {
					throw new Error("Corrupt GGUF: unreasonable array length (" + count + ")");
				}
				if (!keep) {
					for (let i = 0; i < count; i++) this.readValue(elemType, false);
					return count;
				}
				const arr = [];
				for (let i = 0; i < count; i++) arr.push(this.readValue(elemType));
				return arr;
			}
			case GV_UINT64: return this.u64();
			case GV_INT64: return this.i64();
			case GV_FLOAT64: return this.f64();
			default: throw new Error("Unknown GGUF metadata value type: " + type);
		}
	}
	close() {
		try { closeSync(this.fd); } catch { /* ignore */ }
	}
}

function alignUp(x, a) {
	return Math.ceil(x / a) * a;
}

function num(v) {
	return typeof v === "number" ? v : null;
}

/** Routed-expert weight tensor? llama.cpp names every MoE arch's routed
 * experts blk.N.ffn_{gate,down,up,gate_up}_exps (chunked: *_chexps).
 * Shared experts (*shexp*) are deliberately excluded — always active. */
function isExpertTensorName(name) {
	return typeof name === "string"
		&& (name.includes("_exps") || name.includes("chexps"))
		&& !name.includes("shexp");
}

// general.file_type values (LLAMA_FTYPE_* in include/llama.h).
const FILE_TYPE_NAMES = {
	0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0",
	8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M",
	13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
	18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
	23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
	28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
	36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE", 39: "NVFP4",
};

function parseGGUF(path) {
	const r = new FileReader(path);
	try {
		const magicBuf = r.readRaw(4);
		const magicStr = String.fromCharCode(magicBuf[0], magicBuf[1], magicBuf[2], magicBuf[3]);
		if (magicStr !== "GGUF") throw new Error("Not a GGUF file (magic = " + JSON.stringify(magicStr) + ")");
		const version = r.u32();
		if (version < 2 || version > 3) {
			throw new Error("Unsupported GGUF version " + version + " (only v2/v3 are supported).");
		}
		const tensorCount = r.u64();
		const metadataKvCount = r.u64();
		if (!Number.isSafeInteger(tensorCount) || tensorCount > r.size ||
			!Number.isSafeInteger(metadataKvCount) || metadataKvCount > r.size) {
			throw new Error("Corrupt GGUF header (tensor/metadata count is bogus).");
		}
		const raw = {};
		const arrayCounts = {}; // key → element count for arrays counted, not kept
		for (let i = 0; i < metadataKvCount; i++) {
			const key = r.readString();
			const type = r.u32();
			if (type === GV_ARRAY && COUNT_ONLY_ARRAY_KEYS.has(key)) {
				arrayCounts[key] = r.readValue(type, false);
				continue;
			}
			raw[key] = r.readValue(type);
		}
		let firstOffset = -1, lastOffset = 0, monotonic = true;
		const tensors = []; // {name, offset} — kept for MoE expert-byte accounting
		for (let i = 0; i < tensorCount; i++) {
			const name = r.readString();
			const nDims = r.u32();
			for (let d = 0; d < nDims; d++) r.u64();
			r.u32(); // ggml type
			const offset = r.u64();
			tensors.push({ name, offset });
			if (i === 0) firstOffset = offset;
			if (offset < lastOffset) monotonic = false;
			lastOffset = offset;
		}
		const headerPos = r.pos;
		let alignment = num(raw["general.alignment"]);
		if (!alignment || alignment < 1) alignment = 32;
		const dataStart = alignUp(headerPos, alignment);
		const weightsBytes = r.size - dataStart;
		if (weightsBytes <= 0 || weightsBytes > r.size) {
			throw new Error("Failed to parse GGUF layout (computed weights size is invalid).");
		}
		const arch = typeof raw["general.architecture"] === "string" ? raw["general.architecture"] : null;
		const p = (suffix) => (arch ? num(raw[arch + "." + suffix]) : null);
		const nLayers = p("block_count");
		const nEmbd = p("embedding_length");
		const nHeads = p("attention.head_count");
		const nKvHeadsRaw = p("attention.head_count_kv");
		const keyLen = p("attention.key_length");
		const valLen = p("attention.value_length");
		// MLA (latent KV) models carry separate compressed key/value head
		// lengths; llama.cpp's is_mla() keys off exactly these two.
		const mlaKeyLen = p("attention.key_length_mla");
		const mlaValLen = p("attention.value_length_mla");
		// <arch>.vocab_size is optional and often absent (every Qwen3.5/3.8
		// GGUF omits it); the tokenizer's own token list is the real count —
		// llama.cpp's n_vocab is exactly that array's length.
		const nVocab = p("vocab_size") ?? arrayCounts["tokenizer.ggml.tokens"] ?? null;
		const contextLength = p("context_length");
		const slidingWindow = p("attention.sliding_window");
		const fullAttnInterval = p("full_attention_interval");
		const nextnPredictLayers = p("nextn_predict_layers");
		const expertCount = p("expert_count");
		const expertUsedCount = p("expert_used_count");
		const expertSharedCount = p("expert_shared_count");
		const expertFfnLength = p("expert_feed_forward_length") ?? p("expert_chunk_feed_forward_length");
		const moeEveryNLayers = p("moe_every_n_layers");
		const leadingDenseBlocks = p("leading_dense_block_count");
		const valueExpertCount = p("attention.value_expert_count");
		const valueExpertUsedCount = p("attention.value_expert_used_count");
		const ssmConvKernel = p("ssm.conv_kernel");
		const ssmStateSize = p("ssm.state_size");
		const ssmNGroup = p("ssm.group_count");
		const ssmDtRank = p("ssm.time_step_rank");
		const ssmInnerSize = p("ssm.inner_size");
		const nKvHeads = nKvHeadsRaw ?? nHeads;
		const isMla = mlaKeyLen != null && mlaValLen != null;
		let headDim = keyLen ?? valLen ?? null;
		if (headDim === null && nEmbd && nHeads) headDim = nEmbd / nHeads;
		const warnings = [];
		if (firstOffset !== 0) warnings.push("First tensor offset is " + firstOffset + " bytes into the data region — non-standard writer.");
		if (!monotonic) warnings.push("Tensor offsets are not monotonically increasing (unusual layout).");
		if (!arch) warnings.push("No architecture metadata found.");
		if (nLayers === null) warnings.push("Could not read block_count (n_layers).");
		if (headDim === null || nKvHeads === null) warnings.push("Could not read attention dimensions; KV cache estimate unavailable.");
		if (nVocab === null) warnings.push("Could not read vocab size (<arch>.vocab_size and tokenizer.ggml.tokens both missing); the compute estimate assumes 32K tokens.");
		if (isMla) warnings.push("MLA (latent KV) layout — llama.cpp requires the same cache type for K and V on this model.");
		if (arch && /deepseek/i.test(arch)) warnings.push("Architecture looks like DeepSeek (MLA). The standard KV-cache formula may under-estimate real usage.");
		if (slidingWindow != null && slidingWindow > 0) warnings.push("Hybrid sliding-window model (window = " + slidingWindow + " tokens). Local layers cap their KV cache at the window; a ~1/6 share of layers is assumed full-attention.");
		// --- MoE detection ------------------------------------------------
		// Primary signal: <arch>.expert_count > 0. Fallback: expert tensor
		// names (llama.cpp maps every MoE arch's routed experts to
		// blk.N.ffn_{gate,down,up}[_gate_up]_exps / *_chexps).
		const tensorHasExperts = tensors.some((t) => isExpertTensorName(t.name));
		const isMoe = (expertCount != null && expertCount > 0) || tensorHasExperts;
		const moeSource = (expertCount != null && expertCount > 0) ? "metadata" : (tensorHasExperts ? "tensors" : null);
		if (tensorHasExperts && !(expertCount != null && expertCount > 0)) {
			warnings.push("Expert tensors found but no expert_count metadata — treating as MoE; expert-byte estimates are approximate.");
		}
		// Expert weights in bytes, via offset deltas (valid only for the
		// standard dense-packed layout, i.e. monotonic offsets). Shared
		// experts (*shexp*) stay resident — they are always active — so they
		// are NOT counted here. Null when the layout is non-standard.
		let expertBytes = null, expertBytesByLayer = null;
		if (tensorHasExperts && monotonic) {
			const ordered = [...tensors].sort((a, b) => a.offset - b.offset);
			expertBytes = 0;
			expertBytesByLayer = {};
			for (let i = 0; i < ordered.length; i++) {
				const t = ordered[i];
				if (!isExpertTensorName(t.name)) continue;
				const end = i + 1 < ordered.length ? ordered[i + 1].offset : weightsBytes;
				const size = end - t.offset;
				if (size <= 0) continue;
				expertBytes += size;
				const m = /^blk\.(\d+)\./.exec(t.name);
				if (m) {
					const layer = Number(m[1]);
					expertBytesByLayer[layer] = (expertBytesByLayer[layer] ?? 0) + size;
				}
			}
		}
		const meta = {
			version, arch,
			name: typeof raw["general.name"] === "string" ? raw["general.name"] : null,
			paramCount: num(raw["general.parameter_count"]),
			fileType: num(raw["general.file_type"]),
			fileTypeName: num(raw["general.file_type"]) != null ? (FILE_TYPE_NAMES[num(raw["general.file_type"])] ?? "type " + num(raw["general.file_type"])) : "unknown",
			nLayers, nEmbd, nHeads, nKvHeads, headDim, nVocab, contextLength, slidingWindow,
			isMla,
			fullAttnInterval, nextnPredictLayers, ssmConvKernel, ssmStateSize, ssmNGroup, ssmDtRank, ssmInnerSize,
			expertCount, expertUsedCount, expertSharedCount, expertFfnLength,
			moeEveryNLayers, leadingDenseBlocks, valueExpertCount, valueExpertUsedCount,
			isMoe, moeSource, expertBytes, expertBytesByLayer,
			alignment, tensorCount, metadataKvCount, warnings,
		};
		return { fileName: basename(path), filePath: path, fileSize: r.size, weightsBytes, dataStart, meta };
	} finally {
		r.close();
	}
}

const ggufCache = new Map();
export function parseGGUFCached(path) {
	let st;
	try {
		st = statSync(path);
	} catch {
		throw new Error("cannot stat file: " + path);
	}
	if (!st.isFile()) throw new Error("not a file: " + path);
	const hit = ggufCache.get(path);
	if (hit && hit.size === st.size && hit.mtimeMs === st.mtimeMs) return hit.result;
	const result = parseGGUF(path);
	ggufCache.set(path, { size: st.size, mtimeMs: st.mtimeMs, result });
	if (ggufCache.size > 8) {
		const first = ggufCache.keys().next().value;
		ggufCache.delete(first);
	}
	return result;
}
// ---------------------------------------------------------------------------
// Process manager.
// ---------------------------------------------------------------------------
function slug(name) {
	return String(name)
		.replace(/\.gguf$/i, "")
		.replace(/[^a-zA-Z0-9_-]+/g, "-")
		.replace(/^-+|-+$/g, "")
		.toLowerCase()
		.slice(0, 40) || "local-model";
}

// Reasoning-effort helpers (exported for tests). 'off' disables thinking;
// any other level is passed through to llama-server at launch time — this
// llama.cpp build does NOT parse per-request reasoning_effort, so the level
// is a server-launch setting.
//
// preserveThinking controls --reasoning-preserve vs --no-reasoning-preserve
// (upstream default is preserve ON; this plugin historically forced OFF, so
// the toggle defaults to false to keep existing profiles byte-identical).
// Accepts both camelCase (preserveThinking) and snake_case (preserve_thinking)
// profile keys.
export function normalizePreserveThinking(raw) {
	return raw?.preserveThinking === true || raw?.preserve_thinking === true;
}

export function normalizeEffort(effort) {
	return THINKING_LEVELS.includes(effort) ? effort : "medium";
}

// KV-cache helpers (exported for tests). The tab exposes the full set of types
// upstream llama-server accepts; an absent or unknown value falls back to the
// tuned default (Q5_0 K / Q4_1 V) so legacy profiles and API calls keep the
// historical launch args.
export function normalizeKvCacheType(value, fallback) {
	return typeof value === "string" && KV_CACHE_TYPES.includes(value) ? value : fallback;
}

/** True when llama.cpp demands type_k === type_v for this model. The runtime
 * check is `hparams.is_mla() || arch == LLM_ARCH_DEEPSEEK4` (llama-context.cpp:
 * "model does not support different K and V cache types"); is_mla() is exactly
 * "the GGUF carries both attention.key_length_mla and value_length_mla", which
 * is what the parser stores as `isMla`. Deliberately narrow — a false positive
 * would block a launch llama-server would have accepted, while a false
 * negative only means llama-server reports the error itself. */
export function kvTypesMustMatch(meta) {
	return meta?.isMla === true || meta?.arch === "deepseek4";
}

export function thinkingArgsFor(effort, preserveThinking = false) {
	const level = normalizeEffort(effort);
	const preserve = typeof preserveThinking === "object" && preserveThinking !== null
		? normalizePreserveThinking(preserveThinking)
		: normalizePreserveThinking({ preserveThinking });
	return level === "off"
		? ["--reasoning", "off"]
		: ["--reasoning", "auto", "--reasoning-format", "deepseek", preserve ? "--reasoning-preserve" : "--no-reasoning-preserve", "--reasoning-effort", level];
}

/** --sleep-idle-seconds for launch args. Empty (upstream default = disabled)
 * when the timer is off (0). The sleeping server keeps answering /health
 * and /models and reloads automatically on the next request (one slow
 * request), so this is VRAM eviction, not a shutdown. Exported for tests. */
/** Effective --sleep-idle-seconds for a timer value, or null when off. */
export function sleepIdleSecsFor(mins) {
	return Number.isInteger(mins) && mins > 0 ? mins * 60 : null;
}

export function sleepIdleArgsFor(mins) {
	const s = sleepIdleSecsFor(mins);
	return s === null ? [] : ["--sleep-idle-seconds", String(s)];
}

// ---------------------------------------------------------------------------
// Multi-GPU helpers (exported for tests): how the model is spread over the
// GPUs llama.cpp sees. The tab only offers these when more than one GPU was
// detected; the API still accepts them, so a saved profile keeps launching the
// same way on a machine where detection differs.
// ---------------------------------------------------------------------------

/** `--split-mode` values upstream accepts. "layer" is llama.cpp's own default
 * (layers pipelined across GPUs) and "row" is the usual "tensor parallelism"
 * setting; "tensor" is upstream's experimental full tensor split. */
export const SPLIT_MODES = ["layer", "row", "tensor", "none"];
export const SPLIT_MODE_DEFAULT = "layer";

export function normalizeSplitMode(value) {
	return typeof value === "string" && SPLIT_MODES.includes(value) ? value : SPLIT_MODE_DEFAULT;
}

/** `--tensor-split` proportions: comma (or slash/space) separated positive
 * numbers, one per GPU, e.g. "3,1". Anything else becomes null = upstream's
 * own proportional-to-VRAM default. */
export function normalizeTensorSplit(value) {
	if (typeof value !== "string") return null;
	const parts = value.trim().split(/[,/\s]+/).filter(Boolean);
	if (parts.length === 0) return null;
	const nums = [];
	for (const p of parts) {
		const n = Number(p);
		if (!Number.isFinite(n) || n <= 0) return null;
		nums.push(n);
	}
	return nums.join(",");
}

/** Split-related launch args. The default (layer, no explicit split) adds
 * nothing, so an untouched tab spawns the same argv as before the control
 * existed. `tensorSplit` alone is still meaningful with layer/row. */
export function splitArgsFor(splitMode, tensorSplit) {
	const mode = normalizeSplitMode(splitMode);
	const split = normalizeTensorSplit(tensorSplit);
	const args = mode === SPLIT_MODE_DEFAULT ? [] : ["--split-mode", mode];
	if (split) args.push("--tensor-split", split);
	return args;
}

/** Even-ish default proportions for N GPUs, weighted by VRAM when known:
 * what the tab prefills in the tensor-split box. Returns "" for a single GPU
 * (nothing to split). */
export function evenTensorSplit(gpus) {
	const list = Array.isArray(gpus) ? gpus.filter((g) => g && g.vramBytes > 0) : [];
	if (list.length < 2) return "";
	const min = Math.min(...list.map((g) => g.vramBytes));
	return list.map((g) => Math.max(1, Math.round(g.vramBytes / min))).join(",");
}

// ---------------------------------------------------------------------------
// MoE helpers (exported for tests). Dense models ignore every MoE option.
// ---------------------------------------------------------------------------

/** Normalize the MoE launch config. cpuMoe wins over nCpuMoe; expertUsed is
 * the requested top-k override (null = stock value from the GGUF). */
export function normalizeMoEConfig(raw) {
	const cpuMoe = raw?.cpuMoe === true;
	const nCpuMoe = Number.isInteger(raw?.nCpuMoe) && raw.nCpuMoe > 0 ? raw.nCpuMoe : 0;
	const expertUsed = Number.isInteger(raw?.expertUsed) && raw.expertUsed > 0 ? raw.expertUsed : null;
	const arch = typeof raw?.arch === "string" && raw.arch ? raw.arch : null;
	return { cpuMoe, nCpuMoe, expertUsed, arch };
}

/** CLI args for the MoE config. Returns [] for dense models. Throws when the
 * requested top-k exceeds the model's expert count. */
export function moeArgsFor(moe, meta) {
	const cfg = normalizeMoEConfig(moe);
	if (!meta?.isMoe) return [];
	const args = [];
	if (cfg.cpuMoe) args.push("--cpu-moe");
	else if (cfg.nCpuMoe > 0) args.push("--n-cpu-moe", String(cfg.nCpuMoe));
	if (cfg.expertUsed != null) {
		if (meta.expertCount != null && cfg.expertUsed > meta.expertCount) {
			throw new Error("top-k " + cfg.expertUsed + " exceeds this model's " + meta.expertCount + " experts");
		}
		const arch = cfg.arch || meta.arch;
		if (!arch) throw new Error("cannot override top-k: unknown architecture");
		args.push("--override-kv", arch + ".expert_used_count=int:" + cfg.expertUsed);
	}
	return args;
}

/** VRAM-resident weights for the estimate: subtract the expert bytes kept on
 * CPU. Returns null when the split cannot be computed (unknown layout). */
export function moeVramWeights(weightsBytes, meta, moe) {
	const cfg = normalizeMoEConfig(moe);
	if (!meta?.isMoe || (!cfg.cpuMoe && cfg.nCpuMoe <= 0)) return weightsBytes;
	if (meta.expertBytes == null) return null;
	if (cfg.cpuMoe) return Math.max(weightsBytes - meta.expertBytes, 0);
	const byLayer = meta.expertBytesByLayer ?? {};
	let cpuBytes = 0, known = false;
	for (const [layer, bytes] of Object.entries(byLayer)) {
		if (Number(layer) < cfg.nCpuMoe) { cpuBytes += bytes; known = true; }
	}
	if (!known) return null;
	return Math.max(weightsBytes - cpuBytes, 0);
}

function createManager() {
	mkdirSync(dataDir(), { recursive: true });
	const state = {
		status: "idle", // idle | starting | ready | stopped
		pid: null,
		model: null,
		modelPath: null,
		alias: null,
		ctx: null,
		mtpHeads: 0,
		reasoningEffort: "medium",
		preserveThinking: false,
		splitMode: SPLIT_MODE_DEFAULT,
		tensorSplit: null,
		kvTypeK: KV_CACHE_DEFAULT_K,
		kvTypeV: KV_CACHE_DEFAULT_V,
		cpuMoe: false,
		nCpuMoe: 0,
		expertUsed: null,
		mode: "single", // single | router
		presetsPath: null,
		loadedModels: [],
		mmprojPath: null,
		mmprojCpu: MMPROJ_CPU,
		port: PORT,
		startedAt: null,
		error: null,
		logPath: LOG_PATH,
		binPath: null, // llama-server actually spawned (set by launch())
		sleepIdleSecs: null, // effective idle-eviction timer (null = off)
	};
	let child = null;
	let healthTimer = null;
	let runSeq = 0;

	function stopHealthPolling() {
		if (healthTimer !== null) { clearInterval(healthTimer); healthTimer = null; }
	}

	/** Resolve the binary for a launch, or explain what to fix. Re-resolved on
	 * every launch (not cached at import) so pointing the Runtime card at a new
	 * build takes effect without restarting dsh. */
	function requireBin() {
		const bin = currentBin();
		if (!bin.resolved) {
			const tried = bin.candidates.slice(0, 6).map((c) => c.path).join(", ");
			return { error: "llama-server not found — set the llama.cpp binaries folder in the Runtime card (searched: " + tried + (bin.candidates.length > 6 ? ", …" : "") + ")" };
		}
		// Last line of defense: a hand-edited settings.json must not make this
		// spawn an arbitrary executable.
		const bad = binIdentityError(bin.resolved);
		if (bad) return { error: "refusing to launch: " + bad };
		return { path: bin.resolved };
	}

	async function stop() {
		stopHealthPolling();
		if (child !== null) {
			const c = child;
			child = null;
			killChild(c);
			state.status = "stopped";
			state.pid = null;
			return "child";
		}
		if (await portUp()) {
			const killed = killPortListener();
			state.status = "stopped";
			state.pid = null;
			return killed ? "port" : "none";
		}
		state.status = "idle";
		state.error = null;
		return "none";
	}

	async function run(modelPath, ctxSize, mtpHeads, mmprojPath, effort, mmprojCpu = MMPROJ_CPU, moe = null, preserveThinking = false, kvTypeK = KV_CACHE_DEFAULT_K, kvTypeV = KV_CACHE_DEFAULT_V, splitMode = SPLIT_MODE_DEFAULT, tensorSplit = null) {
		const bin = requireBin();
		if (bin.error) return { ok: false, error: bin.error };
		if (!existsSync(modelPath)) return { ok: false, error: "model file missing: " + modelPath };
		if (mmprojPath && !existsSync(mmprojPath)) return { ok: false, error: "mmproj file missing: " + mmprojPath };
		if (await portUp()) return { ok: false, error: "a server is already listening on port " + PORT + " — Stop it first" };

		const ctx = Number.isInteger(ctxSize) && ctxSize > 0 ? ctxSize : 8192;
		// Fixed MTP draft (upstream draft-mtp): any depth 0-MTP_MAX is valid;
		// upstream clamps the effective depth to the model's nextn depth.
		const mtp = Number.isInteger(mtpHeads) && mtpHeads > 0 ? Math.min(mtpHeads, MTP_MAX) : 0;
		const alias = slug(basename(modelPath));
		const mmproj = typeof mmprojPath === "string" && mmprojPath ? mmprojPath : null;
		const mmprojCpuFinal = typeof mmprojCpu === "boolean" ? mmprojCpu : MMPROJ_CPU;
		const effortLevel = normalizeEffort(effort);
		const preserve = normalizePreserveThinking({ preserveThinking });
		// MoE: read the header for arch/validation (best-effort — a parse
		// failure only forfeits the top-k override, never the launch).
		let moeMeta = null;
		try {
			moeMeta = parseGGUFCached(modelPath).meta;
		} catch {
			moeMeta = null;
		}
		if (moe?.arch && moeMeta) moeMeta = { ...moeMeta, arch: moe.arch };
		else if (moe?.arch) moeMeta = { isMoe: true, arch: moe.arch, expertCount: null };
		let moeArgs;
		try {
			moeArgs = moeArgsFor(moe, moeMeta);
		} catch (err) {
			return { ok: false, error: err?.message ?? String(err) };
		}
		const moeCfg = normalizeMoEConfig(moe);

		// KV cache type comes from the tab's K/V selectors; the defaults
		// (Q5_0 K / Q4_1 V) are the 2026 ladder sweet spot for a 16 GiB card
		// (same tail precision as Q8_0/Q4_0 at 20% less VRAM; K-first, but
		// Q8_0 K only pays off with V >= Q5_1). Unknown values fall back to
		// that pair. Quantized V needs flash-attn (always on below); MLA /
		// DeepSeek4 models reject mixed K/V types, so refuse up front instead
		// of letting the child die on "does not support different K and V".
		const kvK = normalizeKvCacheType(kvTypeK, KV_CACHE_DEFAULT_K);
		const kvV = normalizeKvCacheType(kvTypeV, KV_CACHE_DEFAULT_V);
		// Multi-GPU placement. The default (layer, no explicit proportions)
		// emits nothing, so an untouched tab spawns the same argv as always.
		const split = normalizeSplitMode(splitMode);
		const splitArgs = splitArgsFor(split, tensorSplit);
		if (kvK !== kvV && kvTypesMustMatch(moeMeta)) {
			return { ok: false, error: "this model (MLA latent KV) needs the same cache type for K and V — set both to " + kvK };
		}
		// The draft is unconditional whenever MTP is on: there is no ctx
		// ceiling any more (the old "ignore softcap" toggle is gone, always-on).
		// Draft KV is pinned to Q4_0 — upstream has no draft auto-quant — and a
		// deep draft at large ctx can still OOM or collapse decode, so the tab
		// defaults to the measured sweet spot (3).
		const specArgs = mtp > 0
			? ["--spec-type", "draft-mtp", "--spec-draft-n-max", String(mtp), "--spec-draft-p-min", String(MTP_P_MIN),
				"--spec-draft-type-k", "q4_0", "--spec-draft-type-v", "q4_0"]
			: [];

		const args = [
			"-m", modelPath,
			"-ngl", "999",
			"-c", String(ctx),
			"-b", "2048", "-ub", "512",
			"-t", "4", "-np", "1", "--poll", "0",
			"--cont-batching",
			"--jinja",
			...thinkingArgsFor(effortLevel, preserve),
			"--flash-attn", "on",
			"--kv-unified",
			"--cache-type-k", kvK,
			"--cache-type-v", kvV,
			...splitArgs,
			...specArgs,
			...moeArgs,
			// Qwen-VL needs a 1024-token image minimum (load_hparams warning).
			...(mmproj ? ["--mmproj", mmproj, "--image-min-tokens", "1024", ...(mmprojCpuFinal ? ["--no-mmproj-offload"] : [])] : []),
			"--alias", alias,
			"--host", "127.0.0.1",
			"--port", String(PORT),
			// Idle VRAM eviction (server-managed; wakes on next request).
			...sleepIdleArgsFor(readSettings().autoUnloadMins),
		];

		return launch(args, {
			mode: "single",
			bin: bin.path,
			modelLabel: basename(modelPath),
			modelPath,
			alias,
			ctx,
			mtpHeads: mtp,
			reasoningEffort: effortLevel,
			preserveThinking: preserve,
			mmproj,
			mmprojCpu: mmprojCpuFinal,
			cpuMoe: moeCfg.cpuMoe,
			nCpuMoe: moeCfg.nCpuMoe,
			expertUsed: moeCfg.expertUsed,
			kvTypeK: kvK,
			kvTypeV: kvV,
			splitMode: split,
			tensorSplit: normalizeTensorSplit(tensorSplit),
			sleepIdleSecs: sleepIdleSecsFor(readSettings().autoUnloadMins),
			logNote: "starting " + basename(modelPath) + " (ctx " + ctx + ", mtp " + mtp + ", effort " + effortLevel + (preserve ? ", preserve" : ", no-preserve") + ", kv " + kvK + "/" + kvV + (moeMeta?.isMoe ? ", experts " + (moeCfg.cpuMoe ? "cpu" : moeCfg.nCpuMoe > 0 ? "cpu-first-" + moeCfg.nCpuMoe : "gpu") + (moeCfg.expertUsed != null ? " top-" + moeCfg.expertUsed : "") : "") + ", mmproj " + (mmproj ? basename(mmproj) + (mmprojCpuFinal ? " (cpu)" : " (gpu)") : "none") + (splitArgs.length > 0 ? ", split " + split + (normalizeTensorSplit(tensorSplit) ? " " + normalizeTensorSplit(tensorSplit) : "") : "") + ", bin " + bin.path + ")",
		});
	}

	async function startRouter(presetsPath) {
		const bin = requireBin();
		if (bin.error) return { ok: false, error: bin.error };
		if (!existsSync(presetsPath)) return { ok: false, error: "presets file missing: " + presetsPath };
		if (await portUp()) return { ok: false, error: "a server is already listening on port " + PORT + " — Stop it first" };
		const args = [
			"--models-preset", presetsPath,
			"--host", "127.0.0.1",
			"--port", String(PORT),
			"--models-max", String(ROUTER_MAX),
			// Idle VRAM eviction (server-managed; wakes on next request).
			...sleepIdleArgsFor(readSettings().autoUnloadMins),
		];
		return launch(args, { mode: "router", bin: bin.path, presetsPath, modelLabel: "router (" + basename(presetsPath) + ")", sleepIdleSecs: sleepIdleSecsFor(readSettings().autoUnloadMins), logNote: "starting router (" + basename(presetsPath) + ", bin " + bin.path + ")" });
	}

	async function launch(args, info) {
		appendFileSync(LOG_PATH, "\n=== " + new Date().toISOString() + " " + info.logNote + " ===\n");

		const next = spawn(info.bin, args, {
			env: { ...process.env, RADV_PERFTEST: "nogttspill" },
			stdio: ["ignore", "pipe", "pipe"],
		});

		const sink = (data) => {
			try { appendFileSync(LOG_PATH, data); } catch { /* ignore */ }
		};
		next.stdout.on("data", sink);
		next.stderr.on("data", sink);
		next.on("error", (err) => {
			stopHealthPolling();
			clearKillTimer(next);
			if (child === next) child = null;
			state.status = "stopped";
			state.error = "spawn failed: " + err.message;
		});
		next.on("exit", (code, signal) => {
			stopHealthPolling();
			clearKillTimer(next);
			if (child === next) child = null;
			if (state.status !== "stopped") {
				state.status = "stopped";
				state.error = state.error ?? "server exited before becoming healthy (" + (signal ?? "code " + code) + ")";
			}
		});

		child = next;
		state.status = "starting";
		state.pid = next.pid;
		state.mode = info.mode;
		state.binPath = info.bin ?? null;
		state.model = info.modelLabel ?? null;
		if (info.modelPath !== undefined) state.modelPath = info.modelPath;
		if (info.alias !== undefined) state.alias = info.alias;
		if (info.ctx !== undefined) state.ctx = info.ctx;
		if (info.mtpHeads !== undefined) state.mtpHeads = info.mtpHeads;
		if (info.reasoningEffort !== undefined) state.reasoningEffort = info.reasoningEffort;
		if (info.preserveThinking !== undefined) state.preserveThinking = info.preserveThinking;
		if (info.splitMode !== undefined) state.splitMode = info.splitMode;
		if (info.tensorSplit !== undefined) state.tensorSplit = info.tensorSplit;
		if (info.kvTypeK !== undefined) state.kvTypeK = info.kvTypeK;
		if (info.kvTypeV !== undefined) state.kvTypeV = info.kvTypeV;
		if (info.mmproj !== undefined) state.mmprojPath = info.mmproj;
		if (info.mmprojCpu !== undefined) state.mmprojCpu = info.mmprojCpu;
		if (info.cpuMoe !== undefined) state.cpuMoe = info.cpuMoe;
		if (info.nCpuMoe !== undefined) state.nCpuMoe = info.nCpuMoe;
		if (info.expertUsed !== undefined) state.expertUsed = info.expertUsed;
		if (info.presetsPath !== undefined) state.presetsPath = info.presetsPath;
		if (info.sleepIdleSecs !== undefined) state.sleepIdleSecs = info.sleepIdleSecs;
		state.loadedModels = [];
		state.port = PORT;
		state.startedAt = Date.now();
		state.error = null;

		const seq = ++runSeq;
		const deadline = Date.now() + START_TIMEOUT_S * 1000;
		healthTimer = setInterval(async () => {
			if (seq !== runSeq) { stopHealthPolling(); return; }
			if (await portUp()) {
				stopHealthPolling();
				state.status = "ready";
				state.error = null;
				return;
			}
			if (Date.now() > deadline) {
				stopHealthPolling();
				if (state.status === "starting") {
					state.status = "stopped";
					state.error = "server did not become healthy within " + START_TIMEOUT_S + "s — see " + LOG_PATH;
					const c = child;
					child = null;
					if (c) killChild(c);
				}
			}
		}, 1000);

		return { ok: true, pid: next.pid };
	}

	async function status() {
		const s = { ...state };
		const settings = readSettings();
		// Runtime card state: what the setting says, what it resolves to right
		// now, and which layer of the chain produced that path.
		s.binConfigured = settings.binPath;
		s.bin = currentBin();
		// Only a version we already probed (Runtime card's Check): status must
		// never spawn the binary, and the poll runs every 2 s.
		s.binVersion = s.bin.resolved ? cachedBinVersion(s.bin.resolved) : null;
		s.shortcuts = shortcutDirs().map((s) => {
			// exists is reported, never enforced: an unmounted disk must not make
			// its shortcut vanish from the list (or from the tab).
			let exists = false;
			try { exists = statSync(s.path).isDirectory(); } catch { exists = false; }
			return { ...s, exists };
		});
		s.shortcutsSource = settings.shortcuts !== null ? "settings" : (SHORTCUTS_ENV.length > 0 ? "env" : "none");
		// GPU inventory for the tab's VRAM budget + the multi-GPU controls.
		// peekGpuInventory never awaits nvidia-smi: a pending probe reports the
		// assumption and the next poll (2 s later) has the real numbers.
		const inventory = peekGpuInventory();
		const overrideBytes = settings.vramGb > 0 ? Math.round(settings.vramGb * 1024 ** 3) : 0;
		s.gpus = inventory.gpus;
		s.vramTotalBytes = overrideBytes || inventory.totalBytes || VRAM_ASSUMED_BYTES;
		s.vramSource = overrideBytes > 0
			? "override"
			: inventory.pending === true ? "probing" : (inventory.assumed ? "assumed" : inventory.source);
		s.vramOverrideGb = settings.vramGb;
		s.gpuCount = inventory.gpus.length;
		s.portUp = await portUp();
		s.uptimeMs = state.startedAt ? Date.now() - state.startedAt : null;
		try {
			s.presetsCount = cachedProfiles().length;
		} catch (err) {
			// A corrupt store must not break the poll; the tab keeps working
			// and the message explains the 0.
			s.presetsCount = 0;
			s.profilesError = err?.message ?? String(err);
		}
		s.routerMax = ROUTER_MAX;
		s.autostartRouter = readSettings().autostartRouter;
		s.autoUnloadMins = readSettings().autoUnloadMins;
		s.sleepIdleSecs = state.sleepIdleSecs;
		s.logSize = 0;
		try { s.logSize = statSync(LOG_PATH).size; } catch { /* no log yet */ }
		if (state.mode === "router" && s.portUp) {
			try {
				const res = await fetch("http://127.0.0.1:" + PORT + "/models", { signal: AbortSignal.timeout(1500) });
				if (res.ok) {
					const list = await res.json();
					const items = Array.isArray(list) ? list : (Array.isArray(list?.data) ? list.data : []);
					s.routerModels = items.map((m) => {
						const id = typeof m?.id === "string" ? m.id : (typeof m?.model === "string" ? m.model : null);
						const value = typeof m?.status?.value === "string" ? m.status.value : "unknown";
						const modes = Array.isArray(m?.architecture?.input_modalities)
							? m.architecture.input_modalities.filter((x) => x === "text" || x === "image")
							: null;
						return id === null ? null : { id, value, modes };
					}).filter(Boolean);
					s.loadedModels = s.routerModels.filter((m) => m.value !== "unloaded").map((m) => m.id);
				}
			} catch {
				s.loadedModels = [];
				s.routerModels = [];
			}
		}
		return s;
	}

	function dispose() {
		stopHealthPolling();
		const c = child;
		child = null;
		if (c) killChild(c);
	}

	return { run, stop, status, dispose, startRouter };
}
// ---------------------------------------------------------------------------
// HTTP helpers.
// ---------------------------------------------------------------------------
function readBody(req) {
	return new Promise((resolve, reject) => {
		let data = "";
		req.on("data", (chunk) => { data += chunk; });
		req.on("end", () => resolve(data));
		req.on("error", reject);
	});
}

function sendJson(res, code, body) {
	const payload = JSON.stringify(body);
	res.writeHead(code, {
		"content-type": "application/json; charset=utf-8",
		"cache-control": "no-store",
	});
	res.end(payload);
}

/** Roots the file endpoints (browse/gguf-meta/run, and profile saves) may
 * touch: the user's home, the browser shortcut dirs (Runtime card, else
 * LOCAL_MODELS_SHORTCUTS), the directories of the models already declared in
 * saved profiles — a saved profile must always stay loadable, whatever the
 * shortcut list says — and the llama.cpp dir when it really holds a
 * llama-server. Resolves lexically and requires containment so inputs can't
 * wander the filesystem.
 *
 * Nothing here is taken on faith: a path only widens the roots once it is a
 * directory the user configured, a profile that already exists, or a binary
 * that passes isLlamaServerPath(). */
function allowedRoots() {
	const roots = [homedir(), ...shortcutDirs().map((s) => s.path)];
	try {
		for (const profile of cachedProfiles()) {
			for (const declared of [profile?.modelPath, profile?.mmprojPath]) {
				if (typeof declared === "string" && declared) roots.push(dirname(declared));
			}
		}
	} catch {
		/* a corrupt store must not break the file endpoints */
	}
	// The llama.cpp tree: the configured directory (it must hold a llama-server
	// to have been saved) and the binary the chain actually resolves to, but
	// only when that binary is a real llama-server.
	const configured = readSettings().binPath;
	if (configured) {
		const abs = expandHome(configured, homedir());
		try {
			if (statSync(abs).isDirectory()) roots.push(abs);
		} catch {
			/* not on disk (yet) */
		}
	}
	const resolved = currentBin().resolved;
	if (resolved && isLlamaServerPath(resolved)) roots.push(resolved);
	// A root naming the binary itself (not a dir) also opens the folder holding
	// it, so "Up"/"Browse…" from the picked path keeps working. Checked on a
	// snapshot: pushing while iterating the live array never terminates.
	for (const r of [...roots]) {
		const abs = resolve(r);
		try {
			if (!statSync(abs).isDirectory()) roots.push(dirname(abs));
		} catch {
			/* nothing on disk to open */
		}
	}
	return roots;
}

/** A path we are willing to register as a browsable root: absolute, no ".."
 * segment, and not a filesystem root (which would list everything). Exported
 * for tests. */
export function isSafeRootPath(p) {
	if (typeof p !== "string" || !p.trim()) return false;
	const raw = p.trim();
	if (!isAbsolute(raw)) return false;
	// Check the raw segments: resolve() collapses ".." before we could see it.
	if (raw.split(/[\\/]+/).includes("..")) return false;
	const abs = resolve(raw);
	return dirname(abs) !== abs;
}

function isPathAllowed(p) {
	if (typeof p !== "string" || !p) return false;
	let abs;
	try {
		abs = resolve(p);
	} catch {
		return false;
	}
	return allowedRoots().some((root) => {
		let r;
		try {
			r = resolve(root);
		} catch {
			return false;
		}
		return abs === r || abs.startsWith(r + sep);
	});
}

function listDir(target, allFiles) {
	let st;
	try {
		st = statSync(target);
	} catch {
		return null;
	}
	if (!st.isDirectory()) return null;
	const entries = [];
	let truncated = false;
	for (const name of readdirSync(target)) {
		if (name.startsWith(".")) continue;
		const full = join(target, name);
		let s;
		try {
			s = statSync(full);
		} catch {
			continue;
		}
		if (s.isDirectory()) entries.push({ name, type: "dir" });
		else if (s.isFile() && (allFiles || /\.gguf$/i.test(name))) entries.push({ name, type: "file", size: s.size });
		if (entries.length > 2000) { truncated = true; break; }
	}
	entries.sort((a, b) => a.type === b.type ? a.name.localeCompare(b.name) : (a.type === "dir" ? -1 : 1));
	return { path: target, parent: dirname(target) === target ? null : dirname(target), entries, truncated };
}

// ---------------------------------------------------------------------------
// Provider profile: what gets written into llm-pi-ai when the user clicks
// "Register in dsh". Pure and exported so the vision-modality logic can be
// tested without a running server.
// ---------------------------------------------------------------------------
export function buildProviderProfile(st, requestedRoute) {
	const modelId = st.alias || "local-model";
	const fallbackRoute = "local-" + String(modelId).replace(/[^a-zA-Z0-9_-]+/g, "-");
	const route = typeof requestedRoute === "string" && /^[a-zA-Z0-9_-]+$/.test(requestedRoute)
		? requestedRoute
		: fallbackRoute;
	const vision = Boolean(st.mmprojPath);
	// Fall back to the single-mode launch default (8192), not a larger
	// window: the advertisement must never exceed what an equivalent run()
	// would actually get.
	const contextWindow = st.ctx ?? 8192;
	const profile = {
		api: "openai-completions",
		baseURL: "http://127.0.0.1:" + st.port + "/v1",
		apiKeyEnv: "LOCAL_MODELS_API_KEY",
		displayName: "Local llama-server — " + (st.model || modelId) + (vision ? " (vision)" : ""),
		// Provider-neutral default thinking level, kept in sync with the tab's
		// launch-time selection so dsh selectors show the same default.
		reasoning: normalizeEffort(st.reasoningEffort),
		// Route-level wire-compatibility switches. This llama-server build
		// honors per-request reasoning_effort and stream_options include_usage.
		compat: {
			supportsReasoningEffort: true,
			supportsUsageInStreaming: true,
		},
		models: [{
			id: modelId,
			name: modelId,
			contextWindow,
			// Capped at half the window (see MAX_OUTPUT_TOKENS): the qwen3.x
			// series is a heavy thinker and at xhigh its thinking block alone
			// blows past small ceilings, so 32K is the default — but never more
			// than half the window, or compaction's `W - O - headroom` budget
			// collapses and proactive compaction never fires. NOTE: under the
			// core defaults (headroom 65536) a 131072 window still thresholds
			// at ~32K — see compactionBudgetFor(); mid-size windows need a
			// per-model headroom override, not a smaller O. Need more room
			// for a monster thinking block? Raise per-request maxTokens
			// explicitly instead.
			maxTokens: defaultMaxOutput(contextWindow),
			// Declaring image input is what makes a hand-declared vision model
			// usable: without it dsh rejects image attachments ("model does not
			// declare image input"). mmprojPath is set when a --mmproj file was
			// loaded, i.e. the model is actually multimodal.
			input: vision ? ["text", "image"] : ["text"],
			// Probe-validated thinking levels: the chat template accepts only
			// low/medium/xhigh as reasoning_effort values ("none" disables
			// thinking); advertising others would make dsh send a value the
			// template rejects with HTTP 500.
			reasoningEfforts: {
				off: "none",
				low: "low",
				medium: "medium",
				xhigh: "xhigh",
			},
			compat: { supportsReasoningEffort: true },
		}],
	};
	if (vision) {
		// Vision guards: keep oversized base64 images from stalling sessions —
		// dsh degrades overlarge images to text placeholders instead of failing.
		profile.maxRequestImageBytes = IMAGE_MAX_BYTES;
		profile.requestImagePixelBudget = IMAGE_PIXEL_BUDGET;
		profile.requestImageMaxBytes = IMAGE_MAX_BYTES;
	}
	return { profile, modelId, route };
}

// ---------------------------------------------------------------------------
// Saved profiles: named configurations persisted to
// $DSH_HOME/local-models/profiles.json. Pure + exported for tests.
// ---------------------------------------------------------------------------
export function slugProfile(name) {
	const base = String(name).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
	if (base.length <= 40) return base || "profile";
	// Long names: truncating to 40 chars would let two distinct names that
	// share a prefix silently overwrite each other on upsert. Keep a prefix
	// plus a hash of the full name so the id stays deterministic and unique.
	let h = 0;
	for (let i = 0; i < base.length; i++) h = (h * 31 + base.charCodeAt(i)) >>> 0;
	return base.slice(0, 32) + "-" + h.toString(36);
}

export function profilesFile() {
	return join(dataDir(), "profiles.json");
}

export function readProfiles() {
	let raw;
	try {
		raw = readFileSync(profilesFile(), "utf8");
	} catch (err) {
		// No store yet = no profiles. Anything else (permissions, I/O) is
		// surfaced instead of masquerading as "no profiles".
		if (err?.code === "ENOENT") return [];
		throw new Error("cannot read profiles store: " + (err?.message ?? String(err)));
	}
	let list;
	try {
		list = JSON.parse(raw);
	} catch {
		throw new Error("profiles.json is corrupt: " + profilesFile());
	}
	return Array.isArray(list) ? list : [];
}

function writeJsonAtomic(file, value) {
	mkdirSync(dirname(file), { recursive: true });
	const tmp = file + ".tmp-" + process.pid;
	writeFileSync(tmp, JSON.stringify(value, null, 2), "utf8");
	renameSync(tmp, file);
}

function writeProfiles(list) {
	writeJsonAtomic(profilesFile(), list);
	presetsCache = list;
}

// Plugin settings (dataDir/settings.json). Missing or corrupt file =
// defaults; the toggle rewrites valid JSON on next change.
export function settingsFile() {
	return join(dataDir(), "settings.json");
}

export function normalizeSettings(raw) {
	// Idle-eviction timer (minutes, 0 = off). Default 30: an idle 27B pins
	// ~10 GB of the 16 GB card, while reloading costs tens of seconds —
	// 30 min is the "stepped away" boundary; shorter evicts between
	// back-to-back tasks, longer pins VRAM all evening.
	const mins = Number.isInteger(raw?.autoUnloadMins) && raw.autoUnloadMins >= 0
		? raw.autoUnloadMins
		: 30;
	// llama.cpp binaries (the Runtime card). "" = auto-detect through the
	// candidate chain; stored as typed (a dir or the binary itself) and never
	// validated here, so a stale path can't brick the settings file — status
	// reports it as unresolved and Load refuses with a readable error.
	const binPath = typeof raw?.binPath === "string" ? raw.binPath.trim() : "";
	// File-browser shortcuts (the Runtime card's folder list). null = never
	// configured, so LOCAL_MODELS_SHORTCUTS still applies; [] = the user took
	// the list over and emptied it (removing the last env-derived folder must
	// stick, not silently fall back to the env var). Non-strings and blanks are
	// dropped instead of breaking the 2 s status poll.
	const shortcuts = raw?.shortcuts === undefined || raw?.shortcuts === null
		? null
		: Array.isArray(raw.shortcuts)
			? raw.shortcuts
				.filter((p) => typeof p === "string" && p.trim())
				.map((p) => p.trim())
				.filter((p, i, a) => a.indexOf(p) === i)
				.slice(0, 12)
			: null;
	// VRAM budget override in GiB (0 = auto-detect). Only needed on hardware the
	// probes cannot read (or to cap a multi-GPU total by hand): the tab's
	// estimator otherwise uses what detectGpus() found, falling back to the
	// historic 16 GiB assumption.
	const vramGb = Number.isFinite(raw?.vramGb) && raw.vramGb > 0 && raw.vramGb <= 1024
		? Math.round(raw.vramGb * 100) / 100
		: 0;
	return { autostartRouter: raw?.autostartRouter === true, autoUnloadMins: mins, binPath, shortcuts, vramGb };
}

export function readSettings() {
	try {
		return normalizeSettings(JSON.parse(readFileSync(settingsFile(), "utf8")));
	} catch {
		return normalizeSettings(null);
	}
}

export function writeSettings(raw) {
	const next = normalizeSettings(raw);
	writeJsonAtomic(settingsFile(), next);
	return next;
}

// Cached profile list for the status poll (every ~2s): invalidated by
// writeProfiles. External edits (the ops skill) take effect on restart,
// which is already the documented flow for node-half changes.
let presetsCache = null;
function cachedProfiles() {
	if (presetsCache === null) presetsCache = readProfiles();
	return presetsCache;
}

/** Validate + normalize one profile config; throws on bad input. */
export function normalizeProfileConfig(raw) {
	const modelPath = typeof raw?.modelPath === "string" && raw.modelPath ? raw.modelPath : "";
	if (!modelPath || !existsSync(modelPath)) throw new Error("profile needs a modelPath that exists on disk");
	if (!isPathAllowed(modelPath)) throw new Error("modelPath is outside the folders this tab may read: " + modelPath + " — add its folder under Model folders in the Runtime card first");
	const ctx = Number.isInteger(raw?.ctx) && raw.ctx > 0 ? raw.ctx : 8192;
	const mtpHeads = Number.isInteger(raw?.mtpHeads) && raw.mtpHeads >= 0 ? Math.min(raw.mtpHeads, MTP_MAX) : 0;
	const mmprojPath = typeof raw?.mmprojPath === "string" && raw.mmprojPath ? raw.mmprojPath : null;
	if (mmprojPath && !existsSync(mmprojPath)) throw new Error("mmprojPath does not exist: " + mmprojPath);
	if (mmprojPath && !isPathAllowed(mmprojPath)) throw new Error("mmprojPath is outside the folders this tab may read: " + mmprojPath + " — add its folder under Model folders in the Runtime card first");
	const effort = normalizeEffort(raw?.effort);
	const preserveThinking = normalizePreserveThinking(raw);
	// Multi-GPU placement. Legacy profiles have neither field and keep
	// llama.cpp's own default (layer split, proportional tensor split).
	const splitMode = normalizeSplitMode(raw?.splitMode);
	const tensorSplit = normalizeTensorSplit(raw?.tensorSplit);
	// KV cache quantization pair (--cache-type-k / --cache-type-v). Legacy
	// profiles without either field keep the tuned default pair.
	const kvTypeK = normalizeKvCacheType(raw?.kvTypeK, KV_CACHE_DEFAULT_K);
	const kvTypeV = normalizeKvCacheType(raw?.kvTypeV, KV_CACHE_DEFAULT_V);
	// Vision projector placement (--no-mmproj-offload). Profiles saved before
	// this field existed launched with the projector on CPU, so missing/legacy
	// values default to true; an explicit false keeps vision on the GPU.
	const mmprojCpu = raw?.mmprojCpu === false ? false : raw?.mmprojCpu === true ? true : MMPROJ_CPU;
	// MoE launch config (arch is informational: which <arch>.expert_used_count
	// key a top-k override targets; re-validated from the GGUF at launch).
	const moe = normalizeMoEConfig(raw);
	return { modelPath, ctx, mtpHeads, mmprojPath, mmprojCpu, effort, preserveThinking, kvTypeK, kvTypeV, splitMode, tensorSplit, ...moe };
}

/** Save (or upsert by slugified name) one profile. Returns the saved profile. */
export function saveProfile(name, config) {
	const trimmed = typeof name === "string" ? name.trim() : "";
	if (!trimmed) throw new Error("profile needs a non-empty name");
	const normalized = normalizeProfileConfig(config);
	const id = slugProfile(trimmed);
	const entry = { id, name: trimmed, ...normalized, updatedAt: new Date().toISOString() };
	const list = readProfiles();
	const next = [entry, ...list.filter((p) => (p?.id ?? slugProfile(p?.name ?? "")) !== id)];
	writeProfiles(next);
	return entry;
}

/** Remove a profile by name (or id). Returns true if something was removed. */
export function removeProfile(name) {
	const id = slugProfile(name);
	const list = readProfiles();
	const next = list.filter((p) => (p?.id ?? slugProfile(p?.name ?? "")) !== id);
	if (next.length === list.length) return false;
	writeProfiles(next);
	return true;
}

// ---------------------------------------------------------------------------
// Router mode: generate llama-server router presets from saved profiles and
// build a multi-model dsh provider route. Pure + exported for tests.
// ---------------------------------------------------------------------------
export function generateRouterPresets(profiles) {
	const lines = [
		"[*]",
		"flash-attn = on",
		// Shared KV defaults; each profile may override them below (the preset
		// loader cascades [*] first, then the model's own section).
		"cache-type-k = " + KV_CACHE_DEFAULT_K,
		"cache-type-v = " + KV_CACHE_DEFAULT_V,
		"spec-draft-type-k = q4_0",
		"spec-draft-type-v = q4_0",
		"kv-unified = 1",
		"cont-batching = 1",
		"jinja = 1",
		"t = 4",
		// Single sequence: the daily use is one model at a time, and MTP
		// buffer sharing is cheapest at -np 1.
		"parallel = 1",
		"poll = 0",
		"",
	];
	for (const p of profiles) {
		const id = p.id || slugProfile(p.name);
		lines.push("[" + id + "]");
		lines.push("model = " + p.modelPath);
		lines.push("ctx-size = " + (p.ctx || 8192));
		lines.push("alias = " + id);
		lines.push("batch-size = 2048");
		lines.push("ubatch-size = 512");
		lines.push("n-gpu-layers = 999");
		// KV cache quantization, explicit per model (legacy profiles without
		// the fields land on the same pair the [*] section carries).
		lines.push("cache-type-k = " + normalizeKvCacheType(p.kvTypeK, KV_CACHE_DEFAULT_K));
		lines.push("cache-type-v = " + normalizeKvCacheType(p.kvTypeV, KV_CACHE_DEFAULT_V));
		// MoE expert placement: all experts in RAM (--cpu-moe) or the first
		// N layers' experts (--n-cpu-moe); dense profiles skip both.
		if (p.cpuMoe === true) lines.push("cpu-moe = 1");
		else if (Number.isInteger(p.nCpuMoe) && p.nCpuMoe > 0) lines.push("n-cpu-moe = " + p.nCpuMoe);
		// MoE top-k override (needs the profile's stored arch for the key).
		if (Number.isInteger(p.expertUsed) && p.expertUsed > 0 && p.arch) {
			lines.push("override-kv = " + p.arch + ".expert_used_count=int:" + p.expertUsed);
		}
		if (p.mmprojPath) {
			lines.push("mmproj = " + p.mmprojPath);
			// Qwen-VL needs a 1024-token image minimum (load_hparams warning).
			lines.push("image-min-tokens = 1024");
			// Vision projector placement, same default as single mode
			// (--no-mmproj-offload): legacy profiles without the field stay
			// on CPU; an explicit mmprojCpu: false keeps vision on the GPU.
			if (p.mmprojCpu !== false) lines.push("no-mmproj-offload = 1");
		}
		// MTP draft spec: unconditional whenever MTP is on — the old ctx
		// ceiling / "ignore softcap" toggle is gone.
		const mtp = Number.isInteger(p.mtpHeads) && p.mtpHeads > 0 ? Math.min(p.mtpHeads, MTP_MAX) : 0;
		// Preset keys are CLI args without the leading dashes; the default split
		// (layer, no proportions) stays implicit exactly like in single mode.
		const split = normalizeSplitMode(p.splitMode);
		const tensorSplit = normalizeTensorSplit(p.tensorSplit);
		const ctxSize = Number.isInteger(p.ctx) && p.ctx > 0 ? p.ctx : 8192;
		if (mtp > 0) {
			lines.push("spec-type = draft-mtp");
			lines.push("spec-draft-n-max = " + mtp);
			lines.push("spec-draft-p-min = " + MTP_P_MIN);
		}
		if (split !== SPLIT_MODE_DEFAULT) lines.push("split-mode = " + split);
		if (tensorSplit) lines.push("tensor-split = " + tensorSplit);
		if (p.effort === "off") lines.push("reasoning = off");
		else if (p.effort) lines.push("chat-template-kwargs = " + JSON.stringify({ reasoning_effort: p.effort }));
		// Reasoning-trace preservation: explicit either way (upstream defaults
		// to preserve ON, while this plugin's historical default is OFF), so
		// existing profiles without the field stay byte-identical (0).
		// Skipped when reasoning is off — there is no trace to preserve.
		if (p.effort !== "off") {
			const preserve = normalizePreserveThinking(p);
			lines.push("reasoning-preserve = " + (preserve ? "1" : "0"));
		}
		lines.push("");
	}
	return lines.join("\n");
}

/** Best-effort id -> input-modalities map from the running router. */
export async function routerModalities() {
	try {
		const res = await fetch("http://127.0.0.1:" + PORT + "/models", { signal: AbortSignal.timeout(1500) });
		if (!res.ok) return {};
		const list = await res.json();
		const items = Array.isArray(list) ? list : (Array.isArray(list?.data) ? list.data : []);
		const out = {};
		for (const m of items) {
			const id = typeof m?.id === "string" ? m.id : null;
			if (id === null) continue;
			const modes = (m?.architecture?.input_modalities || []).filter((x) => x === "text" || x === "image");
			if (modes.length > 0) out[id] = modes;
		}
		return out;
	} catch {
		return {};
	}
}

export function buildRouterProfile(profiles, modalitiesByModel) {
	const models = profiles.map((p) => {
		const id = p.id || slugProfile(p.name);
		const detected = modalitiesByModel && modalitiesByModel[id];
		const input = (detected && detected.length > 0)
			? detected
			: p.mmprojPath ? ["text", "image"] : ["text"];
		return {
			id,
			name: id,
			contextWindow: p.ctx ?? 8192,
			maxTokens: defaultMaxOutput(p.ctx ?? 8192),
			input,
			reasoningEfforts: { off: "none", low: "low", medium: "medium", xhigh: "xhigh" },
			compat: { supportsReasoningEffort: true },
		};
	});
	return {
		modelId: "router",
		route: "local-router",
		profile: {
			api: "openai-completions",
			baseURL: "http://127.0.0.1:" + PORT + "/v1",
			apiKeyEnv: "LOCAL_MODELS_API_KEY",
			displayName: "Local llama-server (router — " + profiles.length + " models)",
			reasoning: "medium",
			compat: { supportsReasoningEffort: true, supportsUsageInStreaming: true },
			models,
		},
	};
}

// ---------------------------------------------------------------------------
// Plugin apply: register routes + effect-owned process.
// ---------------------------------------------------------------------------
export function apply(ctx) {
	const manager = createManager();

	// Publish one provider route to dsh (shared by the register button and
	// boot autostart). Throws on settings failure.
	async function publishProvider(route, profile, modelId) {
		const API_KEY_REF = "LOCAL_MODELS_API_KEY";
		// Assumes ctx.settings.update() merges per provider key: a
		// wholesale replace here would wipe other providers, so if
		// registering a second route ever drops the first, this
		// call is the place to revisit.
		await ctx.settings.update("llm-pi-ai", { providers: { [route]: profile } });
		// llama-server accepts any key; provision a non-empty placeholder so the
		// provider is usable. Rejects if the live environment already supplies it.
		let keyNote = "";
		try {
			await ctx.credentials.set(API_KEY_REF, "local-llama-server");
			keyNote = "placeholder api key written to " + API_KEY_REF;
		} catch (ke) {
			keyNote = "api key from environment: " + (ke?.message ?? String(ke));
		}
		return { ok: true, route, modelId, profile, keyNote };
	}

	const routeSpec = [
		{
			kind: "exact",
			path: "/local-models/browse",
			handler: async (req, res) => {
				const url = new URL(req.url ?? "/", "http://x");
				const dir = url.searchParams.get("dir");
				const allFiles = url.searchParams.get("all") === "1";
				const target = dir && dir.indexOf("/") === 0
					? dir
					: dir ? join(homedir(), dir) : homedir();
				if (!isPathAllowed(target)) {
					sendJson(res, 400, { error: "path is outside the folders this tab may read (home + Model folders + the llama.cpp dir): " + target });
					return;
				}
				const listing = listDir(target, allFiles);
				if (listing === null) {
					sendJson(res, 400, { error: "not a readable directory: " + target });
					return;
				}
				sendJson(res, 200, listing);
			},
		},
		{
			kind: "exact",
			path: "/local-models/gguf-meta",
			handler: async (req, res) => {
				let body;
				try {
					body = JSON.parse((await readBody(req)) || "{}");
				} catch {
					sendJson(res, 400, { error: "invalid JSON body" });
					return;
				}
				const path = typeof body.path === "string" ? body.path : "";
				if (!path) {
					sendJson(res, 400, { error: "missing `path`" });
					return;
				}
				if (!isPathAllowed(path)) {
					sendJson(res, 400, { error: "path is outside the folders this tab may read (home + Model folders + the llama.cpp dir)" });
					return;
				}
				try {
					sendJson(res, 200, parseGGUFCached(path));
				} catch (err) {
					sendJson(res, 422, { error: err?.message ?? String(err) });
				}
			},
		},
		{
			kind: "exact",
			path: "/local-models/status",
			handler: async (_req, res) => sendJson(res, 200, await manager.status()),
		},
		{
			// Terminal tail: serve llama-server.log from a byte offset (no offset =
			// last `max` bytes). nextOffset is the cursor for the next poll.
			kind: "exact",
			path: "/local-models/logs",
			handler: async (req, res) => {
				const url = new URL(req.url ?? "/", "http://x");
				const maxParam = Number(url.searchParams.get("max"));
				const max = Number.isFinite(maxParam) && maxParam > 0
					? Math.min(Math.floor(maxParam), 1024 * 1024)
					: 256 * 1024;
				const offsetParam = url.searchParams.get("offset");
				const offset = offsetParam === null || offsetParam === "" ? null : Number(offsetParam);
				let size = 0;
				try { size = statSync(LOG_PATH).size; } catch { size = 0; }
				let start = 0;
				let text = "";
				let got = 0;
				if (size > 0) {
					// No valid offset, or the file shrank below it (restarted/rotated) → rebase on the tail.
					if (offset === null || !Number.isFinite(offset) || offset < 0 || offset >= size) {
						start = Math.max(0, size - max);
					} else {
						start = Math.max(0, Math.floor(offset));
					}
					const len = Math.min(max, size - start);
					if (len > 0) {
						try {
							const fd = openSync(LOG_PATH, "r");
							try {
								const buf = Buffer.alloc(len);
								got = readSync(fd, buf, 0, len, start);
								text = buf.toString("utf8", 0, got);
							} finally {
								closeSync(fd);
							}
						} catch {
							text = "";
							got = 0;
						}
					}
				}
				// Cursor for the next poll: bytes consumed, not the byte length
				// of the decoded slice (a slice starting/ending mid-UTF-8-char
				// decodes to U+FFFD, whose 3 re-encoded bytes would drift us).
				const nextOffset = start + got;
				sendJson(res, 200, { text, offset: start, nextOffset, size, atEOF: nextOffset >= size });
			},
		},
		{
			kind: "exact",
			path: "/local-models/run",
			handler: async (req, res) => {
				let body;
				try {
					body = JSON.parse((await readBody(req)) || "{}");
				} catch {
					sendJson(res, 400, { error: "invalid JSON body" });
					return;
				}
				const path = typeof body.path === "string" ? body.path : "";
				const ctx = body.ctx === undefined ? undefined : Number(body.ctx);
				const mtp = body.mtp === undefined ? 0 : Number(body.mtp);
				const mmproj = typeof body.mmproj === "string" && body.mmproj ? body.mmproj : null;
				const effort = typeof body.effort === "string" ? body.effort : undefined;
				const kvTypeK = normalizeKvCacheType(body.kvTypeK, KV_CACHE_DEFAULT_K);
				const kvTypeV = normalizeKvCacheType(body.kvTypeV, KV_CACHE_DEFAULT_V);
				const moe = {
					cpuMoe: body.cpuMoe === true,
					nCpuMoe: body.nCpuMoe === undefined ? 0 : Number(body.nCpuMoe),
					expertUsed: body.expertUsed === undefined || body.expertUsed === null || body.expertUsed === "" ? null : Number(body.expertUsed),
					arch: typeof body.arch === "string" ? body.arch : null,
				};
				if (!path) {
					sendJson(res, 400, { error: "missing `path`" });
					return;
				}
				if (!isPathAllowed(path)) {
					sendJson(res, 400, { error: "model path is outside the folders this tab may read: " + path + " — add its folder under Model folders in the Runtime card" });
					return;
				}
				if (mmproj && !isPathAllowed(mmproj)) {
					sendJson(res, 400, { error: "mmproj path is outside the folders this tab may read: " + mmproj + " — add its folder under Model folders in the Runtime card" });
					return;
				}
				const result = await manager.run(path, Number.isFinite(ctx) ? ctx : 8192, Number.isFinite(mtp) ? mtp : 0, mmproj, effort, typeof body.mmprojCpu === "boolean" ? body.mmprojCpu : MMPROJ_CPU, moe, normalizePreserveThinking(body), kvTypeK, kvTypeV, normalizeSplitMode(body.splitMode), normalizeTensorSplit(body.tensorSplit));
				sendJson(res, result.ok ? 202 : 409, result);
			},
		},
		{
			kind: "exact",
			path: "/local-models/stop",
			handler: async (_req, res) => {
				const stopped = await manager.stop();
				sendJson(res, 200, { stopped });
			},
		},
		{
			kind: "exact",
			path: "/local-models/profiles",
			handler: async (req, res) => {
				if (req.method === "POST") {
					let body;
					try {
						body = JSON.parse((await readBody(req)) || "{}");
					} catch {
						sendJson(res, 400, { error: "invalid JSON body" });
						return;
					}
					try {
						const profile = saveProfile(
							typeof body.name === "string" ? body.name : "",
							{
								modelPath: body.modelPath,
								ctx: body.ctx,
								mtpHeads: body.mtpHeads,
								mmprojPath: body.mmprojPath,
								mmprojCpu: body.mmprojCpu,
								effort: body.effort,
								preserveThinking: normalizePreserveThinking(body),
								kvTypeK: body.kvTypeK,
								kvTypeV: body.kvTypeV,
								splitMode: body.splitMode,
								tensorSplit: body.tensorSplit,
								cpuMoe: body.cpuMoe,
								nCpuMoe: body.nCpuMoe,
								expertUsed: body.expertUsed,
								arch: body.arch,
							},
						);
						sendJson(res, 200, { ok: true, profile, profiles: readProfiles() });
					} catch (err) {
						sendJson(res, 400, { error: err?.message ?? String(err) });
					}
					return;
				}
				try {
					sendJson(res, 200, { profiles: readProfiles() });
				} catch (err) {
					sendJson(res, 500, { error: err?.message ?? String(err) });
				}
			},
		},
		{
			kind: "exact",
			path: "/local-models/profiles/remove",
			handler: async (req, res) => {
				if (req.method !== "POST") {
					sendJson(res, 405, { error: "POST required" });
					return;
				}
				let body;
				try {
					body = JSON.parse((await readBody(req)) || "{}");
				} catch {
					sendJson(res, 400, { error: "invalid JSON body" });
					return;
				}
				let removed;
				let profiles;
				try {
					removed = removeProfile(typeof body.name === "string" ? body.name : "");
					profiles = readProfiles();
				} catch (err) {
					sendJson(res, 500, { error: err?.message ?? String(err) });
					return;
				}
				sendJson(res, 200, { ok: true, removed, profiles });
			},
		},
		{
			kind: "exact",
			path: "/local-models/settings",
			handler: async (req, res) => {
				if (req.method === "POST") {
					let body;
					try {
						body = JSON.parse((await readBody(req)) || "{}");
					} catch {
						sendJson(res, 400, { error: "invalid JSON body" });
						return;
					}
					const patch = {};
					if (body.autostartRouter !== undefined) {
						if (body.autostartRouter !== true && body.autostartRouter !== false) {
							sendJson(res, 400, { error: "autostartRouter must be true or false" });
							return;
						}
						patch.autostartRouter = body.autostartRouter;
					}
					if (body.autoUnloadMins !== undefined) {
						if (!Number.isInteger(body.autoUnloadMins) || body.autoUnloadMins < 0) {
							sendJson(res, 400, { error: "autoUnloadMins must be a non-negative integer (minutes, 0 = off)" });
							return;
						}
						patch.autoUnloadMins = body.autoUnloadMins;
					}
					// llama.cpp binaries: "" = auto-detect. A non-empty value must
					// resolve to an existing llama-server, so a typo can't be saved
					// and then fail at the next Load. The Check button lets the user
					// test a path (and read its --version) before committing it.
					if (body.binPath !== undefined) {
						if (typeof body.binPath !== "string") {
							sendJson(res, 400, { error: "binPath must be a string (a directory or the llama-server binary; \"\" = auto-detect)" });
							return;
						}
						const wanted = body.binPath.trim();
						if (wanted) {
							const probe = resolveBin({
								configured: wanted,
								home: homedir(),
								search: false,
							});
							if (!probe.resolved) {
								sendJson(res, 400, { error: "no llama-server found at " + wanted + " — expected the directory holding it (e.g. ~/llama.cpp/build/bin) or the binary itself" });
								return;
							}
							// Existing is not enough: it must BE a llama-server. This is
							// also what keeps a stray file from becoming a browse root.
							const bad = binIdentityError(probe.resolved);
							if (bad) {
								sendJson(res, 400, { error: bad + " — point at the directory holding llama-server (e.g. ~/llama.cpp/build/bin) or at the binary itself" });
								return;
							}
						}
						patch.binPath = wanted;
					}
					// File-browser shortcuts are browsable roots, so they are the one
					// place where a bad value widens the file endpoints: absolute,
					// no "..", never the filesystem root, at most 12 entries.
					// Missing dirs are still kept (an unmounted disk must not
					// silently lose its shortcut); the tab marks them as missing.
					if (body.shortcuts !== undefined) {
						if (!Array.isArray(body.shortcuts) || body.shortcuts.some((p) => typeof p !== "string")) {
							sendJson(res, 400, { error: "shortcuts must be an array of directory paths" });
							return;
						}
						const list = body.shortcuts.map((p) => p.trim()).filter(Boolean);
						if (list.length > 12) {
							sendJson(res, 400, { error: "at most 12 shortcut folders (got " + list.length + ")" });
							return;
						}
						const bad = list.find((p) => !isSafeRootPath(p));
						if (bad) {
							sendJson(res, 400, { error: "not a usable folder: " + bad + " — shortcuts must be absolute paths, without \"..\", and not the filesystem root" });
							return;
						}
						patch.shortcuts = list;
					}
					// VRAM budget override in GiB (0 = auto-detect through the GPU
					// probes). Bounded so a typo cannot produce a nonsense budget.
					if (body.vramGb !== undefined) {
						if (typeof body.vramGb !== "number" || !Number.isFinite(body.vramGb) || body.vramGb < 0 || body.vramGb > 1024) {
							sendJson(res, 400, { error: "vramGb must be a number between 0 (auto) and 1024" });
							return;
						}
						patch.vramGb = body.vramGb;
					}
					sendJson(res, 200, { ok: true, settings: writeSettings({ ...readSettings(), ...patch }) });
					return;
				}
				sendJson(res, 200, { settings: readSettings() });
			},
		},
		{
			// Runtime card's "Check": resolve + `<bin> --version`, without saving.
			// Body { binPath } ("" = whatever is in effect now).
			kind: "exact",
			path: "/local-models/runtime/check",
			handler: async (req, res) => {
				if (req.method !== "POST") { sendJson(res, 405, { error: "POST required" }); return; }
				let body;
				try {
					body = JSON.parse((await readBody(req)) || "{}");
				} catch {
					sendJson(res, 400, { error: "invalid JSON body" });
					return;
				}
				if (body.binPath !== undefined && typeof body.binPath !== "string") {
					sendJson(res, 400, { error: "binPath must be a string" });
					return;
				}
				const override = typeof body.binPath === "string" ? body.binPath.trim() : "";
				// An explicit override is tested on its own (no fallback to PATH
				// or the home build dirs), otherwise the user could "verify" a
				// path that is not the one that would be launched.
				const bin = override
					? resolveBin({ configured: override, home: homedir(), search: false })
					: currentBin();
				// Identity first: a green check has to mean "this is a llama-server
				// that starts", not "some file printed a banner".
				const identity = bin.resolved
					? binIdentityError(bin.resolved)
					: "no " + BIN_NAME + " found in " + (override || "the search path");
				const version = identity ? null : await probeBin(bin.resolved);
				sendJson(res, 200, {
					ok: !identity && (!version || version.ok),
					error: identity,
					bin,
					version,
				});
			},
		},
		{
			kind: "exact",
			path: "/local-models/router/start",
			handler: async (req, res) => {
				if (req.method !== "POST") { sendJson(res, 405, { error: "POST required" }); return; }
				let profiles;
				try {
					profiles = readProfiles();
				} catch (err) {
					sendJson(res, 500, { error: err?.message ?? String(err) });
					return;
				}
				if (profiles.length === 0) { sendJson(res, 400, { error: "save at least one profile first" }); return; }
				try {
					mkdirSync(dataDir(), { recursive: true });
					writeFileSync(ROUTER_PRESETS_FILE, generateRouterPresets(profiles), "utf8");
					const result = await manager.startRouter(ROUTER_PRESETS_FILE);
					sendJson(res, result.ok ? 200 : 409, { ok: result.ok, error: result.error, presets: profiles.length, iniPath: ROUTER_PRESETS_FILE });
				} catch (err) {
					sendJson(res, 500, { error: err?.message ?? String(err) });
				}
			},
		},
		{
			kind: "exact",
			path: "/local-models/router/unload-all",
			handler: async (req, res) => {
				if (req.method !== "POST") { sendJson(res, 405, { error: "POST required" }); return; }
				try {
					const resp = await fetch("http://127.0.0.1:" + PORT + "/models", { signal: AbortSignal.timeout(1500) });
					const list = await resp.json().catch(() => []);
					const items = Array.isArray(list) ? list : (Array.isArray(list?.data) ? list.data : []);
					const results = [];
					for (const m of items) {
						if (typeof m?.status?.value !== "string" || m.status.value === "unloaded") continue;
						const id = typeof m?.id === "string" ? m.id : null;
						if (!id) continue;
						try {
							const r = await fetch("http://127.0.0.1:" + PORT + "/models/unload", {
								method: "POST",
								headers: { "content-type": "application/json" },
								body: JSON.stringify({ model: id }),
								signal: AbortSignal.timeout(15000),
							});
							results.push({ id, ok: r.ok });
						} catch (err) {
							results.push({ id, ok: false, error: err?.message ?? String(err) });
						}
					}
					sendJson(res, 200, { ok: true, unloaded: results.filter((r) => r.ok).length, results });
				} catch (err) {
					sendJson(res, 502, { error: err?.message ?? String(err) });
				}
			},
		},
		{
			kind: "exact",
			path: "/local-models/router/unload",
			handler: async (req, res) => {
				if (req.method !== "POST") { sendJson(res, 405, { error: "POST required" }); return; }
				let body;
				try { body = JSON.parse((await readBody(req)) || "{}"); } catch { sendJson(res, 400, { error: "invalid JSON body" }); return; }
				const model = typeof body.model === "string" && body.model ? body.model : "";
				if (!model) { sendJson(res, 400, { error: "missing model id" }); return; }
				try {
					const r = await fetch("http://127.0.0.1:" + PORT + "/models/unload", {
						method: "POST",
						headers: { "content-type": "application/json" },
						body: JSON.stringify({ model }),
						signal: AbortSignal.timeout(15000),
					});
					const payload = await r.json().catch(() => ({}));
					sendJson(res, r.ok ? 200 : 502, { ok: r.ok, payload });
				} catch (err) {
					sendJson(res, 502, { error: err?.message ?? String(err) });
				}
			},
		},
		{
			kind: "exact",
			path: "/local-models/register",
			handler: async (req, res) => {
				let body;
				try {
					body = JSON.parse((await readBody(req)) || "{}");
				} catch {
					sendJson(res, 400, { error: "invalid JSON body" });
					return;
				}
				const st = await manager.status();
				if (st.status !== "ready") {
					sendJson(res, 409, { error: "the server is not ready — wait for /health then register" });
					return;
				}
				let savedProfiles = [];
				if (st.mode === "router") {
					try {
						savedProfiles = readProfiles();
					} catch (err) {
						sendJson(res, 500, { error: err?.message ?? String(err) });
						return;
					}
				}
				const { profile, modelId, route } = st.mode === "router"
					? buildRouterProfile(savedProfiles, await routerModalities())
					: buildProviderProfile(st, typeof body.route === "string" ? body.route : undefined);
				try {
					sendJson(res, 200, await publishProvider(route, profile, modelId));
				} catch (err) {
					sendJson(res, 500, { error: "settings update failed: " + (err?.message ?? err) });
				}
			},
		},
	];

	ctx.effect(() => {
		const disposers = routeSpec.map((route) => ctx.webServer.register(route));
		return () => {
			for (const dispose of disposers) dispose();
			manager.dispose();
		};
	}, "local-models: routes");

	// Boot autostart: when the user enabled "start router with dsh", launch
	// the router in the background and register its route once healthy, so
	// models are usable without opening the tab. Fire-and-forget: every step
	// is logged to llama-server.log (visible in the tab's terminal tail).
	(async () => {
		const note = (msg) => {
			try { appendFileSync(LOG_PATH, "[autostart] " + msg + "\n"); } catch { /* ignore */ }
		};
		try {
			if (!readSettings().autostartRouter) return;
			let profiles;
			try {
				profiles = readProfiles();
			} catch (err) {
				note("aborted: " + (err?.message ?? String(err)));
				return;
			}
			if (profiles.length === 0) {
				note("aborted: autostart enabled but no profiles saved");
				return;
			}
			note("starting router (" + profiles.length + " profiles)...");
			mkdirSync(dataDir(), { recursive: true });
			writeFileSync(ROUTER_PRESETS_FILE, generateRouterPresets(profiles), "utf8");
			const result = await manager.startRouter(ROUTER_PRESETS_FILE);
			if (!result.ok) {
				note("router failed to start: " + (result.error ?? "unknown error"));
				return;
			}
			const deadline = Date.now() + (START_TIMEOUT_S + 30) * 1000;
			for (;;) {
				const st = await manager.status();
				if (st.status === "ready") break;
				if (st.status === "stopped" || Date.now() > deadline) {
					note("router never became ready (status=" + st.status + (st.error ? ": " + st.error : "") + ")");
					return;
				}
				await new Promise((r) => setTimeout(r, 2000));
			}
			const built = buildRouterProfile(profiles, await routerModalities());
			await publishProvider(built.route, built.profile, built.modelId);
			note("router ready and registered as " + built.route);
		} catch (err) {
			note("failed: " + (err?.message ?? String(err)));
		}
	})();
}
// Self-test: node lib/index.js /path/to/model.gguf
// Compares against the resolved invocation path (not string-concatenated
// argv[1], which mismatches on relative paths and symlinks).
const invokedAsScript = (() => {
	try {
		return !!process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1]);
	} catch {
		return false;
	}
})();
if (invokedAsScript && process.argv[2]) {
	try {
		console.log(JSON.stringify(parseGGUFCached(process.argv[2]), null, 2));
	} catch (e) {
		console.error(e?.message ?? String(e));
		process.exitCode = 2;
	}
}
