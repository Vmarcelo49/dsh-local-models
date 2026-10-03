// Unit tests for the pure, exported helpers in lib/index.js.
// Run: npm test (node's built-in runner). No server, no models needed.
import { describe, it, before } from "node:test";
import assert from "node:assert/strict";
import { chmodSync, mkdirSync, mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";

// Isolate the profiles store: lib/index.js resolves its data dir from
// DSH_HOME at import time, so point it at a temp dir before importing.
const TMP = mkdtempSync(join(tmpdir(), "dsh-local-models-test-"));
process.env.DSH_HOME = join(TMP, ".dsh");
// Read at import time too: the env fallback for the browser shortcuts, which
// the Runtime card's own list must be able to take over (and empty out). The
// temp sandbox is one of them, because profile paths must live inside a
// browsable root (allowedRoots) — that is what keeps a profile save from
// widening the file endpoints.
process.env.LOCAL_MODELS_SHORTCUTS = "models=/mnt/models:" + TMP;
const TMP_SHORTCUT = { label: basename(TMP), path: TMP };
const FAKE_MODEL = join(TMP, "model.gguf");
writeFileSync(FAKE_MODEL, "gguf-stub");

const lib = await import("../lib/index.js");

describe("normalizeKvCacheType", () => {
	it("passes known ids through", () => {
		assert.equal(lib.normalizeKvCacheType("q4_0", "q5_0"), "q4_0");
		assert.equal(lib.normalizeKvCacheType("f16", "q5_0"), "f16");
	});
	it("falls back on unknown / non-string values", () => {
		assert.equal(lib.normalizeKvCacheType("bogus", "q5_0"), "q5_0");
		assert.equal(lib.normalizeKvCacheType(undefined, "q4_1"), "q4_1");
		assert.equal(lib.normalizeKvCacheType(null, "q4_1"), "q4_1");
	});
});

describe("kvTypesMustMatch", () => {
	it("requires matching types for MLA / deepseek4", () => {
		assert.equal(lib.kvTypesMustMatch({ isMla: true }), true);
		assert.equal(lib.kvTypesMustMatch({ arch: "deepseek4" }), true);
	});
	it("allows mixed types otherwise", () => {
		assert.equal(lib.kvTypesMustMatch({}), false);
		assert.equal(lib.kvTypesMustMatch(null), false);
		assert.equal(lib.kvTypesMustMatch({ arch: "qwen3" }), false);
	});
});

describe("thinkingArgsFor", () => {
	it("disables reasoning for off", () => {
		assert.deepEqual(lib.thinkingArgsFor("off"), ["--reasoning", "off"]);
	});
	it("passes the level through with preserve flags", () => {
		const args = lib.thinkingArgsFor("medium", true);
		assert.ok(args.includes("--reasoning-preserve"));
		assert.ok(args.includes("medium"));
		const args2 = lib.thinkingArgsFor("xhigh", false);
		assert.ok(args2.includes("--no-reasoning-preserve"));
	});
	it("normalizes unknown levels to medium", () => {
		assert.ok(lib.thinkingArgsFor("bogus").includes("medium"));
	});
});

describe("normalizeMoEConfig", () => {
	it("defaults to dense/stock behavior", () => {
		assert.deepEqual(lib.normalizeMoEConfig({}), {
			cpuMoe: false, nCpuMoe: 0, expertUsed: null, arch: null,
		});
	});
	it("rejects non-positive / non-integer inputs", () => {
		const cfg = lib.normalizeMoEConfig({ cpuMoe: "yes", nCpuMoe: -2, expertUsed: 0 });
		assert.deepEqual(cfg, { cpuMoe: false, nCpuMoe: 0, expertUsed: null, arch: null });
	});
});

describe("moeArgsFor", () => {
	it("returns [] for dense models", () => {
		assert.deepEqual(lib.moeArgsFor({ cpuMoe: true }, null), []);
		assert.deepEqual(lib.moeArgsFor({ cpuMoe: true }, { isMoe: false }), []);
	});
	it("emits placement flags for MoE models", () => {
		const meta = { isMoe: true, arch: "qwen3moe", expertCount: 128 };
		assert.deepEqual(lib.moeArgsFor({ cpuMoe: true }, meta), ["--cpu-moe"]);
		assert.deepEqual(lib.moeArgsFor({ nCpuMoe: 8 }, meta), ["--n-cpu-moe", "8"]);
		assert.deepEqual(
			lib.moeArgsFor({ expertUsed: 4, arch: "qwen3moe" }, meta),
			["--override-kv", "qwen3moe.expert_used_count=int:4"]);
	});
	it("throws when top-k exceeds the expert count or arch is unknown", () => {
		const meta = { isMoe: true, arch: "qwen3moe", expertCount: 8 };
		assert.throws(() => lib.moeArgsFor({ expertUsed: 16, arch: "qwen3moe" }, meta), /exceeds/);
		assert.throws(() => lib.moeArgsFor({ expertUsed: 4 }, { isMoe: true }), /unknown architecture/);
	});
});

describe("moeVramWeights", () => {
	it("passes dense weights through", () => {
		assert.equal(lib.moeVramWeights(100, null, null), 100);
	});
	it("subtracts CPU-resident experts", () => {
		const meta = { isMoe: true, expertBytes: 30, expertBytesByLayer: { 0: 10, 1: 10, 2: 10 } };
		assert.equal(lib.moeVramWeights(100, meta, { cpuMoe: true }), 70);
		assert.equal(lib.moeVramWeights(100, meta, { nCpuMoe: 2 }), 80);
	});
	it("returns null when the split is unknowable", () => {
		assert.equal(lib.moeVramWeights(100, { isMoe: true }, { cpuMoe: true }), null);
	});
});

describe("generateRouterPresets", () => {
	const profile = (over = {}) => ({
		id: "qwen", name: "qwen", modelPath: FAKE_MODEL, ctx: 32768,
		mtpHeads: 2, effort: "medium", ...over,
	});
	it("emits one section per profile with the MTP draft", () => {
		const ini = lib.generateRouterPresets([profile()]);
		assert.ok(ini.includes("[qwen]"));
		assert.ok(ini.includes("ctx-size = 32768"));
		assert.ok(ini.includes("spec-type = draft-mtp"));
		assert.ok(ini.includes("reasoning-preserve = 0"));
	});
	it("keeps the mmproj on CPU like single mode", () => {
		const ini = lib.generateRouterPresets([profile({ mmprojPath: "/tmp/mmproj.gguf" })]);
		assert.ok(ini.includes("mmproj = /tmp/mmproj.gguf"));
		assert.ok(ini.includes("no-mmproj-offload = 1"));
		const plain = lib.generateRouterPresets([profile()]);
		assert.ok(!plain.includes("mmproj"));
	});
	it("keeps the draft on at any ctx (no softcap) and clamps the depth to 7", () => {
		const big = lib.generateRouterPresets([profile({ ctx: 262144 })]);
		assert.ok(big.includes("spec-type = draft-mtp"));
		assert.ok(big.includes("spec-draft-n-max = 2"));
		// A legacy profile carrying the removed "ignore softcap" flag behaves
		// exactly the same: the draft is unconditional now.
		const nocap = lib.generateRouterPresets([profile({ ctx: 262144, ignoreCtxCap: true })]);
		assert.equal(nocap, big);
		// The ceiling moved from 3 to 7 (upstream clamps to the model's nextn).
		const deep = lib.generateRouterPresets([profile({ mtpHeads: 7 })]);
		assert.ok(deep.includes("spec-draft-n-max = 7"));
		const beyond = lib.generateRouterPresets([profile({ mtpHeads: 12 })]);
		assert.ok(beyond.includes("spec-draft-n-max = 7"));
	});
});

describe("slugProfile", () => {
	it("slugifies short names and falls back on empty", () => {
		assert.equal(lib.slugProfile("Qwen 3 27B!"), "qwen-3-27b");
		assert.equal(lib.slugProfile(""), "profile");
	});
	it("keeps long names with a shared prefix distinct", () => {
		const prefix = "a".repeat(50);
		const a = lib.slugProfile(prefix + "-one");
		const b = lib.slugProfile(prefix + "-two");
		assert.notEqual(a, b);
		assert.ok(a.length <= 40 && b.length <= 40);
	});
});

describe("buildProviderProfile / buildRouterProfile", () => {
	it("defaults ctx to the single-mode 8192 and caps maxTokens below the window", () => {
		const { profile } = lib.buildProviderProfile({ alias: "m", port: 8080 });
		assert.equal(profile.models[0].contextWindow, 8192);
		// Half the window: the advertised maxTokens becomes compaction's
		// reserved output `O` in `W - O - headroom`, so `O = W` would leave
		// no message budget and proactive compaction would never fire.
		assert.equal(profile.models[0].maxTokens, 4096);
		// Full-window (daily 131K) launches cap at the 32K adapter default
		// instead of the whole window; xhigh thinking past that must raise
		// per-request maxTokens explicitly.
		const big = lib.buildProviderProfile({ alias: "m", port: 8080, ctx: 131072 });
		assert.equal(big.profile.models[0].maxTokens, 32768);
	});
	it("router models follow the same rule", () => {
		const { profile } = lib.buildRouterProfile([{ id: "m", ctx: 8192 }], {});
		assert.equal(profile.models[0].contextWindow, 8192);
		assert.equal(profile.models[0].maxTokens, 4096);
		const big = lib.buildRouterProfile([{ id: "m", ctx: 131072 }], {});
		assert.equal(big.profile.models[0].maxTokens, 32768);
	});
});

describe("compactionBudgetFor", () => {
	it("prices the 131K profile at a ~32K threshold under core defaults", () => {
		// The reported loop: W = 131072, O = 32768, headroom 65536 →
		// threshold = min(104857, 32768) = 32768, so compaction fires at
		// ~32-38K pressure tokens and xhigh thinking refills it every turn.
		const b = lib.compactionBudgetFor(131072);
		assert.equal(b.maxTokens, 32768);
		assert.equal(b.messageBudget, 98304);
		assert.equal(b.pressureBudget, 32768);
		assert.equal(b.thresholdTokens, 32768);
		assert.equal(b.retainTokens, Math.floor(98304 * 0.16));
		assert.equal(b.viable, true);
	});
	it("flags the 96K-window cliff as not viable", () => {
		// W = 98304, O = 32768 leaves pressure exactly 0: proactive
		// compaction is disabled entirely (core warns once, recovers only
		// on overflow).
		const b = lib.compactionBudgetFor(98304);
		assert.equal(b.pressureBudget, 0);
		assert.equal(b.viable, false);
	});
	it("recommends a headroom that lands the threshold near 70% of W", () => {
		for (const W of [98304, 131072, 196608, 256000]) {
			const b = lib.compactionBudgetFor(W);
			assert.ok(b.recommendedHeadroom >= 4096 && b.recommendedHeadroom <= 65536);
			const tuned = lib.compactionBudgetFor(W, { headroomTokens: b.recommendedHeadroom });
			assert.equal(tuned.viable, true);
			// ~70% target: accept rounding / the 0.8-ratio ceiling.
			assert.ok(tuned.thresholdTokens >= Math.floor(W * 0.6), `W=${W} threshold=${tuned.thresholdTokens}`);
		}
		// Spot check the 131K headline number: headroom ~6.5K → ~92K threshold.
		const b = lib.compactionBudgetFor(131072);
		assert.equal(b.recommendedHeadroom, 98304 - Math.floor(131072 * 0.7));
		const tuned = lib.compactionBudgetFor(131072, { headroomTokens: b.recommendedHeadroom });
		assert.equal(tuned.thresholdTokens, 98304 - b.recommendedHeadroom);
	});
	it("honors explicit maxTokens / headroom overrides", () => {
		const b = lib.compactionBudgetFor(131072, { maxTokens: 16384, headroomTokens: 16384 });
		assert.equal(b.messageBudget, 114688);
		assert.equal(b.thresholdTokens, Math.min(Math.floor(131072 * 0.8), 114688 - 16384));
	});
});

describe("profile store", () => {
	before(() => {
		rmSync(lib.profilesFile(), { force: true });
		rmSync(lib.settingsFile(), { force: true });
	});
	it("round-trips save/remove with a temp DSH_HOME", () => {
		const saved = lib.saveProfile("Test Model", { modelPath: FAKE_MODEL, ctx: 8192 });
		assert.equal(saved.id, "test-model");
		assert.deepEqual(lib.readProfiles().map((p) => p.id), ["test-model"]);
		// Upsert by slug: saving again replaces instead of duplicating.
		lib.saveProfile("Test Model", { modelPath: FAKE_MODEL, ctx: 32768 });
		assert.equal(lib.readProfiles().length, 1);
		assert.equal(lib.readProfiles()[0].ctx, 32768);
		assert.equal(lib.removeProfile("Test Model"), true);
		assert.deepEqual(lib.readProfiles(), []);
		assert.equal(lib.removeProfile("Test Model"), false);
	});
	it("returns [] when no store exists", () => {
		rmSync(lib.profilesFile(), { force: true });
		assert.deepEqual(lib.readProfiles(), []);
	});
	it("throws (instead of returning []) on a corrupt store", () => {
		writeFileSync(lib.profilesFile(), "{not json", "utf8");
		assert.throws(() => lib.readProfiles(), /corrupt/);
		rmSync(lib.profilesFile(), { force: true });
	});
	it("rejects empty names and missing model files", () => {
		assert.throws(() => lib.saveProfile("  ", { modelPath: FAKE_MODEL }), /non-empty name/);
		assert.throws(
			() => lib.saveProfile("x", { modelPath: join(TMP, "nope.gguf") }),
			/modelPath/);
	});
});

describe("plugin settings", () => {
	// Defaults: no router autostart, 30 min idle eviction, binary auto-detected
	// (binPath ""), browser shortcuts not configured yet (null → env fallback).
	const DEFAULTS = { autostartRouter: false, autoUnloadMins: 30, binPath: "", shortcuts: null, vramGb: 0 };
	it("normalizes to known shape", () => {
		assert.deepEqual(lib.normalizeSettings(null), DEFAULTS);
		assert.deepEqual(lib.normalizeSettings({}), DEFAULTS);
		assert.deepEqual(lib.normalizeSettings({ autostartRouter: true }), { ...DEFAULTS, autostartRouter: true });
		assert.deepEqual(lib.normalizeSettings({ autostartRouter: "yes" }), DEFAULTS);
	});
	it("keeps an explicit timer, including off (0)", () => {
		assert.deepEqual(lib.normalizeSettings({ autoUnloadMins: 0 }), { ...DEFAULTS, autoUnloadMins: 0 });
		assert.deepEqual(lib.normalizeSettings({ autoUnloadMins: 15 }), { ...DEFAULTS, autoUnloadMins: 15 });
	});
	it("falls back to the 30 min default on invalid timers", () => {
		for (const bad of [-5, 2.5, "30", null]) {
			assert.deepEqual(lib.normalizeSettings({ autoUnloadMins: bad }), DEFAULTS);
		}
	});
	it("defaults when no settings file exists", () => {
		rmSync(lib.settingsFile(), { force: true });
		assert.deepEqual(lib.readSettings(), DEFAULTS);
	});
	it("round-trips both settings", () => {
		assert.deepEqual(
			lib.writeSettings({ autostartRouter: true, autoUnloadMins: 15 }),
			{ ...DEFAULTS, autostartRouter: true, autoUnloadMins: 15 });
		assert.deepEqual(lib.readSettings(), { ...DEFAULTS, autostartRouter: true, autoUnloadMins: 15 });
		assert.deepEqual(
			lib.writeSettings({ autostartRouter: false, autoUnloadMins: 0 }),
			{ ...DEFAULTS, autoUnloadMins: 0 });
		assert.deepEqual(lib.readSettings(), { ...DEFAULTS, autoUnloadMins: 0 });
		rmSync(lib.settingsFile(), { force: true });
	});
	it("falls back to defaults on a corrupt file", () => {
		writeFileSync(lib.settingsFile(), "{not json", "utf8");
		assert.deepEqual(lib.readSettings(), DEFAULTS);
		rmSync(lib.settingsFile(), { force: true });
	});
	it("keeps the llama.cpp path and the shortcut folder list", () => {
		const next = lib.normalizeSettings({
			binPath: "  /opt/llama.cpp/build/bin  ",
			shortcuts: ["/mnt/models/GGUF", "  ", 42, "/mnt/models/GGUF", "/data/gguf"],
		});
		assert.equal(next.binPath, "/opt/llama.cpp/build/bin");
		// Blank/non-string entries dropped, duplicates collapsed.
		assert.deepEqual(next.shortcuts, ["/mnt/models/GGUF", "/data/gguf"]);
	});
	it("keeps a sane VRAM override and rejects the rest", () => {
		assert.equal(lib.normalizeSettings({ vramGb: 24 }).vramGb, 24);
		assert.equal(lib.normalizeSettings({ vramGb: 15.94 }).vramGb, 15.94);
		for (const bad of [0, -4, "16", null, NaN, 4096]) {
			assert.equal(lib.normalizeSettings({ vramGb: bad }).vramGb, 0, "vramGb " + String(bad));
		}
	});
	it("ignores a non-string binPath and a non-array shortcuts list", () => {
		assert.equal(lib.normalizeSettings({ binPath: 42 }).binPath, "");
		assert.equal(lib.normalizeSettings({ shortcuts: "a:b" }).shortcuts, null);
		// An explicitly empty list is NOT the same as never configured: it is
		// what stops LOCAL_MODELS_SHORTCUTS from coming back.
		assert.deepEqual(lib.normalizeSettings({ shortcuts: [] }).shortcuts, []);
	});
	it("round-trips binPath + shortcuts and keeps them on unrelated writes", () => {
		lib.writeSettings({ binPath: "/opt/bin", shortcuts: ["/data/gguf"] });
		assert.deepEqual(lib.readSettings().shortcuts, ["/data/gguf"]);
		assert.equal(lib.readSettings().binPath, "/opt/bin");
		// The /settings route merges readSettings() before writing: a patch that
		// only touches the timer must not clear the runtime fields.
		const merged = lib.writeSettings({ ...lib.readSettings(), autoUnloadMins: 5 });
		assert.equal(merged.binPath, "/opt/bin");
		assert.deepEqual(merged.shortcuts, ["/data/gguf"]);
		assert.equal(merged.autoUnloadMins, 5);
		rmSync(lib.settingsFile(), { force: true });
	});
});

describe("shortcutDirs", () => {
	it("falls back to LOCAL_MODELS_SHORTCUTS until the tab saves its own list", () => {
		rmSync(lib.settingsFile(), { force: true });
		assert.deepEqual(lib.shortcutDirs(), [{ label: "models", path: "/mnt/models" }, TMP_SHORTCUT]);
	});
	it("uses the saved list, with labels from the dir basename", () => {
		lib.writeSettings({ ...lib.readSettings(), shortcuts: ["/data/gguf", "/mnt/models/GGUF"] });
		assert.deepEqual(lib.shortcutDirs(), [
			{ label: "gguf", path: "/data/gguf" },
			{ label: "GGUF", path: "/mnt/models/GGUF" },
		]);
	});
	it("an explicitly empty list overrides the env var for good", () => {
		lib.writeSettings({ ...lib.readSettings(), shortcuts: [] });
		assert.deepEqual(lib.shortcutDirs(), []);
		// An unrelated write must not resurrect the env fallback either.
		lib.writeSettings({ ...lib.readSettings(), autoUnloadMins: 7 });
		assert.deepEqual(lib.shortcutDirs(), []);
		rmSync(lib.settingsFile(), { force: true });
	});
	it("a corrupt settings file degrades to the env fallback", () => {
		writeFileSync(lib.settingsFile(), "{not json", "utf8");
		assert.deepEqual(lib.shortcutDirs(), [{ label: "models", path: "/mnt/models" }, TMP_SHORTCUT]);
		rmSync(lib.settingsFile(), { force: true });
	});
});

describe("parseShortcutSpec", () => {
	it("parses colon-separated dirs with optional labels", () => {
		assert.deepEqual(lib.parseShortcutSpec("/mnt/models/GGUF"), [{ label: "GGUF", path: "/mnt/models/GGUF" }]);
		assert.deepEqual(lib.parseShortcutSpec("models=/mnt/models/GGUF:/data/gguf"), [
			{ label: "models", path: "/mnt/models/GGUF" },
			{ label: "gguf", path: "/data/gguf" },
		]);
	});
	it("is empty for an empty/absent spec, dropping blank entries", () => {
		assert.deepEqual(lib.parseShortcutSpec(""), []);
		assert.deepEqual(lib.parseShortcutSpec(undefined), []);
		assert.deepEqual(lib.parseShortcutSpec(" : :/data/x:"), [{ label: "x", path: "/data/x" }]);
	});
});

describe("expandHome", () => {
	it("expands a leading ~/ against the given home", () => {
		assert.equal(lib.expandHome("~/build/bin", "/home/u"), "/home/u/build/bin");
		assert.equal(lib.expandHome("~", "/home/u"), "/home/u");
		assert.equal(lib.expandHome("/abs/path", "/home/u"), "/abs/path");
	});
	it("trims and leaves ~ alone when there is no home to expand against", () => {
		assert.equal(lib.expandHome("  /abs  ", "/home/u"), "/abs");
		assert.equal(lib.expandHome("~/x", ""), "~/x");
		assert.equal(lib.expandHome(null, "/home/u"), "");
	});
});

describe("binCandidates", () => {
	// isDir is injected: no dependence on what is installed on this machine.
	const noDirs = () => false;
	it("orders the chain: setting → env → PATH → home build dirs → system", () => {
		const list = lib.binCandidates({
			configured: "/cfg/bin",
			env: "/env/llama-server",
			pathEnv: "/usr/bin:/opt/bin",
			home: "/home/u",
			isDir: noDirs,
		});
		assert.deepEqual(list.map((c) => c.source), ["setting", "env", "path", "path", "home", "home", "home", "system"]);
		assert.deepEqual(list.slice(0, 3).map((c) => c.path), ["/cfg/bin", "/env/llama-server", "/usr/bin/llama-server"]);
		assert.ok(list.some((c) => c.path === "/home/u/llama.cpp/build/bin/llama-server"));
		// /usr/bin came from PATH first, so the system layer only adds the rest.
		assert.ok(list.some((c) => c.path === "/usr/local/bin/llama-server"));
		// Every layer is home-relative or an OS convention: no absolute path of
		// any particular machine may appear here.
		for (const c of list) {
			assert.ok(
				c.path.startsWith("/cfg") || c.path.startsWith("/env") || c.path.startsWith("/opt")
				|| c.path.startsWith("/home/u") || c.path.startsWith("/usr"),
				"unexpected absolute candidate: " + c.path);
		}
	});
	it("appends the binary name when an entry names a directory", () => {
		const list = lib.binCandidates({
			configured: "~/llama.cpp/build/bin",
			home: "/home/u",
			pathEnv: "",
			isDir: (p) => p === "/home/u/llama.cpp/build/bin",
		});
		assert.equal(list[0].path, "/home/u/llama.cpp/build/bin/llama-server");
	});
	it("accepts the binary itself and a trailing slash as a directory", () => {
		const list = lib.binCandidates({
			configured: "/opt/llama.cpp/build/bin/llama-server",
			pathEnv: "",
			home: "",
			isDir: noDirs,
		});
		assert.equal(list[0].path, "/opt/llama.cpp/build/bin/llama-server");
		const slash = lib.binCandidates({ configured: "/opt/bin/", pathEnv: "", home: "", isDir: noDirs });
		assert.equal(slash[0].path, "/opt/bin/llama-server");
	});
	it("drops empty entries and dedupes", () => {
		const list = lib.binCandidates({
			configured: "",
			env: "",
			pathEnv: "::/usr/bin:",
			home: "",
			isDir: noDirs,
		});
		assert.equal(list[0].path, "/usr/bin/llama-server");
		assert.equal(list.filter((c) => c.path === "/usr/bin/llama-server").length, 1);
	});
	it("adds no home or system layer when there is no home", () => {
		const list = lib.binCandidates({ configured: "/opt/bin", pathEnv: "", home: "", isDir: noDirs });
		assert.deepEqual(list.map((c) => c.source), ["setting", "system", "system"]);
		assert.equal(list[0].path, "/opt/bin");
	});
	it("search: false tests one path only (the Runtime Check button)", () => {
		// LOCAL_MODELS_BIN and the rest of the chain must not be able to mask a
		// bad path: "Check" on a typed value reports on that value alone.
		const list = lib.binCandidates({
			configured: "/nope",
			env: "/env/llama-server",
			pathEnv: "/usr/bin",
			home: "/home/u",
			search: false,
			isDir: noDirs,
		});
		assert.deepEqual(list.map((c) => c.path), ["/nope"]);
		// Still expands ~ so a typed "~/llama.cpp/build/bin" is testable.
		const tilde = lib.binCandidates({ configured: "~/b", home: "/home/u", search: false, isDir: noDirs });
		assert.deepEqual(tilde.map((c) => c.path), ["/home/u/b"]);
	});
});

describe("resolveBin", () => {
	const isDir = (p) => p === "/opt/llama.cpp/build/bin";
	it("picks the first candidate that exists and reports its provenance", () => {
		const bin = lib.resolveBin({
			configured: "/opt/llama.cpp/build/bin",
			env: "/env/llama-server",
			pathEnv: "/usr/bin",
			home: "/home/u",
			isDir,
			isFile: (p) => p === "/env/llama-server",
		});
		assert.equal(bin.resolved, "/env/llama-server");
		assert.equal(bin.source, "env");
		assert.equal(bin.sourceLabel, "LOCAL_MODELS_BIN");
	});
	it("resolves a home build dir when nothing is configured", () => {
		const bin = lib.resolveBin({
			pathEnv: "",
			home: "/home/u",
			isDir: () => false,
			isFile: (p) => p === "/home/u/Projetos/llama.cpp/build/bin/llama-server",
		});
		assert.equal(bin.resolved, "/home/u/Projetos/llama.cpp/build/bin/llama-server");
		assert.equal(bin.source, "home");
	});
	it("returns null when nothing exists, keeping the search list", () => {
		const bin = lib.resolveBin({ configured: "/nope", pathEnv: "", home: "", isDir: () => false, isFile: () => false });
		assert.equal(bin.resolved, null);
		assert.equal(bin.source, null);
		// The system dirs stay in the list: an empty home must not shrink the
		// search space the error message reports.
		assert.equal(bin.candidates[0].path, "/nope");
		assert.ok(bin.candidates.some((c) => c.path === "/usr/bin/llama-server"));
	});
	it("reflects the configured value verbatim, trimmed", () => {
		const bin = lib.resolveBin({ configured: "  /opt/bin  ", pathEnv: "", home: "", isDir, isFile: () => false });
		assert.equal(bin.configured, "/opt/bin");
	});
});


describe("GPU inventory", () => {
	const MIB = 1024 * 1024;
	it("parses nvidia-smi csv (MiB → bytes) and skips junk", () => {
		const out = [
			"NVIDIA GeForce RTX 4090, 24564",
			"NVIDIA GeForce RTX 3060, 12288",
		].join("\n");
		assert.deepEqual(lib.parseNvidiaSmi(out), [
			{ vendor: "nvidia", name: "NVIDIA GeForce RTX 4090", vramBytes: 24564 * MIB },
			{ vendor: "nvidia", name: "NVIDIA GeForce RTX 3060", vramBytes: 12288 * MIB },
		]);
		assert.deepEqual(lib.parseNvidiaSmi(""), []);
		assert.deepEqual(lib.parseNvidiaSmi("name, memory.total\nGPU, 8192"), [
			{ vendor: "nvidia", name: "GPU", vramBytes: 8192 * MIB },
		]);
		// Missing/zero memory is dropped, not reported as a 0-byte GPU.
		assert.deepEqual(lib.parseNvidiaSmi("GPU, [N/A]\nGPU2, 0"), []);
	});
	it("resolves a PCI id against pci.ids", () => {
		const ids = [
			"# comment",
			"1002  Advanced Micro Devices, Inc. [AMD/ATI]",
			"\t7550  Navi 48 [Radeon RX 9070/9070 XT/9070 GRE]",
			"\t\t1eae 8811  Subsystem",
			"10de  NVIDIA Corporation",
			"\t2684  AD104 [GeForce RTX 4070]",
		].join("\n");
		assert.equal(lib.lookupPciName(ids, "1002", "7550"), "Navi 48 [Radeon RX 9070/9070 XT/9070 GRE]");
		assert.equal(lib.lookupPciName(ids, "10de", "2684"), "AD104 [GeForce RTX 4070]");
		// A device id from the other vendor's section must not match.
		assert.equal(lib.lookupPciName(ids, "1002", "2684"), null);
		assert.equal(lib.lookupPciName(ids, "1002", "ffff"), null);
		assert.equal(lib.lookupPciName("", "1002", "7550"), null);
		assert.equal(lib.lookupPciName(ids, null, null), null);
	});
	it("reads AMD cards from sysfs, ignoring other vendors and render nodes", () => {
		const files = {
			"/drm/card0/device/vendor": "0x1002",
			"/drm/card0/device/mem_info_vram_total": "17095983104",
			"/drm/card0/device/uevent": "DRIVER=amdgpu\nPCI_ID=1002:7550\n",
			"/drm/card1/device/vendor": "0x8086",
			"/drm/card1/device/mem_info_vram_total": "1024",
		};
		const gpus = lib.readAmdGpus({
			drmDir: "/drm",
			listDir: () => ["card0", "card1", "renderD128", "card0-DP-1", "version"],
			read: (p) => {
				if (!(p in files)) throw new Error("ENOENT " + p);
				return files[p];
			},
		});
		assert.equal(gpus.length, 1);
		assert.equal(gpus[0].vendor, "amd");
		assert.equal(gpus[0].vramBytes, 17095983104);
		assert.equal(gpus[0].card, "card0");
	});
	it("tolerates a missing /sys/class/drm", () => {
		assert.deepEqual(lib.readAmdGpus({ listDir: () => { throw new Error("ENOENT"); } }), []);
	});
	it("honors CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES", () => {
		const gpus = [{ vendor: "nvidia", name: "a" }, { vendor: "nvidia", name: "b" }, { vendor: "nvidia", name: "c" }];
		assert.deepEqual(lib.filterVisibleGpus(gpus, undefined), gpus);
		assert.deepEqual(lib.filterVisibleGpus(gpus, "all"), gpus);
		assert.deepEqual(lib.filterVisibleGpus(gpus, "1,2"), [gpus[1], gpus[2]]);
		assert.deepEqual(lib.filterVisibleGpus(gpus, ""), []);
		assert.deepEqual(lib.filterVisibleGpus(gpus, "-1"), []);
		// UUID/MIG entries cannot be mapped from here: keep the full list.
		assert.deepEqual(lib.filterVisibleGpus(gpus, "GPU-abc123"), gpus);
	});
	it("detects NVIDIA first, then AMD, then stays unknown", async () => {
		const nv = await lib.detectGpus({
			runner: async () => ({ err: null, stdout: "RTX 4090, 24564\nRTX 4090, 24564", stderr: "" }),
			env: {},
		});
		assert.equal(nv.source, "nvidia-smi");
		assert.equal(nv.gpus.length, 2);
		assert.equal(nv.totalBytes, 2 * 24564 * MIB);
		assert.equal(nv.assumed, false);

		// nvidia-smi missing → sysfs (AMD), with the VRAM totals summed.
		const amdFiles = {
			"/drm/card0/device/vendor": "0x1002",
			"/drm/card0/device/mem_info_vram_total": "17179869184",
			"/drm/card0/device/uevent": "PCI_ID=1002:7550",
			"/drm/card1/device/vendor": "0x1002",
			"/drm/card1/device/mem_info_vram_total": "17179869184",
			"/drm/card1/device/uevent": "PCI_ID=1002:7550",
		};
		const amd = await lib.detectGpus({
			runner: async () => ({ err: new Error("ENOENT"), stdout: "", stderr: "" }),
			env: { HIP_VISIBLE_DEVICES: "0" },
			drmDir: "/drm",
			listDir: () => ["card0", "card1"],
			read: (p) => {
				if (!(p in amdFiles)) throw new Error("ENOENT " + p);
				return amdFiles[p];
			},
		});
		assert.equal(amd.source, "sysfs");
		assert.equal(amd.gpus.length, 1); // HIP_VISIBLE_DEVICES=0 hides card1
		assert.equal(amd.totalBytes, 17179869184);

		const unknown = await lib.detectGpus({
			runner: async () => ({ err: new Error("ENOENT"), stdout: "", stderr: "" }),
			env: {},
			listDir: () => [],
		});
		assert.deepEqual(unknown.gpus, []);
		assert.equal(unknown.assumed, true);
		assert.equal(unknown.totalBytes, 0);
	});
});

describe("multi-GPU split", () => {
	it("normalizes the split mode, defaulting to upstream's layer", () => {
		for (const m of ["layer", "row", "tensor", "none"]) assert.equal(lib.normalizeSplitMode(m), m);
		for (const bad of ["bogus", "", null, 7, undefined]) assert.equal(lib.normalizeSplitMode(bad), "layer");
	});
	it("normalizes tensor-split proportions", () => {
		assert.equal(lib.normalizeTensorSplit("3,1"), "3,1");
		assert.equal(lib.normalizeTensorSplit(" 3 / 1 "), "3,1");
		assert.equal(lib.normalizeTensorSplit("1"), "1");
		for (const bad of ["", "  ", "a,b", "1,0", "1,-2", null, 3, "1,,x"]) {
			assert.equal(lib.normalizeTensorSplit(bad), null, String(bad));
		}
	});
	it("emits no args for the default (historic argv stays identical)", () => {
		assert.deepEqual(lib.splitArgsFor("layer", null), []);
		assert.deepEqual(lib.splitArgsFor(undefined, undefined), []);
		assert.deepEqual(lib.splitArgsFor("layer", ""), []);
	});
	it("emits --split-mode / --tensor-split when configured", () => {
		assert.deepEqual(lib.splitArgsFor("row", null), ["--split-mode", "row"]);
		assert.deepEqual(lib.splitArgsFor("row", "3,1"), ["--split-mode", "row", "--tensor-split", "3,1"]);
		assert.deepEqual(lib.splitArgsFor("none", null), ["--split-mode", "none"]);
		// Proportions alone still mean something with the default layer split.
		assert.deepEqual(lib.splitArgsFor("layer", "1,1"), ["--tensor-split", "1,1"]);
		assert.deepEqual(lib.splitArgsFor("bogus", "junk"), []);
	});
	it("prefills even proportions from the detected VRAM", () => {
		assert.equal(lib.evenTensorSplit([{ vramBytes: 24 }, { vramBytes: 24 }]), "1,1");
		assert.equal(lib.evenTensorSplit([{ vramBytes: 24576 }, { vramBytes: 12288 }]), "2,1");
		assert.equal(lib.evenTensorSplit([{ vramBytes: 16 }]), "");
		assert.equal(lib.evenTensorSplit([]), "");
		assert.equal(lib.evenTensorSplit(null), "");
	});
});

describe("path guards", () => {
	// Each fixture lives in its own dir and is named exactly like upstream's
	// binary: the name is part of the guard, not an accident of the fixture.
	let n = 0;
	const makeBin = (mode) => {
		const dir = join(TMP, "guards", String(++n));
		mkdirSync(dir, { recursive: true });
		const file = join(dir, "llama-server");
		writeFileSync(file, "#!/bin/sh\n");
		chmodSync(file, mode);
		return file;
	};
	it("only a llama-server-named executable may be launched", () => {
		assert.equal(lib.isLlamaServerPath(process.execPath), false); // not named llama-server
		assert.equal(lib.isLlamaServerPath(join(TMP, "absent", "llama-server")), false);
		assert.equal(lib.isLlamaServerPath(null), false);
		const notExec = makeBin(0o644);
		assert.equal(lib.isLlamaServerPath(notExec), false, "not executable yet");
		assert.equal(lib.isLlamaServerPath(makeBin(0o755)), true);
	});
	it("explains why a path is not usable, name first", () => {
		assert.match(lib.binIdentityError("/bin/echo"), /not a llama-server binary/);
		assert.match(lib.binIdentityError(""), /no llama-server path/);
		assert.match(lib.binIdentityError(makeBin(0o644)), /not executable/);
		assert.equal(lib.binIdentityError(makeBin(0o755)), null);
	});
	it("rejects browsable roots that would open the whole filesystem", () => {
		for (const bad of ["/", "//", "/etc/../..", "relative/path", "", "  ", null, 42, "/mnt/../etc"]) {
			assert.equal(lib.isSafeRootPath(bad), false, String(bad));
		}
		for (const ok of ["/mnt/models", "/mnt/raid0/GGUF", TMP]) {
			assert.equal(lib.isSafeRootPath(ok), true, ok);
		}
	});
});

describe("sleepIdleArgsFor / sleepIdleSecsFor", () => {
	it("maps minutes to --sleep-idle-seconds", () => {
		assert.equal(lib.sleepIdleSecsFor(30), 1800);
		assert.deepEqual(lib.sleepIdleArgsFor(30), ["--sleep-idle-seconds", "1800"]);
		assert.deepEqual(lib.sleepIdleArgsFor(5), ["--sleep-idle-seconds", "300"]);
	});
	it("is empty when off or invalid", () => {
		assert.equal(lib.sleepIdleSecsFor(0), null);
		assert.deepEqual(lib.sleepIdleArgsFor(0), []);
		assert.deepEqual(lib.sleepIdleArgsFor(-1), []);
		assert.deepEqual(lib.sleepIdleArgsFor("30"), []);
	});
});

describe("parseGGUF vocab fallback", () => {
	// Minimal synthetic GGUF v3 (header + one dummy tensor + 64 B of data) —
	// enough for the header-only parser, so no multi-GB model is needed.
	// `arch` gets no <arch>.vocab_size, exactly like a Qwen3.5/3.8-family file.
	function syntheticGguf({ arch = "qwen35", tokens = 300, vocabSize = null } = {}) {
		const u32 = (n) => { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b; };
		const u64 = (n) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(n)); return b; };
		const str = (s) => { const b = Buffer.from(s, "utf8"); return Buffer.concat([u64(b.length), b]); };
		const kvStr = (k, v) => Buffer.concat([str(k), u32(8), str(v)]); // type 8 = string
		const kvU32 = (k, v) => Buffer.concat([str(k), u32(4), u32(v)]); // type 4 = uint32
		// type 9 = array, `elems` already encoded with the right element type
		const kvArray = (k, elemType, elems) => Buffer.concat(
			[str(k), u32(9), u32(elemType), u64(elems.length)].concat(elems));
		const kv = [kvStr("general.architecture", arch), kvU32(arch + ".block_count", 4)];
		if (vocabSize != null) kv.push(kvU32(arch + ".vocab_size", vocabSize));
		if (tokens != null) {
			// The three big arrays a real tokenizer writes, in llama.cpp's order
			// (i32 token_type between the two string arrays). merges is one
			// shorter, so mixing up the two counts cannot pass unnoticed.
			kv.push(kvArray("tokenizer.ggml.tokens", 8, Array.from({ length: tokens }, (_, i) => str("tok" + i))));
			kv.push(kvArray("tokenizer.ggml.token_type", 5, Array.from({ length: tokens }, () => u32(1))));
			kv.push(kvArray("tokenizer.ggml.merges", 8, Array.from({ length: Math.max(tokens - 1, 0) }, (_, i) => str("a b" + i))));
		}
		// KV pairs after the token arrays: reading them proves the parser walked
		// each payload to exactly the right byte instead of skipping it.
		kv.push(kvU32("tokenizer.ggml.eos_token_id", 1), kvU32(arch + ".context_length", 4096));
		const header = Buffer.concat([
			Buffer.from("GGUF", "ascii"), u32(3), u64(1), u64(kv.length), ...kv,
			str("token_embd.weight"), u32(1), u64(tokens ?? 1), u32(0), u64(0),
		]);
		const pad = (32 - (header.length % 32)) % 32;
		return Buffer.concat([header, Buffer.alloc(pad + 64)]);
	}
	let seq = 0;
	function parse(buf) {
		const path = join(TMP, "synthetic-" + seq++ + ".gguf");
		writeFileSync(path, buf);
		return lib.parseGGUFCached(path).meta;
	}
	it("counts tokenizer.ggml.tokens when <arch>.vocab_size is absent", () => {
		const meta = parse(syntheticGguf({ tokens: 300 }));
		assert.equal(meta.nVocab, 300);
		assert.equal(meta.contextLength, 4096); // key after the array still lands
	});
	it("prefers an explicit <arch>.vocab_size", () => {
		assert.equal(parse(syntheticGguf({ tokens: 300, vocabSize: 151936 })).nVocab, 151936);
	});
	it("stays null (and warns) when neither source is present", () => {
		const meta = parse(syntheticGguf({ tokens: null }));
		assert.equal(meta.nVocab, null);
		assert.ok(meta.warnings.some((w) => /vocab size/i.test(w)));
	});
	it("walks big arrays without keeping their elements", () => {
		// 100k entries per array: the string arrays would be materialised as
		// 100k JS strings each before the fix. Only counts are used, and the
		// KV pairs after them must still be read (tokens, not merges, wins).
		const meta = parse(syntheticGguf({ tokens: 100000 }));
		assert.equal(meta.nVocab, 100000);
		assert.equal(meta.contextLength, 4096);
	});
});
