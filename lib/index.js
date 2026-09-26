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
 *   POST  /local-models/register        → add the ready server as an llm-pi-ai
 *                                          provider route via the settings service
 *
 * GGUF parsing is a port of the header-only parser from the user's
 * vram-calculator project (src/gguf.ts) over Node fs instead of Blob slices.
 */
import { spawn, execSync } from "node:child_process";
import {
	appendFileSync,
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
import { basename, join, dirname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { homedir } from "node:os";
import process from "node:process";

export const inject = ["settings", "credentials", "webServer"];

const PORT_RAW = Number(process.env.LOCAL_MODELS_PORT ?? 8080);
const PORT = Number.isFinite(PORT_RAW) && PORT_RAW > 0 && PORT_RAW < 65536
	? Math.floor(PORT_RAW)
	: 8080;
const BIN = process.env.LOCAL_MODELS_BIN
	?? "/home/marcelo/Projetos/llama.cpp/build/bin/llama-server";

/** File-browser shortcut buttons (colon-separated dirs). Label defaults to the
 * dir's basename; "name=path" entries allow custom labels. */
const DIR_SHORTCUTS = (process.env.LOCAL_MODELS_SHORTCUTS ?? "/mnt/raid0/GGUF")
	.split(":").map((s) => s.trim()).filter(Boolean).map((s) => {
		const eq = s.indexOf("=");
		return eq > 0
			? { label: s.slice(0, eq).trim() || "dir", path: s.slice(eq + 1).trim() }
			: { label: basename(s) || s, path: s };
	}).filter((s) => s.path);
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
 * Fixed MTP draft (upstream --spec-type draft-mtp) stays on up to this ctx
 * ceiling; above it the draft is dropped. Upstream has no draft-KV
 * auto-quant, and fixed depth > 3 collapses at large ctx (measured: fixed
 * n=6 at 131K ctx decodes ~12 tok/s vs ~57 at n=3 — per-step VRAM thrash),
 * so depth is capped at 3 everywhere below.
 */
const VRAM_CTX_LIMIT = 131072;
/**
 * Max output advertised when a server is registered in dsh: 131K tokens,
 * the full daily window. The qwen3.x series is a heavy thinker — at xhigh
 * the thinking block alone regularly exceeds the old 32K default and the
 * reply gets cut off — so registration offers the whole window as max
 * output. Still capped at the launched ctx in the profile builders below:
 * a small-ctx profile must never advertise more output than it has room
 * for.
 */
const MAX_OUTPUT_TOKENS = 131072;
/** Fixed-MTP ceiling: upstream fixed draft depth above 3 collapses at large
 * ctx (see VRAM_CTX_LIMIT). The tab offers 0-3; the API clamps here too. */
const MTP_MAX = 3;
/** Draft acceptance floor for fixed MTP (validated daily value). */
const MTP_P_MIN = 0.75;
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
		ignoreCtxCap: false,
		kvTypeK: KV_CACHE_DEFAULT_K,
		kvTypeV: KV_CACHE_DEFAULT_V,
		cpuMoe: false,
		nCpuMoe: 0,
		expertUsed: null,
		mode: "single", // single | router
		presetsPath: null,
		loadedModels: [],
		mmprojPath: null,
		port: PORT,
		startedAt: null,
		error: null,
		logPath: LOG_PATH,
		sleepIdleSecs: null, // effective idle-eviction timer (null = off)
	};
	let child = null;
	let healthTimer = null;
	let runSeq = 0;

	function stopHealthPolling() {
		if (healthTimer !== null) { clearInterval(healthTimer); healthTimer = null; }
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

	async function run(modelPath, ctxSize, mtpHeads, mmprojPath, effort, mmprojCpu = MMPROJ_CPU, moe = null, ignoreCtxCap = false, preserveThinking = false, kvTypeK = KV_CACHE_DEFAULT_K, kvTypeV = KV_CACHE_DEFAULT_V) {
		if (!existsSync(BIN)) return { ok: false, error: "llama-server binary missing: " + BIN };
		if (!existsSync(modelPath)) return { ok: false, error: "model file missing: " + modelPath };
		if (mmprojPath && !existsSync(mmprojPath)) return { ok: false, error: "mmproj file missing: " + mmprojPath };
		if (await portUp()) return { ok: false, error: "a server is already listening on port " + PORT + " — Stop it first" };

		const ctx = Number.isInteger(ctxSize) && ctxSize > 0 ? ctxSize : 8192;
		// Fixed MTP draft (upstream draft-mtp): any depth 0-3 is valid, but
		// clamp — fixed depth > 3 collapses at large ctx (see VRAM_CTX_LIMIT).
		const mtp = Number.isInteger(mtpHeads) && mtpHeads > 0 ? Math.min(mtpHeads, MTP_MAX) : 0;
		const alias = slug(basename(modelPath));
		const mmproj = typeof mmprojPath === "string" && mmprojPath ? mmprojPath : null;
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
		if (kvK !== kvV && kvTypesMustMatch(moeMeta)) {
			return { ok: false, error: "this model (MLA latent KV) needs the same cache type for K and V — set both to " + kvK };
		}
		// MTP draft is only enabled up to VRAM_CTX_LIMIT: above it the f16
		// draft KV risks VRAM eviction. Draft KV is pinned to Q4_0 (upstream
		// has no draft auto-quant; draft tokens are cheaply discarded).
		// ignoreCtxCap (the tab's "ignore softcap" checkbox) forces the
		// draft on anyway — may OOM or collapse decode on big ctx.
		const noCap = ignoreCtxCap === true;
		const specArgs = mtp > 0 && (ctx <= VRAM_CTX_LIMIT || noCap)
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
			...specArgs,
			...moeArgs,
			// Qwen-VL needs a 1024-token image minimum (load_hparams warning).
			...(mmproj ? ["--mmproj", mmproj, "--image-min-tokens", "1024", ...(mmprojCpu ? ["--no-mmproj-offload"] : [])] : []),
			"--alias", alias,
			"--host", "127.0.0.1",
			"--port", String(PORT),
			// Idle VRAM eviction (server-managed; wakes on next request).
			...sleepIdleArgsFor(readSettings().autoUnloadMins),
		];

		return launch(args, {
			mode: "single",
			modelLabel: basename(modelPath),
			modelPath,
			alias,
			ctx,
			mtpHeads: mtp,
			reasoningEffort: effortLevel,
			preserveThinking: preserve,
			ignoreCtxCap: noCap,
			mmproj,
			cpuMoe: moeCfg.cpuMoe,
			nCpuMoe: moeCfg.nCpuMoe,
			expertUsed: moeCfg.expertUsed,
			kvTypeK: kvK,
			kvTypeV: kvV,
			sleepIdleSecs: sleepIdleSecsFor(readSettings().autoUnloadMins),
			logNote: "starting " + basename(modelPath) + " (ctx " + ctx + ", mtp " + mtp + (noCap ? ", nocap" : "") + ", effort " + effortLevel + (preserve ? ", preserve" : ", no-preserve") + ", kv " + kvK + "/" + kvV + (moeMeta?.isMoe ? ", experts " + (moeCfg.cpuMoe ? "cpu" : moeCfg.nCpuMoe > 0 ? "cpu-first-" + moeCfg.nCpuMoe : "gpu") + (moeCfg.expertUsed != null ? " top-" + moeCfg.expertUsed : "") : "") + ", mmproj " + (mmproj ? basename(mmproj) + (mmprojCpu ? " (cpu)" : " (gpu)") : "none") + ")",
		});
	}

	async function startRouter(presetsPath) {
		if (!existsSync(BIN)) return { ok: false, error: "llama-server binary missing: " + BIN };
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
		return launch(args, { mode: "router", presetsPath, modelLabel: "router (" + basename(presetsPath) + ")", sleepIdleSecs: sleepIdleSecsFor(readSettings().autoUnloadMins), logNote: "starting router (" + basename(presetsPath) + ")" });
	}

	async function launch(args, info) {
		appendFileSync(LOG_PATH, "\n=== " + new Date().toISOString() + " " + info.logNote + " ===\n");

		const next = spawn(BIN, args, {
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
		state.model = info.modelLabel ?? null;
		if (info.modelPath !== undefined) state.modelPath = info.modelPath;
		if (info.alias !== undefined) state.alias = info.alias;
		if (info.ctx !== undefined) state.ctx = info.ctx;
		if (info.mtpHeads !== undefined) state.mtpHeads = info.mtpHeads;
		if (info.reasoningEffort !== undefined) state.reasoningEffort = info.reasoningEffort;
		if (info.preserveThinking !== undefined) state.preserveThinking = info.preserveThinking;
		if (info.ignoreCtxCap !== undefined) state.ignoreCtxCap = info.ignoreCtxCap;
		if (info.kvTypeK !== undefined) state.kvTypeK = info.kvTypeK;
		if (info.kvTypeV !== undefined) state.kvTypeV = info.kvTypeV;
		if (info.mmproj !== undefined) state.mmprojPath = info.mmproj;
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
		s.shortcuts = DIR_SHORTCUTS;
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

/** Roots the file endpoints (browse/gguf-meta/run) may touch: the user's
 * home plus the browser shortcut dirs (env-extensible via
 * LOCAL_MODELS_SHORTCUTS). Resolves symlinks-free (lexically) and requires
 * containment so absolute-path inputs can't wander the filesystem. */
function allowedRoots() {
	return [homedir(), ...DIR_SHORTCUTS.map((s) => s.path)];
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
			// Advertise the full 131K window as max output: the qwen3.x series
			// is a heavy thinker and at xhigh its thinking block alone blows
			// past the old 32K ceiling, so long replies get truncated. Still
			// never more than the window itself — a small-ctx profile must not
			// offer more output than it has room for.
			maxTokens: Math.min(MAX_OUTPUT_TOKENS, contextWindow),
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
	return { autostartRouter: raw?.autostartRouter === true, autoUnloadMins: mins };
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
	const ctx = Number.isInteger(raw?.ctx) && raw.ctx > 0 ? raw.ctx : 8192;
	const mtpHeads = Number.isInteger(raw?.mtpHeads) && raw.mtpHeads >= 0 ? Math.min(raw.mtpHeads, 3) : 0;
	const mmprojPath = typeof raw?.mmprojPath === "string" && raw.mmprojPath ? raw.mmprojPath : null;
	if (mmprojPath && !existsSync(mmprojPath)) throw new Error("mmprojPath does not exist: " + mmprojPath);
	const effort = normalizeEffort(raw?.effort);
	const ignoreCtxCap = raw?.ignoreCtxCap === true;
	const preserveThinking = normalizePreserveThinking(raw);
	// KV cache quantization pair (--cache-type-k / --cache-type-v). Legacy
	// profiles without either field keep the tuned default pair.
	const kvTypeK = normalizeKvCacheType(raw?.kvTypeK, KV_CACHE_DEFAULT_K);
	const kvTypeV = normalizeKvCacheType(raw?.kvTypeV, KV_CACHE_DEFAULT_V);
	// MoE launch config (arch is informational: which <arch>.expert_used_count
	// key a top-k override targets; re-validated from the GGUF at launch).
	const moe = normalizeMoEConfig(raw);
	return { modelPath, ctx, mtpHeads, mmprojPath, effort, preserveThinking, ignoreCtxCap, kvTypeK, kvTypeV, ...moe };
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
		}
		// MTP draft spec, same gating rule as single mode: fixed draft-mtp up
		// to the ctx ceiling, dropped above it unless the profile ignores it.
		const noCap = p.ignoreCtxCap === true;
		const mtp = Number.isInteger(p.mtpHeads) && p.mtpHeads > 0 ? Math.min(p.mtpHeads, MTP_MAX) : 0;
		const ctxSize = Number.isInteger(p.ctx) && p.ctx > 0 ? p.ctx : 8192;
		if (mtp > 0 && (ctxSize <= VRAM_CTX_LIMIT || noCap)) {
			lines.push("spec-type = draft-mtp");
			lines.push("spec-draft-n-max = " + mtp);
			lines.push("spec-draft-p-min = " + MTP_P_MIN);
		}
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
			maxTokens: Math.min(MAX_OUTPUT_TOKENS, p.ctx ?? 8192),
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
					sendJson(res, 400, { error: "path is outside the allowed roots (home + LOCAL_MODELS_SHORTCUTS): " + target });
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
					sendJson(res, 400, { error: "path is outside the allowed roots (home + LOCAL_MODELS_SHORTCUTS)" });
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
					sendJson(res, 400, { error: "model path is outside the allowed roots (home + LOCAL_MODELS_SHORTCUTS)" });
					return;
				}
				const result = await manager.run(path, Number.isFinite(ctx) ? ctx : 8192, Number.isFinite(mtp) ? mtp : 0, mmproj, effort, body.mmprojCpu === false ? false : MMPROJ_CPU, moe, body.ignoreCtxCap === true, normalizePreserveThinking(body), kvTypeK, kvTypeV);
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
								effort: body.effort,
								preserveThinking: normalizePreserveThinking(body),
								ignoreCtxCap: body.ignoreCtxCap === true,
								kvTypeK: body.kvTypeK,
								kvTypeV: body.kvTypeV,
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
					sendJson(res, 200, { ok: true, settings: writeSettings({ ...readSettings(), ...patch }) });
					return;
				}
				sendJson(res, 200, { settings: readSettings() });
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
