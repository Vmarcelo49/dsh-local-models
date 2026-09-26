// Unit tests for the pure, exported helpers in lib/index.js.
// Run: npm test (node's built-in runner). No server, no models needed.
import { describe, it, before } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// Isolate the profiles store: lib/index.js resolves its data dir from
// DSH_HOME at import time, so point it at a temp dir before importing.
const TMP = mkdtempSync(join(tmpdir(), "dsh-local-models-test-"));
process.env.DSH_HOME = join(TMP, ".dsh");
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
	it("drops the draft above the ctx ceiling unless ignored", () => {
		const big = lib.generateRouterPresets([profile({ ctx: 262144 })]);
		assert.ok(!big.includes("spec-type = draft-mtp"));
		const nocap = lib.generateRouterPresets([profile({ ctx: 262144, ignoreCtxCap: true })]);
		assert.ok(nocap.includes("spec-type = draft-mtp"));
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
	it("defaults ctx to the single-mode 8192 and caps maxTokens", () => {
		const { profile } = lib.buildProviderProfile({ alias: "m", port: 8080 });
		assert.equal(profile.models[0].contextWindow, 8192);
		assert.equal(profile.models[0].maxTokens, 8192);
		// Full-window (daily 131K) launches advertise the whole window as
		// max output: heavy-thinking qwen3.x models at xhigh blow past the
		// old 32K ceiling inside the thinking block.
		const big = lib.buildProviderProfile({ alias: "m", port: 8080, ctx: 131072 });
		assert.equal(big.profile.models[0].maxTokens, 131072);
	});
	it("router models follow the same rule", () => {
		const { profile } = lib.buildRouterProfile([{ id: "m", ctx: 8192 }], {});
		assert.equal(profile.models[0].contextWindow, 8192);
		assert.equal(profile.models[0].maxTokens, 8192);
		const big = lib.buildRouterProfile([{ id: "m", ctx: 131072 }], {});
		assert.equal(big.profile.models[0].maxTokens, 131072);
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
	it("normalizes to known shape", () => {
		assert.deepEqual(lib.normalizeSettings(null), { autostartRouter: false, autoUnloadMins: 30 });
		assert.deepEqual(lib.normalizeSettings({}), { autostartRouter: false, autoUnloadMins: 30 });
		assert.deepEqual(lib.normalizeSettings({ autostartRouter: true }), { autostartRouter: true, autoUnloadMins: 30 });
		assert.deepEqual(lib.normalizeSettings({ autostartRouter: "yes" }), { autostartRouter: false, autoUnloadMins: 30 });
	});
	it("keeps an explicit timer, including off (0)", () => {
		assert.deepEqual(lib.normalizeSettings({ autoUnloadMins: 0 }), { autostartRouter: false, autoUnloadMins: 0 });
		assert.deepEqual(lib.normalizeSettings({ autoUnloadMins: 15 }), { autostartRouter: false, autoUnloadMins: 15 });
	});
	it("falls back to the 30 min default on invalid timers", () => {
		for (const bad of [-5, 2.5, "30", null]) {
			assert.deepEqual(lib.normalizeSettings({ autoUnloadMins: bad }), { autostartRouter: false, autoUnloadMins: 30 });
		}
	});
	it("defaults when no settings file exists", () => {
		rmSync(lib.settingsFile(), { force: true });
		assert.deepEqual(lib.readSettings(), { autostartRouter: false, autoUnloadMins: 30 });
	});
	it("round-trips both settings", () => {
		assert.deepEqual(
			lib.writeSettings({ autostartRouter: true, autoUnloadMins: 15 }),
			{ autostartRouter: true, autoUnloadMins: 15 });
		assert.deepEqual(lib.readSettings(), { autostartRouter: true, autoUnloadMins: 15 });
		assert.deepEqual(
			lib.writeSettings({ autostartRouter: false, autoUnloadMins: 0 }),
			{ autostartRouter: false, autoUnloadMins: 0 });
		assert.deepEqual(lib.readSettings(), { autostartRouter: false, autoUnloadMins: 0 });
		rmSync(lib.settingsFile(), { force: true });
	});
	it("falls back to defaults on a corrupt file", () => {
		writeFileSync(lib.settingsFile(), "{not json", "utf8");
		assert.deepEqual(lib.readSettings(), { autostartRouter: false, autoUnloadMins: 30 });
		rmSync(lib.settingsFile(), { force: true });
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
