#!/usr/bin/env node
// dsh-local-models-ops: profiles.json audit.
// Checks: file exists/parses, model/mmproj paths exist and are not on the
// legacy /mnt/disco1, MTP depth is within the upstream fixed-MTP range
// (0-3), KV cache ids are known, and the K/V pair matches on MLA-style
// profiles where it matters. Exit 0 = clean (warnings allowed),
// exit 1 = hard errors found.
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const PROFILES = process.env.DSH_LOCAL_MODELS_PROFILES || join(homedir(), ".dsh/local-models/profiles.json");
const LEGACY = "/mnt/disco1";
const MTP_MAX = 3; // upstream fixed draft: deeper collapses at large ctx
// Accepted --cache-type-k/v ids (mirrors KV_CACHE_TYPES in lib/index.js);
// unknown ids are normalized to the q5_0/q4_1 default on save/launch.
const KV_TYPES = ["f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "iq4_nl", "q4_0"];
let errors = 0;
let warnings = 0;

const fail = (msg) => { errors++; console.log(`  [FAIL] ${msg}`); };
const warn = (msg) => { warnings++; console.log(`  [warn] ${msg}`); };

let profiles;
try {
  profiles = JSON.parse(readFileSync(PROFILES, "utf8"));
} catch (e) {
  if (e?.code === "ENOENT") {
    console.log("[ok] no profiles saved (store does not exist yet)");
    process.exit(0);
  }
  console.error(`[FAIL] cannot read/parse ${PROFILES}: ${e.message}`);
  process.exit(1);
}
if (!Array.isArray(profiles) || profiles.length === 0) {
  console.log("[ok] no profiles saved (or empty list)");
  process.exit(0);
}

for (const p of profiles) {
  const id = p.id || p.name || "?";
  const ok = (f) => (f ? (existsSync(f) ? "ok" : "MISSING") : "none");
  const legacy = [p.modelPath, p.mmprojPath].some((f) => f && f.includes(LEGACY));
  const preserve = p.preserveThinking ?? p.preserve_thinking ?? false;
  const metrics = `ctx=${p.ctx} mtp=${p.mtpHeads} effort=${p.effort ?? "?"} preserve=${preserve === true ? "on" : "off"}${p.ignoreCtxCap === true ? " nocap" : ""}`;
  console.log(`- ${id}: model=${ok(p.modelPath)} mmproj=${ok(p.mmprojPath)} | ${metrics}${legacy ? " | !!legacy disco1" : ""}`);
  if (legacy) warn(`legacy ${LEGACY} path (disk is failing; move to /mnt/raid0/GGUF/...)`);
  if (!p.modelPath) fail("missing modelPath");
  else if (!existsSync(p.modelPath)) fail(`model file missing: ${p.modelPath}`);
  if (p.mmprojPath && !existsSync(p.mmprojPath)) fail(`mmproj file missing: ${p.mmprojPath}`);
  if (Number.isInteger(p.mtpHeads) && p.mtpHeads > MTP_MAX) {
    warn(`mtpHeads=${p.mtpHeads} exceeds the upstream fixed-MTP cap (${MTP_MAX}) - left over from the fork era`);
  }
  if (p.kvStreamMib !== undefined) {
    warn(`stale fork-era kvStreamMib=${p.kvStreamMib} field (ignored by the plugin)`);
  }
  for (const [field, value] of [["kvTypeK", p.kvTypeK], ["kvTypeV", p.kvTypeV]]) {
    if (value !== undefined && value !== null && !KV_TYPES.includes(value)) {
      warn(`${field}=${value} is not a known cache type (normalized to the default on save/launch)`);
    }
  }
  // The plugin refuses mixed K/V only on MLA (latent-KV) models, which the
  // audit cannot detect from profiles.json alone — so a mismatch is a heads-up,
  // not a failure. Verify against the GGUF (isMla / deepseek4 arch) if unsure.
  if (p.kvTypeK && p.kvTypeV && p.kvTypeK !== p.kvTypeV) {
    console.log(`  [info] mixed KV pair ${p.kvTypeK}/${p.kvTypeV} (refused at launch only on MLA models)`);
  }
}

if (errors > 0) {
  console.log(`[FAIL] ${errors} error(s), ${warnings} warning(s)`);
  process.exit(1);
}
console.log(warnings === 0 ? "[ok] profiles look healthy" : `[ok] profiles usable with ${warnings} warning(s)`);
process.exit(0);
