#!/usr/bin/env node
// dsh-local-models-ops: profiles.json audit.
// Checks: file exists/parses, model/mmproj paths exist and are not on the
// legacy /mnt/disco1, MTP depth is within the upstream fixed-MTP range
// (0-3), and reports ctx/MTP per profile. Exit 0 = clean, 1 = issues found.
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const PROFILES = process.env.DSH_LOCAL_MODELS_PROFILES || join(homedir(), ".dsh/local-models/profiles.json");
const LEGACY = "/mnt/disco1";
const MTP_MAX = 3; // upstream fixed draft: deeper collapses at large ctx
let issues = 0;

let profiles;
try {
  profiles = JSON.parse(readFileSync(PROFILES, "utf8"));
} catch (e) {
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
  const metrics = `ctx=${p.ctx} mtp=${p.mtpHeads} effort=${p.effort ?? "?"}${p.ignoreCtxCap === true ? " nocap" : ""}`;
  console.log(`- ${id}: model=${ok(p.modelPath)} mmproj=${ok(p.mmprojPath)} | ${metrics}${legacy ? " | !!legacy disco1" : ""}`);
  if (legacy) issues++;
  if (p.modelPath && !existsSync(p.modelPath)) issues++;
  if (p.mmprojPath && !existsSync(p.mmprojPath)) issues++;
  if (Number.isInteger(p.mtpHeads) && p.mtpHeads > MTP_MAX) {
    console.log(`  [warn] mtpHeads=${p.mtpHeads} exceeds the upstream fixed-MTP cap (${MTP_MAX}) - left over from the fork era`);
    issues++;
  }
  if (p.kvStreamMib !== undefined) {
    console.log(`  [warn] stale fork-era kvStreamMib=${p.kvStreamMib} field (ignored by the plugin)`);
    issues++;
  }
}

console.log(issues === 0 ? "[ok] profiles look healthy" : `[FAIL] ${issues} issue(s) found`);
process.exit(issues === 0 ? 0 : 1);
