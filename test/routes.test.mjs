// Route-level tests: the HTTP surface the tab talks to, mounted on a fake ctx
// (no dsh, no server, no real llama-server). These cover the confinement and
// validation rules that the pure-helper tests cannot reach: what /settings
// accepts as a binary or as a browsable folder, what /runtime/check is allowed
// to call green, and where the file endpoints may look.
import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";

// Isolate the plugin state and make the sandbox a browsable root before the
// module resolves its data dir / env at import time.
const TMP = mkdtempSync(join(tmpdir(), "dsh-local-models-routes-"));
const DSH = join(TMP, ".dsh");
const MODELS = join(TMP, "models");
const BIN_DIR = join(TMP, "llama.cpp", "build", "bin");
mkdirSync(join(DSH, "local-models"), { recursive: true });
mkdirSync(MODELS, { recursive: true });
mkdirSync(BIN_DIR, { recursive: true });
const MODEL = join(MODELS, "model.gguf");
writeFileSync(MODEL, "gguf-stub");
const FAKE_BIN = join(BIN_DIR, "llama-server");
writeFileSync(FAKE_BIN, "#!/bin/sh\necho 'version: fake 1'\n", "utf8");
chmodSync(FAKE_BIN, 0o755);
// Same name, no +x: the name check passes, the executable check must not.
const RO_DIR = join(TMP, "ro-bin");
mkdirSync(RO_DIR, { recursive: true });
const NOT_EXEC = join(RO_DIR, "llama-server");
writeFileSync(NOT_EXEC, "#!/bin/sh\n", "utf8");
chmodSync(NOT_EXEC, 0o644);

process.env.DSH_HOME = DSH;
process.env.LOCAL_MODELS_SHORTCUTS = MODELS;
process.env.LOCAL_MODELS_PORT = "18089";
delete process.env.LOCAL_MODELS_BIN;

const lib = await import("../lib/index.js");

/** Minimal fake ctx: captures the routes apply() registers and runs the
 * effects immediately, which is all the plugin needs to be driven. */
function mount() {
	const routes = new Map();
	lib.apply({
		settings: { update: async () => {} },
		credentials: { set: async () => {} },
		webServer: { register: (r) => { routes.set(r.path, r.handler); return () => {}; } },
		effect: (fn) => { fn(); },
	});
	const request = async (path, { method = "GET", body = null, query = "" } = {}) => {
		const handler = routes.get(path);
		assert.ok(handler, "route not registered: " + path);
		const res = {
			code: null,
			body: null,
			writeHead(code) { this.code = code; },
			end(payload) { this.body = JSON.parse(payload); },
		};
		const req = {
			method,
			url: path + query,
			on(ev, cb) {
				if (ev === "data" && body !== null) cb(Buffer.from(JSON.stringify(body)));
				if (ev === "end") cb();
				return req;
			},
		};
		await handler(req, res);
		return res;
	};
	return { request };
}

describe("routes: /settings validation", () => {
	let api;
	before(() => { api = mount(); rmSync(lib.settingsFile(), { force: true }); });
	after(() => rmSync(lib.settingsFile(), { force: true }));

	it("refuses a binPath that is not a llama-server (the file endpoints must not widen)", async () => {
		for (const p of ["/etc/hosts", "/bin/echo", "/usr/bin/env"]) {
			const res = await api.request("/local-models/settings", { method: "POST", body: { binPath: p } });
			assert.equal(res.code, 400, p);
			assert.match(res.body.error, /llama-server/, p);
		}
		// Nothing was stored, so the browse guard did not move either.
		assert.equal(lib.readSettings().binPath, "");
	});

	it("refuses a llama-server that is not executable", async () => {
		const res = await api.request("/local-models/settings", { method: "POST", body: { binPath: NOT_EXEC } });
		assert.equal(res.code, 400);
		assert.match(res.body.error, /not executable/);
	});

	it("accepts the directory holding llama-server and the binary itself", async () => {
		for (const p of [BIN_DIR, FAKE_BIN, "~"]) {
			if (p === "~") continue; // no llama-server in the test home
			const res = await api.request("/local-models/settings", { method: "POST", body: { binPath: p } });
			assert.equal(res.code, 200, p);
		}
		assert.equal(lib.readSettings().binPath, FAKE_BIN);
	});

	it("refuses shortcuts that would widen the roots to the whole filesystem", async () => {
		for (const list of [["/"], ["/etc/../.."], ["a/relative"], [42], ["/ok", "/also-ok", "/3", "/4", "/5", "/6", "/7", "/8", "/9", "/10", "/11", "/12", "/13"]]) {
			const res = await api.request("/local-models/settings", { method: "POST", body: { shortcuts: list } });
			assert.equal(res.code, 400, JSON.stringify(list));
		}
	});

	it("accepts an absolute folder list and stores it verbatim", async () => {
		const res = await api.request("/local-models/settings", { method: "POST", body: { shortcuts: [MODELS, "/mnt/models"] } });
		assert.equal(res.code, 200);
		assert.deepEqual(res.body.settings.shortcuts, [MODELS, "/mnt/models"]);
	});

	it("bounds the VRAM override", async () => {
		for (const bad of ["20", -1, 4096, null]) {
			const res = await api.request("/local-models/settings", { method: "POST", body: { vramGb: bad } });
			assert.equal(res.code, 400, String(bad));
		}
		const ok = await api.request("/local-models/settings", { method: "POST", body: { vramGb: 20 } });
		assert.equal(ok.code, 200);
		assert.equal(ok.body.settings.vramGb, 20);
	});
});

describe("routes: /runtime/check verdict", () => {
	let api;
	before(() => { api = mount(); });
	after(() => rmSync(lib.settingsFile(), { force: true }));

	it("is not green for a file that merely prints a banner", async () => {
		const res = await api.request("/local-models/runtime/check", { method: "POST", body: { binPath: "/usr/bin/env" } });
		assert.equal(res.code, 200);
		assert.equal(res.body.ok, false);
		assert.match(res.body.error, /not a llama-server/);
	});

	it("is green for a llama-server that starts, and reports its banner", async () => {
		const res = await api.request("/local-models/runtime/check", { method: "POST", body: { binPath: FAKE_BIN } });
		assert.equal(res.code, 200);
		assert.equal(res.body.ok, true, JSON.stringify(res.body));
		assert.equal(res.body.version.version, "version: fake 1");
		assert.equal(res.body.bin.resolved, FAKE_BIN);
	});

	it("reports a miss without falling back to another layer", async () => {
		const res = await api.request("/local-models/runtime/check", { method: "POST", body: { binPath: join(TMP, "nope") } });
		assert.equal(res.body.ok, false);
		assert.deepEqual(res.body.bin.candidates.map((c) => c.path), [join(TMP, "nope")]);
	});
});

describe("routes: file confinement", () => {
	let api;
	before(() => {
		api = mount();
		lib.writeSettings({ binPath: BIN_DIR, shortcuts: [MODELS] });
	});
	after(() => {
		rmSync(lib.settingsFile(), { force: true });
	});

	it("lists the home and the declared folders", async () => {
		assert.equal((await api.request("/local-models/browse", { query: "?dir=" + encodeURIComponent(MODELS) })).code, 200);
		assert.equal((await api.request("/local-models/browse")).code, 200);
	});

	it("refuses /etc and the filesystem root", async () => {
		for (const dir of ["/etc", "/"]) {
			const res = await api.request("/local-models/browse", { query: "?dir=" + encodeURIComponent(dir) });
			assert.equal(res.code, 400, dir);
		}
	});

	it("keeps the model of a saved profile loadable after its folder is removed", async () => {
		// A profile declares the folder; removing the shortcut must not make the
		// profile unloadable.
		assert.equal((await api.request("/local-models/profiles", { method: "POST", body: { name: "p", modelPath: MODEL, ctx: 8192 } })).code, 200);
		lib.writeSettings({ ...lib.readSettings(), shortcuts: [] });
		assert.equal((await api.request("/local-models/browse", { query: "?dir=" + encodeURIComponent(MODELS) })).code, 200);
		assert.equal((await api.request("/local-models/browse", { query: "?dir=" + encodeURIComponent("/etc") })).code, 400);
	});

	it("refuses a profile pointing outside the declared folders", async () => {
		lib.writeSettings({ ...lib.readSettings(), shortcuts: [] });
		const res = await api.request("/local-models/profiles", { method: "POST", body: { name: "evil", modelPath: "/etc/hosts", ctx: 8192 } });
		assert.equal(res.code, 400);
		assert.match(res.body.error, /outside the folders/);
		// The refused save must not have widened anything.
		assert.equal((await api.request("/local-models/browse", { query: "?dir=" + encodeURIComponent("/etc") })).code, 400);
	});

	it("refuses a run whose mmproj lives outside the declared folders", async () => {
		lib.writeSettings({ ...lib.readSettings(), shortcuts: [MODELS] });
		const res = await api.request("/local-models/run", { method: "POST", body: { path: MODEL, mmproj: "/etc/hosts", ctx: 8192 } });
		assert.equal(res.code, 400);
		assert.match(res.body.error, /mmproj path is outside/);
	});
});

describe("routes: /status shape", () => {
	it("reports the binary, the folders and the GPU budget", async () => {
		lib.writeSettings({ binPath: BIN_DIR, shortcuts: [MODELS], vramGb: 0 });
		const { request } = mount();
		const res = await request("/local-models/status");
		assert.equal(res.code, 200);
		assert.equal(res.body.binConfigured, BIN_DIR);
		assert.equal(res.body.bin.resolved, FAKE_BIN);
		assert.equal(res.body.bin.source, "setting");
		assert.deepEqual(res.body.shortcuts.map((s) => s.path), [MODELS]);
		assert.equal(res.body.shortcuts[0].exists, true);
		assert.equal(res.body.shortcutsSource, "settings");
		assert.ok(res.body.vramTotalBytes > 0, "a VRAM budget is always reported");
		assert.ok(Array.isArray(res.body.gpus));
		assert.equal(res.body.gpuCount, res.body.gpus.length);
		rmSync(lib.settingsFile(), { force: true });
	});
});

describe("probeBin cache", () => {
	// A counting runner proves what the cache does without executing anything.
	const banner = { err: null, stdout: "version: injected\n", stderr: "" };
	it("caches a success and re-probes after chmod changes the mode", async () => {
		const bin = join(TMP, "cache-probe-llama-server");
		writeFileSync(bin, "#!/bin/sh\n", "utf8");
		chmodSync(bin, 0o644);
		let calls = 0;
		const runner = async () => { calls++; return banner; };
		assert.equal((await lib.probeBin(bin, runner)).ok, true);
		assert.equal(calls, 1);
		await lib.probeBin(bin, runner);
		assert.equal(calls, 1, "second probe must come from the cache");
		chmodSync(bin, 0o755);
		await lib.probeBin(bin, runner);
		assert.equal(calls, 2, "chmod must invalidate the cache");
	});
	it("never caches a failure (a fixed binary must not stay broken)", async () => {
		const bin = join(TMP, "failing-llama-server");
		writeFileSync(bin, "#!/bin/sh\n", "utf8");
		let calls = 0;
		const runner = async () => { calls++; return { err: null, stdout: "", stderr: "" }; };
		assert.equal((await lib.probeBin(bin, runner)).ok, false);
		assert.equal((await lib.probeBin(bin, runner)).ok, false);
		assert.equal(calls, 2);
	});
	it("cachedBinVersion never spawns anything", () => {
		assert.equal(lib.cachedBinVersion(join(TMP, "never-probed-llama-server")), null);
	});
});
