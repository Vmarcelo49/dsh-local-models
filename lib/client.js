/**
 * dsh-local-models - browser half.
 *
 * Local Models settings tab: pick a .gguf via a file browser, choose context
 * (8K-step slider + fine input), the KV cache types for K and V, and fixed MTP
 * heads (0-7), see a live VRAM estimate against the detected GPU total, then
 * Load / Stop /
 * Register the server. The VRAM math is ported from the user's
 * vram-calculator project.
 */
window.__ModuleLoader__.load({
	id: "dsh-local-models",
	factory: (require) => {
		var module = { exports: {} };
		var exports = module.exports;
		Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });

		var React = require("react");
		var h = React.createElement;
		var useState = React.useState, useEffect = React.useEffect, useCallback = React.useCallback, useMemo = React.useMemo;

		// ---- VRAM math (ported from vram-calculator/src/vram.ts) ----
		// KV cache quantization: the exact set upstream llama-server accepts
		// (common/arg.cpp `kv_cache_types`), highest precision first, with the
		// ggml bytes-per-element each one costs (block header included). The
		// ids match lib/index.js; the defaults are the tuned 16 GiB pair.
		var KV_TYPES = [
			{ id: "f32", bpe: 4 },
			{ id: "f16", bpe: 2 },
			{ id: "bf16", bpe: 2 },
			{ id: "q8_0", bpe: 34 / 32 },
			{ id: "q5_1", bpe: 24 / 32 },
			{ id: "q5_0", bpe: 22 / 32 },
			{ id: "q4_1", bpe: 20 / 32 },
			{ id: "iq4_nl", bpe: 18 / 32 },
			{ id: "q4_0", bpe: 18 / 32 },
		];
		var KV_BPE = {};
		for (var kvi = 0; kvi < KV_TYPES.length; kvi++) KV_BPE[KV_TYPES[kvi].id] = KV_TYPES[kvi].bpe;
		var KV_DEFAULT_K = "q5_0", KV_DEFAULT_V = "q4_1";
		function normalizeKvType(id, fallback) { return KV_BPE[id] != null ? id : fallback; }
		// Thinking levels the loaded template accepts (mirrors THINKING_LEVELS
		// in lib/index.js; the server's normalizeEffort is the authority).
		var EFFORT_LEVELS = ["off", "low", "medium", "xhigh"];
		function normalizeEffort(effort) { return EFFORT_LEVELS.indexOf(effort) >= 0 ? effort : "medium"; }
		function kvLabel(id) { return String(id).toUpperCase(); }
		function kvOptionLabel(t) { return t.id + " (" + t.bpe.toFixed(2) + " B/elt)"; }
		var GRAPH_BASE = 24 * 1024 * 1024, SCRATCH_PER_TOKEN = 3072;
		var HYBRID_GRAPH_BASE = 172 * 1024 * 1024, HYBRID_SCRATCH_PER_TOKEN = 1025;
		var OVERHEAD_BYTES = 200 * 1024 * 1024;
		// Fallback budget for the estimate: only used before the first status
		// poll, or on hardware the node half could not identify. The real value
		// comes from status.vramTotalBytes (detected, or pinned in the Runtime
		// card). Mirrors VRAM_ASSUMED_BYTES in lib/index.js.
		var TOTAL_VRAM_BYTES = 16 * 1024 * 1024 * 1024;
		var SAFE_MARGIN_BYTES = 700 * 1024 * 1024;
		var CTX_STEP = 8192, FALLBACK_MAX_CTX = 32768;
		// Fixed-MTP ceiling (mirrors MTP_MAX in lib/index.js): upstream accepts any
		// --spec-draft-n-max and clamps the effective depth to the model's nextn
		// depth at load. 0 = off.
		var MTP_MAX = 7;
		var MTP_OPTIONS = [];
		for (var mtpI = 0; mtpI <= MTP_MAX; mtpI++) MTP_OPTIONS.push(mtpI);
		// Multi-GPU placement (mirrors SPLIT_MODES in lib/index.js). "layer" is
		// llama.cpp's own default, so it emits no flag at all.
		var SPLIT_MODES = [
			{ id: "layer", label: "layer — pipelined (default)" },
			{ id: "row", label: "row — tensor parallel" },
			{ id: "tensor", label: "tensor — experimental" },
			{ id: "none", label: "none — first GPU only" },
		];
		var SPLIT_DEFAULT = "layer";
		function normalizeSplitMode(id) {
			for (var si = 0; si < SPLIT_MODES.length; si++) if (SPLIT_MODES[si].id === id) return id;
			return SPLIT_DEFAULT;
		}
		// Even proportions across the detected GPUs, weighted by VRAM (mirrors
		// evenTensorSplit() in lib/index.js).
		function evenSplit(gpus) {
			var list = (gpus || []).filter(function (g) { return g && g.vramBytes > 0; });
			if (list.length < 2) return "";
			var min = Math.min.apply(null, list.map(function (g) { return g.vramBytes; }));
			return list.map(function (g) { return Math.max(1, Math.round(g.vramBytes / min)); }).join(",");
		}
		// Mirrors normalizeTensorSplit() in lib/index.js: proportions or "".
		function normSplitText(v) {
			if (typeof v !== "string") return "";
			var parts = v.trim().split(/[,/\s]+/).filter(Boolean);
			if (parts.length === 0) return "";
			for (var pi = 0; pi < parts.length; pi++) {
				var n = Number(parts[pi]);
				if (!isFinite(n) || n <= 0) return "";
			}
			return parts.join(",");
		}
		function fmtVramGb(bytes) {
			var gb = bytes / (1024 * 1024 * 1024);
			var r = Math.round(gb * 10) / 10;
			return String(r);
		}

		var DEFAULT_FULL_ATTN_SHARE = 1 / 6;

		function gdnLayout(nLayers, fullAttnInterval, nextnPredictLayers) {
			var nextn = nextnPredictLayers ?? 0;
			var trunk = Math.max((nLayers ?? 0) - nextn, 0);
			var interval = fullAttnInterval ?? 0;
			var hasSsmSig = interval > 1;
			var attnTrunk = hasSsmSig ? Math.floor(trunk / interval) : trunk;
			return {
				hybrid: hasSsmSig && nextn >= 0 && (nLayers ?? 0) > 0 && attnTrunk < trunk,
				interval: interval || 1,
				trunk: trunk,
				attnTrunk: attnTrunk,
				mtp: nextn,
			};
		}

		function gdnRecurrentBytes(recrLayerCount, ssmStateSize, ssmInnerSize, ssmNGroup, ssmDtRank, ssmConvKernel, ssmF16) {
			if (!recrLayerCount || !ssmStateSize || !ssmInnerSize) return 0;
			var ssmBytes = ssmStateSize * ssmInnerSize * (ssmF16 ? 2 : 4);
			var convDim = 2 * ssmStateSize * (ssmNGroup || 1) + ssmStateSize * (ssmDtRank || 1);
			var convBytes = convDim * (ssmConvKernel || 4) * 4;
			return recrLayerCount * (ssmBytes + convBytes);
		}

		function kvBytesFor(context, nLayers, nKvHeads, headDim, kvTypeK, kvTypeV, slidingWindow, fullAttnShare, kvLayersOverride) {
			if (!nKvHeads || !headDim) return 0;
			var bpeK = KV_BPE[kvTypeK] ?? 2;
			var bpeV = KV_BPE[kvTypeV] ?? 2;
			var perTokenPerLayer = nKvHeads * headDim * (bpeK + bpeV);
			if (kvLayersOverride != null && kvLayersOverride > 0) {
				return Math.floor(kvLayersOverride * context * perTokenPerLayer);
			}
			if (!nLayers) return 0;
			var hasSwa = !!slidingWindow && slidingWindow > 0 && context > slidingWindow;
			if (!hasSwa) {
				return Math.floor(nLayers * context * perTokenPerLayer);
			}
			var share = fullAttnShare == null || fullAttnShare < 0 || fullAttnShare > 1 ? DEFAULT_FULL_ATTN_SHARE : fullAttnShare;
			var fullLayers = nLayers * share;
			var localLayers = nLayers - fullLayers;
			var effLocal = Math.min(context, slidingWindow);
			return Math.floor((fullLayers * context + localLayers * effLocal) * perTokenPerLayer);
		}

		function estimateComputeBytes(context, nVocab, hybrid) {
			var vocab = Math.min(Math.max(nVocab ?? 32000, 1024), 262144);
			var outputs = Math.min(512, context);
			var logits = outputs * vocab * 4;
			var base = hybrid ? HYBRID_GRAPH_BASE : GRAPH_BASE;
			var perToken = hybrid ? HYBRID_SCRATCH_PER_TOKEN : SCRATCH_PER_TOKEN;
			var scratch = perToken * Math.max(context - 1, 0) + base;
			return Math.floor(logits + scratch);
		}

		function maxContextFor(targetBytes, input) {
			if (!input.nKvHeads || !input.headDim) return null;
			var hybrid = input.kvLayersOverride != null && input.kvLayersOverride > 0;
			var fixedExtra = hybrid ? (input.recurrentFixedBytes ?? 0) : 0;
			var base = hybrid ? HYBRID_GRAPH_BASE : GRAPH_BASE;
			var perToken = hybrid ? HYBRID_SCRATCH_PER_TOKEN : SCRATCH_PER_TOKEN;
			var vocab = Math.min(Math.max(input.nVocab ?? 32000, 1024), 262144);
			var logitsConst = 512 * vocab * 4;
			var computeBase = base + logitsConst;
			var avail = targetBytes - input.weightsBytes - input.overheadBytes - computeBase - fixedExtra;
			if (avail <= 0) return 0;
			var fitsCtx = function (ctx) {
				var kv = kvBytesFor(ctx, input.nLayers, input.nKvHeads, input.headDim, input.kvTypeK, input.kvTypeV, input.slidingWindow, input.fullAttnShare, input.kvLayersOverride);
				return kv + perToken * ctx <= avail;
			};
			if (!fitsCtx(1)) return 0;
			var lo = 1, hi = 2;
			while (hi <= 1073741824 && fitsCtx(hi)) hi = hi * 2;
			if (hi > 1073741824) return hi;
			while (hi - lo > 1) {
				var mid = (lo + hi) >> 1;
				if (fitsCtx(mid)) lo = mid; else hi = mid;
			}
			return lo;
		}

		function fmtInt(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
		function fmtCtx(n) { return n % 1024 === 0 ? fmtInt(n / 1024) + "K" : fmtInt(n); }
		function fmtGiB(b) { return (b / (1024 * 1024 * 1024)).toFixed(2) + " GiB"; }
		function fmtMiB(b) { return (b / (1024 * 1024)).toFixed(1) + " MiB"; }

		// ---- tiny fetch helpers (never throw raw JSON.parse errors) ----
		function parseJson(r) {
			return r.json().catch(function () {
				return { error: "empty or invalid response (HTTP " + r.status + ")" };
			});
		}
		function get(path) {
			return fetch(path).then(function (r) { return parseJson(r); });
		}
		function post(path, body) {
			return fetch(path, {
				method: "POST",
				headers: { "content-type": "application/json" },
				body: JSON.stringify(body),
			}).then(function (r) {
				return parseJson(r).then(function (parsed) { return { ok: r.ok, body: parsed }; });
			});
		}

		// ---- shared hooks ----
		// Shorthand for the file's dominant state pattern (one line per state).
		function useStatePair(initial) {
			var s = useState(initial);
			return [s[0], s[1]];
		}
		// Poll fn every ms (plus once on mount). The latest fn is always
		// used via a ref, so inline (non-memoized) callbacks don't tear
		// down and rebuild the interval on every render.
		function usePoll(fn, ms) {
			var fnRef = React.useRef(fn);
			fnRef.current = fn;
			useEffect(function () {
				var wrapped = function () { fnRef.current(); };
				wrapped();
				var timer = setInterval(wrapped, ms);
				return function () { clearInterval(timer); };
			}, [ms]);
		}

		// ---- styles ----
		var style = {
			page: { maxWidth: 720, display: "flex", flexDirection: "column", gap: 10, color: "var(--dsw-alias-label-primary)" },
			title: { margin: 0, fontSize: 16, fontWeight: 500, lineHeight: "24px", color: "var(--dsw-alias-label-primary)" },
			intro: { margin: 0, fontSize: 14, lineHeight: "22px", color: "var(--dsw-alias-label-tertiary)" },
			card: { border: "1px solid var(--dsw-alias-border-l2)", borderRadius: 12, padding: "10px 12px", display: "flex", flexDirection: "column", gap: 8 },
			cardTitle: { margin: 0, fontSize: 13, fontWeight: 500, color: "var(--dsw-alias-label-secondary)" },
			row: { display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" },
			button: { boxSizing: "border-box", height: 34, font: "inherit", cursor: "pointer", border: "none", borderRadius: 17, padding: "0 14px", fontSize: 14, lineHeight: "22px", background: "var(--dsw-alias-button-primary-fill)", color: "var(--dsw-alias-label-primary-foreground)" },
			secondaryButton: { boxSizing: "border-box", height: 34, font: "inherit", cursor: "pointer", border: "1px solid var(--dsw-alias-border-l2)", borderRadius: 17, padding: "0 14px", fontSize: 14, lineHeight: "22px", background: "transparent", color: "var(--dsw-alias-label-primary)" },
			disabled: { opacity: 0.4, cursor: "default" },
			small: { fontSize: 12, lineHeight: "18px", color: "var(--dsw-alias-label-tertiary)" },
			mono: { fontFamily: "var(--ds-font-family-code)", fontSize: 12, lineHeight: "18px" },
			input: { boxSizing: "border-box", border: "1px solid var(--dsw-alias-border-l2)", width: 150, height: 32, font: "inherit", background: "var(--dsw-alias-bg-layer-1)", color: "var(--dsw-alias-label-primary)", borderRadius: 8, padding: "0 10px", fontSize: 14 },
			range: { width: 280, accentColor: "var(--dsw-alias-button-primary-fill)" },
			select: { boxSizing: "border-box", border: "1px solid var(--dsw-alias-border-l2)", height: 32, font: "inherit", background: "var(--dsw-alias-bg-layer-1)", color: "var(--dsw-alias-label-primary)", borderRadius: 8, padding: "0 8px", fontSize: 14 },
			error: { margin: 0, fontSize: 12, lineHeight: "18px", color: "var(--dsw-alias-state-error-primary)" },
			warn: { margin: 0, fontSize: 12, lineHeight: "18px", color: "var(--dsw-alias-state-warn-label)" },
			good: { color: "var(--dsw-alias-state-success-primary)", fontWeight: 500 },
			bad: { color: "var(--dsw-alias-state-error-primary)", fontWeight: 500 },
			typedef: { display: "flex", justifyContent: "space-between", gap: 12 },
			overlay: { position: "fixed", inset: 0, background: "rgba(0,0,0,0.45)", display: "flex", alignItems: "flex-start", justifyContent: "center", zIndex: 1000, paddingTop: 48 },
			modal: { width: 560, maxWidth: "92vw", maxHeight: "70vh", background: "var(--dsw-alias-bg-layer-1)", border: "1px solid var(--dsw-alias-border-l2)", borderRadius: 14, display: "flex", flexDirection: "column", gap: 10, padding: 16, boxShadow: "0 12px 40px rgba(0,0,0,0.4)" },
			terminalOverlay: { position: "fixed", inset: 0, background: "rgba(0,0,0,0.45)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1100 },
			terminal: { width: 980, maxWidth: "94vw", height: "80vh", background: "var(--dsw-alias-bg-layer-1)", border: "1px solid var(--dsw-alias-border-l2)", borderRadius: 14, display: "flex", flexDirection: "column", gap: 8, padding: 14, boxShadow: "0 12px 40px rgba(0,0,0,0.4)" },
			termBody: { flex: 1, overflowY: "auto", margin: 0, background: "var(--dsw-alias-bg-layer-2, rgba(0,0,0,0.05))", border: "1px solid var(--dsw-alias-border-l3)", borderRadius: 10, padding: 10, fontFamily: "var(--ds-font-family-code)", fontSize: 12, lineHeight: "17px", whiteSpace: "pre-wrap", wordBreak: "break-word", color: "var(--dsw-alias-label-primary)" },
			list: { margin: 0, padding: 0, listStyle: "none", overflowY: "auto", flex: 1, minHeight: 160, maxHeight: 320, border: "1px solid var(--dsw-alias-border-l2)", borderRadius: 10 },
			item: { padding: "6px 10px", cursor: "pointer", fontSize: 13, lineHeight: "20px", borderBottom: "1px solid var(--dsw-alias-border-l3)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" },
			itemSel: { background: "var(--dsw-alias-selection-fill, rgba(120,140,255,0.18))" },
			itemDir: { color: "var(--dsw-alias-label-primary)", fontWeight: 500 },
			itemFile: { color: "var(--dsw-alias-label-secondary)" },
			pathBar: { fontSize: 12, lineHeight: "18px", color: "var(--dsw-alias-label-secondary)", fontFamily: "var(--ds-font-family-code)", wordBreak: "break-all" },
		};

		// ---- shared presentational pieces ----
		// One-line labeled row: label + control, hint as a caption below.
		function FieldRow(props) {
			return h(
				"div",
				{ style: { display: "flex", flexDirection: "column", gap: 2 } },
				h(
					"div",
					{ style: style.row },
					h("span", { style: style.cardTitle }, props.label),
					props.control,
				),
				props.hint ? h("span", { style: style.small }, props.hint) : null,
			);
		}
		// Key/value line for the estimate + status cards.
		function KV(props) {
			return h(
				"div",
				{ style: props.strong ? Object.assign({}, style.typedef, { borderTop: "1px solid var(--dsw-alias-border-l2)", paddingTop: 6, marginTop: 2, fontWeight: 500 }) : style.typedef },
				h("span", {}, props.label),
				h("span", { style: props.tone || style.mono }, props.children),
			);
		}
		// One-line status summary for the actions row + status card.
		function statusLine(status, portUp) {
			if (status.status === "ready" && status.mode === "router") {
				var pc = status.presetsCount != null ? status.presetsCount : status.presets;
				return "router ready — " + String((status.loadedModels || []).length) + " model(s) loaded (" + String(pc != null ? pc : "?") + " presets)";
			}
			if (status.status === "ready") {
				var moeBits = status.cpuMoe === true ? ", experts CPU" : (status.nCpuMoe > 0 ? ", experts cpu-first-" + status.nCpuMoe : "");
				if (status.expertUsed != null) moeBits += ", top-" + status.expertUsed;
				var preserveBit = status.preserveThinking === true ? ", preserve" : (status.preserveThinking === false ? ", no-preserve" : "");
				var effortBit = status.reasoningEffort != null ? ", effort " + status.reasoningEffort : "";
				var visionBit = status.mmprojPath ? ", vision " + status.mmprojPath.split("/").pop() + (status.mmprojCpu === false ? " (gpu)" : " (cpu)") : "";
				var splitBit = status.splitMode && status.splitMode !== SPLIT_DEFAULT ? ", split " + status.splitMode + (status.tensorSplit ? " " + status.tensorSplit : "") : (status.tensorSplit ? ", tensor-split " + status.tensorSplit : "");
				return "ready — " + (status.model || "model") + " (pid " + status.pid + ", ctx " + fmtCtx(status.ctx || 0) + ", mtp " + status.mtpHeads + moeBits + preserveBit + effortBit + visionBit + splitBit + ")";
			}
			if (status.status === "starting") {
				return "loading — " + (status.model || "model") + " (pid " + status.pid + ", waiting for /health)";
			}
			if (portUp) {
				return "a server is already listening on port " + status.port + " (not started by this tab — Stop will reap it)";
			}
			if (status.status === "stopped" && status.error) {
				return "stopped — " + status.error;
			}
			if (status.status === "stopped") {
				return "stopped";
			}
			return "nothing loaded";
		}

		// ---- File browser modal ----
		function FileBrowserModal(props) {
			var dirPair = useStatePair(props.startDir || "");
			var dir = dirPair[0], setDir = dirPair[1];
			var entPair = useStatePair(null);
			var entries = entPair[0], setEntries = entPair[1];
			var parPair = useStatePair(null);
			var parent = parPair[0], setParent = parPair[1];
			var loadPair = useStatePair(false);
			var loading = loadPair[0], setLoading = loadPair[1];
			var errPair = useStatePair(null);
			var err = errPair[0], setErr = errPair[1];
			var selPair = useStatePair(null);
			var selected = selPair[0], setSelected = selPair[1];
			// Remembered folders (persisted): jump buttons + toggle for the current dir.
			var LS_FOLDERS = "dsh-local-models:remembered-folders";
			var readFolders = function () {
				try {
					var a = JSON.parse(localStorage.getItem(LS_FOLDERS) || "[]");
					return Array.isArray(a) ? a.filter(function (x) { return typeof x === "string" && x; }) : [];
				} catch { return []; }
			};
			var remPair = useStatePair(readFolders);
			var remembered = remPair[0], setRemembered = remPair[1];
			var saveFolders = function (list) {
				setRemembered(list);
				try { localStorage.setItem(LS_FOLDERS, JSON.stringify(list)); } catch { /* storage unavailable */ }
			};
			var isRemembered = dir !== "" && remembered.indexOf(dir) !== -1;
			var toggleRemember = function () {
				if (!dir) return;
				saveFolders(isRemembered
					? remembered.filter(function (d) { return d !== dir; })
					: remembered.concat([dir]).slice(-8));
			};

			// dirMode: pick a directory (the llama.cpp binaries folder) instead of
			// a file — files are shown for context but never selected, and the
			// confirm button returns the directory currently listed. Listing
			// every file matters here: a build/bin has no .gguf in it, and an
			// empty-looking folder reads as a broken picker.
			var dirMode = props.dirMode === true;
			var allFiles = props.allFiles === true || dirMode;
			var load = useCallback(function (d) {
				setLoading(true);
				setErr(null);
				get("/local-models/browse" + (d ? "?dir=" + encodeURIComponent(d) : "") + (allFiles ? (d ? "&all=1" : "?all=1") : ""))
					.then(function (body) {
						if (body && Array.isArray(body.entries)) {
							setDir(body.path || d);
							setEntries(body.entries);
							setParent(body.parent != null ? body.parent : null);
							setSelected(null);
						} else {
							setErr((body && body.error) || "could not list directory");
						}
					})
					.catch(function (e2) { setErr(String(e2)); })
					.finally(function () { setLoading(false); });
			}, []);

			useEffect(function () { load(dir); }, [load]);

			return h(
				"div",
				{ style: style.overlay, onClick: props.onClose },
				h(
					"div",
					{ style: style.modal, onClick: function (ev) { ev.stopPropagation(); } },
					h("h3", { style: style.cardTitle }, dirMode ? "Choose the llama.cpp binaries folder" : allFiles ? "Choose a vision mmproj file" : "Choose a GGUF model"),
					h("div", { style: style.pathBar }, dir || "~ (home)"),
					h(
						"div",
						{ style: style.row },
						h("button", { style: style.secondaryButton, onClick: function () { load(""); } }, "Home"),
						(props.shortcuts || []).map(function (sc, i) {
							return h("button", { key: i, style: style.secondaryButton, onClick: function () { load(sc.path); }, title: sc.path }, sc.label);
						}),
						h("button", { style: parent ? style.secondaryButton : Object.assign({}, style.secondaryButton, style.disabled), disabled: !parent, onClick: function () { if (parent) load(parent); } }, "Up"),
						remembered.map(function (f, i) {
							return h("button", { key: i, style: style.secondaryButton, title: f, onClick: function () { load(f); } }, f);
						}),
						h("button", { style: dir ? style.secondaryButton : Object.assign({}, style.secondaryButton, style.disabled), disabled: !dir, onClick: toggleRemember }, isRemembered ? "Forget Folder" : "Remember Folder"),
						loading ? h("span", { style: style.small }, "loading…") : null,
						err ? h("span", { style: style.error }, String(err)) : null,
					),
					h(
						"ul",
						{ style: style.list },
						(entries || []).map(function (en, i) {
							if (en.type === "dir") {
								return h("li", { key: i, style: Object.assign({}, style.item, style.itemDir), onClick: function () { load(dir + "/" + en.name); } }, en.name + "/");
							}
							var sel = selected === en.name;
							return h("li", {
								key: i,
								style: Object.assign({}, style.item, style.itemFile, sel ? style.itemSel : {}, dirMode ? style.disabled : {}),
								onClick: dirMode ? undefined : function () { setSelected(en.name); },
								title: en.name,
							}, en.name + (en.size != null ? "  (" + fmtGiB(en.size) + ")" : ""));
						}),
					),
					h(
						"div",
						{ style: style.row },
						h("button", {
							style: (dirMode ? dir : selected) ? style.button : Object.assign({}, style.button, style.disabled),
							disabled: dirMode ? !dir : !selected,
							onClick: function () { if (dirMode) { if (dir) props.onPick(dir); } else if (selected) props.onPick(dir + "/" + selected); },
						}, dirMode ? "Use this folder" : "Use this file"),
						h("button", { style: style.secondaryButton, onClick: props.onClose }, "Cancel"),
					),
				),
			);
		}
		// ---- Terminal modal: live tail of llama-server.log ----
		var TERM_MAX_LINES = 3000; // in-memory line cap (drop head)
		var TERM_SEED_BYTES = 262144; // 256 KiB seed window / per-poll cap
		// ANSI SGR palette, tuned for dark backgrounds (30 maps to dim gray
		// instead of pure black so "black" log text stays readable).
		var ANSI_FG = ["#5f5f5f", "#f07178", "#c3e88d", "#ffcb6b", "#82aaff", "#c792ea", "#89ddff", "#eeffff"];
		var ANSI_FG_BRIGHT = ["#8a8f98", "#ff8b92", "#d4f7a3", "#ffdf80", "#9cc4ff", "#d7aefb", "#a6f0ff", "#ffffff"];
		function ansi256Color(n) {
			if (n < 8) return ANSI_FG[n];
			if (n < 16) return ANSI_FG_BRIGHT[n - 8];
			if (n < 232) {
				var v = n - 16, r = Math.floor(v / 36), g = Math.floor((v % 36) / 6), b = v % 6;
				var comp = function (c) { return c === 0 ? 0 : c * 40 + 55; };
				return "rgb(" + comp(r) + "," + comp(g) + "," + comp(b) + ")";
			}
			var gray = 8 + (n - 232) * 10;
			return "rgb(" + gray + "," + gray + "," + gray + ")";
		}
		function blankSgrStyle() { return { fg: null, bg: null, b: false, i: false, u: false }; }
		// Apply one SGR parameter list onto a style object (mutated).
		function applySgr(style, params) {
			for (var k = 0; k < params.length; k++) {
				var p = params[k];
				if (p === 0) { style.fg = null; style.bg = null; style.b = false; style.i = false; style.u = false; }
				else if (p === 1) style.b = true;
				else if (p === 3) style.i = true;
				else if (p === 4) style.u = true;
				else if (p === 22) style.b = false;
				else if (p === 23) style.i = false;
				else if (p === 24) style.u = false;
				else if (p >= 30 && p <= 37) style.fg = ANSI_FG[p - 30];
				else if (p >= 90 && p <= 97) style.fg = ANSI_FG_BRIGHT[p - 90];
				else if (p === 39) style.fg = null;
				else if (p >= 40 && p <= 47) style.bg = ANSI_FG[p - 40];
				else if (p >= 100 && p <= 107) style.bg = ANSI_FG_BRIGHT[p - 100];
				else if (p === 49) style.bg = null;
				else if ((p === 38 || p === 48) && k + 2 < params.length && params[k + 1] === 5) {
					if (params[k + 2] >= 0 && params[k + 2] <= 255) {
						if (p === 38) style.fg = ansi256Color(params[k + 2]);
						else style.bg = ansi256Color(params[k + 2]);
					}
					k += 2;
				} else if ((p === 38 || p === 48) && k + 4 < params.length && params[k + 1] === 2) {
					var rgb = "rgb(" + params[k + 2] + "," + params[k + 3] + "," + params[k + 4] + ")";
					if (p === 38) style.fg = rgb; else style.bg = rgb;
					k += 4;
				}
			}
		}
		// Parse one physical line (no \n) into styled spans, mutating the
		// running SGR style. Returns { spans, plain }.
		function parseAnsiLine(raw, style) {
			var spans = [], plain = "";
			var parts = raw.split(/(\x1b\[[0-9;]*m)/);
			for (var i = 0; i < parts.length; i++) {
				var seg = parts[i];
				if (seg === "") continue;
				if (seg.charAt(0) === "\x1b") {
					var nums = seg.slice(2, -1).split(";");
					var params = [];
					for (var j = 0; j < nums.length; j++) {
						params.push(nums[j] === "" ? 0 : Number(nums[j]));
					}
					applySgr(style, params);
				} else if (seg) {
					spans.push({ t: seg, fg: style.fg, bg: style.bg, b: style.b, i: style.i, u: style.u });
					plain += seg;
				}
			}
			return { spans: spans, plain: plain };
		}
		// Parse a raw chunk into display lines. A chunk usually ends mid-line,
		// so the text after the last \n is held back raw (st.pending) and
		// prepended to the next chunk — \r overwrites and escape sequences
		// therefore always resolve whole. Style state mutates on parsed
		// (complete-line) text only.
		function parseAnsiChunk(text, st) {
			text = (st.carry || "") + (st.pending || "") + text;
			st.carry = "";
			st.pending = "";
			var cut = text.lastIndexOf("\n");
			var head = cut === -1 ? "" : text.slice(0, cut);
			st.pending = cut === -1 ? text : text.slice(cut + 1);
			if (st.pending.length > 1048576) st.pending = st.pending.slice(-1048576);
			var tail = /(\x1b(?:\[[0-9;]*)?)$/.exec(st.pending);
			if (tail) { st.pending = st.pending.slice(0, st.pending.length - tail[1].length); st.carry = tail[1]; }
			if (head === "") return [];
			head = head.replace(/\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)/g, "");
			head = head.replace(/\x1b[()][0-9A-Z]/g, "");
			// Strip non-SGR CSI (SGR ends in m/M: m feeds the color parser
			// below, M deletes a line so dropping it is a no-op for a tail).
			head = head.replace(/\x1b\[[0-9;?]*[A-LN-Za-ln-z]/g, "");
			var out = [];
			var raws = head.split("\n");
			for (var i = 0; i < raws.length; i++) {
				var line = raws[i];
				if (line.charAt(line.length - 1) === "\r") line = line.slice(0, -1);
				var segs = line.split("\r");
				out.push(parseAnsiLine(segs[segs.length - 1], st.style));
			}
			return out;
		}
		function newTermState() { return { style: blankSgrStyle(), carry: "" }; }
		function spanStyle(sp) {
			var s = {};
			if (sp.fg) s.color = sp.fg;
			if (sp.bg) s.backgroundColor = sp.bg;
			if (sp.b) s.fontWeight = 700;
			if (sp.i) s.fontStyle = "italic";
			if (sp.u) s.textDecoration = "underline";
			return s;
		}
		function TerminalModal(props) {
			var linesPair = useStatePair([]);
			var lines = linesPair[0], setLines = linesPair[1];
			var errPair = useStatePair(null);
			var err = errPair[0], setErr = errPair[1];
			var filterPair = useStatePair("");
			var filter = filterPair[0], setFilter = filterPair[1];
			var newPair = useStatePair(0);
			var newCount = newPair[0], setNewCount = newPair[1];
			var bodyRef = React.useRef(null);
			var cursorRef = React.useRef(null); // server's nextOffset; null until the seed fetch lands
			var pinnedRef = React.useRef(true); // auto-follow while the user is at the bottom
			var inflightRef = React.useRef(false);
			var termRef = React.useRef(null); // parser state {style, carry, pending}, survives polls
			var lineNoRef = React.useRef(0); // stable keys for appended lines
			var prevLenRef = React.useRef(0); // follow-scroll only when lines actually grew
			if (termRef.current === null) termRef.current = newTermState();
			var raf = typeof requestAnimationFrame === "function"
				? requestAnimationFrame
				: function (f) { f(); };
			var rafIdRef = React.useRef(0);

			var onScroll = function () {
				if (rafIdRef.current) return;
				rafIdRef.current = 1;
				raf(function () {
					rafIdRef.current = 0;
					var el = bodyRef.current;
					if (!el) return;
					var pinned = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
					pinnedRef.current = pinned;
					if (pinned) setNewCount(0);
				});
			};

			var appendText = function (text) {
				var parsed = parseAnsiChunk(text, termRef.current);
				if (parsed.length === 0) return;
				var numbered = [];
				for (var i = 0; i < parsed.length; i++) {
					numbered.push({ n: lineNoRef.current++, spans: parsed[i].spans, plain: parsed[i].plain });
				}
				setLines(function (prev) {
					var next = prev.concat(numbered);
					return next.length > TERM_MAX_LINES ? next.slice(next.length - TERM_MAX_LINES) : next;
				});
				if (!pinnedRef.current) {
					setNewCount(function (c) { return c + numbered.length; });
				}
			};

			useEffect(function () {
				var cancelled = false;
				var timer = null;
				var pull = function (params) {
					if (inflightRef.current) return;
					inflightRef.current = true;
					get("/local-models/logs" + params)
						.then(function (body) {
							if (cancelled || !body || typeof body.text !== "string") return;
							cursorRef.current = body.nextOffset;
							setErr(null);
							if (body.text.length > 0) appendText(body.text);
						})
						.catch(function (e2) { if (!cancelled) setErr(String(e2)); })
						.finally(function () { inflightRef.current = false; });
				};
				setLines([]);
				setNewCount(0);
				cursorRef.current = null;
				pinnedRef.current = true;
				pull("?max=" + TERM_SEED_BYTES);
				timer = setInterval(function () {
					pull(cursorRef.current === null ? "" : "?offset=" + cursorRef.current + "&max=" + TERM_SEED_BYTES);
				}, 2000);
				return function () {
					cancelled = true;
					clearInterval(timer);
				};
			}, []);

			// Follow: scroll to bottom after renders that actually grew the buffer.
			useEffect(function () {
				if (lines.length !== prevLenRef.current) {
					prevLenRef.current = lines.length;
					if (pinnedRef.current) {
						var el = bodyRef.current;
						if (el) el.scrollTop = el.scrollHeight;
					}
				}
				return undefined;
			});

			var jumpToEnd = function () {
				pinnedRef.current = true;
				setNewCount(0);
				var el = bodyRef.current;
				if (el) el.scrollTop = el.scrollHeight;
			};
			var clearView = function () {
				setLines([]);
				setNewCount(0);
			};

			var q = filter.trim().toLowerCase();
			var view = q ? lines.filter(function (l) { return l.plain.toLowerCase().indexOf(q) !== -1; }) : lines;

			return h(
				"div",
				{ style: style.terminalOverlay, onClick: props.onClose },
				h(
					"div",
					{ style: style.terminal, onClick: function (ev) { ev.stopPropagation(); } },
					h(
						"div",
						{ style: style.row },
						h("h3", { style: style.cardTitle }, "llama-server terminal"),
						h("span", { style: style.small }, "live tail of " + (props.logPath || "llama-server.log")),
						h("span", { style: { flex: 1 } }),
						h("input", {
							style: Object.assign({}, style.input, { width: 170, height: 30 }),
							value: filter, placeholder: "Filter lines…",
							onChange: function (ev) { setFilter(ev.target.value); }, spellCheck: false,
						}),
						!pinnedRef.current && newCount > 0 && !q
							? h("button", { style: style.secondaryButton, onClick: jumpToEnd }, "↓ " + newCount + " new")
							: h("button", { style: style.secondaryButton, onClick: jumpToEnd }, "Jump to end"),
						h("button", { style: style.secondaryButton, onClick: clearView }, "Clear view"),
						h("button", { style: style.secondaryButton, onClick: props.onClose }, "Close"),
					),
					err ? h("p", { style: style.error }, String(err)) : null,
					h(
						"div",
						{ ref: bodyRef, style: style.termBody, onScroll: onScroll },
						view.length > 0
							? view.map(function (l) {
								return h("div", { key: l.n }, l.spans.length > 0
									? l.spans.map(function (sp, si) {
										return h("span", { key: si, style: spanStyle(sp) }, sp.t);
									})
									: "\u00a0");
							})
							: h("span", { style: style.small }, lines.length > 0 ? "no lines match the filter" : "no output yet"),
					),
				),
			);
		}

		// ---- main section ----
		function LocalModelsSection() {
			var selPair = useStatePair(null);
			var selectedPath = selPair[0], setSelectedPath = selPair[1];
			var metaPair = useStatePair(null);
			var metaRoot = metaPair[0], setMetaRoot = metaPair[1];
			var metaErrPair = useStatePair(null);
			var metaErr = metaErrPair[0], setMetaErr = metaErrPair[1];
			var metaBusyPair = useStatePair(false);
			var metaBusy = metaBusyPair[0], setMetaBusy = metaBusyPair[1];
			var pickerPair = useStatePair(false);
			var pickerOpen = pickerPair[0], setPickerOpen = pickerPair[1];
			var ctxPair = useStatePair(8192);
			var ctx = ctxPair[0], setCtx = ctxPair[1];
			var ctxTextPair = useStatePair("8192");
			var ctxText = ctxTextPair[0], setCtxText = ctxTextPair[1];
			var mtpPair = useStatePair(2);
			var mtp = mtpPair[0], setMtp = mtpPair[1];
			// MoE expert placement: "gpu" (all experts in VRAM), "cpu" (all in
			// RAM via --cpu-moe), "layers" (first N layers in RAM via
			// --n-cpu-moe N). expertUsedText "" = stock top-k from the GGUF.
			var moeModePair = useStatePair("gpu");
			var moeMode = moeModePair[0], setMoeMode = moeModePair[1];
			var nCpuMoePair = useStatePair("8");
			var nCpuMoeText = nCpuMoePair[0], setNCpuMoeText = nCpuMoePair[1];
			var expUsedPair = useStatePair("");
			var expertUsedText = expUsedPair[0], setExpertUsedText = expUsedPair[1];
			var effortPair = useStatePair("medium");
			var effort = effortPair[0], setEffort = effortPair[1];
			// KV cache quantization, one selector per store (--cache-type-k /
			// --cache-type-v). Defaults = the tuned 16 GiB pair.
			var kvKPair = useStatePair(KV_DEFAULT_K);
			var kvTypeK = kvKPair[0], setKvTypeK = kvKPair[1];
			var kvVPair = useStatePair(KV_DEFAULT_V);
			var kvTypeV = kvVPair[0], setKvTypeV = kvVPair[1];
			// Preserve the full reasoning trace in history (--reasoning-preserve).
			// Off (default, historical behavior) sends --no-reasoning-preserve.
			var preservePair = useStatePair(false);
			var preserveThinking = preservePair[0], setPreserveThinking = preservePair[1];
			var mmPair = useStatePair(null);
			var mmprojPath = mmPair[0], setMmprojPath = mmPair[1];
			var mmCpuPair = useStatePair(true);
			var mmprojCpu = mmCpuPair[0], setMmprojCpu = mmCpuPair[1];
			var mmMetaPair = useStatePair(null);
			var mmMetaRoot = mmMetaPair[0], setMmMetaRoot = mmMetaPair[1];
			var mmErrPair = useStatePair(null);
			var mmErr = mmErrPair[0], setMmErr = mmErrPair[1];
			var mmPickerPair = useStatePair(false);
			var mmPickerOpen = mmPickerPair[0], setMmPickerOpen = mmPickerPair[1];
			var termPair = useStatePair(false);
			var termOpen = termPair[0], setTermOpen = termPair[1];
			var statusPair = useStatePair({ status: "idle", portUp: false });
			var status = statusPair[0], setStatus = statusPair[1];
			var busyPair = useStatePair(false);
			var busy = busyPair[0], setBusy = busyPair[1];
			var errPair = useStatePair(null);
			var error = errPair[0], setError = errPair[1];
			var regPair = useStatePair(null);
			var registered = regPair[0], setRegistered = regPair[1];
			var profPair = useStatePair([]);
			var profiles = profPair[0], setProfiles = profPair[1];
			var profNamePair = useStatePair("");
			var profileName = profNamePair[0], setProfileName = profNamePair[1];
			// Runtime card (llama.cpp binaries folder + browser shortcuts). This
			// is a server-wide setting, not a per-launch field: it is persisted
			// through /local-models/settings and adopted from the status poll,
			// and profiles deliberately do NOT carry it — a profile must stay
			// portable between machines, and the router uses one binary anyway.
			var binPair = useStatePair("");
			var binDraft = binPair[0], setBinDraft = binPair[1];
			// Last value the server confirmed: binDraft !== binSaved means the
			// user has an unsaved edit the poll must not overwrite.
			var binSavedPair = useStatePair("");
			var binSaved = binSavedPair[0], setBinSaved = binSavedPair[1];
			var binChkPair = useStatePair(null);
			var binCheck = binChkPair[0], setBinCheck = binChkPair[1];
			var binBusyPair = useStatePair(false);
			var binBusy = binBusyPair[0], setBinBusy = binBusyPair[1];
			var binPickerPair = useStatePair(false);
			var binPickerOpen = binPickerPair[0], setBinPickerOpen = binPickerPair[1];
			var scDraftPair = useStatePair("");
			var shortcutDraft = scDraftPair[0], setShortcutDraft = scDraftPair[1];
			var scBusyPair = useStatePair(false);
			var scBusy = scBusyPair[0], setScBusy = scBusyPair[1];
			// Mutation base for the folder list: the poll snapshot is up to 2 s
			// old, so adding/removing from it loses a just-applied change. This
			// ref is updated from every POST response (authoritative), and a poll
			// may only overwrite it once it reports the list we just wrote (or
			// after a 5 s grace, in case someone else changed it meanwhile).
			var scListRef = React.useRef(null);
			var scPendingRef = React.useRef(null);
			// VRAM budget override (GiB, "" = auto-detect): a machine-level
			// setting like binPath, so it follows the same draft/saved dance.
			var vramPair = useStatePair("");
			var vramDraft = vramPair[0], setVramDraft = vramPair[1];
			var vramSavedPair = useStatePair("");
			var vramSaved = vramSavedPair[0], setVramSaved = vramSavedPair[1];
			// Multi-GPU placement. Defaults reproduce llama.cpp's own behavior
			// (layer split, proportions from VRAM), so an untouched tab launches
			// exactly what it launched before this control existed.
			var splitPair = useStatePair(SPLIT_DEFAULT);
			var splitMode = splitPair[0], setSplitMode = splitPair[1];
			var tsPair = useStatePair("");
			var tensorSplitText = tsPair[0], setTensorSplitText = tsPair[1];

			var meta = metaRoot ? metaRoot.meta : null;
			var trainCtx = meta ? meta.contextLength : null;
			var ctxMax = trainCtx && trainCtx > 0 ? trainCtx : FALLBACK_MAX_CTX;

			// MLA models (DeepSeek-style latent KV) reject mixed K/V cache
			// types in llama.cpp — surface it and keep Load disabled. Mirrors
			// kvTypesMustMatch() in lib/index.js (`is_mla() || deepseek4`).
			var kvMustMatch = !!(meta && (meta.isMla === true || meta.arch === "deepseek4"));
			var kvPairOk = !kvMustMatch || kvTypeK === kvTypeV;

			var LS_PATH = "dsh-local-models:model-path";
			var LS_CTX = "dsh-local-models:ctx";
			var LS_MTP = "dsh-local-models:mtp";
			var LS_EFFORT = "dsh-local-models:effort";
			var LS_KVTYPE_K = "dsh-local-models:kv-type-k";
			var LS_KVTYPE_V = "dsh-local-models:kv-type-v";
			var LS_PRESERVE = "dsh-local-models:preserve-thinking";
			var LS_MMPROJ = "dsh-local-models:mmproj-path";
			var LS_MMPROJ_CPU = "dsh-local-models:mmproj-cpu";
			var LS_EXPERTUSED = "dsh-local-models:expert-used";
			var LS_MOEMODE = "dsh-local-models:moe-mode";
			var LS_NCPUMOE = "dsh-local-models:n-cpu-moe";
			// Runtime: the saved value is restored into BOTH the draft and the
			// "saved" baseline, so a tab that opens with a stale cache still
			// adopts the server's value on the first poll (equal values = clean).
			var LS_BIN = "dsh-local-models:bin-path";
			var LS_VRAM = "dsh-local-models:vram-gb";
			var LS_SPLIT = "dsh-local-models:split-mode";
			var LS_TSPLIT = "dsh-local-models:tensor-split";

			// Restore the previous selection on mount (session survived a refresh).
			useEffect(function () {
				try {
					var p = localStorage.getItem(LS_PATH);
					if (p) { setSelectedPath(p); fitAppliedRef.current = true; }
					var mp = localStorage.getItem(LS_MMPROJ);
					if (mp) setMmprojPath(mp);
					var mc = localStorage.getItem(LS_MMPROJ_CPU);
					if (mc === "1") setMmprojCpu(true);
					else if (mc === "0") setMmprojCpu(false);
					var c = Number(localStorage.getItem(LS_CTX));
					if (Number.isFinite(c) && c > 0) { setCtx(c); setCtxText(String(c)); }
					var m = Number(localStorage.getItem(LS_MTP));
					if (Number.isFinite(m) && m >= 0 && m <= MTP_MAX) setMtp(m);
					setEffort(normalizeEffort(localStorage.getItem(LS_EFFORT)));
					setKvTypeK(normalizeKvType(localStorage.getItem(LS_KVTYPE_K), KV_DEFAULT_K));
					setKvTypeV(normalizeKvType(localStorage.getItem(LS_KVTYPE_V), KV_DEFAULT_V));
					if (localStorage.getItem(LS_PRESERVE) === "1") setPreserveThinking(true);
					var mm = localStorage.getItem(LS_MOEMODE);
					if (mm === "cpu" || mm === "layers" || mm === "gpu") setMoeMode(mm);
					var nc = Number(localStorage.getItem(LS_NCPUMOE));
					if (Number.isFinite(nc) && nc > 0) setNCpuMoeText(String(Math.floor(nc)));
					var eu = localStorage.getItem(LS_EXPERTUSED);
					if (eu != null && eu !== "" && Number.isFinite(Number(eu)) && Number(eu) > 0) setExpertUsedText(String(Math.floor(Number(eu))));
					var bp = localStorage.getItem(LS_BIN);
					if (bp != null) { setBinDraft(bp); setBinSaved(bp); }
					var vg = localStorage.getItem(LS_VRAM);
					if (vg != null) { setVramDraft(vg); setVramSaved(vg); }
					setSplitMode(normalizeSplitMode(localStorage.getItem(LS_SPLIT)));
					var ts = localStorage.getItem(LS_TSPLIT);
					if (ts) setTensorSplitText(ts);
				} catch { /* storage unavailable */ }
				return undefined;
			}, []);

			// Plain (non-memoized) function: usePoll always invokes the latest
			// render's copy via its ref, so the poll sees fresh state (e.g.
			// the current doRegister/status instead of first-render ones).
			var refresh = function () {
				get("/local-models/status").then(function (body) {
					if (!body) return;
					setStatus(body);
					// Router auto-register: starting the router always refreshes
					// the local-router route, no manual Register press needed.
					if (pendingRouterRegRef.current && body.mode === "router" && body.status === "ready") {
						pendingRouterRegRef.current = false;
						doRegister();
						return;
					}
					// The server outlives page refreshes: adopt the running model so the
					// tab re-syncs (Stops stay available) even without any stored state.
					if (body.modelPath && !selectedPathRef.current) {
						setSelectedPath(body.modelPath);
						if (body.ctx) { setCtx(body.ctx); setCtxText(String(body.ctx)); }
						if (body.mtpHeads != null && Number.isFinite(Number(body.mtpHeads))) setMtp(Math.min(Math.max(0, Math.floor(Number(body.mtpHeads))), MTP_MAX));
						setKvTypeK(normalizeKvType(body.kvTypeK, KV_DEFAULT_K));
						setKvTypeV(normalizeKvType(body.kvTypeV, KV_DEFAULT_V));
						if (body.reasoningEffort != null) setEffort(normalizeEffort(body.reasoningEffort));
						else if (body.effort != null) setEffort(normalizeEffort(body.effort));
						if (body.preserveThinking != null) setPreserveThinking(body.preserveThinking === true);
						else if (body.preserve_thinking != null) setPreserveThinking(body.preserve_thinking === true);
						if (body.cpuMoe === true) setMoeMode("cpu");
						else if (body.nCpuMoe > 0) { setMoeMode("layers"); setNCpuMoeText(String(body.nCpuMoe)); }
						if (body.expertUsed != null) setExpertUsedText(String(body.expertUsed));
						if (body.mmprojCpu != null) setMmprojCpu(body.mmprojCpu !== false);
						if (body.splitMode != null) setSplitMode(normalizeSplitMode(body.splitMode));
						if (body.tensorSplit != null) setTensorSplitText(String(body.tensorSplit));
					}
					if (body.mmprojPath && !mmprojPathRef.current && !clearedMmprojRef.current) setMmprojPath(body.mmprojPath);
					if (body.mmprojPath && !mmprojPathRef.current && !clearedMmprojRef.current && body.mmprojCpu != null) setMmprojCpu(body.mmprojCpu !== false);
					// Runtime: adopt the server value only while the field is
					// clean — an unsaved edit must survive the 2 s poll.
					if (!binDirtyRef.current) {
						var serverBin = typeof body.binConfigured === "string" ? body.binConfigured : "";
						setBinDraft(serverBin);
						setBinSaved(serverBin);
					}
					var serverPaths = (Array.isArray(body.shortcuts) ? body.shortcuts : []).map(function (x) { return x.path; });
					var pending = scPendingRef.current;
					if (!pending || Date.now() - pending.at > 5000 || samePaths(serverPaths, pending.list)) {
						scListRef.current = serverPaths;
						scPendingRef.current = null;
					}
					if (!vramDirtyRef.current) {
						var serverVram = body.vramOverrideGb > 0 ? String(body.vramOverrideGb) : "";
						setVramDraft(serverVram);
						setVramSaved(serverVram);
					}
				}).catch(function () {});
			};

			var selectedPathRef = React.useRef(selectedPath);
			selectedPathRef.current = selectedPath;
			var mmprojPathRef = React.useRef(mmprojPath);
			mmprojPathRef.current = mmprojPath;
			// Explicit user "Clear" suppresses the status poll from re-feeding
			// the server's retained mmprojPath until a new file is picked.
			var clearedMmprojRef = React.useRef(false);
			// Set by "Load" on a profile so the GGUF-meta effect keeps the
			// profile's context instead of resetting to min(8192, train_ctx).
			var pendingProfileRef = React.useRef(null);
			// Fit-to-VRAM auto-apply runs once per freshly picked file (not on
			// restore or profile loads — those own their settings already).
			var fitAppliedRef = React.useRef(false);
			// Set when the router starts: the refresh poll registers the
			// router route in dsh as soon as the server reports ready.
			var pendingRouterRegRef = React.useRef(false);
			// Unsaved Runtime edit (typed path ≠ saved path): the poll must not
			// clobber it, exactly like the model/mmproj adopt guards.
			var binDirtyRef = React.useRef(false);
			binDirtyRef.current = binDraft.trim() !== binSaved;
			var vramDirtyRef = React.useRef(false);
			vramDirtyRef.current = vramDraft.trim() !== vramSaved;
			// Same list, same order: what the server echoes for a write of ours.
			var samePaths = function (a, b) {
				if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
				for (var i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
				return true;
			};

			usePoll(refresh, 2000);

			// Persist the selection so a refresh restores it.
			useEffect(function () {
				try {
					if (selectedPath) localStorage.setItem(LS_PATH, selectedPath);
					else localStorage.removeItem(LS_PATH);
					localStorage.setItem(LS_CTX, String(ctx));
					localStorage.setItem(LS_MTP, String(mtp));
					localStorage.setItem(LS_EFFORT, effort);
					localStorage.setItem(LS_KVTYPE_K, kvTypeK);
					localStorage.setItem(LS_KVTYPE_V, kvTypeV);
					localStorage.setItem(LS_PRESERVE, preserveThinking ? "1" : "0");
					localStorage.setItem(LS_MOEMODE, moeMode);
					localStorage.setItem(LS_NCPUMOE, nCpuMoeText);
					localStorage.setItem(LS_MMPROJ_CPU, mmprojCpu ? "1" : "0");
					if (expertUsedText !== "" && Number.isFinite(Number(expertUsedText))) localStorage.setItem(LS_EXPERTUSED, String(expertUsedText));
					else localStorage.removeItem(LS_EXPERTUSED);
					if (mmprojPath) localStorage.setItem(LS_MMPROJ, mmprojPath);
					else localStorage.removeItem(LS_MMPROJ);
					localStorage.setItem(LS_BIN, binDraft);
					localStorage.setItem(LS_VRAM, vramDraft);
					localStorage.setItem(LS_SPLIT, splitMode);
					if (tensorSplitText) localStorage.setItem(LS_TSPLIT, tensorSplitText);
					else localStorage.removeItem(LS_TSPLIT);
				} catch { /* storage unavailable */ }
				return undefined;
			}, [selectedPath, ctx, mtp, effort, preserveThinking, mmprojPath, mmprojCpu, moeMode, nCpuMoeText, expertUsedText, kvTypeK, kvTypeV, binDraft, splitMode, tensorSplitText, vramDraft]);
			// Fetch GGUF metadata when a file is picked.
			useEffect(function () {
				if (!selectedPath) return;
				var cancelled = false;
				setMetaBusy(true);
				setMetaErr(null);
				// New file: back to the stock top-k until the user overrides it
				// (a profile load re-applies its own value after this effect).
				if (!pendingProfileRef.current || pendingProfileRef.current.expertUsed == null) setExpertUsedText("");
				post("/local-models/gguf-meta", { path: selectedPath })
					.then(function (res) {
						if (cancelled) return;
						if (res.ok && res.body && res.body.meta) {
							setMetaRoot(res.body);
							var tc = res.body.meta.contextLength;
							var pending = pendingProfileRef.current;
							var def = pending && pending.ctx ? pending.ctx : (tc && tc > 0 ? Math.min(8192, tc) : 8192);
							setCtx(def);
							setCtxText(String(def));
						} else {
							setMetaRoot(null);
							setMetaErr((res.body && res.body.error) || "could not parse GGUF");
						}
					})
					.catch(function (e2) { if (!cancelled) { setMetaRoot(null); setMetaErr(String(e2)); } })
					.finally(function () { pendingProfileRef.current = null; if (!cancelled) setMetaBusy(false); });
				return function () { cancelled = true; };
			}, [selectedPath]);

			// ---- GPU / VRAM budget (detected by the node half, or pinned) ----
			// The estimate and the "fits" verdict are only as good as this
			// number, so the source is always shown next to it.
			var gpus = Array.isArray(status.gpus) ? status.gpus : [];
			var gpuCount = gpus.length;
			var vramBytes = Number.isFinite(status.vramTotalBytes) && status.vramTotalBytes > 0
				? status.vramTotalBytes
				: TOTAL_VRAM_BYTES;
			var vramGb = fmtVramGb(vramBytes);
			var vramSource = status.vramSource || "assumed";
			var splitVisible = gpuCount > 1 || splitMode !== SPLIT_DEFAULT || tensorSplitText.trim() !== "";
			var gpuLabel = gpuCount > 0
				? gpus.map(function (g) { return g.name + " (" + fmtVramGb(g.vramBytes) + " GB)"; }).join(" + ")
				: vramSource === "probing" ? "detecting…"
					: vramSource === "override" ? "manual budget"
						: "not detected — assuming " + vramGb + " GB";

			// Live VRAM estimate.
			var estimate = useMemo(function () {
				if (!meta) return null;
				if (!meta.nLayers || !meta.nKvHeads || !meta.headDim) {
					return { unavailable: true };
				}
				var gdn = gdnLayout(meta.nLayers, meta.fullAttnInterval, meta.nextnPredictLayers);
				// The fixed MTP draft is always on when mtp > 0: no ctx ceiling
				// (the old softcap is gone), so the draft KV always counts.
				var specOn = mtp > 0;
				var kvLayersOverride = gdn.hybrid ? gdn.attnTrunk + (specOn ? gdn.mtp : 0) : null;
				// Upstream keeps the SSM recurrent state in f32 (the fork's
				// f16-state env is gone), so estimate with f32 state bytes.
				var recrFixed = gdn.hybrid && meta.ssmInnerSize
					? gdnRecurrentBytes(gdn.trunk - gdn.attnTrunk, meta.ssmStateSize ?? 128, meta.ssmInnerSize, meta.ssmNGroup ?? 16, meta.ssmDtRank ?? 48, meta.ssmConvKernel ?? 4, false)
					: 0;
				var kv = kvBytesFor(ctx, meta.nLayers, meta.nKvHeads, meta.headDim, kvTypeK, kvTypeV, meta.slidingWindow, null, kvLayersOverride);
				var compute = estimateComputeBytes(ctx, meta.nVocab, gdn.hybrid);
				// mmproj on CPU lives in RAM: only a GPU mmproj costs VRAM.
				var mmprojBytes = (!mmprojCpu && mmMetaRoot) ? mmMetaRoot.weightsBytes : 0;
				// MoE expert split: experts kept on CPU live in RAM, so only
				// the VRAM-resident weights count toward the total. Mirrors
				// moeVramWeights() in lib/index.js (kept in sync by hand).
				var cpuExpertBytes = 0, moeUnknown = false;
				if (meta.isMoe && moeMode !== "gpu") {
					if (meta.expertBytes == null) {
						moeUnknown = true;
					} else if (moeMode === "cpu") {
						cpuExpertBytes = meta.expertBytes;
					} else {
						var nl = parseInt(nCpuMoeText, 10);
						var byLayer = meta.expertBytesByLayer || {};
						var sum = 0, known = false, k;
						for (k in byLayer) {
							if (Number(k) < nl) { sum += byLayer[k]; known = true; }
						}
						if (!known || !isFinite(nl) || nl <= 0) moeUnknown = true;
						else cpuExpertBytes = sum;
					}
				}
				var weightsVram = Math.max(metaRoot.weightsBytes - cpuExpertBytes, 0);
				var total = weightsVram + mmprojBytes + recrFixed + kv + compute + OVERHEAD_BYTES;
				// Fit-to-VRAM: smallest first-N-layers CPU split that brings the
				// total within the safe budget. n = -1 means even all experts
				// on CPU miss (KV too big → reduce ctx). Null when the expert
				// split is unknown or the model already fits on GPU.
				var fit = null;
				if (meta.isMoe && meta.expertBytes != null && meta.expertBytesByLayer) {
					var baseNonWeights = mmprojBytes + recrFixed + kv + compute + OVERHEAD_BYTES;
					var budget = vramBytes - SAFE_MARGIN_BYTES;
					var byL = meta.expertBytesByLayer, maxL = -1, kk;
					for (kk in byL) { maxL = Math.max(maxL, Number(kk)); }
					var layerCount = meta.nLayers || (maxL + 1);
					var prefix = function (n) { var s = 0, key; for (key in byL) { if (Number(key) < n) s += byL[key]; } return s; };
					var fitsAt = function (n) { return metaRoot.weightsBytes - prefix(n) + baseNonWeights <= budget; };
					if (!fitsAt(0) && layerCount > 0) {
						if (!fitsAt(layerCount)) {
							fit = { n: -1, total: null, cpuAll: false };
						} else {
							var lo = 0, hi = layerCount, mid;
							while (hi - lo > 1) { mid = (lo + hi) >> 1; if (fitsAt(mid)) hi = mid; else lo = mid; }
							fit = { n: hi, total: metaRoot.weightsBytes - prefix(hi) + baseNonWeights, cpuAll: hi >= layerCount };
						}
					}
				}
				var input = {
					weightsBytes: weightsVram + mmprojBytes,
					nLayers: meta.nLayers, nKvHeads: meta.nKvHeads, headDim: meta.headDim,
					kvTypeK: kvTypeK, kvTypeV: kvTypeV,
					nVocab: meta.nVocab, slidingWindow: meta.slidingWindow, fullAttnShare: null,
					kvLayersOverride: kvLayersOverride, recurrentFixedBytes: recrFixed,
					overheadBytes: OVERHEAD_BYTES,
				};
				var maxCtx = maxContextFor(vramBytes, input);
				return {
					gdn: gdn, specOn: specOn, kvLayersOverride: kvLayersOverride, recrFixed: recrFixed,
					kv: kv, compute: compute, total: total, weights: metaRoot.weightsBytes, weightsVram: weightsVram,
					cpuExpertBytes: cpuExpertBytes, moeUnknown: moeUnknown, mmprojBytes: mmprojBytes,
					fit: fit, overBudget: total > vramBytes - SAFE_MARGIN_BYTES,
					fits: total <= vramBytes,
					fitsSafe: total <= vramBytes - SAFE_MARGIN_BYTES,
					maxCtx: maxCtx,
					pct: Math.round(100 * total / vramBytes),
				};
			}, [meta, metaRoot, mmMetaRoot, mmprojCpu, ctx, mtp, moeMode, nCpuMoeText, kvTypeK, kvTypeV, vramBytes]);

			// Smart default: once per freshly picked file, auto-apply the
			// fit-to-VRAM split when the current settings exceed the budget.
			// Waits for metaBusy so the context default has landed first.
			useEffect(function () {
				if (!estimate || metaBusy || !selectedPath || fitAppliedRef.current) return;
				fitAppliedRef.current = true;
				if (estimate.fit && estimate.fit.n >= 0 && estimate.overBudget) applyFit();
				return undefined;
			});

			// Fetch the vision projector (mmproj) metadata when picked.
			useEffect(function () {
				if (!mmprojPath) { setMmMetaRoot(null); setMmErr(null); return; }
				var cancelled = false;
				setMmErr(null);
				post("/local-models/gguf-meta", { path: mmprojPath })
					.then(function (res) {
						if (cancelled) return;
						if (res.ok && res.body && res.body.meta) setMmMetaRoot(res.body);
						else setMmErr((res.body && res.body.error) || "could not parse mmproj GGUF");
					})
					.catch(function (e2) { if (!cancelled) setMmErr(String(e2)); });
				return function () { cancelled = true; };
			}, [mmprojPath]);

			var acting = busy || status.status === "starting";
			var running = status.status === "ready";
			var portUp = status.portUp === true;
			var stoppable = running || portUp;
			var canRun = !!meta && !!selectedPath && !acting && !running && status.mode !== "router" && kvPairOk;
			var statusText = statusLine(status, portUp);

			var onCtxSlider = function (ev) {
				var v = Number(ev.target.value);
				setCtx(v);
				setCtxText(String(v));
			};
			var onCtxText = function (ev) {
				setCtxText(ev.target.value);
				var n = Number(ev.target.value);
				if (Number.isFinite(n) && n > 0) setCtx(Math.min(Math.floor(n), ctxMax));
			};

			// MoE launch payload shared by Load and Save-profile. Invalid inputs
			// fall back to stock behavior (experts on GPU, stock top-k).
			var moePayload = function () {
				var n = Math.floor(Number(nCpuMoeText));
				var k = Math.floor(Number(expertUsedText));
				return {
					cpuMoe: moeMode === "cpu",
					nCpuMoe: moeMode === "layers" && Number.isFinite(n) && n > 0 ? n : 0,
					expertUsed: expertUsedText !== "" && Number.isFinite(k) && k > 0 ? k : null,
					arch: meta ? meta.arch : null,
				};
			};

			// One-click fit-to-VRAM: switch to the computed expert split.
			var applyFit = function () {
				if (!estimate || !estimate.fit || estimate.fit.n < 0) return;
				if (estimate.fit.cpuAll) {
					setMoeMode("cpu");
				} else {
					setMoeMode("layers");
					setNCpuMoeText(String(estimate.fit.n));
				}
			};

			// Clamp a ctx candidate the way the textbox does: finite, > 0,
			// capped at the model's train ctx (or the fallback ceiling).
			// Submit paths must use this — never raw ctxText.
			var clampCtx = function (n, fallback) {
				var v = Math.floor(Number(n));
				if (!Number.isFinite(v) || v <= 0) v = fallback;
				return Math.min(v, ctxMax);
			};

			// Shared skeleton for the tab's action buttons: busy spinner,
			// clear-then-set error, POST, per-action follow-up, always
			// release busy. `then` receives the post() result.
			var runAction = function (path, body, then) {
				setBusy(true);
				setError(null);
				post(path, body)
					.then(then)
					.catch(function (e2) { setError(String(e2)); })
					.finally(function () { setBusy(false); });
			};

			// A 404/405 from a /local-models route means the running dsh still
			// hosts the plugin from before the change (the node half is composed
			// at boot), so say what to do instead of showing a bare HTTP code.
			var withRestartHint = function (msg) {
				return /HTTP 40[45]/.test(msg)
					? msg + " — restart dsh web so it picks up the updated plugin, then try again"
					: msg;
			};

			var doRun = function () {
				runAction("/local-models/run", Object.assign({ path: selectedPath, ctx: clampCtx(ctxText, ctx), mtp: mtp, mmproj: mmprojPath || null, effort: effort, preserveThinking: preserveThinking, mmprojCpu: mmprojCpu, kvTypeK: kvTypeK, kvTypeV: kvTypeV, splitMode: splitMode, tensorSplit: tensorSplitText.trim() || null }, moePayload()),
					function (res) {
						if (!res.ok) setError((res.body && res.body.error) || "start failed");
						refresh();
					});
			};

			var doStop = function () {
				pendingRouterRegRef.current = false;
				runAction("/local-models/stop", {},
					function (res) { if (!res.ok) setError((res.body && res.body.error) || "stop failed"); refresh(); });
			};

			var doRegister = function () {
				runAction("/local-models/register", { route: "local-" + (status.alias || "model") },
					function (res) {
						if (res.ok) setRegistered(res.body.route + " → " + res.body.modelId);
						else setError((res.body && res.body.error) || "register failed");
						refresh();
					});
			};

			var loadProfiles = function () {
				get("/local-models/profiles").then(function (body) {
					if (body && Array.isArray(body.profiles)) setProfiles(body.profiles);
				}).catch(function () {});
			};
			useEffect(function () { loadProfiles(); }, []);

			var doSaveProfile = function () {
				var name = profileName.trim();
				if (!name || !selectedPath) return;
				runAction("/local-models/profiles", Object.assign({ name: name, modelPath: selectedPath, ctx: clampCtx(ctxText, ctx), mtpHeads: mtp, mmprojPath: mmprojPath || null, mmprojCpu: mmprojCpu, effort: effort, preserveThinking: preserveThinking, kvTypeK: kvTypeK, kvTypeV: kvTypeV, splitMode: splitMode, tensorSplit: tensorSplitText.trim() || null }, moePayload()),
					function (res) {
						if (!res.ok) setError((res.body && res.body.error) || "profile save failed");
						else { setProfileName(""); loadProfiles(); }
					});
			};

			var doLoadProfile = function (p) {
				pendingProfileRef.current = { ctx: p.ctx, expertUsed: p.expertUsed };
				// Profiles own their MoE settings — no auto-fit override.
				fitAppliedRef.current = true;
				setSelectedPath(p.modelPath || "");
				if (p.ctx) { setCtx(p.ctx); setCtxText(String(p.ctx)); }
				// Clamp: old profiles may carry MTP 4-6, unsupported since the
				// move to upstream fixed MTP (depth > 3 collapses at large ctx).
				if (p.mtpHeads != null) setMtp(Math.min(p.mtpHeads, MTP_MAX));
				// Legacy profiles carry no KV pair: fall back to the defaults.
				setKvTypeK(normalizeKvType(p.kvTypeK, KV_DEFAULT_K));
				setKvTypeV(normalizeKvType(p.kvTypeV, KV_DEFAULT_V));
				setEffort(normalizeEffort(p.effort));
				setPreserveThinking(p.preserveThinking === true || p.preserve_thinking === true);
				// Profiles without MoE fields predate expert placement (launched
				// all-GPU), so reset to GPU instead of keeping a stale selection;
				// expertUsed "" = stock top-k.
				if (p.cpuMoe === true) setMoeMode("cpu");
				else if (p.nCpuMoe > 0) { setMoeMode("layers"); setNCpuMoeText(String(p.nCpuMoe)); }
				else setMoeMode("gpu");
				if (p.expertUsed != null) setExpertUsedText(String(p.expertUsed));
				else setExpertUsedText("");
				if (p.mmprojCpu === false) setMmprojCpu(false);
				else setMmprojCpu(true);
				// Legacy profiles have no split fields: llama.cpp's own default
				// (layer, proportions from VRAM) is what they launched with.
				setSplitMode(normalizeSplitMode(p.splitMode));
				setTensorSplitText(typeof p.tensorSplit === "string" ? p.tensorSplit : "");
				if (p.mmprojPath) {
					setMmprojPath(p.mmprojPath);
					clearedMmprojRef.current = false;
				} else {
					// No mmproj in this profile: drop it AND keep the status poll
					// from resurrecting the previous server's retained value.
					setMmprojPath(null);
					clearedMmprojRef.current = true;
				}
				setError(null);
			};

			var doRemoveProfile = function (p) {
				runAction("/local-models/profiles/remove", { name: p.name },
					function (res) {
						if (!res.ok) setError((res.body && res.body.error) || "profile delete failed");
						loadProfiles();
					});
			};

			var doStartRouter = function () {
				runAction("/local-models/router/start", {},
					function (res) {
						if (!res.ok) setError((res.body && res.body.error) || "router start failed");
						else pendingRouterRegRef.current = true;
						refresh();
					});
			};

			var doUpdateSettings = function (patch) {
				runAction("/local-models/settings", patch,
					function (res) {
						if (!res.ok) {
							setError(withRestartHint((res.body && res.body.error) || "settings update failed"));
						}
						refresh();
					});
			};

			var doSetAutostart = function (enabled) {
				doUpdateSettings({ autostartRouter: enabled });
			};

			var doSetAutoUnload = function (mins) {
				doUpdateSettings({ autoUnloadMins: mins });
			};

			// ---- Runtime card (llama.cpp binaries + browser folders) ----
			// "Check" tests a candidate without saving it: it resolves the path
			// through the same chain the launcher uses and runs `<bin> --version`.
			// Returns its promise so Save can chain the verification.
			var doCheckBin = function (candidate) {
				setBinBusy(true);
				setBinCheck(null);
				return post("/local-models/runtime/check", { binPath: candidate })
					.then(function (res) {
						if (!res.ok) setBinCheck({ ok: false, error: withRestartHint((res.body && res.body.error) || "check failed") });
						else setBinCheck(res.body);
					})
					.catch(function (e2) { setBinCheck({ ok: false, error: String(e2) }); })
					.finally(function () { setBinBusy(false); });
			};

			// "Save" persists the path. The node half refuses a path that does
			// not resolve to a real llama-server, so a typo cannot be stored.
			var doSaveBin = function () {
				var next = binDraft.trim();
				setBinBusy(true);
				setError(null);
				post("/local-models/settings", { binPath: next })
					.then(function (res) {
						if (!res.ok) {
							setError(withRestartHint((res.body && res.body.error) || "could not save the binaries folder"));
							return undefined;
						}
						var saved = res.body.settings && typeof res.body.settings.binPath === "string"
							? res.body.settings.binPath
							: next;
						setBinDraft(saved);
						setBinSaved(saved);
						// Verify what was just stored (and read its --version).
						return doCheckBin(saved);
					})
					.catch(function (e2) { setError(String(e2)); })
					.finally(function () { setBinBusy(false); refresh(); });
			};

			// VRAM budget: committed on blur/Enter (no second Save button in the
			// card). "" or 0 = auto-detect through the node half's GPU probes.
			var doSaveVram = function () {
				if (!vramDirtyRef.current) return;
				var raw = vramDraft.trim().replace(",", ".");
				var gb = raw === "" ? 0 : Number(raw);
				if (!Number.isFinite(gb) || gb < 0 || gb > 1024) {
					setError("VRAM budget must be a number of GB between 0 and 1024 (empty = auto-detect)");
					return;
				}
				setError(null);
				post("/local-models/settings", { vramGb: gb })
					.then(function (res) {
						if (!res.ok) {
							setError(withRestartHint((res.body && res.body.error) || "could not save the VRAM budget"));
							return;
						}
						var savedGb = res.body.settings && res.body.settings.vramGb > 0 ? res.body.settings.vramGb : 0;
						var txt = savedGb > 0 ? String(savedGb) : "";
						setVramDraft(txt);
						setVramSaved(txt);
					})
					.catch(function (e2) { setError(String(e2)); })
					.finally(refresh);
			};

			// Current list to mutate from: the local ref when we have one, else
			// the poll snapshot (first render after a page load).
			var shortcutBase = function () {
				if (Array.isArray(scListRef.current)) return scListRef.current;
				return (Array.isArray(status.shortcuts) ? status.shortcuts : []).map(function (s) { return s.path; });
			};

			var saveShortcuts = function (list, failMsg) {
				setScBusy(true);
				setError(null);
				post("/local-models/settings", { shortcuts: list })
					.then(function (res) {
						if (!res.ok) {
							setError(withRestartHint((res.body && res.body.error) || failMsg));
							scPendingRef.current = null;
							return;
						}
						// The server echoes the stored list: adopt it instead of
						// guessing, so a rapid second edit builds on the real state.
						var saved = res.body.settings && Array.isArray(res.body.settings.shortcuts)
							? res.body.settings.shortcuts
							: list;
						scListRef.current = saved;
						scPendingRef.current = { list: saved, at: Date.now() };
					})
					.catch(function (e2) { setError(String(e2)); })
					.finally(function () { setScBusy(false); refresh(); });
			};

			var doAddShortcut = function () {
				var next = shortcutDraft.trim();
				if (!next) return;
				var list = shortcutBase();
				if (list.indexOf(next) >= 0) { setShortcutDraft(""); return; }
				setShortcutDraft("");
				saveShortcuts(list.concat([next]), "could not add the folder");
			};

			var doRemoveShortcut = function (target) {
				saveShortcuts(shortcutBase().filter(function (p) { return p !== target; }), "could not remove the folder");
			};

			var doUnloadAll = function () {
				runAction("/local-models/router/unload-all", {},
					function (res) {
						if (!res.ok) setError((res.body && res.body.error) || "unload-all failed");
						refresh();
					});
			};

			// ---- render: actions + status (Model tab only) ----
			var loadBtnStyle = canRun ? style.button : Object.assign({}, style.button, style.disabled);
			var stopBtnStyle = !stoppable ? Object.assign({}, style.secondaryButton, style.disabled) : style.secondaryButton;
			var actionsRow = h(
				"div",
				{ style: style.row },
				h("button", { style: loadBtnStyle, onClick: doRun, disabled: !canRun }, "Load model"),
				h("button", { style: stopBtnStyle, onClick: doStop, disabled: !stoppable }, "Stop"),
				// Single-model mode only: the router registers itself on start.
				running && status.mode !== "router" ? h("button", { style: style.button, onClick: doRegister, disabled: busy }, "Register in dsh") : null,
				h("span", { style: style.small }, statusText),
			);
			var statusCard = h(
				"div",
				{ style: { display: "flex", flexDirection: "column", gap: 4, borderTop: "1px solid var(--dsw-alias-border-l2)", paddingTop: 8 } },
				h("div", {}, h("span", {}, "Status: "), h("span", { style: style.mono }, statusText)),
				h(KV, { label: "PID" }, String(status.pid || "—")),
				h(KV, { label: "ctx" }, String(status.ctx != null ? fmtCtx(status.ctx) : "—")),
				h(KV, { label: "mtp" }, String(status.mtpHeads != null ? status.mtpHeads : "—")),
				status.kvTypeK && status.kvTypeV ? h(KV, { label: "KV cache" }, kvLabel(status.kvTypeK) + " K / " + kvLabel(status.kvTypeV) + " V") : null,
				status.reasoningEffort != null ? h(KV, { label: "effort" }, String(status.reasoningEffort)) : null,
				status.preserveThinking != null ? h(KV, { label: "preserve" }, status.preserveThinking === true ? "on (--reasoning-preserve)" : "off (--no-reasoning-preserve)") : null,
				status.mmprojPath ? h(KV, { label: "vision" }, status.mmprojPath.split("/").pop() + (status.mmprojCpu === false ? " (gpu)" : " (cpu)")) : null,
				status.cpuMoe === true ? h(KV, { label: "experts" }, "CPU (all in RAM)" + (status.expertUsed != null ? ", top-" + status.expertUsed : "")) : (status.nCpuMoe > 0 ? h(KV, { label: "experts" }, "cpu-first-" + status.nCpuMoe + (status.expertUsed != null ? ", top-" + status.expertUsed : "")) : (status.expertUsed != null ? h(KV, { label: "experts" }, "top-" + status.expertUsed) : null)),
				status.splitMode && (status.splitMode !== SPLIT_DEFAULT || status.tensorSplit) ? h(KV, { label: "split" }, status.splitMode + (status.tensorSplit ? " " + status.tensorSplit : "")) : null,
				status.logPath ? h("div", {}, h("span", {}, "Log: "), h("span", { style: style.mono }, status.logPath)) : null,
				h("div", { style: style.row },
					h("button", { style: style.secondaryButton, onClick: function () { setTermOpen(true); } }, "Open terminal"),
					h("span", { style: style.small }, "live tail of the llama-server output (the log above)"),
				),
			);
			// ---- render: pick + meta + launch options ----
			var pickedName = selectedPath ? selectedPath.split("/").pop() : "";
			var modelBody = h(
				"div",
				{ style: { display: "flex", flexDirection: "column", gap: 8 } },
				h(
					"div",
					{ style: style.row },
					h("button", { style: style.button, onClick: function () { setPickerOpen(true); } }, "Choose GGUF…"),
					selectedPath
						? h("span", { style: { fontSize: 13, fontWeight: 500 } }, pickedName)
						: h("span", { style: style.small }, "no model selected — pick a .gguf file"),
				),
				selectedPath ? h("div", { style: { fontSize: 12, lineHeight: "18px", color: "var(--dsw-alias-label-secondary)", fontFamily: "var(--ds-font-family-code)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }, title: selectedPath }, selectedPath) : null,
				selectedPath ? h(
					"div",
					{ style: { display: "flex", flexDirection: "column", gap: 2 } },
					metaBusy ? h("span", { style: style.small }, "parsing header…") : null,
					metaErr ? h("p", { style: style.error }, String(metaErr)) : null,
					meta ? h("span", { style: style.small }, "weights " + fmtGiB(metaRoot.weightsBytes)) : null,
					(meta && meta.warnings && meta.warnings.length > 0 ? meta.warnings.map(function (w, i) { return h("p", { key: i, style: style.warn }, "⚠ " + w); }) : null),
				) : null,
				meta ? h(
					"div",
					{ style: style.row },
					h("span", { style: style.cardTitle }, "Context"),
					h("input", { type: "range", min: 8192, max: Math.max(8192, ctxMax), step: CTX_STEP, value: ctx, style: style.range, onChange: onCtxSlider, disabled: !meta }),
					h("input", { style: style.input, value: ctxText, onChange: onCtxText, disabled: !meta, spellCheck: false }),
				) : null,
				meta ? h(FieldRow, {
					label: "KV cache",
					control: h("span", { style: style.row },
						h("span", { style: style.small }, "K"),
						h("select", { style: style.select, value: kvTypeK, title: "--cache-type-k (key store)", onChange: function (ev) { setKvTypeK(ev.target.value); }, disabled: !meta },
							KV_TYPES.map(function (t) { return h("option", { key: t.id, value: t.id }, kvOptionLabel(t)); }),
						),
						h("span", { style: style.small }, "V"),
						h("select", { style: style.select, value: kvTypeV, title: "--cache-type-v (value store)", onChange: function (ev) { setKvTypeV(ev.target.value); }, disabled: !meta },
							KV_TYPES.map(function (t) { return h("option", { key: t.id, value: t.id }, kvOptionLabel(t)); }),
						),
					),
					hint: "key / value cache quantization (--cache-type-k / --cache-type-v). This pair costs ~" + Math.round(100 * (KV_BPE[kvTypeK] + KV_BPE[kvTypeV]) / (KV_BPE[KV_DEFAULT_K] + KV_BPE[KV_DEFAULT_V])) + "% of the default " + KV_DEFAULT_K + "/" + KV_DEFAULT_V + " KV bytes; quantized V needs flash-attn (always on here) and the MTP draft KV stays q4_0.",
				}) : null,
				meta && !kvPairOk ? h("p", { style: style.warn }, "⚠ MLA model (latent KV): llama-server requires the same cache type for K and V — set both to " + kvTypeK + ".") : null,
				meta ? h(FieldRow, {
					label: "Max MTP head",
					control: h("select", { style: style.select, value: mtp, onChange: function (ev) { setMtp(Number(ev.target.value)); }, disabled: !meta },
						MTP_OPTIONS.map(function (n) { return h("option", { key: n, value: n }, n === 0 ? "off" : String(n)); }),
					),
				}) : null,
				// Rendered with >1 GPU, and also when a profile or the stored tab
				// state carries a non-default split — otherwise that value would
				// keep reaching /run with no way to see or clear it.
				splitVisible ? h(FieldRow, {
					label: "Multi-GPU split",
					control: h("span", { style: style.row },
						h("select", {
							style: style.select,
							value: splitMode,
							title: "--split-mode: how the model is spread over the " + gpuCount + " detected GPUs",
							onChange: function (ev) { setSplitMode(ev.target.value); },
							disabled: !meta,
						},
							SPLIT_MODES.map(function (m) { return h("option", { key: m.id, value: m.id }, m.label); }),
						),
						h("input", {
							style: Object.assign({}, style.input, { width: 96 }),
							value: tensorSplitText,
							placeholder: evenSplit(gpus) || "1,1",
							title: "--tensor-split: fraction of the model per GPU, e.g. 3,1",
							onChange: function (ev) { setTensorSplitText(ev.target.value); },
							disabled: !meta,
							spellCheck: false,
						}),
						h("span", { style: style.small }, "per-GPU proportions (--tensor-split)"),
					),
					hint: gpuCount > 1
						? gpuCount + " GPUs detected — layer pipelines the model across them, row/tensor parallelize each layer (tensor parallelism). Empty proportions = split in proportion to VRAM. The estimate stays a whole-machine budget: only a real per-GPU measurement can tell whether one card is over."
						: "set by the loaded profile or the stored tab state, but only " + gpuCount + " GPU was detected — llama.cpp ignores the split flags with a single device. Switch back to layer and clear the proportions to drop them.",
				}) : null,
				meta ? h(FieldRow, {
					label: "Force thinking level",
					control: h("select", { style: style.select, value: effort, onChange: function (ev) { setEffort(ev.target.value); }, disabled: !meta },
						EFFORT_LEVELS.map(function (level) { return h("option", { key: level, value: level }, level); }),
					),
				}) : null,
				meta && effort !== "off" ? h(
					"div",
					{ style: style.row },
					h("label", { style: { display: "flex", flexDirection: "row", alignItems: "center", gap: 6 } },
						h("input", { type: "checkbox", checked: preserveThinking, onChange: function (ev) { setPreserveThinking(ev.target.checked); } }),
						h("span", { style: style.small }, "preserve thinking across turns (--reasoning-preserve; off = --no-reasoning-preserve)"),
					),
				) : null,
				h(
					"div",
					{ style: style.row },
					h("button", { style: style.secondaryButton, onClick: function () { setMmPickerOpen(true); } }, "Vision mmproj (optional)"),
					mmprojPath ? h("span", { style: style.mono }, mmprojName(mmprojPath) + (mmMetaRoot ? "  (" + fmtGiB(mmMetaRoot.weightsBytes) + ")" : "")) : h("span", { style: style.small }, "none — vision models need --mmproj"),
					mmprojPath ? h("button", { style: style.secondaryButton, onClick: function () { setMmprojPath(null); setMmMetaRoot(null); setMmErr(null); clearedMmprojRef.current = true; } }, "Clear") : null,
					mmprojPath ? h("label", { style: { display: "flex", flexDirection: "row", alignItems: "center", gap: 6 } },
						h("input", { type: "checkbox", checked: mmprojCpu, onChange: function (ev) { setMmprojCpu(ev.target.checked); } }),
						h("span", { style: style.small }, "mmproj on CPU (weights in RAM, frees ~0.9 GiB VRAM; encode ~15-22 s)"),
					) : null,
					mmErr ? h("span", { style: style.error }, String(mmErr)) : null,
				),
				moeCard(meta, estimate, applyFit, moeMode, setMoeMode, nCpuMoeText, setNCpuMoeText, expertUsedText, setExpertUsedText, vramGb),
				estimate ? h(
					"div",
					{ style: { display: "flex", flexDirection: "column", gap: 4, borderTop: "1px solid var(--dsw-alias-border-l2)", paddingTop: 8 } },
					h("span", { style: style.cardTitle }, "VRAM estimate"),
					estimateBlock(),
				) : null,
				actionsRow,
				statusCard,
			);

			// ---- render: VRAM estimate (lives inside the Model tab) ----
			function estimateBlock() {
				if (!estimate) return null;
				if (estimate.unavailable) {
					return h("p", { style: style.small }, "Attention dimensions missing from the GGUF header — KV estimate unavailable.");
				}
				var layout = estimate.gdn.hybrid
					? String(estimate.gdn.attnTrunk) + " attn + " + (estimate.specOn ? estimate.gdn.mtp : 0) + " MTP · " + (estimate.gdn.trunk - estimate.gdn.attnTrunk) + " linear (GDN)"
					: (meta.nLayers != null ? String(meta.nLayers) + " attn" : "—");
				return h(
					"div",
					{ style: { display: "flex", flexDirection: "column", gap: 4, fontSize: 13, lineHeight: "20px" } },
					h(KV, { label: "Weights" + (estimate.cpuExpertBytes > 0 ? " (VRAM-resident)" : "") }, fmtGiB(estimate.weightsVram != null ? estimate.weightsVram : estimate.weights)),
					estimate.cpuExpertBytes > 0 ? h(KV, { label: "Experts in RAM" }, fmtGiB(estimate.cpuExpertBytes)) : null,
					estimate.moeUnknown ? h("p", { style: style.small }, "Expert split unknown for this layout — the estimate assumes all weights in VRAM.") : null,
					h(KV, { label: "KV cache (" + kvLabel(kvTypeK) + " K / " + kvLabel(kvTypeV) + " V, " + layout + ")" }, fmtGiB(estimate.kv)),
					estimate.recrFixed > 0 ? h(KV, { label: "GDN recurrent state" }, fmtGiB(estimate.recrFixed)) : null,
					h(KV, { label: "Compute / graph" }, fmtGiB(estimate.compute)),
					estimate.mmprojBytes > 0 ? h(KV, { label: "Vision encoder (mmproj)" }, fmtGiB(estimate.mmprojBytes)) : null,
					h(KV, { label: "Overhead" }, fmtMiB(OVERHEAD_BYTES)),
					h(KV, { strong: true, label: "Total (" + vramGb + " GB VRAM)" }, fmtGiB(estimate.total) + "  (" + estimate.pct + "%)"),
					h(KV, { label: "GPU", tone: gpuCount > 1 ? style.good : undefined }, gpuLabel),
					h(KV, { label: "Fits " + vramGb + " GB", tone: estimate.fits ? style.good : style.bad }, estimate.fits ? "yes" : "no"),
					h(KV, { label: "Fits with 700 MB margin", tone: estimate.fitsSafe ? style.good : style.bad }, estimate.fitsSafe ? "yes" : "no"),
					estimate.maxCtx != null ? h(KV, { label: "Max ctx that fits " + vramGb + " GB" }, fmtCtx(estimate.maxCtx)) : null,
				);
			}

			// ---- render: profiles ----
			var profilesBody = h(
				"div",
				{ style: { display: "flex", flexDirection: "column", gap: 8 } },
				h("p", { style: style.small }, "Save the current selection to reload it in one click (model, ctx, MTP, thinking level + preserve, mmproj)."),
				h("div", { style: style.row },
					h("input", { style: Object.assign({}, style.input, { width: 260 }), value: profileName, onChange: function (ev) { setProfileName(ev.target.value); }, placeholder: "Profile name", disabled: !selectedPath }),
					h("button", { style: style.button, onClick: doSaveProfile, disabled: !selectedPath || busy }, "Save current"),
				),
				profiles.length === 0
					? h("p", { style: style.small }, "No profiles yet — save one to reload everything in a click.")
					: h("div", { style: { display: "flex", flexDirection: "column", gap: 8 } },
						profiles.map(function (p) {
							var pSplit = normSplitText(p.tensorSplit);
							return h("div", { key: String(p.id || p.name), style: { border: "1px solid var(--dsw-alias-border-l3)", borderRadius: 10, padding: "6px 10px", display: "flex", flexDirection: "row", alignItems: "center", gap: 8 } },
								h("div", { style: { display: "flex", flexDirection: "column", gap: 0, flex: 1, minWidth: 0 } },
									h("span", { style: style.mono }, String(p.name || p.id || "")),
									h("span", { style: style.small }, String(p.modelPath || "").split("/").pop() + " · ctx " + fmtCtx(p.ctx || 0) + " · " + (p.effort || "medium") + ((p.preserveThinking === true || p.preserve_thinking === true) && p.effort !== "off" ? " · preserve" : "") + (p.mtpHeads > 0 ? " · mtp " + p.mtpHeads : "") + " · KV " + normalizeKvType(p.kvTypeK, KV_DEFAULT_K) + "/" + normalizeKvType(p.kvTypeV, KV_DEFAULT_V) + (p.mmprojPath ? " · vision(" + (p.mmprojCpu === false ? "gpu" : "cpu") + ")" : "") + (p.cpuMoe === true ? " · experts CPU" : (p.nCpuMoe > 0 ? " · experts cpu×" + p.nCpuMoe : "")) + (p.expertUsed > 0 ? " · top-" + p.expertUsed : "") + (normalizeSplitMode(p.splitMode) !== SPLIT_DEFAULT || pSplit ? " · split " + normalizeSplitMode(p.splitMode) + (pSplit ? " " + pSplit : "") : "")),
								),
								h("button", { style: style.secondaryButton, onClick: function () { doLoadProfile(p); } }, "Load"),
								h("button", { style: Object.assign({}, style.secondaryButton, { color: "var(--dsw-alias-state-error-primary)" }), onClick: function () { doRemoveProfile(p); } }, "Delete"),
							);
						}),
					),
			);

			// ---- render: runtime (llama.cpp binaries + browser folders) ----
			// The binary is resolved server-side on every launch; this card only
			// reports what that chain picks and lets the user repoint it.
			var binInfo = status.bin || null;
			var binResolved = binInfo && binInfo.resolved ? binInfo.resolved : null;
			var binOk = Boolean(binResolved) && binInfo.executable === true;
			var binLine = !binInfo
				? "resolving…"
				: binResolved
					? binResolved + "  ·  " + (binInfo.sourceLabel || binInfo.source) + (status.binVersion ? "  ·  " + status.binVersion : "") + (binOk ? "" : "  ·  not executable — chmod +x it")
					: "none found — set the folder above, or install llama-server on PATH";
			var binTone = !binInfo
				? style.mono
				: binOk ? Object.assign({}, style.mono, style.good) : Object.assign({}, style.mono, style.bad);
			var binDirtyNow = binDraft.trim() !== binSaved;
			var binCheckBlock = binCheck
				? (function () {
					var cb = binCheck.bin || null;
					var cv = binCheck.version || null;
					if (binCheck.ok && cb && cb.resolved) {
						return h("p", { style: Object.assign({}, style.small, style.good) },
							"✓ " + cb.resolved + (cv && cv.version ? " — " + cv.version : " — starts"));
					}
					if (cb && cb.resolved) {
						return h("p", { style: Object.assign({}, style.small, style.bad) },
							"✗ " + (binCheck.error || (cb.resolved + (cv && cv.error ? " — " + cv.error : (cb.executable ? "" : " — not executable (chmod +x)")))));
					}
					return h(
						"div",
						{ style: { display: "flex", flexDirection: "column", gap: 2 } },
						h("p", { style: Object.assign({}, style.small, style.bad) },
							"✗ no llama-server in " + (binDraft.trim() || "(auto-detect)") + (binCheck.error ? " — " + binCheck.error : "")),
						cb && cb.candidates && cb.candidates.length > 0
							? h("span", { style: style.small }, "looked at: " + cb.candidates.slice(0, 6).map(function (c) { return c.path; }).join(" · "))
							: null,
					);
				})()
				: null;
			var shortcuts = Array.isArray(status.shortcuts) ? status.shortcuts : [];
			// Optimistic view for the chips: the local list (refreshed from every
			// POST response) wins over the poll snapshot, which can be up to 2 s
			// behind — otherwise a removed folder stays on screen and the next
			// click lands on a stale row. `exists` still comes from the poll.
			var shortcutsView = Array.isArray(scListRef.current)
				? scListRef.current.map(function (path) {
					var hit = shortcuts.filter(function (x) { return x.path === path; })[0];
					return { label: (hit && hit.label) || path.split("/").pop() || path, path: path, exists: hit ? hit.exists : undefined };
				})
				: shortcuts;
			var chipStyle = { display: "flex", alignItems: "center", gap: 6, border: "1px solid var(--dsw-alias-border-l2)", borderRadius: 14, padding: "2px 6px 2px 10px" };
			var chipXStyle = { border: "none", background: "transparent", color: "inherit", font: "inherit", fontSize: 14, lineHeight: "18px", cursor: "pointer", padding: "0 2px" };
			var runtimeBody = h(
				"div",
				{ style: { display: "flex", flexDirection: "column", gap: 8 } },
				// Label on its own line, controls below: the label + a path input
				// + three buttons do not fit one row in this card.
				h(
					"div",
					{ style: { display: "flex", flexDirection: "column", gap: 4 } },
					h("span", { style: style.cardTitle }, "llama.cpp binaries"),
					h("div",
						{ style: style.row },
						h("input", {
							style: Object.assign({}, style.input, { flex: "1 1 240px", width: "auto", minWidth: 160 }),
							value: binDraft,
							placeholder: "auto-detect (PATH, ~/llama.cpp/build/bin…)",
							onChange: function (ev) { setBinDraft(ev.target.value); setBinCheck(null); },
							spellCheck: false,
						}),
						h("button", { style: style.secondaryButton, onClick: function () { setBinPickerOpen(true); } }, "Browse…"),
						h("button", { style: binBusy ? Object.assign({}, style.secondaryButton, style.disabled) : style.secondaryButton, disabled: binBusy, onClick: function () { doCheckBin(binDraft.trim()); } }, binBusy ? "Checking…" : "Check"),
						h("button", { style: !binDirtyNow || binBusy ? Object.assign({}, style.button, style.disabled) : style.button, disabled: !binDirtyNow || binBusy, onClick: doSaveBin }, "Save"),
					),
					h("span", { style: style.small }, "the folder holding llama-server (or the binary itself). Empty = auto-detect: LOCAL_MODELS_BIN, then PATH, then the usual build dirs. Applied on the next start — no dsh restart."),
				),
				h(
					"div",
					{ style: { display: "flex", flexDirection: "column", gap: 2 } },
					h("span", { style: style.cardTitle }, "in use"),
					h("span", { style: Object.assign({}, binTone, { wordBreak: "break-all" }) }, binLine),
					// A saved path that did not win the chain means the setting is
					// stale (bad path, or a llama.cpp tree that moved): say so
					// instead of quietly launching a different binary.
					binInfo && binInfo.configured && binInfo.source !== "setting"
						? h("p", { style: style.warn }, "the saved path (" + binInfo.configured + ") did not resolve — this fallback is what will launch. Fix it above and Save.")
						: null,
					// No `bin` in the poll answer = the running dsh still hosts the
					// node half from before this card existed (it loads at boot).
					!binInfo ? h("span", { style: style.small }, "if this never fills in, restart dsh web (the node half of the plugin loads at boot) and reload this page.") : null,
				),
				binCheckBlock,
				binDirtyNow ? h("p", { style: style.warn }, "unsaved change — press Save to persist it (Check only tests it).") : null,
				h(FieldRow, {
					label: "Model folders",
					control: h("span", { style: style.row },
						h("input", {
							style: Object.assign({}, style.input, { width: 260 }),
							value: shortcutDraft,
							placeholder: "/mnt/models",
							onChange: function (ev) { setShortcutDraft(ev.target.value); },
							onKeyDown: function (ev) { if (ev.key === "Enter") doAddShortcut(); },
							spellCheck: false,
						}),
						h("button", { style: !shortcutDraft.trim() || scBusy ? Object.assign({}, style.secondaryButton, style.disabled) : style.secondaryButton, disabled: !shortcutDraft.trim() || scBusy, onClick: doAddShortcut }, "Add folder"),
					),
					hint: "shortcut buttons in the file picker. " + (status.shortcutsSource === "env" ? "Currently from LOCAL_MODELS_SHORTCUTS (add one here to take over)." : "Home is always reachable."),
				}),
				shortcutsView.length === 0
					? h("span", { style: style.small }, "no shortcut folders")
					: h("div", { style: style.row },
						shortcutsView.map(function (s, i) {
							return h("span", { key: i, style: s.exists === false ? Object.assign({}, chipStyle, style.disabled) : chipStyle, title: s.exists === false ? s.path + " (missing)" : s.path },
								h("span", { style: style.mono }, s.label || s.path),
								s.exists === false ? h("span", { style: style.small }, "not found") : null,
								h("button", { style: chipXStyle, disabled: scBusy, title: "remove " + s.path, onClick: function () { doRemoveShortcut(s.path); } }, "×"));
						}),
					),
				h(FieldRow, {
					label: "VRAM budget",
					control: h("span", { style: style.row },
						h("input", {
							style: Object.assign({}, style.input, { width: 90 }),
							value: vramDraft,
							placeholder: fmtVramGb(vramBytes),
							title: "GiB used by the estimator; empty = auto-detect",
							onChange: function (ev) { setVramDraft(ev.target.value); },
							onBlur: doSaveVram,
							onKeyDown: function (ev) { if (ev.key === "Enter") doSaveVram(); },
							spellCheck: false,
						}),
						h("span", { style: style.small }, "GB — empty = auto (" + gpuLabel + ")"),
					),
					hint: "what the fit estimate counts as available VRAM. Auto reads nvidia-smi (NVIDIA) or sysfs (AMD) and sums every GPU; pin it when the hardware is not detected.",
				}),
			);

			// ---- render: router ----
			var routerBody = h(
				"div",
				{ style: { display: "flex", flexDirection: "column", gap: 8 } },
				h(
					"div",
					{ style: style.row },
					h("button", { style: style.button, onClick: doStartRouter, disabled: busy || status.status === "ready" }, "Start Router"),
					status.mode === "router" && running ? h("button", { style: style.secondaryButton, onClick: doStop, disabled: !stoppable }, "Kill Router") : null,
					status.mode === "router" && (status.loadedModels || []).length > 0 ? h("button", { style: style.secondaryButton, onClick: doUnloadAll, disabled: busy }, "Unload all models") : null,
				),
				h(
					"label",
					{ style: style.small },
					h("input", {
						type: "checkbox",
						checked: status.autostartRouter === true,
						disabled: busy,
						onChange: function (ev) { doSetAutostart(ev.target.checked); },
					}),
					" Start the router automatically when dsh starts (needs at least one saved profile)",
				),
				(function () {
					var mins = status.autoUnloadMins != null ? status.autoUnloadMins : 30;
					var desired = mins > 0 ? mins * 60 : null;
					var live = status.status === "ready" || status.status === "starting";
					var stale = live && status.sleepIdleSecs !== desired;
					return h(
						"div",
						{ style: { display: "flex", flexDirection: "column", gap: 4 } },
						h(
							"label",
							{ style: style.small },
							"Unload models after ",
							h("select", {
								style: style.select,
								value: mins,
								disabled: busy,
								onChange: function (ev) { doSetAutoUnload(Number(ev.target.value)); },
							},
								[0, 5, 15, 30, 60, 120].map(function (m) {
									return h("option", { key: m, value: m }, m === 0 ? "Off (never)" : m + " min idle");
								})),
							" of idleness.",
						),
						h("p", { style: style.small }, "Frees VRAM while idle; the next request reloads automatically (one slow request). Takes effect on next start."),
						stale ? h("p", { style: style.warn }, "Timer changed — restart the server to apply it now.") : null,
					);
				})(),
			);

			// ---- render: page ----
			function titledCard(title, body) {
				return h(
					"div",
					{ style: style.card },
					h("h3", { style: style.cardTitle }, title),
					body,
				);
			}
			return h(
				"div",
				{ style: style.page },
				h("h2", { style: style.title }, "Local Models"),
				h("p", { style: style.intro }, "Pick a GGUF, tune context and MTP, and load it through llama-server — it lives with this dsh process and gets killed when dsh shuts down."),
				titledCard("Runtime", runtimeBody),
				titledCard("Router", routerBody),
				titledCard("Profiles", profilesBody),
				titledCard("Model", modelBody),
				registered !== null ? h("div", {}, h("span", {}, "Registered: "), h("span", { style: style.mono }, registered)) : null,
				error !== null ? h("p", { style: style.error }, String(error)) : null,
				pickerOpen ? h(FileBrowserModal, { allFiles: false, shortcuts: status.shortcuts || [], onClose: function () { setPickerOpen(false); }, onPick: function (p) { setPickerOpen(false); setSelectedPath(p); setMetaRoot(null); fitAppliedRef.current = false; }, startDir: "" }) : null,
				mmPickerOpen ? h(FileBrowserModal, { allFiles: true, shortcuts: status.shortcuts || [], onClose: function () { setMmPickerOpen(false); }, onPick: function (p) { setMmPickerOpen(false); setMmprojPath(p); setMmMetaRoot(null); clearedMmprojRef.current = false; }, startDir: "" }) : null,
				binPickerOpen ? h(FileBrowserModal, {
					dirMode: true,
					shortcuts: status.shortcuts || [],
					onClose: function () { setBinPickerOpen(false); },
					onPick: function (p) { setBinPickerOpen(false); setBinDraft(p); setBinCheck(null); },
					// Start where the binary already is, so "use this folder" is
					// one click for the common case of a rebuild in place.
					startDir: binResolved ? binResolved.replace(/\/[^/]*$/, "") : "",
				}) : null,
				termOpen ? h(TerminalModal, { logPath: status.logPath || "", onClose: function () { setTermOpen(false); } }) : null,
			);
		}

		// ---- MoE card (render helper, not a component: needs live estimate) ----
		function mmprojName(p) { return p ? p.split("/").pop() : ""; }
		function moeCard(meta, estimate, applyFit, moeMode, setMoeMode, nCpuMoeText, setNCpuMoeText, expertUsedText, setExpertUsedText, vramGb) {
			if (!meta || !meta.isMoe) return null;
			var stockTopK = meta.expertUsedCount != null ? String(meta.expertUsedCount) : "?";
			var moeInfo = String(meta.expertCount != null ? meta.expertCount : "?") + " experts · top-" + stockTopK + " active" +
				(meta.expertSharedCount > 0 ? " (+" + meta.expertSharedCount + " shared)" : "") +
				(meta.valueExpertCount > 0 ? " · attention MoE " + meta.valueExpertCount + "/top-" + (meta.valueExpertUsedCount != null ? meta.valueExpertUsedCount : "?") : "") +
				(meta.moeSource === "tensors" ? " · detected from tensors (no metadata)" : "");
			return h(
				"div",
				{ style: { display: "flex", flexDirection: "column", gap: 8 } },
				h("span", { style: style.cardTitle }, "Mixture of Experts"),
				h("span", { style: style.mono }, moeInfo),
				estimate && estimate.fit && estimate.fit.n >= 0 && estimate.overBudget ? h(
					"div",
					{ style: style.row },
					h("span", { style: style.warn }, "Doesn't fit " + vramGb + " GB as configured — " + (estimate.fit.cpuAll ? "all experts to CPU" : "cpu-first-" + estimate.fit.n) + " fits (~" + fmtGiB(estimate.fit.total) + ")."),
					h("button", { style: style.secondaryButton, onClick: applyFit }, "Apply fit"),
				) : null,
				estimate && estimate.fit && estimate.fit.n === -1 ? h("p", { style: style.warn }, "Even all experts on CPU exceeds the " + vramGb + " GB budget — reduce context.") : null,
				h(FieldRow, {
					label: "Experts on",
					control: h("span", { style: style.row },
						h("select", { style: style.select, value: moeMode, onChange: function (ev) { setMoeMode(ev.target.value); } },
							h("option", { value: "gpu" }, "GPU (all in VRAM)"),
							h("option", { value: "cpu" }, "CPU (all in RAM)"),
							h("option", { value: "layers" }, "CPU (first N layers)"),
						),
						moeMode === "layers" ? h("input", { style: Object.assign({}, style.input, { width: 90 }), value: nCpuMoeText, onChange: function (ev) { setNCpuMoeText(ev.target.value); }, spellCheck: false }) : null,
						moeMode === "layers" ? h("span", { style: style.small }, "layers") : null,
					),
				}),
				h(FieldRow, {
					label: "Active experts (top-k)",
					control: h("input", { style: Object.assign({}, style.input, { width: 90 }), value: expertUsedText, placeholder: stockTopK, onChange: function (ev) { setExpertUsedText(ev.target.value); }, spellCheck: false }),
					hint: "empty = stock (" + stockTopK + ") — launches with --override-kv when set",
				}),
				h("span", { style: style.small }, "CPU experts live in RAM and compute on the CPU (not paged to VRAM per token): only the VRAM-resident weights count in the estimate — expect slower decode than full-VRAM loads."),
			);
		}
var inject = ["slots"];
function apply(ctx) {
	ctx.slots.inject("settings.section", function () {
		return ctx.slots.register(
			{
				name: "settings.section",
				id: "local-models",
				order: 20,
				label: function () { return "Local Models"; },
				inject: function () { return {}; },
			},
			LocalModelsSection,
		);
	});
}
exports.apply = apply;
exports.inject = inject;
return module.exports;
	},
});
