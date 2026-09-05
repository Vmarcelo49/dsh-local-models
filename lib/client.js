/**
 * dsh-local-models - browser half.
 *
 * Local Models settings tab: pick a .gguf via a file browser, choose context
 * (8K-step slider + fine input) and fixed MTP heads (0-3), see a live VRAM
 * estimate against 16 GB total, then Load / Stop / Register the server.
 * The VRAM math is ported from the user's vram-calculator project.
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
		var KV_BPE = { q8_0: 34 / 32, q4_0: 18 / 32 };
		var GRAPH_BASE = 24 * 1024 * 1024, SCRATCH_PER_TOKEN = 3072;
		var HYBRID_GRAPH_BASE = 172 * 1024 * 1024, HYBRID_SCRATCH_PER_TOKEN = 1025;
		var OVERHEAD_BYTES = 200 * 1024 * 1024;
		var TOTAL_VRAM_BYTES = 16 * 1024 * 1024 * 1024;
		var SAFE_MARGIN_BYTES = 700 * 1024 * 1024;
		var CTX_STEP = 8192, FALLBACK_MAX_CTX = 32768, VRAM_CTX_LIMIT = 131072;

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
		// Poll fn every ms (plus once on mount).
		function usePoll(fn, ms) {
			useEffect(function () {
				fn();
				var timer = setInterval(fn, ms);
				return function () { clearInterval(timer); };
			}, [fn]);
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
				return "router ready — " + String((status.loadedModels || []).length) + " model(s) loaded (" + String((status.presets) || "?") + " presets)";
			}
			if (status.status === "ready") {
				var moeBits = status.cpuMoe === true ? ", experts CPU" : (status.nCpuMoe > 0 ? ", experts cpu-first-" + status.nCpuMoe : "");
				if (status.expertUsed != null) moeBits += ", top-" + status.expertUsed;
				return "ready — " + (status.model || "model") + " (pid " + status.pid + ", ctx " + fmtCtx(status.ctx || 0) + ", mtp " + status.mtpHeads + moeBits + ")";
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

			var allFiles = props.allFiles === true;
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
					h("h3", { style: style.cardTitle }, allFiles ? "Choose a vision mmproj file" : "Choose a GGUF model"),
					h("div", { style: style.pathBar }, dir || "~ (home)"),
					h(
						"div",
						{ style: style.row },
						h("button", { style: style.secondaryButton, onClick: function () { load(""); } }, "Home"),
						(props.shortcuts || []).map(function (sc, i) {
							return h("button", { key: i, style: style.secondaryButton, onClick: function () { load(sc.path); }, title: sc.path }, sc.label);
						}),
						h("button", { style: parent ? style.secondaryButton : Object.assign({}, style.secondaryButton, style.disabled), disabled: !parent, onClick: function () { if (parent) load(parent); } }, "Up"),
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
								style: Object.assign({}, style.item, style.itemFile, sel ? style.itemSel : {}),
								onClick: function () { setSelected(en.name); },
								title: en.name,
							}, en.name + (en.size != null ? "  (" + fmtGiB(en.size) + ")" : ""));
						}),
					),
					h(
						"div",
						{ style: style.row },
						h("button", {
							style: selected ? style.button : Object.assign({}, style.button, style.disabled),
							disabled: !selected,
							onClick: function () { if (selected) props.onPick(dir + "/" + selected); },
						}, "Use this file"),
						h("button", { style: style.secondaryButton, onClick: props.onClose }, "Cancel"),
					),
				),
			);
		}
		// ---- Terminal modal: live tail of llama-server.log ----
		var TERM_MAX_CHARS = 400000; // in-memory view cap (drop head)
		var TERM_SEED_BYTES = 262144; // 256 KiB seed window / per-poll cap
		function TerminalModal(props) {
			var txtPair = useStatePair("");
			var text = txtPair[0], setText = txtPair[1];
			var errPair = useStatePair(null);
			var err = errPair[0], setErr = errPair[1];
			var bodyRef = React.useRef(null);
			var cursorRef = React.useRef(null); // server's nextOffset; null until the seed fetch lands
			var pinnedRef = React.useRef(true); // auto-follow while the user is at the bottom
			var inflightRef = React.useRef(false);

			var onScroll = function () {
				var el = bodyRef.current;
				if (!el) return;
				pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
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
							if (body.text.length > 0) {
								setText(function (prev) {
									var joined = prev.length > 0 && !prev.endsWith("\n") ? prev + "\n" + body.text : prev + body.text;
									return joined.length > TERM_MAX_CHARS ? joined.slice(-TERM_MAX_CHARS) : joined;
								});
								if (pinnedRef.current) {
									var el = bodyRef.current;
									if (el) el.scrollTop = el.scrollHeight;
								}
							}
						})
						.catch(function (e2) { if (!cancelled) setErr(String(e2)); })
						.finally(function () { inflightRef.current = false; });
				};
				setText("");
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

			var jumpToEnd = function () {
				pinnedRef.current = true;
				var el = bodyRef.current;
				if (el) el.scrollTop = el.scrollHeight;
			};

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
						h("button", { style: style.secondaryButton, onClick: jumpToEnd }, "Jump to end"),
						h("button", { style: style.secondaryButton, onClick: function () { setText(""); } }, "Clear view"),
						h("button", { style: style.secondaryButton, onClick: props.onClose }, "Close"),
					),
					err ? h("p", { style: style.error }, String(err)) : null,
					h(
						"pre",
						{ ref: bodyRef, style: style.termBody, onScroll: onScroll },
						text.length > 0 ? text : h("span", { style: style.small }, "no output yet"),
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
			var tabPair = useStatePair("model");
			var tab = tabPair[0], setTab = tabPair[1];

			var meta = metaRoot ? metaRoot.meta : null;
			var trainCtx = meta ? meta.contextLength : null;
			var ctxMax = trainCtx && trainCtx > 0 ? trainCtx : FALLBACK_MAX_CTX;

			var LS_PATH = "dsh-local-models:model-path";
			var LS_CTX = "dsh-local-models:ctx";
			var LS_MTP = "dsh-local-models:mtp";
			var LS_MMPROJ = "dsh-local-models:mmproj-path";
			var LS_MOEMODE = "dsh-local-models:moe-mode";
			var LS_NCPUMOE = "dsh-local-models:n-cpu-moe";
			var LS_TAB = "dsh-local-models:tab";
			var TABS = ["profiles", "router", "model"];

			// Restore the previous selection on mount (session survived a refresh).
			useEffect(function () {
				try {
					var p = localStorage.getItem(LS_PATH);
					if (p) { setSelectedPath(p); fitAppliedRef.current = true; }
					var mp = localStorage.getItem(LS_MMPROJ);
					if (mp) setMmprojPath(mp);
					var c = Number(localStorage.getItem(LS_CTX));
					if (Number.isFinite(c) && c > 0) { setCtx(c); setCtxText(String(c)); }
					var m = Number(localStorage.getItem(LS_MTP));
					if (Number.isFinite(m) && m >= 0 && m <= 3) setMtp(m);
					var mm = localStorage.getItem(LS_MOEMODE);
					if (mm === "cpu" || mm === "layers" || mm === "gpu") setMoeMode(mm);
					var nc = Number(localStorage.getItem(LS_NCPUMOE));
					if (Number.isFinite(nc) && nc > 0) setNCpuMoeText(String(Math.floor(nc)));
					var tb = localStorage.getItem(LS_TAB);
					if (TABS.indexOf(tb) !== -1) setTab(tb);
				} catch { /* storage unavailable */ }
				return undefined;
			}, []);

			var refresh = useCallback(function () {
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
						if (body.mtpHeads != null) setMtp(body.mtpHeads);
						if (body.cpuMoe === true) setMoeMode("cpu");
						else if (body.nCpuMoe > 0) { setMoeMode("layers"); setNCpuMoeText(String(body.nCpuMoe)); }
						if (body.expertUsed != null) setExpertUsedText(String(body.expertUsed));
					}
					if (body.mmprojPath && !mmprojPathRef.current && !clearedMmprojRef.current) setMmprojPath(body.mmprojPath);
				}).catch(function () {});
			}, []);

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

			usePoll(refresh, 2000);

			// Persist the selection so a refresh restores it.
			useEffect(function () {
				try {
					if (selectedPath) localStorage.setItem(LS_PATH, selectedPath);
					else localStorage.removeItem(LS_PATH);
					localStorage.setItem(LS_CTX, String(ctx));
					localStorage.setItem(LS_MTP, String(mtp));
					localStorage.setItem(LS_MOEMODE, moeMode);
					localStorage.setItem(LS_NCPUMOE, nCpuMoeText);
					localStorage.setItem(LS_TAB, tab);
					if (mmprojPath) localStorage.setItem(LS_MMPROJ, mmprojPath);
					else localStorage.removeItem(LS_MMPROJ);
				} catch { /* storage unavailable */ }
				return undefined;
			}, [selectedPath, ctx, mtp, mmprojPath, moeMode, nCpuMoeText, tab]);
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

			// Live VRAM estimate.
			var estimate = useMemo(function () {
				if (!meta) return null;
				if (!meta.nLayers || !meta.nKvHeads || !meta.headDim) {
					return { unavailable: true };
				}
				var gdn = gdnLayout(meta.nLayers, meta.fullAttnInterval, meta.nextnPredictLayers);
				// Mirrors the launch gate: the fixed MTP draft stays on up to
				// the ctx ceiling, and is dropped above it.
				var specOn = mtp > 0 && ctx <= VRAM_CTX_LIMIT;
				var kvLayersOverride = gdn.hybrid ? gdn.attnTrunk + (specOn ? gdn.mtp : 0) : null;
				// Upstream keeps the SSM recurrent state in f32 (the fork's
				// f16-state env is gone), so estimate with f32 state bytes.
				var recrFixed = gdn.hybrid && meta.ssmInnerSize
					? gdnRecurrentBytes(gdn.trunk - gdn.attnTrunk, meta.ssmStateSize ?? 128, meta.ssmInnerSize, meta.ssmNGroup ?? 16, meta.ssmDtRank ?? 48, meta.ssmConvKernel ?? 4, false)
					: 0;
				var kv = kvBytesFor(ctx, meta.nLayers, meta.nKvHeads, meta.headDim, "q8_0", "q4_0", meta.slidingWindow, null, kvLayersOverride);
				var compute = estimateComputeBytes(ctx, meta.nVocab, gdn.hybrid);
				var mmprojBytes = mmMetaRoot ? mmMetaRoot.weightsBytes : 0;
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
					var budget = TOTAL_VRAM_BYTES - SAFE_MARGIN_BYTES;
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
					kvTypeK: "q8_0", kvTypeV: "q4_0",
					nVocab: meta.nVocab, slidingWindow: meta.slidingWindow, fullAttnShare: null,
					kvLayersOverride: kvLayersOverride, recurrentFixedBytes: recrFixed,
					overheadBytes: OVERHEAD_BYTES,
				};
				var maxCtx = maxContextFor(TOTAL_VRAM_BYTES, input);
				return {
					gdn: gdn, specOn: specOn, kvLayersOverride: kvLayersOverride, recrFixed: recrFixed,
					kv: kv, compute: compute, total: total, weights: metaRoot.weightsBytes, weightsVram: weightsVram,
					cpuExpertBytes: cpuExpertBytes, moeUnknown: moeUnknown, mmprojBytes: mmprojBytes,
					fit: fit, overBudget: total > TOTAL_VRAM_BYTES - SAFE_MARGIN_BYTES,
					fits: total <= TOTAL_VRAM_BYTES,
					fitsSafe: total <= TOTAL_VRAM_BYTES - SAFE_MARGIN_BYTES,
					maxCtx: maxCtx,
					pct: Math.round(100 * total / TOTAL_VRAM_BYTES),
				};
			}, [meta, metaRoot, mmMetaRoot, ctx, mtp, moeMode, nCpuMoeText]);

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
			var canRun = !!meta && !!selectedPath && !acting && !running && status.mode !== "router";
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

			var doRun = function () {
				setBusy(true);
				setError(null);
				post("/local-models/run", Object.assign({ path: selectedPath, ctx: Math.floor(Number(ctxText)) || ctx, mtp: mtp, mmproj: mmprojPath || null, effort: effort, mmprojCpu: mmprojCpu }, moePayload()))
					.then(function (res) {
						if (!res.ok) setError((res.body && res.body.error) || "start failed");
						refresh();
					})
					.catch(function (e2) { setError(String(e2)); })
					.finally(function () { setBusy(false); });
			};

			var doStop = function () {
				setBusy(true);
				setError(null);
				pendingRouterRegRef.current = false;
				post("/local-models/stop", {})
					.then(function (res) { if (!res.ok) setError((res.body && res.body.error) || "stop failed"); refresh(); })
					.catch(function (e2) { setError(String(e2)); })
					.finally(function () { setBusy(false); });
			};

			var doRegister = function () {
				setBusy(true);
				setError(null);
				post("/local-models/register", { route: "local-" + (status.alias || "model") })
					.then(function (res) {
						if (res.ok) setRegistered(res.body.route + " → " + res.body.modelId);
						else setError((res.body && res.body.error) || "register failed");
						refresh();
					})
					.catch(function (e2) { setError(String(e2)); })
					.finally(function () { setBusy(false); });
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
				setBusy(true);
				setError(null);
				post("/local-models/profiles", Object.assign({ name: name, modelPath: selectedPath, ctx: Math.floor(Number(ctxText)) || ctx, mtpHeads: mtp, mmprojPath: mmprojPath || null, effort: effort }, moePayload()))
					.then(function (res) {
						if (!res.ok) setError((res.body && res.body.error) || "profile save failed");
						else { setProfileName(""); loadProfiles(); }
					})
					.catch(function (e2) { setError(String(e2)); })
					.finally(function () { setBusy(false); });
			};

			var doLoadProfile = function (p) {
				pendingProfileRef.current = { ctx: p.ctx, expertUsed: p.expertUsed };
				// Profiles own their MoE settings — no auto-fit override.
				fitAppliedRef.current = true;
				setSelectedPath(p.modelPath || "");
				if (p.ctx) { setCtx(p.ctx); setCtxText(String(p.ctx)); }
				// Clamp: old profiles may carry MTP 4-6, unsupported since the
				// move to upstream fixed MTP (depth > 3 collapses at large ctx).
				if (p.mtpHeads != null) setMtp(Math.min(p.mtpHeads, 3));
				setEffort(p.effort || "medium");
				// Legacy profiles (saved before MoE fields existed) leave the
				// current expert placement alone; expertUsed "" = stock top-k.
				if (p.cpuMoe === true) setMoeMode("cpu");
				else if (p.nCpuMoe > 0) { setMoeMode("layers"); setNCpuMoeText(String(p.nCpuMoe)); }
				else if (p.cpuMoe === false) setMoeMode("gpu");
				if (p.expertUsed != null) setExpertUsedText(String(p.expertUsed));
				else setExpertUsedText("");
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
				setBusy(true);
				setError(null);
				post("/local-models/profiles/remove", { name: p.name })
					.then(function (res) {
						if (!res.ok) setError((res.body && res.body.error) || "profile delete failed");
						loadProfiles();
					})
					.catch(function (e2) { setError(String(e2)); })
					.finally(function () { setBusy(false); });
			};

			var doStartRouter = function () {
				setBusy(true);
				setError(null);
				post("/local-models/router/start", {}).then(function (res) {
					if (!res.ok) setError((res.body && res.body.error) || "router start failed");
					else pendingRouterRegRef.current = true;
					refresh();
				}).catch(function (e2) { setError(String(e2)); }).finally(function () { setBusy(false); });
			};

			var doUnloadModel = function (model) {
				setBusy(true);
				post("/local-models/router/unload", { model: model }).then(function (res) {
					if (!res.ok) setError((res.body && res.body.error) || "unload failed");
					refresh();
				}).catch(function (e2) { setError(String(e2)); }).finally(function () { setBusy(false); });
			};

			var doUnloadAll = function () {
				setBusy(true);
				setError(null);
				post("/local-models/router/unload-all", {}).then(function (res) {
					if (!res.ok) setError((res.body && res.body.error) || "unload-all failed");
					refresh();
				}).catch(function (e2) { setError(String(e2)); }).finally(function () { setBusy(false); });
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
					label: "Max MTP head",
					control: h("select", { style: style.select, value: mtp, onChange: function (ev) { setMtp(Number(ev.target.value)); }, disabled: !meta },
						[0, 1, 2, 3].map(function (n) { return h("option", { key: n, value: n }, n === 0 ? "off" : String(n)); }),
					),
				}) : null,
				meta ? h(FieldRow, {
					label: "Thinking level",
					control: h("select", { style: style.select, value: effort, onChange: function (ev) { setEffort(ev.target.value); }, disabled: !meta },
						["off", "low", "medium", "xhigh"].map(function (level) { return h("option", { key: level, value: level }, level); }),
					),
				}) : null,
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
				moeCard(meta, estimate, applyFit, moeMode, setMoeMode, nCpuMoeText, setNCpuMoeText, expertUsedText, setExpertUsedText),
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
					h(KV, { label: "KV cache (Q8_0 K / Q4_0 V, " + layout + ")" }, fmtGiB(estimate.kv)),
					estimate.recrFixed > 0 ? h(KV, { label: "GDN recurrent state" }, fmtGiB(estimate.recrFixed)) : null,
					h(KV, { label: "Compute / graph" }, fmtGiB(estimate.compute)),
					estimate.mmprojBytes > 0 ? h(KV, { label: "Vision encoder (mmproj)" }, fmtGiB(estimate.mmprojBytes)) : null,
					h(KV, { label: "Overhead" }, fmtMiB(OVERHEAD_BYTES)),
					h(KV, { strong: true, label: "Total (16 GB VRAM)" }, fmtGiB(estimate.total) + "  (" + estimate.pct + "%)"),
					h(KV, { label: "Fits 16 GB", tone: estimate.fits ? style.good : style.bad }, estimate.fits ? "yes" : "no"),
					h(KV, { label: "Fits with 700 MB margin", tone: estimate.fitsSafe ? style.good : style.bad }, estimate.fitsSafe ? "yes" : "no"),
					estimate.maxCtx != null ? h(KV, { label: "Max ctx that fits 16 GB" }, fmtCtx(estimate.maxCtx)) : null,
					estimate.specOn ? null : h("p", { style: style.small }, "MTP draft is off above ctx " + fmtCtx(VRAM_CTX_LIMIT) + " (upstream has no draft-KV auto-quant) — estimate uses no MTP KV."),
				);
			}

			// ---- render: profiles ----
			var profilesBody = h(
				"div",
				{ style: { display: "flex", flexDirection: "column", gap: 8 } },
				h("p", { style: style.small }, "Save the current selection to reload it in one click (model, ctx, MTP, thinking level, mmproj)."),
				h("div", { style: style.row },
					h("input", { style: Object.assign({}, style.input, { width: 260 }), value: profileName, onChange: function (ev) { setProfileName(ev.target.value); }, placeholder: "Profile name", disabled: !selectedPath }),
					h("button", { style: style.button, onClick: doSaveProfile, disabled: !selectedPath || busy }, "Save current"),
				),
				profiles.length === 0
					? h("p", { style: style.small }, "No profiles yet — save one to reload everything in a click.")
					: h("div", { style: { display: "flex", flexDirection: "column", gap: 8 } },
						profiles.map(function (p) {
							return h("div", { key: String(p.id || p.name), style: { border: "1px solid var(--dsw-alias-border-l3)", borderRadius: 10, padding: "8px 10px", display: "flex", flexDirection: "column", gap: 6 } },
								h("div", { style: { display: "flex", flexDirection: "column", gap: 2 } },
									h("span", { style: style.mono }, String(p.name || p.id || "")),
									h("span", { style: style.small }, String(p.modelPath || "").split("/").pop() + " · ctx " + fmtCtx(p.ctx || 0) + " · " + (p.effort || "medium") + (p.mtpHeads > 0 ? " · mtp " + p.mtpHeads : "") + (p.mmprojPath ? " · vision" : "") + (p.cpuMoe === true ? " · experts CPU" : (p.nCpuMoe > 0 ? " · experts cpu×" + p.nCpuMoe : "")) + (p.expertUsed > 0 ? " · top-" + p.expertUsed : "")),
								),
								h("div", { style: style.row },
									h("button", { style: style.secondaryButton, onClick: function () { doLoadProfile(p); } }, "Load"),
									h("button", { style: Object.assign({}, style.secondaryButton, { color: "var(--dsw-alias-state-error-primary)" }), onClick: function () { doRemoveProfile(p); } }, "Delete"),
								),
							);
						}),
					),
			);

			// ---- render: router ----
			var routerBody = h(
				"div",
				{ style: { display: "flex", flexDirection: "column", gap: 8 } },
				h("p", { style: style.small }, "One endpoint — models from your saved profiles load on demand (llama.cpp router mode). Starting re-registers the models in dsh automatically."),
				status.mode === "router"
					? h("div", { style: { display: "flex", flexDirection: "column", gap: 8 } },
						h("span", { style: style.small }, "1 model at a time — the active model unloads before the requested one loads (models-max " + String(status.routerMax ?? 1) + ")"),
						h("div", { style: style.row },
							h("button", { style: style.secondaryButton, onClick: doUnloadAll, disabled: busy || (status.loadedModels || []).length === 0 }, "Unload all"),
							h("span", { style: style.small }, String((status.loadedModels || []).length) + " loaded"),
						),
						(status.routerModels || []).map(function (m) {
							var value = String((m && m.value) || "unknown");
							var act = value === "loaded";
							var rowStyle = act ? style.good : value === "loading" ? style.warn : style.small;
							var gone = value === "unloaded";
							return h("div", { key: String((m && m.id) || "?"), style: style.row },
								h("span", { style: style.mono }, String((m && m.id) || "?")),
								h("span", { style: rowStyle }, value),
								!gone ? h("button", { style: style.secondaryButton, onClick: function () { doUnloadModel((m && m.id) || ""); } }, "Unload") : null,
							);
						}),
					)
					: null,
				h("div", { style: style.row },
					h("button", { style: style.button, onClick: doStartRouter, disabled: busy || status.status === "ready" }, "Start router (from profiles)"),
					status.mode === "router" ? h("button", { style: style.secondaryButton, onClick: doStop, disabled: !stoppable }, "Stop") : null,
				),
			);

			// ---- render: tabs ----
			var tabDefs = [
				{ id: "profiles", label: "Profiles" },
				{ id: "router", label: "Router" },
				{ id: "model", label: "Model" },
			];
			// Stale tab (e.g. estimate from before the merge) falls back to Model.
			var effTab = tab === "profiles" || tab === "router" ? tab : "model";
			var tabBar = h(
				"div",
				{ style: style.row, role: "tablist" },
				tabDefs.map(function (t) {
					var active = effTab === t.id;
					return h("button", {
						key: t.id, role: "tab", "aria-selected": active,
						style: active ? style.button : style.secondaryButton,
						onClick: function () { setTab(t.id); },
					}, t.label);
				}),
			);
			var tabBody = h(
				"div",
				{ style: style.card },
				effTab === "profiles" ? profilesBody
					: effTab === "router" ? routerBody
					: modelBody,
			);
			return h(
				"div",
				{ style: style.page },
				h("h2", { style: style.title }, "Local Models"),
				h("p", { style: style.intro }, "Pick a GGUF, tune context and MTP, and load it through llama-server — it lives with this dsh process and gets killed when dsh shuts down."),
				tabBar,
				tabBody,
				registered !== null ? h("div", {}, h("span", {}, "Registered: "), h("span", { style: style.mono }, registered)) : null,
				error !== null ? h("p", { style: style.error }, String(error)) : null,
				pickerOpen ? h(FileBrowserModal, { allFiles: false, shortcuts: status.shortcuts || [], onClose: function () { setPickerOpen(false); }, onPick: function (p) { setPickerOpen(false); setSelectedPath(p); setMetaRoot(null); fitAppliedRef.current = false; }, startDir: "" }) : null,
				mmPickerOpen ? h(FileBrowserModal, { allFiles: true, shortcuts: status.shortcuts || [], onClose: function () { setMmPickerOpen(false); }, onPick: function (p) { setMmPickerOpen(false); setMmprojPath(p); setMmMetaRoot(null); clearedMmprojRef.current = false; }, startDir: "" }) : null,
				termOpen ? h(TerminalModal, { logPath: status.logPath || "", onClose: function () { setTermOpen(false); } }) : null,
			);
		}

		// ---- MoE card (render helper, not a component: needs live estimate) ----
		function mmprojName(p) { return p ? p.split("/").pop() : ""; }
		function moeCard(meta, estimate, applyFit, moeMode, setMoeMode, nCpuMoeText, setNCpuMoeText, expertUsedText, setExpertUsedText) {
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
					h("span", { style: style.warn }, "Doesn't fit 16 GB as configured — " + (estimate.fit.cpuAll ? "all experts to CPU" : "cpu-first-" + estimate.fit.n) + " fits (~" + fmtGiB(estimate.fit.total) + ")."),
					h("button", { style: style.secondaryButton, onClick: applyFit }, "Apply fit"),
				) : null,
				estimate && estimate.fit && estimate.fit.n === -1 ? h("p", { style: style.warn }, "Even all experts on CPU exceeds the 16 GB budget — reduce context.") : null,
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
