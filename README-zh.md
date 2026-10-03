# dsh-local-models

一个 `dsh` 插件，为 dsh Web GUI 增加一个 **Local Models** 标签页：选择 `.gguf` 文件，调节上下文与推测解码，实时查看显存估算，并通过 `llama-server` 加载它 —— 随后一键把运行中的服务器注册为 dsh 里的 LLM 提供方。

基于上游原版 `llama.cpp`（`llama-server`）构建。没有 fork，没有补丁，也没有构建步骤：客户端 bundle 是手写的 `React.createElement`（没有 JSX 工具链），node 半边零依赖。

## Features

- **模型选择器** —— 应用内的文件浏览器（仅目录 + `.gguf`），由 `POST /local-models/gguf-meta` 提供仅解析头部的 GGUF 解析（架构、量化、层数、上下文长度、MoE 检测）
- **启动选项** —— 上下文滑块（8K 步进，上限为模型训练时的上下文）+ 微调输入框，KV cache 量化选择器（K 一个、V 一个 —— 覆盖 `llama-server` 接受的每一种类型，并显示每元素字节数），固定的 MTP 草稿深度（0–7，上游会截断到模型的 nextn 深度），thinking level（`off`/`low`/`medium`/`xhigh`）+ preserve-thinking 开关（`--reasoning-preserve` 与 `--no-reasoning-preserve`，默认关闭），可选的视觉 `mmproj`（GPU 或 CPU offload），MoE 专家放置（`--cpu-moe` / `--n-cpu-moe` / top-k 覆盖）以及 fit-to-VRAM 辅助工具
- **实时显存估算** —— 权重 + 所选的 K/V cache 类型 + 循环状态 + 计算/图 + 额外开销，与检测到的 GPU 总显存对比（nvidia-smi / amdgpu sysfs，多卡求和，未知时按 16 GB 计），并给出 fits / safe-margin / max-ctx-that-fits 三行结果（Gemma 系列的准确性见 [Known issues](./KNOWN_ISSUES.md)）
- **Profiles** —— 保存具名的启动配置，一键重新加载
- **路由模式** —— 用一个 OpenAI 兼容端点（`--models-preset`）服务所有已保存的 profile；模型按需加载，默认同一时刻只驻留一个。启动路由会自动在 dsh 中（重新）注册它的模型 —— 无需手动点 Register。
- **Register in dsh** —— 把已就绪的服务器写成一个 `llm-pi-ai` 提供方路由（包含视觉模态 + thinking levels，最大输出声明为 32K tokens（上限为窗口的一半，以便 compact 保留压力预算；较长的 xhigh thinking 块请显式提高单次请求的 maxTokens））
- **终端浮层** —— 在标签页里实时跟踪 `llama-server` 日志

## Requirements

- 带有 `web` profile 的 `dsh`（插件会组合进该 profile）
- 一个 `llama-server` 可执行文件（上游 `llama.cpp`，Vulkan/CUDA/CPU —— 取决于你的机器用哪种）
- 显存预算由检测得出（NVIDIA 走 `nvidia-smi`，AMD 走 amdgpu sysfs，所有可见 GPU 求和），也可以在标签页的 Runtime 卡片里手动固定；16 GB 回退值与安全余量位于 `lib/client.js` 顶部（`TOTAL_VRAM_BYTES`、`SAFE_MARGIN_BYTES`）

## Install

插件位于 dsh 的某个 **profile** 内，而 profile 是
`$DSH_HOME/profiles/<name>` 下的一个 pnpm 项目；`dsh plugin` 会把它的参数
转发给该目录下的 pnpm。

```bash
# from the npm registry
dsh plugin --profile web add dsh-local-models

# straight from git (plain ESM, no build step)
dsh plugin --profile web add github:Vmarcelo49/dsh-local-models

# from a local clone, for development (symlinked: edits apply on reload)
dsh plugin --profile web add link:/path/to/dsh-local-models
```

`dsh plugin add` 会写入依赖，**并且**把该包追加到
`$DSH_HOME/profiles/web/package.json` 里的 `dsh.profile.bundles` —— 正是这个
数组完成挂载，所以没有任何需要手工编辑的地方。重启 dsh web 进程（bundle
组合发生在启动时），刷新浏览器，然后打开
Settings → **Local Models**。

不启动也能检查组合结果，移除时用：

```bash
dsh --profile web --dump-config | grep -A 2 dsh-local-models
dsh plugin --profile web remove dsh-local-models
```

- **pnpm 必须在 `PATH` 上。** 安装 `dsh` CLI 本身可以用 npm 或 bun，但 profile
  内的插件管理归 pnpm 管（`dsh plugin` 会调用它，
  否则会打印 `pnpm was not found`）。
- **没有版本门槛，也不需要豁免。** 该包不声明任何 `@deepseek-ai/*` peer 依赖 ——
  它只使用注入的服务（`settings`、`credentials`、`webServer`）和客户端 slot ——
  所以 `dsh plugin` 不会因 dsh 版本不匹配而拒绝它，
  也不需要 `dsh plugin allow-version`。
- **没有构建步骤。** 该包没有 `prepare` 脚本，因此 git 托管的插件会撞上的
  pnpm `allowBuilds` 门槛永远不会触发。
- **清单校验。** [`dsh-plugin-dev check`](https://www.npmjs.com/package/dsh-plugin-guide)
  （来自 `dsh-plugin-guide`）会校验 bundle 清单：`cordis.patch.yml`、
  `dsh.bundle.patch` 指针、`engines` 以及 `files` 白名单。

> node 半边的改动（路由、inject 列表）需要重启 dsh；client 半边的改动只需刷新页面。

## Usage

1. **Choose GGUF…** —— 选择模型文件（Home / Models 快捷入口，Up 向上导航）。
2. 调整 **context**、**KV cache K / V**、**Max MTP head**（固定草稿，0-7；3 是调校好的甜点值 —— 更深的草稿在大 ctx 下会崩溃）、**thinking level** + **preserve thinking** 复选框，以及可选的 **mmproj** 和 **MoE** 设置。
3. **Load model**，观察状态卡片，通过 **Open terminal** 查看输出。
4. **Register in dsh** —— 该路由（默认 `local-<alias>`）会出现在 Models 选择器中。
5. 或者保存 **profiles**，再用 **Start router (from profiles)** 得到一个多模型端点。
6. 勾选 **“Start the router automatically when dsh starts”**（Router 卡片），即可在 dsh 启动时拉起路由，并在它健康后注册 `local-router` 路由 —— 不打开标签页也能继续使用模型。至少需要一个已保存的 profile；进度落在 `llama-server.log`（`[autostart]` 行，可通过 Open terminal 查看）。
7. **Idle eviction**（Router 卡片，“Unload models after …”，默认空闲 30 分钟）会通过上游的 `--sleep-idle-seconds` 释放显存，对单次加载和路由都生效；休眠中的服务器仍会响应 `/health`，并在下一个请求时自动重新加载（该请求会慢一次）。`0` 表示关闭。下次启动时生效 —— 若正在运行的服务器使用了不同的计时器，标签页会给出警告。

## Configuration

| 变量 | 默认值 | 含义 |
|---|---|---|
| `LOCAL_MODELS_PORT` | `8080` | `llama-server` 端口 |
| `LOCAL_MODELS_BIN` | ——（自动检测） | 服务器可执行文件或它所在的目录；Runtime 卡片里的设置优先级更高 |
| `LOCAL_MODELS_SHORTCUTS` | ——（无） | 以冒号分隔的文件浏览器快捷目录（`name=path` 可自定义标签）；Runtime 卡片的文件夹列表一旦保存即接管 |
| `LOCAL_MODELS_MMPROJ_CPU` | `1` | 视觉投影权重放在内存中（`0` = offload 到 GPU） |
| `LOCAL_MODELS_ROUTER_MAX` | `1` | 路由中同时常驻的模型数量上限 |
| `LOCAL_MODELS_MAX_IMAGE_BYTES` | `10485760` | 视觉图像防护上限 |
| `LOCAL_MODELS_IMAGE_PIXEL_BUDGET` | `4194304` | 视觉像素预算 |
| `DSH_HOME` | `~/.dsh` | 数据目录（`local-models/profiles.json`、`local-models/settings.json`、`llama-server.log`） |

标签页的显存预算靠检测而非硬编码：NVIDIA 通过 `nvidia-smi`，
AMD 通过 sysfs（`mem_info_vram_total`，存在 `pci.ids` 时会解析出产品名），
所有可见 GPU 求和，并遵循 `CUDA_VISIBLE_DEVICES` /
`HIP_VISIBLE_DEVICES`。读不到硬件时回退到历史值 16 GiB，
Runtime 卡片的 **VRAM budget** 字段可以手动固定这个数字
（`settings.json` → `vramGb`，0 = 自动）。

启动参数固定为经过验证的日常配置：全量 offload，`-b 2048 -ub 512 -t 4 -np 1`，`--flash-attn on --kv-unified`，推理用 `--reasoning auto --reasoning-format deepseek --reasoning-effort <level>`，并在 preserve 开关（profile 字段 `preserveThinking`）打开时追加 `--reasoning-preserve`，否则用 `--no-reasoning-preserve`；MTP 用 `--spec-type draft-mtp --spec-draft-n-max N --spec-draft-p-min 0`（不加门控 —— 这也是上游自己的默认值；置信度门控只在带宽吃紧的卡上才划算，而在这块 16 GB 卡上，n-max 为 3 时会损失约 32% 的解码速度，却把接受率从 63.5% *提高到* 91.1%，见 [bench/mtp_tuning.md](./bench/mtp_tuning.md)；标签页提供 0-7 的深度，上游会把有效深度截断到模型的 nextn 深度，而且草稿在任何 ctx 下都是无条件的 —— 旧的“忽略 MTP ctx 软上限”复选框已经移除，所以在较大 ctx 下用很深的草稿仍可能 OOM 或让解码崩溃），profile 设置了多 GPU 放置时用 `--split-mode` / `--tensor-split`（默认：llama.cpp 自己的层划分，不带任何参数 —— 只有检测到多于一块 GPU 时才会出现该控件），以及标签页 K/V 选择器给出的 KV cache 组合（`--cache-type-k` / `--cache-type-v`，profile 字段 `kvTypeK` / `kvTypeV`）。该 `llama-server` 接受的每一种类型都会列出（`f32 f16 bf16 q8_0 q5_1 q5_0 q4_1 iq4_nl q4_0`，并标注它的字节/元素）；默认的 `q5_0` K / `q4_1` V 是实测的 16 GB 甜点，缺少这些字段的旧 profile 也会以完全相同的组合启动。量化 V 需要 flash-attn（这里始终开启），而 MTP 草稿的 KV 固定为 `q4_0`。MLA 模型（DeepSeek 风格的 latent KV）在 llama.cpp 中不接受混合的 K/V 类型，因此标签页会警告，并在两者一致前保持 Load 不可用，`/run` 路由也会以明确的错误拒绝这类启动。路由 preset 会携带相同的按 profile 指定的 KV 组合和 `reasoning-preserve = 1/0` 选择。

## HTTP API (mounted under `/local-models`)

| 路由 | 含义 |
|---|---|
| `GET /local-models/browse?dir=` | 目录 + `.gguf` 文件 |
| `POST /local-models/gguf-meta` | `{path}` → 解析出的 GGUF 头部（带缓存） |
| `GET /local-models/status` | 状态 + 实时的 `/health` 探测 |
| `GET /local-models/logs?offset=&max=` | `llama-server.log` 的增量尾部 |
| `POST /local-models/run` | 启动服务器 |
| `POST /local-models/stop` | 停止子进程（或回收端口） |
| `POST /local-models/profiles` / `GET` | 保存（upsert）/ 列出 profile |
| `POST /local-models/profiles/remove` | 删除一个 profile |
| `GET /local-models/settings` / `POST` | 读取 / 更新插件设置（`autostartRouter`、`autoUnloadMins`、`binPath`、`shortcuts`、`vramGb`） |
| `POST /local-models/runtime/check` | `{binPath}` → 解析 + `<bin> --version`（Runtime 卡片的 Check） |
| `POST /local-models/router/start` | 从 profile 构建 preset 并启动路由 |
| `POST /local-models/router/unload` | 卸载一个路由模型 |
| `POST /local-models/router/unload-all` | 卸载所有路由模型 |
| `POST /local-models/register` | 把已就绪的服务器添加为 `llm-pi-ai` 路由 |

## Project layout

```
lib/index.js    node half: process manager, GGUF parser, routes, presets
lib/client.js   browser half: settings tab (single build-free bundle)
skills/         operator skill: spawn-parity checklist, profile audits
docs/           UI mockup
```

纯函数、对外导出的辅助函数（`normalizeEffort`、`moeArgsFor`、`generateRouterPresets`、`buildProviderProfile`、profiles store）由 `npm test` 覆盖（node 内置测试运行器，`test/`）；`node lib/index.js /path/to/model.gguf` 会把解析出的头部打印出来，作为自检。

宿主提供的模块：`@deepseek-ai/dsh-client-runtime` 和 `@deepseek-ai/dsh-client-ui-settings` 由 dsh 宿主在打包 bundle 时注入（见 `package.json` 里的 `dsh.client.inject` 列表），并且刻意**不**放在 `dependencies` 中 —— 它们并不存在于 npm 上，也绝不能安装。

## Known issues

见 [KNOWN_ISSUES.md](./KNOWN_ISSUES.md) —— 其中最值得注意的是，显存估算对 Gemma 系列的结构只是近似值。

## License

MIT —— 详见 [LICENSE](./LICENSE)。
