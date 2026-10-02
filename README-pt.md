# dsh-local-models

Um addon do `dsh` que adiciona uma aba **Local Models** à Web GUI do dsh: escolha um arquivo `.gguf`, ajuste o contexto e a decodificação especulativa, acompanhe uma estimativa de VRAM ao vivo e carregue o modelo pelo `llama-server` — depois registre o servidor em execução como um provedor de LLM no dsh com um clique.

Construído sobre o `llama.cpp` upstream sem modificações (`llama-server`). Sem fork, sem patches, sem etapa de build: o bundle do cliente é `React.createElement` escrito à mão (sem toolchain JSX) e a metade node não tem dependências.

## Features

- **Seletor de modelo** — navegador de arquivos dentro do app (apenas diretórios + `.gguf`) com leitura só do cabeçalho do GGUF (arquitetura, quant, camadas, comprimento de contexto, detecção de MoE) por trás de `POST /local-models/gguf-meta`
- **Opções de inicialização** — slider de contexto (passos de 8K, limitado ao contexto treinado do modelo) + campo de ajuste fino, seletores de quantização do KV cache (um para K, um para V — todos os tipos que o `llama-server` aceita, com bytes por elemento exibidos), profundidade fixa do draft MTP (0–7, o upstream limita à profundidade nextn do modelo), thinking level (`off`/`low`/`medium`/`xhigh`) + toggle de preservação do thinking (`--reasoning-preserve` vs `--no-reasoning-preserve`, padrão desativado), `mmproj` de visão opcional (offload para GPU ou CPU), posicionamento dos experts MoE (`--cpu-moe` / `--n-cpu-moe` / override de top-k) com um auxiliar de ajuste à VRAM
- **Estimativa de VRAM ao vivo** — pesos + os tipos de cache K/V selecionados + estado recorrente + compute/grafo + overhead contra o total detectado de GPU (nvidia-smi / sysfs do amdgpu, somados entre as GPUs, 16 GB presumidos quando desconhecido), com linhas de cabe / margem de segurança / ctx máximo que cabe (veja [Known issues](./KNOWN_ISSUES.md) para a precisão na família Gemma)
- **Profiles** — salve configurações de inicialização nomeadas e recarregue com um clique
- **Modo Router** — serve todos os profiles salvos a partir de um único endpoint compatível com OpenAI (`--models-preset`); os modelos carregam sob demanda, um residente por vez por padrão. Iniciar o router registra (ou re-registra) automaticamente seus modelos no dsh — sem precisar clicar em Register manualmente.
- **Register in dsh** — grava o servidor pronto como uma rota de provedor `llm-pi-ai` (modalidade de visão + thinking levels incluídos, saída máxima anunciada em 131K tokens para que blocos longos de thinking xhigh não sejam truncados)
- **Overlay de terminal** — tail ao vivo do log do `llama-server` direto da aba

## Requirements

- `dsh` com o profile `web` (o plugin se compõe nele)
- Um binário `llama-server` (`llama.cpp` upstream, Vulkan/CUDA/CPU — o que a sua máquina usar)
- O VRAM budget é detectado (`nvidia-smi` para NVIDIA, sysfs do amdgpu para AMD, todas as GPUs visíveis somadas) e pode ser fixado no card Runtime da aba; o fallback de 16 GB e a margem de segurança ficam no topo de `lib/client.js` (`TOTAL_VRAM_BYTES`, `SAFE_MARGIN_BYTES`)

## Install

Um plugin vive dentro de um **profile** do dsh, que é um projeto pnpm em
`$DSH_HOME/profiles/<name>`; o `dsh plugin` repassa seus argumentos para o pnpm
nesse diretório.

```bash
# from the npm registry
dsh plugin --profile web add dsh-local-models

# straight from git (plain ESM, no build step)
dsh plugin --profile web add github:Vmarcelo49/dsh-local-models

# from a local clone, for development (symlinked: edits apply on reload)
dsh plugin --profile web add link:/path/to/dsh-local-models
```

O `dsh plugin add` grava a dependência **e** acrescenta o pacote a
`dsh.profile.bundles` em `$DSH_HOME/profiles/web/package.json` — é esse array
que o monta, então não há nada para editar à mão. Reinicie o processo web do
dsh (a composição do bundle acontece no boot), atualize o navegador e abra
Settings → **Local Models**.

Verifique a composição sem inicializar e remova-o de novo com:

```bash
dsh --profile web --dump-config | grep -A 2 dsh-local-models
dsh plugin --profile web remove dsh-local-models
```

- **O pnpm precisa estar no `PATH`.** npm ou bun conseguem instalar a própria
  CLI do `dsh`, mas o gerenciamento de plugins dentro de um profile é do pnpm
  (o `dsh plugin` invoca o pnpm por baixo e imprime `pnpm was not found` caso contrário).
- **Sem gate de versão, sem exceção.** O pacote não declara nenhuma peer
  dependency `@deepseek-ai/*` — ele usa apenas serviços injetados (`settings`,
  `credentials`, `webServer`) e slots de cliente — então o `dsh plugin` nunca o
  recusa por incompatibilidade de dsh e nenhum `dsh plugin allow-version` é necessário.
- **Sem etapa de build.** Não existe script `prepare`, então o gate
  `allowBuilds` do pnpm que plugins hospedados no git encontram nunca é acionado.
- **Checagem de manifest.** O [`dsh-plugin-dev check`](https://www.npmjs.com/package/dsh-plugin-guide)
  (do `dsh-plugin-guide`) valida o manifest do bundle: `cordis.patch.yml`,
  o ponteiro `dsh.bundle.patch`, `engines` e a whitelist de `files`.

> Mudanças na metade node (rotas, lista de inject) exigem reiniciar o dsh; mudanças na metade cliente exigem apenas atualizar a página.

## Usage

1. **Choose GGUF…** — escolha um arquivo de modelo (atalhos Home / Models, navegação Up).
2. Ajuste **context**, **KV cache K / V**, **Max MTP head** (draft fixo, 0-7; 3 é o ponto ideal ajustado — profundidades maiores colapsam em ctx grande), **thinking level** + checkbox **preserve thinking**, e as configurações opcionais de **mmproj** e **MoE**.
3. **Load model**, acompanhe o card de status e inspecione a saída via **Open terminal**.
4. **Register in dsh** — a rota (padrão `local-<alias>`) aparece no seletor de modelos.
5. Como alternativa, salve **profiles** e use **Start router (from profiles)** para um endpoint com vários modelos.
6. Marque **"Start the router automatically when dsh starts"** (card Router) para iniciar o router no boot e registrar sua rota `local-router` assim que ele estiver saudável — os modelos continuam utilizáveis sem abrir a aba. Requer pelo menos um profile salvo; o progresso vai para `llama-server.log` (linhas `[autostart]`, visíveis via Open terminal).
7. **Despejo por ociosidade** (card Router, "Unload models after …", padrão 30 min de ociosidade) libera VRAM via `--sleep-idle-seconds` do upstream tanto em cargas avulsas quanto no router; o servidor dormindo continua respondendo a `/health` e recarrega automaticamente na próxima requisição (uma requisição lenta). `0` desativa. Passa a valer na próxima inicialização — a aba avisa quando o servidor em execução usa um timer diferente.

## Configuration

| Variável | Padrão | Significado |
|---|---|---|
| `LOCAL_MODELS_PORT` | `8080` | porta do `llama-server` |
| `LOCAL_MODELS_BIN` | — (detecção automática) | binário do servidor ou o diretório que o contém; a configuração do card Runtime tem precedência |
| `LOCAL_MODELS_SHORTCUTS` | — (nenhum) | diretórios de atalho do navegador de arquivos separados por dois-pontos (`name=path` para rótulos personalizados); a lista de pastas do card Runtime assume o controle depois de salva |
| `LOCAL_MODELS_MMPROJ_CPU` | `1` | pesos do projetor de visão na RAM (`0` = offload para a GPU) |
| `LOCAL_MODELS_ROUTER_MAX` | `1` | máximo de modelos do router residentes ao mesmo tempo |
| `LOCAL_MODELS_MAX_IMAGE_BYTES` | `10485760` | limite de imagem da visão |
| `LOCAL_MODELS_IMAGE_PIXEL_BUDGET` | `4194304` | orçamento de pixels da visão |
| `DSH_HOME` | `~/.dsh` | diretório de dados (`local-models/profiles.json`, `local-models/settings.json`, `llama-server.log`) |

O VRAM budget da aba é detectado, não hardcoded: NVIDIA via `nvidia-smi`,
AMD via sysfs (`mem_info_vram_total`, com o nome do produto resolvido a partir de
`pci.ids` quando presente), todas as GPUs visíveis somadas, e `CUDA_VISIBLE_DEVICES` /
`HIP_VISIBLE_DEVICES` respeitadas. Hardware que não pode ser lido cai no valor
histórico de 16 GiB, e o campo **VRAM budget** do card Runtime fixa o número
manualmente (`settings.json` → `vramGb`, 0 = automático).

As flags de inicialização são fixas na configuração diária validada: offload total, `-b 2048 -ub 512 -t 4 -np 1`, `--flash-attn on --kv-unified`, reasoning `--reasoning auto --reasoning-format deepseek --reasoning-effort <level>` mais `--reasoning-preserve` quando o toggle de preservação (campo de profile `preserveThinking`) está ligado, senão `--no-reasoning-preserve`, MTP `--spec-type draft-mtp --spec-draft-n-max N --spec-draft-p-min 0` (sem gate — o próprio padrão do upstream; o gate de confiança só compensa em placas com banda de memória escassa, nesta placa de 16 GB ele custa ~32% de decode em n-max 3 enquanto *aumenta* a aceitação de 63.5% → 91.1%, veja [bench/mtp_tuning.md](./bench/mtp_tuning.md); a aba oferece profundidades 0-7, o upstream limita a profundidade efetiva à profundidade nextn do modelo, e o draft é incondicional em qualquer ctx — o antigo checkbox “ignore the MTP ctx softcap” não existe mais, então um draft profundo em ctx grande ainda pode estourar a memória (OOM) ou colapsar o decode), posicionamento multi-GPU `--split-mode` / `--tensor-split` quando um profile os define (padrão: o próprio layer split do llama.cpp, sem flags — o controle só aparece quando há mais de uma GPU detectada), e o par de KV cache dos seletores K/V da aba (`--cache-type-k` / `--cache-type-v`, campos de profile `kvTypeK` / `kvTypeV`). Todos os tipos que este `llama-server` aceita são oferecidos (`f32 f16 bf16 q8_0 q5_1 q5_0 q4_1 iq4_nl q4_0`, rotulados com seus bytes/elemento); o padrão `q5_0` K / `q4_1` V é o ponto ideal medido para 16 GB, e profiles legados sem esses campos iniciam exatamente com esse par. V quantizado exige flash-attn (sempre ligado aqui) e o KV do draft MTP fica fixo em `q4_0`. Modelos MLA (KV latente no estilo DeepSeek) rejeitam tipos K/V mistos no llama.cpp, então a aba avisa e mantém o Load desabilitado até que ambos coincidam, e a rota `/run` recusa esse tipo de inicialização com um erro claro. Os presets do Router carregam o mesmo par KV por profile e a mesma escolha `reasoning-preserve = 1/0`.

## HTTP API (mounted under `/local-models`)

| Rota | Significado |
|---|---|
| `GET /local-models/browse?dir=` | diretórios + arquivos `.gguf` |
| `POST /local-models/gguf-meta` | `{path}` → cabeçalho GGUF parseado (em cache) |
| `GET /local-models/status` | estado + sonda `/health` recente |
| `GET /local-models/logs?offset=&max=` | tail incremental do `llama-server.log` |
| `POST /local-models/run` | inicia o servidor |
| `POST /local-models/stop` | para o processo filho (ou libera a porta) |
| `POST /local-models/profiles` / `GET` | salva (upsert) / lista profiles |
| `POST /local-models/profiles/remove` | exclui um profile |
| `GET /local-models/settings` / `POST` | lê / atualiza as configurações do plugin (`autostartRouter`, `autoUnloadMins`, `binPath`, `shortcuts`, `vramGb`) |
| `POST /local-models/runtime/check` | `{binPath}` → resolve + `<bin> --version` (o Check do card Runtime) |
| `POST /local-models/router/start` | monta os presets a partir dos profiles + inicia o router |
| `POST /local-models/router/unload` | descarrega um modelo do router |
| `POST /local-models/router/unload-all` | descarrega todos os modelos do router |
| `POST /local-models/register` | adiciona o servidor pronto como uma rota `llm-pi-ai` |

## Project layout

```
lib/index.js    node half: process manager, GGUF parser, routes, presets
lib/client.js   browser half: settings tab (single build-free bundle)
skills/         operator skill: spawn-parity checklist, profile audits
docs/           UI mockup
```

Os helpers puros e exportados (`normalizeEffort`, `moeArgsFor`, `generateRouterPresets`, `buildProviderProfile`, store de profiles) são cobertos por `npm test` (runner embutido do node, `test/`); `node lib/index.js /path/to/model.gguf` despeja um cabeçalho parseado como autoteste.

Módulos fornecidos pelo host: `@deepseek-ai/dsh-client-runtime` e `@deepseek-ai/dsh-client-ui-settings` são injetados pelo host do dsh no momento da composição do bundle (veja a lista `dsh.client.inject` em `package.json`) e deliberadamente **não** estão em `dependencies` — eles não existem no npm e não devem ser instalados.

## Known issues

Veja [KNOWN_ISSUES.md](./KNOWN_ISSUES.md) — principalmente, a estimativa de VRAM é aproximada para layouts da família Gemma.

## License

MIT — veja [LICENSE](./LICENSE).
