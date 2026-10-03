# dsh-local-models

Un addon de `dsh` que añade una pestaña **Local Models** a la GUI web de dsh: elige un archivo `.gguf`, ajusta el contexto y la decodificación especulativa, observa una estimación de VRAM en vivo y cárgalo con `llama-server`; después registra el servidor en ejecución como proveedor de LLM en dsh con un solo clic.

Construido sobre el `llama.cpp` upstream sin modificaciones (`llama-server`). Sin fork, sin parches, sin paso de compilación: el bundle de cliente es `React.createElement` escrito a mano (sin toolchain de JSX) y la parte de node no tiene dependencias.

## Features

- **Selector de modelo** — explorador de archivos dentro de la app (solo directorios + `.gguf`) con un parseo únicamente de la cabecera del GGUF (arquitectura, cuantización, capas, longitud de contexto, detección de MoE) detrás de `POST /local-models/gguf-meta`
- **Opciones de lanzamiento** — slider de contexto (pasos de 8K, limitado al contexto entrenado del modelo) + entrada de ajuste fino, selectores de cuantización de la caché KV (uno para K y otro para V — todos los tipos que acepta `llama-server`, con los bytes por elemento mostrados), profundidad fija del borrador MTP (0–7, upstream la limita a la profundidad nextn del modelo), nivel de razonamiento (`off`/`low`/`medium`/`xhigh`) + toggle de preservar el razonamiento (`--reasoning-preserve` vs `--no-reasoning-preserve`, desactivado por defecto), `mmproj` de visión opcional (offload a GPU o CPU), colocación de expertos MoE (`--cpu-moe` / `--n-cpu-moe` / override del top-k) con un asistente de ajuste a la VRAM
- **Estimación de VRAM en vivo** — pesos + los tipos de caché K/V seleccionados + estado recurrente + cómputo/grafo + overhead frente al total de GPU detectado (nvidia-smi / sysfs de amdgpu, sumados entre todas las GPUs, 16 GB asumidos cuando se desconoce), con filas de entra / margen de seguridad / ctx máximo que entra (ver [Problemas conocidos](./KNOWN_ISSUES.md) para la precisión en la familia Gemma)
- **Profiles** — guarda configuraciones de lanzamiento con nombre y recárgalas con un clic
- **Modo router** — sirve todos los perfiles guardados desde un único endpoint compatible con OpenAI (`--models-preset`); los modelos se cargan bajo demanda, uno residente a la vez por defecto. Arrancar el router registra (o vuelve a registrar) automáticamente sus modelos en dsh — sin pulsar Register a mano.
- **Register in dsh** — escribe el servidor listo como ruta de proveedor `llm-pi-ai` (con modalidad de visión + niveles de razonamiento, y salida máxima anunciada en 32K tokens (limitada a la mitad de la ventana para que la compactación conserve presupuesto de presión; sube maxTokens por request explícitamente para bloques largos de razonamiento xhigh))
- **Overlay de terminal** — tail en vivo del log de `llama-server` desde la pestaña

## Requirements

- `dsh` con el perfil `web` (el plugin se compone dentro de él)
- Un binario `llama-server` (`llama.cpp` upstream, Vulkan/CUDA/CPU — lo que use tu máquina)
- El presupuesto de VRAM se detecta (`nvidia-smi` para NVIDIA, sysfs de amdgpu para AMD, sumando todas las GPUs visibles) y se puede fijar en la tarjeta Runtime de la pestaña; el fallback de 16 GB y el margen de seguridad están al principio de `lib/client.js` (`TOTAL_VRAM_BYTES`, `SAFE_MARGIN_BYTES`)

## Install

Un plugin vive dentro de un **perfil** de dsh, que es un proyecto pnpm bajo
`$DSH_HOME/profiles/<name>`; `dsh plugin` reenvía sus argumentos a pnpm en
ese directorio.

```bash
# from the npm registry
dsh plugin --profile web add dsh-local-models

# straight from git (plain ESM, no build step)
dsh plugin --profile web add github:Vmarcelo49/dsh-local-models

# from a local clone, for development (symlinked: edits apply on reload)
dsh plugin --profile web add link:/path/to/dsh-local-models
```

`dsh plugin add` escribe la dependencia **y** añade el paquete a
`dsh.profile.bundles` en `$DSH_HOME/profiles/web/package.json` — ese array es lo
que lo monta, así que no hay nada que editar a mano. Reinicia el proceso web de
dsh (la composición del bundle ocurre al arrancar), recarga el navegador y abre
Settings → **Local Models**.

Comprueba la composición sin arrancar y vuelve a quitarlo con:

```bash
dsh --profile web --dump-config | grep -A 2 dsh-local-models
dsh plugin --profile web remove dsh-local-models
```

- **pnpm debe estar en el `PATH`.** npm o bun pueden instalar el propio CLI de
  `dsh`, pero la gestión de plugins dentro de un perfil es cosa de pnpm
  (`dsh plugin` lo invoca por debajo e imprime `pnpm was not found` si no está).
- **Sin puerta de versión, sin exención.** El paquete no declara dependencias
  peer de `@deepseek-ai/*` — solo usa servicios inyectados (`settings`,
  `credentials`, `webServer`) y slots de cliente —, así que `dsh plugin` nunca lo
  rechaza por un desajuste de dsh y no hace falta `dsh plugin allow-version`.
- **Sin paso de compilación.** No hay script `prepare`, así que la puerta `allowBuilds`
  de pnpm en la que caen los plugins alojados en git nunca se activa.
- **Comprobación del manifiesto.** [`dsh-plugin-dev check`](https://www.npmjs.com/package/dsh-plugin-guide)
  (de `dsh-plugin-guide`) valida el manifiesto del bundle: `cordis.patch.yml`,
  el puntero `dsh.bundle.patch`, `engines` y la lista blanca `files`.

> Los cambios en la parte de node (rutas, lista de inyecciones) necesitan reiniciar dsh; los cambios en la parte de cliente solo necesitan recargar la página.

## Usage

1. **Choose GGUF…** — elige un archivo de modelo (atajos Home / Models, navegación con Up).
2. Ajusta **context**, **KV cache K / V**, **Max MTP head** (borrador fijo, 0-7; 3 es el punto óptimo ajustado — más profundidad colapsa con ctx grandes), **thinking level** + casilla **preserve thinking**, y los ajustes opcionales de **mmproj** y **MoE**.
3. **Load model**, observa la tarjeta de estado, inspecciona la salida con **Open terminal**.
4. **Register in dsh** — la ruta (por defecto `local-<alias>`) aparece en el selector de modelos.
5. Como alternativa, guarda **perfiles** e **Start router (from profiles)** para tener un endpoint multimodelo.
6. Marca **"Start the router automatically when dsh starts"** (tarjeta Router) para lanzar el router al inicio y registrar su ruta `local-router` en cuanto pase el chequeo de salud — los modelos siguen usándose sin abrir la pestaña. Requiere al menos un perfil guardado; el progreso aparece en `llama-server.log` (líneas `[autostart]`, visibles con Open terminal).
7. **Expulsión por inactividad** (tarjeta Router, "Unload models after …", por defecto 30 min de inactividad) libera VRAM mediante el `--sleep-idle-seconds` de upstream tanto en las cargas individuales como en el router; el servidor dormido sigue respondiendo a `/health` y se recarga solo en la siguiente petición (una petición lenta). `0` la desactiva. Surte efecto en el siguiente arranque — la pestaña avisa cuando el servidor en ejecución usa un temporizador distinto.

## Configuration

| Variable | Por defecto | Significado |
|---|---|---|
| `LOCAL_MODELS_PORT` | `8080` | puerto de `llama-server` |
| `LOCAL_MODELS_BIN` | — (autodetección) | binario del servidor o el directorio que lo contiene; el ajuste de la tarjeta Runtime tiene prioridad sobre él |
| `LOCAL_MODELS_SHORTCUTS` | — (ninguno) | directorios de atajo del explorador de archivos separados por dos puntos (`name=path` para etiquetas personalizadas); la lista de carpetas de la tarjeta Runtime toma el relevo una vez guardada |
| `LOCAL_MODELS_MMPROJ_CPU` | `1` | pesos del proyector de visión en RAM (`0` = offload a la GPU) |
| `LOCAL_MODELS_ROUTER_MAX` | `1` | máximo de modelos del router residentes a la vez |
| `LOCAL_MODELS_MAX_IMAGE_BYTES` | `10485760` | límite de imágenes de visión |
| `LOCAL_MODELS_IMAGE_PIXEL_BUDGET` | `4194304` | presupuesto de píxeles de visión |
| `DSH_HOME` | `~/.dsh` | directorio de datos (`local-models/profiles.json`, `local-models/settings.json`, `llama-server.log`) |

El presupuesto de VRAM de la pestaña se detecta, no está hardcodeado: NVIDIA a
través de `nvidia-smi`, AMD a través de sysfs (`mem_info_vram_total`, con el
nombre de producto resuelto desde `pci.ids` cuando existe), todas las GPUs
visibles sumadas, y `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` respetadas.
El hardware que no se puede leer cae al histórico de 16 GiB, y el campo
**VRAM budget** de la tarjeta Runtime fija el número a mano (`settings.json` → `vramGb`, 0 = auto).

Los flags de lanzamiento están fijados a la configuración diaria validada: offload completo, `-b 2048 -ub 512 -t 4 -np 1`, `--flash-attn on --kv-unified`, razonamiento `--reasoning auto --reasoning-format deepseek --reasoning-effort <level>` más `--reasoning-preserve` cuando el toggle de preservar (campo `preserveThinking` del perfil) está activado y, si no, `--no-reasoning-preserve`, MTP `--spec-type draft-mtp --spec-draft-n-max N --spec-draft-p-min 0` (sin puerta — el propio valor por defecto de upstream; la puerta de confianza solo compensa en tarjetas con poco ancho de banda, en esta tarjeta de 16 GB cuesta ~32% de decode con n-max 3 mientras *sube* la aceptación del 63.5% → 91.1%, ver [bench/mtp_tuning.md](./bench/mtp_tuning.md); la pestaña ofrece profundidades 0-7, upstream limita la profundidad efectiva a la profundidad nextn del modelo, y el borrador es incondicional con cualquier ctx — la antigua casilla “ignore the MTP ctx softcap” ya no existe, así que un borrador profundo con un ctx grande todavía puede dar OOM o colapsar el decode), colocación multi-GPU `--split-mode` / `--tensor-split` cuando un perfil los define (por defecto: el reparto por capas del propio llama.cpp, sin flags — el control solo aparece cuando se detecta más de una GPU), y el par de caché KV de los selectores K/V de la pestaña (`--cache-type-k` / `--cache-type-v`, campos del perfil `kvTypeK` / `kvTypeV`). Se ofrecen todos los tipos que acepta este `llama-server` (`f32 f16 bf16 q8_0 q5_1 q5_0 q4_1 iq4_nl q4_0`, etiquetados con sus bytes/elemento); el `q5_0` K / `q4_1` V por defecto es el punto óptimo medido para 16 GB, y los perfiles heredados sin esos campos arrancan exactamente con ese par. La V cuantizada necesita flash-attn (aquí siempre activo) y la KV del borrador MTP se queda fijada a `q4_0`. Los modelos MLA (KV latente al estilo DeepSeek) rechazan tipos K/V mixtos en llama.cpp, así que la pestaña avisa y mantiene Load deshabilitado hasta que ambos coincidan, y la ruta `/run` rechaza ese lanzamiento con un error claro. Los presets del router llevan el mismo par KV por perfil y la misma elección `reasoning-preserve = 1/0`.

## HTTP API (mounted under `/local-models`)

| Ruta | Significado |
|---|---|
| `GET /local-models/browse?dir=` | directorios + archivos `.gguf` |
| `POST /local-models/gguf-meta` | `{path}` → cabecera GGUF parseada (en caché) |
| `GET /local-models/status` | estado + sonda `/health` fresca |
| `GET /local-models/logs?offset=&max=` | tail incremental de `llama-server.log` |
| `POST /local-models/run` | lanza el servidor |
| `POST /local-models/stop` | detiene el hijo (o libera el puerto) |
| `POST /local-models/profiles` / `GET` | guarda (upsert) / lista perfiles |
| `POST /local-models/profiles/remove` | borra un perfil |
| `GET /local-models/settings` / `POST` | lee / actualiza los ajustes del plugin (`autostartRouter`, `autoUnloadMins`, `binPath`, `shortcuts`, `vramGb`) |
| `POST /local-models/runtime/check` | `{binPath}` → resuelve + `<bin> --version` (el Check de la tarjeta Runtime) |
| `POST /local-models/router/start` | construye los presets desde los perfiles + arranca el router |
| `POST /local-models/router/unload` | descarga un modelo del router |
| `POST /local-models/router/unload-all` | descarga todos los modelos del router |
| `POST /local-models/register` | añade el servidor listo como ruta `llm-pi-ai` |

## Project layout

```
lib/index.js    node half: process manager, GGUF parser, routes, presets
lib/client.js   browser half: settings tab (single build-free bundle)
skills/         operator skill: spawn-parity checklist, profile audits
docs/           UI mockup
```

Los helpers puros y exportados (`normalizeEffort`, `moeArgsFor`, `generateRouterPresets`, `buildProviderProfile`, almacén de perfiles) están cubiertos por `npm test` (el runner integrado de node, `test/`); `node lib/index.js /path/to/model.gguf` vuelca una cabecera parseada a modo de autotest.

Módulos provistos por el host: `@deepseek-ai/dsh-client-runtime` y `@deepseek-ai/dsh-client-ui-settings` los inyecta el host de dsh en tiempo de bundle (ver la lista `dsh.client.inject` de `package.json`) y deliberadamente **no** están en `dependencies` — no existen en npm y no se deben instalar.

## Known issues

Ver [KNOWN_ISSUES.md](./KNOWN_ISSUES.md) — en especial, la estimación de VRAM es aproximada para los layouts de la familia Gemma.

## License

MIT — ver [LICENSE](./LICENSE).
