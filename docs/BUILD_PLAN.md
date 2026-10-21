# Container deployment and RTX 3060 adaptation plan

Status: Slice A diagnostic and Slice B test isolation landed 2026-10-21. Later phases have not started. The baseline is in [MEMORY.md](../MEMORY.md).

Build a useful Studio on this Ubuntu machine in stages: project management and media editing first, one measured local image workflow next, and optional remote H3 video after that. The existing ComfyUI serves another app and, by explicit user instruction, must not be changed at all. Reuse compatible model files read-only before considering downloads. Keep unsupported model routes unavailable with an explanation.

The first release runs the Studio in a container. A managed rendering variant keeps its own ComfyUI and guard in the same container as the job runner, because the existing scripts manage local processes and their lifetimes. The shared host ComfyUI is outside Studio's execution and lifecycle control. Its migration or direct use for Studio submissions is outside this plan.

## Protect the existing application

This is a hard implementation constraint, including during diagnostics, testing, deployment, upgrades, and rollback:

- Leave the shared ComfyUI checkout, virtual environment, packages, custom nodes, configuration, systemd unit, ports, models, permissions and ownership unchanged.
- Never start, stop, restart, migrate, upgrade or reconfigure that service. Do not submit prompts, upload files, clear its queue, interrupt work, or call its model-unload endpoints.
- Limit integration to read-only inventory and read-only access to selected model files. Store all Studio inputs, outputs, caches, logs and runtime state separately.
- Treat its RAM and GPU allocations as belonging to the other application. Never reclaim them or modify the other application's scheduling to make Studio fit.
- Keep app-only mode as the default. A separate renderer still shares the physical GPU and RAM, so local GPU tests require an explicitly arranged exclusive use window without modifying the existing service. An empty queue or idle GPU snapshot does not reserve the device. If such a window is unavailable, continue app-only work or use a separate worker.

The earlier conditional host-service migration option is superseded by this constraint.

## Current environment and constraints

The audit baseline is Studio `3.0.1`, commit `88fca5e`. [MEMORY.md](../MEMORY.md) records the audit and validation history. [AGENTS.md](../AGENTS.md) remains authoritative for engine behavior and repository constraints.

| Item | Observed on this machine | Consequence |
| --- | --- | --- |
| CPU and OS | Intel i3-2120, 2 cores / 4 threads; Ubuntu 24.04 x86_64 | Use existing compatible wheels, limited build concurrency, and no compilation during app startup |
| GPU | RTX 3060, 12 GB VRAM, compute capability 8.6 | Measure a discrete-GPU route; do not use Spark estimates as RTX benchmarks |
| Host RAM | About 16 GB total, 2–3 GB available during inspection | App-only work can proceed; rendering needs substantially more available RAM before testing |
| Free disk | About 139 GB | Budget image layers, caches, outputs, and temporary downloads; reuse model files |
| Docker | Local daemon reachable; Compose 5.6.0; cgroup v2 | Container packaging can proceed without installing Docker |
| NVIDIA runtime | Named `nvidia` runtime; toolkit CLI present | Passthrough verified 2026-10-21: disposable `nvidia-smi` saw the RTX 3060 |
| Host ComfyUI | `/home/mcpadmin/ComfyUI`, version 0.40.0, commit `e0a32bce84f4603f2b61b16b50f8958e68fd6171` | Preserve its checkout, virtual environment, custom nodes, and service |
| ComfyUI service | User systemd service `comfyui.service`, Python 3.12.3, PyTorch `2.14.1+cu130` | It already sees the RTX 3060; do not replace its environment |
| ComfyUI listeners | `127.0.0.1:8188` and `172.17.0.1:8188` | The host service is already reachable through a local/Docker-bridge interface |
| Other listener | container `studio-studio-1` publishes `192.168.50.22:8190` and `:8443`; the socket uid is 0 | Leave that container and both ports alone |
| Existing models | SDXL Turbo, SDXL base/refiner, FLUX 2 Klein variants, Wan variants, and others | Inventory is evidence of files, not proof that each model fits or renders correctly |

The host ComfyUI queue was empty when inspected. That observation grants no exclusive ownership of the service or GPU. Its service starts `main.py` directly, has no systemd memory limit, and no process matching this repository's memory guard was found. It is not presently a verified Studio-managed renderer.

## Delivery scope

| Capability | First app release | After local image validation | Optional later work |
| --- | --- | --- | --- |
| Projects, Slate, Library, uploads and existing-media playback | Available | Available | Preserve |
| FFmpeg export and media probing | Available with bounded resource use | Available | Preserve |
| Text-to-image and scout contact sheets | Explicitly unavailable | One validated model and measured sizes | Additional measured models |
| Existing Qwen and FLUX3-K routes | Unavailable on this RTX profile | Remain unavailable unless separately validated | Preserve original Mac/Spark behavior |
| Local H3, Ref2VA, Drama 3, Music 3 | Unavailable | Unavailable | Requires a separately supported hardware/model route |
| H3 on another Spark | Unavailable | Unavailable until worker integration | Single video jobs through existing SSH transport |
| Generative edits, motion analysis, foley and local video alternatives | Show dependency/capability reasons | No implied support from image generation | Separate scoped milestones |
| Agent CLIs and account integrations | Show unavailable when absent | No host credentials mounted by default | Explicit integration later |

Using SDXL Turbo does not reproduce Qwen's capabilities. Its model card identifies limitations in text rendering and faces. Its preferred 512x512 output is a starting point for measurement, not a substitute for the existing H3 canvas rules. [SDXL Turbo model card](https://huggingface.co/stabilityai/sdxl-turbo)

## Architecture and ComfyUI reuse

### Studio container

Use one Compose service named `studio`, with an app-only image initially. Add an optional managed CUDA image/override later; it replaces that same service rather than starting a second Studio against the same data.

The app-only image contains Python, the lightweight web dependencies, Pillow/NumPy where needed by app functionality, and FFmpeg/ffprobe. It requires no GPU reservation, PyTorch installation, or model download. The managed CUDA variant adds a separate renderer environment and the pinned ComfyUI code. Model libraries stay outside the app's environment and process.

Keep the app and managed ComfyUI in one PID namespace. All rendering still starts through `spark/comfy.sh`; its guard watches the actual ComfyUI PID. Preserve the one-at-a-time queue, GPU lock, unload behavior, and cancellation ownership. Do not split the guard and renderer into unrelated containers or rely on Docker termination as the only guard.

### Network behavior

For this Linux-only initial deployment, use `network_mode: host` and keep Studio bound to `127.0.0.1:7870`. This preserves existing loopback-peer authorization checks in `app/suite.py`, `app/agents.py`, and the updater without expanding trusted clients. Host networking shares the host network namespace and ignores published-port mappings, so the Compose service must not also declare `ports`. [Docker host networking](https://docs.docker.com/engine/network/drivers/host/)

The tradeoff is reduced network isolation and shared port allocation. Validate the listening addresses explicitly. Managed ComfyUI must use a verified unused loopback port, provisionally 8191, while the existing service retains 8188. Refuse startup if another server owns the configured port. Do not expose Studio or ComfyUI to the LAN, enable host PID sharing, or mount the Docker socket.

### Reuse policy

1. **Reuse immediately:** read-only discovery of the existing service and model catalogue; mount the existing model directory read-only when a managed renderer is introduced. Keep outputs, caches, and temporary uploads in Studio-owned storage.
2. **Preserve the existing service:** no installation, configuration or lifecycle changes, and no mutating API calls. Do not submit Studio jobs to 8188, upload files there, edit its systemd unit, update its checkout, reinstall packages, alter custom nodes, clear its queue, call its `/free`, or interrupt jobs.
3. **Default rendering path under current repository rules:** a pinned, guarded Studio renderer may reuse the existing weights. This duplicates a software runtime only when necessary; it does not duplicate weight downloads. Its launch must be exclusive with other GPU workloads and must pass the host-memory floor.
4. **Keep runtime versions independent:** the shared ComfyUI 0.40.0 differs from the repository pin `b9b290e43f1eb379946dfeb013faf9bbefa1a618`. Keep both installations' versions intact. Reject a Studio rendering configuration that points at the shared service: existing code can start, unload and restart servers, and must never apply those operations to it. Compatibility work belongs in Studio's own managed runtime, with no migration of the shared service.

For the first image benchmark, prefer the already present `checkpoints/sd_xl_turbo_1.0_fp16.safetensors` (about 6.46 GiB on disk). Verify provenance, completeness, applicable licence, and node compatibility before use. Disk size is not a peak-memory estimate. FLUX 2 Klein files already here are distinct from this repository's FLUX3-K models and must not be relabeled or routed as them.

### Data and process ownership

| Data | Planned treatment |
| --- | --- |
| Studio projects, uploads, renders and exports | Bind mount a user-owned persistent directory to `/data/studio`; set `STUDIO_HOME` explicitly |
| Existing ComfyUI models | Read-only mount at `/models/existing`; map supported folders through ComfyUI model-path configuration |
| New optional weights | Separate persistent `/models/studio`; explicit downloads only, with checksums and size checks |
| Runtime state and cache | Studio-owned writable directory; no shared host ComfyUI output or user/config directories |
| Application and dependency code | Built into versioned images; immutable during normal operation |
| Logs | Container stdout/stderr plus bounded job/renderer logs where the app needs them |

Run as the user's UID/GID, currently 1000:1000, with an init/reaping process. Do not recursively change ownership of existing media or models. Check writable destinations before starting. Do not mount the entire home directory, host Python environment, or account credential folders.

A container-private PID cannot prove that a host lock owner is stale. Reuse the machine-wide lock location only with an ownership scheme that distinguishes host/container instances; never remove another instance's lock based on an invisible or reused PID. Existing ComfyUI does not automatically honor this lock, so GPU inventory and an explicit exclusive rendering window remain necessary.

## Phase 0 Establish the integration baseline

**Outcome:** an accurate manifest of the host, existing ComfyUI, and the software we can reuse.

- Add a bounded, read-only diagnostic command, proposed as `scripts/doctor.py`, with human-readable and JSON output. Reuse it in setup and bug reports.
- Record OS/architecture, CPU instruction support, available RAM and swap, free storage, GPU UUID/VRAM/driver, Docker/Compose versions, and cgroup limits. Probe hardware with timeouts and cache immutable facts.
- Inspect existing ComfyUI through `/system_stats`, selected `/object_info` endpoints, and queue counts. Do not capture private queued prompts or full process environments.
- Record exact model file paths, sizes, checksum/provenance status, node versions, and runtime pin differences. Compute large hashes only when selecting a model, rather than hashing the whole library on each startup.
- Test GPU visibility in a disposable CUDA container with `nvidia-smi`; validate a small tensor operation later in the renderer environment. No model loading during this step. Compose supports explicit GPU device selection; prefer the RTX GPU UUID over an unstable device index. [Compose GPU support](https://docs.docker.com/compose/how-tos/gpu-support/)
- Inspect which workload owns the 8190 listener and check that the proposed Studio/managed-renderer ports are free. Report conflicts without stopping services.
- Do not reinstall the already registered NVIDIA runtime. Any needed Docker daemon changes must be prepared separately because NVIDIA runtime configuration can alter the host daemon configuration and require a restart. [NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

**Acceptance:** app deployment requirements are known; existing ComfyUI remains running and unchanged; GPU passthrough is either verified or a specific blocker is recorded. App-only work proceeds independently of GPU passthrough.

**Recorded 2026-10-21:** `scripts/doctor.py` is that diagnostic, with text and `--json` output. Tests in `tests/test_doctor.py` cover the read-only limits and do not contact the shared service. The live baseline is summarized in [MEMORY.md](../MEMORY.md). GPU passthrough succeeded in a disposable container using the local `nvidia/cuda:12.6.3-base-ubuntu24.04` image. Shared ComfyUI was still active afterward, and the watched paths were unchanged. App-only work can proceed. Local rendering stays gated: MemAvailable was about 2.5 GB, under the 8000 MB floor, and the shared service is not a Studio renderer.

## Phase 1 Make tests and background lifetimes reliable

**Outcome:** test results describe the implementation rather than the machine they happen to run on.

Primary files: `tests/conftest.py`, `tests/test_sparks.py`, `tests/test_lengths.py`, `tests/test_discover.py`, `app/server.py`, `app/discover.py`.

- Replace module-level assignments to `server.run`, `server.memory_of`, and hub methods with fixtures that restore the original state. Stop worker threads and close fake servers/tunnels during teardown.
- Move queue-worker and discovery startup into an explicit application lifecycle. Keep native startup behavior, while making imports and test collection side-effect free. Ensure lifecycle shutdown joins or signals background tasks.
- Model Mac, Spark, RTX, and no-GPU hosts through hardware fixtures. Keep tests of actual admission refusal separate from frame-count/command-construction tests.
- Keep media integration tests in an environment with FFmpeg/ffprobe. Tests specifically about missing tools should simulate absence. Use explicit skips for optional integrations rather than silently succeeding.
- Replace the multi-Spark test's tight wall-clock speed assertion with evidence of overlapping jobs and one job per worker. Keep real performance measurement outside unit tests.
- Bound tests that can hang and capture thread/process diagnostics on failure. Investigate the broad audit run's voice-stream hang rather than declaring the suite green from focused tests.

**Acceptance:** focused tests pass alone and together; collecting Spark tests does not disable the guard elsewhere; ordinary tests never inspect or mutate the user's real model store; the full non-GPU suite completes with an explicit pass/fail/skip summary.

**Recorded 2026-10-21:** Queue and discovery startup moved into `begin()` / `end()`. Importing the app during pytest collection does not start them. `tests/conftest.py` forces `STUDIO_DISCOVER_DELAY=-1`, starts the queue for the session, and stops it afterwards. Spark tests patch `server.run` and `server.memory_of` only inside a fixture and restore them; teardown checks that `memory_of` still belongs to `server`. Frame and command builds that are not admission tests pass `room=False`. The multi-Spark test checks that jobs overlap and that each worker has one. Hardware profiles `mac`, `spark`, `rtx`, and `none` set `server.SPARK` only for `spark`. RTX is not a production mode. At the end of Phase 1, production `SPARK` was still `sys.platform.startswith("linux")`. Phase 2 replaces that with `hardware.is_spark()`. The 8000 MB floor is unchanged.

The voice-stream hang was not the discover timer. On this host about 3 GB was free, under the stand-in 8 GB kill-guard, so the guard SIGKILLed the fake engine. `Cuda.make` marked the live stream done only after a successful start, and `/live` then waited. Stand-in voice and music engines now set `guard_gb` to 0, and a failed load still marks the stream done. Production `guard_gb` stays 8. Song tests report a large `meminfo` so Spark admission does not use this host's free RAM; the measurement tests still pass their own `available_mb`. Sound command tests pretend ffmpeg is present. Tests that need the real ffmpeg or ffprobe binary, or PyAV, skip.

Focused tests together (`test_isolation.py`, `test_lengths.py`, `test_sparks.py`, `test_memory_guard.py`, `test_discover.py`, `test_doctor.py`, `test_lastframe.py`): 80 passed, 1 skipped, 31.06 s. Full non-GPU suite (`pytest tests`, pytest 9.1.1, `--timeout=240`): **616 passed, 8 skipped, 0 failed**, 152.47 s. The skips are `tests/test_shot.py` (PyAV is not installed, so that module's tests were not collected), `test_mend_paints_over_dull_green_residue`, `test_frame_from_a_clip`, two motion-link tests (ffprobe and the sample clip), two music soundtrack tests, and `test_the_delivery_sound_is_resampled_only_when_asked`. Shared ComfyUI stayed active (MainPID 897018, NRestarts 0). Container `studio-studio-1` stayed healthy on `192.168.50.22:8190` and `:8443`.

## Phase 2 Introduce hardware and capability boundaries

**Outcome:** app-only Linux works without pretending to be a Mac or Spark, and unsupported work is rejected before execution.

Proposed new modules: `app/hardware.py` for hardware observations and `app/capabilities.py` for enabled routes. Keep them lightweight and importable by diagnostics and scripts without importing the web server or model frameworks.

- Separate operating system, CPU architecture, GPU family, memory topology, runtime installation, model availability, and enabled job types. Linux alone must never imply Spark.
- Define application modes: app-only, managed RTX images, existing Mac, existing Spark, and optional controller-only. A mode cannot override physical limits or turn an unmeasured model into a supported route.
- Extend `/api/state` with capabilities and machine-readable refusal reasons. Distinguish unavailable hardware, missing runtime, missing model, incompatible nodes, busy renderer, insufficient memory, and unmeasured configuration.
- Apply the same capability checks to browser actions, `/api/jobs`, Voice/Music/Sound endpoints, Nodes, agent-triggered actions, and direct render scripts. Preserve non-generative project and media operations.
- Update `app/server.py`, `app/voice.py`, `app/music.py`, `app/sound.py`, `app/discover.py`, `app/telemetry.py`, and script dispatch. Preserve explicit Mac/Spark routes and their existing model rules.
- Update `app/static/home.js`, `settings.js`, `nodes.js`, `shell.js`, `machine.js`, `looks.js`, and relevant render tabs. Remove the fallback that silently offers Qwen when `image_models` is empty. Keep existing projects viewable even when their saved model is unavailable.
- Disable model installers that would invoke the large one-shot setup from an app-only or unsupported RTX route. Provide accurate install requirements rather than an image-only fit assertion.

**Acceptance:** this machine reports RTX 3060 correctly; app-only state contains no usable local render route; bypassing a disabled button through the API still yields a clear refusal; Mac/Spark fixture tests preserve their prior valid behavior.

**Recorded 2026-10-21:** `app/hardware.py` and `app/capabilities.py` observe this host as an NVIDIA GeForce RTX 3060 (`nvidia-rtx`), not a DGX Spark. Production `SPARK` is `hardware.is_spark()`. Mode is `app-only`. `/api/state` reports engine `none`, `image_models` `[]`, `local_render` false, and lead reason `unmeasured_configuration`. Available memory was above the 8000 MB floor, so `insufficient_memory` is not cited. Generative jobs, voice takes, music, and `scripts/require-route.sh` refuse before a Spark script runs. Export and soundtrack stay allowed. A forced `spark` or `mac` mode still opens the existing routes; `STUDIO_MODE=spark` on this GPU does not. The page title is the platform label, not Mac. Home Generate, Make image, and Queue this clip are disabled with that reason; Voice and Music keep their credits and stay unavailable.

Full non-GPU suite (`pytest tests`, pytest 9.1.1, `--timeout=240`): **627 passed, 8 skipped, 0 failed**, 152.02 s. Exit 0. Log `/tmp/suite-phase2-b.log`. The 8 skips are the same ffmpeg/PyAV set as Phase 1. A live app on `127.0.0.1:7870` with `STUDIO_DISCOVER_DELAY=-1` confirmed the state payload, the API refusal, and the disabled buttons. Shared ComfyUI and the container on `192.168.50.22:8190` were not changed. Local rendering stays off.

## Phase 3 Ship the app container

**Outcome:** Studio project and media workflows run persistently at `http://127.0.0.1:7870` without model installation.

Proposed files: `Dockerfile.studio`, `.dockerignore`, `compose.yaml`, `.env.example`, `requirements-app.in`, `requirements-app.lock`, `requirements-test.lock`, `scripts/container-entrypoint.sh`, and `scripts/container-healthcheck.py`.

- Pin the base image digest and Python dependency versions for Linux x86_64. Keep the existing Apple/model dependency lock unchanged. Put test tooling and Node-based page tests in a test build stage.
- Install FFmpeg/ffprobe and required process utilities in the image; do not require their installation on the host. Keep compiler tools out of the ordinary runtime image.
- Give the app an interpreter path independent of `$PREFIX/venv`. Audit scripts such as `scout.sh` that currently use the renderer interpreter for lightweight postprocessing.
- Start the app directly in container mode, with a dedicated health endpoint that does not scan disks, query the GPU driver, fetch updates, or load models.
- Disable in-place Git updates and pip installation in the container. Expose the image version/source revision in the UI; retain the existing updater for native deployments. Build replacements and recreate the container for updates.
- Use explicit persistent paths and scoped model discovery. Do not trigger the legacy home-folder migration or scan all mounted disks on startup.
- Make managed ComfyUI's code/custom nodes immutable at runtime: stage required nodes during image construction and retain the current copy behavior only for native installs. Direct outputs, cache, input staging, and `comfy.kind`/log files into writable Studio-owned paths.
- Implement signal handling and shutdown: stop accepting jobs, cancel only owned work, unload owned models, stop the guarded child, release owned locks, and exit within a tested timeout.
- Set explicit container RAM, swap, process and CPU policies based on measured app-only usage. The CPU is shared with existing workloads; use a bounded build concurrency and avoid startup compilation. Docker memory limits constrain the container but do not replace model admission or the guard. [Docker resource constraints](https://docs.docker.com/engine/containers/resource_constraints/)

**Acceptance:** container builds without downloading weights; projects and media survive recreation; the existing ComfyUI service is unchanged; listeners remain loopback; imports, Slate edits, media playback/probing, and a small FFmpeg export work as the unprivileged user; stopping the container leaves no owned renderer or stale lock.

**Recorded 2026-10-21:** `Dockerfile.studio`, `compose.yaml`, and the app-only entrypoint ship that container. Base images are pinned by linux/amd64 digest. `requirements.lock` is unchanged. The runtime image has ffmpeg and ffprobe, runs as uid 10001, and has no gcc, git, nvcc, or pip. `GET /api/health` returns ok, version, revision, and container, and does not scan disks or load models. In-place git and pip updates are off. The container sentence is "This Studio is a container image. Rebuild the image and recreate the container to update." Discovery uses named roots only. Managed ComfyUI nodes are staged at build and are not copied at runtime. Compose publishes `127.0.0.1:7870` only. `HOST` stays unset. `STUDIO_BIND` is `0.0.0.0` inside the bridge so the published loopback port works.

The measured app-only export was about 59–69 MiB RSS, 8–9 pids, and under 35% of one core. The ceiling is headroom over that run: 1 GiB memory, 1 GiB memswap (no extra swap), 64 pids, 1 CPU. It does not replace the 8000 MB admission floor. An export finished under that ceiling. A video job returns 400 `unavailable_hardware` because this image has no `nvidia-smi`. On the host, with the RTX 3060 visible, the lead reason remains `unmeasured_configuration`.

Projects, a Slate edit, an upload, a probe, and a 1-second ffmpeg export survived recreation on volume `imagination-studio-app-data`. `docker stop -t 20` exits 143 because uvicorn re-raises SIGTERM after shutdown. The owned GPU lock was absent afterward. Shared `comfyui.service` stayed active, MainPID 897018, NRestarts 0. Container `studio-studio-1` stayed healthy on `192.168.50.22:8190` and `:8443`.

Full non-GPU suite (`pytest tests`, pytest 9.1.1, `--timeout=240`): **643 passed, 8 skipped, 0 failed**, 156.14 s. Exit 0. Log `/tmp/phase3-build/suite3.log`. The 8 skips are the same ffmpeg/PyAV set as Phase 1. In the browser, on the pre-commit image, Make image and Queue this clip were disabled with "No GPU is available for a local render on this machine." Home Generate, Voice, and Music render controls were disabled. Voice still says "Built with Fish Audio" and Music still says "MiniMax-Music3". A fresh load of Create reported no console errors.

The committed image is `imagination-studio-app:4718c9e` (`sha256:9b07ed901a8ec4c3be9b6e40a4954ad54c5ed15e399d6d5a81e6972e90c61a98`, 788 MB). Its filesystem layers match the image that ran the workflow checks (`sha256:63de09e728d224e54e87a615ee350f785b24dcea94dc8bf727e23b3eb19f20bd`, revision `658fb1a-dirty`). After the commit, only the revision label changed. `GET /api/health` on `4718c9e` returned `{"ok":true,"version":"3.0.1","revision":"4718c9e","container":true}`. `/api/state` was 200: home `/data`, engine `none`, `image_models` `[]`, `local_render` false, platform label `LINUX · NO GPU · APP-ONLY`. The container was stopped afterward. Port 7870 was free. Local rendering stays off. After that record, Compose also publishes `192.168.50.22:7870`, and `127.0.0.1:7870` remains. `HOST` stays unset. There is no login.

## Phase 4 Add RTX admission and guarded lifecycle

**Outcome:** rendering is allowed only when host memory, container memory, GPU memory, and ownership requirements all pass.

Primary files: `spark/memory_guard.py`, `spark/comfy.sh`, `spark/spark_generate.py`, `app/server.py`, `app/telemetry.py`; proposed helpers may live beside the existing guard.

- Preserve Spark measurements, floors, polling behavior, fallback guard, and governor integration. Add an RTX-specific measured route; do not rewrite Spark tables with RTX data.
- Read three independent budgets: host `MemAvailable`, effective cgroup headroom, and selected-GPU free VRAM. Do not add host RAM to VRAM. Handle cgroup v2 limits/usage/pressure and parent limits; an unknown required limit or failed probe must not be treated as unlimited capacity.
- Compute the host allowance after preserving the existing host floor. Compute container headroom after its own reserve. Admission must fit both allowances and the GPU's separate reserve. The present 2–3 GB of available host RAM is below the existing 8,000 MB minimum floor, so a managed render cannot start under current load. Free RAM through an agreed workload change; do not lower the floor or silently terminate other processes.
- Recheck admission immediately before loading, because a queued job's earlier observation may be stale. Include current external GPU allocations, and do not count another service's allocations as reclaimable.
- Add bounded cached VRAM sampling and cgroup enforcement alongside the existing host guard. Stop only the owned renderer on breach; record a reason on the job while keeping the app responsive. Test the plain-loop fallback as well as the Python guard.
- Make model-family unloading and `/free`, cancellation, queue deletion, and server shutdown conditional on renderer ownership. A server answering a configured port is not sufficient proof of ownership.
- Tighten process targeting inside the container and handle stale locks safely across PID namespaces/restarts. Keep one renderer per Studio instance and one admitted local GPU job at a time.
- In managed mode, existing ComfyUI or another GPU workload can cause a refusal. Do not use global interrupts or restart the host service to make room. A resource check cannot prevent the other app from starting work immediately afterward, so enable local GPU testing only during an explicitly arranged exclusive use window. If that cannot be established without changes to the shared service, retain app-only mode or use a separate worker.
- Add regression tests with fake endpoints and process controllers proving that start, cancel, unload, shutdown and recovery target only Studio's owned renderer. Reject the protected host endpoint for rendering and exercise the same behavior when that service occupies a different configured address. No test may issue a mutating request or service-control command to the real shared installation.

**Acceptance:** simulated low host RAM, low cgroup headroom, low VRAM, occupied port, foreign lock, and missing telemetry all refuse safely. Guard tests stop only disposable owned processes. Ordinary image submissions remain disabled until Phase 5 produces a valid measured profile; a bounded calibration command may run the selected test workflow under every existing floor and runtime guard to establish that profile.

## Phase 5 Validate one existing image model

**Outcome:** a prompt produces images and a scout sheet through the real app queue on the RTX 3060.

First candidate: the existing SDXL Turbo FP16 checkpoint. Start at 512x512, batch size one, using the model's documented low-step recipe, mapped correctly to the pinned ComfyUI sampler semantics. Confirm required loader, text-encoding, sampler, decoder and save nodes before a render. Do not assume a Diffusers guidance value is numerically identical to the corresponding ComfyUI CFG setting. [SDXL Turbo usage](https://huggingface.co/stabilityai/sdxl-turbo)

- Add a distinct model identifier and label, such as `sdxl-turbo`, without changing what `qwen`, `flux3k-fast`, or other existing IDs mean. Keep its defaults and limits in a model registry used by both API and scripts.
- Extend the existing ComfyUI workflow path with the required image workflow. All execution goes through the guarded launcher; do not add standalone model inference to the web app.
- Validate the existing checkpoint before enabling it. Prefer verified existing files over a new download. Do not overwrite or move them.
- Use PyTorch attention and FP16 as the initial compatibility candidate. Verify actual flags against the repository's pinned ComfyUI revision. Preserve Spark's existing attention recipe. Low-VRAM controls can shift pressure to host RAM and therefore require measurement here. [ComfyUI CLI source](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/cli_args.py)
- Preserve the user's prompt as written. Do not silently apply Qwen adapters, reference formats, or the FLUX3-K caption upsampler to this model. Initially reject unsupported editing, LoRA, reference, and video-chain options explicitly.
- Decouple standalone image sizes from H3 video presets. Keep image dimensions divisible by 16 and preserve the stricter existing H3 canvas/frame rules for any later video handoff. A 512-row image route must not imply a newly supported video resolution or frame count.
- Generate scouts sequentially at batch size one, retaining seed progression, filenames, job progress, and contact-sheet behavior. Keep a conservative count limit until tested.
- Record exact model/runtime revisions, GPU UUID, precision, image size, steps, sampler, seed, cold-load time, warm-render time, peak host/cgroup memory, peak VRAM, and minimum free memory. Store results in proposed `docs/benchmarks/rtx3060-images.json` with a readable report.
- First run a single minimal render under the full guard and an independent monitor. Expand only after it finishes safely: repeat cold/warm runs, a sequential scout set, cancellation, a later successful job, and container shutdown/restart. Do not intentionally OOM the real machine; exercise refusal and kill behavior with mocks or disposable small processes.

**Acceptance:** repeated completed outputs open correctly; memory stays above every floor; no unrelated workload is interrupted; output metadata identifies the real model and settings; measured limits are enforced at submission and load. If the first candidate cannot fit, stop at the working app release and record the limiting budget before considering another existing model or hardware change.

## Phase 6 Add optional remote H3 video

**Dependency:** a real capable worker, user-provided SSH access, matching Studio code, and its own verified memory guard. This phase is independent of local image benchmarking once Phases 1–3 are complete.

Primary files: `app/sparks.py`, `app/server.py`, `app/remote.py`, `app/static/sparks.js`, `docs/multi-spark.md`.

- Separate job validation from executor admission: validate prompts, paths, frames, model options and references first, select a capable executor, then use that executor's memory check. Do not require this RTX machine to pass local H3 admission for a remote clip.
- Add a controller-only role that never claims a GPU video job locally. Preserve local CPU exports and project operations. Keep the existing `sparks` request semantics and limit for backward compatibility; separately report eligible renderers so a non-rendering controller is not counted as a GPU slot.
- Preserve one job per worker, code-hash matching, each worker's guard, SSH tunnels, media-path restrictions, `LOST`, and the requirement for a person's Check after a worker goes down. Include any new shared modules/configuration in the code manifest.
- Eliminate the existing eventual local fallback for controller-only video jobs. If workers refuse or disappear, keep the job in an explicit bounded retry/wait state or fail with a clear reason; never execute H3 locally and never silently extend the lost-worker timeout.
- Retain the current scope of single `video` jobs only. Images, edits, voice, music, long chains and a single clip split across Sparks are not included.
- Mount only the selected SSH credentials/known-hosts needed for the worker, or an explicitly chosen agent socket, at implementation time. Do not copy credentials into images or the repository.

**Acceptance:** the controller can queue a valid clip it cannot render itself; the worker checks its own resources, renders and returns it; mismatched code, lost worker, cancellation and refusal paths remain bounded; the controller never launches a local video model. Without a worker, validate with fixtures and label live verification pending.

## Phase 7 Finish operations and release verification

**Outcome:** a repeatable local deployment with a tested recovery path.

- Add user instructions for build/start/stop/logs/health, locating outputs, model inventory, refusal reasons, and optional renderer enablement. Example commands must match the files actually implemented.
- Keep `.env.example` free of secrets. Define only necessary settings: data paths, mode, selected GPU, Studio port, managed renderer port, and measured resource profile.
- Restrict discovery to configured model mounts. Distinguish model files present from a model that is compatible, licensed for the intended use, benchmarked, and currently runnable.
- Prevent duplicate Studio instances against the same data and GPU lock. Check existing ComfyUI queue/ownership again before each integration test.
- Define active-job restart semantics. The current queue is in memory: graceful deployment drains/cancels jobs, and interruption must not be shown as success or automatically replay a GPU job. If retaining job history across restarts is included, persist minimal job records atomically and mark interrupted work explicitly; do not imply that mounting project storage alone persists the queue.
- Exercise backup and restore of project/configuration data. Tag images by source revision, retain the prior working image, and test rollback against a copied data directory before any schema migration.
- Bound logs and temporary outputs, remove only Studio-owned temporary files, and reserve disk for incomplete renders. Never prune shared Docker images, caches, or models automatically.
- Update `README.md`, relevant credits/notices, and `MEMORY.md`. Preserve existing attribution and model licence requirements. Record every measured profile's hardware and dependency revisions.

**Acceptance:** a clean deployment can be reproduced; restarting/recreating preserves projects and media; failed jobs remain explainable; rollback works; the host ComfyUI installation remains unchanged; all applicable non-GPU, container and measured GPU checks have recorded outcomes.

## Implementation slices and dependencies

Use small reviewable changes. The estimates below describe scope; elapsed time depends primarily on test repairs and hardware measurements, so do not promise a calendar date before Phase 1 is green.

| Slice | Depends on | Deliverable | Size |
| --- | --- | --- | --- |
| A | None | Diagnostics and existing ComfyUI inventory | Small |
| B | A | Test isolation and explicit background lifetimes | Medium |
| C | B | Hardware/capability API and app-only UI behavior | Medium |
| D | C | App runtime image, Compose, persistence, health, shutdown | Medium |
| E | B, D | RTX host/cgroup/VRAM admission and renderer ownership | Large |
| F | E and available memory | Existing SDXL Turbo workflow and benchmarked limits | Medium, measurement dependent |
| G | C, D and a capable worker | Optional controller-only single-video support | Large, worker dependent |
| H | D, plus whichever of F/G ships | Operational documentation, backup, rollback, release checks | Medium |

The first useful release is A–D plus the app-only parts of H. GPU validation must not delay a working project and media application. The next useful release adds E–F. G is optional and should not be presented as already configured.

## Verification matrix

| Area | Required evidence |
| --- | --- |
| Hardware detection | Mac, Spark, RTX, missing driver, unavailable GPU query, and restricted-container fixtures |
| Capabilities | No fallback to Qwen; direct API and script refusals match UI availability |
| Original model rules | Existing frame counts, canvas constraints, prompt compilation, adapter routes and audio behavior preserved |
| Test isolation | No import-time mocks or discovery scans; tests pass independently and together |
| Container | Non-root writes, explicit storage, startup without GPU/models, loopback-only sockets, clean shutdown |
| Memory | Host floor plus cgroup and VRAM budgets; recheck before load; bounded telemetry failures |
| Existing ComfyUI | No changes to installation/configuration/service/models/permissions; read-only access only; no submissions, uploads, interrupts, queue clears, unloads, restarts or stops |
| Images | Actual outputs, measured peaks, sequential scouts, cancellation and retry under ownership rules |
| Worker mode | Destination admission, no controller fallback, code/guard checks, lost worker and cancellation behavior |
| Operations | Project/media persistence, interrupted-job semantics, backup/restore and image rollback |

## Immediate next work

Phase 3 is done. Start Phase 4 (RTX host, cgroup, and VRAM admission, and renderer ownership). The app container is published on `127.0.0.1:7870` and `192.168.50.22:7870`. Preserve the shared ComfyUI service, its model files, and the container that publishes port 8190. Treat unmeasured hardware and the 8000 MB floor as rendering gates. Do not lower the floor. Local rendering stays off until a measured profile exists.
