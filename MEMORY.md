# Project memory

Last updated: 2026-10-21

## Objective and current direction

Adapt Wasserman Slate AI Studio to run on this machine. The user requested a code audit, then discussed containerizing the adapted application and requested this file to preserve our progress.

The proposed direction is a containerized Studio app, with rendering introduced after capability and memory checks. The user explicitly clarified that the existing ComfyUI also serves another app and **must not be changed at all**. Treat it as an independently owned shared service. Reuse only compatible model files through read-only access; Studio rendering uses its own guarded instance. The detailed implementation roadmap is [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md). Slice A, the read-only baseline, is recorded below. The app container has not been built. The existing upstream Docker image is an installation payload carrier, not a Studio runtime image.

## Hard constraint protecting the existing app

- Do not modify the existing ComfyUI checkout, virtual environment, dependencies, custom nodes, configuration, systemd unit, ports, model files, permissions, or ownership.
- Do not start, stop, restart, migrate, upgrade, or reconfigure its service. Do not submit jobs to it, upload files to it, clear its queue, interrupt work, or unload its models.
- Remove the earlier proposal for conditional host-service migration/reuse from implementation scope. Read-only inventory and read-only model-file access are the only planned integration with it.
- Keep Studio's renderer, inputs, outputs, caches, runtime state and lifecycle separate. An occupied port or insufficient resource budget must cause a refusal rather than a takeover.
- Both applications still share physical RAM and the GPU. Keep app-only mode as the default; local GPU tests require an explicitly arranged exclusive use window that does not change the existing service. An empty queue or idle GPU snapshot is not a reservation. If safe coexistence cannot be established, keep local rendering disabled and use app-only or a separate worker.

Read `AGENTS.md` before making changes. This file records working context; it does not replace those instructions.

## Target machine

Measured during the audit on 2026-10-21:

| Component | Observed |
| --- | --- |
| OS / architecture | Ubuntu 24.04.5 LTS, Linux x86_64 |
| CPU | Intel Core i3-2120, 2 cores / 4 threads |
| GPU | NVIDIA GeForce RTX 3060, 12,288 MiB VRAM, compute capability 8.6 |
| NVIDIA driver | 595.84 |
| System RAM | About 16 GB total; only 2–3 GB available during the audit |
| Swap | 4 GiB total; about 2 GiB used |
| Free disk | About 139 GB reported by `df -h` |
| Available tools | Python 3.12.3, Node 24.18.0, uv, Docker CLI, nvidia-smi |
| Missing tools | ffmpeg, ffprobe, nvcc on PATH; pytest in system Python |

Recheck available memory, disk, and tools before implementation. During planning, Docker daemon access was verified on `mcpdev02`: Compose 5.6.0, cgroup v2, NVIDIA runtime registered, and NVIDIA Container Toolkit CLI 1.20.1 installed. GPU passthrough into a new container has not yet been tested.

## Existing ComfyUI discovered during planning

- Location: `/home/mcpadmin/ComfyUI`, with its own `.venv`; active user systemd service `comfyui.service`.
- Version: ComfyUI 0.40.0, commit `e0a32bce84f4603f2b61b16b50f8958e68fd6171`, Python 3.12.3, PyTorch `2.14.1+cu130`. Its API sees the RTX 3060.
- Listeners: `127.0.0.1:8188` and `172.17.0.1:8188`. `192.168.50.22:8190` is published by container `studio-studio-1` (also `:8443`); the socket uid is 0. Leave that container and both ports alone.
- Model files already include SDXL Turbo (`checkpoints/sd_xl_turbo_1.0_fp16.safetensors`, about 6.46 GiB), SDXL base/refiner, FLUX 2 Klein variants, Wan variants, and supporting encoders/VAEs. Existing files are not proof of runtime fit. FLUX 2 Klein is not this repository's FLUX3-K.
- Custom-node directories include `studio_film` and `studio_scene`. The checkout has an untracked `start.sh`; preserve it.
- Read-only API checks showed an empty queue at inspection time. No prompts were submitted, services stopped, queues cleared, or model files changed.
- The service starts `main.py` directly with `--enable-manager`. No process matching this repository's guard was found; systemd memory limits are unlimited. Its version also differs from this repository's pinned ComfyUI revision. Directly setting `SPARK_COMFY_URL` does not resolve guard, version, or process-ownership requirements.

## Audit findings

Audited checkout: `88fca5e` (Studio version `3.0.1`). Scope focused on installation, runtime routing, memory admission, worker scheduling, and tests. This was not a complete security review or a model performance benchmark.

1. **Linux is treated as DGX Spark hardware.** `app/server.py:140` sets `SPARK` from the operating system alone. Similar assumptions occur in voice, music, discovery, telemetry, and shell scripts. The live state endpoint labeled this RTX 3060 machine as a DGX Spark. Introduce shared hardware detection and explicit backend capabilities.
2. **The render memory model assumes Spark unified memory.** `spark/memory_guard.py` checks host RAM using Spark measurements, without checking separate GPU VRAM. On this machine its smallest 864x480, 22-frame video estimate was about 55 GB for the int8 route or 75 GB for ComfyUI's bfloat16 route, before safety allowances. These are existing Spark estimates, not RTX measurements. The configured H3 stack is not a viable local installation as-is; containerization does not resolve that.
3. **Image-only installation skips memory admission.** `oneshot-setup-spark.sh:90` bypasses the RAM requirement with `--image-only` while claiming that mode fits. `spark/spark_generate.py:image` also starts the server without an equivalent image memory preflight. Image model fit on this machine is unverified.
4. **Remote workers do not yet make this a controller-only installation.** At audit checkout `88fca5e`, local rendering memory was validated before worker selection, and the scheduler could claim a render job on this machine. Phase 6, below, separates that check for one video clip: a controller validates the clip here and asks the worker's guard when the clip is sent. The controller does not claim a video job. Images, edits, voice, music, and long chains stay on their own routes. The tests use a stand-in worker. Live verification on a real worker is pending. The published container stays app-only.
5. **The app installer is coupled to model environments.** `scripts/app.sh:11` requires the main installed environment, although the web app can start with lightweight dependencies. The carrier's prebuilt Spark engines are ARM64; `spark/build-engine.sh` only extracts them on aarch64. Their compatibility with this x86_64/RTX host has not been established through a source build.
6. **Tests leaked shared state.** At audit time `tests/test_sparks.py` replaced `server.run` and `server.memory_of` during collection, and `app/discover.py` started a delayed scan when the router attached. Slice B, below, replaced that with fixtures and an explicit lifecycle.

Do not lower existing memory floors or alter measured Spark limits to make jobs fit. Preserve model isolation, GPU locking, the guarded ComfyUI launch path, and the other constraints in `AGENTS.md`.

## Completed validation

- Ran `bash oneshot-setup.sh --verify`; it correctly refused the default video installation for insufficient system RAM. No installation or weight download followed.
- Created a temporary audit environment at `/tmp/imagination-audit-venv` with lightweight app/test dependencies. Some dependencies were subsequently aligned with `requirements.lock`; this was not a complete locked model environment.
- Started `app/server.py` with temporary storage and prefix directories, bound to loopback on a temporary port. `/`, `/api/state`, `/api/jobs`, and `/api/updates/health` answered successfully. No model was loaded. The server was stopped afterward.
- Parsed all 107 Python files and ran `bash -n` on all 27 shell scripts: no syntax errors.
- Final focused run: **141 passed, 2 failed** across API responsiveness, memory guard, Slate page, lengths, write guard, and discovery tests. Both failures were in length tests encountering the real machine's memory rejection instead of their expected behavior.
- Discovery tests passed independently with background discovery disabled. A memory-admission test that failed alongside the Spark tests passed alone, confirming the global mock leak.
- A broader suite run during the audit did not finish cleanly: it encountered missing FFmpeg, shared-state failures, and a hang, and was terminated. That run is not a pass. Slice B, below, is the later finished run.
- No GPU renders, model downloads, engine builds, or container builds were performed. No source code was changed during the audit. `MEMORY.md` is the first repository change from this conversation.

Temporary logs, if still present: `/tmp/imagination-audit-tests.log`, `/tmp/imagination-audit-focused.log`, and `/tmp/imagination-audit-collected.log`. These are disposable and are not required to resume work.

## Proposed container design

- **Studio runtime container:** web UI, project management, queue, lightweight Python dependencies, and FFmpeg/ffprobe. Provide an app-only startup path independent of model installation.
- **Optional managed rendering variant:** replace the same Studio service with an image containing a separate ComfyUI environment, while keeping the renderer, queue and guard in one container/PID namespace. This refines the earlier separate-container idea because existing lifecycle code relies on local process ownership. Keep the repository's ComfyUI pin and guarded launcher; reuse the host's model directory read-only. SDXL Turbo already on disk is the first candidate to validate, not an enabled or benchmarked model.
- **Existing host renderer:** serves another app and must remain unchanged. No submissions, lifecycle control, migration, or configuration changes are in scope. Only read-only inventory and read-only model-file reuse are planned.
- **Persistent mounts:** project data and outputs survive container replacement. Existing model files are reused read-only; any new downloads go to separate storage. Keep weights out of built images and preserve licensing and attribution requirements.
- **Network:** proposed initial Linux deployment uses host networking with Studio bound to `127.0.0.1:7870`, preserving local-peer API checks without setting `HOST=0.0.0.0`. No published-port mappings in host-network mode. A managed renderer uses a verified free loopback port, provisionally 8191; preserve host ComfyUI on 8188. No LAN/public access was requested.
- **Heavy video:** keep unsupported local routes unavailable. A capable worker is an option after implementing controller-only scheduling, subject to the existing remote-job constraints.

Container boundaries must preserve the guarded ComfyUI launcher, one-at-a-time model execution, GPU lock ownership, and unloading behavior. Do not launch ComfyUI through an unguarded shortcut. Add cgroup and VRAM checks without relaxing the host-memory floor. Available RAM was still only about 2.6 GiB during planning, below the existing 8,000 MB minimum floor; rendering remains gated on memory availability.

## Phase 0 baseline

Rechecked 2026-10-21 with `python3 scripts/doctor.py` (`--json --gpu-container --select-model checkpoints/sd_xl_turbo_1.0_fp16.safetensors`, then a second run with `--skip-models` after the runtime listing was corrected). `tests/test_doctor.py`: 19 passed. Those tests do not contact the shared service. The command is not hooked into `oneshot-setup.sh`, so the Mac and Spark installers are unchanged.

| Check | Result |
| --- | --- |
| App-only blockers | None. Port 7870 is free. Host `ffmpeg` and `ffprobe` are still absent; the app image is expected to supply them |
| Memory | 2,552 MB available of 15,976 MB. Floor 8,000 MB (`max(8000, MemTotal/10)`). Swap about 1.8 GB free of 4 GB. Below the floor, so local rendering stays off |
| Disk | 138.2 GiB free on the home filesystem |
| CPU | i3-2120, 4 logical CPUs, SSE4.2 and AVX, no AVX2 or AVX-512 |
| cgroup | v2, `memory.max` is `max` on this session and the user slice (no cap). User-slice `memory.current` was about 13.5 GiB |
| GPU | RTX 3060, UUID `GPU-82f380a9-0f75-b3d3-8db7-4045bd2dfaeb`, 12,288 MiB, driver 595.84, compute 8.6. About 301 MiB in use by the existing ComfyUI Python process |
| Docker | Server 29.9.0, Compose 5.6.0, named runtimes include `nvidia` |
| GPU container | Succeeded, no image pull. Image `nvidia/cuda:12.6.3-base-ubuntu24.04`, `docker run --rm --network none --read-only --gpus device=<uuid> nvidia-smi`. The container printed the same GPU. No model mount |
| Shared ComfyUI after that probe | Still reachable, version 0.40.0, queue running 0 / pending 0. Watched paths unchanged: `main.py`, `extra_model_paths.yaml`, `custom_nodes`, `models`, and the user unit `comfyui.service` |
| Service | `comfyui.service` active, main PID 897018 at inspection. systemd `MemoryCurrent` about 11.4 GiB. An empty queue is not a GPU reservation |
| Commit | Shared checkout `e0a32bce84f4603f2b61b16b50f8958e68fd6171`. Repository pin `b9b290e43f1eb379946dfeb013faf9bbefa1a618`. Studio HEAD `88fca5e` |
| Nodes present | `CheckpointLoaderSimple`, `CLIPTextEncode`, `KSampler`, `VAEDecode`, `SaveImage`, `EmptyLatentImage`. Custom node names: `example_node.py.example`, `studio_film`, `studio_scene`, `websocket_image_save.py` |
| Models | 48 weight files, about 181 GiB, under `/home/mcpadmin/ComfyUI/models`. Library was not hashed |
| SDXL Turbo candidate | `checkpoints/sd_xl_turbo_1.0_fp16.safetensors`, 6,938,081,905 bytes, sha256 `e869ac7d6942cb327d68d5ed83a40447aadf20e0c3358d98b2cc9e270db0da26`. Provenance against an upstream digest was not checked |
| Port 8191 | Free. This is the provisional managed-renderer port |
| Port 8190 | Container `studio-studio-1` publishes `192.168.50.22:8190` and `:8443`. Leave it alone |

Doctor never enables a render route. Regenerate the inventory with `python3 scripts/doctor.py` or `--json`. Add `--gpu-container` only for another disposable `nvidia-smi` check, and `--select-model <path-inside-models>` to hash one file.

## Slice B test isolation

Recorded 2026-10-21. At the end of this slice, production `SPARK` was still `sys.platform.startswith("linux")`. Phase 2 replaces that with `hardware.is_spark()`. The 8000 MB floor is unchanged. Local rendering stays off. Doctor is still not hooked into the Mac or Spark installers. RTX is only a test profile: `server.SPARK` is set only when the profile name is `spark`.

The queue and discovery no longer start at import during pytest collection. `begin()` / `end()` own them. `tests/conftest.py` forces `STUDIO_DISCOVER_DELAY=-1` and starts the queue for the session. Spark stand-ins restore `server.run`, `server.memory_of`, and the hub methods, and close their fake servers on teardown.

The audit hang was a voice live stream, not the discover timer. This host had about 3 GB free. The stand-in engine's kill-guard (8 GB) SIGKILLed it during load, `Cuda.make` did not mark `/live` done, and the test blocked in `live.content`. Stand-in voice and music engines set `guard_gb` to 0. A failed CUDA load still marks the stream done. Production `guard_gb` stays 8 and `NEED_GB` stays 50. Song tests report a large `meminfo` so this host's RAM does not refuse a stand-in; `tests/test_music.py::test_the_memory_guard_knows_a_song` still passes its own `available_mb`. Sound tests that build commands pretend ffmpeg is present. Tests that need the real binary skip.

| Run | Result |
| --- | --- |
| Focused together: isolation, lengths, sparks, memory guard, discover, doctor, lastframe | 80 passed, 1 skipped (ffmpeg), 31.06 s |
| Full `pytest tests` with `--timeout=240` (pytest 9.1.1) | **616 passed, 8 skipped, 0 failed**, 152.47 s. Exit 0. It did not hang |

Skips: `tests/test_shot.py` (PyAV is not installed; those tests were not collected), `test_mend_paints_over_dull_green_residue` and `test_frame_from_a_clip` (ffmpeg), two motion-link tests (ffprobe and the sample clip), two music soundtrack tests (ffmpeg), and `test_the_delivery_sound_is_resampled_only_when_asked` (ffmpeg).

After both runs, `comfyui.service` was still active, MainPID 897018, NRestarts 0. Container `studio-studio-1` (`40fe133a96d9`) was still healthy on `192.168.50.22:8190` and `:8443`. The suite log is `/tmp/suite-isolation-c.log`.

## Phase 2 hardware and capability boundaries

Recorded 2026-10-21. This machine is an NVIDIA GeForce RTX 3060, family `nvidia-rtx`, product Inspiron 620. It is not a DGX Spark. `app/hardware.py` and `app/capabilities.py` are the boundary. Production `SPARK` is `hardware.is_spark()`, so Linux alone does not select the Spark route. Mode on this host is `app-only`. The 8000 MB floor is unchanged. A live reading had about 13 GB available, so state cites `unmeasured_configuration` and does not cite `insufficient_memory`.

`/api/state` reports engine `none`, an empty `image_models` list, and `local_render` false. Generative jobs (scout, video, studio, long, edit, actions, sound, voice, music) are refused before build, ffmpeg, or a Spark script. Export, soundtrack, project reads, and `POST /api/voice/cut` stay available. `scripts/require-route.sh` stops `image.sh`, `video.sh`, `edit.sh`, `turnaround.sh`, `studio.sh`, and `scout.sh` on this machine before they exec into `spark/`. A mode cannot invent a route: `STUDIO_MODE=spark` stays app-only here; `capabilities.use("spark")` or `use("mac")` opens the existing Mac or Spark behavior for tests. Managed RTX does not render.

The page no longer titles this host as a Mac. Home Generate, Make image, and Queue this clip are disabled with the unmeasured-route reason. A direct `POST /api/jobs` for video returns 400 with code `unmeasured_configuration`. Voice still says "Built with Fish Audio" and Music still says "MiniMax-Music3"; both are unavailable. A saved picture-model id is shown as not on this machine.

| Run | Result |
| --- | --- |
| Full `pytest tests` with `--timeout=240` (pytest 9.1.1) | **627 passed, 8 skipped, 0 failed**, 152.02 s. Exit 0. Log `/tmp/suite-phase2-b.log` |

The 8 skips match Phase 1: `tests/test_shot.py` (PyAV), `test_mend_paints_over_dull_green_residue`, `test_frame_from_a_clip`, two motion-link tests, two music soundtrack tests, and `test_the_delivery_sound_is_resampled_only_when_asked`. Browser check on `http://127.0.0.1:7870` used a temporary `STUDIO_HOME` and `STUDIO_DISCOVER_DELAY=-1`. Shared ComfyUI and `studio-studio-1` were not restarted or given jobs. Local rendering stays off. Phase 3 was not started.

## Phase 3 app container

Recorded 2026-10-21. Commit `4718c9e` (not pushed). The app-only image is `imagination-studio-app:4718c9e`, id `sha256:9b07ed901a8ec4c3be9b6e40a4954ad54c5ed15e399d6d5a81e6972e90c61a98`, 788 MB. It contains no model weights. Base digests are pinned for linux/amd64. `requirements.lock` is unchanged. The runtime has ffmpeg and ffprobe, uid 10001, and no gcc, git, nvcc, or pip. `GET /api/health` on that image returned `{"ok":true,"version":"3.0.1","revision":"4718c9e","container":true}`.

`/api/state` was 200: home `/data`, engine `none`, `image_models` `[]`, `local_render` false, mode `app-only`, platform label `LINUX · NO GPU · APP-ONLY`. The container has no `nvidia-smi`, so the lead reason inside it is `unavailable_hardware`. On the host, with the RTX 3060 visible, the lead reason remains `unmeasured_configuration`. Export stays enabled. A video job is refused. The update message is "This Studio is a container image. Rebuild the image and recreate the container to update."

Compose publishes `127.0.0.1:7870` and `192.168.50.22:7870`. `HOST` is unset. Inside the bridge, `STUDIO_BIND=0.0.0.0` so the proxy can reach the app. The measured export was about 59–69 MiB RSS, 8–9 pids, and under 35% of one core. The applied ceiling is headroom over that run: memory 1 GiB, memswap 1 GiB (no extra swap), pids 64, CPUs 1. That ceiling does not replace the 8000 MB floor. An export completed under it.

Projects, a Slate edit, an upload, a probe, and a 1-second ffmpeg export survived recreation on volume `imagination-studio-app-data`. Those workflow checks ran on image `sha256:63de09e728d224e54e87a615ee350f785b24dcea94dc8bf727e23b3eb19f20bd` (tag `658fb1a-dirty`). The committed rebuild reused those filesystem layers and changed the revision to `4718c9e`. `docker stop -t 20` exits 143: uvicorn re-raises SIGTERM after the lifespan shutdown. No owned GPU lock remained on the volume. The container is published on `127.0.0.1:7870` and `192.168.50.22:7870`. There is no login. Updates, Brains, and Agents stay limited to loopback and Tailscale.

| Run | Result |
| --- | --- |
| Full `pytest tests` with `--timeout=240` (pytest 9.1.1), after the container-mode edits | **643 passed, 8 skipped, 0 failed**, 156.14 s. Exit 0. Log `/tmp/phase3-build/suite3.log` |

The 8 skips match Phase 1. In the browser, on the pre-commit image, Make image and Queue this clip were disabled with "No GPU is available for a local render on this machine." Home Generate, the Voice render control, and Music Generate were disabled. The credits "Built with Fish Audio" and "MiniMax-Music3" were still shown. Updates showed the container rebuild sentence. A fresh Create load reported no console errors. Shared `comfyui.service` stayed active, MainPID 897018, NRestarts 0. `studio-studio-1` stayed healthy on `192.168.50.22:8190` and `:8443`. Local rendering stays off.

## Phase 4 RTX admission

Recorded 2026-10-21. Admission is `spark/rtx_admit.py`. Spark measurements, `spark/memory_guard.py`, `spark/comfy.sh`, and `capabilities.reasons()` are unchanged. The host lead reason stays `unmeasured_configuration`. Inside the app container it stays `unavailable_hardware`. The 8000 MB floor was not lowered.

Host RAM, cgroup headroom, and free VRAM are separate budgets. The cgroup v2 root on this machine has no `memory.max`. A chain of `max` up to that root is a known absence of a cap. A missing `memory.max` below the root is a failed probe. There is no measured profile, so a live admit is not ok. Image submissions stay disabled. No calibration workflow ran, and no job was sent to ComfyUI.

`python3 spark/rtx_admit.py admit` returned `unmeasured_configuration` and exited 1. The same-time reading was host available 12562 MB of 15976 MB, floor 8000 MB, no numeric cgroup cap, free VRAM 11653 MiB, reserve 512 MiB. `spark_generate.call()` exits before any request on this machine. `stop_comfy` posts only for a live owned renderer. Port 8188 stays protected. Tests stop only `sleep` processes they start.

The published container was not rebuilt. It is still `imagination-studio-app:4718c9e` on `127.0.0.1:7870` and `192.168.50.22:7870`. After the suite, `comfyui.service` was still active, MainPID 897018, NRestarts 0. `studio-studio-1` was still healthy on `192.168.50.22:8190` and `:8443`.

| Run | Result |
| --- | --- |
| Full `pytest tests` with `--timeout=240` (pytest 9.1.1), after the cgroup-root fix | **662 passed, 8 skipped, 0 failed**, 160.88 s. Exit 0. Log `/tmp/phase4-suite-cgroup.log` |

The 8 skips match Phase 1.

## Phase 5 SDXL Turbo on the shared ComfyUI

Recorded 2026-10-21. The operator corrected the route: pictures go to the shared ComfyUI at `http://127.0.0.1:8188`. Studio does not start a second server, and it does not start, stop, interrupt, unload, or clear that service. Spark `call()` and `stop_comfy` still refuse port 8188.

Rebuilt 2026-10-21 as `imagination-studio-app:a8edb67` (`sha256:8acb2d19d02c40850db99d3161e70775af7df6fe76701f156f6eaf2c2e4bdf60`, 788 MB), healthy on `127.0.0.1:7870` and `192.168.50.22:7870`. `GET /api/health` returned `{"ok":true,"version":"3.0.1","revision":"a8edb67","container":true}`. `/api/state` lists one picture model, `sdxl-turbo` at 512×512 and 1 step. `local_render` stays false. Scout and video stay disabled. The lead reason inside the container stays `unavailable_hardware`. From `192.168.50.22`, `/api/updates`, `/api/brains`, and `/api/agent/tools` return 403, and `/api/state` returns 200. The checkpoint is mounted read-only at `/models/sd_xl_turbo_1.0_fp16.safetensors`. No GPU is attached. An in-container budget check was ok: host available 13535 MB, free VRAM 11706 MiB, shared URL `http://172.17.0.1:8188`. It did not submit a prompt. Two projects were still listed. Home's Picture chip says SDXL Turbo and Picture Generate is enabled. Clip, Voice, and Song Generate stay disabled. Create offers only 512×512, and Make image was left unclicked. `comfyui.service` stayed active, MainPID 897018, NRestarts 0. `studio-studio-1` stayed healthy on `192.168.50.22:8190` and `:8443`.

The checkpoint size matches 6,938,081,905 bytes. The sha256 is the Phase 0 value; this run did not re-read the file. The licence is the Stability AI Non-Commercial Research Community License. The recipe is 512×512, one step, ComfyUI cfg 1.0, `euler_ancestral`, `sgm_uniform`, batch size 1, count 1. The running server is the installed tree `e0a32bce84f4603f2b61b16b50f8958e68fd6171`. Pinned ComfyUI remains `b9b290e43f1eb379946dfeb013faf9bbefa1a618`. Spark's `--use-ck-attention` is unchanged.

One still finished. Prompt id `24757f89-dd49-404b-960b-8cfd462bab86`, seed 1, 512×512, 267932 bytes, 51.8 s, `~/ImaginationStudio/sdxl-turbo/still_s1.png`. The queue was empty before the prompt and empty after. Before the prompt, host available was 13667 MB and free VRAM was 11653 MiB. During the load, host available fell to 6654 MB, under the 8000 MB floor, and free VRAM fell to 4697 MiB, above the 512 MiB reserve. The floor was not lowered. `docs/benchmarks/rtx3060-images.json` is `enabled` true. A later still still has to clear the floor before it is submitted. The weights stayed loaded. The service stayed active, MainPID 897018, NRestarts 0.

`local_render` stays false. Scout and video routes stay disabled. On this host the lead reason stays `unmeasured_configuration`. With the profile enabled, the only picture model on this machine is `sdxl-turbo`.

| Run | Result |
| --- | --- |
| Full `pytest tests` with `--timeout=240` (pytest 9.1.1), after the shared ComfyUI still | **672 passed, 8 skipped, 0 failed**, 155.76 s. Exit 0. Log `/tmp/phase5-shared2.log` |
| Full `pytest tests` with `--timeout=240` (pytest 9.1.1), after the container still path | **674 passed, 8 skipped, 0 failed**, 157.47 s. Exit 0. Log `/tmp/phase5-container.log` |

The 8 skips match Phase 1. The earlier route suite, before this correction, was 672 passed, 8 skipped, 159.38 s, log `/tmp/phase5-suite.log`.

## Phase 6 optional remote H3 video

Recorded 2026-10-21. `STUDIO_MODE=controller` queues one video clip for a worker. This machine does not render H3. The prompt, frames, paths, and references are checked here. `build()` skips the local memory room for that clip. The worker's own guard is asked when the clip is sent. `sparks` stays 1 to 5. On a controller the number is workers, and this machine is not a GPU slot. On a Spark, 1 still means this Spark, and after two takebacks the clip still renders here. A controller clip that no worker will take fails with a reason. `LOST` stays 60 seconds. A down worker waits for Check. An export stays local. `scripts/video.sh` exits before H3 starts. SSH uses an identity only when both `STUDIO_SSH_IDENTITY` and `STUDIO_SSH_KNOWN_HOSTS` are existing files. The published service stays `STUDIO_MODE=app-only` and does not mount a key.

The tests use a stand-in worker. Live verification on a real worker is pending. No tunnel was opened and H3 was not started.

| Run | Result |
| --- | --- |
| Full `pytest tests` with `--timeout=240` (pytest 9.1.1) | **693 passed, 8 skipped, 0 failed**, 166.43 s. Exit 0. Log `/tmp/phase6-suite.log` |

The 8 skips match Phase 1.

Rebuilt 2026-10-21. The published app container is `imagination-studio-app:5b8857b` (`sha256:e4932ad6ad3fb04b8923b90702273e8194f435536f60076e7588429d0f7c0fff`, 788 MB), healthy on `127.0.0.1:7870` and `192.168.50.22:7870`. `GET /api/health` returned version 3.0.1, revision `5b8857b`, container true. Images `050899a` and `14151f6` are still present. Projects `p1021-061926` and `p1021-062452` were on the volume. The service stays `STUDIO_MODE=app-only`. `/api/state` reported platform `linux`, `local_render` false, mode `app-only`, lead reason `unavailable_hardware`, and picture model `sdxl-turbo`. Video stayed disabled. The Sparks page showed the existing DGX Spark note. Home title was "Imagination Studio · Linux". The sidebar said "for Mac, Spark, and Linux". Clip Generate was disabled. Picture Generate was not clicked. `comfyui.service` stayed active, MainPID 897018, NRestarts 0. `studio-studio-1` stayed healthy. Live verification on a real worker is pending.

## Next steps

1. The published app is `imagination-studio-app:5b8857b` (`sha256:e4932ad6ad3fb04b8923b90702273e8194f435536f60076e7588429d0f7c0fff`) on `127.0.0.1:7870` and `192.168.50.22:7870`. Keep `imagination-studio-app:050899a` and `imagination-studio-app:14151f6`. The published service stays `STUDIO_MODE=app-only`. A controller can queue one video clip for a worker and does not render H3 here. That role is not configured on the published container. Live verification on a real worker is pending. Do not mount an SSH key. The system runs on Mac, DGX Spark, and Linux. The sidebar says "for Mac, Spark, and Linux". The live queue stays in memory. `job-history.json` is a record (`queue` false). One process holds the data-directory lock. Container discovery walks only `STUDIO_SCAN_ROOTS`. Operations are in `docs/OPERATIONS.md`. A still submits to the shared ComfyUI at `http://172.17.0.1:8188` only when its queue is empty and host available is at or above 8000 MB. The floor stays. Do not unload or restart pid 897018. Port 8188 is not published, and no GPU is attached to the app container. Keep the models and the container on port 8190 unchanged.

Phase 0 added `scripts/doctor.py` and ran it. At the end of that slice the only container was a disposable `nvidia-smi` probe. No model weights were downloaded, and the shared ComfyUI service was not changed. The app image arrived in Phase 3.

## Maintenance

Commit each finished slice before starting the next one. Push that slice to the fork `https://github.com/dogemackenzie/imagination-studio` (`fork`). Pull the original from `https://github.com/wassermanproductions/imagination-studio` (`origin`). Do not push to `origin` unless asked. The fork adds to the original. It stays able to take upstream updates.

GitHub on this machine is this checkout only. The CLI is logged in as dogemackenzie with `GH_CONFIG_DIR=/home/mcpadmin/.config/imagination-studio-gh`. The binary is `/home/mcpadmin/.local/share/imagination-studio/gh/bin/gh`. The credential helper is set on this repo's `fork` remote. Do not run `gh auth setup-git`, and do not give other projects that config.

Update this file when decisions are made or work is completed. Record what changed, validation results, unresolved issues, and the next concrete step. Keep proposals distinct from implemented behavior. Never store credentials, access tokens, or private prompt/media contents here.
