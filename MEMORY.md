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
4. **Remote workers do not yet make this a controller-only installation.** `app/server.py:1900` validates local rendering memory before worker selection. `app/sparks.py:646` allows the controller to claim render jobs. A controller-only mode needs scheduling and admission changes. Existing remote execution supports single video jobs only; do not assume images, voice, music, or long-video jobs are remotely supported.
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

## Next steps

1. Phase 3 of `docs/BUILD_PLAN.md`: an app-only container on `127.0.0.1:7870`, with persistent storage, loopback binding, health checks, and a pinned dependency set. Do not enable a renderer. Keep the shared ComfyUI service, its models, and the container on port 8190 unchanged. Do not lower the 8000 MB floor.
2. Add host-RAM, cgroup and VRAM admission plus renderer/lock ownership checks. Keep unsupported models disabled.
3. Verify and benchmark the existing SDXL Turbo file under the guarded managed route once sufficient host RAM is available; establish limits from measurements before enabling it in the UI.
4. Optionally implement controller-only support for eligible H3 video jobs on a capable worker. Keep other job types within their current execution constraints.
5. Complete shutdown, persistence, backup/restore, rollback and documentation checks before calling a release ready.

Phase 0 added `scripts/doctor.py` and ran it. The only container was a disposable `nvidia-smi` probe; the Studio app image was not built. No runtime packages were installed, no model weights were downloaded, and the shared ComfyUI service was not changed.

## Maintenance

Commit each finished slice before starting the next one. Push only when asked.

Update this file when decisions are made or work is completed. Record what changed, validation results, unresolved issues, and the next concrete step. Keep proposals distinct from implemented behavior. Never store credentials, access tokens, or private prompt/media contents here.
