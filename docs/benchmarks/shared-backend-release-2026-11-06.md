# Shared-backend release verification — 2026-11-06

Studio 4.3.1 is deployed from `3587a3a`, merging upstream `c572890c31dbca321df427f63e4cced3ce77d02c`. The previous image `imagination-studio-app:fc9d9ec` remains available. Documentation-only follow-up commits do not alter the image's source revision.

## Scope and results

- Full non-GPU suite in the test image: **2277 passed, 99 skipped, no failures**, 442 seconds. Test-only Git, SSH, procps and Node tools are separate from the runtime. Tests ran with isolated data, no network outside their container, no GPU attachment, and an init process to reap child processes. Platform/model-dependent skips are not feature certification.
- Both SDXL Turbo and Z-Image clients completed mocked workflow/output checks. Coverage includes missing measurements, peak-plus-reserve admission, queue refusal, lost submission responses, cancellation/restart correlation, cross-client locking, ownership-preserving backup and forbidden backend lifecycle calls.
- Browser sweep: **34 tabs, 260 buttons, 38 writes blocked**, zero uncaught page errors, broken media or phone overflow. Three empty-timeline export attempts correctly returned 400. The checker now returns to the app after an unsuccessful download and verifies the requested tab actually rendered.
- After deployment, a second browser check passed Home, Image, Machine and Settings (51 button clicks, 10 writes blocked), with no page errors or phone overflow.
- CPU media: silent and audio clips export successfully. Container exports use bounded FFmpeg threads and Studio's cancellable owned-child runner. Night Shift's final reel normalizes video timestamps to retain 24 fps across AAC segment joins.
- Recovery: backup/restore rehearsal passed. A copied-data upgrade, old-image rollback and re-upgrade retained identical project/media hashes and did not replay jobs. A full backup was taken with production Studio stopped immediately before replacement. The deployed data hashes also match.
- Runtime: health reports version 4.3.1 and revision `3587a3a`; mode is app-only and local rendering is false. Existing bindings, key setting, readonly model/telemetry mounts and resource limits were retained. Credits and license files are present. Runtime has FFmpeg/ffprobe and has no Git, pip, compiler or CUDA toolkit.
- Shared ComfyUI retained its process identity and zero service restarts. The other app stayed healthy. No model installation, service/configuration change, queue mutation, model unload or shared Docker pruning was performed.

## Capacity and limitations

Admission derives cold-load demand from the recorded before/minimum memory readings, then adds 512 MB host and 256 MiB GPU margins. Existing 8000 MB host and 512 MiB GPU reserves remain. When NVML and backend readings are both available, the lower free-VRAM reading wins.

| Recipe | Host headroom including margin | Minimum host available with reserve | GPU headroom including margin |
| --- | ---: | ---: | ---: |
| SDXL Turbo | 7525 MB | 15525 MB | 7212 MiB |
| Z-Image Turbo | 11646 MB | 19646 MB | 11704 MiB |

Live read-only admission checks saw approximately 6.8 GB host available and refused both recipes before submission. **No live cold/warm render was attempted or certified.** Z-Image's host requirement exceeds this machine's total RAM, and its GPU demand also leaves little room on a 12 GB card. The earlier benchmark images breached reserves and remain historical measurements only.

The Studio client lock coordinates Studio processes using the same data directory; it cannot reserve capacity against another application's submissions. ComfyUI owns queue execution. A cancelled client may leave its prompt running; the persisted record prevents automatic duplicate submissions. An uncertain response or missing terminal history requires read-only investigation, never clearing the shared queue.

Optional remote video has simulated-worker coverage but no real worker configured. Local H3, Ref2VA, voice/music engines, additional models and multi-image scouts remain unqualified here. These are external prerequisites or separate milestones, not silently skipped acceptance checks.
