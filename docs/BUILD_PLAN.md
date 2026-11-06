# Shared-backend Studio integration plan

Updated 2026-11-06. The user approved implementation, integration, deployment, and additional testing. ComfyUI is a shared backend, not a Studio-owned service. Earlier separate-renderer proposals are superseded.

## Boundaries

- Use the existing shared ComfyUI API for supported Studio stills. Never change its checkout, environment, nodes, models, configuration, permissions, service, or ports.
- Never start, stop, restart, unload models, clear the queue, or interrupt the shared backend. Cancellation stops Studio's client; a submitted shared prompt may finish independently. Do not report cancellation as remote termination.
- The other application's container and data remain untouched. An empty shared queue is a snapshot, not an exclusive reservation; ComfyUI serializes submissions from all clients.
- Keep Studio's app container and data separate. Preserve the deployed LAN binding and explicit access-key setting; document them accurately. Do not mount credentials or enable unmeasured local model routes.
- Preserve host and GPU reserves. If measured peak demand cannot fit, refuse the render and report the capacity limitation. Testing approval does not justify lowering a floor.

## Existing foundation

Phases 0–3 are implemented: diagnostics, test isolation, hardware/capability boundaries, an unprivileged app container, persistence, health checks, bounded CPU exports and lifecycle management. Job history, an instance lock, backup/restore tooling and bounded logs are implemented. SDXL Turbo and Z-Image Turbo each have one recorded still, but those measurements breached the original reserves and do not constitute completed safety acceptance. Controller-only video has simulated-worker coverage; no real worker is configured.

## Remaining work and evidence

1. **Documentation and baseline** — reconciled MEMORY, this plan and operations with the 3.2.0 baseline and the approved shared-backend boundary. Complete.
2. **Upstream integration** — merge upstream `c572890` (4.3.1) in an isolated checkout; resolve 26 conflicting files without losing local capabilities, container behavior, controller routing or upstream Windows/GPU behavior. Run the full non-GPU suite with isolated data and no live model submissions. In progress.
3. **Shared-backend admission** — budget measured cold-load host/VRAM demand plus reserves, recheck before each submission, fail closed on missing measurements/telemetry, and prevent overlapping Studio submissions even when its client is cancelled or restarted. Never treat another app's allocations as reclaimable. Implemented, including conservative combined NVML/backend readings and persistent prompt ownership.
4. **Image verification** — validate both client workflows, busy/refusal paths, timeout/cancellation/restart behavior and output ownership with a disposable mock backend. Live cold/warm rendering is conditional on sufficient measured headroom. If it cannot fit, retain the refusal and record live validation as blocked; do not label it passed. Both clients pass mocked workflow/output tests; cancellation, lost responses, cross-client locks and capacity refusals are covered. Live renders are blocked by headroom. Multi-image scouts remain outside the one-still profiles.
5. **Release and recovery** — test the container against copied data, check the UI and CPU media flow, rehearse backup/restore and old-image rollback, preserve the current image, build with disk headroom, deploy only Studio, and verify both existing services remain healthy. Copied-data export and old-image rollback passed with identical project/media hashes and no job replay. The browser sweep covered all 34 tabs and 260 buttons with writes blocked. Final regression and production deployment are pending. No shared Docker pruning.
6. **Optional remote video** — maintain and test controller behavior, lost-worker/cancellation/code mismatch paths. A real end-to-end clip requires a configured capable worker and SSH access; do not invent a worker or mount credentials automatically. Simulated coverage remains part of the regression suite; the real-worker prerequisite is still external.

## Completion criteria

- Reviewed upstream integration, relevant unit/API/UI tests and container checks pass; skipped/unavailable features are explicitly identified.
- No new heavyweight route is enabled by Linux detection alone. Shared API lifecycle mutations are excluded from Studio's still path.
- Memory decisions reflect the recorded peaks, not checkpoint file size alone. Insufficient capacity yields an actionable refusal before POST /prompt.
- Studio projects/media survive deployment and rollback rehearsals. Interrupted jobs are history, not replayed work.
- Documentation distinguishes implemented code, simulated validation, live evidence and external blockers.

## Deferred scope

Additional models, multi-image scouts, local H3/Ref2VA/Drama/Music engines, GPU hardware upgrades, and changing either shared service are separate milestones. Upstream availability alone does not qualify those features on this RTX 3060 / 16 GB host.
