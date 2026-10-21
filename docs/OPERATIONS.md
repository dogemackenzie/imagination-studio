# Operations

The system runs on a Mac, a DGX Spark, and Linux. This page is the Linux app container on this machine.
The commands match `compose.yaml`, `Dockerfile.studio`, `scripts/container-entrypoint.sh`, `scripts/container-healthcheck.py`,
`scripts/doctor.py`, `scripts/comfy_preflight.py`, and `scripts/studio-data.py`.

The app listens on `127.0.0.1:7870` and `192.168.50.22:7870`. There is no login. The container does not
attach a GPU. It does not publish 8188, 8190, 8191, or 8443. `HOST` stays unset. `STUDIO_BIND` is `0.0.0.0`
only inside the container so the published addresses can reach the app.

Leave the shared ComfyUI service and the container that publishes port 8190 as they are. Studio does not
start, stop, interrupt, unload, or clear that ComfyUI. `spark/comfy.sh` is the DGX Spark renderer and is
not the start command here.

## Build, start, stop, logs, health

From this repository:

```bash
SOURCE_REV=<git short> docker compose build studio
SOURCE_REV=<git short> docker compose up -d --force-recreate --no-deps studio
docker compose logs --tail=200 studio
curl -sS http://127.0.0.1:7870/api/health
docker compose stop studio
```

`SOURCE_REV` is the image tag and the revision reported by `/api/health`. Keep the previous tag.
`docker compose stop studio` sends SIGTERM. The stop grace period is 20 seconds. That stop is this
service only.

A healthy body is JSON with `ok` true, `version`, `revision`, and `container` true. The check inside
the image is `scripts/container-healthcheck.py`, which GETs `/api/health` and does not scan disks or
load a model.

The in-app updater in a container answers that the image has to be rebuilt and the container
recreated. It does not fetch git.

## Optional controller

The published container stays `STUDIO_MODE=app-only`. A controller is a separate setting,
`STUDIO_MODE=controller`, on a checkout you start yourself. It can queue one video clip for a
worker. This machine does not render that clip. The image does not mount an SSH key, a known-hosts
file, or an agent socket. A real worker has not been verified from this machine. The rules are in
[docs/multi-spark.md](multi-spark.md).

## Outputs

Inside the container the data directory is `/data` (`STUDIO_HOME`). The named volume is
`imagination-studio-app-data`.

| Path | What it holds |
| --- | --- |
| `/data/projects` | Project files |
| `/data/uploads` | Uploads |
| `/data/renders` | Renders |
| `/data/scouts` | Scout sheets |
| `/data/exports` | Exports |
| `/data/slate` | Slate |
| `/data/settings.json` | Settings |
| `/data/job-history.json` | A record of jobs |

`/data/job-history.json` has `"queue": false`. It is a record. Mounting the volume does not restore
the live queue and does not run a job again. The live queue is the memory of the server process.
`GET /api/jobs` is that memory. `GET /api/jobs/history` is the record.

The measured still from the host run is `~/ImaginationStudio/sdxl-turbo/still_s1.png`. That file is
on the host. It is not inside the container volume.

## Model inventory

`python3 scripts/doctor.py` prints a read-only baseline. It GETs the shared server's stats and queue
counts. It does not submit a prompt. `--select-model` is the only way it hashes a file, and this
deployment does not need to hash the checkpoint again.

In the container, discovery walks only `STUDIO_SCAN_ROOTS`. Empty walks nothing: not the home
directory and not `/mnt`, `/media`, `/data`, or the rest of the host. A Mac or a DGX Spark still
walks that machine's own model folders. That walk is how those installs find weights.

Five separate facts:

| Fact | Meaning |
| --- | --- |
| present | A file is on disk |
| compatible | The file matches the expected place or a verified candidate, or the checkpoint size matches the profile |
| licensed | A recorded licence covers the intended use. SDXL Turbo's licence is the Stability AI Non-Commercial Research Community License. That is not a commercial grant |
| benchmarked | A measured profile names the hardware and a finished render |
| runnable | The measured profile is enabled and the file is the recorded size |

A file that is present is not runnable. Catalogue rows from Models & weights keep `licensed`,
`benchmarked`, and `runnable` false. The measured still is `measured` on `GET /api/discover`, from
`spark/sdxl_turbo.py`. Admission, including the 8000 MB floor and an empty queue, is checked again
when a still is submitted. `local_render` stays false. Scout and video routes stay disabled.

The measured profile is `docs/benchmarks/rtx3060-images.json`. Hardware recorded there: NVIDIA
GeForce RTX 3060, UUID `GPU-82f380a9-0f75-b3d3-8db7-4045bd2dfaeb`, 12288 MiB, driver 595.84.
Checkpoint `sd_xl_turbo_1.0_fp16.safetensors`, 6938081905 bytes. ComfyUI pin
`b9b290e43f1eb379946dfeb013faf9bbefa1a618`. Installed shared tree
`e0a32bce84f4603f2b61b16b50f8958e68fd6171`. Host floor 8000 MB. The floor stays.

## Refusal reasons

`app/capabilities.py` is the text the API returns. The lead reason on this host, where the GPU name
is visible, is `unmeasured_configuration`. Inside the app container, where no GPU is visible, the
lead reason is `unavailable_hardware`.

| Code | Reason |
| --- | --- |
| `unavailable_hardware` | No GPU is available for a local render on this machine. |
| `missing_runtime` | The runtime this route needs is not installed. |
| `missing_model` | The model this route needs is not installed. |
| `incompatible_nodes` | The installed nodes cannot run this route. |
| `busy_renderer` | The renderer is busy with another job. |
| `insufficient_memory` | Available memory is under the 8000 MB floor, so a render cannot start. |
| `unmeasured_configuration` | This machine has no measured render route. An RTX card is not a production render mode, and a managed RTX image does not enable one. |

Export and soundtrack stay available. They assemble files that already exist. Offering SDXL Turbo
does not set `local_render` true and does not enable scout or video.

## A still

SDXL Turbo is the measured picture route. The profile's `enabled` field is true. A still is one
prompt to the shared ComfyUI when its queue is empty and host available memory is at or above
8000 MB. The recipe stays 512×512, one step, cfg 1.0, `euler_ancestral`, `sgm_uniform`, batch size 1.
The container uses `SDXL_COMFY_URL=http://172.17.0.1:8188`. The host uses `http://127.0.0.1:8188`.

Before any integration run that would submit a prompt:

```bash
python3 scripts/comfy_preflight.py --owner-pid "$(systemctl --user show comfyui.service -p MainPID --value)"
```

That command GETs `/queue` on an allowed origin (`http://127.0.0.1:8188` or `http://172.17.0.1:8188`)
and checks that the owner pid's command line is still ComfyUI. It exits 1 when the queue is not
empty, the queue cannot be read, or that pid is gone. It does not submit, interrupt, unload, or
clear. Other origins are refused.

## Jobs and restarts

The live queue is in memory. A graceful stop cancels queued and running jobs and writes those rows
as `cancelled`. A process that dies first leaves `queued` or `running` in the record; the next start
rewrites those rows as `interrupted` and does not put them on the queue. `done` is the only success
status. Cancelled and interrupted jobs stay in `GET /api/jobs/history` with the reason.

One Studio process holds `$STUDIO_HOME/studio.lock` for its whole life. A second process on that
same directory exits before it takes work. A GPU lock whose owner pid is still alive also refuses
the start. A lock whose owner is already gone does not.

## Backup, restore, and rollback

Copy project data out of a stopped app container, then rehearse on that copy:

```bash
docker compose stop studio
mkdir -p /tmp/studio-data-copy
docker cp "$(docker compose ps -aq studio)":/data/. /tmp/studio-data-copy
python3 scripts/studio-data.py rehearse --home /tmp/studio-data-copy --work /tmp/studio-rollback \
  --image <the tag you are moving to> --prior-image <the tag you can return to>
SOURCE_REV=<the tag you are moving to> docker compose up -d --force-recreate --no-deps studio
```

`rehearse` writes a backup, bumps `schema.json` on the working copy only after that backup exists,
and restores the backup into `/tmp/studio-rollback/restored`. The restored tree is the
pre-migration data. The command records both image tags. It does not run Docker and it does not
delete an image.

To put a backup into an empty directory:

```bash
python3 scripts/studio-data.py backup --home /tmp/studio-data-copy --to /tmp/studio-backup
python3 scripts/studio-data.py restore --from /tmp/studio-backup --into /tmp/studio-restored
```

Restore copies projects and settings. It does not run a job. There is no schema migration on the
live volume in this release. Rehearse on a copy before adding one.

Return the app to a tag that is already on this machine:

```bash
SOURCE_REV=<prior tag> docker compose up -d --force-recreate --no-deps studio
```

Keep the previous image. Do not prune images, the build cache, or model files.

## Disk, logs, and temporary files

Compose keeps the app log at 10 MB times 3 files. The container `/tmp` is a 256 MB tmpfs. A job's
own log in memory stays at the last 20000 characters. `STUDIO_DISK_RESERVE_BYTES` is 1073741824
(1 GiB) on the data filesystem. A job is refused, with the free space and the reserve in the
message, when less than that is free. The 8000 MB figure is host RAM and is a different floor.

```bash
python3 scripts/studio-data.py sweep --home /tmp/studio-data-copy
```

`sweep` deletes regular files under `comfy/temp` that are older than a day. It skips symlinks.
The server process runs the same sweep on its own data directory when it starts. It does not delete
models, Docker images, or files outside that temp directory.
