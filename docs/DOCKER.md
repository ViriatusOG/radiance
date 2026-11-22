# Radiance in a container

The repository builds into one Docker image that holds the engine, its plugins, its tools and the
recipes. It is assembled from scratch out of the install tree, exactly the shared libraries that
tree loads, a static busybox, and `curl`, `jq` and a CA bundle for `hfget.sh`. The host needs
Docker, an AMD GPU and an `amdgpu` kernel driver that supports ROCm 10.1. It does not need ROCm, a
compiler or FFmpeg.

The image is published as [`stilldeadcode/radiance`](https://hub.docker.com/r/stilldeadcode/radiance),
with its kernels built for RDNA4 (`gfx1201`, libr4d), RDNA3 (`gfx11-generic`, libr3d) and Strix Halo
(`gfx1151`, libhalo): `docker pull stilldeadcode/radiance`. Build it yourself for
other cards or another commit. The examples below use `radiance`, the tag a local build gets; with
the published image, write `stilldeadcode/radiance` instead.

Besides radiance (Apache-2.0, `/opt/radiance/share/doc/radiance/`), the image holds glibc,
libstdc++, the ROCm runtime, FFmpeg, busybox, curl, jq and a few other libraries, each under its own
license, GPL and LGPL among them. `/usr/share/doc/THIRD-PARTY` in the image lists each one with its
version, its license files and its source. `docker/licenses.sh` writes it during the build and fails
the build if a file in the image cannot be traced to a package.

## Build the image

```sh
docker/build.sh                        # HEAD, tagged radiance:<version>, radiance:<commit>, radiance:latest
docker/build.sh -s gpu-host            # stream the commit over ssh; gpu-host builds and keeps the image
docker/build.sh -j 8                   # cap compile jobs (a device kernel takes 2-3 GB under hipcc)
docker/build.sh -J 4                   # tests at a time (default: the compile jobs)
docker/build.sh -B 5                   # every test's CPU budget times 5, for a host slower per thread
docker/build.sh -E tokenizer_perf_test # leave tests matching a regex out of the test stage
docker/build.sh -t 'gfx1200;gfx1201'   # GPU architectures (default 'gfx1201;gfx11-generic;gfx1151')
docker/build.sh -r <commit>            # any other commit
docker/build.sh --target build         # stop at a stage, tagged radiance-<stage>
```

The build context is `git archive` of the commit, so the image holds only what is committed: no
build directory, no uncommitted edit, no model file. It builds with BuildKit (`docker buildx`), which
runs independent stages in parallel and caches each one; with `-s` only the remote machine needs
buildx.

The `Dockerfile` stages:

| stage | what it does |
|---|---|
| `ffmpeg` | FFmpeg 8.1 with only the demuxers `core/mm/media.cpp` admits, the decoders their streams carry (still images, H.264, HEVC, VP8, VP9, AV1 through dav1d, MPEG-1/2/4, Theora, FLV/H.263, and audio) and the three encoders the images API writes (PNG, JPEG, WebP). The distribution's build brings about 200 packages the engine never uses |
| `llvm` | ROCm's `libLLVM` rebuilt from ROCm's llvm-project at the commit the image's `clang --version` names, configured as ROCm ships it, with `docker/llvm-vopd-pairing.patch` applied. ROCm 10.1's clang does not compile the same source to the same code twice (its VOPD pairing follows heap addresses); the patch makes it repeatable. This library is the only compiler file that changes |
| `build` | `rocm/dev-ubuntu-24.04:10.1.0-full` plus cmake, ninja and g++-14. Device code is compiled by ROCm's clang from `/opt/rocm-repeatable`, a shadow of the ROCm tree whose `libLLVM` is the `llvm` stage's (`-DRAD_HIP_VOPD_REPEATABLE=ON`, `--rocm-path=/opt/rocm`); the HIP runtime and comgr in the image are ROCm's own. Configures with kernels for `GPU_TARGETS`, builds, and installs to `/opt/radiance` with the recipes. Device code is not stored compressed: the engine's AQL backend reads the fat binaries itself and refuses a compressed bundle |
| `test` | `docker/llvm-check.sh`, then `ctest -LE 'gpu\|oot'`: every suite that runs without a card. The check recompiles one kernel file four ways: with pairing off, the image's compiler must emit ROCm's device code byte for byte; with pairing on, two compiles must match the build's own object. The image is produced only if both pass |
| `rootfs` | The stripped install tree, every shared library its binaries and plugins load (resolved by `ldd`, glibc and the ROCm runtime included) in one directory the loader searches by default, a static busybox, and `curl`, `jq` and the CA bundle. ROCm's `libdrm_amdgpu` has the card name table compiled in, so no `amdgpu.ids` is needed. Every ELF file is then resolved from inside a chroot of the tree, so the build fails on a library the image lacks |
| `runtime` | `FROM scratch` and the rootfs. This is the image that ships |

The largest files are the LLVM and Clang libraries comgr loads (about 130 MB and 90 MB), without
which the HIP runtime cannot initialise a device. The shipped image has no distribution: no package
manager and no shell beyond busybox (`docker exec -it <name> sh`). JPEG XL input is refused by name,
since its decoder needs libjxl.

A build container has no card to enumerate, so the GPU architecture must be named; a bare-metal
`./build.sh` detects it. `libr4d` builds for gfx12 targets only; any other target in the list gets
no RDNA4 kernels.

The ROCm version in the image must be one the host's kernel driver supports. It is a build argument
(`ROCM_VERSION`, default `10.1.0-full`; from ROCm 10 on, the development image is published only
with the `-full` suffix).

The GPU suites (`mem_test`, `r4d_selftest`, `kbench`, `kbench_moe`) need the card to themselves.
They run from the `build` stage:

```sh
docker/build.sh --target build
docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    radiance-build ctest --test-dir /build -L gpu --output-on-failure
```

## Run it

The entrypoint is `radiance`, which reads its settings from a YAML file named by `--config`
([`docs/GUIDE.md`](GUIDE.md) §4, [`docs/MULTIMODEL.md`](MULTIMODEL.md) §4). The other installed tools (`rad-convert`, `rad-info`, `rad-kbench`,
`rad-tune`, `rad-schemas`, `rad-tokcheck`, `rad-yaml`, and `rad-export`, which runs on the host
since the image has no Python) are on `PATH`, and the recipes are in
`/opt/radiance/share/radiance/recipes/`.

```yaml
# /srv/radiance/radiance.yaml
server:
  host: "::"
  port: 8000
models:
  m:
    model: /models/m.rad
    tp: 2
```

```sh
docker run --rm -v /srv/models/rad:/models:ro --entrypoint rad-info radiance /models/m.rad

docker run -d --name radiance --network host --init --ulimit memlock=-1 \
    --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    --group-add "$(getent group render | cut -d: -f3)" \
    --group-add "$(getent group video | cut -d: -f3)" \
    -v /srv/models/rad:/models:ro \
    -v /srv/radiance/radiance.yaml:/etc/radiance/radiance.yaml:ro \
    radiance --config /etc/radiance/radiance.yaml
```

The options:

- `--device /dev/kfd --device /dev/dri`, with the `render` and `video` groups, give the container
  the GPUs.
- `--security-opt seccomp=unconfined`: Docker's default seccomp profile refuses io_uring and the
  NUMA binding calls the ROCm runtime makes for host memory. Qwen3.8-Flash-Next's n-gram embedding
  table is read from the container through io_uring. Without this flag the table reads fall back
  to synchronous pread: the model starts and serves, the n-gram reads are slower. The flag keeps
  the fast path.
- `--init`: an init process runs as PID 1 and forwards SIGTERM to the engine, which finishes its
  step and releases the cards.
- `--ulimit memlock=-1`: where no device kernel gathers the n-gram table (a run on the host
  backend), `ngram-placement: ram` locks a copy of it in RAM (47.7 GiB for Qwen3.8-Flash-Next, far
  past Docker's default memlock limit). Without the limit raised, `auto` reads the table from the
  container and `ram` refuses the start. On the cards the copy is pinned memory the cards read,
  which the limit does not count. The compose files set it.
- `--network host`: the server binds the host's addresses directly, IPv4 and IPv6, with no NAT in
  front of it.

The image sets `GPU_MAX_ALLOC_PERCENT=100` and `GPU_MAX_HEAP_SIZE=100`, because the HIP runtime's
defaults cap one allocation, and the heap, below what the placement plan hands out.

A plugin of your own (a kernel library for another card, an architecture, a quantiser) is mounted
and put first on the search path; nothing in the image changes:

```sh
docker run ... -v /srv/radiance-plugins:/plugins:ro \
    -e RADIANCE_HOME=/plugins:/opt/radiance/share/radiance \
    radiance --config /etc/radiance/radiance.yaml
```

`/plugins` holds any of `kernels/`, `architectures/` and `quantizers/`, built against the same
radiance version (`docs/PLUGIN.md` §13). The engine in the image starts on any AMD GPU family: the
image's `GPU_TARGETS` are what its bundled kernel library, libr4d, is compiled for, not a limit on
the cards the engine runs on.

## Compose

`deploy/compose/` has two compose files, `tp1.yaml` for one card (`RADIANCE_GPU` names
it) and `tp2.yaml` for two, each with the server's configuration in a file beside it
(`tp1.radiance.yaml`, `tp2.radiance.yaml`). Each runs one container that serves eleven models behind one port and
brings a model onto the cards when a request names it, parking or unloading others to make room
([`docs/MULTIMODEL.md`](MULTIMODEL.md));
[`deploy/compose/README.md`](../deploy/compose/README.md) lists the models of each.

A compose file names its configuration as a compose config (top-level `configs:`, its `file:`
the `.radiance.yaml` beside it), which Compose places in the container at
`/etc/radiance/radiance.yaml`, read-only, where `command: --config /etc/radiance/radiance.yaml`
names it. The configuration is plain radiance YAML: `radiance --config tp2.radiance.yaml --check`
parses and validates it without serving (with `RADIANCE_API_KEY` set, since the file names it),
and Compose substitutes nothing in it, so the port is the
`server.port` the file states (8000), which the compose file's health check asks too. The API key
stays in the container's environment, read through `server.api-key-env`, and never appears in the
configuration. An entry's name under `models:` is the model id a request gives in `model`. Delete
the entries of the containers you do not have; any other model is added as an entry with its keys
from the table in [`README.md`](../README.md#run-without-docker) (a `memory:` key holds for the
whole file). To change a setting, edit the model's entry.

The entries carry no `priority`, so the model used least recently gives way to the one a request
asks for. Priorities are strict: a model never displaces an idle one of higher priority, so with
priorities on the small models MiniMax-Music3 and Hunyuan3D-2 waited out `scheduler.load-timeout`.
On `tp1.yaml`, Flash-Next pins a 42 GiB expert pool in host memory and the fleet keeps 8 GiB of the
host free; on a host with 61 GB of memory it is unloaded when another model loads and answered its
next request after 17 to 19 s, loaded from its container again (its entry's `prefix-cache-mib: 1024`
keeps its host pins within that host). Qwen-Image's `max-model-len` in both files keeps its cache
grant within the cards beside the other models ([`README.md`](../README.md#run-without-docker)).

Copy the directory to the serving host and fill in `.env` from `.env.example`. It sets the image,
the models directory (mounted read-only at `/models`), the API key, the user and GPU
groups the server runs as, and the card `tp1.yaml` runs on. The models directory must be on a
local filesystem that does direct I/O, such as ext4, XFS or btrfs: Qwen3.8-Flash-Next reads its
n-gram table from the container that way. On btrfs with compression on, extents stored compressed
are read through the page cache instead, which works but loses the benefit of direct reads; a
container copied into a directory set `chattr +C` is stored uncompressed. Then:

```sh
docker compose -f tp2.yaml up -d      # start; restarts with the Docker daemon
docker compose -f tp2.yaml logs -f    # follow the log
docker compose -f tp2.yaml down       # stop
```

`tp1.yaml` takes the same commands. The two files listen on the same port, so run one at a time:
`down` one before `up` on the other. The server runs as the host user given in `.env`. `/health`,
`/ping` and the dashboard page answer without the API key, and the health check fetches `/health`
with busybox's `wget`.

A container serving MiniMax-H3 also serves ComfyUI's Radiance MiniMax node
(`integrations/comfyui/ComfyUI-Radiance-MiniMax`), a client of `/v1/videos`. With
`lora-dir: /models/loras/h3` in the model's entry, the node's adapter lists show the files under
`$RADIANCE_MODELS/loras/h3`.
