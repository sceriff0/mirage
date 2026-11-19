# Install

<p class="standfirst">MIRAGE needs three things on your machine: Java, Nextflow and a container
engine (Docker on a laptop or workstation, Apptainer/Singularity on a cluster). Every analysis tool
comes pre-packaged in a container, so you install none of them yourself.</p>

!!! tip "First time?"
    [Your first run](quickstart.md) walks through the same setup step by step, starting
    from an empty computer, and ends with the pipeline running on your slides.

## Requirements

| Requirement | Version / detail | Notes |
|---|---|---|
| **Nextflow** | `>=25.04.0` | Uses the `nf-schema@2.5.1` plugin; older versions will not run. |
| **Java** | 17 or newer | Required by Nextflow. Check with `java -version`. |
| **Container engine** | Docker, or Apptainer/Singularity | Docker on a laptop or workstation; Apptainer/Singularity on a cluster. |
| **Processor** | x86-64 (Intel or AMD) | Every image is built for `linux/amd64` only. See [macOS](#macos) for Apple silicon. |
| **Free disk** | ~40 GB with Docker, ~20 GB with Apptainer/Singularity | The images of a default run are about 18 GB compressed; Docker unpacks them to about twice that. Results and the work directory come on top. |
| **Network access on first run** | outbound access to the Nextflow plugin registry, Docker Hub and the Hugging Face Hub | Nextflow fetches `nf-schema` the first time the pipeline runs. On a network-isolated compute node, pre-provision it — see [Offline / air-gapped execution](hpc.md#offline-air-gapped-execution). |

!!! warning "Nextflow version matters"
    MIRAGE requires Nextflow **25.04.0 or later**. If `nextflow -version` reports
    something older, run `nextflow self-update` before doing anything else.

## Java and Nextflow (every platform)

**Java** — Nextflow runs on it. [SDKMAN](https://sdkman.io/) installs it without
administrator rights:

```bash
curl -s https://get.sdkman.io | bash
source "$HOME/.sdkman/bin/sdkman-init.sh"
sdk install java
```

**Nextflow** — the program that runs the pipeline:

```bash
curl -s https://get.nextflow.io | bash
chmod +x nextflow
mkdir -p "$HOME/.local/bin" && mv nextflow "$HOME/.local/bin/"
export PATH="$HOME/.local/bin:$PATH"
```

Add the last line to your `~/.bashrc` or `~/.zshrc` so `nextflow` is found in every new
terminal. An older Nextflow updates in place with `nextflow self-update`.

Then install the container engine for your platform.

## Linux

Install [Docker Engine](https://docs.docker.com/engine/install/) (or Docker Desktop) and
let your user run it without `sudo`:

```bash
sudo usermod -aG docker "$USER"   # then log out and back in
docker run --rm hello-world
```

Run with `-profile local,docker` (or your own `-c site.config` on a larger machine). With
an NVIDIA GPU, segmentation can use it — see [GPU or no GPU](gpu.md); without one, add
`-params-file params/no_gpu.json`.

## macOS

Install [Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/) and
start it. In **Settings → Resources**, give it as much memory as your Mac can spare
(8 GB at the very least), so segmentation and registration are not killed for lack of
memory. A Mac has no NVIDIA GPU, so always add `-params-file params/no_gpu.json`.

=== "Intel Mac"

    Nothing else to do: the images run natively.

=== "Apple silicon (M1 and later)"

    The images are built for Intel/AMD (`linux/amd64`) only. The `docker` profile passes
    `--platform linux/amd64`, so Docker Desktop runs them under emulation. On an M2 Mac
    with Docker Desktop's QEMU emulation, conversion and STARE registration ran, but
    **StarDist segmentation crashed (exit code 139)**, so a run does not finish.

    You can try turning on **Settings → General → "Use Rosetta for x86_64/amd64
    emulation on Apple Silicon"** in Docker Desktop. **This has not been tested with
    MIRAGE**; if you try it, please report the result on the
    [issue tracker](https://github.com/sceriff0/mirage/issues).

    The reliable options are an Intel or AMD Linux machine or workstation, or a cluster
    ([HPC cluster](#hpc-cluster) below).

## Windows (WSL 2)

MIRAGE runs inside the Windows Subsystem for Linux:

1. Install [WSL 2](https://learn.microsoft.com/windows/wsl/install) with its default
   Ubuntu distribution (`wsl --install` in an administrator PowerShell, then restart).
2. Install [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
   and, in its settings, keep **"Use the WSL 2 based engine"** on and enable integration
   with your Ubuntu distribution (**Settings → Resources → WSL integration**).
3. Open the **Ubuntu** terminal and follow the [Java and Nextflow](#java-and-nextflow-every-platform)
   and [Linux](#linux) steps there (Docker is already provided by Docker Desktop, so skip
   installing Docker Engine).

Keep your slides and the `mirage` folder inside the Linux file system (for example under
`~/`); they can be copied from Windows through `\\wsl$\Ubuntu\home\<you>` in File Explorer.

## HPC cluster

Clusters usually provide Java as a module and Apptainer or Singularity instead of Docker:

```bash
module load java          # the module name varies by site
java -version             # 17 or newer
singularity --version     # or: apptainer --version
```

Install only Nextflow, as above, in your home directory. Apptainer uses the same
`singularity` profile. Then follow [Running on a cluster](hpc.md): make a `site.config`
with your partition, account and job size, and point the image cache at shared storage.

## Get MIRAGE

```bash
git clone https://github.com/sceriff0/mirage.git
cd mirage
```

All commands on this site assume you run them from the repo root (`nextflow run .`).

## Verify

Confirm Nextflow is present and new enough:

```bash
nextflow -version
```

You should see version **25.04.0 or later**. A quick stub run validates the whole wiring
without running the real tools:

```bash
nextflow run . -profile test,docker -stub --outdir results_stub
```

If that completes without `FAILED` processes, your installation is sound. A real run on
synthetic data is described in [Your first run → No slides at hand?](quickstart.md#no-slides-at-hand).

## Next steps

Run MIRAGE on your slides: **[Your first run](quickstart.md)**. To describe your slides in
full detail, see **[The samplesheet](samplesheet.md)**.
