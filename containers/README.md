# Containers

VerSeFusion runs inside a Singularity (Apptainer) container on the Warrior HPC.
The image is built and published on Docker Hub at:

    docker://gschwing/versefusion:v0.1.0

## Pull on Warrior HPC

```bash
cd containers/
module load singularity      # or apptainer, depending on cluster config
singularity pull --force versefusion.sif docker://gschwing/versefusion:v0.1.0
```

The resulting `versefusion.sif` is what every `slurm/*.sh` script binds
in via `singularity exec --bind ...`.  All SLURM scripts read the path
from the env variable `CONTAINER_SIF` (default
`<repo>/containers/versefusion.sif`).

## Image contents

The container is built from `python:3.11-slim` and installs:

  * the system stack: `git`, `wget`, `ca-certificates`, `build-essential`
  * the Python deps in `requirements.txt`
  * this package itself (`pip install -e .`)

A `Dockerfile` is intentionally **not** kept in this repo — the build is
maintained in a separate `gschwing/versefusion-container` repository so
that changes to the runtime environment can be versioned independently of
the pipeline code.  Pin the tag (`v0.1.0`, `v0.2.0`, ...) in
`configs/default.env` to lock the runtime against a known-good build.

## Local development

For local dev you can skip the container entirely:

```bash
pip install -e ".[dev]"
make download
```

The SLURM wrappers still work locally if `CONTAINER_SIF` is unset and
the scripts are invoked directly (not through `sbatch`) — they degrade to
running Python from the active environment.

## License

The container image inherits the upstream `python:3.11-slim` license and
adds only this repo's MIT-licensed code.  The CC-BY-SA 4.0 license on
the VerSe data **does not** attach to the container.
