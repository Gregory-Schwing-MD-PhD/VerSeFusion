# Containers

VerSeFusion runs inside a Singularity (Apptainer) container on the Warrior
HPC.  Its runtime deps (`nibabel`, `numpy`, `pyyaml`, `scikit-learn`, `tqdm`,
`requests`) are a strict subset of CTSpinoPelvic1K's, so **by default the
same container image runs both repos**.

## Recommended: `make hpc-pull` (mirrors CTSpinoPelvic1K's pattern)

From the project root on Warrior:

```bash
make hpc-pull-slurm        # submit as a SLURM job  (≈ 5–15 min, 16G RAM)
#  or
make hpc-pull              # run on the login node
```

What it does (`slurm/hpc_pull.sh` → `scripts/hpc_pull.sh`):

1. Scrubs leaky host env vars (`JAVA_HOME`, `LD_LIBRARY_PATH`, `PYTHONPATH`,
   `R_LIBS*`) so they don't corrupt the pull.
2. Activates the same conda env CTSpinoPelvic1K uses
   (`${HOME}/mambaforge/envs/nextflow`) for the `singularity` binary.
3. Sets up per-job `SINGULARITY_TMPDIR`, `XDG_RUNTIME_DIR`, and the
   shared `NXF_SINGULARITY_CACHEDIR=${HOME}/singularity_cache`.
4. Runs `singularity pull --force containers/versefusion.sif <image>`.
5. Self-checks: execs the SIF and imports `nibabel`, `numpy`, `yaml`,
   `sklearn`, `tqdm`, `requests`.  Bails out if any are missing.

Outputs:

```
containers/versefusion.sif       ← the SIF every slurm/*.sh wrapper finds
logs/hpc_pull_<jobid>.out / .err  ← job logs
```

## What image does it actually pull?

By default, the **CTSpinoPelvic1K image**:

```
docker://gregoryschwingmdphd/ctspinopelvic1k:latest
```

…because that's what's already published and known-good for these deps.
It lands at `containers/versefusion.sif` (not `ctspinopelvic1k.sif`) so the
rest of the VerSeFusion pipeline finds it via the default
`CONTAINER_SIF` path without any further config.

When a dedicated VerSeFusion image exists, flip one env var:

```bash
SOURCE_REPO=versefusion make hpc-pull-slurm
#  or
SOURCE_REPO=versefusion sbatch slurm/hpc_pull.sh
```

All three knobs:

| env var          | default                  | description                          |
|------------------|--------------------------|--------------------------------------|
| `DOCKERHUB_USER` | `gregoryschwingmdphd`    | Docker Hub namespace.                |
| `SOURCE_REPO`    | `ctspinopelvic1k`        | `ctspinopelvic1k` or `versefusion`.  |
| `TAG`            | `latest`                 | Pin to e.g. `v0.1.0` for prod runs.  |

## Alternative: symlink without pulling at all

If you already have a `ctspinopelvic1k.sif` on disk (e.g. from the
CTSpinoPelvic1K repo on the same cluster), skip the pull entirely:

```bash
cd containers/
ln -s /path/to/CTSpinoPelvic1K/containers/ctspinopelvic1k.sif versefusion.sif
```

The SLURM scripts look up `containers/versefusion.sif` via the
`CONTAINER_SIF` env var; they don't care that it's a symlink.  To find
where your existing SIF lives:

```bash
find ~ -name 'ctspinopelvic1k*.sif' 2>/dev/null
find /scratch -name '*.sif' 2>/dev/null | grep -i ctspino
```

## What VerSeFusion needs at runtime

| package        | used by                                                  |
|----------------|----------------------------------------------------------|
| `nibabel`      | reorient, manifest, nifti, label_crosswalk, qc_overview   |
| `numpy`        | nifti, label_crosswalk, manifest, tests                   |
| `pyyaml`       | label_crosswalk (loads `configs/label_scheme.yaml`)       |
| `scikit-learn` | splits (`StratifiedKFold`)                                |
| `tqdm`         | download (progress bars)                                  |
| `requests`     | download (S3 fetcher)                                     |

Optional (not pulled into the SIF — install on host for these):

| package      | needed for                                       |
|--------------|--------------------------------------------------|
| `matplotlib` | `scripts/qc_overview.py`  — `pip install -e .[qc]` |
| `pytest`/`ruff`/`mypy` | dev: `pip install -e .[dev]`           |

All six required packages are already in the CTSpinoPelvic1K container,
which is why `make hpc-pull` "just works" with the default source.

## Local development without a container

```bash
pip install -e .             # core only
pip install -e ".[qc,dev]"   # + matplotlib + test/lint tooling
make download                # runs against host Python
```

The SLURM wrappers also work locally if `CONTAINER_SIF` is unset and
the scripts are invoked directly (not via `sbatch`) — they fall back to
whatever Python is on `PATH`.

## License

The container image inherits its base-image license and adds only this
repo's MIT-licensed code.  The CC-BY-SA 4.0 license on the VerSe data
does **not** attach to the container itself.
