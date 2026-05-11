// =============================================================================
// VerSeFusion — Nextflow pipeline
//
// Mirrors the SLURM-shell entry points one-for-one as DAG nodes so the
// whole pipeline can be run as a single Nextflow invocation on the
// Warrior HPC.  Each process pulls in the Singularity container declared
// in nextflow.config.
//
// Usage:
//     nextflow run nextflow/main.nf \
//         -profile slurm,singularity \
//         --raw_dir       /path/to/data/raw \
//         --unified_dir   /path/to/data/unified \
//         --reoriented_dir /path/to/data/reoriented \
//         --hf_dir        /path/to/data/hf_export
// =============================================================================

nextflow.enable.dsl = 2

// ---------- parameter defaults -----------------------------------------------
params.raw_dir         = "${projectDir}/../data/raw"
params.unified_dir     = "${projectDir}/../data/unified"
params.reoriented_dir  = "${projectDir}/../data/reoriented"
params.hf_dir          = "${projectDir}/../data/hf_export"
params.prefer          = "verse20"
params.n_folds         = 5
params.seed            = 20260511
params.holdout         = "verse20_test"


// =============================================================================
// processes
// =============================================================================

process DOWNLOAD {
    cpus 4
    memory '8 GB'
    time '8h'
    publishDir params.raw_dir, mode: 'copy', overwrite: false

    output:
    path "download_manifest.json", emit: manifest

    script:
    """
    python -m verse_pipeline.download \\
        --out_dir ${params.raw_dir} \\
        --log_level INFO
    cp ${params.raw_dir}/download_manifest.json .
    """
}

process UNIFY {
    cpus 2
    memory '8 GB'
    time '1h'
    publishDir params.unified_dir, mode: 'copy', overwrite: true

    input:
    path download_manifest

    output:
    path "unify_manifest.json", emit: manifest

    script:
    """
    python -m verse_pipeline.unify \\
        --raw_dir ${params.raw_dir} \\
        --out_dir ${params.unified_dir} \\
        --prefer  ${params.prefer} \\
        --mode    symlink
    cp ${params.unified_dir}/unify_manifest.json .
    """
}

process REORIENT {
    cpus 4
    memory '16 GB'
    time '2h'
    publishDir params.reoriented_dir, mode: 'copy', overwrite: true

    input:
    path unify_manifest

    output:
    path "reorient_manifest.json", emit: manifest

    script:
    """
    python -m verse_pipeline.reorient \\
        --in_dir  ${params.unified_dir} \\
        --out_dir ${params.reoriented_dir}
    cp ${params.reoriented_dir}/reorient_manifest.json .
    """
}

process MANIFEST {
    cpus 2
    memory '8 GB'
    time '30m'
    publishDir params.reoriented_dir, mode: 'copy', overwrite: true

    input:
    path reorient_manifest

    output:
    path "placed_manifest.json", emit: manifest

    script:
    """
    python -m verse_pipeline.manifest \\
        --in_dir         ${params.reoriented_dir} \\
        --out_path       ${params.reoriented_dir}/placed_manifest.json \\
        --unify_manifest ${params.unified_dir}/unify_manifest.json
    cp ${params.reoriented_dir}/placed_manifest.json .
    """
}

process SPLITS {
    cpus 1
    memory '4 GB'
    time '15m'
    publishDir "${params.reoriented_dir}/splits", mode: 'copy', overwrite: true

    input:
    path placed_manifest

    output:
    path "cv_5fold.json"
    path "test.json"

    script:
    """
    python -m verse_pipeline.splits \\
        --manifest ${params.reoriented_dir}/placed_manifest.json \\
        --out_dir  ${params.reoriented_dir}/splits \\
        --n_folds  ${params.n_folds} \\
        --seed     ${params.seed} \\
        --holdout  ${params.holdout}
    cp ${params.reoriented_dir}/splits/*.json .
    """
}

process HF_EXPORT {
    cpus 2
    memory '8 GB'
    time '1h'
    publishDir params.hf_dir, mode: 'copy', overwrite: true

    input:
    path placed_manifest

    output:
    path "dataset_card.md"

    script:
    """
    python -m verse_pipeline.hf_export \\
        --in_dir  ${params.reoriented_dir} \\
        --out_dir ${params.hf_dir} \\
        --mode    copy
    cp ${params.hf_dir}/dataset_card.md .
    """
}


// =============================================================================
// workflow
// =============================================================================

workflow {
    DOWNLOAD()
    UNIFY(DOWNLOAD.out.manifest)
    REORIENT(UNIFY.out.manifest)
    MANIFEST(REORIENT.out.manifest)
    SPLITS(MANIFEST.out.manifest)
    HF_EXPORT(MANIFEST.out.manifest)
}
