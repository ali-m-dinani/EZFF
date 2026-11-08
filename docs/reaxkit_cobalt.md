# ReaxKit cobalt/LAMMPS smoke runner

`reaxkit_runner.py` implements the `reaxkit-external-runner-v1` file contract
inside EZFF's own repository. ReaxKit launches its Python executable; neither
project imports the other. LAMMPS runs as a separate executable. No datasets,
engine binaries, or results are distributed with this example.

## Scientific scope

This is an explicitly selected **cobalt cluster/dimer subset**, not the complete
training set. Only the ENERGY groups `Co_clusters` and `Co2_cluster` are used.
For the tested cobalt dataset this means 18 target expressions, 19 structures,
and 12 parameters. The command requires `--acknowledge-cobalt-subset`.

The scalar objective is `sqrt(sum(((prediction-reference)/tolerance)**2))`.
It is a dimensionless weighted Euclidean norm, **not RMSE** or the unrooted sum
of squares. Each predicted energy expression uses the trainset's explicit
divisors. LAMMPS `units real` supplies kcal/mol directly, without an eV conversion.
Do not compare these scores directly to AMS, JAX, or PIBO native scores.

Structures are evaluated without geometry relaxation, with nonperiodic fixed
boundaries and 20 angstrom padding around each coordinate extent. The BGF cell
is deliberately not used as a periodic cell in this isolated-cluster profile.
Charges start at zero and are equilibrated with `qeq/reaxff`, tolerance `1e-6`.
`run 0` performs no integration steps. The force field must contain one cobalt
atom type and one Co-Co bond entry; params supports section 2 (32 values) and
section 3 (16 values), row 1, with explicit min/max ordering. Unsupported inputs
are rejected rather than silently approximated.

The original parameter vector is evaluated first and retained in EZFF's history.
Then `ONEPLUSONE_SO` or `RANDOMSEARCH_SO` runs serially. With population 2 and one
epoch, there are three evaluations total, including the baseline. This is only
a smoke test, not evidence of convergence. Set `seed: null`: reproducible RNG
forwarding has not been implemented. Multiple replicas are not certified as
reproducible independent trials. Pool execution, custom templates, other
algorithms/objectives, and restart are deliberately unsupported in this runner.

Every candidate keeps its rendered field, parameters, named target predictions,
unrounded loss, geometry energies, and LAMMPS logs under `native/evaluations/`.
`cobalt-run.json` records the profile. Candidate failure or timeout aborts the
runner with an error; no fabricated penalty is reported as a successful result.
LAMMPS warnings also require review. Existing results cannot be overwritten by
restarting this runner; prepare a fresh campaign instead.

EZFF's normal `results/<epoch>/errors`, recommended fields, and evaluated NPZ
archives remain available to ReaxKit's existing converter. That converter does
not yet extract this runner's labeled candidate reports. Its generic engine and
objective descriptors are not proof of cross-method comparability; consult
`cobalt-run.json` and `evaluations/*/report.json` for the exact profile.

## Environment

Use a separate `envs/ezff` environment. EZFF's README documents the separate
MOBOpt installation, which is currently missing from `install_requires` despite
the unconditional import. Therefore `pip check` alone is insufficient; verify
`import mobopt; import ezff` and save `pip freeze` for the environment used.
No dependency upgrade is required merely to add this runner.

## Roar smoke procedure

After the runner has been committed/pushed and pulled on Roar, from the campaign
parent containing `reaxkit`, `EZFF`, and the cobalt input files:

```bash
cp -n EZFF/examples/reaxkit-cobalt/settings.yaml cobalt_ezff_smoke.yaml
cp -n EZFF/examples/reaxkit-cobalt/job.sh ezff_job.sh
```

Edit the YAML command paths if your campaign parent differs from the example.
The job requests one node, one task, and one CPU; **no genoa constraint** is used.
Keep the worker contract line and `exec bash "$1"`. ReaxKit, not this template
directly, handles submission.

```bash
source reaxkit/.venv/bin/activate
bash -n ezff_job.sh
reaxkit optimize-ffield --settings cobalt_ezff_smoke.yaml --validate
reaxkit optimize-ffield --settings cobalt_ezff_smoke.yaml --dry-run
(
  set -e
  module purge
  module load gcc/13.4.0 openmpi/5.0.7 lammps/20250722
  envs/ezff/bin/python EZFF/reaxkit_runner.py \
    cobalt_ezff_smoke/runs/ezff/replica-0001/run-config.json \
    --acknowledge-cobalt-subset --check-only
)
```

`--check-only` validates inputs and executable discovery, not engine execution.
After reviewing the staged configuration and successful check output:

```bash
reaxkit optimize-ffield --submit ./cobalt_ezff_smoke
reaxkit optimize-ffield --status ./cobalt_ezff_smoke
```

Submit once. After completion, collect and download the whole campaign for
scientific/output inspection; do not put results in Git.

## Developer checks

```bash
python -m unittest discover -s tests -p test_reaxkit_runner.py -v
```

Tests use synthetic inputs and mocked engine calls. A separate local Windows
LAMMPS/EZFF validation also completed baseline plus two ONEPLUSONE candidates,
writing the native epoch archive and recommended field. This is **not** Roar
validation; Roar submission and runtime evidence are still required.
