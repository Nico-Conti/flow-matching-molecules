# Evaluate the released DeFoG MOSES samples

Run `eval_defog_moses_fcd.py` from this repository using the same Python environment
as the thesis benchmark. It calls `src/evaluate.py::_fcd` directly. No DeFoG
checkpoint, training, generation, or separate DeFoG environment is required.

From the repository root on the GPU machine:

```bash
python scripts/eval_defog_moses_fcd.py --device cuda --output-dir runs/defog_moses_fcd/batch0
```

Activate the thesis environment before calling Python directly. The existing
`runs/evaluate_published_defog_moses_fcd.sh` cluster driver also calls this Python
file directly, forwards arguments, and caps CPU threads at four by default.
Only the Python file under `scripts/` and that run driver are needed for the
cluster launch; `tests/` remains ignored and is not a runtime dependency.

The first run downloads the authors' `generated_samples.pkl` (~570 MB) and the
official random Test and scaffold TestSF CSVs into `data/defog_moses/`. Subsequent
runs reuse those downloads. Use `--cache-dir` to change the cache location.
The graph decoding/reference preparation runs on CPU; ChemNet inference runs on
the requested GPU. `--device cpu` works too; omitting `--device` selects CUDA when
available and otherwise CPU.

The environment must have the repository installed (`python -m pip install -e .`)
and a GPU-compatible PyTorch build. Check `python -c "import torch;
print(torch.cuda.is_available())"` before launching on GPU. This script needs no
Psi4/DFT, graph-tool, or model checkpoint dependencies beyond the thesis environment.

## Inputs and sample budget

By default the script evaluates the **first 25,000 generated graphs**, before
removing invalid molecules. It errors if fewer than the requested number exist.
It does not sample repeatedly until it has 25,000 valid molecules.

Use an already downloaded file, or choose a different contiguous batch:

```bash
python scripts/eval_defog_moses_fcd.py \
  --samples /absolute/path/generated_samples.pkl \
  --n-samples 25000 --offset 25000 \
  --device cuda --output-dir runs/defog_moses_fcd/batch1
```

Only use an offset that fits the saved file; the script prints `n_available`.
Batch boundaries alone do not establish independent seeds. The authors describe
these as samples from a retrained public checkpoint, not the original paper run.
Do not expect an exact reproduction of their reported FCD=1.95.

For an offline run, supply **both** reference paths as well:

```bash
python scripts/eval_defog_moses_fcd.py \
  --samples /absolute/path/generated_samples.pkl \
  --test-csv /absolute/path/test.csv.gz \
  --test-scaffolds-csv /absolute/path/test_scaffolds.csv.gz \
  --device cuda --output-dir runs/defog_moses_fcd/offline
```

Reference files must have the official `SMILES` column; uncompressed CSVs also work.
The script labels supplied files according to these flags; it cannot establish
that a renamed/custom CSV is the official split. File hashes and counts are saved.
Using `--n-samples 100` reduces the generated batch but still processes both full
reference sets; it is not a cheap end-to-end smoke test or a paper-level comparison.

## Exact evaluation protocol

### Historical filtered HF reference experiment

To measure reference-set sensitivity on the same released generated graphs, use:

```bash
GPU=3 bash runs/evaluate_published_defog_moses_fcd.sh --reference-source hf-defog
```

This loads `nico8771/moses_test_defog` (157,526 rows at the time of the audit) and
`nico8771/moses_test_scaffolds_defog` (156,176 rows) directly. These are historical
aromatic DeFoG-style `filter_dataset=True` survivor sets, not the full official
MOSES benchmark and not evidence of the original paper's FCD references.
Their HF storage partition is named `train`, but its contents are the respective
evaluation splits. No reference graph preprocessing is repeated. HF uses its normal
cache (`HF_HOME` can relocate it); `--cache-dir` controls the original CSV/pickle cache.
The generated pickle download is reused, while graph decoding and FCD run again.

The JSON records the source mode, each HF revision, row count and ordered SMILES
hash. Generated decoding, sample budget and deduplication remain unchanged, so
compare against the previous full-reference result (Test 0.803, TestSF 1.457).
Do not combine this mode with `--test-csv` or `--test-scaffolds-csv`.
Omitting the flag retains the original full-reference behavior.

### Default full-reference protocol

- Decode original **class-index** tensors using DeFoG's eight atom classes
  (C, N, S, O, F, Cl, Br, H) and explicit aromatic bond class 4. Do not decode these
  as the thesis's four-class Kekulé tensors.
- Sanitize the whole generated graph first, then keep its largest connected
  fragment. No bond repair or relaxed charge assignment. Failed molecules remain
  invalid; they are not replaced with fresh samples.
- Clean references with the thesis's `sanitize_smiles_dataset`, using MOSES atoms,
  `charge_aware=False`, and `apply_filter=False`, as in its MOSES loader. No reference
  subsampling or molecular property calculation.
- Call the existing `_fcd` on **unique valid generated SMILES**, against each full
  cleaned reference set. This reproduces the thesis's deduplication policy. It is
  not the duplicate-preserving policy in DeFoG's dormant FCD path or standard MOSES.
- Report validity and uniqueness, but not novelty (which would require train).

## Outputs

Each run writes:

- `results.json`: FCD under `fcd.Test` and `fcd.TestSF`, generated/valid/unique
  counts, reference counts and cleaning statistics, offset, source paths/URLs,
  SHA-256 hashes, dependency versions, and the evaluation protocol.
- `generated_smiles.csv`: one row per selected graph with its original sample
  index; invalid molecules have an empty SMILES. Duplicates remain in this export.

Without `--output-dir`, outputs go into a timestamped folder under
`runs/defog_moses_fcd/`. Reusing an explicit output directory replaces its output
files. A failed FCD calculation does not write a new successful `results.json`.

The public pickle loader only allows the tensor reconstruction globals used by
the release and maps tensor storage to CPU. An unsupported pickle format produces
an error rather than silently changing its interpretation.

Sources: [official repository](https://github.com/manuelmlmadeira/DeFoG#checkpoints),
[public samples](https://drive.switch.ch/index.php/s/MG7y2EZoithAywE).

## Recheck our saved MOSES samples with fresh reference statistics

`rescore_moses_fcd.py` reads the saved SMILES CSV, loads full `moses_test` and
`moses_test_scaffolds_ours` HF references, checks their full row counts, and
recomputes all ChemNet statistics in memory. It never reads or writes an FCD
statistics cache. HF dataset downloads can still use their normal cache.
It retains generated duplicates to match the historical MOSES driver; this
differs from the public-sample evaluator's generic `_fcd` deduplication policy.

Sync both Python files under `scripts/` (`rescore_moses_fcd.py` imports the existing
HF loader), and copy `runs/rescore_moses_fcd.sh` to `/data/n.conti/runs/`:

```bash
GPU=3 bash runs/rescore_moses_fcd.sh
```

The default input is
`/data/n.conti/evals/moses_raw/eval_defog_moses_2026-08-23_201213_smiles.csv`,
the eta=50, 500-step run with historical scores 0.3427 / 0.9732.
Override `SMILES_CSV` with an absolute path to check another saved run.
To select 10,000 random generated rows from that CSV before invalid removal:

```bash
GPU=3 bash runs/rescore_moses_fcd.sh --n-samples 10000 --seed 0
```

Sampling is without replacement, reproducible for the selected seed, and retains
the original order of selected rows. Duplicates remain; invalid rows count toward
the requested budget. Full references remain fixed. The JSON records the input
hash, available row count, selected counts, selection method and seed. Leave `N`
at its default 25,000: it checks the input CSV size, whereas `--n-samples` controls
the subset size. Omitting `--n-samples` rescores the entire file as before.

The input is read-only; no checkpoint, training, generation or graph reconstruction
is needed. Results, source hashes, HF revisions and dependency versions are saved
in a new timestamped `runs/moses_fcd_rescore/` directory, alongside `run.log`.

### Official MOSES scorer

Install the published scoring package once, without its legacy model-training
dependencies (the thesis environment already has the scoring dependencies):

```bash
python -m pip install --no-deps molsets==0.3.1
# Alternatively, for a uv-managed environment without pip:
uv pip install --python .venv/bin/python --no-deps molsets==0.3.1
```

After syncing `scripts/rescore_moses_fcd.py`, the existing run driver can call
the same public API as SimGFM's `check_moses.py`:

```bash
GPU=3 bash runs/rescore_moses_fcd.sh --scorer moses
```

It passes all selected CSV rows (including blank invalid entries and duplicates)
to `moses.get_all_metrics(gen=..., device=..., batch_size=..., n_jobs=...)` without
overriding train, Test, TestSF, or reference statistics. Thus it uses the official
package's bundled data/statistics, not our HF references or local .npz caches.
It computes the full metric suite, so SNN and internal diversity add runtime beyond
FCD. The `moses_metrics` JSON field contains all metrics; undefined non-FCD metrics
are saved as null and non-finite FCD still fails the run. Package version and hashes
of the bundled reference CSV/statistics files are recorded.

Two compatibility shims support newer dependencies: MOSES's retired `rdkit.six.iteritems`
import maps to dictionary iteration, and its filter-table import uses pandas.concat
in place of removed DataFrame.append. The pandas shim is removed after import.
No FCD formula, ChemNet weights, or reference statistics are modified. The shims
are listed in the JSON report. This reproduces the public API call, not SimGFM's
entire historical software environment. Use `--n-samples 10000 --seed 0` as well
only to compare against the earlier random10k result; omit these for the25k check.
