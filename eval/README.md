# SimpleVLN Evaluation

Evaluation suite for **SimpleVLN** on the **R2R-CE** and **RxR-CE** `val-unseen`
benchmarks. Habitat provides the simulation environment, and vLLM serves the
SimpleVLN policy through an OpenAI-compatible API. The evaluation pipeline
supports parallel rollouts, distributed model serving, and resumable experiments.

[Installation](#installation) · [Data](#data-preparation) · [Evaluation](#evaluation) · [Metrics](#results-and-resume)

## Installation

The Habitat evaluator and vLLM server use separate Python environments.

### Habitat

**Python 3.10 · Habitat 0.3.3**

```bash
conda create -n simplevln-eval python=3.10 -y
conda activate simplevln-eval
conda install habitat-sim=0.3.3 headless -c conda-forge -c aihabitat
python -m pip install -r eval/requirements.txt
```

Habitat Sim must be installed before the Python dependencies. Habitat Lab and
Baselines are installed from the upstream `v0.3.3` Git tag. This version's
structured configurations require Python 3.10.

### vLLM

Install vLLM 0.18.0 in a separate environment. This release serves Qwen2.5-VL, Qwen3-VL, and Qwen3.5.

```bash
python -m pip install -r eval/requirements-vllm.txt
```

The [vLLM 0.18.0 GPU installation guide](https://docs.vllm.ai/en/v0.18.0/getting_started/installation/gpu/) covers the CUDA build.

## Data Preparation

The evaluation annotations are included in `eval/data/vln_data/`.

| Benchmark | Split | Episodes |
|---|---|---:|
| R2R-CE | `val_unseen` | 1,839 |
| RxR-CE, English guide | `val_unseen` | 3,669 |

The two benchmarks share **11 Matterport3D scenes**. The package also includes
RxR reference trajectories for nDTW and sDTW, scene identifiers, and checksums.
Annotations are loaded directly in `.json.gz` format; no extraction is needed.

Scene assets are subject to the
[Matterport3D Terms of Use](https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf).
Prepare them using either option below. The evaluator uses `eval/data/vln_data/`
as its default data location.

### From a scene archive

If you have a scene archive packaged with `scene_datasets/` at its root, extract
it into the data directory and verify all annotations and scene assets:

```bash
tar -xzf /path/to/mp3d_val_unseen.tar.gz -C eval/data/vln_data
python eval/scripts/prepare_eval_data.py --data-root eval/data/vln_data --check-only
```

The extracted layout must be
`eval/data/vln_data/scene_datasets/v1/tasks/mp3d/<scene>/`, containing
`<scene>.glb`, `<scene>.navmesh`, `<scene>.house`, and `<scene>_semantic.ply`.
The archive should have no additional enclosing directory.

### From an existing MP3D installation

Habitat-format assets can also be obtained through the
[Matterport3D access process](https://niessner.github.io/Matterport/#download).
Copy the required scenes from your local MP3D directory:

```bash
python eval/scripts/prepare_eval_data.py --mp3d-root /path/to/mp3d
```

The source directory should contain `<scene>/<scene>.glb` and its companion
files. The script copies only the 11 required scenes into `eval/data/vln_data/`
and verifies their checksums. Source files remain in place.

<details>
<summary>Custom data locations and validation</summary>

Prepare annotations and scenes at a custom location:

```bash
python eval/scripts/prepare_eval_data.py \
  --mp3d-root /path/to/mp3d \
  --data-root /path/to/eval-data
```

Set `VLN_DATA_ROOT` in the evaluation environment to use that location:

```bash
export VLN_DATA_ROOT=/path/to/eval-data
```

Validate the bundled annotations:

```bash
python eval/scripts/prepare_eval_data.py --check-only --annotations-only
```

Omit `--annotations-only` to verify scene assets as well. Use `--data-root` to
check a custom location. Full directory specifications and data terms are
provided in the [data guide](data/vln_data/README.md) and
[dataset notice](data/vln_data/NOTICE.md).

</details>

## Evaluation

### Model Serving

Launch a SimpleVLN checkpoint in the vLLM environment:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/path/to/SimpleVLN_Qwen3VL_4B_RL \
bash eval/scripts/serve_vllm.sh 1 1
```

`MODEL_PATH` points to the Hugging Face model directory containing the weights,
configuration, tokenizer, and processor. The server becomes available at
`http://localhost:22001/v1` after its health check passes.

The launcher arguments specify **GPU count** and **tensor-parallel size per
server**. For example, `8 1` launches eight model replicas, while `8 2` launches
four replicas with two GPUs each.

<details>
<summary>Multi-GPU serving</summary>

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/path/to/SimpleVLN_Qwen3VL_4B_RL \
bash eval/scripts/serve_vllm.sh 8 1
```

The eight services listen on ports `22001`–`22008`. Set `BASE_PORT` to change the
first port and `MODEL_NAME` to customize the served model name (default: `qwen3vl`).
Server logs and process IDs are recorded under `eval/logs/vllm/run.*/`.

</details>

### Benchmark Evaluation

With the model service running, launch evaluation in the Habitat environment:

```bash
conda activate simplevln-eval
EXP_NAME=simplevln_qwen3vl_4b \
VLLM_NUM_SERVERS=1 \
SPLIT_NUM=8 \
PASS_K=1 \
bash eval/scripts/eval.sh
```

This configuration evaluates both benchmarks with **eight simulator workers**
and **one trial per episode**, using the single model service above. R2R-CE and
RxR-CE are evaluated sequentially. Adjust `SPLIT_NUM` to the available simulator
resources and `VLLM_NUM_SERVERS` to the number of model services.

<details>
<summary>Evaluation options</summary>

| Variable | Launcher default | Description |
|---|---|---|
| `EXP_NAME` | Required | Experiment name for result directories |
| `VLLM_HOST` | `http://localhost` | Model server host |
| `VLLM_PORT_START` | `22001` | First server port |
| `VLLM_NUM_SERVERS` | `8` | Number of consecutive server ports |
| `VLLM_MODEL_NAME` | `qwen3vl` | Served model name |
| `SPLIT_NUM` | `64` | Simulator workers |
| `PASS_K` | `4` | Trials per episode |
| `MAX_ACTION_HISTORY` | `200` | Retained observation-history window |
| `RESULT_ROOT` | `./results` | Output root, relative to the evaluation module |

The example above overrides the worker count, service count, and number of
trials. Use a distinct `EXP_NAME` for each checkpoint and evaluation configuration.

</details>

### Remote Inference

Specify full endpoint URLs to evaluate against services on other hosts or ports:

```bash
EXP_NAME=simplevln_remote \
VLLM_BASE_URL="http://SERVER_A:8001/v1,http://SERVER_B:9001/v1" \
SPLIT_NUM=8 \
PASS_K=1 \
bash eval/scripts/eval.sh
```

`VLLM_BASE_URL` accepts one URL or a comma-separated list and overrides the host,
port range, and service count. HTTPS endpoints are supported. The evaluator's
`VLLM_MODEL_NAME` must match the server's `MODEL_NAME`.

### Single-Benchmark Evaluation

The Python entry point supports an individual benchmark configuration:

```bash
VLN_DATA_ROOT="${VLN_DATA_ROOT:-$PWD/eval/data/vln_data}" \
python eval/vln/eval.py \
  --exp-config eval/config/vln_r2r.yaml \
  --vllm_base_url http://localhost:22001/v1 \
  --split-num 8 \
  --pass-k 1 \
  --max-action-history 200 \
  --result-path eval/results/simplevln_r2r
```

Use `eval/config/vln_rxr.yaml` for RxR-CE and a separate output directory.
The `--save-video` option saves trajectory GIFs for completed trials.

### Evaluation Settings

| Setting | Value |
|---|---|
| Policy observations | Egocentric RGB |
| Image resolution | 640 × 480 |
| Camera field of view | 90° |
| Forward step | 0.25 m |
| Turn angle | 15° |
| Success threshold | 3 m |
| Historical observations per request | Up to 8, uniformly sampled |
| Decoding | Temperature 0.3, top-p 0.95, maximum 512 output tokens |

The current observation is provided alongside the sampled history. On the first
decision, the initial frame occupies both the history and current-observation
slots. The evaluator does not select a separate local-history frame.

The RxR configuration uses a 90° camera rather than the standard 79° setup;
include the camera setting when reporting results. Action arguments must match
the simulator configuration: `FORWARD_DISTANCE=25` cm and `TURN_ANGLE=15` degrees.

## Results and Resume

Evaluation writes per-trial records and aggregate metrics to:

```text
eval/results/simplevln_qwen3vl_4b/
├── r2r_pass1/result.json
└── rxr_pass1/result.json
```

Each file uses JSON Lines format, with one object per line. The aggregate summary
is also printed to the terminal.

| Field | Metric |
|---|---|
| `sucs_all` | Mean success rate (SR) |
| `spls_all` | Mean SPL |
| `oss_all` | Mean oracle success rate |
| `ones_all` | Mean final navigation error, in meters |
| `length` | Mean traveled distance, in meters |
| `steps_avg` | Mean step count |
| `ndtw`, `sdtw` | Mean nDTW and sDTW; enabled for RxR |
| `pass@k` | Fraction of episodes with at least one successful trial |

Rates are stored on a 0–1 scale. Use `sucs_all` for standard SR and multiply by
100 to report percentages. `pass@k` measures success across multiple attempts.
The supplied R2R configuration does not enable nDTW or sDTW.

Rerunning an experiment with the same configuration resumes completed trials.
Incomplete records are evaluated again, and aggregation retains the last complete
record for each trial. Failed navigation trials contribute to the reported metrics.
If a worker fails, completed records are preserved and no partial-run summary is
written. Nonfinite SPL/NE values are retained in aggregation and may produce
NaN/Infinity in the summary.

## Tests

```bash
python -m unittest discover -s eval/tests -v
```

The 20 regression checks cover data preparation, navigation inputs and actions,
model serving, result aggregation, and resume. They use mock services and
simulators; real Habitat execution and checkpoint performance are evaluated
through the pipeline above.
