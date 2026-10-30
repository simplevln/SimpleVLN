# SimpleVLN Data Processing

Tools for converting navigation trajectories into SimpleVLN training samples
and collecting additional trajectories with DAgger.

| Tool | Purpose |
|---|---|
| [convert_vln.py](tool/convert_vln.py) | Convert R2R, RxR, and ScaleVLN trajectories into training annotations |
| [convert_dagger.py](tool/convert_dagger.py) | Convert collected DAgger trajectories using executed actions |
| [dagger_simplevln.sh](scripts/dagger_simplevln.sh) | Collect R2R or RxR trajectories with Habitat and a vLLM server |
| [trajectory_making.py](tool/trajectory_making.py) | Replay EnvDrop or ScaleVLN trajectories to export camera poses and intrinsics |

Commands below assume the DataProcess directory as the working directory.

## Annotation Conversion

Conversion runs on CPU without Habitat or a model checkpoint:

```bash
python -m pip install -r requirements-convert.txt

python tool/convert_vln.py \
    --annotation_path data/raw/trajectories.jsonl \
    --image_root data/trajectories \
    --output_path data/annotations/vln.json \
    --relative_to data \
    --task_type vln --seed 0
```

Input annotations may be JSON arrays or JSONL. A trajectory contains an episode
identifier, a video directory, an instruction, and an action sequence:

```json
{
  "episode_id": 1,
  "video_id": "scene_001/trajectory_001",
  "instruction": ["Walk through the doorway."],
  "actions": [-1, 1, 1, 2, 0]
}
```

The aliases `id`, `video`, and `instructions` are also supported. An instruction
may be a string, a list of strings, or an object with `instruction_text`.
Frames are read from `<image_root>/<video_id>/`, optionally under `rgb/`.
A dataset root containing `images/` may also be used as `--image_root`.

Legacy actions begin with an initial `-1` or numeric `0` sentinel. Executed action
IDs are `0` for STOP, `1` for forward, `2` for left, and `3` for right. Forward and
turn units are 25 cm and 15°. Action sequences must be arrays; each element may
be a numeric ID, a numeric string, or action text.
Each movement needs a successor frame; only terminal STOP may reuse the last
frame. Missing or duplicate numbered frames and movement after STOP are rejected.

The default uses eight uniformly sampled history frames plus the current frame;
at the first decision, the initial frame fills both history and current slots.
`--max_history` changes the history count. `--seed` controls random action merging.
`--task_type` also accepts `idm`, `ffs`, or multiple task names.

### Portable Image Paths

`--relative_to` sets the root used for image paths in the exported JSON. For
example, an image under `data/trajectories/scene_001/000.jpg` becomes
`trajectories/scene_001/000.jpg` with `--relative_to data`. The command rejects
images outside that root.

Distribute the annotations and image directories under the same root, and
configure the training loader to resolve paths from it. Without `--relative_to`,
the converter preserves paths derived from `--image_root`.

## Trajectory Collection

Use Python 3.10 and a separate Habitat environment:

```bash
conda create -n simplevln-data python=3.10 -y
CONDA_ENV=simplevln-data bash tool/habitat.sh
conda activate simplevln-data
```

This installs Habitat Sim and Habitat Lab 0.3.3. Obtain scene assets through the
[Matterport3D](https://github.com/niessner/Matterport) and
[HM3D](https://github.com/matterport/habitat-matterport-3dresearch) access procedures.
Set the dataset root before collection:

```bash
export VLN_DATA_ROOT=/path/to/vln_data
```

The collection script expects the following files under that root:

```text
vln_data/
├── scene_datasets/mp3d/<scene>/<scene>.glb
└── eval_data/
    ├── r2r/train/{train.json.gz,train_gt.json.gz}
    └── rxr/train/{train_guide_en.json.gz,train_guide_gt.json.gz}
```

The annotations use Habitat's `R2RVLN-v1` schema. Override `R2R_DATA_PATH`,
`R2R_GT_ANNOTATIONS_PATH`, `RXR_DATA_PATH`, and `RXR_GT_ANNOTATIONS_PATH` when
using another annotation layout.

Install vLLM in its own environment following the
[vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/),
then serve the selected model:

```bash
vllm serve /path/to/checkpoint --served-model-name simplevln --port 8000
```

Run collection from the Habitat environment:

```bash
DAGGER_DATASET=R2R OUTPUT_ROOT=data/dagger \
VLLM_HOST=localhost VLLM_PORT_START=8000 VLLM_NUM_SERVERS=1 \
VLLM_MODEL_NAME=simplevln PARALLEL_ENVS=8 \
bash scripts/dagger_simplevln.sh
```

Set `DAGGER_DATASET=RxR` for RxR. The launcher also accepts `DAGGER_P`,
`DAGGER_DATA_IT`, `DAGGER_BO_N`, and `DAGGER_UPDATE_SIZE`. Use a fresh output
directory when changing the model or collection settings. Saved frames use the
release layout `<scene>/<episode_id>/rgb/000.jpg`, and the annotation `video`
field is `<scene>/<episode_id>` with no `images/` prefix. Resuming skips
completed episodes after checking their latest saved annotations and frames.
Successful episodes are committed immediately. Merging preserves previously
saved annotations, including runs where older worker shards have been removed.

Convert the collected trajectories:

```bash
python tool/convert_dagger.py \
    --annotation_path data/dagger/R2R/annotations_actions.json \
    --image_root data/dagger/R2R \
    --output_path data/annotations/dagger.json \
    --relative_to data --mode both --seed 0
```

`atomic` uses the same random merging as the standard VLN converter. `merged`
uses recorded model-call spans. Both use executed actions, including expert
interventions, as labels. `both` writes `dagger_atomic.json` and
`dagger_merged.json`; merged conversion requires `model_call_groups`.

### Camera Poses

The pose tool replays raw trajectories and writes `poses.npy` and `intrinsics.npy`
beside each trajectory's frames. It does not render an RGB dataset.

```bash
python tool/trajectory_making.py \
    --dataset scalevln \
    --data_root data/scalevln \
    --data_path data/raw/scalevln.json.gz \
    --annotations_file annotations_bak.json \
    --num_workers 1 --gpu_ids 0
```

Raw annotations must contain `id`, `video`, and sentinel-prefixed `actions`.
`poses.npy` and `intrinsics.npy` are written next to that episode's frames,
under `<data_root>/<video>/`. A leading `images/` in `video` is removed.
For ScaleVLN, HM3D assets are loaded from
`$VLN_DATA_ROOT/scene_datasets/hm3d/`. Existing pose files are skipped when their
shapes and frame counts match the annotations.

## Tests

```bash
python -m pip install -r requirements-convert.txt 'numpy<2' Pillow
python -m unittest discover -s tests -v
```

The tests cover conversion against fixed reference outputs, frame and action
alignment, portable paths, collection bookkeeping, and resume behavior. They use
synthetic trajectories and simulated services; no GPU, private data, or external
source checkout is required.
