# VLN trajectory and instruction generation

Generate navigation trajectories in MP3D and HM3D, label them with a VLM,
and output separate R2R, RxR and ScaleVLN datasets as `rl*.json.gz` files.

## Requirements and inputs

Use Linux with a working rendering GPU and Habitat-Sim, numpy, scipy,
numpy-quaternion, opencv-python and requests installed.

Set `VLN_DATA` to a data directory containing:

```text
scene_datasets/
  mp3d/<scene>/<scene>.glb
  hm3d/train/<scene>/*.basis.glb
datasets/
  r2r/train/train.json.gz
  rxr/train/train_guide_en.json.gz
  scalevln/ScaleVLN/scalevln_subset_150k.json.gz
```

Include the corresponding MP3D semantic assets and HM3D navigation meshes.
RxR also accepts `train_guide.json.gz`, filtering its references to English.
Set `SCALEVLN_REFS` to override the ScaleVLN reference file; it accepts an
absolute path or a path relative to `VLN_DATA/datasets`.
Reference corpora must use the Habitat-style JSON format with an `episodes` list,
`scene_id`, and `instruction.instruction_text`.

## Run

For instruction generation, configure a Qwen3.5 VLM endpoint that accepts images
through the chat-completions API. Set `VLM_MODEL` to the model identifier exposed
by your provider or local deployment; `qwen3.5` below is an example identifier.

```bash
export VLN_DATA=/path/to/data
export VLM_API_KEY=your_key
export VLM_BASE_URL=https://your-provider.example/v1/chat/completions
export VLM_MODEL=qwen3.5
bash run.sh
```

`VLM_BASE_URL` must be the complete chat-completions endpoint URL.
To run individual stages:

```bash
bash run.sh mp3d          # Sample R2R and RxR trajectories
bash run.sh hm3d          # Sample ScaleVLN trajectories
bash run.sh label         # Label and assemble all three datasets
bash run.sh label r2r     # Label and assemble one dataset
```

`PAR` controls MP3D sampling parallelism (default 16), `WORKERS` controls
HM3D sampling and labeling workers (default 32), and `CONC` controls concurrent
VLM requests per labeling worker (default 8).
`SEED` controls MP3D sampling (default 100); `HM3D_SEED` controls HM3D sampling (default 0).

## Prompt and style references

All three datasets share `prompt.txt`. Each instruction uses up to eight unique
reference instructions from the same scene and its own dataset:

| Output | Style references | Visual input |
| --- | --- | --- |
| R2R | R2R train | RGB and semantic information |
| RxR | English RxR train | RGB and semantic information |
| ScaleVLN | ScaleVLN corpus | RGB |

Reference sentences guide expression style; instructions describe the sampled route.
Missing ScaleVLN reference instructions cause labeling to stop with an error.

## Optional step-count ratios

Without ratio settings, trajectories are sampled randomly without step-bin quotas.
Each dataset defaults to a target of 100 trajectories per scene.
Configure each dataset independently:

```bash
export R2R_PER_SCENE=100
export RXR_PER_SCENE=100
export SCALEVLN_PER_SCENE=100
export R2R_STEP_RATIOS="30,40,20,10"
export RXR_STEP_RATIOS="10,20,30,25,15"
export SCALEVLN_STEP_RATIOS="30,40,20,10"
bash run.sh
```

Ratios correspond to 0-49, 50-99, 100-149, ... non-STOP actions, including turns.
At most eight bins are supported; MP3D's final bin also includes 400 steps.
A zero weight or an omitted trailing bin receives zero quota.
Weights are normalized automatically, and integer quotas sum to the per-scene target.
For example, 101 trajectories with weights `30,40,20,10` gives quotas `30,41,20,10`.
Unset a ratio variable or set it empty to select random sampling for that dataset.

The physical action sizes are 0.25 m forward and 15 degrees per turn.
Validity and deduplication filters apply in both sampling modes. Scene geometry and
attempt limits can prevent quotas from filling, so achieved counts may be lower.
Direct CLI options are `sample.py mp3d --step-ratios` and `sample.py hm3d --bin-ratios`.

## Outputs

Final output directories under `VLN_DATA/datasets`:

- `r2r_rl_final/`
- `rxr_rl_final/`
- `scalevln_rl_final/`

Each contains `rl_all.json.gz` and its matching `rl_all_gt.json.gz`.
R2R and RxR also retain per-level `rl.json.gz` and `rl_gt.json.gz` files;
ScaleVLN writes these files at its output root.
Episode and trajectory IDs are retained during instruction assembly.
The three datasets have independent ID spaces; GT keys match the string form of episode IDs.
Existing instruction tokens are rebuilt after text changes using one vocabulary per dataset.
Assembly rejects duplicate episode IDs within a dataset, missing GT and empty instructions.

Override final directories with `R2R_OUT`, `RXR_OUT` and `SCALEVLN_OUT`.
If changing sampling output directories with `MP3D_OUT` or `HM3D_OUT`, also set
`R2R_SRC`, `RXR_SRC` or `SCALEVLN_SRC` to the corresponding labeling input directory.

## Running again

Use fresh working and output directories for independent generation runs.
Labeling records progress and can be retried with `bash run.sh label`.
The runner stops if scene loading or labeling fails; successful records are kept for retry.
Manual assembly may retain nonempty source instructions for records without a generated instruction.

HM3D sampling checkpoints each completed scene. On interruption, rerun with the same
settings and worker count: completed scenes are retained, and the unfinished scene is
regenerated. Outputs without checkpoints or with changed sampling settings require a
new output directory. Step quotas remain targets; reaching the attempt limit completes
that scene with however many valid trajectories were found.

Navigation graphs in `nav_graphs` are cached separately by scene and seed. Use a fresh
cache directory if scene assets or navigation-graph parameters change. Use fresh labeling
outputs if input trajectories, prompts or model settings change, to avoid reusing old progress.

## References

- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
- [VLN-CE](https://github.com/jacobkrantz/VLN-CE)
- [R2R / Matterport3D Simulator](https://github.com/peteanderson80/Matterport3DSimulator)
- [RxR](https://github.com/google-research-datasets/RxR)
- [ScaleVLN](https://github.com/wz0919/ScaleVLN)

Obtain input corpora and scene assets separately under their respective terms.
