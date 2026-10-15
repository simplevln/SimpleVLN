# Evaluation data

This package contains the complete evaluation inputs used by the supplied
SimpleVLN configurations. It contains no training episodes.

| Dataset | Split | Episodes | Scenes |
|---|---|---:|---:|
| R2R-CE | `val_unseen` | 1,839 | 11 |
| RxR-CE, English guide | `val_unseen` | 3,669 | 11 |
| Total | | 5,508 | 11 shared scenes |

The two episode files are byte-for-byte copies of the existing evaluation inputs.
The RxR reference file contains exactly the 3,669 matching records from the full
11,006-record guide ground truth; each retained record is unchanged. This removes
unused reference records without changing evaluation episodes or nDTW targets.
RxR covers both `en-IN` and `en-US` guide instructions.

## Layout

```text
vln_data/
├── eval_data/
│   ├── r2r/val_unseen/val_unseen.json.gz
│   └── rxr/val_unseen/
│       ├── val_unseen_guide_en.json.gz
│       └── val_unseen_guide_gt.json.gz
├── manifest.json
├── required_scenes.txt
├── SHA256SUMS
├── NOTICE.md
├── LICENSE-CC-BY-4.0.txt
└── scene_datasets/v1/tasks/mp3d/    # Prepared locally; excluded from Git
    └── <scene>/
        ├── <scene>.glb
        ├── <scene>.navmesh
        ├── <scene>.house
        └── <scene>_semantic.ply
```

Episode scene paths are relative, such as `mp3d/<scene>/<scene>.glb`. The YAML
configurations prepend `VLN_DATA_ROOT/scene_datasets/v1/tasks`. No paths from the
machine that prepared this package are stored in the annotations or manifest.
There are no pre-rendered image folders: Habitat renders observations from scenes.

## Prepare the scenes

Obtain the Habitat-format Matterport3D assets through the
[official access process](https://niessner.github.io/Matterport/#download).
The [VLN-CE setup guide](https://github.com/jacobkrantz/VLN-CE#scenes-matterport3d)
describes the Habitat archive. This package records hashes for the Habitat release
with the February 2020 texture update.

Once the scene archive is extracted, run from the `eval/` directory:

```bash
python scripts/prepare_eval_data.py --mp3d-root /path/to/mp3d
```

The supplied directory must contain `<scene>/<scene>.glb` and its companion files.
The script **copies** only the 11 required scene directories into the default data
root, verifies checksums, and leaves the source files in place. Existing matching
files are reused; a conflicting file produces an error instead of being overwritten.
The scene assets total 2,370,614,435 bytes (about 2.37 GB).

For a data location outside the checkout:

```bash
python scripts/prepare_eval_data.py \
  --mp3d-root /path/to/mp3d \
  --data-root /path/to/eval-data
```

This copies both the bundled annotations and the required scene assets. Set
`VLN_DATA_ROOT=/path/to/eval-data` when evaluating from that location.

## Validate

From the `eval/` directory, check the bundled annotation files:

```bash
python scripts/prepare_eval_data.py --check-only --annotations-only
```

After preparing scene assets, check the complete data root:

```bash
python scripts/prepare_eval_data.py --check-only
```

Both commands also accept `--data-root /path/to/eval-data`. The full check verifies
all scene-file hashes in `manifest.json`; it does not launch Habitat or a model.
For the small package alone, `sha256sum -c SHA256SUMS` can also be run from this
directory. The gzip files are read directly by the evaluator; do not unpack them.

## Distribution

The annotation package is included with the evaluation repository. Scene files
are separate because Matterport3D has its own access and redistribution terms,
including recipient acceptance requirements for substantial dataset distribution.
See [NOTICE.md](NOTICE.md) for the upstream sources and terms. The evaluation directory's
`.gitignore` excludes locally prepared scene files.
