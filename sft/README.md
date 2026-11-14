# SFT

We use [ms-swift](https://github.com/modelscope/ms-swift) to conduct SFT.

## Environment Setup

Please refer to [ms-swift](https://github.com/modelscope/ms-swift) for more details on installing.
We recommend using the official Docker image.

## SFT Training

We use [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) as our foundation model.
Qwen2.5-VL / Qwen3-VL / Qwen3.5 are also supported.

Please follow the official ms-swift documentation for training.
Training data is available at [honglyhly/SimpleVLN_SFT_Data](https://huggingface.co/datasets/honglyhly/SimpleVLN_SFT_Data).

### 1. Cache Training Data

Edit these two assignments in [cache/cache_dataset.sh](cache/cache_dataset.sh):

```bash
MODEL="/path/to/models/Qwen3-VL-4B-Instruct"
CACHE_DIR="/path/to/cache_data"
```

Converted JSONs are read from `${CACHE_DIR}/annotations/` by default; set
`JSON_DIR` to override. Use absolute paths. Use a separate cache root when
changing tokenizer, template, or image processor settings. These paths are
assigned inside the script; edit the assignments directly.

Cache all five datasets, or select individual datasets:

```bash
bash cache/cache_all.sh

# Alternatively, prepare stage-1 inputs only:
# bash cache/cache_dataset.sh SFT/R2R SFT/RxR
```

`cache_all.sh` calls `cache_dataset.sh` with all five dataset names. The latter
only exports the cache: DAgger JSONs must already be converted with
`--mode atomic`, and every conversion uses `--max_history 8 --seed 0`.


The outputs are:

| Dataset | Converted JSON, relative to `CACHE_DIR` | Cache, relative to `CACHE_DIR` |
|---|---|---|
| SFT/R2R | `annotations/SFT/R2R.json` | `SFT/R2R/train/` |
| SFT/RxR | `annotations/SFT/RxR.json` | `SFT/RxR/train/` |
| SFT/ScaleVLN | `annotations/SFT/ScaleVLN.json` | `SFT/ScaleVLN/train/` |
| DAgger/R2R | `annotations/DAgger/R2R.json` | `DAgger/R2R/train/` |
| DAgger/RxR | `annotations/DAgger/RxR.json` | `DAgger/RxR/train/` |

The cache stores conversations, image paths, and sequence lengths. Training
still loads the images. Generated JSON files contain local image paths, so the
same absolute image paths must be accessible on every training node. Rebuild
the inputs after relocating the images. Use a fresh cache output directory
when regenerating: ms-swift rejects existing export directories by default.

The fast exporter assumes **640×480 images and patch size 16**. Qwen2.5 uses
patch size 14 and trains from the converted JSONs directly.




