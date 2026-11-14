#!/usr/bin/env python3
"""
Fast cached dataset export for multimodal models.

All VLN dataset images are 640x480. This script:
1. Skips all image disk I/O (returns a pre-built dummy image)
2. Skips fetch_image resize (dimensions are already correct)
3. Skips image_processor tensor creation (returns pre-computed grid_thw)

Token lengths only depend on image dimensions, not pixel content.

Usage:
    MAX_PIXELS=1003520 python fast_cache_dataset.py \
        --model /path/to/model \
        --custom_dataset_info /path/to/dataset_info.json \
        --dataset 'R2R' \
        --dataset_num_proc 32 \
        --to_cached_dataset true \
        --output_dir ./cached_datasets/R2R
"""
import os
import math
from PIL import Image

# --- Configuration ---
# All VLN images are 640x480
IMG_W, IMG_H = 640, 480
PATCH_SIZE = 16
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE  # 32

# smart_resize(480, 640, factor=32) → (480, 640), no change needed
# grid_thw per image = [1, 480//16, 640//16] = [1, 30, 40]
# tokens per image = 1 * 30 * 40 // 4 = 300
GRID_H = IMG_H // PATCH_SIZE  # 30
GRID_W = IMG_W // PATCH_SIZE  # 40

_DUMMY_IMAGE = Image.new('RGB', (IMG_W, IMG_H), (128, 128, 128))


# --- Patch 1: skip image loading ---
def _fast_load_image(image):
    if isinstance(image, Image.Image):
        return image
    return _DUMMY_IMAGE.copy()


import swift.template.vision_utils
swift.template.vision_utils.load_image = _fast_load_image

import swift.template.base
swift.template.base.load_image = _fast_load_image


# --- Patch 2: skip fetch_image resize ---
def _fast_fetch_image(ele, image_patch_size=14):
    return _DUMMY_IMAGE.copy()


import qwen_vl_utils.vision_process
qwen_vl_utils.vision_process.fetch_image = _fast_fetch_image
import qwen_vl_utils
qwen_vl_utils.fetch_image = _fast_fetch_image

# Also patch in the swift qwen template module
import swift.template.templates.qwen as qwen_template
if hasattr(qwen_template, 'fetch_image'):
    qwen_template.fetch_image = _fast_fetch_image


# --- Patch 3: skip image_processor tensor creation ---
import torch

_SINGLE_GRID_THW = torch.tensor([[1, GRID_H, GRID_W]])


class _FastImageProcessorOutput(dict):
    """Minimal output that only contains image_grid_thw."""
    pass


def _make_fast_processor(original_processor):
    def _fast_call(images=None, **kwargs):
        if images is None:
            return _FastImageProcessorOutput({'image_grid_thw': _SINGLE_GRID_THW})
        n = len(images) if isinstance(images, list) else 1
        return _FastImageProcessorOutput({
            'image_grid_thw': _SINGLE_GRID_THW.repeat(n, 1)
        })
    return _fast_call


# Patch the image processor class __call__ at the class level
try:
    from transformers import Qwen2VLImageProcessor
    Qwen2VLImageProcessor.__call__ = lambda self, *a, **kw: _make_fast_processor(self)(*a, **kw)
except ImportError:
    pass

try:
    from transformers.models.qwen2_5_vl.image_processing_qwen2_5_vl import Qwen2_5_VLImageProcessor
    Qwen2_5_VLImageProcessor.__call__ = lambda self, *a, **kw: _make_fast_processor(self)(*a, **kw)
except ImportError:
    pass

try:
    from transformers.models.qwen2_5_vl.image_processing_qwen2_5_vl_fast import Qwen2_5_VLImageProcessorFast
    Qwen2_5_VLImageProcessorFast.__call__ = lambda self, *a, **kw: _make_fast_processor(self)(*a, **kw)
except ImportError:
    pass


# --- Run export ---
from swift.pipelines.export.cached_dataset import export_cached_dataset
export_cached_dataset()
