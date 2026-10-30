# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Habitat extensions for StreamVLN
from . import measures

__all__ = ["measures"]  # side-effect import: registers @registry.register_measure classes