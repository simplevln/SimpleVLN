# Third-party notices

## StreamVLN

The collection code and navigation measurements include components adapted from
[StreamVLN](https://github.com/InternRobotics/StreamVLN), by Meng Wei,
Chenyang Wan, Xiqian Yu, Tai Wang, and collaborators (2025):
*Streaming Vision-and-Language Navigation via SlowFast Context Modeling*.

Affected files:

- `vln/dagger.py`, adapted from `streamvln/streamvln_dagger.py`.
- `vln/habitat_extensions/measures.py`, adapted from `streamvln/habitat_extensions/measures.py`.
- Associated Habitat collection configuration and launcher conventions.

Local adaptations include vLLM-based model calls, executed-action and model-call
recording, collection bookkeeping and resume handling, configurable input/output
paths, and Habitat 0.3.3 configuration and measurement support.

Upstream [license statement](https://github.com/InternRobotics/StreamVLN#-license):
CC BY-NC-SA 4.0. Adapted components retain those terms; see
`LICENSES/CC-BY-NC-SA-4.0.txt` and
https://creativecommons.org/licenses/by-nc-sa/4.0/.
This attribution does not imply upstream endorsement.

## Habitat

Files carrying the Meta Platforms copyright notice retain that notice and the MIT
terms in `LICENSES/MIT-Habitat.txt`.

Upstream: [Habitat-Lab](https://github.com/facebookresearch/habitat-lab),
Copyright (c) Meta Platforms, Inc. and its affiliates.

## Data and model resources

Input scenes, navigation annotations and model weights are obtained separately.
Their respective terms are not replaced by these source-code notices.
