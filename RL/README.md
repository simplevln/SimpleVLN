# SimpleVLN Reinforcement Learning

Online reinforcement learning for SimpleVLN with full navigation episodes,
trajectory-level advantages, and DAPO dynamic sampling. The policy interacts with
Habitat through an environment service; the bundled [verl](https://github.com/verl-project/verl)
runtime handles FSDP training and vLLM rollouts.

The [training guide](recipe/vln_simplevln/README.md) covers installation, data,
environment services, multi-node launch, checkpoint export, and CPU tests.
Presets are included for Qwen3-VL 4B/8B and Qwen3.5 4B/9B.

- [Training recipe](recipe/vln_simplevln/)
- [Dataset configuration](recipe/vln_simplevln/config/vln_datasets.yaml)
- [Shared launcher](recipe/vln_simplevln/run_dapo.sh)
- [verl documentation](https://verl.readthedocs.io/en/latest/)

## Acknowledgments

Built on [verl](https://github.com/verl-project/verl),
[Habitat](https://github.com/facebookresearch/habitat-lab),
[VLN-CE](https://github.com/jacobkrantz/VLN-CE), and
[vLLM](https://github.com/vllm-project/vllm).
Upstream copyright notices, [LICENSE](LICENSE), and [Notice.txt](Notice.txt)
are retained. Scene assets remain subject to their respective dataset terms.
