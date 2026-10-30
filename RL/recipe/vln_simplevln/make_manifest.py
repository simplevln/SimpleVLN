"""Build Parquet episode manifests from the datasets served by the environment."""

import argparse
import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def load_annotation(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)['episodes']


def episode_rows(episodes: list[dict], dataset_name: str) -> list[dict]:
    rows = []
    keys = set()
    for episode in episodes:
        episode_id = str(episode['episode_id'])
        scene_id = str(episode['scene_id'])
        key = (Path(scene_id).name.split('.')[0], episode_id)
        if key in keys:
            raise ValueError(f'Duplicate episode in {dataset_name}: {key}')
        keys.add(key)
        instruction = episode.get('instruction', '')
        if isinstance(instruction, dict):
            instruction = instruction.get('instruction_text', '')
        rows.append({
            'data_source': 'vln',
            'prompt': [{'role': 'user', 'content': 'placeholder'}],
            'agent_name': 'vln_full_episode_agent',
            'extra_info': {
                'episode_id': episode_id,
                'scene_id': scene_id,
                'instruction': str(instruction),
                'dataset_name': dataset_name,
            },
            # Habitat reset needs poses, goals and reference_path. They travel in this
            # one parquet; the environment service writes a temporary json.gz at startup.
            'episode': json.dumps(episode, ensure_ascii=False),
        })
    return rows


def write_manifest(rows: list[dict], output: Path):
    if not rows:
        raise ValueError('Cannot create an empty episode manifest')
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output)
    print(f'{output}: {len(rows)} episodes')


def episodes_to_parquet(episodes: list[dict], out_path: Path, dataset_name: str):
    write_manifest(episode_rows(episodes, dataset_name), out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config', type=Path, help='Unified environment YAML; merge all datasets into one manifest')
    source.add_argument('--data-dir', type=Path, help='Legacy R2R directory containing <split>/<split>.json.gz')
    parser.add_argument('--splits', nargs='+', default=['train'])
    parser.add_argument('--dataset-name', help='Routing key for a single --data-dir split')
    parser.add_argument('--out-dir', type=Path, default=Path('data/manifests'))
    parser.add_argument('--output', type=Path, help='Output filename for --config')
    args = parser.parse_args()
    if args.config:
        if args.dataset_name:
            parser.error('--config takes dataset names from the YAML')
        from omegaconf import OmegaConf
        config = OmegaConf.load(args.config)
        specs = config.get('datasets')
        if not specs:
            parser.error('The config must contain a nonempty datasets map')
        rows = []
        for name, spec in specs.items():
            path = Path(spec.data_path.format(split=spec.get('split', 'train')))
            rows.extend(episode_rows(load_annotation(path), name))
        write_manifest(rows, args.output or args.out_dir / 'vln_train.parquet')
    else:
        if args.output:
            parser.error('--output is used with --config; use --out-dir for --data-dir')
        if args.dataset_name and len(args.splits) != 1:
            parser.error('Use one split with --dataset-name to avoid routing validation episodes to training data')
        for split in args.splits:
            name = args.dataset_name or f'r2r_{split}'
            episodes = load_annotation(args.data_dir / split / f'{split}.json.gz')
            episodes_to_parquet(episodes, args.out_dir / f'vln_{name}_{len(episodes)}.parquet', name)


if __name__ == '__main__':
    main()
