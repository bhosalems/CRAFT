"""Merge per-shard {scores,frames,meta}.shard_<i>_of_<N>.json files produced by
feature_extract_folder.py into single scores.json / frames.json / meta.json.

Reconstructs the original pair order: shard i held pairs[i::N], so interleaving
the shards in round-robin reproduces the contiguous list.
"""
import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output_file', type=str, required=True, help='same as feature_extract_folder.py --output_file')
    p.add_argument('--dataset_name', type=str, default='magmar2026')
    p.add_argument('--extract_feature_model', type=str, required=True, help='clip/mclip/blip/sevila')
    p.add_argument('--num_shards', type=int, required=True)
    p.add_argument('--keep_shards', action='store_true', help='do not delete shard files after merge')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.join(args.output_file, args.dataset_name, args.extract_feature_model)

    shard_scores = []
    shard_frames = []
    shard_meta = []
    shard_files = []

    for i in range(args.num_shards):
        sfx = f'.shard_{i}_of_{args.num_shards}'
        sp = os.path.join(out_dir, f'scores{sfx}.json')
        fp = os.path.join(out_dir, f'frames{sfx}.json')
        mp = os.path.join(out_dir, f'meta{sfx}.json')
        if not (os.path.exists(sp) and os.path.exists(fp) and os.path.exists(mp)):
            raise FileNotFoundError(f'shard {i} files missing under {out_dir}: expected {sp}, {fp}, {mp}')
        with open(sp) as f: shard_scores.append(json.load(f))
        with open(fp) as f: shard_frames.append(json.load(f))
        with open(mp) as f: shard_meta.append(json.load(f))
        shard_files.extend([sp, fp, mp])

    merged_scores, merged_frames, merged_meta = [], [], []
    n_max = max(len(s) for s in shard_scores)
    for k in range(n_max):
        for i in range(args.num_shards):
            if k < len(shard_scores[i]):
                merged_scores.append(shard_scores[i][k])
                merged_frames.append(shard_frames[i][k])
                merged_meta.append(shard_meta[i][k])

    out_scores = os.path.join(out_dir, 'scores.json')
    out_frames = os.path.join(out_dir, 'frames.json')
    out_meta = os.path.join(out_dir, 'meta.json')
    with open(out_scores, 'w') as f: json.dump(merged_scores, f)
    with open(out_frames, 'w') as f: json.dump(merged_frames, f)
    with open(out_meta, 'w') as f: json.dump(merged_meta, f)
    print(f'[merge] wrote {len(merged_scores)} entries to:')
    print(f'  {out_scores}')
    print(f'  {out_frames}')
    print(f'  {out_meta}')

    if not args.keep_shards:
        for p in shard_files:
            os.remove(p)
        print(f'[merge] removed {len(shard_files)} shard files (pass --keep_shards to retain)')


if __name__ == '__main__':
    main()
