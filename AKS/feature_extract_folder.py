import os
import sys
# Remove this script's directory from sys.path so the local AKS/datasets/ folder
# doesn't shadow the HuggingFace `datasets` package that sentence-transformers needs.
_self_dir = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _self_dir]

import torch
from PIL import Image

import json
from decord import VideoReader
from decord import cpu
import numpy as np
import re

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract Video Feature')

    parser.add_argument('--dataset_path', type=str, default='/a2il/data/mbhosale/MAGMaR2026_test/', help='folder containing the .mp4 files')
    parser.add_argument('--topic_mapping', type=str, required=True, help='JSON mapping topic -> list of video ids (without .mp4)')
    parser.add_argument('--queries', type=str, required=True, help='JSONL file with one query per line; expected fields: query_id, title, query')
    parser.add_argument('--extract_feature_model', type=str, default='clip', help='clip/mclip/blip/sevila (mclip = multilingual CLIP text encoder + openai CLIP image encoder)')
    parser.add_argument('--output_file', type=str, default='/a2il/data/mbhosale/MAGMaR2026_test/outscores', help='path of output scores and frames')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset_name', type=str, default='magmar2026', help='tag used for the output subfolder')
    parser.add_argument('--num_shards', type=int, default=1, help='total number of parallel shards')
    parser.add_argument('--shard_id', type=int, default=0, help='this process handles pairs[shard_id::num_shards]')

    return parser.parse_args()


def _norm_title(s):
    tokens = re.split(r'[-_\s]+', s.lower())
    out = []
    for t in tokens:
        if not t or re.fullmatch(r'\d{4}', t):
            continue
        if len(t) > 3 and t.endswith('s'):
            t = t[:-1]
        out.append(t)
    return frozenset(out)


def load_pairs(queries_path, topic_mapping_path, dataset_path):
    with open(topic_mapping_path, 'r') as f:
        mapping = json.load(f)
    topic_index = {_norm_title(topic): topic for topic in mapping}

    queries = []
    with open(queries_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    pairs = []
    unmatched_titles = set()
    missing_videos = set()
    matched_queries = 0

    for q in queries:
        key = _norm_title(q['title'])
        topic = topic_index.get(key)
        if topic is None:
            unmatched_titles.add(q['title'])
            continue
        matched_queries += 1
        prompts = q.get('prompts') or [q['query']]
        for vid in mapping[topic]:
            video_path = os.path.join(dataset_path, vid + '.mp4')
            if not os.path.exists(video_path):
                missing_videos.add(video_path)
                continue
            pairs.append({
                'query_id': q.get('query_id'),
                'query_type': q.get('query_type'),
                'language': q.get('language'),
                'query_title': q['title'],
                'topic': topic,
                'video_id': vid,
                'video_path': video_path,
                'prompts': prompts,
            })

    print(f'[info] {matched_queries}/{len(queries)} queries matched a topic with videos')
    if unmatched_titles:
        print(f'[warn] {len(unmatched_titles)} query titles had no matching topic in the mapping (skipped):')
        for t in sorted(unmatched_titles):
            print('  ', t)
    if missing_videos:
        missing_sorted = sorted(missing_videos)
        print(f'[warn] {len(missing_sorted)} unique videos referenced in mapping not found on disk; skipping:')
        for m in missing_sorted:
            print('  ', m)
    return pairs


def main(args):
    device = args.device

    if args.extract_feature_model == 'clip':
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        model.to(device)
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    elif args.extract_feature_model == 'mclip':
        from transformers import CLIPProcessor, CLIPModel
        from sentence_transformers import SentenceTransformer
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        model.to(device)
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        mclip_text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1', device=device)
    elif args.extract_feature_model == 'blip':
        from lavis.models import load_model_and_preprocess
        model, vis_processors, text_processors = load_model_and_preprocess(
            'blip_image_text_matching', 'large', device=device, is_eval=True)
    elif args.extract_feature_model == 'sevila':
        from lavis.models import load_model_and_preprocess
        model, vis_processors, text_processors = load_model_and_preprocess(
            name='sevila', model_type='pretrain_flant5xl', is_eval=True, device=device)
    else:
        raise ValueError('model not supported')

    out_score_path = os.path.join(args.output_file, args.dataset_name, args.extract_feature_model)
    os.makedirs(out_score_path, exist_ok=True)

    pairs = load_pairs(args.queries, args.topic_mapping, args.dataset_path)
    total_pairs = len(pairs)
    if args.num_shards > 1:
        if not 0 <= args.shard_id < args.num_shards:
            raise ValueError(f'shard_id {args.shard_id} out of range for num_shards {args.num_shards}')
        pairs = pairs[args.shard_id::args.num_shards]
        print(f'[info] shard {args.shard_id}/{args.num_shards}: processing {len(pairs)}/{total_pairs} (query, video) pairs')
    else:
        print(f'[info] processing {total_pairs} (query, video) pairs')

    text_feature_cache = {}

    def get_clip_text_features(prompts):
        key = ('clip', tuple(prompts))
        if key not in text_feature_cache:
            inputs_text = processor(text=list(prompts), return_tensors='pt', padding=True, truncation=True).to(device)
            with torch.no_grad():
                text_out = model.text_model(**inputs_text)
                text_feature_cache[key] = model.text_projection(text_out.pooler_output)
        return text_feature_cache[key]

    def get_mclip_text_features(prompts):
        key = ('mclip', tuple(prompts))
        if key not in text_feature_cache:
            with torch.no_grad():
                emb = mclip_text_model.encode(list(prompts), convert_to_tensor=True, normalize_embeddings=False).to(device)
            text_feature_cache[key] = emb
        return text_feature_cache[key]

    scores = []
    fn = []
    meta = []

    suffix = f'.shard_{args.shard_id}_of_{args.num_shards}' if args.num_shards > 1 else ''
    score_path = os.path.join(out_score_path, f'scores{suffix}.json')
    frame_path = os.path.join(out_score_path, f'frames{suffix}.json')
    meta_path = os.path.join(out_score_path, f'meta{suffix}.json')

    for idx, item in enumerate(pairs):
        prompts = item['prompts']
        text = prompts[0]
        video = item['video_path']
        try:
            vr = VideoReader(video, ctx=cpu(0), num_threads=1)
        except Exception as e:
            print(f'[warn] failed to open {video}: {e}; skipping')
            continue
        fps = vr.get_avg_fps()
        if fps <= 0:
            print(f'[warn] fps={fps} for {video}; skipping')
            continue
        step = max(1, int(fps))
        frame_nums = max(1, len(vr) // step)

        score = []
        frame_num = []

        if args.extract_feature_model == 'blip':
            txt = text_processors['eval'](text)
            for j in range(frame_nums):
                raw_image = vr[j * step].asnumpy()
                raw_image = Image.fromarray(raw_image)
                img = vis_processors['eval'](raw_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    blip_output = model({'image': img, 'text_input': txt}, match_head='itm')
                blip_scores = torch.nn.functional.softmax(blip_output, dim=1)
                score.append(blip_scores[:, 1].item())
                frame_num.append(j * step)

        elif args.extract_feature_model in ('clip', 'mclip'):
            if args.extract_feature_model == 'clip':
                text_features = get_clip_text_features(prompts)
            else:
                text_features = get_mclip_text_features(prompts)
            text_features_n = torch.nn.functional.normalize(text_features, dim=-1)
            for j in range(frame_nums):
                raw_image = vr[j * step].asnumpy()
                raw_image = Image.fromarray(raw_image)
                inputs_image = processor(images=raw_image, return_tensors='pt', padding=True).to(device)
                with torch.no_grad():
                    image_out = model.vision_model(pixel_values=inputs_image['pixel_values'])
                    image_features = model.visual_projection(image_out.pooler_output)  # [1, 512]
                image_features_n = torch.nn.functional.normalize(image_features, dim=-1)
                sims = (text_features_n @ image_features_n.t()).squeeze(-1)  # [P]
                score.append(sims.max().item())
                frame_num.append(j * step)

        else:
            prompt = f'Question: {text}. Is this a good frame to answer this question?'
            txt = text_processors['eval'](prompt)
            for j in range(frame_nums):
                raw_image = vr[j * step].asnumpy()
                raw_image = Image.fromarray(raw_image)
                img = vis_processors['eval'](raw_image).unsqueeze(0).unsqueeze(0).to(device)
                samples = {'video': img, 'loc_input': txt}
                sevila_score = float(model.generate_score(samples).squeeze(0).squeeze(0))
                score.append(sevila_score)
                frame_num.append(j * step)

        fn.append(frame_num)
        scores.append(score)
        meta.append({
            'query_id': item['query_id'],
            'query_title': item['query_title'],
            'topic': item['topic'],
            'video_id': item['video_id'],
            'query_type': item['query_type'],
            'language': item['language'],
        })

        if (idx + 1) % 5 == 0 or idx == len(pairs) - 1:
            print(f'[info] {idx + 1}/{len(pairs)} done; latest: q{item["query_id"]} / {item["video_id"]} ({len(score)} frames)', flush=True)

    with open(frame_path, 'w') as f:
        json.dump(fn, f)
    with open(score_path, 'w') as f:
        json.dump(scores, f)
    with open(meta_path, 'w') as f:
        json.dump(meta, f)
    print(f'[done] wrote {score_path}, {frame_path}, {meta_path}')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
