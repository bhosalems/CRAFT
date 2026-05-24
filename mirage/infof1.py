import os
import json
import argparse

from mirage.metrics.infof1 import f1_score
from mirage.prompts import CLAIM_VERIFICATION_TEXT_USER_PROMPT
from data_adapter import load_prediction, load_reference
from typing import List, Dict, Any, Tuple

import datasets
import evaluate

_DESCRIPTION = """\
Info F1 Metric 
"""

_KWARGS_DESCRIPTION = """
Info F1 Metrics

"""

_CITATION = """\

"""

class InfoF1(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Value("string", id="sequence"),
                    }
                )
            ],
            codebase_urls=[""],
            reference_urls=[
                "",
                "",
            ],
        )
    
    def _compute(
        self, 
        predictions,
        references,
        predicted_claims=None,
        reference_claims=None,
        relevant_videos: List[str] = None,
        eval_type: str = "reference",
        text_scorer: str = "Qwen/Qwen2.5-7B-Instruct",
        video_scorer: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        prompt: str = CLAIM_VERIFICATION_TEXT_USER_PROMPT,
    ):
        
        scores = f1_score(
            predicted_claims=predicted_claims,
            predicted_contexts=predictions,
            reference_claims=reference_claims,
            reference_contexts=references,
            relevant_videos=relevant_videos,
            eval_type=eval_type,
            text_scorer=text_scorer,
            video_scorer=video_scorer,
            user_prompt=prompt,
        )
        return scores
    

def parse_args():
    parser = argparse.ArgumentParser(description="Info F1 Metric")
    parser.add_argument(
        '--eval_type',
        choices=['reference', 'collection'],
        default='reference',
        help='Type of evaluation to perform. \
            Reference evaluates precision against the reference text. \
            Collection evaluates precision against the video collection.'
    )
    parser.add_argument(
        '--prediction',
        type=str,
        default='data/wikivideo/model_preds/qwen3_llm_only_relevant_citations.json',
        help='Path to the prediction file'
    )
    parser.add_argument(
        '--reference',
        type=str,
        default='data/wikivideo/human_eval_subset.json',
        help='Path to the reference file'
    )
    parser.add_argument(
        '--video_dir',
        type=str,
        default='/exp/amartin/wikivideo/all_videos',
        help='Path to the videos'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/wikivideo/model_preds/metric_outputs',
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='qwen_7b',
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='/exp/amartin/models/LLMs',
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    prediction_json = load_prediction(args.prediction, args.video_dir)
    reference_json = load_reference(args.reference, args.video_dir)

    relevant_videos = {}
    for instance in reference_json:
        videos = reference_json[instance]['videos']
        relevant_videos[instance] = videos

    infof1 = InfoF1()
    scores = {}
    
    predictions_inputs = {
        "predictions": [],
        "references": [],
        "predicted_claims": [],
        "reference_claims": [],
        "relevant_videos": [],
        "topics": [],
    }

    for instance in prediction_json: 
        predictions_inputs['topics'].append(instance)
        reference_instance = reference_json[instance]
        videos = relevant_videos[instance]

        if args.eval_type == 'collection':
            relevant_video_paths = []
            for video_id in relevant_videos[instance]:
                video_path = os.path.join(args.video_dir, f"{video_id}.mp4")
                if os.path.exists(video_path):
                    relevant_video_paths.append(video_path)
                else:
                    print(f"Warning: Video {video_path} does not exist.")

        prediction = prediction_json[instance]['prediction']
        predicted_claims = []
        for sent_claims in prediction_json[instance]['claims']:
            for claim in sent_claims:
                predicted_claims.append(claim)

        
        reference = reference_instance['article']
        reference_claims = []
        claim_support_notions = reference_instance['claims_to_supporting_videos']
        for claim in claim_support_notions:
            if len(claim_support_notions[claim]['supporting_videos']) > 0:
                reference_claims.append(claim)

         
        predictions_inputs['predictions'].append(prediction)
        predictions_inputs['references'].append(reference)
        predictions_inputs['predicted_claims'].append(predicted_claims)
        predictions_inputs['reference_claims'].append(reference_claims)
        if args.eval_type == 'collection':
            predictions_inputs['relevant_videos'].append(relevant_video_paths)


    if args.model_name == 'qwen_72b':
        text_verifier = 'Qwen/Qwen2.5-72B-Instruct'
        video_verifier = 'Qwen/Qwen2.5-VL-72B-Instruct'
    elif args.model_name == 'qwen_7b':
        text_verifier = 'Qwen/Qwen2.5-7B-Instruct'
        video_verifier = 'Qwen/Qwen2.5-VL-7B-Instruct'
    else:
        text_verifier = 'Qwen/Qwen2.5-3B-Instruct'
        video_verifier = 'Qwen/Qwen2.5-VL-3B-Instruct'

    scores = infof1.compute(
        predictions=predictions_inputs['predictions'],
        references=predictions_inputs['references'],
        predicted_claims=predictions_inputs['predicted_claims'],
        reference_claims=predictions_inputs['reference_claims'],
        relevant_videos=predictions_inputs.get('relevant_videos', None),
        text_scorer=text_verifier,
        video_scorer=video_verifier,
        eval_type=args.eval_type,
    )

    output_scores = {}
    for i, topic in enumerate(predictions_inputs['topics']):
        output_scores[topic] = scores[i]

    def _stats(key):
        vals = [s[key] for s in output_scores.values() if s.get(key) is not None]
        return (sum(vals) / len(vals) if vals else None), len(vals)
    p_mean, p_n = _stats("precision")
    r_mean, r_n = _stats("recall")
    f_mean, f_n = _stats("f1")
    output_scores["average"] = {
        "precision":   p_mean,
        "recall":      r_mean,
        "f1":          f_mean,
        "num_samples": {"precision": p_n, "recall": r_n, "f1": f_n},
    }

    pred_stem, _ = os.path.splitext(os.path.basename(args.prediction))
    output_file_name = f"{pred_stem}_{args.eval_type}_info_f1_scores_{args.model_name.replace('/', '-')}.json"
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, output_file_name), 'w') as f:
        json.dump(output_scores, f, indent=4)
        
    print(f"wrote to {os.path.join(args.output_dir, output_file_name)}")


if __name__ == "__main__":
    main()
        
        

