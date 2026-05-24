import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import datasets
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer

from mirage.metrics.citef1 import f1_score
from data_adapter import load_prediction, load_reference


# DECORD_EOF_RETRY_MAX=10240
# TODO: this is a temp fix, should figure out better solution if possible 
os.environ["DECORD_EOF_RETRY_MAX"] = "20480"
import logging 
logger = logging.getLogger(__name__)

#TODO: write the descriptions and kwargs for the metric 
_DESCRIPTION = """\
Info F1 Metric 
"""

_KWARGS_DESCRIPTION = """
Info F1 Metrics


Args:
    

Returns:
    
"""

_CITATION = """\

"""

class CiteF1(evaluate.Metric):
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
        pred_claims_to_videos,
        pred_videos_to_sentences,
        ref_claims_to_videos,
        ref_videos_to_claims,
        eval_type: str = 'reference',
        text_scorer: str = "Qwen/Qwen2.5-7B-Instruct",
        video_scorer: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    ):
        
        score = f1_score(
            predicted_contexts=predictions,
            predicted_claims=pred_claims_to_videos,
            predicted_videos_to_sentences=pred_videos_to_sentences,
            reference_contexts=references,
            reference_claims=ref_claims_to_videos,
            reference_videos_to_claims=ref_videos_to_claims,
            eval_type=eval_type,
            text_scorer=text_scorer,
            video_scorer=video_scorer,
        )
        
        return score
        
        
    

def parse_args():
    parser = argparse.ArgumentParser(description="Info F1 Metric")
    parser.add_argument(
        '--prediction',
        type=str,
        default='data/wikivideo/model_preds/qwen3_llm_only_relevant_citations.json',
        help='Path to the prediction file'
    )
    parser.add_argument(
        '--eval_type',
        choices=['reference', 'collection'],
        default='reference',
        help='Type of evaluation to perform. \
            Reference evaluates precision against the reference text. \
            Collection evaluates precision against the video collection.'
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
    os.makedirs(args.output_dir, exist_ok=True)
    return args

def main():
    args = parse_args()

    prediction_data = load_prediction(args.prediction, args.video_dir)
    reference_data = load_reference(args.reference, args.video_dir)

    """
    mappings needed:
        (1) reference claims and videos that could support that claim 
        (2) reference videos to the claims they support
        (2) videos cited in the prediction and the sentences that cite them
        (3) predicted claims and the videos that support them 
    """
    # mapping 1: reference claims to supporting videos
    ref_claims_to_videos = {} # {'topic': {'claim1': [video1, video2], 'claim2': [video3, video4]}}
    ref_videos_to_claims = {} # {'topic': {'video1': [claim1, claim2], 'video2': [claim3, claim4]}}
    for topic in reference_data:
        if not topic in ref_claims_to_videos:
            ref_claims_to_videos[topic] = {}
        if not topic in ref_videos_to_claims:
            ref_videos_to_claims[topic] = {}
        claim_to_supporting_videos = reference_data[topic]['claims_to_supporting_videos']
        
        for claim in claim_to_supporting_videos:
            supporting_videos = claim_to_supporting_videos[claim]['supporting_videos']
            if len(supporting_videos) > 0:
                path_to_videos = [os.path.join(args.video_dir, vid +'.mp4') for vid in supporting_videos]
                ref_claims_to_videos[topic][claim] = path_to_videos

                for vid in supporting_videos:
                    video_name = os.path.join(args.video_dir, vid +'.mp4')
                    if video_name not in ref_videos_to_claims[topic]:
                        ref_videos_to_claims[topic][video_name] = []
                    if claim not in ref_videos_to_claims[topic][video_name]:
                        ref_videos_to_claims[topic][video_name].append(claim)

    # mapping 2: videos cited in the prediction and the sentences that cite them
    # mapping 3: predicted claims and the videos that support them
    pred_videos_to_sentences = {} # {'topic': {'video1': [sentence1, sentence2], 'video2': [sentence3, sentence4]}}
    pred_claims_to_videos = {} # {'topic': {'claim1': [video1, video2], 'claim2': [video3, video4]}}
    for topic in prediction_data:
        if not topic in pred_videos_to_sentences:
            pred_videos_to_sentences[topic] = {}
        if not topic in pred_claims_to_videos:
            pred_claims_to_videos[topic] = {}
        
        claims = prediction_data[topic]['claims']
        citations = prediction_data[topic]['citations']
        sentences = prediction_data[topic]['sentences']

        for i, sentence in enumerate(sentences):
            sentence_claims = claims[i]
            sentence_citations = citations[i]
            # mapping 2
            for citation in sentence_citations:
                if citation not in pred_videos_to_sentences[topic]:
                    pred_videos_to_sentences[topic][citation] = []
                pred_videos_to_sentences[topic][citation].append(sentence)
            # mapping 3
            for claim in sentence_claims:
                if claim not in pred_claims_to_videos[topic]:
                    pred_claims_to_videos[topic][claim] = []
                for citation in sentence_citations:
                    if citation not in pred_claims_to_videos[topic][claim]:
                        pred_claims_to_videos[topic][claim].append(citation)


    # load the metric
    predictions_inputs = {
        "predictions": [],
        "references": [],
        "pred_claims_to_videos": [],
        "pred_videos_to_sentences": [],
        "ref_claims_to_videos": [],
        "ref_videos_to_claims": [],
        "topics": [],
    }
    for topic in prediction_data:
        instance_pred_videos_to_sentences = pred_videos_to_sentences[topic]
        instance_pred_claims_to_videos = pred_claims_to_videos[topic]
        instance_ref_claims_to_videos = ref_claims_to_videos[topic]
        instance_ref_videos_to_claims = ref_videos_to_claims[topic]
        prediction = prediction_data[topic]['prediction']
        reference = reference_data[topic]['article']
        predictions_inputs['predictions'].append(prediction)
        predictions_inputs['references'].append(reference)
        predictions_inputs['pred_claims_to_videos'].append(instance_pred_claims_to_videos)
        predictions_inputs['pred_videos_to_sentences'].append(instance_pred_videos_to_sentences)
        predictions_inputs['ref_claims_to_videos'].append(instance_ref_claims_to_videos)
        predictions_inputs['ref_videos_to_claims'].append(instance_ref_videos_to_claims)
        predictions_inputs['topics'].append(topic)
    
    if args.model_name == 'qwen_72b':
        text_verifier = 'Qwen/Qwen2.5-72B-Instruct'
        video_verifier = 'Qwen/Qwen2.5-VL-72B-Instruct'
    elif args.model_name == 'qwen_7b':
        text_verifier = 'Qwen/Qwen2.5-7B-Instruct'
        video_verifier = 'Qwen/Qwen2.5-VL-7B-Instruct'
    else:
        text_verifier = 'Qwen/Qwen2.5-3B-Instruct'
        video_verifier = 'Qwen/Qwen2.5-VL-3B-Instruct'
    
    citef1 = CiteF1()
    scores = citef1.compute(
        predictions=predictions_inputs['predictions'],
        references=predictions_inputs['references'],
        pred_claims_to_videos=predictions_inputs['pred_claims_to_videos'],
        pred_videos_to_sentences=predictions_inputs['pred_videos_to_sentences'],
        ref_claims_to_videos=predictions_inputs['ref_claims_to_videos'],
        ref_videos_to_claims=predictions_inputs['ref_videos_to_claims'],
        eval_type=args.eval_type,
        text_scorer=text_verifier,
        video_scorer=video_verifier,
    )

    # print(f"Scores: {scores}")
    # map scores back to the topics they correspond to and write them to the output 
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
    output_file_name = f"{pred_stem}_{args.eval_type}_cite_f1_scores_{args.model_name.replace('/', '-')}.json"
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, output_file_name), 'w') as f:
        json.dump(output_scores, f, indent=4)
    print(f"wrote to {os.path.join(args.output_dir, output_file_name)}")

    
if __name__ == "__main__":
    main()
        
        

