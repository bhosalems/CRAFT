from typing import List, Dict, Any, Tuple
import logging

from mirage.prompts import (
    CLAIM_VERIFICATION_VIDEOS_SYSTEM_PROMPT, CLAIM_VERIFICATION_VIDEOS_USER_PROMPT,
    CLAIM_VERIFICATION_TEXT_SYSTEM_PROMPT, CLAIM_VERIFICATION_TEXT_USER_PROMPT
)

from mirage.models import (
    Text2TextHfScorer, Text2TextVLLMScorer,
    Text2VideoVLLMScorer, Text2VideoHfScorer
)

from mirage.scoring_functions import (
    score_claims_reference, score_claims_collection
)

logger = logging.getLogger(__name__)



def judge_inference_llm(
    predicted_claims,
    predicted_contexts,
    reference_claims,
    reference_contexts,
    eval_type: str,
    scorer: str,
    system_prompt: str = CLAIM_VERIFICATION_TEXT_SYSTEM_PROMPT,
    user_prompt: str = CLAIM_VERIFICATION_TEXT_USER_PROMPT,
) -> List[Dict[str,Any]]:
    """
    Performs all inferences where LLM inference is needed. 


    Recall always requires LLM inference, while precision may or many not
    depending on eval_type. 

    # TODO: describe args and return value
    """

    model = Text2TextVLLMScorer(model_name=scorer)
    scores = []
    for i, pred_claims in enumerate(predicted_claims):
        pred_context = predicted_contexts[i]
        ref_claims = reference_claims[i]
        ref_context = reference_contexts[i]
        precision = None
        if eval_type == "reference":
            precision = score_claims_reference(
                hypothesis_claims=pred_claims,
                hypothesis_context=pred_context,
                reference_context=ref_context,
                scorer=model,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
            )

        recall = score_claims_reference(
            hypothesis_claims=ref_claims,
            hypothesis_context=ref_context,
            reference_context=pred_context,
            scorer=model,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        
        scores.append({"precision": precision, "recall": recall})

    return scores

def judge_inference_vlm(
    predicted_claims,
    predicted_contexts,
    reference_videos,
    scorer: str,
    system_prompt: str = CLAIM_VERIFICATION_VIDEOS_SYSTEM_PROMPT,
    user_prompt: str = CLAIM_VERIFICATION_VIDEOS_USER_PROMPT,
    collection_type: str = "single_source"
):
    """
   Performs all inferences where VLM inference is needed. 
    """
    model = Text2VideoVLLMScorer(model_name=scorer)

    scores = []
    for i, pred_claims in enumerate(predicted_claims):
        pred_context = predicted_contexts[i]
        ref_videos = reference_videos[i]
        precision = score_claims_collection(
            hypothesis_claims=pred_claims,
            hypothesis_context=pred_context,
            reference_contexts=ref_videos,
            scorer=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            collection_type=collection_type,
        )
        scores.append({"precision": precision})

    return scores


def f1_score(
    predicted_claims,
    predicted_contexts,
    reference_claims,
    reference_contexts,
    relevant_videos = None,
    eval_type: str = "reference",
    text_scorer: str = "Qwen/Qwen2.5-7B-Instruct",
    video_scorer: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    user_prompt: str = CLAIM_VERIFICATION_TEXT_USER_PROMPT,
    system_prompt: str = CLAIM_VERIFICATION_TEXT_SYSTEM_PROMPT,
) -> Dict[str, float]:
    """
    Compute the info f1 score
    """

    # judge all LLM inferences for the evaluation
    judged_outputs_llm = judge_inference_llm(
        predicted_claims=predicted_claims,
        predicted_contexts=predicted_contexts,
        reference_claims=reference_claims,
        reference_contexts=reference_contexts,
        eval_type=eval_type,
        scorer=text_scorer,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    # only need to service VLM inference if eval_type is "collection"
    if eval_type == "collection":
        judged_outputs_vlm = judge_inference_vlm(
            predicted_claims=predicted_claims,
            predicted_contexts=predicted_contexts,
            reference_videos=relevant_videos,
            scorer=video_scorer,
            system_prompt=CLAIM_VERIFICATION_VIDEOS_SYSTEM_PROMPT,
            user_prompt=CLAIM_VERIFICATION_VIDEOS_USER_PROMPT,
        )

    # merge the two judged outputs
    merged_scores = []
    for i, score in enumerate(judged_outputs_llm):
        if eval_type == "collection":
            score['precision'] = judged_outputs_vlm[i]['precision']

        precision = score["precision"]
        recall = score["recall"]
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        score["f1"] = f1

        merged_scores.append(score)

    return merged_scores