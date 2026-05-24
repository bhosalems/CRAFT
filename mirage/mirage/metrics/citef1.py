import os
from typing import List, Dict, Any, Tuple



from mirage.prompts import (
    CITATION_VERIFICATION_TEXT_SYSTEM_PROMPT, CITATION_VERIFICATION_TEXT_USER_PROMPT,
    CLAIM_VERIFICATION_VIDEOS_SYSTEM_PROMPT, CLAIM_VERIFICATION_VIDEOS_USER_PROMPT,
    CLAIM_VERIFICATION_TEXT_SYSTEM_PROMPT, CLAIM_VERIFICATION_TEXT_USER_PROMPT
)

from mirage.models import (
    Text2TextHfScorer, Text2TextVLLMScorer,
    Text2VideoVLLMScorer, Text2VideoHfScorer
)

from mirage.scoring_functions import (
    score_claims_reference, score_claims_collection,
    score_claims_sentences
)

import logging
logger = logging.getLogger(__name__)




def judge_inference_llm(
    predicted_claims,
    predicted_contexts,
    predicted_videos_to_sentences,
    reference_claims,
    reference_contexts,
    reference_videos_to_claims,
    eval_type: str,
    scorer: str,
    system_prompt: str = CITATION_VERIFICATION_TEXT_SYSTEM_PROMPT,
    user_prompt: str = CITATION_VERIFICATION_TEXT_USER_PROMPT,
):
    """
    Performs inferences where LLM inference is needed.


    Recall always requires LLM inference, while precision may or may not depending on eval_type
    """
    model = Text2TextVLLMScorer(model_name=scorer)
    scores = []
    for i, pred_claims in enumerate(predicted_claims):
        pred_context = predicted_contexts[i]
        pred_videos_to_sentences = predicted_videos_to_sentences[i]
        ref_claims = reference_claims[i]
        ref_context = reference_contexts[i]
        ref_videos_to_claims = reference_videos_to_claims[i]


        precision = None
        if eval_type == "reference":
            hypothesis_claims = []
            ref_claims_cited = []
            for claim in pred_claims:
                hypothesis_claims.append(claim)
                cited_videos = pred_claims[claim]
                for video in cited_videos:
                    ref_claims_for_vid = ref_videos_to_claims.get(video, [])

                    # format the claims s.t. it's `Claims from videoID: claim1. claim2. claim3.`
                    claims_as_sent = '. '.join(ref_claims_for_vid)
                    claims_as_sent = f"Claims from {video}: {claims_as_sent}."
                    if claims_as_sent not in ref_claims_cited:
                        ref_claims_cited.append(claims_as_sent)

            precision = score_claims_reference(
                hypothesis_claims=hypothesis_claims,
                hypothesis_context=pred_context,
                reference_context=' '.join(ref_claims_cited),
                scorer=model,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
            )


        hypothesis_claims = []
        sentences_from_prediction = [] # nested list where each outer index corresponds to a claim, the inner list is the sentences from a specific citation
        for claim in ref_claims:
            supportive_videos = ref_claims[claim]
            pred_sentence_groups = []
            for video in supportive_videos:
                sentences = pred_videos_to_sentences.get(video, [])
                pred_sentence_groups.append(sentences)
            sentences_from_prediction.append(pred_sentence_groups)
            hypothesis_claims.append(claim)


        recall = score_claims_sentences(
            hypothesis_claims=hypothesis_claims,
            sentences_from_prediction=sentences_from_prediction,
            hypothesis_context=pred_context,
            scorer=model,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        # print(f"Precision: {precision}, Recall: {recall}")
        scores.append({"precision": precision, "recall": recall})

    return scores
        


def judge_inference_vlm(
    predicted_claims,
    predicted_contexts,
    scorer,
):
    """
    Performs all inferences where VLM inference is needed.
    """
    model = Text2VideoVLLMScorer(model_name=scorer)

    scores = []
    for i, pred_claims in enumerate(predicted_claims):
        pred_context = predicted_contexts[i]
        total_supported_claims = 0
        video_to_citing_claims = {}
        for claim in pred_claims:
            cited_videos = pred_claims[claim]
            for video in cited_videos:
                if video not in video_to_citing_claims:
                    video_to_citing_claims[video] = []
                if claim not in video_to_citing_claims[video]:
                    video_to_citing_claims[video].append(claim)

        for video in video_to_citing_claims:
            # print(f"Video: {video}, Citing Claims: {video_to_citing_claims[video]}")
            if not os.path.exists(video):
                logger.warning(f"Video {video} does not exist. Treating as not supported.")
                continue
            num_supported_claims = score_claims_collection(
                hypothesis_claims=video_to_citing_claims[video],
                hypothesis_context=pred_context,
                reference_contexts=[video],
                scorer=model,
                system_prompt=CLAIM_VERIFICATION_VIDEOS_SYSTEM_PROMPT,
                user_prompt=CLAIM_VERIFICATION_VIDEOS_USER_PROMPT,
            )
            total_supported_claims += num_supported_claims

        precision = total_supported_claims / len(pred_claims) if pred_claims else 0.0
        scores.append({"precision": precision})

    return scores


def f1_score(
    predicted_contexts,
    predicted_claims,
    predicted_videos_to_sentences,
    reference_contexts,
    reference_claims,
    reference_videos_to_claims,
    eval_type,
    text_scorer,
    video_scorer,
):
    """
    Compute the F1 score for claim verification.
    """

    judged_outputs_llm = judge_inference_llm(
        predicted_claims=predicted_claims,
        predicted_contexts=predicted_contexts,
        predicted_videos_to_sentences=predicted_videos_to_sentences,
        reference_claims=reference_claims,
        reference_contexts=reference_contexts,
        reference_videos_to_claims=reference_videos_to_claims,
        eval_type=eval_type,
        scorer=text_scorer,
    )

    if eval_type == "collection":
        judged_outputs_vlm = judge_inference_vlm(
            predicted_claims=predicted_claims,
            predicted_contexts=predicted_contexts,
            scorer=video_scorer,
        )
    
    merged_scores = []
    for i, score in enumerate(judged_outputs_llm):
        # if not score['precision']:
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




    
    