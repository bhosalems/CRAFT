import os
from typing import List, Dict, Any, Tuple

from mirage.models import (
    Text2TextHfScorer, Text2TextVLLMScorer,
    Text2VideoVLLMScorer, Text2VideoHfScorer
)

from mirage.utils import _sanity_check_sources

import logging
logger = logging.getLogger(__name__)


def score_claims_reference(
    hypothesis_claims: List[str],
    hypothesis_context: str,
    reference_context: str,
    scorer: Text2TextHfScorer | Text2TextVLLMScorer,
    system_prompt: str,
    user_prompt: str,
    return_all_judgments: bool = False,
) -> float | Tuple[float, List[Dict[str, Any]]]:
    """
    Scores claims against a reference context (text), 
    returning the proportion of supported claims and optionally the judgment
    for each claim. 
    
    Args:
        hypothesis_claims (List[str]): List of claims to verify.
        hypothesis_context (str): Context where the claims came from.
        reference_context (str): Reference text to verify claims against.
        scorer (Text2TextHfScorer | Text2TextVLLMScorer): Scorer object to compute the support scores.
        system_prompt (str): The system prompt to use for evaluation.
        user_prompt (str): The prompt to use for evaluation.
        return_all_judgments (bool): Whether to return the judgment for each claim.
    Returns:
        score (float): Proportion of claims supported by the reference context.
        judgments (List[Dict[str, Any]]): List of judgments for each claim if return_all_judgments is True.
    """

    # prepare the prompts 
    prompts = []
    for claim in hypothesis_claims:
        prompt_text = user_prompt.replace("[PUT_CONTEXT_HERE]", hypothesis_context)
        prompt_text = prompt_text.replace("[PUT_CLAIM_HERE]", claim)
        prompt_text = prompt_text.replace("[PUT_VERIFICATION_CONTEXT_HERE]", reference_context)
        prompts.append(prompt_text)

    # get the responses from the model
    responses = scorer.score(prompts=prompts, system_prompt=system_prompt)

    # evaluate the responses
    score = 0.0
    if return_all_judgments:
        judgments = []

    for generated_text, claim in zip(responses, hypothesis_claims):
        if "yes" in generated_text.lower():
            score += 1.0
            if return_all_judgments:
                judgments.append({"claim": claim, "supported": True})
        elif "no" in generated_text.lower():
            score += 0.0
            if return_all_judgments:
                judgments.append({"claim": claim, "supported": False})
        else:
            logger.warning(f"Unclear response for claim '{claim}': '{generated_text}', treating as unsupported.")
            if return_all_judgments:
                judgments.append({"claim": claim, "supported": None})

    # proportion of claims supported 
    score = score / len(hypothesis_claims) if hypothesis_claims else 0.0

    if return_all_judgments:
        return score, judgments
    return score
    

def score_claims_collection_single_source(
    hypothesis_claims: List[str],
    hypothesis_context: str,
    reference_contexts: List[str],
    scorer: Text2TextHfScorer | Text2TextVLLMScorer | Text2VideoVLLMScorer | Text2VideoHfScorer,
    system_prompt: str,
    user_prompt: str,
    return_all_judgments: bool = False,
) -> float | Tuple[float, List[Dict[str, Any]]]:
    """
    Scores claims against a collection of reference contexts (text or video), 
    returning the proportion of supported claims and optionally the judgment
    for each claim. 

    Iterates through the reference contexts and if ANY context supports the claim,
    it is scored as supported. 

    Args:
        hypothesis_claims (List[str]): List of claims to verify.
        hypothesis_context (str): Context where the claims came from.
        reference_contexts (List[str]): List of reference contexts to verify claims against.
        scorer (Text2TextHfScorer | Text2TextVLLMScorer | Text2VideoVLLMScorer | Text2VideoHfScorer): Scorer object to compute the support scores.
        system_prompt (str): The system prompt to use for evaluation.
        user_prompt (str): The prompt to use for evaluation.
        return_all_judgments (bool): Whether to return the judgment for each claim.
    Returns:
        score (float): Proportion of claims supported by any of the reference contexts.
        judgments (List[Dict[str, Any]]): List of judgments for each claim if return_all_judgments is True.
    """

    # prepare the prompts
    prompts = []
    for claim in hypothesis_claims:
        prompt_text = user_prompt.replace("[PUT_CONTEXT_HERE]", hypothesis_context)
        prompt_text = prompt_text.replace("[PUT_CLAIM_HERE]", claim)
        prompts.append(prompt_text)

    # get the responses from the model
    verified_prompt = [False] * len(prompts)
    for context in reference_contexts:
        """
        Logic of this function:
        for each context,
            check all claims against context_i
            if any claim is supported by video_i, mark it as supported and do not check it again
        """
        prompts_for_batch_inference = [
            prompt for i, prompt in enumerate(prompts) if not verified_prompt[i]
        ]
        logger.info(f"Scoring {len(prompts_for_batch_inference)} claims against current context.")

        responses = scorer.score(
            prompts=prompts_for_batch_inference,
            context_path=context,
            system_prompt=system_prompt
        )

        for i, (generated_text, claim) in enumerate(zip(responses, prompts_for_batch_inference)):
            if "yes" in generated_text.lower():
                # mark the claim as verified
                original_index = prompts.index(claim)
                verified_prompt[original_index] = True
            elif "no" in generated_text.lower():
                pass
            else:
                logger.warning(f"Unclear response for claim '{claim}': '{generated_text}', treating as unsupported.")

        if all(verified_prompt):
            logger.info("All claims have been verified as supported; stopping further checks.")
            break

    # proportion of claims supported
    score = sum(1.0 for v in verified_prompt if v) / len(hypothesis_claims) if hypothesis_claims else 0.0
    
    if return_all_judgments:
        judgments = []
        for i, claim in enumerate(hypothesis_claims):
            if verified_prompt[i]:
                judgments.append({"claim": claim, "supported": True})
            else:
                judgments.append({"claim": claim, "supported": False})
        return score, judgments
    
    return score

    
def score_claims_collection_multi_source(
    hypothesis_claims: List[str],
    hypothesis_context: str,
    reference_contexts: List[str],
    scorer: Text2TextHfScorer | Text2TextVLLMScorer | Text2VideoVLLMScorer | Text2VideoHfScorer,
    system_prompt: str,
    user_prompt: str,
    return_all_judgments: bool = False,
) -> float | Tuple[float, List[Dict[str, Any]]]:
    raise NotImplementedError("Multi-source scoring not implemented yet.")



def score_claims_collection(
    hypothesis_claims: List[str],
    hypothesis_context: str,
    reference_contexts: List[str],
    scorer: Text2TextHfScorer | Text2TextVLLMScorer | Text2VideoVLLMScorer | Text2VideoHfScorer,
    system_prompt: str,
    user_prompt: str,
    return_all_judgments: bool = False,
    collection_type: str = "single_source"  # or "multi_source"
) -> float | Tuple[float, List[Dict[str, Any]]]:
    """
    calls either single_source or multi_source based on the input
    """
    
    # first do a sanity check on the sources
    _sanity_check_sources(reference_contexts) # this raises an error if any source is invalid

    if collection_type == "single_source":
        return score_claims_collection_single_source(
            hypothesis_claims=hypothesis_claims,
            hypothesis_context=hypothesis_context,
            reference_contexts=reference_contexts,
            scorer=scorer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            return_all_judgments=return_all_judgments
        )
    elif collection_type == "multi_source":
        return score_claims_collection_multi_source(
            hypothesis_claims=hypothesis_claims,
            hypothesis_context=hypothesis_context,
            reference_contexts=reference_contexts,
            scorer=scorer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            return_all_judgments=return_all_judgments
        )
    else:
        raise ValueError(f"Unknown collection eval type: {collection_type}. Must be 'single_source' or 'multi_source'.")


def score_claims_sentences(
    hypothesis_claims: List[str],
    hypothesis_context: str,
    sentences_from_prediction: List[List[str]],
    scorer: Text2TextHfScorer | Text2TextVLLMScorer,
    system_prompt: str,
    user_prompt: str,
    return_all_judgments: bool = False,
) -> float | Tuple[float, List[Dict[str, Any]]]:
    """
    Scores claims against a list of sentences.

    This function is unique to CiteF1 - Recall
    """
    
    score = 0.0
    for i, claim in enumerate(hypothesis_claims):
        prompts = []
        sentence_groups = sentences_from_prediction[i] # list of lists where each inner list is the sentences from a specific citation
        for sentences in sentence_groups:
            if len(sentences) == 0:
                continue
            sentences_from_citation = ' '.join(sentences)
            prompt_text = user_prompt.replace("[PUT_CONTEXT_HERE]", hypothesis_context)
            prompt_text = prompt_text.replace("[PUT_CLAIM_HERE]", claim)
            prompt_text = prompt_text.replace("[PUT_VERIFICATION_CONTEXT_HERE]", sentences_from_citation)
            prompts.append(prompt_text)
        responses = scorer.score(prompts=prompts, system_prompt=system_prompt)
        for response in responses:
            if "yes" in response.lower():
                score += 1.0
                break


    score = score / len(hypothesis_claims) if hypothesis_claims else 0.0
    return score 


