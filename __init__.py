"""Lightweight namespace for note-taking utilities."""

from .prompts import (
    call_llm_with_retry,
    contains_speculative_language,
    parse_json_with_expected_key,
    prompt_grounded_notes,  # LEGACY
    prompt_grounded_notes_retry,  # LEGACY
    prompt_observation_notes,  # LEGACY
    prompt_observation_notes_retry,  # LEGACY
    prompt_observation_video,  # LEGACY
    prompt_observation_video_retry,  # LEGACY
    prompt_query_conditioned_expanded,  # LEGACY
    prompt_query_conditioned_expanded_retry,  # LEGACY
    prompt_query_conditioned_single,  # LEGACY
    prompt_query_conditioned_single_retry,  # LEGACY
    strip_code_fences,
    # New prompts (PR2)
    prompt_general_notes,
    prompt_general_notes_retry,
    # New prompts (PR3)
    prompt_query_claims,
    prompt_query_claims_retry,
    prompt_query_claims_expanded,
    prompt_query_claims_expanded_retry,
    # New prompts (PR6)
    prompt_higher_level_inference,
    prompt_higher_level_inference_retry,
)

__all__ = [
    "call_llm_with_retry",
    "contains_speculative_language",
    "parse_json_with_expected_key",
    "prompt_grounded_notes",
    "prompt_grounded_notes_retry",
    "prompt_observation_notes",
    "prompt_observation_notes_retry",
    "prompt_observation_video",
    "prompt_observation_video_retry",
    "prompt_query_conditioned_expanded",
    "prompt_query_conditioned_expanded_retry",
    "prompt_query_conditioned_single",
    "prompt_query_conditioned_single_retry",
    "strip_code_fences",
    "prompt_general_notes",
    "prompt_general_notes_retry",
    "prompt_query_claims",
    "prompt_query_claims_retry",
    "prompt_query_claims_expanded",
    "prompt_query_claims_expanded_retry",
    "prompt_higher_level_inference",
    "prompt_higher_level_inference_retry",
]
