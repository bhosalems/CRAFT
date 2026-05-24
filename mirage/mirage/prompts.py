
MULTI_VIDEO_VERIFICATION_SYSTEM_PROMPT = (
    "You are an expert at verifying information."
    + " You will be given a set of videos and a sentence."
    + " Your task is to determine if the sentence is fully supported by the videos."
    + " You will output <response>yes<response> if the sentence is fully supported by the videos, "
    + "or <response>no<response> if the sentence is not fully supported by the videos."
)


MULTI_VIDEO_VERIFICATION_USER_PROMPT = (
    "Sentence: [PUT_SENTENCE_HERE]"
    + "Is the sentence fully supported by the videos?"
    + " Only respond with <response>yes<response> or <response>no<response>."
)


CITATION_VERIFICATION_TEXT_SYSTEM_PROMPT = (
    "You are an expert in evaluating and verifying claims."
    + " You will be given a a claim, the context the claim came from, and a list of claims to verify the claim against"
    + " Your task is to determine if the claim is supported by the list of claims."
    + " You will output <response>yes<response> if the claim is supported by the list of claims, "
    + "or <response>no<response> if the claim is not supported by the list of claims."
)

CITATION_VERIFICATION_TEXT_USER_PROMPT = (
    "Here is the list of claims to verfiy against: <verification_context> [PUT_VERIFICATION_CONTEXT_HERE] <verification_context>."
    + "\nHere is the context the claim came from: <claim_context> [PUT_CONTEXT_HERE] <claim_context>."
    + "\nHere is the claim: <claim> [PUT_CLAIM_HERE] <claim>"
    + "\n\nOnly respond with <response>yes<response> or <response>no<response>."
    + " Is the claim: [PUT_CLAIM_HERE], supported by list of claims to verify against?"
)


CLAIM_VERIFICATION_VIDEOS_SYSTEM_PROMPT = (
    "You are an expert in evaluating and verifying claims."
    + " You will be given a video, a claim, and the context the claim came from."
    + " Your task is to determine if the claim is supported by the video."
    + " You will output <response>yes<response> if the claim is supported by the video, "
    + "or <response>no<response> if the claim is not supported by the video."
)

CLAIM_VERIFICATION_VIDEOS_USER_PROMPT = (
    "Here is the context the claim came from: <claim_context> [PUT_CONTEXT_HERE] <claim_context>."
    + "\nHere is the claim: <claim> [PUT_CLAIM_HERE] <claim>"
    + "\n Only respond with <response>yes<response> or <response>no<response>."
    + " Is the claim: [PUT_CLAIM_HERE], supported by the video?"
)


CLAIM_VERIFICATION_TEXT_SYSTEM_PROMPT = (
    "You are an expert in evaluating and verifying claims."
    + " You will be given a passage of text, a claim, and the context the claim came from."
    + " Your task is to determine if the claim is supported by the passage of text."
    + " You will output <response>yes<response> if the claim is supported by the passage, "
    + "or <response>no<response> if the claim is not supported by the passage."
)

CLAIM_VERIFICATION_TEXT_USER_PROMPT = (
    "Here is the passage: <verification_context> [PUT_VERIFICATION_CONTEXT_HERE] <verification_context>."
    + "\nHere is the context the claim came from: <claim_context> [PUT_CONTEXT_HERE] <claim_context>."
    + "\nHere is the claim: <claim> [PUT_CLAIM_HERE] <claim>"
    + "\n\nOnly respond with <response>yes<response> or <response>no<response>."
    + " Is the claim: [PUT_CLAIM_HERE], supported by the passage?"
)
