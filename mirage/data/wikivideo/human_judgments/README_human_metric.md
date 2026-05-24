# Annotation Instructions for human metric judgments


## Overview 
In this task, you'll be given a set of 3 json files for each prediction and 3 json files for the reference. The reference json files will all be the same, but each will correspond to your annotations for a different system's predictions. In each json file, there will be the same 10 topics. 



For every system prediction json, you'll see 
1. `prediction`: the predicted article for the topic. This prediction is the model's attempt to write an article for the topic.
2. `claims`: this is a sub dictionary that contains the claims and the sentences they came from. It is formatted as 
```
{
  "sentence": {
    "claim": "the claim to be verified",
    "judgment": [False|True]
  },
  ...
}
```
where the judgment is whether the claim is supported by the reference article or not.


For every reference json, you'll see
1. `reference`: this article is the human written gold article for the topic. It is 100% factual and the best article for the topic.
2. `claims`: this is a sub dictionary that contains the claims and the sentences they came from. It is formatted as 
```
{
  "sentence": {
    "claim": "the claim to be verified",
    "judgment": [False|True]
  },
  ...
}
```
where the judgment is whether the claim is supported by the prediction article or not. 

## Annotation Task
The goal of this task is to mirror how we evaluate multimodal RAG in InfoF1. The basic premise is as follows.

1. For every claim in the prediction, check if it is supported by the reference article. If it is mark the judgment as True, otherwise mark it as False. (The default value is None)
2. For every claim in the reference, check if it is supported by the prediction article. If it is mark the judgment as True, otherwise mark it as False. (The default value is None)

## Claim Annotation Criteria 
When deciding whether a **claim is supported** by the other article (reference or prediction), apply the following criteria. The goal is to make judgments **only when there is no reasonable doubt** that the claim is supported by the other text.

A claim is **SUPPORTED (True)** if:
1. **All factual elements** in the claim are explicitly stated or can be **directly inferred** from the other article without needing external knowledge.
   * Example: Claim: *“The Eiffel Tower is located in Paris.”*
     → Supported if the other article states *“The Eiffel Tower is in Paris.”*
2. The **meaning and intent** of the claim are **fully consistent** with the information in the article.
   * Minor wording differences are acceptable if they don’t change meaning.
3. The **temporal or causal context** matches (e.g., dates, events, outcomes are consistent).
4. If the claim includes **quantitative or categorical facts** (numbers, names, locations, affiliations), these details must exactly match what is stated in the other article.

A claim is **NOT SUPPORTED (False)** if:
1. The other article **contradicts** any part of the claim.
2. The other article **omits or is ambiguous** about key details needed to verify the claim.
3. The claim requires **inference beyond what’s stated**, such as outside knowledge, assumptions, or general reasoning not grounded in the text.
4. The claim is **partially supported**, but not fully — i.e., some parts are correct while others are missing or uncertain.

**General rule:**

> Only mark a claim as *True* (supported) if you can clearly point to a sentence or set of sentences in the other article that fully confirm it, leaving **no doubt** about its accuracy. Otherwise, mark it as *False* (not supported).
