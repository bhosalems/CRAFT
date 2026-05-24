# Annotation Instructions for human preference judgments 


## Overview 
In this task, you'll be given a set of 10 topics. For every instance in the topic, you'll also see:

1. `reference`: this article is the human written gold article for the topic. It is 100% factual and the best article for the topic. 
2. `n_prediction`: you will be given three predictions `first_prediction`, `second_prediction`, `third_prediction` from three different models. These predictions are the models attempt to write an article for the topic. 
3. `best_prediction_x`: these two keys `best_prediction_reference` and `best_prediction_topic` are slots to be filled in by you depending on which prediction you think best satisifies the criteria below.
4. `n_prediction_likert`: you will be given three keys to put likert scores `first_prediction_likert`, `second_prediction_likert`, `third_prediction_likert` that correspond to each prediction. Also see the annotation criteria for them below. 


## Best Prediction Criteria 
There two best prediction judgments to be made. `best_prediction_reference` and `best_prediction_topic`.

### Best Prediction Reference
For this judgment, you should choose the prediction that you think is the best representation of the reference article. Some things to help you think about this are:

1. Does the prediction capture the core facts of the reference article? 
2. Does the prediction leave out any important information from the reference article?
3. Does the prediction add any information that is not in the reference article?
4. Does the prediction make any factual errors with respect to the reference article?



### Best Prediction Topic
For thhis judgmenet, you should choose the prediction that you think is the best representation of the topic. The reference is the best article on the topic. Some things to help you think about this are:

1. Is the prediction on the topic or some other topic?
2. Does the prediction capture the core facts of the topic?


#### Alex note
I think Will is write and after defining these it feels they are the same


## Likert Scale Criteria 
For each prediction, you should give a likert score from 1 to 5. The criteria for giving a score is as follows (in descending importance):

- **Consistency/Factuality:** Does the article make only true statements about the topic in question, given what the reference says about that topic?
    - Articles that make factual errors, omissions, or hallucinations of any kind should be penalized.

- **Adequacy:** Does the article adequately capture all of the information contained in the reference article?
    - Articles that omit details about any of the important aspects of the topic should be penalized.

- **Coherence:** Does the article make sense on its own, as a standalone description of the topic?
    - Articles that require you to go read the reference in order to understand what they mean (or that don't make sense even then) should be penalized.
    - Articles that don't provide substance on the event should also be penalized (e.g. "the [event] happened at [a time] in [a place]")

- **Relevancy:** Does the article include only information that is relevant to the topic in question?
    - Articles that include irrelevant or superfluous information, or information about some topic other than the one represented by the reference article, should be penalized.

- **Fluency:** Does the article sound reasonable natural (like something a native English speaker might actually write)?
    - Articles that are disfluent or that sound unnatural should be penalized.


Oftentimes, some of the summaries may be very similar to each other. It is totally fine to give multiple summaries the same score if you think they are of comparable quality!

You should enter your score for each summary in the score field. The default value for each summary is 0, meaning you have not yet annotated the likert judgment. Please do not use half scores (1.5, 2.5, etc).

When you are done, please return commit the file under `preference_json_N_completed.json` depending on your number (N).
