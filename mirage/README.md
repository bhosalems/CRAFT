# MiRAGE: Multimodal Retrieval-Augmented Generation Evaluation

<div align="center">
<a href="" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<!-- <a href="" target="_blank"><img src=https://img.shields.io/badge/HuggingFace-Evaluate-FF6D00?logo=huggingface></a> -->
</div>

MiRAGE: Multimodal Retrieval-Augmented Generation Evaluation. 

## Contents
* [Features](#features)
* [Supported Tasks](#supported-tasks)
* [Installation](#installation)
* [MiRAGE Usage](#mirage-usage)
* [Citation](#citation)
* [Contact](#contact)


## Features
- Evaluating multimodal retrieval-augmented generation systems.
- Integration with vLLM, DeepSpeed, FlashAttention, and other efficient inference techniques.
- Easy-to-use command line interface for running various metrics.
- Evaluation for generation from videos.

## Supported Tasks
### Video RAG
- WikiVideo: [repo](https://github.com/alexmartin1722/wikivideo), [paper](https://arxiv.org/abs/2504.00939)

## Installation
<details><summary><b>From Scratch</b></summary>

```bash
conda create -n video_rag_eval python=3.12 -y 
conda activate video_rag_eval
pip install --upgrade uv
uv pip install vllm --torch-backend=cu128
pip install evaluate 
pip install qwen-vl-utils[decord]==0.0.8
pip install peft
```
</details>

## MiRAGE Usage

### VideoRAG Evaluation
<details><summary><b>Data Prep</b></summary>

When evaluating VideoRAG, you will need the following data:

- predictions, 
- references, 
- video directory, containing all the videos possible to use in RAG (for collection eval only),

#### WikiVideo Data 
We provide everything need to evaluate WikiVideo RAG systems in `data/wikivideo`
- Human judgments for grounding `data/wikivideo/human_judgments/grounding_judgments`
- Human preference judgments (EQJs in the paper) `data/wikivideo/human_preference`
- Metric preference judgments (ICJs in the paper) `data/wikivideo/metric_preference`
- Model predictions for various systems in `data/wikivideo/model_preds/`
- Eval subset from WikiVideo used in the human eval and paper `data/wikivideo/human_eval_subset.json`

For any reference evaluation, you'll need to download the videos used in WikiVideo, which can be found on [huggingface](https://huggingface.co/datasets/hltcoe/wikivideo).


#### Custom data
To run our code as is, you'll need to format your data and system predictions in the following formats:

- **System Predictions**: A JSON file where each entry's key is the topic ID and the values associated with that ID are
  - `prediction`: The generated text from the RAG system. We recommend stripping the citations from this so it is pure text. 
  - `sentences`: The sentence tokenized version of the prediction.
  - `claims`: A list where each index corresponds to a sentence and at each index is the subclaims for that sentence
  - `citations`: A list where each index corresponds to a sentence and at each index is the citations for that sentence. We use the video path as the citation text
    
    Example:
    ```json
    {
      "Topic_ID" : {
        "prediction": "Generated text here...",
        "sentences": ["Generated sentence 1.", "Generated sentence 2."],
        "claims": [["Subclaim 1 for sentence 1.", "Subclaim 2 for sentence 1."], ["Subclaim 1 for sentence 2."]],
        "citations": [["path to citation 1", "path to citation 2"], ["path to citation 3"]]
      }
    }
    ```

- **References**: A JSON file where each entry's key is the topic ID and the values associated with that ID are 
  - `article`: The ground truth text for the topic written by a human. 
  - `claims_to_supporting_videos`: a mapping between the claims of the reference and the videos that support those claims. This is a dictionary formatted s.t. each key is a claim and the values are (1) supporting videos and (2) the modalities from the videos that support the claim.
  
    Example:
    ```json
    {
      "Topic_ID" : {
        "article": "Ground truth article text here...",
        "claims_to_supporting_videos": {
          "Claim 1": {
            "supporting_videos": ["video_id", "video_id"],
            "videos_modalities": {
              "video_id": ["video", "audio"],
              "video_id": ["video"]
            }
          },
          "Claim 2": {
            "supporting_videos": ["video_id"],
            "videos_modalities": {
              "video_id": ["video", "audio", "ocr"]
            }
          }
        }
    }
    ```

</details>

<details><summary><b>Evaluation</b></summary>

When evaluating the RAG tasks, our metrics are driven by two files `infof1.py` and `citef1.py` for InfoF1 and CiteF1 respectively. 

#### InfoF1:
```bash
python infof1.py \
    --eval_type [reference|collection] \
    --prediction [path_to_system_prediction] \
    --reference [path_to_human_eval_json] \
    --video_dir [path_to_videos] \ #only needed for collection eval
    --output_dir [path_to_output_directory] \
    --model_name [qwen_7b|qwen_72b]
```
```bash
python infof1.py \
    --eval_type collection \
    --prediction data/wikivideo/model_preds/qwen_72b_cag_relevant_citations.json \
    --reference data/wikivideo/human_eval_subset.json \
    --video_dir /exp/amartin/wikivideo/all_videos \
    --output_dir data/wikivideo/model_preds/metric_outputs \
    --model_name qwen_7b
```
#### CiteF1:
```bash
python citef1.py \
    --eval_type [reference|collection] \
    --prediction [path_to_system_prediction] \
    --reference [path_to_human_eval_json] \
    --video_dir [path_to_videos] \ #only needed for collection eval
    --output_dir [path_to_output_directory] \
    --model_name [qwen_7b|qwen_72b]
```
```bash
python citef1.py \
    --eval_type collection \
    --prediction data/wikivideo/model_preds/qwen_72b_cag_relevant_citations.json \
    --reference data/wikivideo/human_eval_subset.json \
    --video_dir /exp/amartin/wikivideo/all_videos \
    --output_dir data/wikivideo/model_preds/metric_outputs \
    --model_name qwen_7b
```

</details>



## Citation
If you find MiRAGE useful in your research, please consider citing the following paper:

```
```

## Contact
If you have MiRAGE specific questions, would like a new feature, model support, supported dataset, etc., feel free to open an issue. 

You can also reach out to me for general comments/suggestions/questions through email. 
- Alexander Martin, amart233@jhu.edu
    - if the email listed there is out of date, you can find my current email on my [personal website](https://alexmartin1722.github.io/).
