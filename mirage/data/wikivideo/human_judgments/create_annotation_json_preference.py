import os
import json 
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--first_prediction",
        type=str,
        default='wikivideo/model_preds/qwen_72b_rag.json'
    )
    parser.add_argument(
        "--second_prediction",
        type=str,
        default='wikivideo/model_preds/qwen2vl_72b_cag_2_no_citations.json'
    )
    parser.add_argument(
        '--third_prediction',
        type=str,
        default='wikivideo/model_preds/qwen3_llm_only_no_citations.json'
    )
    parser.add_argument(
        '--reference',
        type=str,
        default='wikivideo/human_eval_subset.json'
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='/home/hltcoe/amartin/report_gen/modality-agnostic-eval/wikivideo/human_judgments'
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    first_preds = json.load(open(args.first_prediction))
    second_preds = json.load(open(args.second_prediction))
    third_preds = json.load(open(args.third_prediction))
    references = json.load(open(args.reference))


    output_json = {}
    key = {}
    for topic in references:
        reference = references[topic]['article']
        first_pred = first_preds[topic]['prediction']
        second_pred = second_preds[topic]['prediction']
        third_pred = third_preds[topic]['prediction']

        # shuffle the predictions, but make sure you can recover which is which 

        id_preds = [('qwen_72b_rag', first_pred), ('qwen2vl_72b_cag_2_no_citations', second_pred), ('qwen3_llm_only', third_pred)]
        random.shuffle(id_preds)

        key[topic] = {
            'first_prediction': id_preds[0][0],
            'second_prediction': id_preds[1][0],
            'third_prediction': id_preds[2][0],
        }

        output_json[topic] = {
            'reference': reference,
            # 'first_prediction': first_pred,
            # 'second_prediction': second_pred,
            # 'third_prediction': third_pred,
            'first_prediction': id_preds[0][1],
            'second_prediction': id_preds[1][1],
            'third_prediction': id_preds[2][1],
            'best_prediction_reference': None,
            'best_prediction_topic': None,
            'first_prediction_likert': 0,
            'second_prediction_likert': 0,
            'third_prediction_likert': 0,
        }
    os.makedirs(args.output_dir, exist_ok=True)

    # dump 3 of the same file (output_json) for 3 different annotators 
    for i in range(3):
        with open(os.path.join(args.output_dir, f'preference_json_{i+1}.json'), 'w') as f:
            json.dump(output_json, f, indent=4)
    with open(os.path.join(args.output_dir, f'preference_key.json'), 'w') as f:
        json.dump(key, f, indent=4)

    

    



if __name__ == "__main__":
    main()