import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from transformers import CLIPProcessor, CLIPModel

import json
from decord import VideoReader
from decord import cpu, gpu
import numpy as np
import os
import cv2

import numpy as np
import pickle

import argparse
import json


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract Video Feature')
    parser.add_argument('--dataset_path', type=str, default='',help='your path of the dataset')
    parser.add_argument('--topic_mapping', type=str, default='',help='your path of the topic mapping')
    parser.add_argument('--extract_feature_model', type=str,default='blip', help='clip')
    parser.add_argument('--output_file', type=str,default='/a2il/data/mbhosale/MAGMaR2026_test/out_scores/',help='path of output scores and frames')
    parser.add_argument('--device', type=str,default='cuda')

    return parser.parse_args()


def main(args):
    if os.path.exists(args.topic_mapping):
        with open(args.dataset_path,'r') as f:
            datas = json.load(f)
    else:
        raise OSError("the label file does not exist")
    device = args.device
    
    if args.extract_feature_model == 'blip':
        model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)
    elif args.extract_feature_model == 'clip':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model.to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    elif args.extract_feature_model == 'sevila':
        raise ValueError('Not support sevila')
        # model, vis_processors, text_processors = load_model_and_preprocess(name="sevila", model_type="pretrain_flant5xl", is_eval=True, device=device)
    else:
        raise ValueError("model not support")

    os.makedirs(os.path.join(args.output_file,args.model_name), exist_ok=True)
    out_score_path = os.path.join(args.output_file,args.model_name,args.extract_feature_model)
    if not os.path.exists(out_score_path):
        os.mkdir(out_score_path)
   

    scores = []
    fn = []
    embeddings = []
    score_path = os.path.join(out_score_path,'scores.json')
    frame_path = os.path.join(out_score_path,'frames.json')
    embedding_path = os.path.join(out_score_path,'embedding.pkl')

    print(data)
    for data in datas:
        text = data['']             
        duration = data['duration']
        vr = VideoReader(video, ctx=cpu(0), num_threads=1)
        fps = vr.get_avg_fps()
        frame_nums = int(len(vr)/int(fps))

        score = []
        frame_num = []
        embedding = []

        if args.extract_feature_model == 'blip':
            txt = text_processors["eval"](text)
            for j in range(frame_nums):
                raw_image = np.array(vr[j*int(fps)])
                raw_image = Image.fromarray(raw_image)
                img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    # blip_output = model({"image": img, "text_input": txt}, match_head="itm") 
                    blip_output,blip_embedding = model({"image": img, "text_input": txt}, match_head="itm-e")              
                blip_scores = torch.nn.functional.softmax(blip_output, dim=1)
                blip_embedding = blip_embedding.cpu() 
                embedding.append(blip_embedding)
                score.append(blip_scores[:, 1].item())
                frame_num.append(j*int(fps))

        elif args.extract_feature_model == 'clip':
            inputs_text = processor(text=text, return_tensors="pt", padding=True,truncation=True).to(device)
            text_features = model.get_text_features(**inputs_text)
            for j in range(frame_nums):
                raw_image = np.array(vr[j*int(fps)])
                raw_image = Image.fromarray(raw_image)
                inputs_image = processor(images=raw_image, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs_image)
                clip_score = torch.nn.CosineSimilarity(dim=-1)(text_features, image_features)
                clip_embedding = image_features.cpu() 
                embedding.append(clip_embedding)
                score.append(clip_score.item())
                frame_num.append(j*int(fps))

        else:
            raise ValueError('Not support sevila')
            # text = 'Question: ' + data['question'] + ' Candidate: ' 
            # if args.dataset_name == 'longvideobench':
            #     for j,cad in enumerate(data['candidates']):
            #         text = text + ". ".join([chr(ord("A")+j), cad]) + ' '
            # else:   
            #     for j in data['options']:
            #         text = text + j
            # text = text + '. Is this a good frame can answer the question?'
            # txt = text_processors["eval"](text)
            # for j in range(frame_nums):
            #     raw_image = np.array(vr[j*int(fps)])
            #     raw_image = Image.fromarray(raw_image)
            #     img = vis_processors["eval"](raw_image).unsqueeze(0).unsqueeze(0).to(device)
            #     samples = {'video':img,'loc_input':txt}
            #     sevila_score = float(model.generate_score(samples).squeeze(0).squeeze(0))
            #     score.append(sevila_score)
            #     frame_num.append(j*int(fps))

        fn.append(frame_num)
        scores.append(score)
        embeddings.append(embedding)
        
    with open(frame_path,'w') as f:
        json.dump(fn,f)
    with open(score_path,'w') as f:
        json.dump(scores,f)
    with open(embedding_path,'wb') as f:
        pickle.dump(embeddings,f)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)