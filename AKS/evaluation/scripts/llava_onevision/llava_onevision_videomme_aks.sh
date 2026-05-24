base_score_path=./selected_frames/videomme/blip
score_type=videomme_32
dataset_name=videomme

python ./evaluation/change_score.py \
    --base_score_path $base_score_path \
    --score_type $score_type \
    --dataset_name $dataset_name 

frame_num=32
use_topk=True

python ./evaluation/insert_frame_num.py \
    --frame_num $frame_num \
    --use_topk $use_topk

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=./checkpoints/llava-onevision-qwen2-7b-ov,use_topk=True \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_onevision_7b \
    --output_path ./results/llavaonevision/${score_type}
