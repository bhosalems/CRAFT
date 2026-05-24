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

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model qwen2_vl \
    --model_args pretrained=./checkpoints/Qwen2-VL-7B-Instruct,use_topk=True,nframes=32 \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen2_vl_7b \
    --output_path /nfs/zm/aks/AKS-main/results/qwen2_vl_all_test/videomme/qwen2_vl_aks_32

