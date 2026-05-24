frame_num=32
use_topk=False

python ./evaluation/change_score.py \
    --frame_num $frame_num \
    --use_topk $use_topk

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model qwen2_vl \
    --model_args pretrained=./checkpoints/Qwen2-VL-7B-Instruct,use_topk=False,nframes=32 \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen2_vl_7b \
    --output_path ./results/qwen2_vl_all_test/videomme/qwen2_vl_uni_32