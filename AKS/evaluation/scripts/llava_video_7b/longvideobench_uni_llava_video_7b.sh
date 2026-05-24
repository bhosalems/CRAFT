score_type=longvideobench_uni

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval\
    --model llava_vid \
    --model_args pretrained=./checkpoints/LLaVA-NeXT-Video-7B-Qwen2,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=64,overwrite=False,use_topk=False \
    --tasks longvideobench_val_v \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llavavid_7b_qwen_lvb_v \
    --output_path ./results/${score_type}
