# Config Files for Running Mirage

This directory contains config files for running mirage. These config files specify settings for LLM and VLM inference and are formatted as follows:

```json
{
    "vlm_name": "model name",
    "llm_name": "model name",
    "fps": 1,  
    "cache_dir": "dir/to/cache", 
    "max_videos": 1, 
    "max_retries": 10, 
    "devices": -1
}
```

### specific config notes
- `vlm_name`  and `llm_name` are any model supported by huggingface and vLLM. 
- `fps` is the frames per second to sample from each video. In our paper, we use 1 fps, but this can be adjusted based on video length and available compute.
- `cache_dir` is the directory to cache downloaded models.
- `max_videos` is the number of videos that are supported in a (claim,videos) entailment judgment. In our paper we use 1, however, this can be increased to N. 
- `max_retries` is specifically tied to `max_videos`. We uniformly downsample the framerate of each video until they fit on device memory. If we cannot fit the videos on device memory after `max_retries` attempts, we treat this as not being supported (i.e. return False for that (claim,videos) judgment).
- `devices` is the number of GPUs to use for inference. You can provide any positive number or -1 for all available GPUs.


### Paper config
The experiments from the paper are run on an H200 node for the 1fps framerate for long (10min) videos. 
```json
{
    "vlm_name": "Qwen/Qwen2.5-7B-Instruct",
    "llm_name": "Qwen/Qwen2.5-VL-7B-Instruct",
    "fps": 1,  
    "cache_dir": "dir/to/cache", 
    "max_videos": 1,
    "devices": -1
}
```

### Config for inference on an L40 (46Gb Vram)
You can get results on a single L40 GPU using the following config (also in `vlm_config_minimal.json`)
```json
{
    "vlm_name": "Qwen/Qwen2.5-7B-Instruct",
    "llm_name": "Qwen/Qwen2.5-VL-7B-Instruct",
    "fps": 0.25,  
    "cache_dir": "dir/to/cache", 
    "max_videos": 1, 
    "max_retries": 10, 
    "devices": -1
}
```


### Selecting other VLMs 
