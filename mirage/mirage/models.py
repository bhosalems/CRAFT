from typing import List, Optional, Tuple, Union, Dict
from mirage.prompts import (
    CLAIM_VERIFICATION_TEXT_SYSTEM_PROMPT,
    CLAIM_VERIFICATION_VIDEOS_SYSTEM_PROMPT
)

import gc
import torch
import logging

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


def _release_prior_gpu_memory():
    """Best-effort release of GPU memory held by any prior LLM() instance in
    this process. Collection-mode metric scripts load the text LLM and then the
    VLM inside a single process; vLLM keeps weights + KV cache resident until
    Python GC happens. Call this before instantiating a new LLM so the second
    model has enough free VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

logger = logging.getLogger(__name__)

class Text2TextHfScorer():
    def __init__(self):
        pass


class Text2TextVLLMScorer():
    def __init__(
        self,
        model_name: str,
        vlm_config: Optional[Dict] = None,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.85,
    ):
        self.model_name = model_name
        self.visible_devices = torch.cuda.device_count()
        _release_prior_gpu_memory()
        self.llm = LLM(
            model=model_name,
            enable_prefix_caching=True,
            tensor_parallel_size=self.visible_devices,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
        )

    def score(
        self, 
        prompts,
        system_prompt: str = CLAIM_VERIFICATION_TEXT_SYSTEM_PROMPT,
    ):
        
        all_messages = []
        
        for prompt in prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                # enable_thinking=False
            )
            all_messages.append(prompt)
            # all_messages.append(messages)

        


        outputs = self.llm.generate(all_messages, sampling_params=self.sampling_params)
        return [output.outputs[0].text for output in outputs]
    
class Text2VideoHfScorer():
    def __init__(self):
        pass

class Text2VideoVLLMScorer():
    def __init__(
        self,
        model_name: str,
        max_videos: int = 1,
        max_retries: int = 10,
        max_model_len: int = 65536,
        gpu_memory_utilization: float = 0.85,
    ):
        self.model_name = model_name
        self.max_videos = max_videos
        self.max_retries = max_retries
        self.visible_devices = torch.cuda.device_count()
        _release_prior_gpu_memory()
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=self.visible_devices,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"video": max_videos},
            max_model_len=max_model_len,
        )
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            use_fast=False
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
        )

    def score_potential_oom(
        self, 
        prompt: str, # no batched inference in oom version cuz y make life difficult
        video_paths: List[str],
        system_prompt: str = CLAIM_VERIFICATION_VIDEOS_SYSTEM_PROMPT,
    ):
        assert len(video_paths) <= self.max_videos, \
            f"Number of videos {len(video_paths)} exceeds max_videos {self.max_videos}"
        
        fps = 1
        max_retries = self.max_retries
        while True:
            if max_retries <= 0:
                raise RuntimeError(f"Max retries exceeded for video fps reduction, lowest fps tried was {fps}")
            max_retries -= 1
            try:
                video_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        *[
                            {
                                "type": "video",
                                "video": video_path,
                                "fps": fps,
                            } for video_path in video_paths
                        ]
                    ]}
                ]
                messages = video_messages
                prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                image_inputs, video_inputs = process_vision_info(messages)
                mm_data = {}
                if video_inputs is not None:
                    mm_data["video"] = video_inputs
                llm_inputs = {
                    "prompt": prompt,
                    "multi_modal_data": mm_data,
                }
                outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
                return [o.outputs[0].text for o in outputs]
            except:
                logger.warning(f"OOM with fps={fps}, reducing fps and retrying...")
                fps = fps / 2

    def score(
        self,
        prompts: List[str],
        context_path: str,
        system_prompt: str = CLAIM_VERIFICATION_VIDEOS_SYSTEM_PROMPT,
    ):
        all_llm_inputs = []
        for prompt in prompts:
            video_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "video",
                        "video": context_path,
                        "fps": 0.25
                    }
                ]}
            ]
            messages = video_messages
            processed_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            try:
                image_inputs, video_inputs = process_vision_info(messages)
            except:
                return ["no" for _ in prompts]
            mm_data = {}
            if video_inputs is not None:
                mm_data["video"] = video_inputs
            llm_inputs = {
                "prompt": processed_prompt,
                "multi_modal_data": mm_data,
            }
            all_llm_inputs.append(llm_inputs)
        outputs = self.llm.generate(all_llm_inputs, sampling_params=self.sampling_params)
        return [o.outputs[0].text for o in outputs]