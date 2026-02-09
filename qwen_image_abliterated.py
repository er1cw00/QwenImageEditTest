import os
import sys
import json
import math
import time
import torch
import ctypes
import argparse
import warnings

from PIL import Image
from huggingface_hub import hf_hub_download
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image
from transformers import Qwen2_5_VLForConditionalGeneration

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_precision
from nunchaku_lora import load_nunchaku_lora

# 屏蔽环境变量相关的警告
os.environ["PYTHON_GIL"] = "1" # 或者干脆不管它
# 在代码中忽略特定的 UserWarning

warnings.filterwarnings("ignore", category=UserWarning, message=".*Perplexity.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Python GIL is enabled.*")


precision = get_precision()
print(f"precision: {precision}")

def print_gpu_mem():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


def check_size(side):
    side = max(512, min(4096, side))   # 先限制在范围内
    side = (side // 8) * 8              # 再向下对齐到最近的8的倍数
    return side 

def resize(image: Image.Image, max_size: int) -> Image.Image:
    """
    将 PIL Image 等比例缩放到最长边不超过 1024 像素。
    """
    #max_size = 1024
    width, height = image.size

    # 计算缩放比例
    scale = min(max_size / width, max_size / height, 1.0)
    new_size = (int(width * scale), int(height * scale))

    # 仅当图像大于1024时才缩放
    if scale < 1.0:
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


num_inference_steps = 4
torch_dtype = torch.bfloat16

nunchaku_repo_id="QuantFunc/Nunchaku-Qwen-Image-EDIT-2511"
nunchaku_filename="nunchaku_qwen_image_edit_2511_balance_int4.safetensors"
transformer_path = hf_hub_download(repo_id=nunchaku_repo_id, filename=nunchaku_filename)
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(transformer_path)


lighting_4step_repo = "lightx2v/Qwen-Image-Edit-2511-Lightning"
lighting_4step_filename = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
lighting_4step_path = hf_hub_download(repo_id=lighting_4step_repo, filename=lighting_4step_filename)
load_nunchaku_lora(transformer, lighting_4step_path, strength=1.0)

qwen_vl_repo = "huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated"
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_vl_repo, dtype=torch_dtype, device_map="auto")

qwen_repo = "Qwen/Qwen-Image-Edit-2511"
pipeline = QwenImageEditPlusPipeline.from_pretrained(qwen_repo, text_encoder=text_encoder, transformer=transformer, torch_dtype=torch_dtype)

transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=32) # increase num_blocks_on_gpu if you have more VRAM
pipeline._exclude_from_cpu_offload.append("transformer")
pipeline.enable_sequential_cpu_offload()
pipeline.precision = precision

def image_edit(images, prompt, width=None, height=None):
    inputs = {
        "image": images,
        "prompt": prompt,
        "negative_prompt": " ",
        "true_cfg_scale": 1.1,
        "num_inference_steps": num_inference_steps,
    }
    if width is not None and height is not None:
        inputs['width'] = width
        inputs['height'] = height
        print(f'resolution: {width}x{height}')

    output = pipeline(**inputs)
    return output.images

prompt = "Make Pikachu hold a sign that says 'Nunchaku is awesome', yarn art style, detailed, vibrant colors"
image_url = f'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png'
image = load_image(image_url)

image = resize(image, 1200)

if any(dim >= 1024 for dim in image.size): # Corrected check for image dimensions
    new_width, new_height = image.size
    print(f'resolution: {new_width}x{new_height}')
else:
    new_width, new_height = None, None

output = image_edit([image], prompt, new_width, new_height)[0]
output.save(f"qwen-image-edit-lora.png")