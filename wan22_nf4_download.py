cat nf4_model.py 
import os
import sys
import time
import requests
import traceback
from huggingface_hub import HfApi, login, snapshot_download, hf_hub_download

def download_wan22_diffuser_transformer():
    allow_list = [
        "transformer/*",
        # "text_encoder/*.safetensors",
        # "text_encoder/*.bin",
        # "text_encoder/*.pt",
        # "text_encoder/*.model"  # 有些模型包含 .model 文件
    ]

    print("下载 magespace/Wan2.2-I2V-A14B-Lightning-Diffusers")
    cache_path = snapshot_download(
        repo_id="magespace/Wan2.2-I2V-A14B-Lightning-Diffusers",
        allow_patterns=allow_list
    )
    return cache_path

cache_path = download_wan22_diffuser_transformer()
print(f"{cache_path}")
