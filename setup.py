import os
import sys
import time
import requests
import traceback
from huggingface_hub import HfApi, login, snapshot_download, hf_hub_download
from kernels import get_kernel


HF_TOKEN=''
HF_ENDPOINT="https://hf-mirror.com"
HF_HOME="/root/autodl-tmp/huggingface/"

def try_login():
    api = HfApi()
    try:
        user_info = api.whoami()
        print(f"✅ 已登录为: {user_info['name']}")
    except Exception:
        print("❌ 尚未登录")
        login(token=HF_TOKEN, add_to_git_credential=True)

def download_flash_attn2():
    print("下载 kernels-community/flash-attn2")
    get_kernel("kernels-community/flash-attn2")
    
def download_qwen_image_edit_2511():
    ignore_list = [
        "transformer/*", 
        # "text_encoder/*.safetensors", 
        # "text_encoder/*.bin", 
        # "text_encoder/*.pt",
        # "text_encoder/*.model"  # 有些模型包含 .model 文件
    ]

    print("下载 Qwen/Qwen-Image-Edit-2511")
    cache_path = snapshot_download(
        repo_id="Qwen/Qwen-Image-Edit-2511",
        ignore_patterns=ignore_list
    )
    return cache_path


def download_nunchaku_transformer():
    print("下载 Qwen/Qwen-Image-Edit-2511")
    nunchaku_repo_id="QuantFunc/Nunchaku-Qwen-Image-EDIT-2511"
    nunchaku_filename="nunchaku_qwen_image_edit_2511_balance_int4.safetensors"
    transformer_path = hf_hub_download(repo_id=nunchaku_repo_id, filename=nunchaku_filename)

    return transformer_path
    
def download_lightning_4step_lora():
    lighting_4step_repo = "lightx2v/Qwen-Image-Edit-2511-Lightning"
    lighting_4step_filename = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    lighting_4step_path = hf_hub_download(repo_id=lighting_4step_repo, filename=lighting_4step_filename)
    return lighting_4step_path
    
def download_qwen25_vl_7b_abliterated():
    qwen_vl_repo = "huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated"
    print("下载 Qwen/Qwen-Image-Edit-2511")
    cache_path = snapshot_download(
        repo_id=qwen_vl_repo,
    )
    return cache_path
    
path = download_qwen_image_edit_2511()
print(f'qwen_image_edit-2511: {path}')

path = download_lightning_4step_lora()
print(f'lightning-4steps-lora: {path}')

path = download_nunchaku_transformer()
print(f'nunchaku transformer: {path}')

path = download_qwen25_vl_7b_abliterated()
print(f'qwen25_vl_7b_abliterated: {path}')