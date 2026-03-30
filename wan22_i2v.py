import torch
import numpy as np
from diffusers import WanImageToVideoPipeline
from diffusers.utils import load_image, export_to_video
from transformers import BitsAndBytesConfig

model_id = "lopho/Wan2.2-I2V-A14B-Diffusers_nf4"
image_path = "./test.jpg"
output_path = "./output_video.mp4"
num_frames = 50
fps = 5
prompt = "一个穿着白色连衣裙的亚洲女孩，慢慢地掀起裙子，一边扭动着自己的身体"
height = 832
width = 480
num_inference_steps = 30
# 1. 配置 4-bit (NF4) 量化参数
# bnb_4bit_compute_dtype 使用 float16 以保证计算稳定性
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

print(f"正在加载 NF4 量化模型: {model_id}...")
    
# 2. 加载 Pipeline
# 注意：low_cpu_mem_usage=True 是必须的，防止加载时内存溢出
pipe = WanImageToVideoPipeline.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto", # 自动分配权重到显存/内存
    torch_dtype=torch.float16
)

# 3. 准备输入图像
image = load_image(image_path)
max_area = 480 * 832
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))

print("开始生成视频（这可能需要几分钟，取决于你的显卡）...")

# 4. 执行推理
# Wan2.2 推荐参数：guidance_scale 5.0 左右，steps 40-50
video_frames = pipe(
    image=image,
    prompt=prompt,
    negative_prompt="low quality, blurry, distorted, static",
    num_frames=num_frames,           # 视频帧数
    height=height,
    width=width,
    num_inference_steps=num_inference_steps,  # 步数
    guidance_scale=5.0,
).frames[0]

# 5. 导出视频
export_to_video(video_frames, output_path, fps=fps)
print(f"视频生成成功，已保存至: {output_path}")
    
    