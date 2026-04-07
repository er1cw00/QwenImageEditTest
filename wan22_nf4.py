import torch
from diffusers import WanTransformer3DModel
from transformers import BitsAndBytesConfig

model_id = "magespace/Wan2.2-I2V-A14B-Lightning-Diffusers"
save_path = "./transformer_2"

# 1. 配置量化参数
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 2. 单独加载并量化 Transformer
# device_map="auto" 会确保量化过程在 GPU 上正确执行
print("正在加载并量化 Transformer...")
transformer = WanTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer_2",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 3. 保存到本地
print(f"正在保存至 {save_path}...")
transformer.save_pretrained(save_path)
print("保存完成！")
