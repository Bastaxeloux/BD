import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

model1 = "stabilityai/stable-diffusion-xl-refiner-1.0"
model2 = "stablediffusionapi/disney-pixar-cartoon"

pipeline = AutoPipelineForImage2Image.from_pretrained(
    model1, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
#pipeline.enable_xformers_memory_efficient_attention()

# prepare image
url = "https://www.cbr.com/wp-content/uploads/2017/04/Spaceman-Cover-.jpg"
url2 = "https://www.cbr.com/wp-content/uploads/2017/04/Judge-Dredd-Mega-City-2-City-Shot.jpg"
init_image = load_image(url)
init_image2 = load_image(url2)

prompt = "atmospheric scenes with a post-apocalyptic setting characterized by desolation, ruins, and decay. Sense of isolation, amidst a landscape strewn with debris and remnants of a once-advanced civilization. contrast between the gritty, dilapidated surroundings, mysterious, eerie blue glow, dark and muted color palette, occasional vibrant accents to highlight key elements. Realistic, slightly fantastical art style, detailed textures, dystopian world"


# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, strength=0.9, guidance_scale=8).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
image.save("bd.png")