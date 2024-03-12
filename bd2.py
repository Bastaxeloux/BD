import torch
from diffusers import (
    StableDiffusionXLPipeline, 
    EulerAncestralDiscreteScheduler,
    AutoencoderKL
)

# Load VAE component
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", 
    torch_dtype=torch.float16
)

# Configure the pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "cagliostrolab/animagine-xl-3.0", 
    vae=vae,
    torch_dtype=torch.float16, 
    use_safetensors=True, 
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to('cuda')

# Define prompts and generate image
case1 = "close-up of a destroyed ground, littered with debris, twisted metal and equipment once so technologically advanced and now futile in this world where the only concern is to feed."
case2 = "very close-up of military boots and the legs of the character running across the image, sensation of speed"
case3 = "in the foreground solitary figure standing in the middle of the ruins, back to the viewer. He is dressed in tattered clothes, with a hood over his head."
case4 = "Close-up of the character's hand reaching toward something in the distance, a faint glow emanating from its palm."
case6 = "Close-up on the character's face,under the dark of his hood."
ambiance = "black and white image, Akira style, Katsuhiro Ōtomo in 1988, desolation, ruins, dystopian world "
negative_prompt = "color, nsfw, lowres, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, artist name"

image = pipe(
    case6+ambiance, 
    negative_prompt=negative_prompt, 
    width=1376,
    height=992,
    guidance_scale=7,
    num_inference_steps=36
).images[0].save("case6.png")

dechet1 = "desolation, ruins, dystopian world, sense of isolation, amidst a landscape strewn with debris and remnants of a once-advanced civilization in the background"
dechet = "atmospheric scenes with a post-apocalyptic setting characterized by , decay, slightly fantastical art style, , . contrast between the gritty, dilapidated surroundings, mysterious, occasional vibrant accents to highlight key elements. bad anatomy, bad hands, "
format = "832,1216", "752,1376"