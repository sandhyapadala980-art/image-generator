import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
model_id="runwayml/stable-diffusion-v1-5"
pipe=StableDiffusionPipeline.from_pretrained(
    model_id
).to("cuda")
prompt="A beautiful flowers glow in garden around"
result=pipe(prompt).images[0]
plt.figure(figsize=(8,6))
plt.imshow(result)
plt.axis("off")
plt.show()
