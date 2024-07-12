import demo_util
import numpy as np
import torch
from PIL import Image
import imagenet_classes
from IPython.display import display

torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)

config = demo_util.get_config("configs/titok_l32.yaml")


titok_tokenizer = demo_util.get_titok_tokenizer(config)
# titok_generator = demo_util.get_titok_generator(config)

device = "cuda"

titok_tokenizer = titok_tokenizer.to(device)
# titok_generator = titok_generator.to(device)


def tokenize_and_reconstruct(img_path):
    original_image = Image.open(img_path)
    image = torch.from_numpy(np.array(original_image).astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
    encoded_tokens = titok_tokenizer.encode(image.to(device))[1]["min_encoding_indices"]
    reconstructed_image = titok_tokenizer.decode_tokens(encoded_tokens)
    reconstructed_image = torch.clamp(reconstructed_image, 0.0, 1.0)
    reconstructed_image = (reconstructed_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
    reconstructed_image = Image.fromarray(reconstructed_image)
    print(f"Input Image is represented by codes {encoded_tokens} with shape {encoded_tokens.shape}")
    print("orginal image:")
    display(original_image)
    print("reconstructed image:")
    display(reconstructed_image)

img_path = "assets/ILSVRC2012_val_00010240.png"
original_image = Image.open(img_path)
image = torch.from_numpy(np.array(original_image).astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
encoded_tokens = titok_tokenizer.encode(image.to(device))[1]["min_encoding_indices"]
reconstructed_image = titok_tokenizer.decode_tokens(encoded_tokens)
reconstructed_image = torch.clamp(reconstructed_image, 0.0, 1.0)
reconstructed_image = (reconstructed_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
reconstructed_image = Image.fromarray(reconstructed_image)
