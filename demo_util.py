"""Demo file for sampling images from TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""


import torch

from omegaconf import OmegaConf
from modeling.titok import TiTok
from modeling.titok_kl import TiTok_KL
# from modeling.maskgit import ImageBert
import math
from functools import partial

# step scheduler
def fn_LinearWarmup(warmup_steps, step):
    if step < warmup_steps:  # linear warmup
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0

def Scheduler_LinearWarmup(warmup_steps):
    return partial(fn_LinearWarmup, warmup_steps)


def fn_LinearWarmup_CosineDecay(warmup_steps, max_steps, multipler_min, step):
    if step < warmup_steps:  # linear warmup
        return float(step) / float(max(1, warmup_steps))
    else:  # cosine learning rate schedule
        multipler = 0.5 * (math.cos((step - warmup_steps) / (max_steps - warmup_steps) * math.pi) + 1)
        return max(multipler, multipler_min)

def Scheduler_LinearWarmup_CosineDecay(warmup_steps, max_steps, multipler_min):
    return partial(fn_LinearWarmup_CosineDecay, warmup_steps, max_steps, multipler_min)



def get_config_cli():
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf

def get_config(config_path):
    conf = OmegaConf.load(config_path)
    return conf

def get_titok_tokenizer(config):
    tokenizer = TiTok(config)
    tokenizer.load_state_dict(torch.load(config.experiment.tokenizer_checkpoint, map_location="cpu"), strict=False)
    # tokenizer.eval()
    # tokenizer.requires_grad_(False)
    return tokenizer

def get_titok_tokenizer_kl_train(config):
    tokenizer = TiTok_KL(config)
    # tokenizer.load_state_dict(torch.load(config.experiment.tokenizer_checkpoint, map_location="cpu"), strict=False)
    # tokenizer.eval()
    # tokenizer.requires_grad_(False)
    return tokenizer

# def get_titok_generator(config):
#     generator = ImageBert(config)
#     generator.load_state_dict(torch.load(config.experiment.generator_checkpoint, map_location="cpu"))
#     generator.eval()
#     generator.requires_grad_(False)
#     return generator

@torch.no_grad()
def sample_fn(generator,
              tokenizer,
              labels=None,
              guidance_scale=3.0,
              randomize_temperature=2.0,
              num_sample_steps=8,
              device="cuda"):
    generator.eval()
    tokenizer.eval()
    if labels is None:
        # goldfish, chicken, tiger, cat, hourglass, ship, dog, race car, airliner, teddy bear, random
        labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, torch.randint(0, 999, size=(1,))]

    labels = torch.LongTensor(labels).to(device)

    generated_tokens = generator.generate(
        condition=labels,
        guidance_scale=guidance_scale,
        randomize_temperature=randomize_temperature,
        num_sample_steps=num_sample_steps)
    
    generated_image = tokenizer.decode_tokens(
        generated_tokens.view(generated_tokens.shape[0], -1)
    )

    generated_image = torch.clamp(generated_image, 0.0, 1.0)
    generated_image = (generated_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

    return generated_image