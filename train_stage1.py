# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import demo_util
from demo_util import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay
import torch.nn.functional as F

import data
from data.faceshq import FFHQ
from torchvision.utils import save_image
from Network.Taming.models.vqgan import VQModel

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def configure_optimizers(model,opt_config,max_steps):
    # print(opt_config.learning_rate)
    lr = opt_config.learning_rate
    opt_ae = torch.optim.Adam(list(model.encoder.parameters())+
                                list(model.decoder.parameters())+
                                list(model.quantize.parameters())+
                                [model.latent_tokens],
                                # [param for param in model.pixel_quantize.parameters() if param.requires_grad] +
                                # [param for param in model.pixel_decoder.parameters() if param.requires_grad],
                                lr=lr, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(model.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
    
    warmup_steps = max_steps * opt_config.warmup_epochs

    if opt_config.scheduler_type == "linear-warmup":
        scheduler_ae = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_ae, Scheduler_LinearWarmup(warmup_steps)), "interval": "step", "frequency": 1,
        }
        scheduler_disc = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup(warmup_steps)), "interval": "step", "frequency": 1,
        }
    elif opt_config.scheduler_type == "linear-warmup_cosine-decay":
        multipler_min = opt_config.min_learning_rate / opt_config.learning_rate
        scheduler_ae = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_ae, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=opt_config.max_steps, multipler_min=multipler_min)), "interval": "step", "frequency": 1,
        }
        scheduler_disc = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=opt_config.max_steps, multipler_min=multipler_min)), "interval": "step", "frequency": 1,
        }
    else:
        raise NotImplementedError()

    return [opt_ae, opt_disc] , [scheduler_ae, scheduler_disc]

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new 1d model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model       # e.g., large
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        recon_dir = f"{experiment_dir}/reconstructions"  # Stores image reconstructions
        os.makedirs(recon_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        recon_dir = None
        logger = create_logger(None)

    # 广播 recon_dir 给所有进程
    recon_dir = [recon_dir] if rank == 0 else [None]
    dist.broadcast_object_list(recon_dir, src=0)
    recon_dir = recon_dir[0]

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    # recon_dir = dist.broadcast_object_list([recon_dir], src=0)[0]



    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):

    # Setup data:
    # transform = transforms.Compose([
    #     transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    # ])
    dataset =  FFHQ(split='train', resolution=256, is_eval=False)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    max_steps = len(loader)




    config = demo_util.get_config(args.config)
    config_VQGAN = demo_util.get_config(args.config_VQGAN)
    config_VQGAN = config_VQGAN.model.params


        # Step 2: 从配置中提取 VQModel 初始化所需的参数
    ddconfig = config_VQGAN['ddconfig']
    lossconfig = config_VQGAN['lossconfig']
    n_embed = config_VQGAN['n_embed']
    embed_dim = config_VQGAN['embed_dim']

    # 提取需要的配置
    ddconfig = config_VQGAN.ddconfig
    lossconfig = config_VQGAN.lossconfig
    n_embed = config_VQGAN.n_embed
    embed_dim = config_VQGAN.embed_dim

    # 初始化 VQModel
    vqgan = VQModel(
        ddconfig=ddconfig,
        lossconfig=lossconfig,
        n_embed=n_embed,
        embed_dim=embed_dim,
        ckpt_path=ddconfig.get("ckpt", None)  # 如果需要加载预训练权重
    ).to(device)


    

    model = demo_util.get_titok_tokenizer(config).to(device)
    # model.init_from_ckpt_de_and_qu("/private/task/wubin/1d-tokenizer-main/ckpt/decoder_weights.pth",
    #                                "/private/task/wubin/1d-tokenizer-main/ckpt/quantize_weights.pth")
    for param in model.parameters():
        param.requires_grad_(True)   
    for param in model.pixel_quantize.parameters():
        param.requires_grad_(False)
    for param in model.pixel_decoder.parameters():
        param.requires_grad_(False)

    # for param in model.latent_tokens.parameters():
    #     param.requires_grad_(False)

    model_without_ddp = model
    # Note that parameter initialization is done within the DiT constructor
    # ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    # requires_grad(ema, False)
    logger.info(f"Titok Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt_config = config.model.optimizer
    [opt_ae, opt_disc], [scheduler_ae, scheduler_disc] = configure_optimizers(model_without_ddp,opt_config,max_steps)
    optimizer = [opt_ae, opt_disc]



    # Prepare models for training:
    # update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    # ema.eval()  # EMA model should always be in eval mode
    model = DDP(model.to(device), device_ids=[rank])

    # vqgan = VQModel(config_VQGAN)
    # vqgan.init_from_ckpt(config_VQGAN.ckpt)
    for param in vqgan.parameters():
        param.requires_grad_(False)
    # vqgan=DDP(vqgan.to(device), device_ids=[rank])
    

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x in loader:
            # print(x["image"])
            x = x["image"]
            # print(x)
            x = x.to(device)
            # print(model)
            x_1D, result_dict = model.module.encode(x) 
            x_1D = model.module.decoder(x_1D) # 这里的size是[b,1024,16,16]
            
            
            x_1D=torch.einsum(
            'nchw,cd->ndhw', x_1D.softmax(1),
            vqgan.quantize.embedding.weight) # 这里的size是[b,256,16,16]
            reconstructed_image=vqgan.decode(x_1D)

            x_VQ,_,_=vqgan.encode(x) # 这里的size是[b,256,16,16]
            _, _, (_, _, min_encoding_indices)=vqgan.quantize(x_VQ)
            _, _, (_, _, _1D_min_encoding_indices)=vqgan.quantize(x_1D)

            reconstructed_image1=vqgan.decode(x_VQ)

            
            
            # x_VQ=vqgan.decode(x_VQ)
            # reconstructed_image=vqgan.decode(x_1D)
            quantizer_loss = result_dict["quantizer_loss"]


            
            opt_ae.zero_grad()


            # print(last_layer)
            # aeloss,_ = model.module.loss(quantizer_loss,x, reconstructed_image,0,train_steps, last_layer= model.module.get_last_layer(), split="train")
            

            # 版本1，单纯比较中间产物的loss，没有其他任何loss
            aeloss=F.mse_loss(x_1D,x_VQ)

            # 版本2，版本1的loss加上quantizer_loss
            #aeloss=F.mse_loss(x_1D,x_VQ)+quantizer_loss.mean()

            # 版本3：最终重建图片的loss
            # aeloss=F.mse_loss(x,reconstructed_image)

            # 版本4：版本3的loss加上quantizer_loss
            # aeloss,_=model.module.loss(quantizer_loss,x,reconstructed_image,0,train_steps,last_layer= None,split="train")

            # 版本5：版本4的loss加上版本1的loss
            # aeloss,_=model.module.loss(quantizer_loss,x,reconstructed_image,0,train_steps,last_layer= None,split="train")
            # aeloss+=F.mse_loss(x_1D,x_VQ)

            # 版本6：离散索引的loss
            # aeloss=F.mse_loss(min_encoding_indices.float(),_1D_min_encoding_indices.float())



            # print(0)
            # print("Before step:",model.module.encoder.transformer[0].ln_1.weight)
            aeloss.backward()
            opt_ae.step()
            scheduler_ae["scheduler"].step()
          
            # Log loss values:
            running_loss += aeloss.item()
            log_steps += 1
            train_steps += 1

            
            

            if train_steps % args.log_every == 0:

                with torch.no_grad():
                    # Save reconstructions:
                    save_image(x, f"{recon_dir}/{train_steps:07d}input.png", nrow=4)
                    save_image(reconstructed_image, f"{recon_dir}/{train_steps:07d}recon.png", nrow=4)
                    # save_image(reconstructed_image1, f"{recon_dir}/{train_steps:07d}recon_vqgan.png", nrow=4)
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        # "ema": ema.state_dict(),
                        # "opt": opt.state_dict(),
                        # "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()


    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, default="large")
    parser.add_argument("--config", type=str, default="configs/titok_vae_large.yaml")
    parser.add_argument("--config_VQGAN", type=str, default="configs/pretrained_maskgit_VQGAN_model.yaml")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--global-batch-size", type=int, default=18)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    args = parser.parse_args()
    main(args)
