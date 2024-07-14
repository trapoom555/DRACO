import sys
import os


parent_dir = os.path.abspath(
    os.path.join(
        os.getcwd(),
        "..",
    )
)

sys.path.append(parent_dir)

import time
import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from data.data import CalvinDataset
from diffusers.training_utils import EMAModel
from diffusion_policy import ConditionalUnet1D
from encoders import TextEncoder, VisionEncoder
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


# parameters
device = 'cuda'
num_epochs = 100
num_diffusion_iters = 100
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
vision_feature_dim = 512
text_feature_dim = 512
lowdim_obs_dim = 7
obs_dim = vision_feature_dim + lowdim_obs_dim
action_dim = 7

# dataloader
dataloader = CalvinDataset(
    '../data/calvin_debug_dataset', 
    obs_horizon=obs_horizon, 
    pred_horizon=pred_horizon
)

# create a network object
vision_encoder = VisionEncoder()
text_encoder = TextEncoder()
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon + text_feature_dim
).to(device)

# scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon'
)

# optimization
ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    power=0.75
)

optimizer = torch.optim.AdamW(
    params=noise_pred_net.parameters(),
    lr=1e-4, weight_decay=1e-6
)

lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

image = torch.zeros((1, obs_horizon,3,96,96)).to(device)

# training loop
with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # data normalized in dataset
                # device transfer
                nimage = torch.Tensor(nbatch['rgb_static']).to(device)
                nagent_pos = torch.Tensor(nbatch['actions']).to(device)
                naction = torch.Tensor(nbatch['next_actions']).to(device)

                # encoder vision features
                image_features = vision_encoder.encode(nimage.flatten(end_dim=0))
                image_features = image_features.reshape(*image.shape[:2],-1)

				# encode text
                text_features = text_encoder.encode(nbatch['text'])

                # concatenate vision feature and low-dim obs
                obs_features = torch.cat([image_features, nagent_pos.unsqueeze(0)], dim=-1)
                
                obs_cond = obs_features.flatten(start_dim=1)
                obs_cond = torch.cat([obs_cond, text_features], dim=-1)

                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (1,), device=device
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps
                )
                
                noisy_actions = noisy_actions.unsqueeze(0)
                noise = noise.unsqueeze(0)

                # predict the noise residual
                noise_pred = noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs_cond)

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(noise_pred_net.parameters())

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)
        tglobal.set_postfix(loss=np.mean(epoch_loss))

# Weights of the EMA model
# is used for inference
ema_nets = noise_pred_net
ema.copy_to(ema_nets.parameters())

timestr = time.strftime("%Y%m%d-%H%M%S")
torch.save(ema_nets.state_dict(), f'../ckpt/{timestr}.pt')