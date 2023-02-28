import torch
from torch import nn
from einops import rearrange, repeat
import matplotlib.pyplot as plt

import time

def train_2d_heatmap(
        cfg,
        train_loader,
        model, 
        criterion,
        optimizer,
        epoch,
        result_dir,
        writer,
):
    model.train()

    end = time.time()
    for step, (input, target) in enumerate(train_loader):
        writer.add_scalar('data time', time.time() - end)
        
        input = input.to(cfg.DEVICE)
        output = model(input)
        output = rearrange(output, 'b n (h w) -> b n h w', h = cfg.DATASET.HEATMAP_SIZE[0])
        global_iter_num = epoch * len(train_loader) + step + 1

        target_weight = torch.ones_like(target)
        loss = criterion(output, target[:,:,:2], target_weight[:,:,:2], writer, global_iter_num)

        print(f'iter : {global_iter_num}        loss is {loss.item()}')
        writer.add_scalar('loss', loss.item(), global_iter_num)

        # if global_iter_num %  == 0:
        # for idx in range(cfg.DATASET.NUM_JOINTS):
        #     writer.add_images(f'joint {idx}', repeat( output[0,idx].detach().cpu().numpy(), 'h w -> h w 3' ))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('batch time', time.time() - end,  global_iter_num)
        end = time.time()