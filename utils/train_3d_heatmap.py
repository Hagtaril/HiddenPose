import torch
from torch import nn

import time
from einops import rearrange


def train_3d_heatmap(
        cfg,
        train_loader,
        model,
        criterion,
        optimizer,
        epoch,
        final_output_dir,
        writer,
):
    model.train()

    end = time.time()
    for step, (input, target) in enumerate(train_loader):
        global_iter_num = epoch * len(train_loader) + step + 1

        writer.add_scalar('data time', time.time() - end,global_iter_num)

        input = input.to(cfg.DEVICE)
        output = model(input)

        gt = rearrange(target, 'b n d -> b (n d)').to(cfg.DEVICE)
        gt_vis = torch.ones_like(gt).to(cfg.DEVICE)


        loss = criterion(output, gt, gt_vis, writer, global_iter_num)

        print(f'iter : {global_iter_num}        loss is {loss.item()}')
        writer.add_scalar('loss', loss.item(), global_iter_num)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('batch time', time.time() - end, global_iter_num)
        end = time.time()


