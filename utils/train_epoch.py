import os
import time
import numpy as np
from einops import rearrange
import torch
from torch import nn
import matplotlib.pyplot as plt
from utils.criterion import softmax_integral_tensor
from utils.visualizer import volume_log, joints_log, threeviews_log

def train_epoch(
        cfg,
        train_loader,
        model,
        criterion,
        voxel_criterion,
        optimizer,
        epoch,
        output_dir,
        writer,
        begin_time,
        save_model_dir, 
        lr_scheduler,
):
    model.train()

    epoch_begin_time = time.time()
    # iter_time = time.time()
    total_step = (cfg.TRAIN.END_EPOCH - cfg.TRAIN.BEGIN_EPOCH) * len(train_loader)
    loss_100iter = 0
    iter100_time = time.time()
    for step, (input,  vol, target_joints, measFile) in enumerate(train_loader):
        batch_begin_time = time.time()
        global_iter_num = epoch * len(train_loader) + step + 1
        # 1.txt is using for logging 
        np.savetxt('./1.txt', target_joints.cpu().numpy().reshape(24, -1))
        input = input.to(cfg.DEVICE)
        output, feature = model(input)
        target_joints = rearrange(target_joints, 'b n d -> b (n d)').to(cfg.DEVICE)
        target_weights = torch.ones_like(target_joints).to(cfg.DEVICE)
        
        joint_loss = criterion(output, target_joints, target_weights)
        voxel_loss = voxel_criterion(feature.reshape(output.shape[0], -1), vol.reshape(vol.shape[0], -1).to(cfg.DEVICE))
        loss = joint_loss + voxel_loss
        
        loss_100iter+= loss.item()
        print("loss : ", loss.item(), "  ", measFile)

        if global_iter_num % 100 == 0:
            volume_log(vol, './results/volume', f"volume_{global_iter_num}", global_iter_num)
            volume_log(output, './results/volume', f"output_{global_iter_num}", global_iter_num)
            volume_log(feature, './results/volume', f'feature_{global_iter_num}', global_iter_num)

            preds = softmax_integral_tensor(output, cfg.DATASET.NUM_JOINTS, True,
                                           cfg.DATASET.HEATMAP_SIZE[0], cfg.DATASET.HEATMAP_SIZE[1], cfg.DATASET.HEATMAP_SIZE[2])
            
            joints_log(preds[0].reshape(cfg.DATASET.NUM_JOINTS, 3).detach().cpu().numpy(),
                       './results/figure/joints',
                       f"pred_joints_{global_iter_num}",
                       global_iter_num)

            joints_log(target_joints[0].reshape(cfg.DATASET.NUM_JOINTS, 3).detach().cpu().numpy(),
                       './results/figure/joints',
                       f"gt_joints_{global_iter_num}",
                       global_iter_num)

            threeviews_log(feature, './results/figure/threeviews',
                           f'feature_{global_iter_num}', global_iter_num)
            threeviews_log(output, './results/figure/threeviews',
                           f'output_{global_iter_num}', global_iter_num)
            threeviews_log(vol, './results/figure/threeviews',
                           f'volume_{global_iter_num}', global_iter_num)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if global_iter_num % 10000 == 0:
            os.makedirs(save_model_dir, exist_ok=True)
            path = save_model_dir + f"/NlosPose_dict_iter{global_iter_num}.pth"
            if os.path.isdir(save_model_dir) == False:
                os.mkdir(save_model_dir)
            state_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "global_iter_num" : global_iter_num,
            }
            torch.save(state_dict, path)

        if global_iter_num % 100 == 0:

            loss_100iter /= 100.
            print(f'global iter is {global_iter_num}, loss is {loss_100iter}, iter100 time is {time.time()-iter100_time}, \
                   {global_iter_num / total_step *100}%, epoch left {(time.time() - epoch_begin_time)  * (total_step - step) / global_iter_num / 60 / 60} h')
            writer.add_scalar('Train Loss', loss_100iter, global_iter_num)
            loss_100iter = 0
        iter_time = time.time()

        # writer.add_scalar('Train Loss', loss.item(), global_iter_num)

        batch_time = time.time() - epoch_begin_time
        writer.add_scalar('batch_time', batch_time, epoch)
    


