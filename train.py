import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import argparse
import torch.optim
import numpy as np
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.parallel
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from config.config_noise import _C as cfg
from utils.nlos_pose_dataloader import NlosPoseDataset
from utils.nlos_dataloader import NlosDataset
from models.NlosPose import NlosPose
from models.optimizer import get_optimizer

from models.feature_extraction import FeatureExtraction
from utils.criterion import NMTNORMCritierion, L2JointLocationLoss, JointsMSELoss, BCEDiceLoss
from utils.train_simdr import train_simdr
from utils.train_epoch import train_epoch
from utils.train_3d_heatmap import train_3d_heatmap
from utils.train_2d_heatmap import train_2d_heatmap
from utils.record import updata_config, create_logger

import time


def parse_args():
    parser = argparse.ArgumentParser(description='HiddenPose network args')

    parser.add_argument(
        '--modelDir',
        help='model directory',
        type=str,
        default=''
    )
    parser.add_argument(
        '--logDir',
        help='log directory',
        type=str,
        default=''
    )
    parser.add_argument(
        '--dataDir',
        help='data directory',
        type=str,
        default=''
    )
    parser.add_argument(
        '--DEVICE',
        help='index of GPU to use',
        type=int,
        default=0
    )

    parser.add_argument(
        '--PHASE',
        type = str,
        default = 'train',
        help = ' \'eval\' or \'continue_train\'  or \'train\'  or \'test\' ',
    )
    

    args = parser.parse_args()

    return args


def updata_config_t128_128x128(cfg):
    cfg.defrost()

    cfg.MODEL.BIN_LEN = cfg.MODEL.BIN_LEN * 4
    cfg.MODEL.TIME_SIZE = cfg.MODEL.TIME_SIZE // 4
    cfg.MODEL.IMAGE_SIZE = [128, 128]
    cfg.MODEL.GRID_DIM = 128
    cfg.MODEL.TIME_SIZE = 128

    cfg.freeze()


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)



def main():
    seed_everything(410)
    args = parse_args()
    updata_config(cfg, args)
    # upsample args update
    updata_config_t128_128x128(cfg)

    # cudnn related setting
    # cudnn.benchmark = cfg.CUDNN.BENCHMARKcd
    # cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    # cudnn.enabled = cfg.CUDNN.ENABLED

    model = NlosPose(cfg)
    model.to(cfg.DEVICE)
    # model = torch.nn.DataParallel(model.to('cuda:1'), device_ids=[0,1], output_device=[1])
    # device = 'cuda'

    for name, param in model.named_parameters():
        print(name, '\t\t\t', param.shape)

    # if cfg.DATASET.NAME == "NlosDataset":
    #     train_data = NlosDataset(cfg, cfg.DATASET.TRAIN_PATH)
    if cfg.DATASET.NAME == "NlosPoseDataset":
        train_data = NlosPoseDataset(cfg, cfg.DATASET.TRAIN_PATH)
    train_loader = DataLoader(train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WOKERS,
                              pin_memory=True)

    # count number of parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {pytorch_total_params}")

    # cfg.LOSS.TYPE == 'L2JointLocationLoss':  # 3D
    criterion = L2JointLocationLoss(output_3d=True)

    voxel_criterion = BCEDiceLoss()
    # TODO
    # Data normalize

    best_perf = 0.0
    last_epoch = -1
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    optimizer = get_optimizer(cfg, model)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR, last_epoch=last_epoch)

    log_dir = cfg.LOG_DIR + f'/{time.gmtime().tm_mon}_{time.gmtime().tm_mday}_{cfg.LOSS.TYPE}_{cfg.MODEL.COORD_REPRESENTATION}'
    writer = SummaryWriter(log_dir)

    save_model_dir = cfg.RESULT.FINAL_OUTPUT_DIR + f'/{time.gmtime().tm_mon}_{time.gmtime().tm_mday}_{cfg.LOSS.TYPE}_{cfg.MODEL.COORD_REPRESENTATION}'



    begin_time = time.time()
    if cfg.PHASE == 'continue_train':
        checkpoint = torch.load("./checkpoints/3_25_L2JointLocationLoss_sa-simdr/NlosPose_final_dict_9.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        begin_epoch = checkpoint['epoch'] + 1
        for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
            epoch_begin_time = time.time()
            lr_scheduler.step()
            if cfg.MODEL.COORD_REPRESENTATION == '3DHeatmap':
                train_epoch(cfg, train_loader, model, criterion, voxel_criterion, optimizer, epoch, cfg.RESULT.FINAL_OUTPUT_DIR, writer,
                            begin_time, save_model_dir, lr_scheduler)
        
            end_time = time.time()

            if epoch % 1 == 0:
                path = save_model_dir + f"/NlosPose_epoch{epoch}.pth"
                if os.path.isdir(save_model_dir) == False:
                    os.makedirs(save_model_dir, exist_ok=True)

                # torch.save(model, path)
            epoch_time = end_time - epoch_begin_time
            print(f'epoch {epoch} used {epoch_time}, left {epoch_time * (cfg.TRAIN.END_EPOCH - epoch - 1) / 60 / 60} hours')

            if epoch % 1 == 0:
                path = save_model_dir + f"/NlosPose_final_dict_{epoch}.pth"
                if os.path.isdir(save_model_dir) == False:
                    os.mkdir(save_model_dir)
                state_dict = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                }
                torch.save(state_dict, path)


    # train  
    else: 
        
        for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
            epoch_begin_time = time.time()
            lr_scheduler.step()

            if cfg.MODEL.COORD_REPRESENTATION == '3DHeatmap':
                train_epoch(cfg, train_loader, model, criterion, voxel_criterion, optimizer, epoch, cfg.RESULT.FINAL_OUTPUT_DIR, writer,
                            begin_time, save_model_dir, lr_scheduler)
        
            end_time = time.time()

            if epoch % 1 == 0:
                path = save_model_dir + f"/NlosPose_epoch{epoch}.pth"
                if os.path.isdir(save_model_dir) == False:
                    os.makedirs(save_model_dir, exist_ok=True)

                # torch.save(model, path)
            epoch_time = end_time - epoch_begin_time
            print(f'epoch {epoch} used {epoch_time}, left {epoch_time * (cfg.TRAIN.END_EPOCH - epoch - 1) / 60 / 60} hours')

            if epoch % 1 == 0:
                path = save_model_dir + f"/NlosPose_final_dict_{epoch}.pth"
                if os.path.isdir(save_model_dir) == False:
                    os.mkdir(save_model_dir)
                state_dict = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                }
                torch.save(state_dict, path)

        path = save_model_dir + f"\\NlosPose_final.pth"
        torch.save(model, path)
        print(f'finished training')

        cfg_path = save_model_dir + f'/NlosPose.log'
        fh = open(cfg_path, 'w', encoding='utf-8')
        fh.write(str(cfg))
        fh.close()




if __name__ == '__main__':
    main()
    print('finished')
