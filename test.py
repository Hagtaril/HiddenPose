import os
from einops import rearrange, repeat
from tensorboard.compat.proto.event_pb2 import TaggedRunMetadata
import torch
import argparse
import torch.optim
import numpy as np
import scipy.io as scio
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.parallel
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


from config.config_noise import _C as cfg
from utils.nlos_pose_dataloader import NlosPoseDataset
from utils.loadrealdata import load_realdata
from models.NlosPose import NlosPose
from models.optimizer import get_optimizer


from utils.record import updata_config
from utils.criterion import softmax_integral_tensor
from utils.visualizer import volume_log, joints_log, threeviews_log

import time


def parse_args():
    parser = argparse.ArgumentParser(description='Train nlospose network')

    parser.add_argument(
        '--model',
        help='model directory',
        type=str,
        default='./trained_weights/NlosPose_final_dict_14.pth'
    )
    parser.add_argument(
        '--test',
        help='test options: test_realdata, test_pose_v2, test_fk',
        type=str,
        default='test_realdata'
    )
    parser.add_argument(
        '--log',
        help='log directory',
        type=str,
        default='./log'
    )
    parser.add_argument(
        '--data',
        help='data directory',
        type=str,
        default=''
    )
    parser.add_argument(
        '--device',
        help='index of GPU to use',
        type=int,
        default=0
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

    updata_config_t128_128x128(cfg)

    cfg.defrost()
    cfg.PHASE = 'test'
    cfg.DATASET.PHASE = 'test'
    cfg.DEVICE = 0
    cfg.freeze()


    model = NlosPose(cfg)
    model.to(cfg.DEVICE)


    best_perf = 0.0
    last_epoch = -1
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    optimizer = get_optimizer(cfg, model)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR, last_epoch=last_epoch)

    log_dir = cfg.LOG_DIR + f'/{time.gmtime().tm_mon}_{time.gmtime().tm_mday}_{cfg.LOSS.TYPE}_{cfg.MODEL.COORD_REPRESENTATION}'
    writer = SummaryWriter(log_dir)

    save_model_dir = cfg.RESULT.FINAL_OUTPUT_DIR + f'/{time.gmtime().tm_mon}_{time.gmtime().tm_mday}_{cfg.LOSS.TYPE}_{cfg.MODEL.COORD_REPRESENTATION}'

    # options : which data to be test 
    test_realdata = False # real data from hiddenpose
    test_pose_v2 = False # test dataset in hiddenpose
    test_fk = False # real data form fk
    if cfg.TEST.TYPE == 'test_realdata':
        test_realdata = True
    elif cfg.TEST.TYPE == 'test_pose_v2':
        test_pose_v2 = True
    elif cfg.TEST.TYPE == 'test_fk':
        test_fk = True

    # load model
    if test_realdata or test_pose_v2 or test_fk:
        # checkpoint = torch.load("checkpoints/4_6_L2JointLocationLoss_sa-simdr/NlosPose_final_dict_14.pth")
        checkpoint = torch.load(cfg.MODEL.LOCATION, map_location='cuda:0')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    model.eval()
    
    if test_fk == True:
        fk_data_path = '/data2/nlospose/fk_64_to_128'
        fk_files = os.listdir(fk_data_path)
        with torch.no_grad():
            idx = 0
            for data_file in fk_files:
                data_path = os.path.join(fk_data_path, data_file)
                input = scio.loadmat(data_path)
                input = torch.tensor(input['meas'])
                # input = rearrange(input, '(b1 b2 t) h w -> b1 b2 t h w', b1 = 1, b2=1)
                K = 2
                for k in range(K):
                    input = (input[:,:,::2] + input[:,:,1::2]) / 2
                input = input[:,:,64:64+128]
                input = rearrange(input, 'h w t -> t h w')
                input = repeat(input, 't h w -> b c t h w', b = 2, c = 1).float().to(cfg.DEVICE)
                output, feature = model(input)
                preds = softmax_integral_tensor(output, cfg.DATASET.NUM_JOINTS, True,
                                                        cfg.DATASET.HEATMAP_SIZE[0], cfg.DATASET.HEATMAP_SIZE[1], cfg.DATASET.HEATMAP_SIZE[2])
                            # preds = torch.cat((x, y, z), dim=2)
                            # preds = preds.reshape((preds.shape[0], 24 * 3))

                name = data_file.split('.')[0]
                threeviews_log(feature, './plot_fk/threeviews', f'name_{name}', idx)
                joints_log(preds[0].reshape(cfg.DATASET.NUM_JOINTS, 3).detach().cpu().numpy(),
                            './plot_fk/joints',
                            f"{idx}_pred_real_joints_{name}",
                            113)
                idx = idx + 1


    elif test_realdata:
        #  test our collected data
        idx = 3
        if idx == 0:
            input = load_realdata("data/lct256_human.mat")
            name = "lct256_human"
        elif idx == 1:
            input = load_realdata("data/1-12-2031_lct256_human.mat")
            name = "1-12-2031_lct256_human"
        elif idx == 2:
            input = load_realdata("data/1-13-1546_lct256_human.mat")
            name = "1-13-1546_lct256_human"
        elif idx == 3:
            input = load_realdata("data/1-14-0147_lct256_human.mat")
            name = "idx==3_1-14-0147_lct256_human"
        
        with torch.no_grad():

            input = rearrange(input, '(b1 b2 t) h w -> b1 b2 t h w', b1 = 1, b2=1)
            input = repeat(input, 'b c t h w -> (b b1) c t h w', b1 = 2).float().to(cfg.DEVICE)
            
                
            output, feature = model(input)

            preds = softmax_integral_tensor(output, cfg.DATASET.NUM_JOINTS, True,
                                                cfg.DATASET.HEATMAP_SIZE[0], cfg.DATASET.HEATMAP_SIZE[1], cfg.DATASET.HEATMAP_SIZE[2])

            threeviews_log(feature, './plot_realdata/threeviews', f'name_{idx}', idx)
            joints_log(preds[0].reshape(cfg.DATASET.NUM_JOINTS, 3).detach().cpu().numpy(),
                            './plot_realdata/joints',
                            f"{idx}_pred_real_joints_{name}",
                            113)
    elif test_pose_v2:
        test_data = NlosPoseDataset(cfg, cfg.DATASET.TEST_PATH)
        test_loader = DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WOKERS,
                                pin_memory=True)
        with torch.no_grad():
            for step, (input, vol, target_joints, person_id) in enumerate(test_loader):
                input = input.to(cfg.DEVICE)
                output, feature = model(input)
                
                preds = softmax_integral_tensor(output, cfg.DATASET.NUM_JOINTS, True,
                                            cfg.DATASET.HEATMAP_SIZE[0], cfg.DATASET.HEATMAP_SIZE[1], cfg.DATASET.HEATMAP_SIZE[2])
                os.makedirs("./test_results/", exist_ok=True)
                pred_file_0 = f"preds_{person_id[0]}"
                joints_log(preds[0].reshape(cfg.DATASET.NUM_JOINTS, 3).detach().cpu().numpy(),
                        './test_results/joints',
                        pred_file_0,
                        )
                pred_file_1 = f"preds_{person_id[1]}"
                joints_log(preds[1].reshape(cfg.DATASET.NUM_JOINTS, 3).detach().cpu().numpy(),
                        './test_results/joints',
                        pred_file_1,
                        )
                
                gt = rearrange(target_joints, 'b n d -> b (n d)')
                gt_file_0 = f"gt_{person_id[0]}" 
                
                joints_log(gt[0].reshape(cfg.DATASET.NUM_JOINTS, 3).detach().cpu().numpy(),
                        './test_results/joints',
                        gt_file_0,
                        )

                gt_file_1 = f"gt_{person_id[1]}" 
                joints_log(gt[1].reshape(cfg.DATASET.NUM_JOINTS, 3).detach().cpu().numpy(),
                        './test_results/joints',
                        gt_file_1,
                        )



    print("finished")



if __name__ == '__main__':
    main()

