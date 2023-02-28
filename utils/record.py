import logging
import time
import os
from pathlib import Path


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    if not root_output_dir.exists():
        print(f'=> creating {root_output_dir}')
        root_output_dir.mkdir()

    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    dataset = cfg.DATASET.NAME

    final_output_dir = root_output_dir / f'dataset_{dataset}' / f'model_{model}' / f'cfg_name_{cfg_name}'
    print(f'=> creating {final_output_dir}')
    final_output_dir.mkdir(parents = True, exist_ok = True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = f'{cfg_name}_{time_str}_{phase}'
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / f'dataset_{dataset}' / f'model_{model}' / (cfg_name + '_' + time_str)
    print(f'=> creating {tensorboard_log_dir}')
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)





def updata_config(cfg, args):
    cfg.defrost()

    if args.model:
        cfg.MODEL.LOCATION = args.model
    
    if args.test:
        cfg.TEST.TYPE = args.test
 
    if args.log:
        cfg.LOG_DIR = args.log

    if args.data:
        cfg.DATA_DIR = args.data

    if args.device:
        cfg.DEVICE = args.DEVICE

    cfg.freeze()