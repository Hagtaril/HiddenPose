import os
import logging
import time
from collections import namedtuple
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn

def create_logger(cfg, cfg_name, phase = 'train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)

    if not root_output_dir.exists():
        print(f'=> creating {root_output_dir}')
        root_output_dir.mkdir()

