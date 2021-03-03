import cv2
import numpy as np
import time
import torch
from video_mAP import simulate
from transformer import Transformer


if __name__ == "__main__":
    TF = Transformer('compression')
    sim_result = simulate('ucf101-24', data_range=(0,20), TF=TF, C_param=None)
    print(sim_result)