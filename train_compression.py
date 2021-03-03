import cv2
import numpy as np
import time
import torch
from video_mAP import simulate
from transformer import Transformer


if __name__ == "__main__":
    TF = Transformer('compression')
    # the first 6 parameters are the weights of 6 features (-1,1)
    # the 7th parameter is for the value of the function when score=1 (-1,1)
    # the 8th parameter is the sigma of the function (0,...)
    C_param = np.zeros(8)
    sim_result = simulate('ucf101-24', data_range=(0,20), TF=TF, C_param=C_param)
    print(sim_result)