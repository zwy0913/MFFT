import os
import sys
import glob
import time

import cv2
import torch
import numpy as np

from tqdm import tqdm
from torch import einsum
from Nets.MFFT import MFFT
from Utilities import Consistency
import Utilities.DataLoaderFM as DLr
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from Utilities.CUDA_Check import GPUorCPU
from Utilities.GuidedFiltering import guided_filter
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode



class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)


class Fusion:
    def __init__(self,
                 # modelpath='debug_model.ckpt',
                 modelpath='RunTimeData/2023-06-25 13.21.25/best_network.pth',
                 dataroot='./Datasets/Eval',
                 dataset_name='Lytro',
                 threshold=0.0015,
                 window_size=5,
                 ):
        self.DEVICE = GPUorCPU().DEVICE
        self.MODELPATH = modelpath
        self.DATAROOT = dataroot
        self.DATASET_NAME = dataset_name
        self.THRESHOLD = threshold
        self.window_size = window_size
        self.window = torch.ones([1, 1, self.window_size, self.window_size], dtype=torch.float).to(self.DEVICE)

    def __call__(self, *args, **kwargs):
        if self.DATASET_NAME != None:
            self.SAVEPATH = '/' + self.DATASET_NAME
            self.DATAPATH = self.DATAROOT + '/' + self.DATASET_NAME
            MODEL = self.LoadWeights(self.MODELPATH)
            EVAL_LIST_A, EVAL_LIST_B = self.PrepareData(self.DATAPATH)
            self.FusionProcess(MODEL, EVAL_LIST_A, EVAL_LIST_B, self.SAVEPATH, self.THRESHOLD)
        else:
            print("Test Dataset required!")
            pass

    def LoadWeights(self, modelpath):
        model = MFFT().to(self.DEVICE)
        model.load_state_dict(torch.load(modelpath))
        model.eval()
        # num_params = 0
        # for p in model.parameters():
        #     num_params += p.numel()
        # # print(model)
        # print("The number of model parameters: {} M\n\n".format(round(num_params / 10e5, 6)))
        # from thop import profile, clever_format
        # flops, params = profile(model, inputs=(torch.rand(1, 3, 520, 520).cuda(), torch.rand(1, 3, 520, 520).cuda()))
        # flops, params = clever_format([flops, params], "%.5f")
        # print('flops: {}, params: {}\n'.format(flops, params))
        return model

    def PrepareData(self, datapath):
        eval_list_A = sorted(glob.glob(os.path.join(datapath, 'sourceA', '*.*')))
        eval_list_B = sorted(glob.glob(os.path.join(datapath, 'sourceB', '*.*')))
        return eval_list_A, eval_list_B

    def ConsisVerif(self, img_tensor, threshold):
        Verified_img_tensor = Consistency.Binarization(img_tensor)
        if threshold != 0:
            Verified_img_tensor = Consistency.RemoveSmallArea(img_tensor=Verified_img_tensor, threshold=threshold)
        return Verified_img_tensor

    def FusionProcess(self, model, eval_list_A, eval_list_B, savepath, threshold):
        if not os.path.exists('./Results' + savepath):
            os.mkdir('./Results' + savepath)
        eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
        eval_loader = DataLoader(dataset=eval_data,
                                 batch_size=1,
                                 shuffle=False, )
        eval_loader_tqdm = tqdm(eval_loader, colour='blue', leave=True, file=sys.stdout)
        cnt = 1
        running_time = []
        with torch.no_grad():
            for A, B in eval_loader_tqdm:
                start_time = time.time()

                NetOut, D = model(A, B)
                D = torch.where(D > 0.5, 1., 0.)
                D = self.ConsisVerif(D, threshold)

                if self.window_size > 0:
                    # 做边缘修正
                    decisionmap = F.conv2d(D, self.window, padding=self.window_size // 2)
                    decisionmap = torch.where(decisionmap == 0., 999., decisionmap)
                    for aa in range(1, self.window_size * self.window_size):
                        decisionmap = torch.where(decisionmap == float(aa), 9999., decisionmap)
                    decisionmap = torch.where(decisionmap == self.window_size * self.window_size, 99999., decisionmap)[0]
                    A = read_image(eval_list_A[cnt - 1], mode=ImageReadMode.RGB).to(self.DEVICE)
                    B = read_image(eval_list_B[cnt - 1], mode=ImageReadMode.RGB).to(self.DEVICE)
                    fused_img = torch.cat([decisionmap.detach(), decisionmap.detach(), decisionmap.detach()], dim=0)
                    fused_img = torch.where(fused_img == 99999., A, fused_img)
                    fused_img = torch.where(fused_img == 999., B, fused_img)
                    fused_img = torch.where(fused_img == 9999., NetOut[0] * 255, fused_img)
                    fused_img = einsum('c w h -> w h c', fused_img).clone().detach().cpu().numpy()
                    cv2.imwrite('./Results' + savepath + '/' + self.DATASET_NAME + '-' + str(cnt).zfill(2) + '.png',
                                cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR))
                else:
                    # 不做边缘修正
                    D = einsum('c w h -> w h c', D[0]).clone().detach().cpu().numpy()
                    A = cv2.imread(eval_list_A[cnt - 1])
                    B = cv2.imread(eval_list_B[cnt - 1])
                    IniF = A * D + B * (1 - D)
                    D_GF = guided_filter(IniF, D, 4, 0.1)
                    Final_fused = A * D_GF + B * (1 - D_GF)
                    cv2.imwrite('./Results' + savepath + '/' + self.DATASET_NAME + '-' + str(cnt).zfill(2) + '.png', Final_fused)
                cnt += 1
                # print("process_time: {} s".format(time.time() - start_time))
                running_time.append(time.time() - start_time)
        running_time_total = 0
        for i in range(len(running_time)):
            print("process_time: {} s".format(running_time[i]))
            if i != 0:
                running_time_total += running_time[i]
        print("\navg_process_time: {} s".format(running_time_total / (len(running_time) - 1)))
        print("\nResults are saved in: " + "./Results" + savepath)


if __name__ == '__main__':
    f = Fusion()
    f()
