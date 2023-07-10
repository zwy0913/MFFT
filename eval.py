import os
import glob
import time

import cv2
import torch
from torchvision.utils import save_image
from tqdm import tqdm

import data_loader
import numpy as np
from model_s2 import MFIFT
from project_parameters import pp
from torch.utils.data import DataLoader
from helper import post_remove_small_objects, to_same_size, guided_filter
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch import einsum

model_num = 34
if model_num == 0:
    MODELPATH = './model.ckpt'
else:
    # MODELPATH = './models/model' + str(model_num) + '.ckpt'
    MODELPATH = 'L:\\Res-CvT\\models\\2022-12-15 16.03.10 models最后就用的这个\\model' + str(model_num) + '.ckpt'
    # MODELPATH = 'C:\\Users\\zheng\\Desktop\\仅transformer\\model' + str(model_num) + '.ckpt'
    # MODELPATH = 'C:\\Users\\zheng\\Desktop\\仅MSFE\\model' + str(model_num) + '.ckpt'
    # MODELPATH = 'C:\\Users\\zheng\\Desktop\\全关\\model' + str(model_num) + '.ckpt'
E_model = MFIFT().to(pp.device)
E_model.load_state_dict(torch.load(MODELPATH))
E_model.eval()

test_dataset = './lytro'
eval_list_A = glob.glob(os.path.join(test_dataset, 'sourceA', '*.*'))
eval_list_B = glob.glob(os.path.join(test_dataset, 'sourceB', '*.*'))
eval_data = data_loader.Eval_Data_loader(eval_list_A, eval_list_B)
eval_loader = DataLoader(dataset=eval_data, batch_size=1, shuffle=False)
cnt = 1
window = torch.tensor([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1],], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).to(pp.device)
padding = 1

running_time = []
for E_A, E_B, O_E_A, O_E_B, W, H in eval_loader:
    torch.no_grad()
    start_time = time.time()
    E_A = E_A.to(pp.device)
    E_B = E_B.to(pp.device)
    O_E_A = O_E_A.to(pp.device)
    O_E_B = O_E_B.to(pp.device)
    # torch.onnx.export(E_model, (E_A, E_B), f="model_best.onnx", verbose=True)
    decision_map = E_model(E_A, E_B).to(pp.device)
    # a = decision_map[0].squeeze(dim=0).clone().detach().cpu().numpy()
    if decision_map.size(2) != O_E_A.size(2) or decision_map.size(3) != O_E_A.size(3):
        decision_map = to_same_size(O_E_A, decision_map)
    decision_map = post_remove_small_objects(decision_map, threshold=0.00).float()

    transf = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.Resize((int(H), int(W)))
        ]
    )

    s = torch.where(decision_map > 0.5, 1., 0.)
    s = F.conv2d(s, window, padding=1)
    s = torch.where(s == 0., 0., s)
    # s = torch.where(s == 1., 128, s)
    # s = torch.where(s == 2., 128, s)
    # s = torch.where(s == 3., 128, s)
    # s = torch.where(s == 4., 128, s)
    # s = torch.where(s == 5., 128, s)
    # s = torch.where(s == 6., 128, s)
    # s = torch.where(s == 7., 128, s)
    # s = torch.where(s == 8., 128, s)
    s = torch.where(s == 9., 255, s)
    s = s[0].squeeze(dim=0).clone().detach().cpu().numpy()
    ss = Image.fromarray(s.astype(dtype=np.uint8))
    # ss.show()
    ss.save('./lytro/fused_img/lytro-' + str(cnt).zfill(2) + '-fm.png')
    # O_E_A = transf(O_E_A)
    # save_image(O_E_A.data.cpu(), './lytro/A/lytro-A-' + str(cnt).zfill(2) + '.png', nrow=1, padding=0)
    # O_E_B = transf(O_E_B)
    # save_image(O_E_B.data.cpu(), './lytro/B/lytro-B-' + str(cnt).zfill(2) + '.png', nrow=1, padding=0)

    decision_map = torch.where(decision_map > 0.5, 1., 0.)

    '''
    decision_map = F.conv2d(decision_map, window, padding=padding)
    decision_map = torch.where(decision_map == 0., 999., decision_map)
    for aa in range(1, 9):
        decision_map = torch.where(decision_map == float(aa), 9999., decision_map)
    decision_map = torch.where(decision_map == 9., 99999., decision_map)
    a = decision_map[0].squeeze(dim=0).clone().detach().cpu().numpy()
    fused_img = torch.cat([decision_map.detach(), decision_map.detach(), decision_map.detach()], dim=1)
    fused_img = torch.where(fused_img == 99999., O_E_A, fused_img)
    fused_img = torch.where(fused_img == 999., O_E_B, fused_img)
    fused_img = torch.where(fused_img == 9999., O_E_C, fused_img)
    '''

    '''
    tt = transforms.ToTensor()
    ti = transforms.ToPILImage()
    decision_map = transf(decision_map)
    ini_fused_img = (O_E_A * decision_map + O_E_B * (1 - decision_map))
    decision_map = einsum('b c w h -> b w h c', decision_map)[0].squeeze(dim=0).clone().detach().cpu().numpy()
    ini_fused_img = einsum('b c w h -> b w h c', ini_fused_img)[0].squeeze(dim=0).clone().detach().cpu().numpy()
    decision_map_guided = guided_filter(ini_fused_img, decision_map, 1, 0.1)
    a = tt(decision_map_guided).to(pp.device)
    sss = a[0].squeeze(dim=0).clone().detach().cpu().numpy()
    out = sss * 255
    out = Image.fromarray(out.astype(dtype=np.uint8))
    out.save('./lytro/fused_img/lytro-' + str(cnt).zfill(2) + '-fm.png')
    # decision_map_guided = einsum('w h c -> c w h', )
    fused_img = (O_E_A * a + O_E_B * (1 - a))
    '''

    D = einsum('b c w h -> b w h c', decision_map)[0].clone().detach().cpu().numpy()
    A = cv2.imread(eval_list_A[cnt - 1])
    B = cv2.imread(eval_list_B[cnt - 1])
    IniF = A * D + B * (1 - D)
    D_GF = guided_filter(IniF, D, 4, 0.1)
    Final_fused = A * D_GF + B * (1 - D_GF)
    print("process_time: {} s".format(time.time() - start_time))
    running_time.append(time.time() - start_time)
    # test = cv2.applyColorMap(Final_fused.astype("uint8"), cv2.COLORMAP_JET)
    # g1 = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    # g2 = cv2.cvtColor(Final_fused, cv2.COLOR_BGR2GRAY)
    # resImg = cv2.absdiff(cv2.cvtColor(A, cv2.COLOR_BGR2GRAY), cv2.cvtColor(Final_fused.astype("uint8"), cv2.COLOR_BGR2GRAY))
    # resImg = Final_fused - A
    # resImg = cv2.normalize(resImg, 1) * 255
    # cv2.imwrite('./lytro/fused_img/lytro_D-' + str(cnt).zfill(2) + '.png', resImg*128)
    cv2.imwrite('./lytro/fused_img/lytro-' + str(cnt).zfill(2) + '.png', Final_fused)
    # fused_img = (O_E_A * decision_map + O_E_B * (1 - decision_map))
    # save_image(fused_img, './lytro/fused_img/lytro-' + str(cnt).zfill(2) + '.png', nrow=1, padding=0)

    cnt += 1

running_time = np.array(running_time)
xxxxx = 0
