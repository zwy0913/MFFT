import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode
from Utilities.CUDA_Check import GPUorCPU
from Loss_funcs.SSIM_Torch import ssim
from torchvision.models.vgg import vgg16
from Utilities.General import denorm

DEVICE = GPUorCPU().DEVICE

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernelx = torch.cat([kernelx, kernelx, kernelx], dim=1)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        kernely = torch.cat([kernely, kernely, kernely], dim=1)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        sobelx = (sobelx+4) / 8
        sobely = (sobely+4) / 8
        return sobelx * 0.5 + sobely * 0.5


class Loss_Intensity(nn.Module):
    def __init__(self):
        super(Loss_Intensity, self).__init__()
        self.loss_abs = nn.L1Loss()

    def forward(self, image_fused, GT):
        # intensity_joint = torch.mean(torch.cat([image_A, image_B]), dim=0)
        # print('intensity_joint shape:', intensity_joint.shape)
        # intensity_joint = torch.max(denorm(image_A), denorm(image_B))
        # s1 = image_A[0][0].clone().detach().cpu().numpy()
        # s2 = image_B[0][0].clone().detach().cpu().numpy()
        # s3 = intensity_joint[0][0].clone().detach().cpu().numpy()
        Loss_intensity = self.loss_abs(image_fused, GT)
        return Loss_intensity


class Loss_Gradient(nn.Module):
    def __init__(self):
        super(Loss_Gradient, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_fused, GT):
        # gradient_A = self.sobelconv(denorm(image_A))
        # gradient_B = self.sobelconv(denorm(image_B))
        gradient_fused = self.sobelconv(image_fused)
        gradient_GT = self.sobelconv(GT)
        # gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_Gradient = F.l1_loss(gradient_fused, gradient_GT)
        return Loss_Gradient


class Loss_SSIM(nn.Module):
    def __init__(self):
        super(Loss_SSIM, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self,image_fused, GT):
        # image_A = denorm(image_A)
        # image_B = denorm(image_B)
        # gradient_A = self.sobelconv(image_A)
        # gradient_B = self.sobelconv(image_B)
        # weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        # weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        # Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
        Loss_SSIM = ssim(image_fused, GT).item()
        return Loss_SSIM


class Loss_Color(nn.Module):

    def __init__(self):
        super(Loss_Color, self).__init__()
        self.sobelconv = Sobelxy()

    def cos_sim(self, a, b, eps=1e-8):
        vector = torch.multiply(a, b)
        up = torch.sum(vector)
        down = torch.sqrt(torch.sum(torch.square(a))) * torch.sqrt(torch.sum(torch.square(b)))
        down = torch.max(down, torch.tensor(eps).to(DEVICE))
        if up > down:
            return torch.tensor(1.)
        else:
            theta = torch.acos(up / down)  # 弧度值
            result = 1 - theta / torch.pi / 2
            return result


    def forward(self, image_fused, GT):
        b, c, h, w = image_fused.shape
        # image_A = denorm(image_A)
        # image_B = denorm(image_B)
        # gradient_A = self.sobelconv(image_A)
        # gradient_B = self.sobelconv(image_B)
        # weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        # weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        # # s1 = image_A[:, 0, :, :].clone().detach().cpu().numpy()
        # # s2 = image_fused[:, 0, :, :].clone().detach().cpu().numpy()
        # Ar, Ag, Ab = torch.chunk(image_A, 3, dim=1)
        # Br, Bg, Bb = torch.chunk(image_B, 3, dim=1)
        # Fr, Fg, Fb = torch.chunk(image_fused, 3, dim=1)
        # # s = self.cos_sim(Ar, Ar)
        # cos_sim_A = self.cos_sim(Ar, Fr) / 3 + self.cos_sim(Ag, Fg) / 3 + self.cos_sim(Ab, Fb) / 3
        # cos_sim_B = self.cos_sim(Br, Fr) / 3 + self.cos_sim(Bg, Fg) / 3 + self.cos_sim(Bb, Fb) / 3
        # Loss_Color = weight_A * cos_sim_A + weight_B * cos_sim_B
        Fr, Fg, Fb = torch.chunk(image_fused, 3, dim=1)
        Gr, Gg, Gb = torch.chunk(GT, 3, dim=1)
        Loss_Color = self.cos_sim(Fr, Gr) / 3 + self.cos_sim(Fg, Gg) / 3 + self.cos_sim(Fb, Gb) / 3
        return Loss_Color


class fusion_loss_mff(nn.Module):
    def __init__(self):
        super(fusion_loss_mff, self).__init__()
        self.L_Grad = Loss_Gradient()
        self.L_Inten = Loss_Intensity()
        self.L_SSIM = Loss_SSIM()
        self.L_Color = Loss_Color()

        # print(1)

    def forward(self, image_fused, GT):
        loss_l1 = 0.7 * self.L_Inten(image_fused, GT)
        loss_gradient = 0.1 * self.L_Grad(image_fused, GT)
        loss_SSIM = 0.1 * (1 - self.L_SSIM(image_fused, GT))
        loss_color = 0.1 * (1 - self.L_Color(image_fused, GT))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM + loss_color
        return fusion_loss


if __name__ == '__main__':
    GT = read_image('1.jpg', mode=ImageReadMode.RGB).to('cuda') / 255
    test_tensor_B = torch.zeros((1, 3, 149, 162)).to('cuda')
    test_tensor_F = read_image('1.jpg', mode=ImageReadMode.RGB).to('cuda') / 255
    loss_mff = fusion_loss_mff()
    print(loss_mff(test_tensor_F.unsqueeze(0), GT.unsqueeze(0)))
