import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, einsum
from torch.nn import functional as F


# BR
class Basic_Residual_Module(nn.Module):
    def __init__(self, input_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(48, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = x + residual
        x = self.relu(x)
        return x


# MSFE
class Multi_Scale_Feature_Extract_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.Initial = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.dilatation_conv_1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.dilatation_conv_2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=2, stride=1, dilation=2),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.dilatation_conv_3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=4, stride=1, dilation=4),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.dilatation_conv_4 = nn.Sequential(
            nn.Conv2d(16 * 3, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.Initial(x)
        x1 = self.dilatation_conv_1(x)
        x2 = self.dilatation_conv_2(x)
        x3 = self.dilatation_conv_3(x)
        concatenation = torch.cat([x1, x2, x3], dim=1)
        x4 = self.dilatation_conv_4(concatenation)
        x = x4 + residual
        x = self.relu(x)
        return x



























class CS_LayerNorm(nn.Module):  # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class CS_PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = CS_LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class CS_DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, scale_factor, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      dilation=scale_factor, bias=bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class CS_Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        padding = proj_kernel // 2
        self.heads = heads
        scale_factor = [1, 2, 4]
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.Multi_scale_Token_Embeding = nn.ModuleList([])
        for i in range(len(scale_factor)):
            self.Multi_scale_Token_Embeding.append(nn.ModuleList([
                CS_DepthWiseConv2d(dim, inner_dim, proj_kernel, padding=scale_factor[i], stride=1,
                                   scale_factor=scale_factor[i], bias=False),
                CS_DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, padding=scale_factor[i], stride=kv_proj_stride,
                                   scale_factor=scale_factor[i], bias=False),
            ]))

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim * 3, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        # b, d, h, w = x.shape
        Q, K, V = [], [], []
        for to_q, to_kv in self.Multi_scale_Token_Embeding:
            q = to_q(x)
            k, v = to_kv(x).chunk(2, dim=1)
            q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h=h), (q, k, v))
            Q.append(q)
            K.append(k)
            V.append(v)

        dots0 = einsum('b i d, b j d -> b i j', Q[2], K[1]) * self.scale
        attn0 = self.attend(dots0)
        attn0 = self.dropout(attn0)
        out0 = einsum('b i j, b j d -> b i d', attn0, V[0])
        out0 = rearrange(out0, '(b h) (x y) d -> b (h d) x y', h=h, y=y)

        dots1 = einsum('b i d, b j d -> b i j', Q[0], K[2]) * self.scale
        attn1 = self.attend(dots1)
        attn1 = self.dropout(attn1)
        out1 = einsum('b i j, b j d -> b i d', attn1, V[1])
        out1 = rearrange(out1, '(b h) (x y) d -> b (h d) x y', h=h, y=y)

        dots2 = einsum('b i d, b j d -> b i j', Q[1], K[0]) * self.scale
        attn2 = self.attend(dots2)
        attn2 = self.dropout(attn2)
        out2 = einsum('b i j, b j d -> b i d', attn2, V[2])
        out2 = rearrange(out2, '(b h) (x y) d -> b (h d) x y', h=h, y=y)

        out = torch.cat([out0, out1, out2], dim=1)

        return self.to_out(out)


class CS_FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)\


# CSIT
class CS_Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head=64, mlp_mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CS_PreNorm(dim, CS_Attention(dim, proj_kernel=proj_kernel, kv_proj_stride=kv_proj_stride, heads=heads,
                                             dim_head=dim_head, dropout=dropout)),
                CS_PreNorm(dim, CS_FeedForward(dim, mlp_mult, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class CS_conv3x3(nn.Module):
    "3x3 convolution with padding"

    def __init__(self, input_dim, output_dim, stride=1):
        super().__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_dim, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv3x3(x)
        return x


# CSVA
class Cross_Scale_Visual_Attention_Module(nn.Module):
    def __init__(self, input_dim=32, dropout=0.):
        super().__init__()
        self.CvT = nn.Sequential(

            nn.Conv2d(32, 48, kernel_size=7, stride=4, padding=3),
            CS_LayerNorm(48),
            CS_Transformer(dim=48, proj_kernel=3, kv_proj_stride=2, heads=1, depth=1, mlp_mult=2,
                                 dropout=dropout),

            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            CS_LayerNorm(64),
            CS_Transformer(dim=64, proj_kernel=3, kv_proj_stride=2, heads=2, depth=1, mlp_mult=4,
                                 dropout=dropout),

        )
        self.relu = nn.ReLU(inplace=True)
        self.conv3x3_1 = CS_conv3x3(input_dim=64, output_dim=48)
        self.conv3x3_2 = CS_conv3x3(input_dim=48, output_dim=32)

    def forward(self, x):
        residual = x
        x = self.CvT(x)
        x = F.interpolate(x, mode='bilinear', size=(x.shape[2] * 2, x.shape[3] * 2))
        x = self.conv3x3_1(x)
        x = F.interpolate(x, mode='bilinear', size=(residual.shape[2], residual.shape[3]))
        x = self.conv3x3_2(x)
        x = x + residual
        x = self.relu(x)
        return x






















class CD_LayerNorm(nn.Module):  # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class CD_PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = CD_LayerNorm(dim)
        self.fn = fn

    def forward(self, x, y, **kwargs):
        x = self.norm(x)
        y = self.norm(y)
        return self.fn(x, y, **kwargs)


class CD_DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      bias=bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class CD_Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = CD_DepthWiseConv2d(dim, inner_dim, proj_kernel, padding=padding, stride=1, bias=False)
        self.to_kv = CD_DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, padding=padding, stride=kv_proj_stride,
                                        bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x, y):
        shapex = x.shape
        bx, nx, _x, wx, hx = *shapex, self.heads
        qx = self.to_q(x)
        kx, vx = self.to_kv(x).chunk(2, dim=1)
        qx, kx, vx = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h=hx), (qx, kx, vx))
        shapey = y.shape
        by, ny, _y, wy, hy = *shapey, self.heads
        qy = self.to_q(y)
        ky, vy = self.to_kv(y).chunk(2, dim=1)
        qy, ky, vy = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h=hy), (qy, ky, vy))

        dotsx = einsum('b i d, b j d -> b i j', qx, ky) * self.scale
        attnx = self.attend(dotsx)
        attnx = self.dropout(attnx)
        outx = einsum('b i j, b j d -> b i d', attnx, vy)
        outx = rearrange(outx, '(b h) (x y) d -> b (h d) x y', h=hx, y=wx)

        dotsy = einsum('b i d, b j d -> b i j', qy, kx) * self.scale
        attny = self.attend(dotsy)
        attny = self.dropout(attny)
        outy = einsum('b i j, b j d -> b i d', attny, vx)
        outy = rearrange(outy, '(b h) (x y) d -> b (h d) x y', h=hy, y=wy)

        # out = torch.cat([outx, outy], dim=1)

        return self.to_out(outx), self.to_out(outy)


class CD_FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x, y):
        return self.net(x), self.net(y)


# CDIT
class CD_Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head=64, mlp_mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CD_PreNorm(dim, CD_Attention(dim, proj_kernel=proj_kernel, kv_proj_stride=kv_proj_stride, heads=heads,
                                          dim_head=dim_head, dropout=dropout)),
                CD_PreNorm(dim, CD_FeedForward(dim, mlp_mult, dropout=dropout))
            ]))

    def forward(self, x, y):
        for attn, ff in self.layers:
            x1, y1 = attn(x, y)
            x2 = x1 + x
            y2 = y1 + y
            x3, y3 = ff(x2, y2)
            x4 = x3 + x2
            y4 = y3 + y2
        return x4, y4


class CD_conv3x3(nn.Module):
    "3x3 convolution with padding"

    def __init__(self, input_dim, output_dim, stride=1):
        super().__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_dim, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv3x3(x)
        return x


# CDC
class Cross_Domian_Constrains(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=7, stride=4, padding=3),
            CD_LayerNorm(48),
        )
        self.cdit = CD_Transformer(dim=48, proj_kernel=3, kv_proj_stride=4, heads=1, depth=1, mlp_mult=2,
                                    dropout=dropout)
        # self.down2 = nn.Sequential(
        #     nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
        #     CC_LayerNorm(64),
        # )
        # self.cdtb2 = CD_Transformer_Block(dim=64, proj_kernel=3, kv_proj_stride=2, heads=2, depth=1, mlp_mult=4,
        #                          dropout=dropout)

        self.relu = nn.ReLU(inplace=True)
        # self.conv3x3_1 = CD_conv3x3(input_dim=64, output_dim=48)
        self.conv3x3_2 = CD_conv3x3(input_dim=48, output_dim=32)

    def forward(self, x, y):
        residualx = x
        residualy = y

        x, y = self.cdit(self.down1(x), self.down1(y))
        # x, y = self.cdtb2(self.down2(x), self.down2(y))


        # x = F.interpolate(x, mode='bilinear', size=(x.shape[2] * 2, x.shape[3] * 2))
        # x = self.conv3x3_1(x)
        x = F.interpolate(x, mode='bilinear', size=(residualx.shape[2], residualx.shape[3]))
        x = self.conv3x3_2(x)
        x = x + residualx
        x = self.relu(x)

        # y = F.interpolate(y, mode='bilinear', size=(y.shape[2] * 2, y.shape[3] * 2))
        # y = self.conv3x3_1(y)
        y = F.interpolate(y, mode='bilinear', size=(residualy.shape[2], residualy.shape[3]))
        y = self.conv3x3_2(y)
        y = y + residualy
        y = self.relu(y)
        return x, y



class MFFT(nn.Module):
    def __init__(
            self,
            *,
            img_channels=3,
            dropout=0.
    ):
        super().__init__()

        # Shared feature extraction
        self.sfe = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
            Multi_Scale_Feature_Extract_Module(),
            Cross_Scale_Visual_Attention_Module(),
        )

        # Feature fusion
        self.mixer = nn.Sequential(
            nn.Conv2d(32 * 2, 32, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
            Basic_Residual_Module(),
        )

        # Generative task
        self.csva_g_1 = Cross_Scale_Visual_Attention_Module()
        self.csva_g_2 = Cross_Scale_Visual_Attention_Module()
        self.br_g = Basic_Residual_Module()
        self.outconv_g = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid(),
        )
        # Discriminative task
        self.csva_d_1 = Cross_Scale_Visual_Attention_Module()
        self.br_d = Basic_Residual_Module()
        self.outconv_d = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid(),
        )

        # Soft sharing
        self.cdc1 = Cross_Domian_Constrains()
        self.cdc2 = Cross_Domian_Constrains()

    def forward(self, A, B):
        Feature_A = self.sfe(A)
        Feature_B = self.sfe(B)

        concatenation = torch.cat([Feature_A, Feature_B], dim=1)
        F = self.mixer(concatenation)

        FG1 = self.csva_g_1(F)
        FD1 = self.csva_d_1(F)

        FG1, FD1 = self.cdc1(FG1, FD1)

        FG2 = self.csva_g_2(FG1)

        FG3 = self.br_g(FG2)
        FD2 = self.br_d(FD1)

        FG3, FD2 = self.cdc2(FG3, FD2)


        FGOut = self.outconv_g(FG3)
        FDOut = self.outconv_d(FD2)


        return FGOut, FDOut


if __name__ == '__main__':
    test_tensor_A = torch.zeros((1, 3, 224, 224)).to('cuda')
    test_tensor_B = torch.rand((1, 3, 224, 224)).to('cuda')
    model = MFFT().to('cuda')
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of model parameters: {} M\n\n".format(round(num_params / 10e5, 6)))
    FG, FD = model(test_tensor_A, test_tensor_B)
    print(FG.shape)
    print(FD.shape)
