import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
from einops import rearrange


class Conv2d(nn.Conv2d): # For Weight Standardization

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def convnxn(in_planes, out_planes, kernel_size, stride=1):
    """nxn convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        #nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        conv1x1(inp, oup),
        #nn.BatchNorm2d(oup),
        nn.GroupNorm(16, oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        #nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        convnxn(inp, oup, kernal_size, stride),
        #nn.BatchNorm2d(oup),
        nn.GroupNorm(16, oup),
        nn.SiLU()
    )

class Conv1dLast(nn.Module): # For Last Conv layer
    def __init__(self, in_channels, out_channels):
        super(Conv1dLast, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        #self.conv = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.conv = conv1x1(self.in_channels, self.out_channels)
        #self.bn = nn.BatchNorm2d(self.out_channels)
        self.bn = nn.GroupNorm(16, self.out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        out = self.conv(x)
        out1 = self.bn(out)
        out2 = self.act(out1)
        return out2, out2

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features)) # C x d i.e., 1000 x 640
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, 0, 0.01)
        if self.sigma is not None:
            self.sigma.data.fill_(1) #for initializaiton of sigma
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):       
        if self.bias is not None:
            input = torch.cat((input, (torch.ones(len(input),1).cuda())), dim=1)
            concat_weight = torch.cat((self.weight, self.bias), dim=1)
            out = F.linear(F.normalize(input,p=2,dim=1,eps=1e-8), F.normalize(concat_weight,p=2,dim=1,eps=1e-8))
        else:
            out = F.linear(F.normalize(input,p=2,dim=1,eps=1e-8), F.normalize(self.weight,p=2,dim=1,eps=1e-8))
            ## N:B: eps 1e-8 is better than default 1e-12

        if self.sigma is not None:
            out = self.sigma * out
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                #nn.BatchNorm2d(hidden_dim),
                nn.GroupNorm(16, hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                #nn.BatchNorm2d(oup),
                nn.GroupNorm(16, oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                #nn.BatchNorm2d(hidden_dim),
                nn.GroupNorm(16, hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                #nn.BatchNorm2d(hidden_dim),
                nn.GroupNorm(16, hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                #nn.BatchNorm2d(oup),
                nn.GroupNorm(16, oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    
    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        self.stem = conv_nxn_bn(3, channels[0], stride=2)

        self.stages0 = MV2Block(channels[0], channels[1], 1, expansion)

        self.stages1 = nn.Sequential(
            MV2Block(channels[1], channels[2], 2, expansion),
            MV2Block(channels[2], channels[3], 1, expansion),
            MV2Block(channels[2], channels[3], 1, expansion),
        )   # Repeat

        self.stages2 = nn.Sequential(
            MV2Block(channels[3], channels[4], 2, expansion),
            MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2)),
            )

        self.stages3 = nn.Sequential(
            MV2Block(channels[5], channels[6], 2, expansion), ## After this the top part starts
            MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*4)),
            )

        self.stages4 = nn.Sequential(
            MV2Block(channels[7], channels[8], 2, expansion),
            MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2]*4)),
            )

        #self.final_conv = conv_1x1_bn(channels[-2], channels[-1])
        self.final_conv = Conv1dLast(channels[-2], channels[-1])

        self.pool = nn.AvgPool2d(ih//32, 1)
        #self.fc = nn.Linear(channels[-1], num_classes, bias=True)
        self.fc = CosineLinear(channels[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages0(x)
        x = self.stages1(x)      # Repeat
        x = self.stages2(x)
        x = self.stages3(x)
        x = self.stages4(x)
        x, _ = self.final_conv(x)
        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x


def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT((256, 256), dims, channels, num_classes=1000, expansion=2)


def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


### Stages 3 ### 16 x 16 x 128

class MobViT_StartAt_Stages3(nn.Module):
    def __init__(self, num_classes=None):
        super(MobViT_StartAt_Stages3, self).__init__()

        self.model = mobilevit_s()

        ## Remove stem and first 3 stages
        del self.model.stem
        del self.model.stages0
        del self.model.stages1
        del self.model.stages2
        del self.model.stages3[0]
        
        if num_classes is not None:
            print('Changing output layer to contain %d classes.' % num_classes)
            self.model.fc = CosineLinear(640, num_classes)

    def forward(self, x): # x dim: N x 128 x 16 x 16
        out = self.model.stages3(x) # N x 128 x 16 x 16
        out = self.model.stages4(out) # N x 160 x 8 x 8
        out, _ = self.model.final_conv(out) # N x 640 x 8 x 8
        out = F.avg_pool2d(out, out.size()[3]) # N x 640 x 1 x 1
        out = out.view(out.size(0), -1) # N x 640
        out = self.model.fc(out) # N x 1000
        return out

    def get_penultimate_feature(self, x): # x dim: N x 128 x 16 x 16
        out = self.model.stages3(x) # N x 128 x 16 x 16
        out = self.model.stages4(out) # N x 160 x 8 x 8
        _, out = self.model.final_conv(out) # N x 640 x 8 x 8
        out = F.avg_pool2d(out, out.size()[3]) # N x 640 x 1 x 1
        out = out.view(out.size(0), -1) # N x 640
        return out


### Stages 3 ### 16 x 16 x 128

class BaseMobViTClassifyAfterStages3(nn.Module):
    def __init__(self, num_del=0, num_classes=None):
        super(BaseMobViTClassifyAfterStages3, self).__init__()

        self.model = mobilevit_s()

        for _ in range(0, num_del):
            del self.model.stages3[-1]

        if num_classes is not None:
            print("Changing num_classes to {}".format(num_classes))
            self.model.fc = CosineLinear(640, num_classes)

    def forward(self, x):
        out = self.model(x)
        return out

class MobViTClassifyAfterStages3(BaseMobViTClassifyAfterStages3):
    def __init__(self, num_classes=None):
        super(MobViTClassifyAfterStages3, self).__init__(num_del=0, num_classes=num_classes)

