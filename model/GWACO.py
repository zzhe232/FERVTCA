import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision
from torchvision.models.resnet import ResNet34_Weights


#  refer to SubSection 3.3
# input: img(batchsize,c,h,w)--->output: img_feature_map(batchsize,c,h,w)
# in FER+ (b,3,48,48)
class GWA(nn.Module): ##第一步
    @staticmethod
    def weight_init(m):
        if isinstance(m,nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
        elif isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __init__(self):
        super(GWA, self).__init__()
        # low level feature extraction
        # 每个gird使用inverted bottleneck neural block去作为local feature extraction network
        self.conv1 = nn.Conv2d(3, 64, 1)
        self.bn1 = nn.BatchNorm2d(64)
        #恢复原通道（inverted bottleleneck）
        self.conv2 = nn.Conv2d(64, 3, 1)
        self.bn2 = nn.BatchNorm2d(3)
        # 图像分割，每块16x16，用一个卷积层实现，形成gird(4*4),9408=56*56*3
        self.patch_embeddings = nn.Conv2d(in_channels=3,
                                          out_channels=9408,
                                          kernel_size=(56, 56),
                                          stride=(56, 56))
        # 使用自适应pool压缩一维
        self.aap = nn.AdaptiveAvgPool2d((1, 1))

        self.apply(self.weight_init)

    def forward(self, x):
        #原图片
        img = x
        #获得batchsize(32,3,224, 224)获得32
        batchsize = x.shape[0]
        #将输入分成gird格子
        x = self.patch_embeddings(x)
        #(batchsize, 9408, 4, 4)->(batchsize, 9408, 16)展平后->(batchsize, 16, 9408)交换最后俩个维度->view(batchsize, 16, 3, 56, 56)保持原格式不变
        x = x.flatten(2).transpose(-1, -2).view(batchsize, 16, 3, 56, 56)  # （batchsize,9,768）（batchsize,9,3,256)每个批次有16个块，为56*56*3
        temp = []
        #对于每一个块（16)提取低级特征
        for i in range(x.shape[1]):
            temp.append(F.leaky_relu(self.bn2(self.conv2(
                F.leaky_relu(self.bn1(self.conv1(x[:, i, :, :, :])))))).unsqueeze(0).transpose(0, 1))
        #处理后的块品阶起来
        x = torch.cat(tuple(temp), dim=1)
        #见文章低级注意力部分，x是每个块的低级特征
        query = x
        #key是交换3，4维度即宽和高
        key = torch.transpose(query, 3, 4)
        #atten形状(batch_size, num_patches, height*width, height*width)
        attn = F.softmax(torch.matmul(query, key) / 56,dim=1)
        temp = []
        #对每个注意力块进行池化，增加一个维度然后对于0,1调换位置
        for i in range(attn.shape[1]):
            temp.append(self.aap(attn[:, i, :, :, :]).unsqueeze(0).transpose(0, 1))
        #文中（4）进行操作
        pattn = torch.ones(56, 56).cuda() * torch.cat(tuple(temp), dim=1)
        pattn = pattn.permute(0,2,3,1,4).contiguous()
        #还原为全局注意力
        pattn = pattn.view(batchsize, 3, 224, 224).cuda()
        #获得注意力图（1，2）步
        map = pattn * img  # (b,3,48,48)
        return img, map


class GWA_Fusion(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, 0.1)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __init__(self):
        super(GWA_Fusion, self).__init__()
        # 原图特征转换网络
        self.convt1 = nn.Conv2d(3, 3, (3, 3), 1, 1)
        self.bnt1 = nn.BatchNorm2d(3)
        # map特征转换网络
        self.convt2 = nn.Conv2d(3, 3, (3, 3), 1, 1)
        self.bnt2 = nn.BatchNorm2d(3)
        # RFN参与特征融合网络
        self.convrfn1 = nn.Conv2d(3,3,(3,3),1,1)
        self.bnrfn1 = nn.BatchNorm2d(3)
        self.prelu1 = nn.PReLU(3)
        self.convrfn2 = nn.Conv2d(3, 3, (3, 3), 1, 1)
        self.bnrfn2 = nn.BatchNorm2d(3)
        self.prelu2 = nn.PReLU(3)
        self.convrfn3 = nn.Conv2d(3, 3, (3, 3), 1, 1)
        self.sigmod = nn.Sigmoid()

        self.apply(self.weight_init)

    def forward(self, img, map):
        img_trans = F.relu(self.bnt1(self.convt1(img)))
        map_trans = F.relu(self.bnt2(self.convt1(map)))
        result = self.prelu1(self.bnrfn1(self.convrfn1(img_trans + map_trans)))
        result = self.prelu2(self.bnrfn2(self.convrfn2(result)))
        result = self.sigmod(self.convrfn3(result+img_trans + map_trans))
        #result: 3, 224, 224
        return result


# backbone + token_embedding + position_embedding
class Backbone(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __init__(self):
        super(Backbone, self).__init__()

        resnet = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        #  feature resize networks
        # shape trans 128，将所有金字塔提取的特征转化为同一个形状
        self.convtran1 = nn.Conv2d(128, 3, 21, 1)
        self.bntran1 = nn.BatchNorm2d(3)
        self.convtran2 = nn.Conv2d(256, 3, 7, 1)
        self.bntran2 = nn.BatchNorm2d(3)
        self.convtran3 = nn.Conv2d(512, 3, 2, 1,1)
        self.bntran3 = nn.BatchNorm2d(3)
        self.weight1 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.weight2 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.weight3 = nn.Parameter(torch.randn(1), requires_grad=True)

        self.apply(self.weight_init)

    def forward(self, x):
        batchsize = x.shape[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)

        x = self.layer2(x)

        # L1  feature transformation from the pyramid features
        l1 = F.leaky_relu(self.bntran1(self.convtran1(x))) #32,3,8,8

        x = self.layer3(x)
        l2 = F.leaky_relu(self.bntran2(self.convtran2(x)))

        x = self.layer4(x)
        l3 = F.leaky_relu(self.bntran3(self.convtran3(x)))
        x = self.weight1 * l1 + self.weight2 * l2 + self.weight3 * l3
        return x
#torch.Size([32, 4, 192])


class CoordAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = reduction
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        batch, channel, height, width = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [height, width], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = x * a_h.expand_as(x) * a_w.expand_as(x)

        return out

class CAAtt(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(CAAtt, self).__init__()

        self.coord_attention = CoordAttention(in_channels, in_channels)
        self.bn = nn.BatchNorm1d(192)
        self.fc = nn.Linear(in_channels * 8 * 8, num_classes)

    def forward(self, x):
        x = self.coord_attention(x)
        x = x.flatten(start_dim=1)
        x = self.bn(x)
        x = self.fc(x)
        return x


class FERVT(nn.Module):
    def __init__(self, device):
        super(FERVT, self).__init__()
        self.gwa = GWA()
        self.gwa.to(device)
        self.gwa_f = GWA_Fusion()
        self.gwa_f.to(device)
        self.backbone = Backbone()
        self.backbone.to(device)
        self.vta = CAAtt()
        self.vta.to(device)

        self.to(device)
        # Evaluation mode on

    def forward(self, x):
        img,map = self.gwa(x)
        emotions = self.vta(self.backbone(self.gwa_f(img,map)))
        return emotions


