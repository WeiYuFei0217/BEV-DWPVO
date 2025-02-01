import torch
import torch.nn.functional as F
from networks.layers import DoubleConv, DoubleConv_inc, OutConv, Down, Up

class UNet(torch.nn.Module):

    def __init__(self, config, NO_BN=False):
        super().__init__()
        bilinear = config['networks']['unet']['bilinear']
        first_feature_dimension = config['networks']['unet']['first_feature_dimension']
        self.score_sigmoid = config['networks']['unet']['score_sigmoid']
        outc_score_dim = 1
        real_input_channels = 80

        first_feature_dimension = 8

        self.inc = DoubleConv(real_input_channels, first_feature_dimension, NO_BN=NO_BN)
        self.down1 = Down(first_feature_dimension, first_feature_dimension * 2, NO_BN=NO_BN) 
        self.down2 = Down(first_feature_dimension * 2, first_feature_dimension * 4, NO_BN=NO_BN)
        self.down3 = Down(first_feature_dimension * 4, first_feature_dimension * 8, NO_BN=NO_BN)
        self.down4 = Down(first_feature_dimension * 8, first_feature_dimension * 16, NO_BN=NO_BN)

        self.up1_pts = Up(first_feature_dimension * (16 + 8), first_feature_dimension * 8, bilinear, NO_BN=NO_BN)
        self.up2_pts = Up(first_feature_dimension * (8 + 4), first_feature_dimension * 4, bilinear, NO_BN=NO_BN)
        self.up3_pts = Up(first_feature_dimension * (4 + 2), first_feature_dimension * 2, bilinear, NO_BN=NO_BN)
        self.up4_pts = Up(first_feature_dimension * (2 + 1), first_feature_dimension * 1, bilinear, NO_BN=NO_BN)
        self.outc_pts = OutConv(first_feature_dimension, 1)

        self.up1_score = Up(first_feature_dimension * (16 + 8), first_feature_dimension * 8, bilinear, NO_BN=NO_BN)
        self.up2_score = Up(first_feature_dimension * (8 + 4), first_feature_dimension * 4, bilinear, NO_BN=NO_BN)
        self.up3_score = Up(first_feature_dimension * (4 + 2), first_feature_dimension * 2, bilinear, NO_BN=NO_BN)
        self.up4_score = Up(first_feature_dimension * (2 + 1), first_feature_dimension * 1, bilinear, NO_BN=NO_BN)
        self.outc_score = OutConv(first_feature_dimension, outc_score_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        _, _, height, width = x.size()

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x4_up_pts = self.up1_pts(x5, x4)
        x3_up_pts = self.up2_pts(x4_up_pts, x3)
        x2_up_pts = self.up3_pts(x3_up_pts, x2)
        x1_up_pts = self.up4_pts(x2_up_pts, x1)
        detector_scores = self.outc_pts(x1_up_pts)

        x4_up_score = self.up1_score(x5, x4)
        x3_up_score = self.up2_score(x4_up_score, x3)
        x2_up_score = self.up3_score(x3_up_score, x2)
        x1_up_score = self.up4_score(x2_up_score, x1)
        weight_scores = self.outc_score(x1_up_score)
        if self.score_sigmoid:
            weight_scores = self.sigmoid(weight_scores)

        f1 = F.interpolate(x1, size=(height, width), mode='bilinear')
        f2 = F.interpolate(x2, size=(height, width), mode='bilinear')
        f3 = F.interpolate(x3, size=(height, width), mode='bilinear')
        f4 = F.interpolate(x4, size=(height, width), mode='bilinear')
        f5 = F.interpolate(x5, size=(height, width), mode='bilinear')

        feature_list = [f1, f2, f3, f4, f5]
        descriptors = torch.cat(feature_list, dim=1)

        return detector_scores, weight_scores, descriptors
