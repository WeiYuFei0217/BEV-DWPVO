import torch
import torch.nn.functional as F
from networks.layers import DoubleConv, OutConv, Down, Up


class UNet(torch.nn.Module):
    """Predicts keypoint position weights (W_pos), keypoint validity weights (W_valid)
    and multi-scale keypoint descriptors (D_key) from a BEV feature map."""

    def __init__(self, config):
        super().__init__()
        bilinear = config['unet']['bilinear']
        c = config['unet']['first_feature_dimension']
        self.score_sigmoid = config['unet']['score_sigmoid']
        in_channels = config['in_channels']

        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)

        self.up1_pts = Up(c * (16 + 8), c * 8, bilinear)
        self.up2_pts = Up(c * (8 + 4), c * 4, bilinear)
        self.up3_pts = Up(c * (4 + 2), c * 2, bilinear)
        self.up4_pts = Up(c * (2 + 1), c * 1, bilinear)
        self.outc_pts = OutConv(c, 1)

        self.up1_score = Up(c * (16 + 8), c * 8, bilinear)
        self.up2_score = Up(c * (8 + 4), c * 4, bilinear)
        self.up3_score = Up(c * (4 + 2), c * 2, bilinear)
        self.up4_score = Up(c * (2 + 1), c * 1, bilinear)
        self.outc_score = OutConv(c, 1)
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
        position_scores = self.outc_pts(x1_up_pts)

        x4_up_score = self.up1_score(x5, x4)
        x3_up_score = self.up2_score(x4_up_score, x3)
        x2_up_score = self.up3_score(x3_up_score, x2)
        x1_up_score = self.up4_score(x2_up_score, x1)
        validity = self.outc_score(x1_up_score)
        if self.score_sigmoid:
            validity = self.sigmoid(validity)

        f1 = F.interpolate(x1, size=(height, width), mode='bilinear')
        f2 = F.interpolate(x2, size=(height, width), mode='bilinear')
        f3 = F.interpolate(x3, size=(height, width), mode='bilinear')
        f4 = F.interpolate(x4, size=(height, width), mode='bilinear')
        f5 = F.interpolate(x5, size=(height, width), mode='bilinear')

        descriptors = torch.cat([f1, f2, f3, f4, f5], dim=1)

        return position_scores, validity, descriptors
