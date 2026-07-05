# DPGLO_SRNet: Glo and ShadowAwareDecoder
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
from math import sqrt
    
def initialize_weights(init_type='gaussian'):
    def init_func(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_func

class Cvi(nn.Module):
    def __init__(self, in_channels, out_channels, before=None, after=False, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Cvi, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv.apply(initialize_weights('gaussian'))

        if after == 'BN':
            self.after = nn.InstanceNorm2d(out_channels)
        elif after == 'Tanh':
            self.after = torch.tanh
        elif after == 'sigmoid':
            self.after = torch.sigmoid

        if before == 'ReLU':
            self.before = nn.ReLU(inplace=True)
        elif before == 'LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x


class CvTi(nn.Module):
    def __init__(self, in_channels, out_channels, before=None, after=False, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=False):
        super(CvTi, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                        output_padding=0, bias=bias)  # Specify output_padding here
        self.conv.apply(initialize_weights('gaussian'))

        if after == 'BN':
            self.after = nn.InstanceNorm2d(out_channels)
        elif after == 'Tanh':
            self.after = torch.tanh
        elif after == 'sigmoid':
            self.after = torch.sigmoid

        if before == 'ReLU':
            self.before = nn.ReLU(inplace=True)
        elif before == 'LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x
    

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x_flat + attn_out

        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x = x + x_mlp

        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x

    
#  Attention Block 
class AttentionBlock(nn.Module):
    def __init__(self, in_ch):
        super(AttentionBlock, self).__init__()
        self.norm = nn.BatchNorm2d(in_ch)
        self.qkv = nn.Conv2d(in_ch, in_ch * 3, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(B, 3, C, H * W).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, C, H*W]

        attn = self.softmax(torch.matmul(q.transpose(-2, -1), k) / (C ** 0.5))  # [B, H*W, H*W]
        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1).reshape(B, C, H, W)
        out = self.proj(out)
        return out + x  # residual connection

#  channel-wise Squeeze-and-Excitation Block  
class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super(SEBlock, self).__init__()
        reduced_ch = max(1, ch // reduction)  # prevent 0 channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, reduced_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_ch, ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.avg_pool(x))
        return x * w

    
#  Smart Residual Block 
class SmartResidualBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super(SmartResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch)
        )
        self.se = SEBlock(ch, reduction)

    def forward(self, x):
        out = self.block(x)
        out = self.se(out)
        return x + out

    
#  Multi-Resolution Contextual Integration 
class MRCI(nn.Module):
    def __init__(self, in_ch_list, out_ch):
        super(MRCI, self).__init__()
        self.fuse = nn.Conv2d(sum(in_ch_list), out_ch, kernel_size=1)

    def forward(self, features):
        resized = [F.interpolate(f, size=features[0].shape[2:], mode='bilinear', align_corners=False) for f in features]
        concat = torch.cat(resized, dim=1)
        return self.fuse(concat)
    
        # Residual-enhanced fusion
        fused = self.fuse(concat) + sum(resized) / len(resized)

        return fused


# Encoder (Global-Local Context)
class Encoder(nn.Module):
    def __init__(self, input_channels=3):
        super(Encoder, self).__init__()
        self.Cv0 = Cvi(input_channels, 64)
        self.Cv1 = Cvi(64, 128, before='LReLU', after='BN')
        self.Cv2 = Cvi(128, 256, before='LReLU', after='BN')
        self.Cv3 = Cvi(256, 512, before='LReLU', after='BN')
        self.Cv4 = Cvi(512, 512, before='LReLU', after='BN')
        self.Cv5 = Cvi(512, 512, before='LReLU')
        
        #  GLO 
        self.trans = TransformerBlock(dim=512)
        self.attn = AttentionBlock(512)
        
        # Degradation-specific branches
        self.shared_residual = SmartResidualBlock(512)

        # Shadow refinement
        self.shadow_trans = TransformerBlock(dim=512)
        self.shadow_attn = AttentionBlock(512)
        self.shadow_refine = nn.Sequential(
            SmartResidualBlock(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        # Combine and fuse with MRCI
        self.dynamic_fuse = nn.Sequential(
            nn.Conv2d(1024, 512, 1),  # 512 + 512 = 1024  # concatenate combined + shadow_refined
            nn.Sigmoid()
            )

        # MRCI for skip connections
        self.mrcf = MRCI([512, 512], 512)  # total = 1024

    def forward(self, input):
        #encoder
        x0 = self.Cv0(input)
        x1 = self.Cv1(x0)
        x2 = self.Cv2(x1)
        x3 = self.Cv3(x2)
        x4_1 = self.Cv4(x3)
        x4_2 = self.Cv4(x4_1)
        x4_3 = self.Cv4(x4_2)
        x5 = self.Cv5(x4_3)
        
        #  GLO 
        glo_feat = self.trans(x5)
        global_feat = self.attn(glo_feat)

        # Degradation separation
        combined = self.shared_residual(global_feat)

        # Shadow refinement
        shadow_feat = self.shadow_trans(combined)
        shadow_feat = self.shadow_attn(shadow_feat)
        shadow_refined = self.shadow_refine(shadow_feat)

        # Combine and fuse with MRCI
        alpha = self.dynamic_fuse(torch.cat([combined, shadow_refined], dim=1))
        alpha = torch.clamp(alpha, min=0.1, max=0.9)  # Prevent saturation
        fused_feat0 = alpha * combined + (1 - alpha) * shadow_refined
        
        fused_feat1 = self.mrcf([x5, fused_feat0])
                                                                                       
        fused_feat2 = F.interpolate(fused_feat1, size=x5.shape[2:], mode='bilinear', align_corners=False)  #fused_feat2 = fused_feat1.view(-1, 512, 1, 1)

        feature_dic = {
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4_1": x4_1,
            "x4_2": x4_2,
            "x4_3": x4_3,
            "x5": x5,
            "fused_feat2": fused_feat2
            
        }

        return feature_dic

#  Residual Decoder with GLo = ViT + Attention + Residual + Refine + MRCI 
class DeGloNet_Decoder(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(DeGloNet_Decoder, self).__init__()
        self.CvT6 = CvTi(2048, 1024, before='ReLU', after='BN')
        self.CvT7 = CvTi(2048, 1024, before='ReLU', after='BN')
        self.CvT8 = CvTi(2048, 512, before='ReLU', after='BN')
        self.CvT9 = CvTi(1024, 256, before='ReLU', after='BN')
        self.CvT10 = CvTi(512, 128, before='ReLU', after='BN')
        self.CvT11 = CvTi(256, output_channels, before='ReLU', after='Tanh')
        
        # SR blocks for enhanced feature emphasis
        self.sr6 = SmartResidualBlock(1024)
        self.sr7_1 = SmartResidualBlock(1024)
        self.sr7_2 = SmartResidualBlock(1024)
        self.sr7_3 = SmartResidualBlock(1024)
        self.sr8 = SmartResidualBlock(512)
        self.sr9 = SmartResidualBlock(256)
        self.sr10 = SmartResidualBlock(128)
        
        #self.final_sr = SmartResidualBlock(output_channels)  # output_channels is typically 3

        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        )

    def forward(self, input_feats, guide_feats):
        #decoder
        # x5
        cat_0 = torch.cat([input_feats["x5"], guide_feats["x5"], input_feats["fused_feat2"], guide_feats["fused_feat2"]], dim=1) #concat(2048,1024)
        x6 = self.CvT6(cat_0) #channel=1024
        x6 = self.sr6(x6)
        
        # x4_3
        # Align before concatenation
        x4_3_input = F.interpolate(input_feats["x4_3"], size=x6.shape[2:], mode='bilinear', align_corners=False)
        x4_3_guide = F.interpolate(guide_feats["x4_3"], size=x6.shape[2:], mode='bilinear', align_corners=False)
        cat1_1 = torch.cat([x6, x4_3_input, x4_3_guide], dim=1)
        x7_1 = self.CvT7(cat1_1)
        x7_1 = self.sr7_1(x7_1)
        
        # x4_2
        # Align before concatenation
        x4_2_input = F.interpolate(input_feats["x4_2"], size=x7_1.shape[2:], mode='bilinear', align_corners=False)
        x4_2_guide = F.interpolate(guide_feats["x4_2"], size=x7_1.shape[2:], mode='bilinear', align_corners=False)  
        cat1_2 = torch.cat([x7_1, x4_2_input, x4_2_guide], dim=1) #concat(1024, 1024)
        x7_2 = self.CvT7(cat1_2)
        x7_2 = self.sr7_2(x7_2)
       
        # x4_1
        # Align before concatenation
        x4_1_input = F.interpolate(input_feats["x4_1"], size=x7_2.shape[2:], mode='bilinear', align_corners=False)
        x4_1_guide = F.interpolate(guide_feats["x4_1"], size=x7_2.shape[2:], mode='bilinear', align_corners=False) 
        cat1_3 = torch.cat([x7_2, x4_1_input, x4_1_guide], dim=1) #concat(1024, 1024)
        x7_3 = self.CvT7(cat1_3)
        x7_3 = self.sr7_3(x7_3)
        
        # x3
        # Align before concatenation
        x3_input = F.interpolate(input_feats["x3"], size=x7_3.shape[2:], mode='bilinear', align_corners=False)
        x3_guide = F.interpolate(guide_feats["x3"], size=x7_3.shape[2:], mode='bilinear', align_corners=False)
        cat2 = torch.cat([x7_3,  x3_input, x3_guide], dim=1) #concat(1024, 1024)
        x8 = self.CvT8(cat2)
        x8 = self.sr8(x8)
        
        # x2
        # Align before concatenation
        x2_input = F.interpolate(input_feats["x2"], size=x8.shape[2:], mode='bilinear', align_corners=False)
        x2_guide = F.interpolate(guide_feats["x2"], size=x8.shape[2:], mode='bilinear', align_corners=False)
        cat3 = torch.cat([x8,  x2_input, x2_guide], dim=1) #concat(512, 512)
        x9 = self.CvT9(cat3)
        x9 = self.sr9(x9)


        # x1
        # Align before concatenation
        x1_input = F.interpolate(input_feats["x1"], size=x9.shape[2:], mode='bilinear', align_corners=False)
        x1_guide = F.interpolate(guide_feats["x1"], size=x9.shape[2:], mode='bilinear', align_corners=False)
        cat4 = torch.cat([x9, x1_input, x1_guide], dim=1) #concat(256, 256)
        x10 = self.CvT10(cat4)
        x10 = self.sr10(x10)
        
        # x1
        x0_input = F.interpolate(input_feats["x0"], size=x10.shape[2:], mode='bilinear', align_corners=False)
        x0_guide = F.interpolate(guide_feats["x0"], size=x10.shape[2:], mode='bilinear', align_corners=False)
        cat5 = torch.cat([x10, x0_input, x0_guide], dim=1)
        #concat(128, 128)
        out = self.CvT11(cat5)
        
        residual = self.refine(out)
        # Final refinement
        out = out + residual

        return out

#  SHADOW-AWARE DECODER 

class ShadowAwareDecoder(nn.Module):
    def __init__(self):
        super(ShadowAwareDecoder, self).__init__()
        
        # Shadow mask processing - extracts shadow-specific features
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),  # Downsample
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # Downsample
            nn.ReLU()
        )
        
        # Use specific features from encoder
        # use x3 (512), x2 (256), x1 (128), x0 (64) for skip connections
        encoder_channels = [512, 256, 128, 64]
        
        # Feature fusion blocks for each encoder level
        self.fusion_blocks = nn.ModuleList([
            nn.Conv2d(ch + 64, ch, 1) for ch in encoder_channels  # +64 for mask features
        ])
        
        # Decoder path with channel reduction
        self.decode_blocks = nn.ModuleList([
            # 512 -> 256
            nn.Sequential(
                nn.Conv2d(512, 256, 3, 1, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 256, 4, 2, 1)  # Upsample
            ),
            # 256 -> 128
            nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 128, 4, 2, 1)  # Upsample
            ),
            # 128 -> 64
            nn.Sequential(
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 64, 4, 2, 1)  # Upsample
            ),
            # 64 -> 32
            nn.Sequential(
                nn.Conv2d(64, 32, 3, 1, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, 4, 2, 1)  # Upsample
            )
        ])
        
        # Final output layer
        self.output = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, encoder_features, shadow_mask):
        # Extract specific features from the encoder dictionary
        # Use x3, x2, x1, x0 for multi-scale processing
        feat_x3 = encoder_features["x3"]     # 512 channels
        feat_x2 = encoder_features["x2"]     # 256 channels
        feat_x1 = encoder_features["x1"]     # 128 channels
        feat_x0 = encoder_features["x0"]     # 64 channels
        
        # Process shadow mask to get multi-scale features
        mask_feats = self.mask_encoder(shadow_mask)
        
        # Start from x3 (512 channels)
        x = feat_x3
        
        # Fuse with mask features at x3 scale
        mask_feat_resized = F.interpolate(mask_feats, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = self.fusion_blocks[0](torch.cat([x, mask_feat_resized], dim=1))
        
        # Decode with skip connections
        # Stage 1: 512 -> 256
        x = self.decode_blocks[0](x)
        # Add skip from x2
        if x.shape[2:] != feat_x2.shape[2:]:
            x = F.interpolate(x, size=feat_x2.shape[2:], mode='bilinear', align_corners=False)
        mask_feat_resized = F.interpolate(mask_feats, size=feat_x2.shape[2:], mode='bilinear', align_corners=False)
        skip_x2 = self.fusion_blocks[1](torch.cat([feat_x2, mask_feat_resized], dim=1))
        x = x + skip_x2
        
        # Stage 2: 256 -> 128
        x = self.decode_blocks[1](x)
        # Add skip from x1
        if x.shape[2:] != feat_x1.shape[2:]:
            x = F.interpolate(x, size=feat_x1.shape[2:], mode='bilinear', align_corners=False)
        mask_feat_resized = F.interpolate(mask_feats, size=feat_x1.shape[2:], mode='bilinear', align_corners=False)
        skip_x1 = self.fusion_blocks[2](torch.cat([feat_x1, mask_feat_resized], dim=1))
        x = x + skip_x1
        
        # Stage 3: 128 -> 64
        x = self.decode_blocks[2](x)
        # Add skip from x0
        if x.shape[2:] != feat_x0.shape[2:]:
            x = F.interpolate(x, size=feat_x0.shape[2:], mode='bilinear', align_corners=False)
        mask_feat_resized = F.interpolate(mask_feats, size=feat_x0.shape[2:], mode='bilinear', align_corners=False)
        skip_x0 = self.fusion_blocks[3](torch.cat([feat_x0, mask_feat_resized], dim=1))
        x = x + skip_x0
        
        # Stage 4: 64 -> 32
        x = self.decode_blocks[3](x)
        
        # Upsample to original size if needed
        if x.shape[2:] != shadow_mask.shape[2:]:
            x = F.interpolate(x, size=shadow_mask.shape[2:], mode='bilinear', align_corners=False)
            
        return self.output(x)


#  DPGLO_SRNet 
class DPGLO_SRNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(DPGLO_SRNet, self).__init__()

        self.encoder1 = Encoder(input_channels)
        self.encoder2 = Encoder(input_channels)
        # Residual Decoder
        self.decoder = DeGloNet_Decoder(output_channels)
        self.placeholder = None
        
        # Shadow-aware decoder
        self.shadow_decoder = ShadowAwareDecoder()
        
        # Fusion for shadow decoder
        self.fusion_attention = nn.Sequential(
            nn.Conv2d(output_channels * 2, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input, GT):
        # Extract shadow masks
        shadow_mask_input = input[:, 3:4, :, :]  # From images_mask
        shadow_mask_gt = GT[:, 3:4, :, :]        # From gt_mask

        input_feats = self.encoder1(input)
        guide_feats = self.encoder2(GT)
        
        out1 = self.decoder(input_feats, input_feats)
        out2 = self.decoder(guide_feats, guide_feats)
        
        # Using shadow decoder
        shadow_out1 = self.shadow_decoder(input_feats, shadow_mask_input)
        shadow_out2 = self.shadow_decoder(guide_feats, shadow_mask_gt)
        
        # shadow outputs, decoder output
        if shadow_out1.shape[2:] != out1.shape[2:]:
            shadow_out1 = F.interpolate(shadow_out1, size=out1.shape[2:], mode='bilinear', align_corners=False)
            shadow_out2 = F.interpolate(shadow_out2, size=out2.shape[2:], mode='bilinear', align_corners=False)
        
        # Attention-based fusion
        # Shadow to shadow free
        attention1 = self.fusion_attention(torch.cat([out1, shadow_out1], dim=1))
        # shadow free
        attention2 = self.fusion_attention(torch.cat([out2, shadow_out2], dim=1))
        
        # Weighted combination
        # Shadow to shadow free
        final_out1 = attention1 * shadow_out1 + (1 - attention1) * out1
        # shadow free
        final_out2 = attention2 * shadow_out2 + (1 - attention2) * out2
        
        return final_out1, final_out2
    
    def test_set(self, input):
        shadow_mask = input[:, 3:4, :, :]
        
        input_feats = self.encoder1(input)
        test_out = self.decoder(input_feats, input_feats)
        
        # Shadow decoder
        shadow_out = self.shadow_decoder(input_feats, shadow_mask)
        if shadow_out.shape[2:] != test_out.shape[2:]:
            shadow_out = F.interpolate(shadow_out, size=test_out.shape[2:], mode='bilinear', align_corners=False)
        
        attention = self.fusion_attention(torch.cat([test_out, shadow_out], dim=1))
        final_out = attention * shadow_out + (1 - attention) * test_out
        
        return final_out
    
    def train_set(self, input, return_shadow=True):
        shadow_mask = input[:, 3:4, :, :]
        
        input_feats = self.encoder1(input)
        
        if self.placeholder is None or self.placeholder["x1"].size(0) != input_feats["x1"].size(0):
            self.placeholder = {}
            for key in input_feats.keys():
                self.placeholder[key] = torch.zeros(input_feats[key].shape, requires_grad=False) \
                    .to(torch.device(input_feats["x1"].device))
        
        shadow_removal_image = self.decoder(input_feats, input_feats)
        
        # Shadow decoder
        shadow_out = self.shadow_decoder(input_feats, shadow_mask)
        if shadow_out.shape[2:] != shadow_removal_image.shape[2:]:
            shadow_out = F.interpolate(shadow_out, size=shadow_removal_image.shape[2:], mode='bilinear', align_corners=False)
        
        attention = self.fusion_attention(torch.cat([shadow_removal_image, shadow_out], dim=1))
        enhanced_output = attention * shadow_out + (1 - attention) * shadow_removal_image
        
        if return_shadow:
            return enhanced_output, shadow_out
        else:
            return enhanced_output
        
        

# Discriminator    
class Discriminator(nn.Module):
    def __init__(self, input_channels=4):
        super(Discriminator, self).__init__()

        self.CB0 = Cvi(input_channels, 64)

        self.CB1 = Cvi(64, 128, before='LReLU', after='BN')

        self.CB2 = Cvi(128, 256, before='LReLU', after='BN')

        self.CB3 = Cvi(256, 512, before='LReLU', after='BN')

        self.CB4 = Cvi(512, 1, before='LReLU', after='sigmoid')

    def forward(self, input): 
        x0 = self.CB0(input)
        x1 = self.CB1(x0)
        x2 = self.CB2(x1)
        x3 = self.CB3(x2)
        out = self.CB4(x3)

        return out

if __name__ == '__main__':
    #BCHW
    size = (3, 3, 256, 256)
    input1 = torch.ones(size)
    input2 = torch.ones(size)
    l1 = nn.L1Loss()
   
    # #Discriminator test
    size = (3, 3, 256, 256)
    input = torch.ones(size)
    
# Assume the model is an instance of a class
#model = DPGLO_SRNet(input_channels=3, output_channels=3)  # Replace with the model you are using

# Print parameter shapes
#for name, param in model.named_parameters():
    #print(f"{name}: {param.shape}")

# Calculate the total number of parameters
#total_params = sum(p.numel() for p in model.parameters())
#print(f"Total number of parameters of DPGLO_SRNet: {total_params}")
