"""PyTorch inference layout matching NASA-IMPACT's released HLS crop model.

Architecture follows geospatial_fm/geospatial_fm.py and the published crop
configuration (Apache-2.0). All encoder, neck and both head tensors load strictly.
No randomly initialized layer is accepted as a pretrained model.
"""
import torch
from torch import nn
from timm.models.vision_transformer import Block

class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Conv3d(6,768,(1,16,16),stride=(1,16,16))
    def forward(self,x):return self.proj(x).flatten(2).transpose(1,2)

class TemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.patch_embed=PatchEmbed()
        self.cls_token=nn.Parameter(torch.zeros(1,1,768));self.pos_embed=nn.Parameter(torch.zeros(1,589,768),requires_grad=False)
        self.blocks=nn.ModuleList([Block(768,8,4,qkv_bias=True,norm_layer=nn.LayerNorm) for _ in range(6)])
        self.norm=nn.LayerNorm(768)
    def forward(self,x):
        x=self.patch_embed(x)+self.pos_embed[:,1:]
        x=torch.cat([(self.cls_token+self.pos_embed[:,:1]).expand(x.shape[0],-1,-1),x],1)
        for block in self.blocks:x=block(x)
        return self.norm(x)

class Norm2d(nn.Module):
    def __init__(self):super().__init__();self.ln=nn.LayerNorm(2304,eps=1e-6)
    def forward(self,x):return self.ln(x.permute(0,2,3,1)).permute(0,3,1,2).contiguous()

class TemporalNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.fpn1=nn.Sequential(nn.ConvTranspose2d(2304,2304,2,2),Norm2d(),nn.GELU(),nn.ConvTranspose2d(2304,2304,2,2))
        self.fpn2=nn.Sequential(nn.ConvTranspose2d(2304,2304,2,2),Norm2d(),nn.GELU(),nn.ConvTranspose2d(2304,2304,2,2))
    def forward(self,x):
        x=x[:,1:].permute(0,2,1).reshape(x.shape[0],2304,14,14)
        return self.fpn2(self.fpn1(x))

class ConvModule(nn.Module):
    def __init__(self,channels):
        super().__init__();self.conv=nn.Conv2d(channels,256,3,padding=1,bias=False);self.bn=nn.BatchNorm2d(256);self.activate=nn.ReLU(inplace=True)
    def forward(self,x):return self.activate(self.bn(self.conv(x)))

class FCNHead(nn.Module):
    def __init__(self,count):
        super().__init__();self.convs=nn.Sequential(*[ConvModule(2304 if i==0 else 256) for i in range(count)])
        self.dropout=nn.Dropout2d(.1);self.conv_seg=nn.Conv2d(256,13,1)
    def forward(self,x):return self.conv_seg(self.dropout(self.convs(x)))

class PrithviCrop(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=TemporalEncoder();self.neck=TemporalNeck();self.decode_head=FCNHead(1);self.auxiliary_head=FCNHead(2)
    def forward(self,x):return self.decode_head(self.neck(self.backbone(x)))
