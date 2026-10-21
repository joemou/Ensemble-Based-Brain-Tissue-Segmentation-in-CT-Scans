## https://github.com/vlbthambawita/divergent-nets/blob/2aaadc565c4f93f656d0d44f07593afab70fe3ff/my_models/triple_models.py

import torch
from typing import Optional, Union, List
import segmentation_models_pytorch as smp
from .unet2022 import unet2022
from monai.networks.nets.swin_unetr import SwinUNETR

class TripleUnet(smp.UnetPlusPlus):
    
    def __init__(self, 
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,):
        
        super().__init__()
        
        self.in_model_1 = smp.Unet(
                encoder_name=encoder_name,
                in_channels=in_channels, 
                encoder_weights=encoder_weights, 
                classes=classes, 
                activation=activation,)
        
        self.in_model_2 = smp.Unet(
                encoder_name=encoder_name,
                in_channels=in_channels, 
                encoder_weights=encoder_weights, 
                classes=classes, 
                activation=activation,)
        
        self.out_model = smp.Unet(
                encoder_name=encoder_name,
                in_channels=2, 
                encoder_weights=encoder_weights, 
                classes=classes, 
                activation=activation,)
        
    def forward(self,x):

        mask_1 = self.in_model_1(x)
        mask_2 = self.in_model_2(x)
        
        mask_concat = torch.cat((mask_1, mask_2), 1)
        
        mask = self.out_model(mask_concat)

        return mask

class TripleUnet_2(smp.UnetPlusPlus):
    
    def __init__(self, 
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,):
        
        super().__init__()
        
        self.in_model_1 = smp.Unet(
                encoder_name=encoder_name,
                in_channels=in_channels, 
                encoder_weights=encoder_weights, 
                classes=classes, 
                activation=activation,)
        
        self.in_model_2 = smp.Unet(
                encoder_name=encoder_name,
                in_channels=in_channels, 
                encoder_weights=encoder_weights, 
                classes=classes, 
                activation=activation,)
        
        self.out_model = smp.Unet(
                encoder_name=encoder_name,
                in_channels=3, 
                encoder_weights=encoder_weights, 
                classes=classes, 
                activation=activation,)
        
    def forward(self,x):

        mask_1 = self.in_model_1(x)
        mask_2 = self.in_model_2(x)

        mask_3 = (mask_1+mask_2)/2
        mask__1 = torch.sigmoid(mask_1)
        mask__1[mask__1>0.5]=1
        mask__1[mask__1<=0.5]=0
        mask__2 = torch.sigmoid(mask_2)
        mask__2[mask__2>0.5]=1
        mask__2[mask__2<=0.5]=0
        filt2 = mask__1.clone().detach().type(torch.int8) | mask__2.clone().detach().type(torch.int8)
        filt = mask__1.clone().detach().type(torch.int8) & mask__2.clone().detach().type(torch.int8)
        mask_5 = filt2 * mask_3 #union
        mask_4 = filt * mask_3 #intersection


        # mask_concat = torch.cat((mask_4, mask_5), 1)
        mask_concat = torch.cat((mask_1, mask_2, mask_4), 1)
        
        mask = self.out_model(mask_concat)
        # mask = self.out_model(mask_4)

        return mask 

class TripleUnet2022(smp.UnetPlusPlus):
    
    def __init__(self, n_channels=1, n_classes=1):
        
        super().__init__()
        
        self.in_model_1 = unet2022(n_channels=n_channels, n_classes=1)
        
        self.in_model_2 = unet2022(n_channels=n_channels, n_classes=1)
        
        self.out_model = unet2022(n_channels=2, n_classes=1)
        
    def forward(self,x):

        mask_1 = self.in_model_1(x)
        mask_2 = self.in_model_2(x)
        
        mask_concat = torch.cat((mask_1, mask_2), 1)
        
        mask = self.out_model(mask_concat)

        return mask 

class TripleUnet2022_2(smp.UnetPlusPlus):
    
    def __init__(self, n_channels=1, n_classes=1):
        
        super().__init__()
        
        self.in_model_1 = unet2022(n_channels=n_channels, n_classes=1)
        self.in_model_2 = unet2022(n_channels=n_channels, n_classes=1)      
        self.out_model = unet2022(n_channels=3, n_classes=1)
        
    def forward(self,x):

        mask_1 = self.in_model_1(x)
        mask_2 = self.in_model_2(x)


        mask_3 = (mask_1+mask_2)/2
        mask__1 = torch.sigmoid(mask_1)
        mask__1[mask__1>0.5]=1
        mask__1[mask__1<=0.5]=0
        mask__2 = torch.sigmoid(mask_2)
        mask__2[mask__2>0.5]=1
        mask__2[mask__2<=0.5]=0
        filt2 = mask__1.clone().detach().type(torch.int8) | mask__2.clone().detach().type(torch.int8)
        filt = mask__1.clone().detach().type(torch.int8) & mask__2.clone().detach().type(torch.int8)
        mask_5 = filt2 * mask_3 #union
        mask_4 = filt * mask_3 #fusion


        
        mask_concat = torch.cat((mask_1, mask_2, mask_4), 1)
        
        mask = self.out_model(mask_concat)
        # mask = self.out_model(mask_4)

        return mask 


class TripleUnet_UNet2022(smp.UnetPlusPlus):
    
    def __init__(self, 
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,):
        
        super().__init__()
        
        # self.in_model_1 = smp.Unet(
        #         encoder_name=encoder_name,
        #         in_channels=in_channels, 
        #         encoder_weights=encoder_weights, 
        #         classes=classes, 
        #         activation=activation,)
        
        # self.in_model_1 = smp.DeepLabV3Plus(
        #         encoder_name = "efficientnet-b3",
        #         encoder_weights= None,
        #         in_channels=in_channels, 
        #         classes=1
        #     )

        self.in_model_1 = SwinUNETR(in_channels=in_channels, out_channels=1, img_size=256, feature_size=48, norm_name=("group", {"num_groups": 12}), spatial_dims=2)
        
        self.in_model_2 = unet2022(n_channels=in_channels, n_classes=1)
        
        self.out_model = smp.Unet(
                encoder_name=encoder_name,
                in_channels=3, 
                encoder_weights=encoder_weights, 
                classes=classes, 
                activation=activation,)
        # self.out_model = unet2022(n_channels=2, n_classes=1)
        
    def forward(self,x):

        mask_1 = self.in_model_1(x)
        mask_2 = self.in_model_2(x)

        mask_3 = (mask_1+mask_2)/2
        # mask_3 = (torch.sigmoid(mask_1)+torch.sigmoid(mask_2))/2
        mask__1 = torch.sigmoid(mask_1)
        mask__1[mask__1>0.5]=1
        mask__1[mask__1<=0.5]=0
        mask__2 = torch.sigmoid(mask_2)
        mask__2[mask__2>0.5]=1
        mask__2[mask__2<=0.5]=0
        filt2 = mask__1.clone().detach().type(torch.int8) | mask__2.clone().detach().type(torch.int8)
        filt = mask__1.clone().detach().type(torch.int8) & mask__2.clone().detach().type(torch.int8)
        mask_5 = filt2 * mask_3 #union
        mask_4 = filt * mask_3 #fusion
        
        mask_concat = torch.cat((mask_1, mask_2, mask_4), 1)
        # mask_concat = torch.cat((mask_1, mask_2), 1)
        
        mask = self.out_model(mask_concat)
        # mask = self.out_model(mask_4)

        return mask