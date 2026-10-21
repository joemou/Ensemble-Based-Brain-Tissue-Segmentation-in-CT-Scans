import argparse
import logging
import torch
import sys
import os
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

from utils.dataset import BasicDataset, BasicDataset2
from combine.combineImage import fusion, union
from ranger2020 import Ranger
from model import UNet_3Plus, unet2022, TripleUnet

from monai.losses import DiceFocalLoss, DiceLoss, FocalLoss, DiceCELoss, GeneralizedDiceLoss
from monai.networks.nets.unetr import UNETR
from monai.networks.nets.swin_unetr import SwinUNETR
import segmentation_models_pytorch as smp

# ===============================================================================================
#  路徑設定
# ===============================================================================================
dir_train_img = '../../stroke_TVGH_ISLES/ISLES_remove_background_8_2/fold_0/train/imgs/'
dir_train_mask = '../../stroke_TVGH_ISLES/ISLES_remove_background_8_2/fold_0/train/masks/'

# dir_train_img = '../../stroke_TVGH_ISLES/train_TVGH/imgs/'
# dir_train_mask = '../../stroke_TVGH_ISLES/train_TVGH/masks/'

## CTP path
dir_CBF = '../../stroke_TVGH_ISLES/ISLES_remove_background_8_2/CBF/'
dir_CBV = '../../stroke_TVGH_ISLES/ISLES_remove_background_8_2/CBV/'
dir_MTT = '../../stroke_TVGH_ISLES/ISLES_remove_background_8_2/MTT/'
dir_Tmax = '../../stroke_TVGH_ISLES/ISLES_remove_background_8_2/Tmax/'

dir_checkpoint = 'checkpoints/'

# ===============================================================================================
# 參數設定
# ===============================================================================================
epoch = 50
epoch_store = 100 #how many epoch to store
batch = 4
lr = 0.0001
torch.manual_seed(0)
torch.cuda.manual_seed(0)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=epoch, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=batch, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=lr,
                        help='Learning rate', dest='lr')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    return parser.parse_args()

def train_model(model, model2, device, device2, device3, epochs=5, batch_size=1, learning_rate=0.001,
				img_scale=1, save_checkpoint=True, amp=False, gradient_clipping=1.0):
	
	# train = BasicDataset(dir_train_img, dir_train_mask, img_scale, transform=True) ## CT: 1 channel
	train = BasicDataset2(dir_train_img, dir_train_mask, img_scale, dir_CBF, dir_CBV, dir_MTT, dir_Tmax, transform=True) ## CT+CTP: 5 channels

	n_train = len(train)
	loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=False,)
	train_loader = DataLoader(train, shuffle=True, **loader_args, drop_last=True)

	logging.info(f'''Starting training:
		Epochs:          {epochs}
		Batch size:      {batch_size}
		Learning rate:   {learning_rate}
		Training rate:   {n_train}
		Checkpoints:     {save_checkpoint}
		Device:          {device}
		Images scaling:  {img_scale}
		Mixed Precision: {amp}
	''')

	## ======================================================================================================
	## Optimizer!!!!!!!!!!
	## ======================================================================================================
	optimizer = Ranger(model.parameters(), lr=learning_rate)
	optimizer2 = Ranger(model2.parameters(), lr=learning_rate)

	## ======================================================================================================
	## Scheduler!!!!!!!!!!
	## ======================================================================================================
	scheduler=optim.lr_scheduler.LinearLR(optimizer,start_factor=1,end_factor=1e-5,total_iters=epochs)
	scheduler2 = optim.lr_scheduler.LinearLR(optimizer2, start_factor=1,end_factor=1e-5,total_iters=epochs)

	## ======================================================================================================
	## Loss Function
	## ======================================================================================================
	criterion = DiceCELoss(sigmoid=True, lambda_dice=1, lambda_ce=1)
	criterion2 = DiceCELoss(sigmoid=False, lambda_dice=1, lambda_ce=1)

	## ======================================================================================================
	## Initial
	## ======================================================================================================
	global_step = 0
	best_dice = 10.
	best_model1_dice = 10.
	best_model2_dice = 10.

	best_loss = 10.
	best_loss_model1 = 10.
	best_loss_model2 = 10.

	best_model_params = copy.deepcopy(model.state_dict())
	best_model_params2 = copy.deepcopy(model2.state_dict())
	best_loss_model_params = copy.deepcopy(model.state_dict())
	best_loss_model_params2 = copy.deepcopy(model2.state_dict())

	## =================================================================================================
	## Start Training
	## =================================================================================================

	for epoch in range(1, epochs+1):
		model.train()
		model2.train()

		with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
			epoch_loss = 0
			epoch_loss1 = 0
			epoch_loss2 = 0
			num_batch = 0

			for batch in train_loader:
				num_batch += 1
				images, true_masks = batch['image'], batch['mask']
				images2 = images

				images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
				true_masks = true_masks.to(device=device, dtype=torch.float32)
				images2 = images2.to(device=device2, dtype=torch.float32, memory_format=torch.channels_last)
				true_masks2 = true_masks.to(device=device2, dtype=torch.long)

				with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
					masks_pred = model(images)
					masks_pred2 = model2(images2)

					## ===================================================================
					## Ensmeble
					## ===================================================================
					# fusion_pred = fusion(masks_pred, masks_pred2, device)
					union_pred = union(masks_pred, masks_pred2, device)

					loss1 = criterion(masks_pred, true_masks)
					loss2 = criterion(masks_pred2, true_masks2)
					loss = criterion2(union_pred.to(device3), true_masks.to(device3))

				optimizer.zero_grad(set_to_none=True) # 清空前一次的gradient
				optimizer2.zero_grad(set_to_none=True)

				final_loss = loss.to(device) #+ loss1 + loss2.to(device)
				final_loss.backward()

				torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
				torch.nn.utils.clip_grad_norm_(model2.parameters(), gradient_clipping)
				optimizer.step() ## 更新權重
				optimizer2.step()

				pbar.update(images.shape[0])
				global_step += 1
				epoch_loss += loss.item() #+ loss1.item() + loss2.item()
				epoch_loss1 += loss1.item()
				epoch_loss2 += loss2.item()
				pbar.set_postfix(**{'loss (batch)': loss.item()})

			scheduler.step()
			scheduler2.step()
			logging.info('Ensemble Loss: {}'.format(epoch_loss/num_batch))
			logging.info('(Training Loss) Model1: {} Model2: {}'.format(epoch_loss1/num_batch, epoch_loss2/num_batch))

			with open('./txt/train_loss.txt', 'a') as f:
				f.write(str(epoch_loss/num_batch))
				f.write('\n')
				f.close()
			with open('./txt/train_loss1.txt', 'a') as f:
				f.write(str(epoch_loss1/num_batch))
				f.write('\n')
				f.close()
			with open('./txt/train_loss2.txt', 'a') as f:
				f.write(str(epoch_loss2/num_batch))
				f.write('\n')
				f.close()

			## store best checkpoint
			val_loss = (epoch_loss1/num_batch)
			val_loss2 = (epoch_loss2/num_batch)

			if save_checkpoint:
				if val_loss < best_model1_dice:
					best_model1_dice = val_loss
					best_model_params = copy.deepcopy(model.state_dict())
					logging.info('Model1 Save Best Model: {}!!'.format(best_model1_dice))

				if val_loss2 < best_model2_dice:
					best_model2_dice = val_loss2
					best_model_params2 = copy.deepcopy(model2.state_dict())
					logging.info('Model2 Save Best Model: {}!!'.format(best_model2_dice))

			if save_checkpoint and (epoch_loss/num_batch) < best_loss:
				best_loss = epoch_loss/num_batch
				best_loss_model1 = val_loss
				best_loss_model_params = copy.deepcopy(model.state_dict())
				best_loss_model2 = val_loss2
				best_loss_model_params2 = copy.deepcopy(model2.state_dict())
				logging.info('Best Loss {} save !'.format(best_loss))

		## 每隔多少epoch_store存一次模型
		if save_checkpoint and (epoch % epoch_store == 0):
			if val_loss < val_loss2:
				torch.save(model, dir_checkpoint + f'Epoch{epoch}_Ensemble_valDice_{val_loss}.pth')
				logging.info('Checkpoint {} save !'.format(epoch))
			else:
				torch.save(model2, dir_checkpoint + f'Epoch{epoch}_Ensemble_valDice_{val_loss2}.pth')
				logging.info('Checkpoint {} save !'.format(epoch))

	model.load_state_dict(best_model_params)
	torch.save(model, dir_checkpoint + f'Best_Ensemble_{best_dice}_model1_{best_model1_dice}.pth')
	model2.load_state_dict(best_model_params2)
	torch.save(model2, dir_checkpoint + f'Best_Ensemble_{best_dice}_model2_{best_model2_dice}.pth')

	model.load_state_dict(best_loss_model_params)
	torch.save(model, dir_checkpoint + f'Best_Ensemble_combination_{best_loss}_model1_{best_loss_model1}.pth')
	model2.load_state_dict(best_loss_model_params2)
	torch.save(model2, dir_checkpoint + f'Best_Ensemble_combination_{best_loss}_model2_{best_loss_model2}.pth')


if __name__ == '__main__':
	args = get_args()
	logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logging.info(f'Using device {device}')
	device1 = "cuda:0"
	device2 = "cuda:1"
	device3 = "cuda:2"


	# model = smp.Unet(
	# 	encoder_name = "vgg16",
	# 	encoder_weights= None,
	# 	in_channels=1, 
	# 	classes=1,
	# 	decoder_use_batchnorm=False
	# )

	# model = UNet_3Plus(in_channels=1, n_classes=1)
	# model = unet2022(n_channels=1, n_classes=1)

	# model = smp.DeepLabV3Plus(
	# 	encoder_name = "efficientnet-b3",
	# 	encoder_weights= None,
	# 	in_channels=1, 
	# 	classes=1
	# )

	# model = UNETR(in_channels=1, out_channels=1, img_size=256, feature_size=32, norm_name=("group", {"num_groups": 16}), spatial_dims=2)
	# model = SwinUNETR(in_channels=1, out_channels=1, img_size=256, feature_size=48, norm_name=("group", {"num_groups": 12}), spatial_dims=2)

	# model = TripleUnet(
	# 	encoder_name = "vgg16",
	# 	encoder_weights= None,
	# 	in_channels=1, 
	# 	classes=1,
	# 	decoder_use_batchnorm=False
	# )


	## ===================================================================
	## Training with Ensemble (choose two models)
	## ===================================================================

	model = smp.Unet(
		encoder_name = "vgg16",
		encoder_weights= None,
		in_channels=5, 
		classes=1,
		decoder_use_batchnorm=False
	)
	model = model.to(device1)

	model2 = unet2022(n_channels=5, n_classes=1)
	model2 = model2.to(device2)

	## ===================================================================


	try:
		train_model(
			model = model,
			model2 = model2,
			epochs = args.epochs,
			batch_size = args.batch_size,
			learning_rate = args.lr,
			device = device,
			device2 = device2,
			device3 = device3,
			img_scale = args.scale,
			amp = args.amp
		)
	except KeyboardInterrupt:
		torch.save(model, 'INTERUPT.pth')
		try:
			sys.exit(0)
		except SystemExit:
			os._exit(0)