import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.iou_score import multiclass_iou_coeff, iou_coeff
from  utils.hdLoss import HausdorffDTLoss
import torchvision

# =============================================================================
#  路徑設定
# =============================================================================
inputPath = './output/'
# maskPath = '../../stroke_TVGH_ISLES/ISLES_remove_background_8_2/fold_0/val/masks/'
# maskPath = '../../stroke_TVGH_ISLES/train_TVGH/masks/'

# maskPath = 'dataset/fold_0/test/masks/'
# maskPath = 'TVGH_dataset/fold_0/test/masks/'

maskPath = '../dataset/brain_TGVH_ISLES/test_ISLES/masks_hard/'
# maskPath = 'kaggle_dataset/fold_0/test/masks/'



torch.manual_seed(0)
torch.cuda.manual_seed(0)


def preprocess(pil_img, scale):
	w, h = pil_img.size
	newW, newH = int(scale * w), int(scale * h)
	assert newW > 0 and newH > 0, 'Scale is too small'
	pil_img = pil_img.resize((newW, newH))
	img = np.asarray(pil_img)
	flag = True

	if img.ndim == 2:
		img = img[np.newaxis, ...]
	else:
		img = img.transpose((2, 0, 1))

	if(img > 1).any():
		## 預測影像為全黑?
		if np.max(img) - np.min(img) == 0:
			img = img
			flag = False
		else:
			img = (img - np.min(img)) / (np.max(img) - np.min(img))

	return img, flag


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
	in_files = next(os.walk(inputPath))[2]
	n_test = len(in_files)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	img_scale = 1
	n_class = 1

	dice_score = 0
	iou_score = 0
	loss_loc = HausdorffDTLoss()
	hd = 0
	FPN_score = 0
	cnt = 0

	with tqdm(total=n_test, desc='Test Round', unit='batch', leave=False) as pbar:
		for i in in_files:
			pred = Image.open(inputPath + i)
			pred = pred.convert('L')
			pred, flag1 = preprocess(pred, img_scale)
			pred = torch.from_numpy(pred).type(torch.FloatTensor)
			pred = pred.to(device=device, dtype=torch.float32)

			mask = Image.open(maskPath + i)
			mask = mask.convert('L')

			lx = torchvision.transforms.Compose([
				torchvision.transforms.Resize([pred.shape[1],pred.shape[1]])
			])
			mask = lx(mask)
			mask, flag2 = preprocess(mask, img_scale)
			mask = torch.from_numpy(mask).type(torch.FloatTensor)
			mask = mask.to(device=device, dtype=torch.float32)

			dice_score += dice_coeff(pred, mask, reduce_batch_first=False)
			iou_score += iou_coeff(pred.squeeze(1), mask, reduce_batch_first=False)

			## ===========================================================================
			## confusion matrix
			## ===========================================================================
			# target = mask.cpu().unsqueeze(0)
			# preds = pred.cpu().unsqueeze(0)

			# FP = len(np.where(preds - target  == 1)[0])
			# FN = len(np.where(preds - target  == -1)[0])
			# TP = len(np.where(preds + target ==2)[0])
			# TN = len(np.where(preds + target == 0)[0])
			## ===========================================================================


			
			## 預測結果為全黑時，不計算HD
			if flag1 == True:
				if n_class == 1:	
					cnt += 1		
					hd += loss_loc.forward(pred.unsqueeze(0), mask.unsqueeze(0))
			pbar.update()

	DSC = (dice_score / max(n_test, 1)).item()
	IoU = (iou_score / max(n_test, 1)).item()
	HD = (hd / cnt).item()

	logging.info('Dice coeff: {}'.format(DSC))
	logging.info('IoU coeff: {}'.format(IoU))
	logging.info('HD: {}'.format(HD))