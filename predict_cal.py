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

from matplotlib import pyplot as plt # modified, add new package
from datetime import datetime # modified, add new package

# =============================================================================
#  路徑設定
# =============================================================================

inputPath = './output/'
# fusionPath = './fusionImg/' # 2024.07.01 modified

## brain tissue segmentataion
# maskPath = '../dataset/brain_TGVH_ISLES/test_ISLES/masks_hard/'
# maskPath = '../dataset/brain_TGVH_ISLES/test_ISLES/masks/'
# maskPath = 'dataset/fold_0/test/masks/'
# maskPath = 'TVGH_dataset/fold_0/test/masks/'
# maskPath = 'kaggle_dataset/fold_0/test/masks/'

## stroke segmentataion
maskPath = '../dataset/stroke_TVGH_ISLES/ISLES_remove_background_5_5/fold_0/val/masks/'
# maskPath = '../../stroke_TVGH_ISLES/ISLES_remove_background_8_2/fold_0/val/masks/'
# maskPath = '../../stroke_TVGH_ISLES/train_TVGH/masks/'



torch.manual_seed(0)
torch.cuda.manual_seed(0)

# 2024.07.01 modified, 把預測結果加入mask視覺化
def fusionResult(pred, mask):
	opacity = 0.5 		# 不透明度
	alpha = 1 - opacity # 透明度
	pred_img = Image.open(pred).convert("L")
	pred_alpha = Image.new("RGBA", pred_img.size) # 要蓋上去的有透明度的圖片(pred)

	for x in range(pred_img.width): # img x axis
		for y in range(pred_img.height): # img y axis
			alpha_value = pred_img.getpixel((x,y))
			# (R,G,B,A) -> (Red,0,0,alpha)
			pred_alpha.putpixel((x, y), (alpha_value, 0, 0, int(alpha*255)))

	mask_overlay = Image.open(mask).convert("RGBA")
	pred_alpha = pred_alpha.resize(mask_overlay.size) # 確認和mask原圖大小相同
	combine_image = Image.alpha_composite(mask_overlay, pred_alpha)

	return combine_image



def preprocess(pil_img, scale): # scale -> 圖片放大幅度
	w, h = pil_img.size
	newW, newH = int(scale * w), int(scale * h)
	assert newW > 0 and newH > 0, 'Scale is too small'
	pil_img = pil_img.resize((newW, newH))
	img = np.asarray(pil_img)
	flag = True

	if img.ndim == 2: # ndim -> numpy的number of dumention
		img = img[np.newaxis, ...]
	else:
		img = img.transpose((2, 0, 1))

	# normalize
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
	loss_loc = HausdorffDTLoss() # HD
	hd = 0
	FPN_score = 0
	cnt = 0

	min_dice = 0

	with tqdm(total=n_test, desc='Test Round', unit='batch', leave=False) as pbar:
		
		TP,TN,FP,FN = [],[],[],[] # 2024.05.13 modified, add confusion matrix list
		# dice_oneImage,totalFalse_oneImage = [],[]

		for i in in_files:
			# 2024.07.01 modified, 將evaluation輸出加入mask做視覺化輸出，因為要以RGBA的方式打開，所以寫在最前面
			# fusionImage = fusionResult(inputPath + i, maskPath + i)

			pred = Image.open(inputPath + i) # ./output/...png
			pred = pred.convert('L') # 轉成灰階
			pred, flag1 = preprocess(pred, img_scale) # input
			pred = torch.from_numpy(pred).type(torch.FloatTensor)
			pred = pred.to(device=device, dtype=torch.float32)

			mask = Image.open(maskPath + i)
			mask = mask.convert('L')
			lx = torchvision.transforms.Compose([
				torchvision.transforms.Resize([pred.shape[1],pred.shape[1]])
			])
			mask = lx(mask)
			mask, flag2 = preprocess(mask, img_scale) # mask
			mask = torch.from_numpy(mask).type(torch.FloatTensor)
			mask = mask.to(device=device, dtype=torch.float32)
			
			# 2024.06.24 modified, 讓每個dice結果分別輸出，儲存於dice_oneImage list
			dice = dice_coeff(pred, mask, reduce_batch_first=False)
			# dice_oneImage.append(float(dice))
			dice_score += dice
			iou_score += iou_coeff(pred.squeeze(1), mask, reduce_batch_first=False)

			# fusionImage.save(fusionPath + str(dice.item())[:4] + i) # ./fusionImg/...png
			# fusionImage.save(fusionPath + i) # ./fusionImg/...png

			# 2024.07.02 modified, 找出最小的dice，方便找戰犯
			# if (dice > min_dice):
			# 	min_dice = dice
				
			## ===========================================================================
			## 2024.05.13 modified, unmarked confusion matrix
			## ===========================================================================
			target = mask.cpu().unsqueeze(0)
			preds = pred.cpu().unsqueeze(0)

			TP.append(len(np.where(preds + target == 2)[0]))
			TN.append(len(np.where(preds + target == 0)[0]))
			FP.append(len(np.where(preds - target == 1)[0]))
			FN.append(len(np.where(preds - target == -1)[0]))

			# totalFalse_oneImage.append(len(np.where(preds != target)[0]))
			## ===========================================================================


			
			## 預測結果為全黑時，不計算HD
			if flag1 == True:
				if n_class == 1:	
					cnt += 1		
					hd += loss_loc.forward(pred.unsqueeze(0), mask.unsqueeze(0))
			pbar.update()

	# # 2024.06.24 modified, 新增predict_dice檔案紀錄所有dice的數據
	# with open('./false_record/predict_dice.txt', 'w') as f:
	# 	f.write(str(dice_oneImage))
	# 	f.write('\n')
	# 	f.close()

	# # 2024.06.27 modified, 新增predict_totalFalse檔案紀錄所有dice的數據
	# with open('./false_record/predict_totalFalse.txt', 'w') as f:
	# 	f.write(str(totalFalse_oneImage))
	# 	f.write('\n')
	# 	f.close()

	# 2024.06.27 modified, 新增Dice分布plt
	# base_path = os.path.dirname(os.path.abspath(__file__))
	# x = list(range(0, len(dice_oneImage)))
	# y = dice_oneImage
	# plt.figure()
	# plt.plot(x,y)
	# plt.ylim([0.7, 1.0])
	# plt.xlabel("Images")
	# plt.ylabel("Validation DSC")
	# plt.grid(True)
	# plt.savefig(os.path.join(base_path, "fusionImg", "ValidCurve_" + datetime.today().strftime('%Y%m%d%H%M%S') + ".png"))

	# # 2024.06.28 modified, 新增False Pixel分布plt
	# x = list(range(0, len(totalFalse_oneImage)))
	# y = totalFalse_oneImage
	# plt.figure()
	# plt.plot(x,y)
	# plt.xlabel("Images")
	# plt.ylabel("False Pixel")
	# plt.grid(True)
	# plt.savefig(os.path.join(base_path, "fusionImg", "FalseCurve_" + datetime.today().strftime('%Y%m%d%H%M%S') + ".png"))



	DSC = (dice_score / max(n_test, 1)).item()
	IoU = (iou_score / max(n_test, 1)).item()
	HD = (hd / cnt).item()

	logging.info('Dice coeff: {}'.format(DSC))
	logging.info('IoU coeff: {}'.format(IoU))
	logging.info('HD: {}'.format(HD))

	# 2024.05.13 modified, log confusion matrix
	logging.info('---------- confusion matrix ----------')
	total_pixel = n_test*256*256
	logging.info(f'TP: {sum(TP)/total_pixel:.4f}    TP amount: {sum(TP)}')
	logging.info(f'TN: {sum(TN)/total_pixel:.4f}    TN amount: {sum(TN)}')
	logging.info(f'FP: {sum(FP)/total_pixel:.4f}    FP amount: {sum(FP)}') # overkill
	logging.info(f'FN: {sum(FN)/total_pixel:.4f}    FN amount: {sum(FN)}') 
	logging.info(f'\t\t\tTotal False: {sum(FN)+sum(FP)}')
	logging.info('--------------------------------------')

	# print(sum(totalFalse_oneImage))
	# print("Min dice:", min(dice_oneImage))
