import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from utils.dataset import BasicDataset
from os.path import splitext, isfile, join
import cv2

# =============================================================================
#  路徑設定
# =============================================================================

modelPath = './checkpoints/123.pth'
modelPath2 = './checkpoints/789.pth'

## brain tissue segmentataion
# input_path = '../dataset/brain_TGVH_ISLES/test_ISLES/imgs_hard/'
# input_path = '../dataset/brain_TGVH_ISLES/test_ISLES/imgs/'

## stroke segmentataion
input_path = '../dataset/stroke_TVGH_ISLES/ISLES_remove_background_5_5/fold_0/val/imgs/'
# input_path = '../../stroke_TVGH_ISLES/train_TVGH/imgs/'



## CTP path
# dir_CBF = '../../stroke_TVGH_ISLES/ISLES_remove_background_8_2/CBF/'
# dir_CBV = '../../stroke_TVGH_ISLES/ISLES_remove_background_8_2/CBV/'
# dir_MTT = '../../stroke_TVGH_ISLES/ISLES_remove_background_8_2/MTT/'
# dir_Tmax = '../../stroke_TVGH_ISLES/ISLES_remove_background_8_2/Tmax/'

output_path = './output/'

# =============================================================================
# 參數設定
# =============================================================================
torch.manual_seed(0)
torch.cuda.manual_seed(0)
output_size = 512

def fusion(img1, img2, device):
	imgFusion = (torch.sigmoid(img1.to(device)) + torch.sigmoid(img2.to(device)))/2

	img1 = torch.sigmoid(img1)
	img1[img1>0.5]=1
	img1[img1<=0.5]=0

	img2 = torch.sigmoid(img2)
	img2[img2>0.5]=1
	img2[img2<=0.5]=0

	filt2 = img1.to(device).clone().detach().type(torch.int8) & img2.to(device).clone().detach().type(torch.int8)
	return filt2 * imgFusion

def union(img1, img2, device):
	imgFusion = (torch.sigmoid(img1.to(device)) + torch.sigmoid(img2.to(device)))/2

	img1 = torch.sigmoid(img1)
	img1[img1>0.5]=1
	img1[img1<=0.5]=0

	img2 = torch.sigmoid(img2)
	img2[img2>0.5]=1
	img2[img2<=0.5]=0

	filt2 = img1.to(device).clone().detach().type(torch.int8) | img2.to(device).clone().detach().type(torch.int8)
	return filt2 * imgFusion

def predict_img(net,
				net2,
				full_img,
				# cbf,
				# cbv,
				# mtt,
				# tmax,
				device,
				scale_factor=1,
				out_threshold=0.5,
				folder='',
				flag='.png'):
	
	net.eval()
	net2.eval()
	if flag == '.png':
		img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
	else:
		full_img = (full_img - np.min(full_img)) / (np.max(full_img) - np.min(full_img))
		img = torch.as_tensor(full_img).unsqueeze(0)

		# cbf = (cbf - np.min(cbf)) / (np.max(cbf) - np.min(cbf))
		# cbf2 = torch.as_tensor(cbf).unsqueeze(0)

		# cbv = (cbv - np.min(cbv)) / (np.max(cbv) - np.min(cbv))
		# cbv2 = torch.as_tensor(cbv).unsqueeze(0)

		# mtt = (mtt - np.min(mtt)) / (np.max(mtt) - np.min(mtt))
		# mtt2 = torch.as_tensor(mtt).unsqueeze(0)

		# tmax = (tmax - np.min(tmax)) / (np.max(tmax) - np.min(tmax))
		# tmax2 = torch.as_tensor(tmax).unsqueeze(0)
	
	img = img.unsqueeze(0)

	# cbf2 = cbf2.unsqueeze(0)
	# cbv2 = cbv2.unsqueeze(0)
	# mtt2 = mtt2.unsqueeze(0)
	# tmax2 = tmax2.unsqueeze(0)
	# img = torch.cat((img, cbf2, cbv2, mtt2, tmax2), 1)

	img = img.to(device=device, dtype=torch.float32)

	with torch.no_grad():
		output = net(img).cpu()
		output2 = net2(img).cpu()

		## 2024.05.14 modified, 將回傳圖片更改為交集/聯集後的結果
		new = fusion(output, output2, device)
		# new = union(output, output2, device)

		## 先放大成原圖的size再threshold
		if flag == '.png':
			# output = F.interpolate(output, (output_size, output_size), mode='bilinear') # 不使用output
			new = F.interpolate(output, (output_size, output_size), mode='bilinear') # 使用兩者交互作用的圖
		
		# mask = torch.sigmoid(output) > out_threshold # 不使用output
		mask = torch.sigmoid(new) > out_threshold # 使用兩者交互作用的圖

	return mask[0].long().squeeze().cpu().numpy() # 2024.05.14 modified, add Tensor.cpu()


def mask_to_image(mask):
	return Image.fromarray((mask.astype(np.uint8)*255).astype(np.uint8))


def get_args():
	parser = argparse.ArgumentParser(description='Predict masks from input images')
	parser.add_argument('--model', '-m', default=modelPath, metavar='FILE',
						help='Specify the files in which the model is stored')
	parser.add_argument('--model2', '-m2', default=modelPath2, metavar='FILE',
						help='Specify the files in which the model is stored')
	parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
						help='Minimum probability value to consider a mask pixel white')
	parser.add_argument('--scale', '-s', type=float, default=1,
						help='Scale factor for the input images')
	parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
	parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
	return parser.parse_args()

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
	args = get_args()
	in_files = next(os.walk(input_path))[2]
	out_files = in_files.copy()

	for i in range(len(out_files)):
		spl = out_files[i].split('.', 1)
		out_files[i] = spl[0] + '.png'

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	net = torch.load(args.model, map_location=device)
	net = net.cuda(device)
	net.eval()

	net2 = torch.load(args.model2, map_location=device)
	net2 = net2.cuda(device)
	net2.eval()

	logging.info("Model Load !")

	for i, fileName in enumerate(in_files):
		flag = splitext(fileName)[1] 
		if splitext(fileName)[1] == '.tiff':
			img = cv2.imread(input_path + str(fileName), cv2.IMREAD_UNCHANGED)

			# cbf = cv2.imread(dir_CBF + str(fileName), cv2.IMREAD_UNCHANGED)
			# cbv = cv2.imread(dir_CBV + str(fileName), cv2.IMREAD_UNCHANGED)
			# mtt = cv2.imread(dir_MTT + str(fileName), cv2.IMREAD_UNCHANGED)
			# tmax = cv2.imread(dir_Tmax + str(fileName), cv2.IMREAD_UNCHANGED)
		else:
			img = Image.open(input_path + fileName)
			img = img.convert('L')
			lx = transforms.Compose([
					transforms.Resize([output_size, output_size])
				])
			img = lx(img)

		mask = predict_img(net=net,
							net2=net2,
							full_img=img,
							# cbf=cbf,
							# cbv=cbv,
							# mtt=mtt,
							# tmax=tmax,
							scale_factor=args.scale,
							out_threshold=args.mask_threshold,
							device=device,
							flag=flag)

		if not args.no_save:
			out_filename = out_files[i]
			result = mask_to_image(mask)
			result.save(output_path+out_filename)
			logging.info('Mask saved to {}'.format(output_path+out_filename))
