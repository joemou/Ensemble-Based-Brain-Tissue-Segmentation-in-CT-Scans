import cv2
import numpy as np
import os
from PIL import Image
import torch

def fusion(img1, img2, device):
	imgFusion = (torch.sigmoid(img1) + torch.sigmoid(img2.to(device)))/2

	img1 = torch.sigmoid(img1)
	img1[img1>0.5]=1
	img1[img1<=0.5]=0

	img2 = torch.sigmoid(img2)
	img2[img2>0.5]=1
	img2[img2<=0.5]=0

	# and -> 交集
	filt2 = img1.clone().detach().type(torch.int8) & img2.to(device).clone().detach().type(torch.int8)
	return filt2 * imgFusion


def union(img1, img2, device):
	imgFusion = (torch.sigmoid(img1) + torch.sigmoid(img2.to(device)))/2

	img1 = torch.sigmoid(img1)
	img1[img1>0.5]=1
	img1[img1<=0.5]=0

	img2 = torch.sigmoid(img2)
	img2[img2>0.5]=1
	img2[img2<=0.5]=0
	
	# or -> 聯集
	filt2 = img1.clone().detach().type(torch.int8) | img2.to(device).clone().detach().type(torch.int8)
	return filt2 * imgFusion