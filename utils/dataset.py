from torch.utils.data import Dataset
from os.path import splitext, isfile, join
from os import listdir
import logging
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torchvision
from torchvision import transforms
import random
import torchvision.transforms.functional as TF
import cv2
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

def load_image(filename):
	ext = splitext(filename)[1]
	if ext == '.npy':
		return Image.fromarray(np.load(filename))
	elif ext in ['.pt', '.pth']:
		return Image.fromarray(torch.load(filename).numpy())
	elif ext == '.tiff':
		return cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
		# return Image.fromarray(cv2.imread(str(filename), cv2.IMREAD_UNCHANGED))
	else:
		return Image.open(filename)
		# return cv2.imread(filename, cv2.IMREAD_UNCHANGED)
		# return Image.fromarray(cv2.imread(filename, cv2.IMREAD_UNCHANGED))

def unique_mask_values(idx, mask_dir, mask_suffix):
	mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
	mask = np.asarray(load_image(mask_file))
	if mask.ndim == 2:
		return np.unique(mask)
	elif mask.ndim == 3:
		mask = mask.reshape(-1, mask.shape[-1])
		return np.unique(mask, axis=0)
	else:
		raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
	def __init__(self, imgs_dir, masks_dir, scale=1.0, mask_suffix='', transform=False):
		self.imgs_dir = Path(imgs_dir)
		self.masks_dir = Path(masks_dir)
		assert 0 < scale <= 1, 'Scale must be between 0 and 1'
		self.scale = scale
		# self.mask_suffix = mask_suffix
		self.transform = transform

		self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
					if isfile(join(imgs_dir, file)) and not file.startswith('.')]
		if not self.ids:
			raise RuntimeError(f'No input file found in {imgs_dir}, make sure you put your images there')

		logging.info(f'Creating dataset with {len(self.ids)} examples')
	
	def __len__(self):
		return len(self.ids)

	@staticmethod
	def preprocess(pil_img, scale, is_mask=False):
		w, h = pil_img.size
		newW, newH = int(scale * w), int(scale * h)
		assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
		pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
		
		img_nd = np.array(pil_img)
		if len(img_nd.shape) == 2:
			img_nd = np.expand_dims(img_nd, axis=2)

		# HWC to CHW
		img_trans = img_nd.transpose((2, 0, 1))
		if img_trans.max() > 1:
			img_trans = (img_trans - np.min(img_trans)) / (np.max(img_trans) - np.min(img_trans))

		return img_trans


	def data_transform(self, img, mask):
		## random transform
		prob = random.randint(0, 10)
		if prob >= 8:
			rot = random.randint(0, 45)
			img = TF.rotate(img, rot)
			mask = TF.rotate(mask, rot)

		prob1 = random.randint(0, 10)
		if prob1 >= 8:
			img = TF.hflip(img)
			mask = TF.hflip(mask)

		prob2 = random.randint(0, 10)
		if prob2 >= 8:
			img = TF.vflip(img)
			mask = TF.vflip(mask)

		return img, mask

	def __getitem__(self, idx):
		name = self.ids[idx]
		# mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
		mask_file = list(self.masks_dir.glob(name + '.*'))
		# img_file = list(self.imgs_dir.glob(name + '.*'))
		img_file = list(self.imgs_dir.glob(name + '.*'))

		assert len(img_file) == 1, \
			f'Either no image or multiple images found for the ID {name}: {img_file}'
		assert len(mask_file) == 1, \
			f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

		mask = load_image(mask_file[0])
		mask = mask.convert('L')
		img = load_image(img_file[0])

		lx = torchvision.transforms.Compose([
				torchvision.transforms.Resize([256, 256])
			])
		mask = lx(mask)
		mask = self.preprocess(mask, self.scale, True)

		if splitext(img_file[0])[1] == '.tiff':
			if np.max(img) - np.min(img) != 0:
				img = (img - np.min(img)) / (np.max(img) - np.min(img)) # normalize

			img2 = torch.as_tensor(img.copy()).float().contiguous().unsqueeze(0)
			mask2 = torch.as_tensor(mask.copy()).long().contiguous()

			if self.transform == True:
				img2, mask2 = self.data_transform(img2, mask2)

			return {
				'image': img2,
				'mask': mask2
			}

		else:
			img = img.convert('L')
			img = lx(img)
			img = self.preprocess(img, self.scale, False)
			assert img.size == mask.size, \
			f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

			img2 = torch.as_tensor(img.copy()).float().contiguous()
			mask2 = torch.as_tensor(mask.copy()).long().contiguous()

			if self.transform == True:
				img2, mask2 = self.data_transform(img2, mask2)

			return {
				'image': img2,
				'mask': mask2
			}


## for CTP image
class BasicDataset2(Dataset):
	def __init__(self, imgs_dir, masks_dir, scale=1.0, dir_CBF='', dir_CBV='', dir_MTT='', dir_Tmax='', mask_suffix='', transform=None):
		self.imgs_dir = Path(imgs_dir)
		self.masks_dir = Path(masks_dir)

		self.dir_CBF = Path(dir_CBF)
		self.dir_CBV = Path(dir_CBV)
		self.dir_MTT = Path(dir_MTT)
		self.dir_Tmax = Path(dir_Tmax)
		assert 0 < scale <= 1, 'Scale must be between 0 and 1'
		self.scale = scale

		self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
					if isfile(join(imgs_dir, file)) and not file.startswith('.')]
		if not self.ids:
			raise RuntimeError(f'No input file found in {imgs_dir}, make sure you put your images there')

		logging.info(f'Creating dataset with {len(self.ids)} examples')
	
	def __len__(self):
		return len(self.ids)

	@staticmethod
	def preprocess(pil_img, scale, is_mask=False):
		w, h = pil_img.size
		newW, newH = int(scale * w), int(scale * h)
		assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
		pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
		
		img_nd = np.array(pil_img)
		if len(img_nd.shape) == 2:
			img_nd = np.expand_dims(img_nd, axis=2)

		# HWC to CHW
		img_trans = img_nd.transpose((2, 0, 1))
		if img_trans.max() > 1:
			img_trans = (img_trans - np.min(img_trans)) / (np.max(img_trans) - np.min(img_trans))

		return img_trans


	def data_transform(self, img, mask):
		prob = random.randint(0, 10)
		if prob >= 8:
			rot = random.randint(0, 45)
			img = TF.rotate(img, rot)
			mask = TF.rotate(mask, rot)

		prob1 = random.randint(0, 10)
		if prob1 >= 8:
			img = TF.hflip(img)
			mask = TF.hflip(mask)

		prob2 = random.randint(0, 10)
		if prob2 >= 8:
			img = TF.vflip(img)
			mask = TF.vflip(mask)

		return img, mask

	def __getitem__(self, idx):
		name = self.ids[idx]
		# mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
		mask_file = list(self.masks_dir.glob(name + '.*'))
		# img_file = list(self.imgs_dir.glob(name + '.*'))
		img_file = list(self.imgs_dir.glob(name + '.*'))

		CBF = list(self.dir_CBF.glob(name + '.*'))
		CBV = list(self.dir_CBV.glob(name + '.*'))
		MTT = list(self.dir_MTT.glob(name + '.*'))
		Tmax = list(self.dir_Tmax.glob(name + '.*'))

		assert len(img_file) == 1, \
			f'Either no image or multiple images found for the ID {name}: {img_file}'
		assert len(mask_file) == 1, \
			f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

		mask = load_image(mask_file[0])
		mask = mask.convert('L')
		img = load_image(img_file[0])

		perfusion_cbf = load_image(CBF[0])
		perfusion_cbv = load_image(CBV[0])
		perfusion_mtt = load_image(MTT[0])
		perfusion_tmax = load_image(Tmax[0])

		lx = torchvision.transforms.Compose([
				torchvision.transforms.Resize([256, 256])
			])
		mask = lx(mask)
		mask = self.preprocess(mask, self.scale, True)

		if splitext(img_file[0])[1] == '.tiff':
			if np.max(img) - np.min(img) != 0:
				img = (img - np.min(img)) / (np.max(img) - np.min(img)) # normalize
				perfusion_cbf = (perfusion_cbf - np.min(perfusion_cbf)) / (np.max(perfusion_cbf) - np.min(perfusion_cbf))
				perfusion_cbv = (perfusion_cbv - np.min(perfusion_cbv)) / (np.max(perfusion_cbv) - np.min(perfusion_cbv))
				perfusion_mtt = (perfusion_mtt - np.min(perfusion_mtt)) / (np.max(perfusion_mtt) - np.min(perfusion_mtt))
				perfusion_tmax = (perfusion_tmax - np.min(perfusion_tmax)) / (np.max(perfusion_tmax) - np.min(perfusion_tmax))

			img2 = torch.as_tensor(img.copy()).float().contiguous().unsqueeze(0)
			perfusion_cbf2 = torch.as_tensor(perfusion_cbf.copy()).float().contiguous().unsqueeze(0)
			perfusion_cbv2 = torch.as_tensor(perfusion_cbv.copy()).float().contiguous().unsqueeze(0)
			perfusion_mtt2 = torch.as_tensor(perfusion_mtt.copy()).float().contiguous().unsqueeze(0)
			perfusion_tmax2 = torch.as_tensor(perfusion_tmax.copy()).float().contiguous().unsqueeze(0)

			mask2 = torch.as_tensor(mask.copy()).long().contiguous()
			img2 = torch.cat((img2, perfusion_cbf2, perfusion_cbv2, perfusion_mtt2, perfusion_tmax2), 0)
			img2, mask2 = self.data_transform(img2, mask2)

			return {
				'image': img2,
				'mask': mask2
			}

		else:
			img = img.convert('L')
			img = lx(img)
			img = self.preprocess(img, self.scale, False)
			assert img.size == mask.size, \
			f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

			img2 = torch.as_tensor(img.copy()).float().contiguous()
			mask2 = torch.as_tensor(mask.copy()).long().contiguous()
			img2, mask2 = self.data_transform(img2, mask2)

			return {
				'image': img2,
				'mask': mask2
			}


