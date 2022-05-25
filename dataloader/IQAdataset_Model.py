import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, make_label_dataset
from PIL import Image
import math
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import util.util as util
from motion_blur.degrad_image import DegradImage
from PIL import Image
import numpy as np
import cv2
import json


class IQAdataset(BaseDataset):
	def initialize(self, opt):
		self.opt = opt
		self.root = opt.dataroot
		self.gt_im_paths = os.path.join(opt.dataroot, 'gt_img_ori')
		# self.gt_spot_paths = os.path.join(opt.dataroot, 'mask_spot')
		self.gt_blur_paths = os.path.join(opt.dataroot, 'gt_img_blur')
		self.region_paths = os.path.join(opt.dataroot, 'mask_ori')
		self.dir_IQA_label = os.path.join(opt.dataroot, 'IQA_label')
		self.labels_records = make_label_dataset(self.dir_IQA_label)
		
		self.transform = get_transform(opt)
		
		# self.degrad = DegradImage(opt)
		# transform_list = [transforms.ToTensor(),
		# transforms.Normalize((0.5, 0.5, 0.5),
		# (0.5, 0.5, 0.5))]
		# self.transform = transforms.Compose(transform_list)
	
	def __getitem__(self, index):
		
		gt_im_path = os.path.join(self.gt_im_paths, self.labels_records.iloc[index].image)+'.png'
		# gt_spot_path = os.path.join(self.gt_spot_paths, self.labels_records.iloc[index].image)+'.png'
		gt_blur_path = os.path.join(self.gt_blur_paths, self.labels_records.iloc[index].image)+'.png'
		region_path = os.path.join(self.region_paths, self.labels_records.iloc[index].image)+'.png'
		
		gt_img = Image.open(gt_im_path).convert('RGB')
		gt_img = gt_img.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
		
		gt_blur = Image.open(gt_blur_path).convert('RGB')
		gt_blur = gt_blur.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
		
		
		region = Image.open(region_path).convert('L')
		region = region.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
		region = np.expand_dims(region, axis=2)
		region = np.array(region, np.float32).transpose(2, 0, 1) / 255.0
		region[region >= 0.5] = 1
		region[region < 0.5] = 0
	
		# gt_spot = np.array(Image.open(gt_spot_path).convert('L'))
		# gt_spot = cv2.resize(gt_spot, (self.opt.fineSize, self.opt.fineSize))
		# gt_spot = np.expand_dims(gt_spot, axis=2)
		# gt_spot = np.array(gt_spot, np.float32).transpose(2, 0, 1) / 255.0
		# gt_spot[gt_spot >= 0.5] = 1
		# gt_spot[gt_spot < 0.5] = 0
		
		gt_img = self.transform(gt_img)
		gt_blur = self.transform(gt_blur)
		
		gt_img = gt_img.numpy()
		gt_blur = gt_blur.numpy()
	
		norm_img = gt_img * 3.2 - 1.6
		
		labels_cls = torch.tensor(self.labels_records.iloc[index].quality)
		
		return {'gt_img': gt_img, 'gt_im_paths': gt_im_path, 'gt_blur': gt_blur, 'norm_img': norm_img, #'gt_spot': gt_spot,
		        'region': region, 'labels_cls': labels_cls}
	
	def __len__(self):
		return len(self.labels_records)
	
	def name(self):
		return 'IQAdataset'
	
	def tensor_to_np(tensor):
		img = tensor.mul(255).byte()
		img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
		return img


def DE_COLOR(img, brightness=0.0, contrast=0.0, saturation=0.0):
	"""Randomly change the brightness, contrast and saturation of an gt_image_512"""
	if brightness > 0:
		brightness_factor = random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
		img = F.adjust_brightness(img, brightness_factor)
	if contrast > 0:
		contrast_factor = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
		img = F.adjust_contrast(img, contrast_factor)
	if saturation > 0:
		saturation_factor = random.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation)
		img = F.adjust_saturation(img, saturation_factor)
	return img


def DE_SPOT(img, h, w, center=None, radius=None):
	s_num = random.randint(5, 20)
	mask0 = np.zeros((h, w))
	for i in range(s_num):
		
		radius = random.randint(math.ceil(0.026 * h), int(0.05 * h))
		center = [random.randint(radius + 1, w - radius - 1), random.randint(radius + 1, h - radius - 1)]
		Y, X = np.ogrid[:h, :w]
		dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
		circle = dist_from_center <= (int(radius / 2))
		
		k = 0.7 + (1.0 - 0.7) * random.random()
		beta = 0.5 + (1.5 - 0.5) * random.random()
		A = k * np.ones((3, 1))
		d = 0.3 * random.random()
		t = math.exp(-beta * d)
		
		mask = np.zeros((h, w))
		mask[circle] = np.multiply(A[0], (1 - t))
		mask0 = mask0 + mask
		mask = cv2.GaussianBlur(mask, (5, 5), 1.8)
		mask = np.array([mask, mask, mask])
		img = img + mask
		img = np.maximum(img, 0)
		img = np.minimum(img, 1)
	
	return img, mask0


def DE_HALO(img, h, w, center=None, radius=None):
	w0_a = random.randint(w / 2 - int(w / 8), w / 2 + int(w / 8))
	h0_a = random.randint(h / 2 - int(h / 8), h / 2 + int(h / 8))
	center_a = [w0_a, h0_a]
	
	wei_dia_a = 0.75 + (1.0 - 0.75) * random.random()
	dia_a = min(h, w) * wei_dia_a
	Y_a, X_a = np.ogrid[:h, :w]
	dist_from_center_a = np.sqrt((X_a - center_a[0]) ** 2 + (Y_a - center_a[1]) ** 2)
	circle_a = dist_from_center_a <= (int(dia_a / 2))
	
	mask_a = np.zeros((h, w))
	mask_a[circle_a] = np.mean(img)
	
	center_b = center_a
	Y_b, X_b = np.ogrid[:h, :w]
	dist_from_center_b = np.sqrt((X_b - center_b[0]) ** 2 + (Y_b - center_b[1]) ** 2)
	
	dia_b_max = 2 * int(np.sqrt(
		max(center_a[0], h - center_a[0]) * max(center_a[0], h - center_a[0]) + max(center_a[1], h - center_a[1]) * max(
			center_a[1], w - center_a[1]))) / min(w, h)
	wei_dia_b = 1.0 + (dia_b_max - 1.0) * random.random()
	dia_b = min(h, w) * wei_dia_b + abs(max(center_b[0] - w / 2, center_b[1] - h / 2) + max(w, h) / 2)
	
	circle_b = dist_from_center_b <= (int(dia_b / 2))
	
	mask_b = np.zeros((h, w))
	mask_b[circle_b] = np.mean(img)  # np.multiply(A[0], (1 - t))
	delta_circle = np.abs(mask_a - mask_b)
	
	sigma = random.randint(int(min(w, h) / 6), min(w, h) / 2) / 3
	gauss_rad = int(sigma * 1.5)
	if (gauss_rad % 2) == 0:
		gauss_rad = gauss_rad + 1
	delta_circle = cv2.GaussianBlur(delta_circle, (gauss_rad, gauss_rad), sigma)
	
	weight_r = [255 / 255, 141 / 255, 162 / 255]
	weight_g = [255 / 255, 238 / 255, 205 / 255]
	weight_b = [255 / 255, 238 / 255, 90 / 255]
	
	num = random.randint(0, 2)
	delta_circle = np.array([weight_r[num] * delta_circle, weight_g[num] * delta_circle, weight_b[num] * delta_circle])
	img = img + delta_circle
	
	img = np.maximum(img, 0)
	img = np.minimum(img, 1)
	
	return img


def DE_BLUR(img, h, w, center=None, radius=None):
	img = (np.transpose(img, (1, 2, 0)))
	
	sigma = 0 + (15 - 0) * random.random()
	rad_w = random.randint(int(sigma / 3), int(sigma / 2))
	rad_h = random.randint(int(sigma / 3), int(sigma / 2))
	if (rad_w % 2) == 0: rad_w = rad_w + 1
	if (rad_h % 2) == 0: rad_h = rad_h + 1
	
	img = cv2.GaussianBlur(img, (rad_w, rad_h), sigma)
	img = (np.transpose(img, (2, 0, 1)))
	
	img = np.maximum(img, 0)
	img = np.minimum(img, 1)
	
	return img


def DE_HOLE(img, h, w, center=None, diameter=None):
	diameter_circle = random.randint(int(0.3 * w), int(0.5 * w))
	center = [random.randint(1, w - 1), random.randint(1, h - 1)]
	Y, X = np.ogrid[:h, :w]
	dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
	circle = dist_from_center <= (int(diameter_circle / 2))
	
	mask = np.zeros((h, w))
	mask[circle] = 1
	
	brightness = -0.05
	brightness_factor = random.uniform((brightness - 0.2 * 1), min(brightness, 0))
	
	mask = mask * brightness_factor

	sigma = random.uniform(diameter_circle / 4, diameter_circle / 3)
	
	rad_w = random.randint(int(diameter_circle / 4), int(diameter_circle / 3))
	rad_h = random.randint(int(diameter_circle / 4), int(diameter_circle / 3))
	if (rad_w % 2) == 0: rad_w = rad_w + 1
	if (rad_h % 2) == 0: rad_h = rad_h + 1
	
	mask = cv2.GaussianBlur(mask, (rad_w, rad_h), sigma)
	mask = np.array([mask, mask, mask])
	img = img + mask
	img = np.maximum(img, 0)
	img = np.minimum(img, 1)
	
	return img
