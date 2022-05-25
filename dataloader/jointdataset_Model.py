import math
import torchvision.transforms.functional as F
import numpy as np
import cv2
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, make_label_dataset
from data.base_dataset import BaseDataset, get_transform
from PIL import Image

class jointdataset(BaseDataset):
	
	def initialize(self, opt):
		self.opt = opt
		self.root = opt.dataroot
		self.gt_im_paths = os.path.join(opt.dataroot, 'gt_image')
		self.de_im_paths = os.path.join(opt.dataroot, 'de_image')
		self.image_mask_paths = os.path.join(opt.dataroot, 'gt_image_mask')
		self.image_ves_mask_paths = os.path.join(opt.dataroot, 'gt_image_ves_mask')
		
		self.dir_IQA_label = os.path.join(opt.dataroot, 'IQA_label')
		self.dir_CLS_label = os.path.join(opt.dataroot, 'CLS_label')
		self.labels_IQA_records = make_label_dataset(self.dir_IQA_label)
		self.labels_CLS_records = make_label_dataset(self.dir_CLS_label)
		
		self.transform = get_transform(opt)
	
		# self.degrad = DegradImage(opt)
		# transform_list = [transforms.ToTensor(),
		# transforms.Normalize((0.5, 0.5, 0.5),
		# (0.5, 0.5, 0.5))]
		# self.transform = transforms.Compose(transform_list)
	
	def __getitem__(self, index):
		gt_im_path = os.path.join(self.gt_im_paths, self.labels_CLS_records.iloc[index].image) + '.png'
		de_im_path = os.path.join(self.de_im_paths, self.labels_CLS_records.iloc[index].image) + '.png'
		image_mask_path = os.path.join(self.image_mask_paths, self.labels_CLS_records.iloc[index].image) + '.png'
		image_ves_mask_path = os.path.join(self.image_ves_mask_paths, self.labels_CLS_records.iloc[index].image) + '-mask.png'
		
		gt_img = Image.open(gt_im_path).convert('RGB')
		gt_img = gt_img.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
		gt_img = self.transform(gt_img)
		gt_img = gt_img.numpy()
		
		de_img = Image.open(de_im_path).convert('RGB')
		de_img = de_img.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
		de_img = self.transform(de_img)
		de_img = de_img.numpy()
		
		image_ves_mask = np.array(Image.open(image_ves_mask_path).convert('L'))
		image_ves_mask = cv2.resize(image_ves_mask, (self.opt.fineSize, self.opt.fineSize))
		image_ves_mask = np.expand_dims(image_ves_mask, axis=2)
		image_ves_mask = np.array(image_ves_mask, np.float32).transpose(2, 0, 1) / 255.0
		image_ves_mask[image_ves_mask >= 0.5] = 1
		image_ves_mask[image_ves_mask < 0.5] = 0
		
		image_mask = Image.open(image_mask_path).convert('L')
		image_mask = image_mask.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
		image_mask = np.expand_dims(image_mask, axis=2)
		image_mask = np.array(image_mask, np.float32).transpose(2, 0, 1) / 255.0
		image_mask[image_mask >= 0.5] = 1
		image_mask[image_mask < 0.5] = 0
		
		norm_image = de_img * 3.2 - 1.6
		
		labels_IQA = torch.tensor(self.labels_IQA_records.iloc[index].quality)
		labels_cls = torch.tensor(self.labels_CLS_records.iloc[index].DR_grade)
		
		return {'de_img': de_img, 'gt_img': gt_img, 'gt_img_path': gt_im_path, 'norm_image': norm_image,
		        'mask_region': image_mask, 'ves_mask_region': image_ves_mask, 'labels_IQA': labels_IQA,
		        'labels_cls': labels_cls}
	
	def __len__(self):
		return len(self.labels_IQA_records)
	
	def name(self):
		return 'jointdataset'