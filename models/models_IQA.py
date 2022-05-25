import torch
import util.util as utilss
import numpy as np

from os.path import join, exists
from torch.autograd import Variable

from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from util.metrics import *
from models.base_model import BaseModel

from . import networks
from models.losses import init_loss

np.seterr(divide='ignore',invalid='ignore')
class model_IQA(BaseModel):
	def name(self):
		return 'model_IQA'

	def initialize(self, opt):
		BaseModel.initialize(self, opt)
		self.isTrain = opt.isTrain
		self.input_gt_img = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
		self.input_norm_img = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
		self.input_gt_blur = self.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
		self.input_gt_spot = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
		self.input_region = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
		self.label = self.Tensor(opt.batchSize)

		if opt.model_choice == 'EyesImage_IQA_with_masklabel' or opt.model_choice == 'EyesImage_IQAresnet_with_masklabel':
			if exists(join(opt.project_root, opt.segm_spot_weight_root)):
				opt.segm_spot_weight_root = opt.segm_spot_weight_root
			else:
				opt.segm_spot_weight_root = 'none'
			self.net = networks.define_net(opt.model_choice, opt.segm_spot_weight_root, opt.norm, self.gpu_ids)
		elif opt.model_choice == 'EyesImage_IQA_with_image_blur_label' or opt.model_choice == 'EyesImage_IQAresnet_with_image_blur_label':
			if exists((join(opt.project_root, opt.SR_blur_weight_root))):
				opt.SR_blur_weight_root = opt.SR_blur_weight_root
			else:
				opt.SR_blur_weight_root = 'none'
			self.net = networks.define_net(opt.model_choice, opt.SR_blur_weight_root, opt.norm, self.gpu_ids)

		if not self.isTrain or opt.continue_train:

			if opt.model_choice == 'EyesImage_IQA_with_masklabel':
				self.load_network(self.net, 'IQA_mask', opt.which_epoch)
			elif opt.model_choice == 'EyesImage_IQAresnet_with_masklabel':
				self.load_network(self.net, 'IQAres_mask', opt.which_epoch)
			elif opt.model_choice == 'EyesImage_IQA_with_image_blur_label':
				self.load_network(self.net, 'IQA_blur', opt.which_epoch)
			elif opt.model_choice == 'EyesImage_IQAresnet_with_image_blur_label':
				self.load_network(self.net, 'IQA_res_blur', opt.which_epoch)

		if self.isTrain:
			self.old_lr = opt.lr

			self.optimizer_G = torch.optim.Adam(self.net.parameters(),
												lr=opt.lr, betas=(opt.beta1, 0.999))

			self.clsLoss, self.segmenLoss = init_loss(opt, self.Tensor)

		print('---------- Networks initialized -------------')
		networks.print_network(self.net)
		print('-----------------------------------------------')

	def set_input(self, input):
		input_gt_img = input['gt_img']
		input_gt_blur = input['gt_blur']
		input_norm_img = input['norm_img']
		# input_gt_spot = input['gt_spot']
		input_region = input['region']
		label = input['labels_cls']
		self.image_paths = input['gt_im_paths']
		self.label = self.Tensor(label.shape)
		self.label.copy_(label)
		self.input_gt_img.resize_(input_gt_img.size()).copy_(input_gt_img)
		self.input_gt_blur.resize_(input_gt_blur.size()).copy_(input_gt_blur)
		self.input_norm_img.resize_(input_norm_img.size()).copy_(input_norm_img)
		# self.input_gt_spot.resize_(input_gt_spot.size()).copy_(input_gt_spot)
		self.input_region.resize_(input_region.size()).copy_(input_region)

	def forward(self, opt):
		self.input_norm_img = Variable(self.input_norm_img)
		self.in_gt_img = Variable(self.input_gt_img)
		self.in_gt_blur = Variable(self.input_gt_blur)
		# self.input_gt_spot = Variable(self.input_gt_spot)
		self.input_region = Variable(self.input_region)
		self.in_labels = Variable(self.label)
		self.in_labels2 = self.in_labels.long()
		if opt.model_choice == 'EyesImage_IQA_with_masklabel' or opt.model_choice == 'EyesImage_IQAresnet_with_masklabel':
			self.labels_pre, self.spot_map = self.net.forward(self.in_gt_img, self.input_norm_img)

		elif opt.model_choice == 'EyesImage_IQA_with_image_blur_label' or opt.model_choice == 'EyesImage_IQAresnet_with_image_blur_label':
			self.labels_pre, self.imgblur = self.net.forward(self.in_gt_img)

	def backward_net(self, opt, epoch):
		if opt.model_choice == 'EyesImage_IQA_with_masklabel' or opt.model_choice == 'EyesImage_IQAresnet_with_masklabel':
			self.loss_net_Segmen = self.segmenLoss.get_loss(self.input_gt_spot * self.input_region,
															self.spot_map * self.input_region)  # * self.opt.lambda_S
			psnr_list = []
			ssim_list = []
			for i in range(len(self.image_paths)):
				in_gt_spot = self.input_gt_spot[i] * self.input_region[i]
				in_gt_spot = torch.unsqueeze(in_gt_spot, 0)
				spot_map = self.spot_map[i] * self.input_region[i]
				spot_map = torch.unsqueeze(spot_map, 0)
				in_gt_spot = utilss.tensor2im(in_gt_spot.data)
				in_gt_spot = in_gt_spot.astype(np.uint8)
				spot_map = utilss.tensor2im(spot_map.data)
				spot_map = spot_map.astype(np.uint8)
				psnr = PSNR(in_gt_spot, spot_map)
				psnr_list.append(psnr)
				ssim = SSIM(in_gt_spot, spot_map, channel_axis=-1)
				ssim_list.append(ssim)

			self.loss_labels_pre = self.clsLoss.get_loss(self.labels_pre, self.in_labels2)
			out = self.labels_pre
			label = self.in_labels2
			self.acc = calculate_accuracy(out, label)
			self.kappa = compute_kappa(out, label)
			self.f1 = compute_f1(out, label)

			self.loss_G = 0.1*self.loss_net_Segmen + self.loss_labels_pre

			self.loss_G.backward()

			return psnr_list, ssim_list, self.acc, self.kappa, self.f1, self.loss_net_Segmen, self.loss_labels_pre

		elif opt.model_choice == 'EyesImage_IQA_with_image_blur_label' or opt.model_choice == 'EyesImage_IQAresnet_with_image_blur_label':
			self.loss_net_Segmen = self.segmenLoss.get_loss(self.in_gt_blur * self.input_region,
															self.imgblur * self.input_region)  # * self.opt.lambda_S
			psnr_list = []
			ssim_list = []
			for i in range(len(self.image_paths)):
				in_gt_blur = self.in_gt_blur[i] * self.input_region[i]
				in_gt_blur = torch.unsqueeze(in_gt_blur, 0)
				imgblur = self.imgblur[i] * self.input_region[i]
				imgblur = torch.unsqueeze(imgblur, 0)
				in_gt_img = utilss.tensor2im(in_gt_blur.data)
				in_gt_img = in_gt_img.astype(np.uint8)
				imgblur = utilss.tensor2im(imgblur.data)
				imgblur = imgblur.astype(np.uint8)
				psnr = PSNR(in_gt_img, imgblur)
				psnr_list.append(psnr)
				ssim = SSIM(in_gt_img, imgblur, channel_axis=-1)# multichannel=True
				ssim_list.append(ssim)

			self.loss_labels_pre = self.clsLoss.get_loss(self.labels_pre, self.in_labels2)
			out = self.labels_pre
			label = self.in_labels2
			self.acc = calculate_accuracy(out, label)
			self.kappa = compute_kappa(out, label)
			self.f1 = compute_f1(out, label)

			self.loss_G = self.loss_net_Segmen + self.loss_labels_pre

			self.loss_G.backward()
			return psnr_list, ssim_list, self.acc, self.kappa, self.f1, self.loss_net_Segmen, self.loss_labels_pre

	def optimize_parameters(self, opt, epoch):
		self.forward(opt)
		self.optimizer_G.zero_grad()
		psnr, ssim, acc, kappa, f1, loss_blur, loss_IQA = self.backward_net(opt, epoch)
		self.optimizer_G.step()
		return psnr, ssim, acc, kappa, f1, loss_blur, loss_IQA

	def save(self, opt, label):
		if opt.model_choice == 'EyesImage_IQA_with_masklabel':
			self.save_network(self.net, 'IQA_mask', label, self.gpu_ids)
		elif opt.model_choice == 'EyesImage_IQAresnet_with_masklabel':
			self.save_network(self.net, 'IQAres_mask', label, self.gpu_ids)
		elif opt.model_choice == 'EyesImage_IQA_with_image_blur_label':
			self.save_network(self.net, 'IQA_blur', label, self.gpu_ids)
		elif opt.model_choice == 'EyesImage_IQAresnet_with_image_blur_label':
			self.save_network(self.net, 'IQAres_blur', label, self.gpu_ids)

	def update_learning_rate(self):
		lrd = self.opt.lr / self.opt.niter_decay
		lr = self.old_lr - lrd
		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = lr
		print('update learning rate: %f -> %f' % (self.old_lr, lr))
		self.old_lr = lr
