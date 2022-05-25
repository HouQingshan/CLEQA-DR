import torch
import torch.nn as nn
import torch.nn.functional as F


class ClsLoss():
	def initialize(self, loss):
		self.criterion = loss
	
	def get_loss(self, prlabel, reallabel):
		return self.criterion(prlabel, reallabel)


class ContentLoss():
	def initialize(self, loss):
		self.criterion = loss
	
	def get_loss(self, fakeIm, realIm):
		return self.criterion(fakeIm, realIm)


def init_loss(opt, tensor):
	
	if opt.model_choice == 'EyesImage_IQA_with_masklabel' or opt.model_choice == 'EyesImage_IQAresnet_with_masklabel':
		cls_loss = ClsLoss()
		cls_loss.initialize(nn.CrossEntropyLoss())
		segmen_loss = dice_bce_loss()
		segmen_loss.initialize()
		return cls_loss, segmen_loss

	elif opt.model_choice == 'EyesImage_IQA_with_image_blur_label' or opt.model_choice == 'EyesImage_IQAresnet_with_image_blur_label':
		cls_loss = ClsLoss()
		cls_loss.initialize(nn.CrossEntropyLoss())
		segmen_loss = ContentLoss()
		segmen_loss.initialize(nn.L1Loss())
		return cls_loss, segmen_loss
	elif opt.model_choice == 'EyesImage_CLS_with_resnet' or opt.model_choice == 'EyesImage_CLS_with_MyNet':
		cls_loss = ClsLoss()
		cls_loss.initialize(nn.CrossEntropyLoss())
		return cls_loss
	elif opt.model_choice == 'EyesImage_SR':
		content_loss = ContentLoss()
		content_loss.initialize(nn.L1Loss())
		segmen_loss = dice_bce_loss()
		segmen_loss.initialize()
		return content_loss, segmen_loss
	elif opt.model_choice == 'EyesImage_joint':
		cls_loss = ClsLoss()
		cls_loss.initialize(nn.CrossEntropyLoss())
		
		content_loss = ContentLoss()
		content_loss.initialize(nn.L1Loss())
		
		segmen_loss = dice_bce_loss()
		segmen_loss.initialize()
		
		return cls_loss, content_loss, segmen_loss
	else:
		raise ValueError("Model [%s] not recognized." % opt.model)


class dice_bce_loss(nn.Module):
	def initialize(self, batch=True):
		super(dice_bce_loss, self).__init__()
		self.batch = batch
		self.bce_loss = nn.BCELoss()
		self.ce_loss = nn.CrossEntropyLoss()
		self.softmax = torch.nn.Softmax(dim=1)
	
	def soft_dice_coeff(self, y_true, y_pred):
		smooth = 0.0  # may change
		if self.batch:
			i = torch.sum(y_true)
			j = torch.sum(y_pred)
			intersection = torch.sum(y_true * y_pred)
		else:
			i = y_true.sum(1).sum(1).sum(1)
			j = y_pred.sum(1).sum(1).sum(1)
			intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
		score = (2. * intersection + smooth) / (i + j + smooth)
		# score = (intersection + smooth) / (i + j - intersection + smooth)#iou
		return score.mean()
	
	def soft_dice_loss(self, y_true, y_pred):
		loss = 1 - self.soft_dice_coeff(y_true, y_pred)
		return loss
	
	def get_loss(self, y_true, y_pred):
		a = self.bce_loss(y_pred, y_true)
		# b = self.soft_dice_loss(y_true, y_pred)
		return a
