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


class model_joint(BaseModel):
    def name(self):
        return 'model_joint'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.input_gt_img = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.input_de_img = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.input_norm_img = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.img_region_mask = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
        self.img_ves_mask = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
        self.label_cls = self.Tensor(opt.batchSize)
        self.label_IQA = self.Tensor(opt.batchSize)

        if exists(join(opt.project_root, opt.segm_vessel_weight_root)):
            opt.segm_vessel_weight_root = 'null'
        self.net = networks.define_net(opt.model_choice, opt.segm_vessel_weight_root, opt.norm, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.net, 'joint', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr

            self.optimizer_G = torch.optim.Adam(self.net.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.clsLoss, self.contentLoss, self.segmenLoss = init_loss(opt, self.Tensor)

        print('---------- Networks initialized -------------')
        networks.print_network(self.net)
        print('-----------------------------------------------')

    def set_input(self, input):
        # {'de_img': de_img, 'gt_img': gt_img, 'gt_img_path': gt_im_path, 'norm_image': norm_image,
        # 'mask_region': image_mask, 'ves_mask_region': image_ves_mask, 'labels_IQA': labels_IQA,
        # 'labels_cls': labels_cls}
        input_de_img = input['de_img']
        input_gt_img = input['gt_img']
        input_norm_image = input['norm_image']
        input_mask_region = input['mask_region']
        input_ves_mask_region = input['ves_mask_region']
        label_IQA = input['labels_IQA']
        label_cls = input['labels_cls']
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_paths = input['gt_img_path']
        self.label_IQA = self.Tensor(label_IQA.shape).to(device)
        self.label_cls = self.Tensor(label_cls.shape).to(device)
        self.label_IQA.copy_(label_IQA).to(device)
        self.label_cls.copy_(label_cls).to(device)
        self.input_de_img.resize_(input_de_img.size()).copy_(input_de_img).to(device)
        self.input_gt_img.resize_(input_gt_img.size()).copy_(input_gt_img).to(device)
        self.input_norm_img.resize_(input_norm_image.size()).copy_(input_norm_image).to(device)
        self.img_region_mask.resize_(input_mask_region.size()).copy_(input_mask_region).to(device)
        self.img_ves_mask.resize_(input_ves_mask_region.size()).copy_(input_ves_mask_region).to(device)

    def forward(self, opt):
        self.in_de_img = Variable(self.input_de_img)
        self.in_gt_img = Variable(self.input_gt_img)
        self.in_img_norm = Variable(self.input_norm_img)
        self.in_img_mask = Variable(self.img_region_mask)
        self.in_img_ves_mask = Variable(self.img_ves_mask)
        self.in_labels_IQA = Variable(self.label_IQA)
        self.in_labels_IQA2 = self.in_labels_IQA.long()
        self.in_labels_cls = Variable(self.label_cls)
        self.in_labels_cls2 = self.in_labels_cls.long()
        # out_ves, output_512, label_pre_cls, out_blur, label_pre_IQA
        self.ves_mask_pre, self.SR_image, self.labels_cls_pre, self.blur_image, self.labels_IQA_pre = self.net.forward(
            self.in_de_img, self.in_img_norm)


    def backward_net(self, opt, epoch):
        # CLS
        self.labels_cls_loss = self.clsLoss.get_loss(self.labels_cls_pre, self.in_labels_cls2)
        out = self.labels_cls_pre
        label = self.in_labels_cls2
        self.cls_acc = calculate_accuracy(out, label)
        self.cls_kappa = compute_kappa(out, label)
        self.cls_f1 = compute_f1(out, label)

        # IQA
        self.labels_IQA_loss = self.clsLoss.get_loss(self.labels_IQA_pre, self.in_labels_IQA2)
        out = self.labels_IQA_pre
        label = self.in_labels_IQA2
        self.IQA_acc = calculate_accuracy(out, label)
        self.IQA_kappa = compute_kappa(out, label)
        self.IQA_f1 = compute_f1(out, label)

        # ves_Dice
        self.seg_ves_loss = self.segmenLoss.get_loss(self.in_img_ves_mask * self.in_img_mask,
                                                     self.ves_mask_pre * self.in_img_mask)

        # # SR
        # self.SR_image_loss = self.contentLoss.get_loss(self.SR_image * self.in_img_mask,
        #                                                self.in_gt_img * self.in_img_mask)
        # SR_psnr_list = []
        # SR_ssim_list = []
        # for i in range(len(self.image_paths)):
        #     in_gt_img = self.in_gt_img[i] * self.in_img_mask[i]
        #     in_gt_img = torch.unsqueeze(in_gt_img, 0)
        #     SR_image = self.SR_image[i] * self.in_img_mask[i]
        #     SR_image = torch.unsqueeze(SR_image, 0)
        #     in_gt_img = utilss.tensor2im(in_gt_img.data)
        #     in_gt_img = in_gt_img.astype(np.uint8)
        #     SR_image = utilss.tensor2im(SR_image.data)
        #     SR_image = SR_image.astype(np.uint8)
        #     psnr = PSNR(in_gt_img, SR_image)
        #     SR_psnr_list.append(psnr)
        #     ssim = SSIM(in_gt_img, SR_image, channel_axis=-1)
        #     SR_ssim_list.append(ssim)

        # SR_blur
        self.SR_blur_image_loss = self.contentLoss.get_loss(self.blur_image * self.in_img_mask,
                                                            self.in_de_img * self.in_img_mask)
        blur_psnr_list = []
        blur_ssim_list = []
        for i in range(len(self.image_paths)):
            in_gt_blur = self.in_de_img[i] * self.in_img_mask[i]
            in_gt_blur = torch.unsqueeze(in_gt_blur, 0)
            imgblur = self.blur_image[i] * self.in_img_mask[i]
            imgblur = torch.unsqueeze(imgblur, 0)
            in_gt_img = utilss.tensor2im(in_gt_blur.data)
            in_gt_img = in_gt_img.astype(np.uint8)
            imgblur = utilss.tensor2im(imgblur.data)
            imgblur = imgblur.astype(np.uint8)
            psnr = PSNR(in_gt_img, imgblur)
            blur_psnr_list.append(psnr)
            ssim = SSIM(in_gt_img, imgblur, channel_axis=-1)
            blur_ssim_list.append(ssim)

        # self.loss_all = self.labels_cls_loss + self.labels_IQA_loss + self.seg_ves_loss + self.SR_image_loss + self.SR_blur_image_loss
        self.loss_all = self.labels_cls_loss + self.labels_IQA_loss + 0.1*self.seg_ves_loss +  self.SR_blur_image_loss # 0.4 0.12
        self.loss_all.backward()

        # return self.cls_acc, self.cls_kappa, self.cls_f1, self.IQA_acc, self.IQA_kappa, self.IQA_f1, SR_psnr_list, SR_ssim_list, blur_psnr_list, blur_ssim_list, self.labels_cls_loss, self.labels_IQA_loss, self.seg_ves_loss, self.SR_image_loss, self.SR_blur_image_loss
        return self.cls_acc, self.cls_kappa, self.cls_f1, self.IQA_acc, self.IQA_kappa, self.IQA_f1, blur_psnr_list, blur_ssim_list, self.labels_cls_loss, self.labels_IQA_loss, self.seg_ves_loss,self.SR_blur_image_loss
    def optimize_parameters(self, opt, epoch):
        self.forward(opt)
        self.optimizer_G.zero_grad()
        # cls_acc, cls_kappa, cls_f1, IQA_acc, IQA_kappa, IQA_f1, SR_psnr, SR_ssim, blur_psnr, blur_ssim, cls_loss, IQA_loss, ves_loss, SR_loss, blur_loss = self.backward_net(
        #     opt, epoch)
        cls_acc, cls_kappa, cls_f1, IQA_acc, IQA_kappa, IQA_f1, blur_psnr, blur_ssim, cls_loss, IQA_loss, ves_loss, blur_loss = self.backward_net(
             opt, epoch)
        self.optimizer_G.step()
        # return cls_acc, cls_kappa, cls_f1, IQA_acc, IQA_kappa, IQA_f1, SR_psnr, SR_ssim, blur_psnr, blur_ssim, cls_loss, IQA_loss, ves_loss, SR_loss, blur_loss
        return cls_acc, cls_kappa, cls_f1, IQA_acc, IQA_kappa, IQA_f1, blur_psnr, blur_ssim, cls_loss, IQA_loss, ves_loss, blur_loss
    def save(self, opt, label):
        self.save_network(self.net, 'joint', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
