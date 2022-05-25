from os.path import join, exists
from torch.autograd import Variable
from models.base_model import BaseModel
from . import networks
from models.losses import init_loss

from util.metrics import *

np.seterr(divide='ignore',invalid='ignore')
class model_CLS(BaseModel):
    def name(self):
        return 'model_CLS'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.input_img = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.input_img_mask = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
        self.label = self.Tensor(opt.batchSize)

        if exists(join(opt.project_root, opt.CLS_weight_root)):
            opt.CLS_weight_root = opt.CLS_weight_root
        else:
            opt.CLS_weight_root = 'none'
        self.net = networks.define_net(opt.model_choice, opt.CLS_weight_root, opt.norm, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            if opt.model_choice == 'EyesImage_CLS_with_resnet':
                self.load_network(self.net, 'CLS_res', opt.which_epoch)
            elif opt.model_choice == 'EyesImage_CLS_with_MyNet':
                self.load_network(self.net, 'CLS_MyNet', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.optimizer_G = torch.optim.Adam(self.net.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.clsLoss = init_loss(opt, self.Tensor)

        print('---------- Networks initialized -------------')
        networks.print_network(self.net)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_gt_img = input['gt_img']
        input_region = input['region']
        label = input['labels_cls']
        self.label = self.Tensor(label.shape)
        self.label.copy_(label)
        self.input_img.resize_(input_gt_img.size()).copy_(input_gt_img)
        self.input_img_mask.resize_(input_region.size()).copy_(input_region)

    def forward(self, opt):
        self.in_gt_img = Variable(self.input_img)
        self.input_region = Variable(self.input_img_mask)
        self.in_labels = Variable(self.label)
        self.in_labels2 = self.in_labels.long()
        self.label_pre = self.net.forward(self.in_gt_img)

    def backward_net(self, opt, epoch):
        self.loss_labels_pre = self.clsLoss.get_loss(self.label_pre, self.in_labels2)
        out = self.label_pre
        label = self.in_labels2
        self.acc = calculate_accuracy(out, label)
        self.kappa = compute_kappa(out, label)
        self.f1 = compute_f1(out, label)
        # print('epoch = %d  loss_net: %f  acc: %f' % (epoch, self.loss_labels_pre, self.acc))

        self.loss_labels_pre.backward()

        return self.acc, self.kappa, self.f1, self.loss_labels_pre

    def optimize_parameters(self, opt, epoch):
        self.forward(opt)
        self.optimizer_G.zero_grad()
        acc, kappa, f1, loss = self.backward_net(opt, epoch)
        self.optimizer_G.step()

        return acc, kappa, f1, loss

    def save(self, opt, label):
        if opt.model_choice == 'EyesImage_CLS_with_resnet':
            self.save_network(self.net, 'CLS_res', label, self.gpu_ids)
        elif opt.model_choice == 'EyesImage_CLS_with_MyNet':
            self.save_network(self.net, 'CLS_MyNet', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
