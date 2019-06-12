import argparse
import os
from util import util
from ipdb import set_trace as st
# for gray scal : input_nc, output_nc, ngf, ndf, gpu_ids, batchSize, norm

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--G', type=str, default='UnetINDiv4_CCAM', help='choice of network for Generator')
        self.parser.add_argument('--dataroot', type=str, default='./../../Hdd_DATA/BRATS2015_mat_std_sbjnorm', help='data root')
        self.parser.add_argument('--savepath', type=str, default='./results', help='savepath')
        self.parser.add_argument('--nEpoch', type=int, default=1000, help='number of Epoch iteration')
        self.parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
        self.parser.add_argument('--lr_D', type=float, default=0.00001, help='learning rate for D')
        self.parser.add_argument('--lr_C', type=float, default=0.00001, help='learning rate for C')
        self.parser.add_argument('--disp_div_N', type=int, default=100, help=' display N per epoch')
        self.parser.add_argument('--nB', type=int, default=1, help='input batch size')
        self.parser.add_argument('--DB_small', action='store_true', help='use small DB')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2.')
        self.parser.add_argument('--name', type=str, default='demo_exp_CollaGAN_BRATS', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--w_decay', type=float, default=0.01, help='weight decay for generator')
        self.parser.add_argument('--w_decay_D', type=float, default=0., help='weight decay for discriminator')
        self.parser.add_argument('--lambda_l1_cyc', type=float, default=10, help='lambda_L1_cyc, StarGAN cyc loss rec')
        self.parser.add_argument('--lambda_l2_cyc', type=float, default=0., help='lambda_L2_cyc, StarGAN cyc loss rec')
        self.parser.add_argument('--lambda_ssim_cyc', type=float, default=1., help='lambda_ssim')
        self.parser.add_argument('--lambda_l2', type=float, default=0., help='lambda_L2')
        self.parser.add_argument('--lambda_l1', type=float, default=0., help='lambda_L1')
        self.parser.add_argument('--lambda_ssim', type=float, default=0., help='lambda_ssim')
        self.parser.add_argument('--lambda_GAN', type=float, default=1., help='lambda GAN')
        self.parser.add_argument('--lambda_G_clsf', type=float, default=1., help='generator classification loss. fake to be well classified')
        self.parser.add_argument('--lambda_D_clsf', type=float, default=1., help='discriminator classification loss. fake to be well classified')
        self.parser.add_argument('--lambda_cyc', type=float, default=1, help='lambda_cyc')
        self.parser.add_argument('--nEpochDclsf', type=int, default=0, help='# of nEpoch for Discriminator pretrain')
        self.parser.add_argument('--nCh_D', type=int, default=4, help='# of ngf for Discriminator')
        self.parser.add_argument('--nCh_C', type=int, default=16, help='# of ngf for Classifier')
        self.parser.add_argument('--use_lsgan', action='store_true', help='use lsgan, if not defualt GAN')
        self.parser.add_argument('--use_1x1Conv', action='store_true', help='use 1x1Conv, if not defualt 3x3conv')
        self.parser.add_argument('--wo_norm_std', action='store_true', help='NOT use std normalization')
        self.parser.add_argument('--N_null', type=int, default=1, help='# of nulling in input images')
        self.parser.add_argument('--ngf', type=int, default=64, help=' ngf')
        self.parser.add_argument('--dropout', type=float, default=0.5, help='droptout ')
        self.parser.add_argument('--test_mode', action='store_true', help='not train. just test')
        self.parser.add_argument('--AUG', action='store_true', help='use augmentation')
        self.parser.add_argument('--nEpochD', type=int, default=2, help = 'nEpochD update while 1 G update')


        self.initialized = True
    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        #self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        #if len(self.opt.gpu_ids) > 0:
        #    torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.savepath, self.opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
    @staticmethod
    def load_opts(opt,exp_name):
        #optLists = ['model','dataroot','savepath','nEpoch','lr','disp_div_N','batchSize','input_nc','gpu_ids','name','use_residual','no_flip','lambda_cost','weight_decay','use_dropout','optimizer','ri','normalize']
        exp_dir = os.path.join(opt.savepath,exp_name)
        with open(os.path.join(exp_dir,'opt.txt'),'r') as opt_file:
            for aLine in opt_file.readlines():
                idx = aLine.find(':')
                if idx==-1:
                    continue
                else:
                    cur_opt = aLine[:idx]
                    cur_val = aLine[idx+2:-1]
                    if cur_opt=='model':
                        opt.model      = cur_val
                    elif cur_opt=='dataroot':
                        opt.dataroot   = cur_val
                    elif cur_opt=='savepath':
                        opt.savepath   = cur_val
                    elif cur_opt=='nEpoch':
                        opt.savepath   = cur_val
                    elif cur_opt=='lr':
                        opt.lr         = float(cur_val)
                    elif cur_opt=='disp_div_N':
                        opt.disp_div_N = int(cur_val)
                    elif cur_opt=='batchSize':
                        opt.batchSize  = int(cur_val)
                    elif cur_opt=='input_nc':
                        opt.input_nc   = int(cur_val)
                    elif cur_opt=='gpu_ids':
                        cur_val = cur_val[1:-1]
                        opt.gpu_ids    = [int(cur_val)]
                        print('Use GPU id......')
                    elif cur_opt=='name':
                        opt.name       = cur_val
                    elif cur_opt=='use_residual':
                        opt.use_residual= (cur_val=='True')
                    elif cur_opt=='no_flip':
                        opt.use_residual= (cur_val=='True')
                    elif cur_opt=='lambda_cost':
                        opt.lambda_cost = float(cur_val)
                    elif cur_opt=='weight_decay':
                        opt.weight_decay= float(cur_val)
                    elif cur_opt=='use_dropout':
                        opt.use_dropout = (cur_val=='True')
                    elif cur_opt=='optimizer':
                        opt.optimizer= cur_val
                    elif cur_opt=='ri':
                        opt.ri = (cur_val=='True')
                    elif cur_opt=='normalize':
                        opt.normalize = (cur_val=='True')
                    else:
                        st()
        return opt





