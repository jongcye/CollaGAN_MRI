import argparse
import os
from util import util
from ipdb import set_trace as st

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--G', type=str, default='NVDLMED', help='choice of network')
        self.parser.add_argument('--dataroot', type=str, default='/Hdd_2/BRATS_colla/BRATS2015_mat_std_sbjnorm_D', help='data root')
        self.parser.add_argument('--savepath', type=str, default='./seg_results', help='savepath')
        self.parser.add_argument('--nEpoch', type=int, default=1000, help='number of Epoch iteration')
        self.parser.add_argument('--lr', type=float, default=0.000001, help='learning rate')
        self.parser.add_argument('--disp_div_N', type=int, default=10, help=' display N per epoch')
        self.parser.add_argument('--nB', type=int, default=1, help='input batch size')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2.')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--w_decay', type=float, default=0.00001, help='weight decay')
        self.parser.add_argument('--lambda_l2', type=float, default=0.1, help='lambda_L2')
        self.parser.add_argument('--lambda_KL', type=float, default=0.1, help='lambda_L2')
        self.parser.add_argument('--ngf', type=int, default=64, help=' ngf')
        self.parser.add_argument('--dropout', type=float, default=0.2, help='droptout ')
        self.parser.add_argument('--test_mode', action='store_true', help='not train. just test')
        self.parser.add_argument('--AUG', action='store_true', help='use augmentation')
        self.parser.add_argument('--lambda_WT', type=float, default=1.0, help='lambda_WT')
        self.parser.add_argument('--lambda_TC', type=float, default=1.0, help='lambda_TC')
        self.parser.add_argument('--lambda_EC', type=float, default=1.0, help='lambda_EC')
        self.parser.add_argument('--lambda_precision', type=float, default=0.0, help='lambda_precision')
        self.parser.add_argument('--lambda_recall', type=float, default=0.0, help='lambda_recall')
        self.parser.add_argument('--tumor', type=int, default=0, help='0:WT, 1:TC, 2:EC')


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





