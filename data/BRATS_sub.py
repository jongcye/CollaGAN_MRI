from os import listdir
from os.path import join, isfile
import random
from scipy import io as sio
import numpy as np
import copy
from ipdb import set_trace as st
from math import ceil

class BRATS_sub():
    def __init__(self,opt,phase):
        super(BRATS_sub, self).__init__()
        random.seed(0)
        self.dataroot = opt.dataroot
        self.root   = join(self.dataroot,phase)
        self.flist  = []
        
        self.root   = self.dataroot

        ## numpy file is a list of filename paths for each slice of brain images.
        ## For fast data loading, the brain images are saved and loaded by slice-by-slice .
        self.flist = np.load(join(self.dataroot, phase+'_flist_main_z.npy'))

        self.N = 4
        self.nZ      = 3  #150
        self.nCh_seg = 1
        self.nCh_in  = self.N
        self.nCh_out = self.N
        self.nY      = 240 
        self.nX      = 240
        self.len     = len(self.flist) 
        self.use_aug = (phase=='train') and opt.AUG
        self.tumor = opt.tumor

    def get_info(self,opt):
        opt.nCh_in = self.nCh_in
        opt.nCh_out= self.nCh_out
        opt.nCh_seg= self.nCh_seg
        opt.nY     = self.nY
        opt.nX     = self.nX
        opt.nZ     = self.nZ 
        return opt

    def getBatch(self, start, end):
        nB = end-start
        end   = min([end,self.len])
        start = end-nB
        batch = self.flist[start:end]
        
        sz_p   = [end-start,self.nCh_in*2, self.nY, self.nX]
        sz_a   = [end-start,  self.nCh_in, self.nY, self.nX]
        sz_M   = [end-start, self.nCh_seg, self.nY, self.nX]

        input_pre = np.empty(sz_p, dtype=np.float32)
        input_img = np.empty(sz_a, dtype=np.float32)
        input_post = np.empty(sz_p,dtype=np.float32)
        target_img = np.empty(sz_M, dtype=np.float32)
        
        
        for iB, aBatch in enumerate(batch):
            i_str = len(aBatch.rstrip("0123456789"))
            z_s = int( aBatch[i_str:] )
            _Batch = aBatch[:i_str] 
            
            input_pre[iB,0,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s-2)+'O.mat'))
            input_pre[iB,1,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s-2)+'C.mat'))
            input_pre[iB,2,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s-2)+'T.mat'))
            input_pre[iB,3,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s-2)+'F.mat'))
            input_pre[iB,4,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s-1)+'O.mat'))
            input_pre[iB,5,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s-1)+'C.mat'))
            input_pre[iB,6,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s-1)+'T.mat'))
            input_pre[iB,7,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s-1)+'F.mat'))
            
            input_img[iB,0,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s)+'O.mat'))
            input_img[iB,1,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s)+'C.mat'))
            input_img[iB,2,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s)+'T.mat'))
            input_img[iB,3,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s)+'F.mat'))
             
            input_post[iB,0,:,:] = self.read_mat(join(self.root, _Batch+str(z_s+1)+'O.mat'))
            input_post[iB,1,:,:] = self.read_mat(join(self.root, _Batch+str(z_s+1)+'C.mat'))
            input_post[iB,2,:,:] = self.read_mat(join(self.root, _Batch+str(z_s+1)+'T.mat'))
            input_post[iB,3,:,:] = self.read_mat(join(self.root, _Batch+str(z_s+1)+'F.mat'))
            input_post[iB,4,:,:] = self.read_mat(join(self.root, _Batch+str(z_s+2)+'O.mat'))
            input_post[iB,5,:,:] = self.read_mat(join(self.root, _Batch+str(z_s+2)+'C.mat'))
            input_post[iB,6,:,:] = self.read_mat(join(self.root, _Batch+str(z_s+2)+'T.mat'))
            input_post[iB,7,:,:] = self.read_mat(join(self.root, _Batch+str(z_s+2)+'F.mat'))
            
            tmp_S = self.read_mat(join(self.root,_Batch+str(z_s)+'S.mat'))
            target_img[iB,:,:,:] = tmp_S[:,:,self.tumor]

        if self.use_aug:
            scale = np.random.uniform(low=0.9, high=1.1)
            input_pre = input_pre*scale
            input_img = input_img*scale
            input_post= input_post*scale
            
            if random.randint(0,1):
                input_pre = np.flip(input_pre,axis=3)
                input_img = np.flip(input_img,axis=3)
                input_post = np.flip(input_post,axis=3)
                target_img = np.flip(target_img,axis=3)
            
        return input_pre, input_img, input_post, target_img
 
    def getBatch_WTTCEC(self, start, end):
        nB = end-start
        end   = min([end,self.len])
        start = end-nB
        batch = self.flist[start:end]
        
        sz_p   = [end-start,self.nCh_in*2, self.nY, self.nX]
        sz_a   = [end-start,  self.nCh_in, self.nY, self.nX]
        sz_M   = [end-start, self.nCh_seg, self.nY, self.nX]

        input_pre = np.empty(sz_p, dtype=np.float32)
        input_img = np.empty(sz_a, dtype=np.float32)
        input_post = np.empty(sz_p,dtype=np.float32)
        target_img = np.empty(sz_M, dtype=np.float32)
        
        
        for iB, aBatch in enumerate(batch):
            i_str = len(aBatch.rstrip("0123456789"))
            z_s = int( aBatch[i_str:] )
            _Batch = aBatch[:i_str] 
            
            input_pre[iB,0,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s-2)+'O.mat'))
            input_pre[iB,1,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s-2)+'C.mat'))
            input_pre[iB,2,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s-2)+'T.mat'))
            input_pre[iB,3,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s-2)+'F.mat'))
            input_pre[iB,4,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s-1)+'O.mat'))
            input_pre[iB,5,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s-1)+'C.mat'))
            input_pre[iB,6,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s-1)+'T.mat'))
            input_pre[iB,7,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s-1)+'F.mat'))
            
            input_img[iB,0,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s)+'O.mat'))
            input_img[iB,1,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s)+'C.mat'))
            input_img[iB,2,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s)+'T.mat'))
            input_img[iB,3,:,:]  = self.read_mat(join(self.root, _Batch+str(z_s)+'F.mat'))
             
            input_post[iB,0,:,:] = self.read_mat(join(self.root, _Batch+str(z_s+1)+'O.mat'))
            input_post[iB,1,:,:] = self.read_mat(join(self.root, _Batch+str(z_s+1)+'C.mat'))
            input_post[iB,2,:,:] = self.read_mat(join(self.root, _Batch+str(z_s+1)+'T.mat'))
            input_post[iB,3,:,:] = self.read_mat(join(self.root, _Batch+str(z_s+1)+'F.mat'))
            input_post[iB,4,:,:] = self.read_mat(join(self.root, _Batch+str(z_s+2)+'O.mat'))
            input_post[iB,5,:,:] = self.read_mat(join(self.root, _Batch+str(z_s+2)+'C.mat'))
            input_post[iB,6,:,:] = self.read_mat(join(self.root, _Batch+str(z_s+2)+'T.mat'))
            input_post[iB,7,:,:] = self.read_mat(join(self.root, _Batch+str(z_s+2)+'F.mat'))
            
            tmp_S = self.read_mat(join(self.root,_Batch+str(z_s)+'S.mat'))
            target_img[iB,:,:,:] = np.transpose(tmp_S,(2,0,1))

        if self.use_aug:
            scale = np.random.uniform(low=0.9, high=1.1)
            input_pre = input_pre*scale
            input_img = input_img*scale
            input_post= input_post*scale
            
            if random.randint(0,1):
                input_pre = np.flip(input_pre,axis=3)
                input_img = np.flip(input_img,axis=3)
                input_post = np.flip(input_post,axis=3)
                target_img = np.flip(target_img,axis=3)
            
        return input_pre, input_img, input_post, target_img
    
    def shuffle(self, seed=0):
        random.seed(seed)
        random.shuffle(self.flist)

    def name(self):
        return 'BRATSseg dataset'

    def __len__(self):
        return self.len
    
    @staticmethod
    def read_mat(filename, var_name="img"):
        mat = sio.loadmat(filename)
        return mat[var_name]

