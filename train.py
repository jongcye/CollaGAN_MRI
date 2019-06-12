import os
import numpy as np
import tensorflow as tf
from util.util import myNumExt, wpng
import time
from data.BRATS import BRATS as myDB
from ipdb import set_trace as st
from math import ceil
import random
from options.colla_options import BaseOptions
from tqdm import tqdm
import logging 

# device setting
opt = BaseOptions().parse()


# parameter setting
dtype       = tf.float32
eps         = 1e-12
nB          = opt.nB
log_dir     = opt.savepath+'/'+opt.name+'/log_dir/train'
log_dir_v   = opt.savepath+'/'+opt.name+'/log_dir/valid'
ckpt_dir    = opt.savepath+'/'+opt.name+'/ckpt_dir'

# data loading 
DB_train    = myDB(opt,'train')
l_train     = len(DB_train)
DB_valid    = myDB(opt,'valid')
l_valid     = len(DB_valid)
#DB_test     = myDB(opt,'test')
#l_test      = len(DB_test)

# load parameters
opt = DB_train.get_info(opt)
nY  = opt.nY 
nX  = opt.nX
nCh_in      = opt.nCh_in
nCh_out     = opt.nCh_out

from model.CollaGAN import CollaGAN as myModel

nStep_train     = ceil(l_train/nB)
disp_step_train = ceil(nStep_train/opt.disp_div_N)
nStep_valid     = ceil(l_valid/nB)
disp_step_valid = ceil(nStep_valid/opt.disp_div_N)
#nStep_test     = ceil(l_test/nB)

# model initialize
str_ = "/device:GPU:"+str(opt.gpu_ids[0])
print(str_)
with tf.device(str_):
    Colla = myModel(opt)


saver = tf.train.Saver()

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if latest_ckpt==None:
        start_str = "start! initially!"
        tf.global_variables_initializer().run()
        epoch_start=0
    else:
        start_str = ("Start from saved model -"+latest_ckpt)
        saver.restore(sess, latest_ckpt)
        epoch_start=myNumExt(latest_ckpt)+1
    print(start_str)
    
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)   
    valid_writer = tf.summary.FileWriter(log_dir_v, sess.graph) 
    disp_t = 0+epoch_start*disp_step_train
    disp_v = 0+epoch_start*disp_step_valid

    if not opt.test_mode:
        for iEpoch in range(epoch_start, opt.nEpoch+1):
            DB_train.shuffle(seed=iEpoch)   
            print('============================EPOCH # %d # =============' % (iEpoch) )
            s_epoch = time.time()
            out_arg  = [Colla.C_optm, Colla.G_optm, Colla.D_optm, Colla.G_loss, Colla.D_loss]
            out_argm = [Colla.C_optm, Colla.G_optm, Colla.D_optm, Colla.G_loss, Colla.D_loss, Colla.summary_op]
    
            loss_G = 0.
            loss_D = 0.
            cnt=0
            for step in tqdm(range(nStep_train)):
                _tar_class_idx, _a,_b,_c,_d, _am, _bm, _cm, _dm, _tar_class_bools, _tar_img = DB_train.getBatch(step*nB,(step+1)*nB)
                feed_dict = {Colla.is_Training:True, Colla.tar_class_idx:_tar_class_idx,Colla.a_img:_a, Colla.b_img:_b, Colla.c_img:_c, Colla.d_img:_d, Colla.targets:_tar_img,
                        Colla.a_mask:_am, Colla.b_mask:_bm, Colla.c_mask:_cm, Colla.d_mask:_dm,
                        Colla.bool0:_tar_class_bools[0],Colla.bool1:_tar_class_bools[1], Colla.bool2:_tar_class_bools[2],Colla.bool3:_tar_class_bools[3]}
                # train
                if step % disp_step_train == 0:
                    results = sess.run(out_argm, feed_dict=feed_dict)
                    train_writer.add_summary(results[-1],disp_t)
                    disp_t+=1
                    train_writer.flush()
                    loss_G = loss_G + results[-3]
                    loss_D = loss_D + results[-2]
                else:
                    results = sess.run(out_arg, feed_dict = feed_dict)
                    loss_G = loss_G + results[-2]
                    loss_D = loss_D + results[-1]



            #################### VALIDATION loop goes here #######################
            out_arg  = [Colla.G_loss, Colla.D_loss]
            out_argm = [Colla.G_loss, Colla.D_loss, Colla.summary_op]
            
            vloss_G = 0.
            vloss_D = 0.
            
            for step in tqdm(range(nStep_valid)):
                _tar_class_idx, _a,_b,_c,_d, _am, _bm, _cm, _dm, _tar_class_bools, _tar_img = DB_valid.getBatch(step*nB,(step+1)*nB)
                feed_dict = {Colla.is_Training:False, Colla.tar_class_idx:_tar_class_idx,Colla.a_img:_a, Colla.b_img:_b, Colla.c_img:_c, Colla.d_img:_d, Colla.targets:_tar_img,
                        Colla.a_mask:_am, Colla.b_mask:_bm, Colla.c_mask:_cm, Colla.d_mask:_dm,
                        Colla.bool0:_tar_class_bools[0],Colla.bool1:_tar_class_bools[1], Colla.bool2:_tar_class_bools[2],Colla.bool3:_tar_class_bools[3]}
                if step % disp_step_valid == 0:
                    results = sess.run(out_argm, feed_dict = feed_dict)
                    valid_writer.add_summary(results[-1],disp_v)
                    disp_v+=1
                    valid_writer.flush()
                    vloss_G = vloss_G+results[-3]
                    vloss_D = vloss_D+results[-2]
                else:
                    results = sess.run(out_arg, feed_dict = feed_dict)       
                    vloss_G = vloss_G+results[-2]
                    vloss_D = vloss_D+results[-1]
            
            out_train = (' %d epoch -- train loss (G / D) : %.4f /  %.4f' %(iEpoch, loss_G/nStep_train, loss_D/nStep_train))
            out_valid = (' %d epoch -- valid loss (G / D) : %.4f /  %.4f' %(iEpoch, vloss_G/nStep_valid, vloss_D/nStep_valid))
            print(out_train)
            print(out_valid)
    
            if iEpoch %1 ==0:
                path_saved = saver.save(sess, os.path.join(ckpt_dir, "model.ckpt"), global_step=iEpoch)
                logging.info("Model saved in file: %s" % path_saved)


