import os
import numpy as np
import tensorflow as tf
from util.util import myNumExt, wpng_bw
import time
from data.BRATS_sub import BRATS_sub as myDB
from ipdb import set_trace as st
from math import ceil
import random
from options.seg_options import BaseOptions
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

######################
DB_train    = myDB(opt,'train')
l_train     = len(DB_train)
DB_valid    = myDB(opt,'valid')
l_valid     = len(DB_valid)
#DB_test     = myDB(opt,'test')
#l_test      = len(DB_test)

opt = DB_train.get_info(opt)
nY  = opt.nY 
nX  = opt.nX
nCh_in      = opt.nCh_in
nCh_out     = opt.nCh_out


from model.AEseg3d import AEseg3d as myModel

nStep_train     = ceil(l_train/nB)
disp_step_train = ceil(nStep_train/opt.disp_div_N)
nStep_valid     = ceil(l_valid/nB)
disp_step_valid = ceil(nStep_valid/opt.disp_div_N)
#nStep_test     = ceil(l_test/nB)

## model initialize
str_ = "/device:GPU:"+str(opt.gpu_ids[0])
print(str_)
with tf.device(str_):
    AEseg = myModel(opt)

saver = tf.train.Saver()

##
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if latest_ckpt==None:
        print("start! initially!")
        tf.global_variables_initializer().run()
        epoch_start=0
    else:
        print("Start from saved model -"+latest_ckpt)
        saver.restore(sess, latest_ckpt)
        epoch_start=myNumExt(latest_ckpt)+1

    train_writer = tf.summary.FileWriter(log_dir, sess.graph)   
    valid_writer = tf.summary.FileWriter(log_dir_v, sess.graph) 
    disp_t = 0+epoch_start*opt.disp_div_N
    disp_v = 0+epoch_start*opt.disp_div_N

    if not opt.test_mode:

        for iEpoch in range(epoch_start, opt.nEpoch+1):
            DB_train.shuffle(seed=iEpoch)   
            print('============================EPOCH # %d # =============' % (iEpoch) )
            s_epoch = time.time()
                
            out_arg  = [AEseg.optm, AEseg.total_loss, AEseg.WT_dice]
            out_argm = [AEseg.optm, AEseg.total_loss, AEseg.WT_dice, AEseg.summary_op]
            
            loss_G = 0.
            WT_dice = 0.
            cnt=0
            for step in tqdm(range(nStep_train)):
                _input_pre, _input_img, _input_post, _target = DB_train.getBatch(step*nB,(step+1)*nB)
                if np.sum(_input_pre)==0:
                    continue
                
                feed_dict = {AEseg.is_Training:True, AEseg.inputs_pre:_input_pre, AEseg.inputs:_input_img, AEseg.inputs_post:_input_post, AEseg.targets:_target}
                # train
                if step % disp_step_train == 0:
                    results = sess.run(out_argm, feed_dict=feed_dict)
                    train_writer.add_summary(results[-1],disp_t)
                    disp_t+=1
                    train_writer.flush()
                else:
                    results = sess.run(out_arg, feed_dict = feed_dict)
                cnt=cnt+1
                loss_G = loss_G + results[1] 
                WT_dice= WT_dice+results[2] 
            #################### VALIDATION loop
            out_arg  = [AEseg.total_loss, AEseg.WT_dice]
            out_argm = [AEseg.total_loss, AEseg.WT_dice, AEseg.summary_op]
           
            vcnt=0
            vloss_G = 0.
            vWT_dice= 0. 
            for step in tqdm(range(nStep_valid)):
                _input_pre, _input_img, _input_post, _target = DB_valid.getBatch(step*nB,(step+1)*nB)
                if np.sum(_input_pre)==0:
                    continue
                feed_dict = {AEseg.is_Training:False, AEseg.inputs_pre:_input_pre, AEseg.inputs:_input_img, AEseg.inputs_post:_input_post, AEseg.targets:_target}   
                if step % disp_step_valid == 0:
                    results = sess.run(out_argm, feed_dict = feed_dict)
                    valid_writer.add_summary(results[-1],disp_v)
                    disp_v+=1
                    valid_writer.flush()
                else:
                    results = sess.run(out_arg, feed_dict = feed_dict)       
                vcnt=vcnt+1
                vloss_G = vloss_G+results[0]
                vWT_dice= vWT_dice+results[1]
            print(' %d epoch -- train loss (WT_dice) : %.4f(%.4f) ' %(iEpoch, loss_G/cnt, WT_dice/cnt))
            print(' %d epoch -- valid loss (WT_dice) : %.4f(%.4f) ' %(iEpoch, vloss_G/vcnt, vWT_dice/vcnt))
    
            if iEpoch %1 ==0:
                path_saved = saver.save(sess, os.path.join(ckpt_dir, "model.ckpt"), global_step=iEpoch)
                logging.info("Model saved in file: %s" % path_saved)


