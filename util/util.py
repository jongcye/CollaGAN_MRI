import tensorflow as tf
import numpy as np
from ipdb import set_trace as st
import cv2 
ch_dim=1
dtype = tf.float32
eps = 1e-12

def tf_prec_recall(p_true, p_pred):
    intersect = tf.reduce_mean( p_true*p_pred )
    precision = intersect / ( tf.reduce_mean( p_true ) + eps )
    recall    = intersect / ( tf.reduce_mean( p_pred ) + eps )
    return precision, recall

def tf_dice_score(p_true,p_pred):
    intersect = tf.reduce_mean(p_true*p_pred, axis=[0,1,2])
    union = eps + tf.reduce_mean(p_pred, axis=[0,1,2]) + tf.reduce_mean(p_true,axis=[0,1,2])
    dice = (2.*intersect+eps)/union
    return dice

def wpng(fname, img, nY=240, nX=240):
    img = np.clip(img,0,255)
    img = np.concatenate([img[:,:,2,np.newaxis],img[:,:,1,np.newaxis],img[:,:,0,np.newaxis]],axis=2)
    cv2.imwrite(fname,img)

def wpng_bw(fname, img, nY=240, nX=240):
    img = np.clip(img,0,255)
    cv2.imwrite(fname,img)

def myNumExt(s):
    head = s.rstrip('0123456789')
    tail = s[len(head):]
    return int(tail)

