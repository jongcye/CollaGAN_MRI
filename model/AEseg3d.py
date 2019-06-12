import tensorflow as tf
import tensorflow.contrib.layers as li
from ipdb import set_trace as st
from model.netUtil import NVDLMED 
from util.util import tf_dice_score, tf_prec_recall

dtype = tf.float32
d_form  = 'channels_first'
d_form_ = 'NCHW'
ch_dim  = 1

eps        = 1e-12
class AEseg3d:
    def __init__(self, opt):
        self.nB    = opt.nB
        self.nCh_in = opt.nCh_in
        self.nCh_out = opt.nCh_out
        self.nCh_seg = opt.nCh_seg
        self.nY    = opt.nY
        self.nX    = opt.nX
        self.lr    = opt.lr
        self.nCh   = opt.ngf
        self.scale = 45.0
        self.lambda_l2 = opt.lambda_l2
        self.lambda_KL = opt.lambda_KL
        self.G = Generator('AEseg3d', opt.G, self.nCh_out, self.nCh_seg, nCh=opt.ngf, w_decay=opt.w_decay)
        # placeholders 
        self.targets = tf.placeholder(dtype, [self.nB, self.nCh_seg, self.nY, self.nX])
        self.inputs  = tf.placeholder(dtype, [self.nB, self.nCh_in, self.nY, self.nX])
#        self.inputs_pre  = tf.placeholder(dtype, [self.nB, self.nCh_in, self.nY, self.nX])
#        self.inputs_post = tf.placeholder(dtype, [self.nB, self.nCh_in, self.nY, self.nX])
        self.inputs_pre  = tf.placeholder(dtype, [self.nB, self.nCh_in*2, self.nY, self.nX])
        self.inputs_post = tf.placeholder(dtype, [self.nB, self.nCh_in*2, self.nY, self.nX])
        tf_inputs = tf.concat([self.inputs_pre, self.inputs, self.inputs_post], axis=ch_dim)
        self.decay_step  = 10000
        self.is_Training = tf.placeholder(tf.bool)

        #targ = tf.slice( self.targets, [0,0,40,24,0],[-1,-1,160,192,-1] )
        #inpt = tf.slice( self.inputs,  [0,0,40,24,0],[-1,-1,160,192,-1] )

        ''' inference G '''
        #self.recon, self.segres = self.G( tf_inputs, self.is_Training)
        self.recon, self.segres, mu, sigma = self.G( tf_inputs, self.is_Training)

        ''' losses from here '''
        ## l2 loss for generator
        self.l2_loss = tf.reduce_mean(tf.squared_difference(self.recon, self.inputs))

        # some constants OH labels define here
        self.WT_dice = tf_dice_score( self.targets[:,0,:,:], self.segres[:,0,:,:])
        #self.TC_dice = tf_dice_score( self.targets[:,1,:,:], self.segres[:,1,:,:])
        #self.EC_dice = tf_dice_score( self.targets[:,2,:,:], self.segres[:,2,:,:])
        #self.dice_loss = (3.-self.WT_dice-self.TC_dice-self.EC_dice)
        self.dice_loss = (1.-self.WT_dice)
        #
        self.precision, self.recall = tf_prec_recall(self.targets[:,0,:,:], self.segres[:,0,:,:])
        # l2 for dice
        self.l2_seg_loss = tf.reduce_mean( tf.squared_difference( self.targets, self.segres) )
        # KL div loss
        self.KL_div_loss = tf.reduce_mean( tf.square(mu)+tf.square(sigma) - tf.log(tf.nn.relu(sigma)+eps)-1 )

        #self.total_loss = self.KL_div_loss + self.dice_loss+self.l2_seg_loss + self.l2_loss*self.lambda_l2 
        self.total_loss = self.lambda_KL*self.KL_div_loss + self.dice_loss + self.l2_loss*self.lambda_l2 

        #
        self.AnB = tf.reduce_mean( self.targets[:,0,:,:]*self.segres[:,0,:,:] )
        self.A = tf.reduce_mean( self.targets[:,0,:,:] )
        self.B = tf.reduce_mean(  self.segres[:,0,:,:] )
 
        # Display
        tf.summary.scalar('0loss/0_KL_div_loss', self.KL_div_loss)
        tf.summary.scalar('0loss/1_L2_loss', self.l2_loss)
        tf.summary.scalar('0loss/2_DICE_loss', self.dice_loss)
        tf.summary.scalar('0loss/3_l2_seg_loss', self.l2_seg_loss)

        tf.summary.scalar('1dice/1WT', self.WT_dice)
        #tf.summary.scalar('1dice/2TC', self.TC_dice)
        #tf.summary.scalar('1dice/3EC', self.EC_dice)
  
        tf.summary.scalar('2dice/1precision', self.precision)
        tf.summary.scalar('2dice/1recall', self.recall)
        # display an image
        tf.summary.image('1inputs/1T1w', self.tf_vis( self.inputs[:,0,tf.newaxis,:,:] ) )
        tf.summary.image('1inputs/2T1contrast', self.tf_vis( self.inputs[:,1,tf.newaxis,:,:] ) )
        tf.summary.image('1inputs/3T2w', self.tf_vis( self.inputs[:,2,tf.newaxis,:,:] ) )
        tf.summary.image('1inputs/4FLAIR', self.tf_vis( self.inputs[:,3,tf.newaxis,:,:] ) )

        tf.summary.image('2recon/1T1w', self.tf_vis( self.recon[:,0,tf.newaxis,:,:] ) )
        tf.summary.image('2recon/2T1contrast', self.tf_vis( self.recon[:,1,tf.newaxis,:,:] ) )
        tf.summary.image('2recon/3T2w', self.tf_vis( self.recon[:,2,tf.newaxis,:,:] ) )
        tf.summary.image('2recon/4FLAIR', self.tf_vis( self.recon[:,3,tf.newaxis,:,:] ) )

        tf.summary.image('3Seg/1WT_target',self.tf_visout( self.targets[:,0,tf.newaxis,:,:]  ))
        tf.summary.image('3Seg/1WT_result',self.tf_visout( self.segres[:,0,tf.newaxis,:,:]  ))
#        tf.summary.image('3Target/2TC',self.tf_visout( self.targets[:,1,tf.newaxis,:,:]  ))
#        tf.summary.image('3Target/3ET',self.tf_visout( self.targets[:,2,tf.newaxis,:,:]  ))
 
#        tf.summary.image('4SegRes/1WT',self.tf_visout( self.segres[:,0,tf.newaxis,:,:]  ))
#        tf.summary.image('4SegRes/2TC',self.tf_visout( self.segres[:,1,tf.newaxis,:,:]  ))
#        tf.summary.image('4SegRes/3ET',self.tf_visout( self.segres[:,2,tf.newaxis,:,:]  ))
        self.optimize(self.total_loss)
        # display an image
        self.summary_op = tf.summary.merge_all()



    def tf_visout(self, inp, order=[0,2,3,1]):
        return tf.transpose(inp,order)

    def tf_vis(self, inp, order=[0,2,3,1]):
        return tf.cast( tf.transpose(inp,order)*self.scale,tf.uint8)

    def tf_vis_abs(self, inp, order=[0,2,3,1]):
        return tf.cast( tf.transpose( tf.abs(inp),order)*self.scale,tf.uint8)

    def optimize(self, total_loss):
        def make_optimizer(loss, variables, lr,  name='Adam'):
            global_step = tf.Variable(0,trainable=False)
            lr_         = tf.train.exponential_decay(lr, global_step, self.decay_step,0.99,staircase=True)
            tf.summary.scalar('learning_rate/{}'.format(name), lr_)
            return tf.train.AdamOptimizer( lr_, beta1=0.5 , name=name).minimize(loss,global_step=global_step,var_list=variables)
        
        self.optm  = make_optimizer(total_loss, self.G.variables, self.lr,   name='Adam')

class Generator:
    def __init__(self,name,G, nCh_out, nCh_seg, nCh=16, w_decay=0):
        if G=='NVDLMED':
            self.net = NVDLMED
        else:
            st()
        self.name = name
        self.nCh  = nCh
        self.nCh_out = nCh_out
        self.nCh_seg = nCh_seg
        self.reuse   = False
        self.w_decay = w_decay 
        self.reg_ = tf.contrib.layers.l2_regularizer(scale=self.w_decay) if self.w_decay>0 else None 
 
    def __call__(self, image, is_Training):
        with tf.variable_scope(self.name, reuse=self.reuse):
            rec, seg, mu, sigma = self.net(image, self.nCh_out, self.nCh_seg, is_Training, self.reg_, nCh=self.nCh)        

        self.reuse=True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return rec, seg, mu, sigma

