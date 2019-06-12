import tensorflow as tf
import tensorflow.contrib.layers as li
from ipdb import set_trace as st
from model.netUtil import  Conv2d, Conv2d2x2, lReLU, BN, Conv1x1, UnetINDiv4_CCAM

dtype = tf.float32
d_form  = 'channels_first'
d_form_ = 'NCHW'
ch_dim  = 1

REAL_LABEL = 1.0
eps        = 1e-12
class CollaGAN:
    def __init__(self, opt):
        self.nB    = opt.nB
        self.nCh_in = opt.nCh_in
        self.nCh_out = opt.nCh_out
        self.nY    = opt.nY
        self.nX    = opt.nX
        self.lr    = opt.lr
        self.lr_D  = opt.lr_D
        self.lr_C  = opt.lr_C
        self.nCh   = opt.ngf
        self.nCh_D = opt.nCh_D
        self.nCh_C = opt.nCh_C
        self.use_lsgan = opt.use_lsgan
        self.class_N = 4
        self.lambda_l1_cyc = opt.lambda_l1_cyc
        self.lambda_l2_cyc = opt.lambda_l2_cyc
        self.lambda_l1 = opt.lambda_l1
        self.lambda_l2 = opt.lambda_l2
        self.lambda_GAN = opt.lambda_GAN
        self.lambda_G_clsf = opt.lambda_G_clsf
        self.lambda_D_clsf = opt.lambda_D_clsf
        self.lambda_ssim = opt.lambda_ssim
        self.lambda_ssim_cyc = opt.lambda_ssim_cyc
        self.scale = 255.0 # if opt.wo_norm_std else 25.0 % temporaly for display

        self.G = Generator('G', opt.G, self.nCh_out,nCh=opt.ngf,use_1x1Conv=opt.use_1x1Conv, w_decay=opt.w_decay)
        self.D = Discriminator('D', nCh=self.nCh_D, w_decay_D=opt.w_decay_D,class_N=self.class_N, DR_ratio=opt.dropout)

        # placeholders 
        self.targets = tf.placeholder(dtype, [self.nB, self.nCh_out, self.nY, self.nX])
        self.tar_class_idx = tf.placeholder(tf.uint8)
        self.is_Training= tf.placeholder(tf.bool)
        
        self.a_img = tf.placeholder(dtype,[self.nB,self.nCh_out, self.nY, self.nX])
        self.b_img = tf.placeholder(dtype,[self.nB,self.nCh_out, self.nY, self.nX])      
        self.c_img = tf.placeholder(dtype,[self.nB,self.nCh_out, self.nY, self.nX])
        self.d_img = tf.placeholder(dtype,[self.nB,self.nCh_out, self.nY, self.nX])      

        self.a_mask = tf.placeholder(dtype,[self.nB,1, self.nY, self.nX])
        self.b_mask = tf.placeholder(dtype,[self.nB,1, self.nY, self.nX])
        self.c_mask = tf.placeholder(dtype,[self.nB,1, self.nY, self.nX])
        self.d_mask = tf.placeholder(dtype,[self.nB,1, self.nY, self.nX])


        self.bool0 = tf.placeholder(tf.bool)
        self.bool1 = tf.placeholder(tf.bool)
        self.bool2 = tf.placeholder(tf.bool)
        self.bool3 = tf.placeholder(tf.bool)
        
        ''' generate inputs ( imag + mask ) '''
        tmp_zeros = tf.zeros([self.nB,self.nCh_out,self.nY,self.nX],dtype)
        inp1 = tf.cond(self.bool0, lambda:tmp_zeros, lambda:self.a_img)
        inp2 = tf.cond(self.bool1, lambda:tmp_zeros, lambda:self.b_img)
        inp3 = tf.cond(self.bool2, lambda:tmp_zeros, lambda:self.c_img)
        inp4 = tf.cond(self.bool3, lambda:tmp_zeros, lambda:self.d_img)

        input_contrasts = tf.concat([inp1,inp2,inp3,inp4],axis=ch_dim) 
        self.inputs = tf.concat([input_contrasts, self.a_mask, self.b_mask,self.c_mask,self.d_mask],axis=ch_dim)

        ''' inference G, D for 1st input (not cyc) '''
        self.recon = self.G(self.inputs,self.is_Training)

        ## D(recon)
        RealFake_rec,type_rec = self.D(self.recon, self.is_Training)
        ## D(target)
        RealFake_tar,type_tar = self.D(self.targets, self.is_Training)

        ''' generate inputs for cyc '''
        # for cyc
        cyc1_ = tf.cond(self.bool0, lambda:self.recon, lambda:self.a_img)
        cyc2_ = tf.cond(self.bool1, lambda:self.recon, lambda:self.b_img)
        cyc3_ = tf.cond(self.bool2, lambda:self.recon, lambda:self.c_img)
        cyc4_ = tf.cond(self.bool3, lambda:self.recon, lambda:self.d_img)

        cyc_inp1_ = tf.concat([tmp_zeros,cyc2_,cyc3_,cyc4_],axis=ch_dim)
        cyc_inp2_ = tf.concat([cyc1_,tmp_zeros,cyc3_,cyc4_],axis=ch_dim)
        cyc_inp3_ = tf.concat([cyc1_,cyc2_,tmp_zeros,cyc4_],axis=ch_dim)
        cyc_inp4_ = tf.concat([cyc1_,cyc2_,cyc3_,tmp_zeros],axis=ch_dim)

        atmp_zeros = tf.zeros([self.nB,1,self.nY,self.nX],dtype)
        atmp_ones  = tf.ones([self.nB,1,self.nY,self.nX],dtype)
        cyc_inp1 = tf.concat([cyc_inp1_,atmp_ones,atmp_zeros,atmp_zeros,atmp_zeros],axis=ch_dim)
        cyc_inp2 = tf.concat([cyc_inp2_,atmp_zeros,atmp_ones,atmp_zeros,atmp_zeros],axis=ch_dim)
        cyc_inp3 = tf.concat([cyc_inp3_,atmp_zeros,atmp_zeros,atmp_ones,atmp_zeros],axis=ch_dim)
        cyc_inp4 = tf.concat([cyc_inp4_,atmp_zeros,atmp_zeros,atmp_zeros,atmp_ones],axis=ch_dim)

        ''' inference G, D for cyc inputs'''
        self.cyc1 = self.G(cyc_inp1, self.is_Training)
        self.cyc2 = self.G(cyc_inp2, self.is_Training)
        self.cyc3 = self.G(cyc_inp3, self.is_Training)
        self.cyc4 = self.G(cyc_inp4, self.is_Training)

        ## D(cyc), C(cyc)
        RealFake_cyc1,type_cyc1 = self.D(self.cyc1, self.is_Training)
        RealFake_cyc2,type_cyc2 = self.D(self.cyc2, self.is_Training)
        RealFake_cyc3,type_cyc3 = self.D(self.cyc3, self.is_Training)
        RealFake_cyc4,type_cyc4 = self.D(self.cyc4, self.is_Training)
        
        ## D(tar), C(tar)
        RealFake_tar1, type_tar1 = self.D(self.a_img, self.is_Training)
        RealFake_tar2, type_tar2 = self.D(self.b_img, self.is_Training)
        RealFake_tar3, type_tar3 = self.D(self.c_img, self.is_Training)
        RealFake_tar4, type_tar4 = self.D(self.d_img, self.is_Training)

        ''' Here, loss def starts here'''
        # gen loss for generator
        G_gan_loss_cyc1   = tf.reduce_mean(tf.squared_difference(RealFake_cyc1, REAL_LABEL))
        G_gan_loss_cyc2   = tf.reduce_mean(tf.squared_difference(RealFake_cyc2, REAL_LABEL))
        G_gan_loss_cyc3   = tf.reduce_mean(tf.squared_difference(RealFake_cyc3, REAL_LABEL))
        G_gan_loss_cyc4   = tf.reduce_mean(tf.squared_difference(RealFake_cyc4, REAL_LABEL))

        G_gan_loss_cyc  = G_gan_loss_cyc1 + G_gan_loss_cyc2 + G_gan_loss_cyc3 + G_gan_loss_cyc4 
        G_gan_loss_orig   = tf.reduce_mean(tf.squared_difference(RealFake_rec, REAL_LABEL))
        G_gan_loss = (G_gan_loss_orig + G_gan_loss_cyc)/5.

        ## l2 loss for generator
        cyc_l2_loss1 = tf.reduce_mean(tf.squared_difference(self.cyc1, self.a_img))
        cyc_l2_loss2 = tf.reduce_mean(tf.squared_difference(self.cyc2, self.b_img))
        cyc_l2_loss3 = tf.reduce_mean(tf.squared_difference(self.cyc3, self.c_img))
        cyc_l2_loss4 = tf.reduce_mean(tf.squared_difference(self.cyc4, self.d_img))
        l2_cyc_loss = cyc_l2_loss1 + cyc_l2_loss2 + cyc_l2_loss3 + cyc_l2_loss4 

        l2_loss_orig = tf.reduce_mean(tf.squared_difference(self.recon,self.targets))
        l2_loss = (l2_loss_orig+l2_cyc_loss)/5.


        ## l1 loss for generator
        cyc_l1_loss1 = tf.reduce_mean(tf.losses.absolute_difference(self.cyc1, self.a_img))
        cyc_l1_loss2 = tf.reduce_mean(tf.losses.absolute_difference(self.cyc2, self.b_img))
        cyc_l1_loss3 = tf.reduce_mean(tf.losses.absolute_difference(self.cyc3, self.c_img))
        cyc_l1_loss4 = tf.reduce_mean(tf.losses.absolute_difference(self.cyc4, self.d_img))
        # for cyc
        cyc_l1_loss1 = tf.cond(self.bool0, lambda:0., lambda:cyc_l1_loss1)
        cyc_l1_loss2 = tf.cond(self.bool1, lambda:0., lambda:cyc_l1_loss2)
        cyc_l1_loss3 = tf.cond(self.bool2, lambda:0., lambda:cyc_l1_loss3)
        cyc_l1_loss4 = tf.cond(self.bool3, lambda:0., lambda:cyc_l1_loss4)
      
        l1_cyc_loss = cyc_l1_loss1 + cyc_l1_loss2 + cyc_l1_loss3 + cyc_l1_loss4 

        l1_loss_orig = tf.reduce_mean(tf.losses.absolute_difference(self.recon,self.targets))
        l1_loss = (l1_loss_orig+l1_cyc_loss)/5.

        ## SSIM loss for generator
        ssim1= tf.image.ssim(self.cyc1[0,0,:,:,tf.newaxis], self.a_img[0,0,:,:,tf.newaxis], 5)
        ssim2= tf.image.ssim(self.cyc2[0,0,:,:,tf.newaxis], self.b_img[0,0,:,:,tf.newaxis], 5)
        ssim3= tf.image.ssim(self.cyc3[0,0,:,:,tf.newaxis], self.c_img[0,0,:,:,tf.newaxis], 5)
        ssim4= tf.image.ssim(self.cyc4[0,0,:,:,tf.newaxis], self.d_img[0,0,:,:,tf.newaxis], 5)

        ssimr= tf.image.ssim(self.recon[0,0,:,:,tf.newaxis], self.targets[0,0,:,:,tf.newaxis], 5)
 
        cyc_ssim_loss1 = -tf.log( (1.0+ssim1)/2.0)
        cyc_ssim_loss2 = -tf.log( (1.0+ssim2)/2.0)        
        cyc_ssim_loss3 = -tf.log( (1.0+ssim3)/2.0)  
        cyc_ssim_loss4 = -tf.log( (1.0+ssim4)/2.0)  

        cyc_ssim_loss1 = tf.cond(self.bool0, lambda:0., lambda: -tf.log( (1.0+ssim1)/2.0))
        cyc_ssim_loss2 = tf.cond(self.bool1, lambda:0., lambda: -tf.log( (1.0+ssim2)/2.0))
        cyc_ssim_loss3 = tf.cond(self.bool2, lambda:0., lambda: -tf.log( (1.0+ssim3)/2.0))
        cyc_ssim_loss4 = tf.cond(self.bool3, lambda:0., lambda: -tf.log( (1.0+ssim4)/2.0))
        ssim_cyc_loss  = cyc_ssim_loss1 + cyc_ssim_loss2 + cyc_ssim_loss3 + cyc_ssim_loss4
#
        ssim_loss_orig = -tf.log( (1.0+ssimr)/2.0)   
        ssim_loss = (ssim_loss_orig + ssim_cyc_loss)/4.

        # some constants OH labels define here
         
        OH_label1 = tf.tile(tf.reshape(tf.one_hot(tf.cast(0,tf.uint8),self.class_N),[-1,1,self.class_N]),[self.nB,1,1])
        OH_label2 = tf.tile(tf.reshape(tf.one_hot(tf.cast(1,tf.uint8),self.class_N),[-1,1,self.class_N]),[self.nB,1,1])
        OH_label3 = tf.tile(tf.reshape(tf.one_hot(tf.cast(2,tf.uint8),self.class_N),[-1,1,self.class_N]),[self.nB,1,1])
        OH_label4 = tf.tile(tf.reshape(tf.one_hot(tf.cast(3,tf.uint8),self.class_N),[-1,1,self.class_N]),[self.nB,1,1])
        OH_labelT = tf.tile(tf.reshape(tf.one_hot(tf.cast(self.tar_class_idx,tf.uint8),self.class_N),[-1,1,self.class_N]),[1,1,1])
         
        '''classification loss for generator'''
        G_clsf_cyc_loss1 = tf.losses.softmax_cross_entropy(OH_label1, type_cyc1)
        G_clsf_cyc_loss2 = tf.losses.softmax_cross_entropy(OH_label2, type_cyc2)
        G_clsf_cyc_loss3 = tf.losses.softmax_cross_entropy(OH_label3, type_cyc3)
        G_clsf_cyc_loss4 = tf.losses.softmax_cross_entropy(OH_label4, type_cyc4)

        G_clsf_cyc_loss  = G_clsf_cyc_loss1 + G_clsf_cyc_loss2 + G_clsf_cyc_loss3 + G_clsf_cyc_loss4 
        G_clsf_orig_loss = tf.losses.softmax_cross_entropy(OH_labelT, type_rec)
        G_clsf_loss = (G_clsf_orig_loss + G_clsf_cyc_loss)/5.
        
        ''' total generator loss '''
        self.G_loss       = (self.lambda_GAN*(G_gan_loss_orig + G_gan_loss_cyc) +
                 self.lambda_l1_cyc* ( l1_cyc_loss  ) + self.lambda_l1* (l1_loss_orig) +
                 self.lambda_l2_cyc* ( l2_cyc_loss  ) + self.lambda_l2* (l2_loss_orig) +
                 self.lambda_G_clsf* (G_clsf_orig_loss + G_clsf_cyc_loss) +
                 self.lambda_ssim_cyc* (ssim_cyc_loss)# +
                 )

        # discriminator loss
        C_loss1 = tf.losses.softmax_cross_entropy(OH_label1,type_tar1)
        C_loss2 = tf.losses.softmax_cross_entropy(OH_label2,type_tar2)
        C_loss3 = tf.losses.softmax_cross_entropy(OH_label3,type_tar3)
        C_loss4 = tf.losses.softmax_cross_entropy(OH_label4,type_tar4)

        self.C_loss       = C_loss1 + C_loss2 + C_loss3 + C_loss4

        
        if self.use_lsgan:
            err_real = tf.reduce_mean(tf.squared_difference(RealFake_tar, REAL_LABEL))
            err_fake = tf.reduce_mean(tf.square(RealFake_rec))
            D_err = err_real + err_fake

            cyc_real1 = tf.reduce_mean(tf.squared_difference(RealFake_tar1, REAL_LABEL))
            cyc_fake1 = tf.reduce_mean(tf.square(RealFake_cyc1))
            cyc_err1 = cyc_real1 + cyc_fake1 
            cyc_real2 = tf.reduce_mean(tf.squared_difference(RealFake_tar2, REAL_LABEL))
            cyc_fake2 = tf.reduce_mean(tf.square(RealFake_cyc2))
            cyc_err2 = cyc_real2 + cyc_fake2 
            cyc_real3 = tf.reduce_mean(tf.squared_difference(RealFake_tar3, REAL_LABEL))
            cyc_fake3 = tf.reduce_mean(tf.square(RealFake_cyc3))
            cyc_err3 = cyc_real3 + cyc_fake3 
            cyc_real4 = tf.reduce_mean(tf.squared_difference(RealFake_tar4, REAL_LABEL))
            cyc_fake4 = tf.reduce_mean(tf.square(RealFake_cyc4))
            cyc_err4 = cyc_real4 + cyc_fake4 
        else:
            st()
            err_real = -tf.reduce_mean(tf.log(RealFake_tar+eps))
            err_fake = -tf.reduce_mean(tf.log(1-RealFake_rec+eps))
        D_gan_cyc  = cyc_err1 + cyc_err2 + cyc_err3 + cyc_err4 
        D_gan_loss  = (D_err + D_gan_cyc)/5.
        ##
        self.D_loss = (D_err + D_gan_cyc)/5. + (self.C_loss)/4.
        
        # Display
        tf.summary.scalar('0loss/G:ganfake(0.25-0) + l1(-->0) +clsf(--0))', self.G_loss)
        tf.summary.scalar('0loss/D:realfake(0.5)+clsf(1.386-->0)', self.D_loss)

        tf.summary.scalar('1G/G_gan', G_gan_loss)
        tf.summary.scalar('1G/L2', l2_loss)
        tf.summary.scalar('1G/L1', l1_loss)
        tf.summary.scalar('1G/SSIM', ssim_loss)
        tf.summary.scalar('1G/clsf', G_clsf_loss)
 
        tf.summary.scalar('2D/D_gan_loss:REAL/FAKE(0.5)', D_gan_loss)
        tf.summary.scalar('2D/C_loss(REAL)--1.386-->0', self.C_loss)

        tf.summary.scalar('G_gan(0.25-0)/rec ', G_gan_loss_orig)
        tf.summary.scalar('G_gan(0.25-0)/cyc1', G_gan_loss_cyc1)
        tf.summary.scalar('G_gan(0.25-0)/cyc2', G_gan_loss_cyc2)
        tf.summary.scalar('G_gan(0.25-0)/cyc3', G_gan_loss_cyc3)
        tf.summary.scalar('G_gan(0.25-0)/cyc4', G_gan_loss_cyc4)

        tf.summary.scalar('G_l2/rec ', l2_loss_orig)
        tf.summary.scalar('G_l2/cyc1', cyc_l2_loss1)
        tf.summary.scalar('G_l2/cyc2', cyc_l2_loss2)
        tf.summary.scalar('G_l2/cyc3', cyc_l2_loss3)
        tf.summary.scalar('G_l2/cyc4', cyc_l2_loss4)

        tf.summary.scalar('G_l1/rec ', l1_loss_orig)
        tf.summary.scalar('G_l1/cyc1', cyc_l1_loss1)
        tf.summary.scalar('G_l1/cyc2', cyc_l1_loss2)
        tf.summary.scalar('G_l1/cyc3', cyc_l1_loss3)
        tf.summary.scalar('G_l1/cyc4', cyc_l1_loss4)

        tf.summary.scalar('G_ssim/rec ', ssimr)
        tf.summary.scalar('G_ssim/cyc1', ssim1)
        tf.summary.scalar('G_ssim/cyc2', ssim2)
        tf.summary.scalar('G_ssim/cyc3', ssim3)
        tf.summary.scalar('G_ssim/cyc4', ssim4)


        tf.summary.scalar('G_clsf/rec_', G_clsf_orig_loss)      
        tf.summary.scalar('G_clsf/cyc_rec_a', G_clsf_cyc_loss1)      
        tf.summary.scalar('G_clsf/cyc_rec_b', G_clsf_cyc_loss2)     
        tf.summary.scalar('G_clsf/cyc_rec_c', G_clsf_cyc_loss3)      
        tf.summary.scalar('G_clsf/cyc_rec_d', G_clsf_cyc_loss4)     

        tf.summary.scalar('D_gan_loss(bestForD:1-0.5:bestForG)/Rec_err', D_err)
        tf.summary.scalar('D_gan_loss(bestForD:1-0.5:bestForG)/cyc1_err', cyc_err1)
        tf.summary.scalar('D_gan_loss(bestForD:1-0.5:bestForG)/cyc2_err', cyc_err2)
        tf.summary.scalar('D_gan_loss(bestForD:1-0.5:bestForG)/cyc3_err', cyc_err3)
        tf.summary.scalar('D_gan_loss(bestForD:1-0.5:bestForG)/cyc4_err', cyc_err4)

        tf.summary.scalar('D_gan_loss_detail(bestForD:0-0.25:bestForG)/Rec_err_real', err_real)
        tf.summary.scalar('D_gan_loss_detail(bestForD:0-0.25:bestForG)/Rec_err_fake', err_fake)
        tf.summary.scalar('D_gan_loss_detail(bestForD:0-0.25:bestForG)/cyc1_err_real', cyc_real1)
        tf.summary.scalar('D_gan_loss_detail(bestForD:0-0.25:bestForG)/cyc1_err_fake', cyc_fake1)
        tf.summary.scalar('D_gan_loss_detail(bestForD:0-0.25:bestForG)/cyc2_err_real', cyc_real2)
        tf.summary.scalar('D_gan_loss_detail(bestForD:0-0.25:bestForG)/cyc2_err_fake', cyc_fake2)
        tf.summary.scalar('D_gan_loss_detail(bestForD:0-0.25:bestForG)/cyc3_err_real', cyc_real3)
        tf.summary.scalar('D_gan_loss_detail(bestForD:0-0.25:bestForG)/cyc3_err_fake', cyc_fake3)
        tf.summary.scalar('D_gan_loss_detail(bestForD:0-0.25:bestForG)/cyc4_err_real', cyc_real4)
        tf.summary.scalar('D_gan_loss_detail(bestForD:0-0.25:bestForG)/cyc4_err_fake', cyc_fake4)

        tf.summary.scalar('C/a_img', C_loss1)
        tf.summary.scalar('C/b_img', C_loss2)
        tf.summary.scalar('C/c_img', C_loss3)
        tf.summary.scalar('C/d_img', C_loss4)
        
        # display an image
        tf.summary.image('1inputs/1T1w', self.tf_vis( self.inputs[:,0,tf.newaxis,:,:] ) )
        tf.summary.image('1inputs/2T1contrast', self.tf_vis( self.inputs[:,1,tf.newaxis,:,:] ) )
        tf.summary.image('1inputs/3T2w', self.tf_vis( self.inputs[:,2,tf.newaxis,:,:] ) )
        tf.summary.image('1inputs/4FLAIR', self.tf_vis( self.inputs[:,3,tf.newaxis,:,:] ) )
        tf.summary.image('4outputs/1Target', self.tf_vis( self.targets) )
       
        # residual
        s = 1.
        sc = 2.
        s2 = 0.55
        cyc1_rgbv = self.tf_vis((self.cyc1*s+self.a_img)/(s+1))
        cyc2_rgbv = self.tf_vis((self.cyc2*s+self.b_img)/(s+1))
        cyc3_rgbv = self.tf_vis((self.cyc3*s+self.c_img)/(s+1))
        cyc4_rgbv = self.tf_vis((self.cyc4*s2+self.d_img)/(s2+1))
        self.cyc1_rgb = self.tf_visout((self.cyc1*s+self.a_img)/(s+1))
        self.cyc2_rgb = self.tf_visout((self.cyc2*s+self.b_img)/(s+1))
        self.cyc3_rgb = self.tf_visout((self.cyc3*s+self.c_img)/(s+1))
        self.cyc4_rgb = self.tf_visout((self.cyc4*s2+self.d_img)/(s2+1))

        tf.summary.image('2cycle/1T1w', cyc1_rgbv)
        tf.summary.image('2cycle/2T1contrast', cyc2_rgbv)
        tf.summary.image('2cycle/3T2w', cyc3_rgbv)
        tf.summary.image('2cycle/4FLAIR', cyc4_rgbv)

        self.a_img_rgb = self.tf_visout( self.a_img )
        self.b_img_rgb = self.tf_visout( self.b_img )
        self.c_img_rgb = self.tf_visout( self.c_img )
        self.d_img_rgb = self.tf_visout( self.d_img )

        self.recon_rgb = self.tf_vis( self.recon )
        tf.summary.image('4outputs/2Recon', self.recon_rgb) 
        tf.summary.image('4outputs/3errx3', self.tf_vis_abs( 3*(self.recon-self.targets)))

        # display an image
        self.summary_op = tf.summary.merge_all()

        self.optimize(self.G_loss, self.D_loss, self.C_loss)

    def tf_visout(self, inp, order=[0,2,3,1]):
        return tf.transpose(inp,order)

    def tf_vis(self, inp, order=[0,2,3,1]):
        return tf.cast( tf.transpose(inp,order)*self.scale,tf.uint8)

    def tf_vis_abs(self, inp, order=[0,2,3,1]):
        return tf.cast( tf.transpose( tf.abs(inp),order)*self.scale,tf.uint8)

    def optimize(self, G_loss, D_loss, C_loss):
        def make_optimizer(loss, variables, lr,  name='Adam'):
            global_step = tf.Variable(0,trainable=False)
            decay_step  = 400
            lr_         = tf.train.exponential_decay(lr, global_step, decay_step,0.99,staircase=True)
            tf.summary.scalar('learning_rate/{}'.format(name), lr_)
            return tf.train.AdamOptimizer( lr_, beta1=0.5 , name=name).minimize(loss,global_step=global_step,var_list=variables)
        
        self.G_optm  = make_optimizer(G_loss, self.G.variables, self.lr,   name='Adam_G')
        self.D_optm  = make_optimizer(D_loss, self.D.variables, self.lr_D, name='Adam_D')
        self.C_optm  = make_optimizer(C_loss, self.D.variables, self.lr_C, name='Adam_C')

class Generator:
    def __init__(self,name,G, nCh_out,nCh=16, use_1x1Conv=False, w_decay=0):
        if G=='UnetINDiv4_CCAM':
            self.net = UnetINDiv4_CCAM
        else:
            st()
        self.name = name
        self.nCh  = nCh
        self.nCh_out = nCh_out
        self.reuse = False
        self.use_1x1Conv=use_1x1Conv
        self.w_decay = w_decay 
    def __call__(self, image, is_Training):
        with tf.variable_scope(self.name, reuse=self.reuse):
            reg_ = tf.contrib.layers.l2_regularizer(scale=self.w_decay) if self.w_decay>0 else None 
            out = self.net(image, self.nCh_out, is_Training, reg_, nCh=self.nCh, _1x1Conv=self.use_1x1Conv)        

        self.reuse=True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return out#, logits

class Discriminator:
    def __init__(self, name='D', nCh=4, w_decay_D=0,DR_ratio=0, class_N=4):
        self.name   = name
        self.nCh    = [nCh, int(nCh*2), int(nCh*4),int(nCh*8), int(nCh*16), int(nCh*32), int(nCh*64), int(nCh*128)]
        self.reuse  = False
        self.k = 4
        self.kernel = 5  # 0/(2**(len(self.nCh)-1))
        self.class_N = class_N
        self.w_decay_D = w_decay_D
        self.dropout_ratio = DR_ratio
        self.use_bias = True

    '''Discriminator 3'''
    def __call__(self, input, is_Training):
        with tf.variable_scope(self.name, reuse=self.reuse):
            reg_ = tf.contrib.layers.l2_regularizer(scale=self.w_decay_D) if self.w_decay_D>0 else None 
            
            '''path #1'''
            p1_C1 = lReLU(  Conv2d(input, self.nCh[0], 'p1_C1', reg=reg_, use_bias=self.use_bias), name='p1c1' )
            p1_C2 = lReLU(  Conv2d(p1_C1, self.nCh[0], 'p1_C2', reg=reg_, use_bias=self.use_bias), name='p1c2' )
            p1_C3 = lReLU(  Conv2d(p1_C2, self.nCh[0], 'p1_C3', reg=reg_, use_bias=self.use_bias), name='p1c3' )
            p1_out = tf.layers.conv2d(p1_C3, filters=self.nCh[2], kernel_size=(self.k,self.k), strides=(4,4), padding="VALID",use_bias=False, kernel_initializer=li.xavier_initializer(), kernel_regularizer=reg_, name='p1_out', data_format=d_form)

            '''path #2'''
            p2_C1 = Conv2d(input, self.nCh[0], 'p2_C1', reg=reg_, use_bias=self.use_bias)
            p2_P = tf.layers.max_pooling2d(p2_C1,pool_size=[2,2],strides=[2,2],padding="VALID", data_format=d_form, name='p2_P')
            p2_C2 = Conv2d(p2_P, self.nCh[1], 'p2_C2', reg=reg_, use_bias=self.use_bias) 
            p2_P2= tf.layers.max_pooling2d(p2_C2,pool_size=[2,2],strides=[2,2],padding="VALID", data_format=d_form, name='p2_P2')
            p2_out= lReLU( Conv2d(p2_P2, self.nCh[2], 'p2_C3', reg=reg_, use_bias=self.use_bias), name='p2out' )

            '''path #3'''
            p3_P = tf.layers.conv2d(input, filters=self.nCh[2], kernel_size=(self.k,self.k), strides=(4,4), padding="VALID",use_bias=False, kernel_initializer=li.xavier_initializer(), kernel_regularizer=reg_, name='p3_P',data_format=d_form)
            p3_C1 = lReLU( Conv2d(p3_P, self.nCh[2], 'p3_C1', reg=reg_, use_bias=self.use_bias), name='p3c1' )
            p3_C2 = lReLU( Conv2d(p3_C1, self.nCh[2], 'p3_C2', reg=reg_, use_bias=self.use_bias), name='p3c2' )
            p3_out = Conv2d(p3_C2, self.nCh[2], 'p3_out', reg=reg_, use_bias=self.use_bias)
            
            ''' [nCh+nCh+nCh]*4 x 60 x 60 '''
            cat = tf.concat([p1_out, p2_out, p3_out], axis=1)
            cat_P = lReLU( Conv2d2x2( cat, kernel_size= self.k, ch_out=self.nCh[3], reg=reg_, name='cat_P'), name='catP' )
            cat_P2 = lReLU( Conv2d2x2( cat_P, kernel_size= self.k, ch_out=self.nCh[4], reg=reg_, name='cat_P2'), name='catP2' )
	    
	    #''' 15 x 15 '''
            dropout_cat_P2 =  tf.layers.dropout(cat_P2, rate=self.dropout_ratio,training=is_Training)
            RF_out = tf.layers.conv2d(dropout_cat_P2, filters=1, kernel_size=(3,3), strides=(1,1), padding="VALID", use_bias=False, data_format=d_form, kernel_initializer=li.xavier_initializer(), name='RF_out')

            '''branch for classification here'''
            cat_P3 = lReLU( Conv2d2x2( cat_P2, kernel_size=self.k, ch_out=self.nCh[5], reg=reg_, name='cat_P3'), name='catP3' )
            dropout_cat_P3 =  tf.layers.dropout(cat_P3, rate=self.dropout_ratio,training=is_Training)
            Class_out = tf.layers.conv2d(dropout_cat_P3, filters=self.class_N, kernel_size=(self.kernel,self.kernel), strides=(1,1), padding="VALID", use_bias=False, data_format=d_form, kernel_initializer=li.xavier_initializer(), name='Class_out')
        '''Discriminator 3'''
        self.reuse=True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return RF_out, Class_out[tf.newaxis,:,:,0,0]
