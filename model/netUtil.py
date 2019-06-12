import tensorflow as tf
import tensorflow.contrib.layers as li
from tensorflow.contrib.layers import xavier_initializer as xi
from ipdb import set_trace as st
dtype = tf.float32
d_form  = 'channels_first'
d_form_ = 'NCHW'
ch_dim  = 1

def C(x, ch_out, name,k=3,s=1, reg=None, use_bias=False):
    return tf.layers.conv2d(x, filters=ch_out, kernel_size=(k,k), strides=(s,s), padding="SAME", use_bias=use_bias,data_format=d_form, kernel_initializer=li.xavier_initializer(), kernel_regularizer=reg, name="".join((name,"_Conv")))


def Conv2d(x, ch_out, name, reg=None, use_bias=False,k=3,s=1):
    return tf.layers.conv2d(x, filters=ch_out, kernel_size=(k,k), strides=(s,s), padding="SAME", use_bias=use_bias,data_format=d_form, kernel_initializer=xi(), kernel_regularizer=reg, name="".join((name,"_Conv")))

def BN(x, is_Training, name):
    scope=name+'_bn'
    #return li.batch_norm(x, is_training=True, epsilon=0.000001, center=True, data_format=d_form_, updates_collections=None, scope=scope)
    return tf.cond(is_Training, lambda: li.batch_norm(x, is_training=True, epsilon=1.000001, center=True, data_format=d_form_, updates_collections=None, scope=scope),
            lambda: li.batch_norm(x, is_training=False, updates_collections=None, epsilon=0.000001, center=True, data_format=d_form_,scope=scope, reuse=True) )

def IN(x, name):
    return tf.contrib.layers.instance_norm( x, epsilon=0.000001, center=True, data_format=d_form_, scope=name+'_in')

def Pool2d(x, ch_out, name):
    return tf.layers.conv2d(x, filters=ch_out, kernel_size=(2,2), strides=(2,2), padding="SAME", data_format=d_form,use_bias=False, kernel_initializer=li.xavier_initializer(), name=name)
def Pool2d4x4(x, ch_out, _name):
    return tf.layers.conv2d(x, filters=ch_out, kernel_size=(4,4), strides=(4,4), padding="SAME", data_format=d_form,use_bias=False, kernel_initializer=li.xavier_initializer(), name=_name)


def Conv2dT(x, ch_out, name,k=2,s=2):
    return tf.layers.conv2d_transpose(x, filters=ch_out, kernel_size=(k,k), strides=(s,s), padding="SAME",data_format=d_form,kernel_initializer=li.xavier_initializer(), name=name)

def ReLU(x,name):
    return tf.nn.relu(x, name="".join((name,"_R")))

def lReLU(x,name):
    return tf.nn.leaky_relu(x, name="".join((name,"_lR")))

def Conv1x1(x, ch_out, name, reg=None,use_bias=False):
    return tf.layers.conv2d(x, filters=ch_out, kernel_size=(1,1), strides=(1,1), padding="SAME", use_bias=use_bias, data_format=d_form, kernel_initializer=li.xavier_initializer(), kernel_regularizer=reg, name="".join((name,"_Conv1x1")))

def Conv2d2x2(x, ch_out, name, kernel_size=3,reg=None, use_bias=False):
    return tf.layers.conv2d(x, filters=ch_out, kernel_size=(kernel_size,kernel_size), strides=(2,2), padding="VALID", use_bias=use_bias,data_format=d_form, kernel_initializer=li.xavier_initializer(), kernel_regularizer=reg, name="".join((name,"_Conv2x2")))

def CNR(inp, n_out, k=3, s=1, name='', reg=None, use_bias=False):
    return lReLU( IN( C( inp, n_out, name, k=k, s=s,reg=reg,use_bias=use_bias), name),name)

def CNRCN(inp, n_out, k=3, s=1, _name='', reg=None, use_bias=False):
    CNR = lReLU( IN( C( inp, n_out, _name, k=k, s=s,reg=reg,use_bias=use_bias), _name),_name)
    _name = _name+'2'
    return IN( C( CNR, n_out, _name, k=k, s=s,  reg=reg, use_bias=use_bias), _name)


def CBR(inp, n_out, is_Training, name='', reg=[], _1x1Conv=False):
    if _1x1Conv:
        return lReLU( BN( Conv1x1( inp, n_out, name, reg=reg),is_Training,name),name)
    else:
        return lReLU( BN( Conv2d( inp, n_out, name, reg=reg),is_Training,name),name)

def CCNR(inp, n_out, name='', reg=[]):
    C1 = Conv1x1( inp, int(n_out/2), name, reg=reg)
    C3 = Conv2d( inp, int(n_out/2), name, reg=reg)
    CC  = tf.concat([C1, C3], axis=ch_dim)
    return lReLU( IN( CC,name),name)


def CCBR(inp, n_out, is_Training, name='', reg=[]):
    C1 = Conv1x1( inp, int(n_out/2), name, reg=reg)
    C3 = Conv2d( inp, int(n_out/2), name, reg=reg)
    CC  = tf.concat([C1, C3], axis=ch_dim)
    return lReLU( BN( CC,is_Training,name),name)

def GRCGRC(inp, n_out, is_Training, reg_=[],name_=''):
    ng = 4
    G1 = tf.contrib.layers.group_norm(inp, groups=ng, channels_axis=1, reduction_axes=(-2,-1), activation_fn=tf.nn.relu )
    C1 = Conv2d( G1, n_out, name=name_+'3DConv1st', reg=reg_, use_bias=False, k=3,s=1)
    G2 = tf.contrib.layers.group_norm(C1, groups=ng, channels_axis=1, reduction_axes=(-2,-1), activation_fn=tf.nn.relu )
    C2 = Conv2d( G2, n_out, name=name_+'3DConv2nd', reg=reg_, use_bias=False, k=3,s=1)
    
    return C2+inp

def NVDLMED(inp, n_out, n_seg, is_Training, reg_=[], nCh=32, name_=''):
    # Encoder starts here
    C1   = Conv2d( inp, nCh, name=name_+'inp', reg=reg_, use_bias=False, k=1,s=1)
    C1_1 = GRCGRC(  C1, nCh, is_Training, reg_=reg_, name_=name_+'Unit1')
    
    P1   = Conv2d( C1_1, nCh*2, name=name_+'3DPool1', reg=reg_, use_bias=False, k=2,s=2)
    C2_1 = GRCGRC(   P1, nCh*2, is_Training, reg_=reg_, name_=name_+'Unit2_1')
    C2_2 = GRCGRC( C2_1, nCh*2, is_Training, reg_=reg_, name_=name_+'Unit2_2')
 
    P2   = Conv2d( C2_2, nCh*4, name=name_+'3DPool2', reg=reg_, use_bias=False, k=2,s=2)
    C3_1 = GRCGRC(   P2, nCh*4, is_Training, reg_=reg_, name_=name_+'Unit3_1')
    C3_2 = GRCGRC( C3_1, nCh*4, is_Training, reg_=reg_, name_=name_+'Unit3_2')
    
    P3   = Conv2d( C3_2, nCh*8, name=name_+'3DPool3', reg=reg_, use_bias=False, k=2,s=2)
    C4_1 = GRCGRC(   P3, nCh*8, is_Training, reg_=reg_, name_=name_+'Unit4_1')
    C4_2 = GRCGRC( C4_1, nCh*8, is_Training, reg_=reg_, name_=name_+'Unit4_2')
    C4_3 = GRCGRC( C4_2, nCh*8, is_Training, reg_=reg_, name_=name_+'Unit4_3') 
    C4_4 = GRCGRC( C4_3, nCh*8, is_Training, reg_=reg_, name_=name_+'Unit4_4') 

    # Decoder for segmentation
    U1   = Conv2dT( C4_4, nCh*4,name=name_+'upConv1',k=2,s=2) 
    D1   = GRCGRC(U1 + C3_2, nCh*4, is_Training, reg_=reg_, name_=name_+'DUnit1')
    U2   = Conv2dT(   D1, nCh*2,name=name_+'upConv2',k=2,s=2) 
    D2   = GRCGRC(U2 + C2_2, nCh*2, is_Training, reg_=reg_, name_=name_+'DUnit2')
    U3   = Conv2dT(   D2,   nCh,name=name_+'upConv3',k=2,s=2) 
    D3   = GRCGRC(U3 + C1_1, nCh, is_Training, reg_=reg_, name_=name_+'DUnit3')

    Seg_out = tf.sigmoid( Conv2d(D3, n_seg, reg=reg_, name=name_+'SegOut',k=1))

    # Decoder for VAE
    VAE_GR = tf.contrib.layers.group_norm(C4_4, groups=4, channels_axis=1, reduction_axes=(-2,-1), activation_fn=tf.nn.relu )
    VAE_C  =  Conv2d( VAE_GR, n_out, name=name_+'3DConvVAE', reg=reg_, use_bias=False, k=3,s=2)
    VAE_C  = tf.reshape(VAE_C,[VAE_C.shape[0],4*15*15])

    VAE_mu = tf.layers.dense( VAE_C, 128)
    VAE_sigma = tf.layers.dense( VAE_C, 128)

    z = VAE_mu + VAE_sigma * tf.random_normal( tf.shape(VAE_sigma),0,1,dtype=dtype)
    
    VU__ = tf.nn.relu( tf.layers.dense( z, 4*30*30 ) )
    VU_  = Conv2d( tf.reshape( VU__, [VU__.shape[0], 4,30,30]), 256, name=name_+'Conv1VAEx2ch', reg=reg_, use_bias=False, k=1,s=1)
    VU   = Conv2dT(  VU_, nCh*4,name=name_+'upx2VAE1',k=2,s=2) 
    
    VU2  = GRCGRC(    VU, nCh*4, is_Training, reg_=reg_, name_=name_+'VAE_Up2')
    VU2U = Conv2dT(  VU2, nCh*2, name=name_+'upx2VAE2',k=2,s=2) 
    
    VU3  = GRCGRC(  VU2U, nCh*2, is_Training, reg_=reg_, name_=name_+'VAE_Up3')
    VU3U = Conv2dT(  VU3, nCh, name=name_+'upx2VAE3',k=2,s=2) 

    VU4  = GRCGRC(VU3U, nCh, is_Training, reg_=reg_, name_=name_+'VAE_Up4')
    VAE_out = Conv2d( VU4, n_out, name=name_+'VAE_out', reg=reg_, use_bias=False, k=1,s=1)

    return VAE_out, Seg_out, VAE_mu, VAE_sigma


 
def CCAM(sympara, input_feature, nCh, reduction=4):
    nMLPm = nCh/reduction
    # avgPool
    avgPool = tf.reduce_mean(input_feature,axis=[2,3],keepdims=False)
    # concat
    inp_MLPm = tf.concat([sympara, avgPool], axis=1)
    mhidden1 = tf.layers.dense(inp_MLPm, nMLPm, use_bias=True, activation=tf.nn.leaky_relu)
    mhidden2 = tf.layers.dense(mhidden1, nMLPm, use_bias=True, activation=tf.nn.leaky_relu)
    MLPm     = tf.layers.dense(mhidden2, nCh, use_bias=True)
    #attention
    ch_attention = tf.sigmoid(MLPm)
    refined_feature = tf.multiply(input_feature, ch_attention[:,:,tf.newaxis,tf.newaxis])
    return refined_feature 

def UnetINDiv4_CCAM(inp, n_out, is_Training, reg_=[], nCh=64, name_='', _1x1Conv=False, use_bias=True):
    sympara = inp[:,4:,0,0]   
    r = 4 
    
    mask = inp[:,4:,:,:]   
    ainp = tf.concat( [  inp[:,0:1,:,:],mask],axis=ch_dim)
    binp = tf.concat( [  inp[:,1:2,:,:],mask],axis=ch_dim)
    cinp = tf.concat( [  inp[:,2:3,:,:],mask],axis=ch_dim)
    dinp = tf.concat( [  inp[:,3:4,:,:],mask],axis=ch_dim)
    anCh = nCh/4
    
    '''a path'''
    adown0_1     =    CNR(  ainp,      anCh,  name=name_+'alv0_1', reg=reg_, use_bias=use_bias)
    adown0_2     =    CNR(  adown0_1,  anCh,  name=name_+'alv0_2', reg=reg_, use_bias=use_bias)
    apool1       = Pool2d(  adown0_2,  anCh*2,name=name_+'alv1_p') 
    adown1_1     =    CNR(  apool1,    anCh*2,name=name_+'alv1_1', reg=reg_, use_bias=use_bias) 
    adown1_2     =    CNR(  adown1_1,  anCh*2,name=name_+'alv1_2', reg=reg_, use_bias=use_bias)
    apool2       = Pool2d(  adown1_2,  anCh*4,name=name_+'alv2_p')
    adown2_1     =    CNR(  apool2,    anCh*4,name=name_+'alv2_1', reg=reg_, use_bias=use_bias) 
    adown2_2     =    CNR(  adown2_1,  anCh*4,name=name_+'alv2_2', reg=reg_, use_bias=use_bias)
    apool3       = Pool2d(  adown2_2,  anCh*8,name=name_+'alv3_p')
    adown3_1     =    CNR(  apool3,    anCh*8,name=name_+'alv3_1', reg=reg_, use_bias=use_bias) 
    adown3_2     =    CNR(  adown3_1,  anCh*8,name=name_+'alv3_2', reg=reg_, use_bias=use_bias)
    apool4       = Pool2d(  adown3_2,  anCh*16, name=name_+'alv4_p')

    '''b path'''
    bdown0_1     =    CNR(  binp,      anCh,  name=name_+'blv0_1', reg=reg_, use_bias=use_bias)
    bdown0_2     =    CNR(  bdown0_1,  anCh,  name=name_+'blv0_2', reg=reg_, use_bias=use_bias)
    bpool1       = Pool2d(  bdown0_2,  anCh*2,name=name_+'blv1_p') 
    bdown1_1     =    CNR(  bpool1,    anCh*2,name=name_+'blv1_1', reg=reg_, use_bias=use_bias) 
    bdown1_2     =    CNR(  bdown1_1,  anCh*2,name=name_+'blv1_2', reg=reg_, use_bias=use_bias)
    bpool2       = Pool2d(  bdown1_2,  anCh*4,name=name_+'blv2_p')
    bdown2_1     =    CNR(  bpool2,    anCh*4,name=name_+'blv2_1', reg=reg_, use_bias=use_bias) 
    bdown2_2     =    CNR(  bdown2_1,  anCh*4,name=name_+'blv2_2', reg=reg_, use_bias=use_bias)
    bpool3       = Pool2d(  bdown2_2,  anCh*8,name=name_+'blv3_p')
    bdown3_1     =    CNR(  bpool3,    anCh*8,name=name_+'blv3_1', reg=reg_, use_bias=use_bias) 
    bdown3_2     =    CNR(  bdown3_1,  anCh*8,name=name_+'blv3_2', reg=reg_, use_bias=use_bias)
    bpool4       = Pool2d(  bdown3_2,  anCh*16, name=name_+'blv4_p')

    '''c path'''
    cdown0_1     =    CNR(  cinp,      anCh,  name=name_+'clv0_1', reg=reg_, use_bias=use_bias)
    cdown0_2     =    CNR(  cdown0_1,  anCh,  name=name_+'clv0_2', reg=reg_, use_bias=use_bias)
    cpool1       = Pool2d(  cdown0_2,  anCh*2,name=name_+'clv1_p') 
    cdown1_1     =    CNR(  cpool1,    anCh*2,name=name_+'clv1_1', reg=reg_, use_bias=use_bias) 
    cdown1_2     =    CNR(  cdown1_1,  anCh*2,name=name_+'clv1_2', reg=reg_, use_bias=use_bias)
    cpool2       = Pool2d(  cdown1_2,  anCh*4,name=name_+'clv2_p')
    cdown2_1     =    CNR(  cpool2,    anCh*4,name=name_+'clv2_1', reg=reg_, use_bias=use_bias) 
    cdown2_2     =    CNR(  cdown2_1,  anCh*4,name=name_+'clv2_2', reg=reg_, use_bias=use_bias)
    cpool3       = Pool2d(  cdown2_2,  anCh*8,name=name_+'clv3_p')
    cdown3_1     =    CNR(  cpool3,    anCh*8,name=name_+'clv3_1', reg=reg_, use_bias=use_bias) 
    cdown3_2     =    CNR(  cdown3_1,  anCh*8,name=name_+'clv3_2', reg=reg_, use_bias=use_bias)
    cpool4       = Pool2d(  cdown3_2,  anCh*16, name=name_+'clv4_p')

    '''d path'''
    ddown0_1     =    CNR(  dinp,      anCh,  name=name_+'dlv0_1', reg=reg_, use_bias=use_bias)
    ddown0_2     =    CNR(  ddown0_1,  anCh,  name=name_+'dlv0_2', reg=reg_, use_bias=use_bias)
    dpool1       = Pool2d(  ddown0_2,  anCh*2,name=name_+'dlv1_p') 
    ddown1_1     =    CNR(  dpool1,    anCh*2,name=name_+'dlv1_1', reg=reg_, use_bias=use_bias) 
    ddown1_2     =    CNR(  ddown1_1,  anCh*2,name=name_+'dlv1_2', reg=reg_, use_bias=use_bias)
    dpool2       = Pool2d(  ddown1_2,  anCh*4,name=name_+'dlv2_p')
    ddown2_1     =    CNR(  dpool2,    anCh*4,name=name_+'dlv2_1', reg=reg_, use_bias=use_bias) 
    ddown2_2     =    CNR(  ddown2_1,  anCh*4,name=name_+'dlv2_2', reg=reg_, use_bias=use_bias)
    dpool3       = Pool2d(  ddown2_2,  anCh*8,name=name_+'dlv3_p')
    ddown3_1     =    CNR(  dpool3,    anCh*8,name=name_+'dlv3_1', reg=reg_, use_bias=use_bias) 
    ddown3_2     =    CNR(  ddown3_1,  anCh*8,name=name_+'dlv3_2', reg=reg_, use_bias=use_bias)
    dpool4       = Pool2d(  ddown3_2,  anCh*16, name=name_+'dlv4_p')

    ''' decoder '''
    pool4 = tf.concat([apool4,bpool4,cpool4,dpool4], axis=ch_dim)
    down4_1     =    CNR(    pool4, nCh*16, name=name_+'lv4_1', reg=reg_, use_bias=use_bias) 
    down4_2     =    CNR(  down4_1, nCh*16, name=name_+'lv4_2', reg=reg_, use_bias=use_bias)
    down4_2     =    CCAM(sympara, down4_2, nCh*16, reduction=r) 
    up4         = Conv2dT( down4_2,  nCh*8, name=name_+'lv4__up')
    
    down3_2 = tf.concat([adown3_2,bdown3_2,cdown3_2, ddown3_2], axis=ch_dim)
    CC3         = tf.concat([down3_2, up4], axis=ch_dim)
    up3_1     =    CNR(        CC3,  nCh*8, name=name_+'lv3_1', reg=reg_, use_bias=use_bias) 
    up3_2     =    CNR(      up3_1,  nCh*8, name=name_+'lv3_2', reg=reg_, use_bias=use_bias)
    up3_2     =    CCAM(sympara, up3_2, nCh*8, reduction=r) 
    up3         = Conv2dT(   up3_2,  nCh*4, name=name_+'lv3__up')
    
    down2_2 = tf.concat([adown2_2,bdown2_2,cdown2_2, ddown2_2], axis=ch_dim)
    CC2         = tf.concat([down2_2, up3], axis=ch_dim)
    up2_1       =    CNR(      CC2,  nCh*4, name=name_+'lv2__1', reg=reg_, use_bias=use_bias)
    up2_2       =    CNR(    up2_1,  nCh*4, name=name_+'lv2__2', reg=reg_, use_bias=use_bias)
    up2_2       =    CCAM(sympara, up2_2, nCh*4, reduction=r)
    up2         = Conv2dT(   up2_2,  nCh*2, name=name_+'lv2__up')

    down1_2 = tf.concat([adown1_2,bdown1_2,cdown1_2,ddown1_2], axis=ch_dim)
    CC1         = tf.concat([down1_2, up2], axis=ch_dim)
    up1_1       =    CNR(      CC1,  nCh*2, name=name_+'lv1__1', reg=reg_, use_bias=use_bias)
    up1_2       =    CNR(    up1_1,  nCh*2, name=name_+'lv1__2', reg=reg_, use_bias=use_bias)
    up1_2       =    CCAM(sympara, up1_2, nCh*2, reduction=r)
    up1         = Conv2dT(   up1_2,   nCh, name=name_+'lv1__up')

    down0_2 = tf.concat([adown0_2,bdown0_2,cdown0_2,ddown0_2], axis=ch_dim)
    CC0         = tf.concat([down0_2, up1], axis=ch_dim)
    up0_1       =    CNR(      CC0,   nCh, name=name_+'lv0__1', reg=reg_, use_bias=use_bias)
    up0_2       =    CNR(    up0_1,   nCh, name=name_+'lv0__2', reg=reg_, use_bias=use_bias)

    return  ReLU(Conv1x1(   up0_2, n_out,name=name_+'conv1x1'),name='thelast')




