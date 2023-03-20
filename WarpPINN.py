#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import time
from pyDOE import lhs

# Set up tensorflow in graph mode
tf.compat.v1.disable_eager_execution()

#class WarpPINN(tf.Module):
class WarpPINN:    
    def __init__(self, imr, imt, layers_u, bool_mask, im_mesh, segm_mesh, bg_mesh, lmbmu, pix_crop, reg_mask=1, sigma=None):

        """ 
        The input of the network is of the form (X,Y,Z,t), mesh coordinates of points + time. The output is the deformation u1, u2, u3.
        If using Fourier Feature mappings the input has the form ( cos(B_{1,1}X+B_{1,2}Y+B_{1,3}Z), ..., cos(B_{m,1}X+B_{m,2}Y+B_{m,3}Z), sin(B_{1,1}X+B_{1,2}Y+B_{1,3}Z), ..., sin(B_{m,1}X+B_{m,2}Y+B_{m,3}Z), t)

        - imr: reference image (first frame)
        - imt: template image (all frames, including the first). Registration is performed from imr to each frame in imt.
        - layers_u: list of integers containing width of each layer. Input of network is (x,y,z,t), output is (u1,u2,u3).
                    If Fourier Feature mappings are not used then layers_u = [4, neu_1, ..., neu_L, 3]
                    If Fourier Feature mappings with m features and Ns frequencies are used then layers_u = [2*m+1, neu_1, neu_2, ..., neu_L-1, Ns * neuL, 3]
        - bool_mask: boolean of same shape as imr. True/False if voxel is/is not in left ventricle at reference image.
        - im_mesh: mesh coordinates of image voxels.
        - segm_mesh: mesh coordinates of points in left ventricle. 
        - bg_mesh: mesh coordinates of points in background image.
        - lmbmu: positive real number used to enforce incompressibility stronger in left ventricle than in background image.
        - pix_crop: list containing pixel sizes, and indices of cropping. 
        - reg_mask: positive real, by default equal to 1, that weights the image registration metric to enforce a stronger registration on left ventricle.
                    Minimise |imr(X)-imt(X)| if X is in left ventricle or reg_mask * |imr(X)-imt(X)| if not.
        - sigma: hyperparameter for Fourier Feature mappings. By default, sigma = None, then Fourier Feature mappings are not used.
                If not None, then sigma has to be a list of frequencies [sigma_1,...,sigma_Ns]. 
                Frequencies of input embedding are initialised as N(0, simga_1^2),..., N(0, simga_Ns^2).
                Its choice is not clear and different values should be tried.
        """

        ### Images ###

        # Scaling images to [0,1]
        ub_im = np.max([np.max(imr), np.max(imt)])
        lb_im = np.min([np.min(imr), np.min(imt)])

        self.imr = (imr - lb_im) / (ub_im - lb_im)
        self.imt = (imt - lb_im) / (ub_im - lb_im)

        self.depth = self.imr.shape[0]
        self.height = self.imr.shape[1]
        self.width = self.imr.shape[2]

        ### Coordinates ###

        # Scaling coordinates to [-1,1] x [-1,1] x [-1,1]
        self.ub_coords = np.max(im_mesh, axis=0)
        self.lb_coords = np.min(im_mesh, axis=0)
        ub_lb = self.ub_coords - self.lb_coords

        # Input coordinates are scaled. 
        # These scaling factors have to be used also when computing derivatives.
        lim_y = ub_lb[1] / ub_lb[0]
        lim_z = ub_lb[2] / ub_lb[0]
        sc_factor = np.zeros(3)

        sc_factor[0] = 2 / ub_lb[0]
        sc_factor[1] = 2 * lim_y / ub_lb[1]
        sc_factor[2] = 2 * lim_z / ub_lb[2]

        self.lim = np.array([-1, -lim_y, -lim_z])

        self.sc_factor = sc_factor 

        self.im_sc = self.lim + (im_mesh - self.lb_coords) * self.sc_factor
        self.segm_sc = self.lim + (segm_mesh - self.lb_coords) * self.sc_factor
        self.bg_sc = self.lim + (bg_mesh - self.lb_coords) * self.sc_factor

        self.Xs = self.im_sc[:,0:1]
        self.Ys = self.im_sc[:,1:2]
        self.Zs = self.im_sc[:,2:3]

        X = np.reshape(self.Xs, [self.depth, self.height, self.width])
        Y = np.reshape(self.Ys, [self.depth, self.height, self.width])
        Z = np.reshape(self.Zs, [self.depth, self.height, self.width])

        ### Time ###

        # Frames are assumed to be equispaced
        self.frames = imt.shape[0]
        self.T = np.linspace(0, 1, num=self.frames) 
        self.Ts = np.resize( self.T, self.Xs.shape)

        ### Cropping ###

        # Training can be accelerated by cropping the images.
        # Once it is done, training can be completed with the full-resolution image.
        # By default, cropping is not used during training.
        # IMPORTANT: this feature has not been fully tested. TODO

        Xs_crop = X[:, 0::2, 0::2]
        Ys_crop = Y[:, 0::2, 0::2]
        Zs_crop = Z[:, 0::2, 0::2]

        Xs_crop = Xs_crop.ravel()[:,None]
        Ys_crop = Ys_crop.ravel()[:,None]
        Zs_crop = Zs_crop.ravel()[:,None]

        self.im_sc_crop = np.hstack((Xs_crop, Ys_crop, Zs_crop))

        # Low resolution images.
        self.imr_crop = self.imr[:, 0:-1:2, 0:-1:2]
        self.imt_crop = self.imt[:, 0:-1:2, 0:-1:2]

        self.depth_crop = self.imr_crop.shape[0]
        self.height_crop = self.imr_crop.shape[1]
        self.width_crop = self.imr_crop.shape[2]

        ### Mask ###

        # Used to weight the image registration metric.

        self.mask = bool_mask + reg_mask * (~bool_mask)

        ### Neo Hookean parameters ###

        # lambda/mu factor in Neo Hookean
        self.lamb_mu_in = lmbmu #Incompressibility in the ring
        self.lamb_mu_out = 1 #Compressibility out of the ring

        ### Pixel spacing ###
        self.px = pix_crop[0]
        self.py = pix_crop[1]
        self.pz = pix_crop[2]

        self.crop_x_in, self.crop_y_in = pix_crop[3], pix_crop[4]
        self.crop = np.array([self.crop_x_in, self.crop_y_in, 0])

        ### Neural Network ###
        
        tf.compat.v1.reset_default_graph()
        
        # tf placeholders and graph
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                    log_device_placement=True)) 

        self.weights_u, self.biases_u = self.initialize_NN(layers_u, 'u')

        #tf.compat.v1.disable_eager_execution()

        # Placeholders for image registration
        self.x_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.y_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.z_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        
        self.t_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

        # Placeholders for Neo Hookean
        self.x_e_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.y_e_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.z_e_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

        self.t_e_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

        # Images
        self.imt_tf = tf.compat.v1.placeholder(tf.float32, shape = [self.depth, self.height, self.width])
        self.imr_tf = tf.compat.v1.placeholder(tf.float32, shape = [self.depth, self.height, self.width])

        self.imt_crop_tf = tf.compat.v1.placeholder(tf.float32, shape = [self.depth_crop, self.height_crop, self.width_crop])
        self.imr_crop_tf = tf.compat.v1.placeholder(tf.float32, shape = [self.depth_crop, self.height_crop, self.width_crop])

        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer()

        self.ffm = False

        if sigma is not None:
            # Fourier Feature Mapping 
            
            self.ffm = True

            m = int(layers_u[0]/2)
            self.N_s = len(sigma)
            np.random.seed(0)
            self.B = [np.random.normal(0, s, size=(3,m)).astype('float32') for s in sigma]
            self.B = np.stack(self.B, axis=0)

            self.XYZ_FF = np.matmul(self.im_sc, self.B)        
            self.cos_FF = np.cos(self.XYZ_FF)
            self.sin_FF = np.sin(self.XYZ_FF)

            self.XYZ_FF_crop = np.matmul(self.im_sc_crop, self.B)        
            self.cos_FF_crop = np.cos(self.XYZ_FF_crop)
            self.sin_FF_crop = np.sin(self.XYZ_FF_crop)

            # Placeholders for FFM
            self.cos_tf = tf.compat.v1.placeholder(tf.float32, shape=[self.N_s, None, m])
            self.sin_tf = tf.compat.v1.placeholder(tf.float32, shape=[self.N_s, None, m])

            self.u1_pred, self.u2_pred, self.u3_pred = self.net_u_FFM(self.cos_tf, self.sin_tf, self.t_tf)

        else:
                    
            self.u1_pred, self.u2_pred, self.u3_pred = self.net_u(self.x_tf, self.y_tf, self.z_tf, self.t_tf)

        self.u1x_pred, self.u1y_pred, self.u1z_pred, self.u2x_pred, self.u2y_pred, self.u2z_pred, self.u3x_pred, self.u3y_pred, self.u3z_pred, self.J_pred, self.W_NH_pred = self.net_strain(self.x_e_tf, self.y_e_tf, self.z_e_tf, self.t_e_tf)

        # Prediction / Warping template image 
        self.imr_pred = self.deform(self.imt_tf, self.u1_pred, self.u2_pred, self.u3_pred)
        self.imr_crop_pred = self.deform_crop(self.imt_crop_tf, self.u1_pred, self.u2_pred, self.u3_pred)

        # self.imr_pred_LV = tf.boolean_mask(self.imr_pred, bool_mask)
        # self.imr_tf_LV = tf.boolean_mask(self.imr_tf, bool_mask)
        # self.imr_pred_not_LV = tf.boolean_mask(self.imr_pred, ~bool_mask)
        # self.imr_tf_not_LV = tf.boolean_mask(self.imr_tf, ~bool_mask)

        self.loss_MSE = tf.compat.v1.reduce_mean( self.mask * tf.square( (self.imr_pred - self.imr_tf) ) )
        self.loss_L1 = tf.compat.v1.reduce_mean( self.mask * tf.abs( (self.imr_pred - self.imr_tf) ) )

        self.loss_MSE_crop = tf.compat.v1.reduce_mean( self.mask[:, 0:-1:2, 0:-1:2] * tf.square(self.imr_crop_pred - self.imr_crop_tf))
        self.loss_L1_crop = tf.compat.v1.reduce_mean( self.mask[:, 0:-1:2, 0:-1:2] * tf.abs(self.imr_crop_pred - self.imr_crop_tf))

        self.train_op_Adam_MSE = self.optimizer_Adam.minimize(self.loss_MSE)

        ### LOSS LISTS
        self.lossit_value = []
        self.lossit_MSE = []
        self.lossit_NeoHook = []

        # self.saver = tf.train.Saver()
        self.saver = tf.compat.v1.train.Saver()
        # self.saver = tf.train.Checkpoint()
        # Initialize Tensorflow variables
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        
    def initialize_NN(self, layers, u_str):      
        """ 
        Weights are initialised with Xavier initialisation, bias are initialised as 0.
        """
        def xavier_init(size, name_var, l):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random.normal([in_dim, out_dim], dtype=tf.float32, seed=l) * xavier_stddev, dtype=tf.float32, name=name_var)
        
        weights = []
        biases = []
        num_layers = len(layers) 
        l_end = num_layers - 2
        for l in range(l_end-1):
            W = xavier_init([layers[l], layers[l+1]], 'W_'+u_str+'_'+str(l), l)
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32, name='b_'+u_str+'_'+str(l))
            weights.append(W)
            biases.append(b)    
        W = xavier_init([layers[-2], layers[-1]], 'W_'+u_str+'_'+str(l_end-1), l_end-1)
        b = tf.Variable(tf.zeros([1,layers[-1]], dtype=tf.float32), dtype=tf.float32, name = 'b_'+u_str+'_'+str(l_end-1))
        weights.append(W)
        biases.append(b)          
        return weights, biases

    def neural_net(self, X, weights, biases):
        """ 
        WarpPINN without FFM
        """
        num_layers = len(weights) + 1
        
        H = X
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, y, z, t):
        """ 
        Output: u1, u2 and u3 with WarpPINN.
        """

        u = self.neural_net(tf.concat([x, y, z, t], axis=1), self.weights_u, self.biases_u)

        u1 = u[:,0:1]
        u2 = u[:,1:2]
        u3 = u[:,2:3]

        return u1, u2, u3

    def neural_net_FFM(self, X, weights, biases):
        """  
        WarpPINN-FF
        """
        num_layers = len(weights) + 1
        
        H = X
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        # Reshape to one matrix. Relevant when using more than one value of sigma.
        # TODO: check this step. Seems to be the reason of long computational times for WarpPINN-FF 
        H = tf.transpose(H, [1,0,2])
        H = tf.reshape(H, [-1, tf.shape(W)[0]])
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u_FFM(self, cos, sin, t):
        """ 
        Output: u1, u2 and u3 with WarpPINN-FF.
        """

        t_expanded = tf.reshape(t, [1, tf.shape(t)[0], 1])
        t_expanded = tf.tile(t_expanded, [self.N_s, 1, 1])

        u = self.neural_net_FFM(tf.concat([cos, sin, t_expanded], axis=2), self.weights_u, self.biases_u)

        u1 = u[:,0:1]
        u2 = u[:,1:2]
        u3 = u[:,2:3]

        return u1, u2, u3

    ### Output: strains of u1, u2 and u3
    def net_strain(self, x, y, z, t):

        u1x_str, u1y_str, u1z_str, u2x_str, u2y_str, u2z_str, u3x_str, u3y_str, u3z_str = self.net_u_der(x, y, z, t)

        F1 =  tf.stack([u1x_str, u1y_str, u1z_str])
        F2 =  tf.stack([u2x_str, u2y_str, u2z_str])
        F3 =  tf.stack([u3x_str, u3y_str, u3z_str])

        F = (tf.squeeze(tf.stack([F1,F2,F3])) + tf.expand_dims(tf.eye(3),-1))
        F = tf.transpose(F, [2,0,1]) #batch x 3 x 3
        F = tf.cast(F, dtype = 'float64')

        # Option 1:
        J = tf.linalg.det(F)

        J_segm, J_bg = tf.split(J, 2)
        J_segm = self.lamb_mu_in * tf.square(J_segm - 1)
        J_bg = self.lamb_mu_out * tf.square(J_bg - 1) 
        J_NH = tf.concat([J_segm, J_bg], axis=0)

        C = tf.matmul(tf.transpose(F, [0,2,1]),F)

        W_NeoHook = tf.linalg.trace(C) - 3 - tf.math.log(J**2) + J_NH 

        return u1x_str, u1y_str, u1z_str, u2x_str, u2y_str, u2z_str, u3x_str, u3y_str, u3z_str, tf.cast(J, dtype = 'float32'), tf.cast(W_NeoHook, dtype = 'float32')

    def Fourier_mapping(self, x, y, z):
        xyz_FF = tf.matmul(tf.concat([x,y,z],1), self.B)

        cos_FF = tf.cos(xyz_FF)
        sin_FF = tf.sin(xyz_FF)

        return cos_FF, sin_FF

    def net_u_der(self, x, y, z, t):

        """ 
        Derivatives of predicted u1, u2 and u3.
        Scale factors come from chain rule and scaling of coordinates.
        """

        if self.ffm:
            cos_FF, sin_FF = self.Fourier_mapping(x, y, z)
            u1, u2, u3 = self.net_u_FFM(cos_FF, sin_FF, t)
        else: 
            u1, u2, u3 = self.net_u(x, y, z, t)

        u1x = tf.gradients(u1, x)[0] 
        u1y = tf.gradients(u1, y)[0] * (1/self.sc_factor[0]) * self.sc_factor[1]
        u1z = tf.gradients(u1, z)[0] * (1/self.sc_factor[0]) * self.sc_factor[2]
        
        u2x = tf.gradients(u2, x)[0] * (1/self.sc_factor[1]) * self.sc_factor[0] 
        u2y = tf.gradients(u2, y)[0] 
        u2z = tf.gradients(u2, z)[0] * (1/self.sc_factor[1]) * self.sc_factor[2] 

        u3x = tf.gradients(u3, x)[0] * (1/self.sc_factor[2]) * self.sc_factor[0]
        u3y = tf.gradients(u3, y)[0] * (1/self.sc_factor[2]) * self.sc_factor[1] 
        u3z = tf.gradients(u3, z)[0] 

        return u1x, u1y, u1z, u2x, u2y, u2z, u3x, u3y, u3z 

    def deform(self, imt, u1, u2, u3):
        """ 
        Template image is warped with trilinear interpolation.
        """

        new_coords = self.im_sc + tf.concat([u1, u2, u3], axis=1)

        # Back scaling to mesh coords
        new_coords = (new_coords - self.lim) / self.sc_factor + self.lb_coords

        # Mesh coords to pixel coords
        new_coords = new_coords * np.array([-1/self.px, -1/self.py, 1/self.pz]) - self.crop

        X_new = tf.reshape(new_coords[:,0:1], [self.depth, self.height, self.width])
        Y_new = tf.reshape(new_coords[:,1:2], [self.depth, self.height, self.width])
        Z_new = tf.reshape(new_coords[:,2:3], [self.depth, self.height, self.width])

        #indices
        X0 = tf.cast(tf.floor(X_new), dtype = 'float32')
        Y0 = tf.cast(tf.floor(Y_new), dtype = 'float32')
        Z0 = tf.cast(tf.floor(Z_new), dtype = 'float32')

        X1 = X0 + 1        
        Y1 = Y0 + 1
        Z1 = Z0 + 1

        depth = tf.cast(self.depth, dtype='float32')
        height = tf.cast(self.height, dtype='float32')
        width = tf.cast(self.width, dtype='float32')
        zero = tf.zeros([], dtype='float32')

        X0 = tf.clip_by_value(X0, zero, width - 1)
        X1 = tf.clip_by_value(X1, zero, width - 1)
        Y0 = tf.clip_by_value(Y0, zero, height - 1)
        Y1 = tf.clip_by_value(Y1, zero, height - 1) 
        Z0 = tf.clip_by_value(Z0, zero, depth - 1)
        Z1 = tf.clip_by_value(Z1, zero, depth - 1) 

        X_new = tf.reshape(X_new, [-1])
        Y_new = tf.reshape(Y_new, [-1])
        Z_new = tf.reshape(Z_new, [-1])
        X0 = tf.reshape(X0, [-1])
        X1 = tf.reshape(X1, [-1])
        Y0 = tf.reshape(Y0, [-1])
        Y1 = tf.reshape(Y1, [-1])
        Z0 = tf.reshape(Z0, [-1])
        Z1 = tf.reshape(Z1, [-1])

        Xd = X_new - X0
        Yd = Y_new - Y0
        Zd = Z_new - Z0

        Xdd = X1 - X_new
        Ydd = Y1 - Y_new
        Zdd = Z1 - Z_new

        i000 = tf.cast(X0 + width*Y0 + width*height*Z0, dtype = 'int32') 
        i100 = tf.cast(X1 + width*Y0 + width*height*Z0, dtype = 'int32') 
        i001 = tf.cast(X0 + width*Y0 + width*height*Z1, dtype = 'int32') 
        i101 = tf.cast(X1 + width*Y0 + width*height*Z1, dtype = 'int32') 
        i010 = tf.cast(X0 + width*Y1 + width*height*Z0, dtype = 'int32') 
        i110 = tf.cast(X1 + width*Y1 + width*height*Z0, dtype = 'int32') 
        i011 = tf.cast(X0 + width*Y1 + width*height*Z1, dtype = 'int32') 
        i111 = tf.cast(X1 + width*Y1 + width*height*Z1, dtype = 'int32') 

        imr = tf.reshape(imt, [-1])

        im00 = tf.gather(imr, i000)*Xdd + tf.gather(imr, i100)*Xd
        im10 = tf.gather(imr, i010)*Xdd + tf.gather(imr, i110)*Xd
        im01 = tf.gather(imr, i001)*Xdd + tf.gather(imr, i101)*Xd
        im11 = tf.gather(imr, i011)*Xdd + tf.gather(imr, i111)*Xd

        im0 = im00*Zdd + im01*Zd
        im1 = im10*Zdd + im11*Zd

        return tf.reshape(im0*Ydd+im1*Yd, shape=[self.depth, self.height, self.width])

    def callback_MSE_NeoHook(self, loss, loss_MSE, loss_NeoHook):
        print('Loss: %.3e, Loss MSE: %.3e, Loss NeoHook: %.3e' % 
                            (loss, loss_MSE, loss_NeoHook))
        self.lossit_value.append(loss)
        self.lossit_MSE.append(loss_MSE)
        self.lossit_NeoHook.append(loss_NeoHook)

    def callback_Tuckey_NeoHook(self, loss, loss_Tuckey, loss_NeoHook):
        print('Loss: %.3e, Loss Tuckey: %.3e, Loss NeoHook: %.3e' % 
                            (loss, loss_Tuckey, loss_NeoHook))
        self.lossit_value.append(loss)
        self.lossit_MSE.append(loss_Tuckey)
        self.lossit_NeoHook.append(loss_NeoHook)

    def optimize_LBFGS(self, loss_function):
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss_function, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 500,
                                                                            'maxfun': 500,
                                                                            'maxcor': 50,
                                                                            'maxls': 50,
                                                                            'ftol' : 1.0 * np.finfo(float).eps})
                    
        return optimizer

    ### Pretrain ###
    
    def pretrain(self, tol): 
        """ 
        First, WarpPINN is trained to approximate the identity.
        tol: stop pretraining when loss is below tol.
        """

        preloss = tf.reduce_mean(tf.square(self.u1_pred) + tf.square(self.u2_pred) + tf.square(self.u3_pred))
        train_op_Adam_pretrain = self.optimizer_Adam.minimize(preloss)

        self.prelossit = []
        start_time = time.time()

        if self.ffm:
            tf_dict = {self.cos_tf: self.cos_FF, self.sin_tf: self.sin_FF}
        else:
            tf_dict = {self.x_tf: self.Xs, self.y_tf: self.Ys, self.z_tf: self.Zs}

        it = 0
        seed = 0 # For replicability. 
        while True:
            np.random.seed(seed)
            np.random.shuffle(self.Ts)
            tf_dict[self.t_tf] = self.Ts

            self.sess.run(train_op_Adam_pretrain, tf_dict)
            loss_value = self.sess.run(preloss, tf_dict)
            self.prelossit.append(loss_value)
            # Print
            elapsed = time.time() - start_time
            print('It: %d, Loss: %.3e, Time: %.2f' % 
                (it, loss_value, elapsed))
            start_time = time.time()
            it += 1
            seed += 1
            if loss_value < tol:
                break

    ### Training ### MSE + NEO HOOKEAN

    def train_Adam_MSE_NeoHook(self, nEpoch, mu_NeoHook, size=100, crop=False): 
        """ 
        Minimises using L2-norm.
        nEpoch: number of epochs
        mu_NeoHook: regulariser. Positive number.
        size: size of batch used to evalute the Neo Hookean.
        """
        
        reg_NeoHook = tf.reduce_mean(self.W_NH_pred) #tf.reduce_mean(tf.abs(self.W_NH_pred))
        
        if self.ffm:
            if crop:
                loss_MSE = self.loss_MSE_crop
                loss_NeoHook = loss_MSE + mu_NeoHook * reg_NeoHook
                tf_dict = {self.cos_tf: self.cos_FF_crop, self.sin_tf: self.sin_FF_crop,
                            self.imt_crop_tf: self.imt_crop, self.imr_crop_tf: self.imr_crop} 
            else:
                loss_MSE = self.loss_MSE
                loss_NeoHook = loss_MSE + mu_NeoHook * reg_NeoHook
                tf_dict = {self.cos_tf: self.cos_FF, self.sin_tf: self.sin_FF,
                            self.imr_tf: self.imr}
        else:
            if crop:
                loss_MSE = self.loss_MSE_crop
                loss_NeoHook = loss_MSE + mu_NeoHook * reg_NeoHook
                tf_dict = {self.x_tf: self.Xs_crop, self.y_tf: self.Ys_crop, self.z_tf: self.Zs_crop,
                            self.imt_crop_tf: self.imt_crop, self.imr_crop_tf: self.imr_crop} 
            else:
                loss_MSE = self.loss_MSE
                loss_NeoHook = loss_MSE + mu_NeoHook * reg_NeoHook
                tf_dict = {self.x_tf: self.Xs, self.y_tf: self.Ys, self.z_tf: self.Zs,
                            self.imr_tf: self.imr}    

        train_op_Adam_NeoHook = self.optimizer_Adam.minimize(loss_NeoHook)

        start_time = time.time()

        idx_global = np.arange(self.segm_sc.shape[0])

        idx_t = 0
        for ep in range(nEpoch):
            # Different shuffle for each epoch
            np.random.shuffle(idx_global)
            splits = np.array_split(idx_global, idx_global.shape[0]//size)
            for it, idx in enumerate(splits):
                
                tf_dict[self.x_e_tf] = np.vstack((self.segm_sc[idx,0:1], self.bg_sc[idx,0:1]))
                tf_dict[self.y_e_tf] = np.vstack((self.segm_sc[idx,1:2], self.bg_sc[idx,1:2]))
                tf_dict[self.z_e_tf] = np.vstack((self.segm_sc[idx,2:3], self.bg_sc[idx,2:3]))
                tf_dict[self.t_e_tf] = np.random.uniform(0, 1, size=[2 * len(idx), 1]) #np.ones([2*len(idx),1]) * self.T[idx_t] #idx_global.shape[0]//size, 1])  

                tf_dict[self.t_tf] =  np.ones_like( self.Xs ) *  self.T[idx_t]
                tf_dict[self.imt_tf] = self.imt[idx_t, :, :, :]       

                idx_t = (idx_t + 1) % self.frames 

                self.sess.run(train_op_Adam_NeoHook, tf_dict)
                
                # Print
                iter = it + ep*idx_global.shape[0]//size
                if iter % 100 == 0:

                    loss_value = self.sess.run(loss_NeoHook, tf_dict)
                    loss_value_MSE = self.sess.run(loss_MSE, tf_dict)
                    loss_value_NH = self.sess.run(reg_NeoHook, tf_dict)
                    self.lossit_value.append(loss_value)
                    self.lossit_MSE.append(loss_value_MSE)
                    self.lossit_NeoHook.append(loss_value_NH)

                    elapsed = time.time() - start_time
                    print('Epoch: %d, It: %d, Loss: %.3e, Loss MSE: %.3e, Loss NeoHook: %.3e, Time: %.2f' % 
                            (ep, iter, loss_value, loss_value_MSE, loss_value_NH, elapsed))
                    start_time = time.time() 
    
    def train_BFGS_MSE_NeoHook(self, mu_NeoHook, crop=False): 
        reg_NeoHook = tf.reduce_mean(self.W_NH_pred) 
        
        for idx_t in range(self.frames):

            if self.ffm:
                if crop:
                    loss_MSE = self.loss_MSE_crop
                    loss_NeoHook = loss_MSE + mu_NeoHook * reg_NeoHook
                    tf_dict = {self.cos_tf: self.cos_FF_crop, self.sin_tf: self.sin_FF_crop,
                                self.imt_crop_tf: self.imt_crop[idx_t, :, :, :], self.imr_crop_tf: self.imr_crop} 
                else:
                    loss_MSE = self.loss_MSE
                    loss_NeoHook = loss_MSE + mu_NeoHook * reg_NeoHook
                    tf_dict = {self.cos_tf: self.cos_FF, self.sin_tf: self.sin_FF,
                                self.imt_tf: self.imt[idx_t, :, :, :], self.imr_tf: self.imr}
            else:
                if crop:
                    loss_MSE = self.loss_MSE_crop
                    loss_NeoHook = loss_MSE + mu_NeoHook * reg_NeoHook
                    tf_dict = {self.x_tf: self.Xs_crop, self.y_tf: self.Ys_crop, self.z_tf: self.Zs_crop,
                                self.imt_crop_tf: self.imt_crop[idx_t, :, :, :], self.imr_crop_tf: self.imr_crop} 
                else:
                    loss_MSE = self.loss_MSE
                    loss_NeoHook = loss_MSE + mu_NeoHook * reg_NeoHook
                    tf_dict = {self.x_tf: self.Xs, self.y_tf: self.Ys, self.z_tf: self.Zs,
                                self.imt_tf: self.imt[idx_t, :, :, :], self.imr_tf: self.imr} 

            tf_dict[self.x_e_tf] = np.vstack((self.segm_sc[:,0:1], self.bg_sc[:,0:1]))
            tf_dict[self.y_e_tf] = np.vstack((self.segm_sc[:,1:2], self.bg_sc[:,1:2]))
            tf_dict[self.z_e_tf] = np.vstack((self.segm_sc[:,2:3], self.bg_sc[:,2:3]))
            tf_dict[self.t_e_tf] = np.ones( (2 * self.segm_sc.shape[0],1) ) *  self.T[idx_t]
            tf_dict[self.t_tf] = np.ones_like( self.Xs ) *  self.T[idx_t]

            # Call SciPy's L-BFGS otpimizer
            self.optimizer_MSE = self.optimize_LBFGS(loss_NeoHook)

            self.optimizer_MSE.minimize(self.sess, 
                                    feed_dict = tf_dict,         
                                    fetches = [loss_NeoHook, loss_MSE, reg_NeoHook], 
                                    loss_callback = self.callback_MSE_NeoHook)

    ##################################################################################

    # L1 + NEO HOOKEAN

    def train_Adam_L1_NeoHook(self, nEpoch, mu_NeoHook, size=100, crop=False): 
        """ 
        Minimises using L1-norm.
        nEpoch: number of epochs
        mu_NeoHook: regulariser. Positive number.
        size: size of batch used to evalute the Neo Hookean.
        """
        
        reg_NeoHook = tf.reduce_mean(self.W_NH_pred)
        
        if self.ffm:
            if crop:
                loss_L1 = self.loss_L1_crop
                loss_NeoHook = loss_L1 + mu_NeoHook * reg_NeoHook
                tf_dict = {self.cos_tf: self.cos_FF_crop, self.sin_tf: self.sin_FF_crop,
                            self.imt_crop_tf: self.imt_crop, self.imr_crop_tf: self.imr_crop} 
            else:
                loss_L1 = self.loss_L1
                loss_NeoHook = loss_L1 + mu_NeoHook * reg_NeoHook
                tf_dict = {self.cos_tf: self.cos_FF, self.sin_tf: self.sin_FF,
                            self.imr_tf: self.imr}
        else:
            if crop:
                loss_L1 = self.loss_L1_crop
                loss_NeoHook = loss_L1 + mu_NeoHook * reg_NeoHook
                tf_dict = {self.x_tf: self.Xs_crop, self.y_tf: self.Ys_crop, self.z_tf: self.Zs_crop,
                            self.imt_crop_tf: self.imt_crop, self.imr_crop_tf: self.imr_crop} 
            else:
                loss_L1 = self.loss_L1
                loss_NeoHook = loss_L1 + mu_NeoHook * reg_NeoHook
                tf_dict = {self.x_tf: self.Xs, self.y_tf: self.Ys, self.z_tf: self.Zs,
                            self.imr_tf: self.imr}    

        train_op_Adam_NeoHook = self.optimizer_Adam.minimize(loss_NeoHook)

        start_time = time.time()

        #https://github.com/fsahli/EikonalNet/blob/master/models_tf.py
        idx_global = np.arange(self.segm_sc.shape[0])

        idx_t = 0
        seed = 0
        for ep in range(nEpoch):
            # Different shuffle for each epoch
            np.random.seed(seed)
            np.random.shuffle(idx_global)
            splits = np.array_split(idx_global, idx_global.shape[0]//size)
            for it, idx in enumerate(splits):
                
                tf_dict[self.x_e_tf] = np.vstack((self.segm_sc[idx,0:1], self.bg_sc[idx,0:1]))
                tf_dict[self.y_e_tf] = np.vstack((self.segm_sc[idx,1:2], self.bg_sc[idx,1:2]))
                tf_dict[self.z_e_tf] = np.vstack((self.segm_sc[idx,2:3], self.bg_sc[idx,2:3]))               
                tf_dict[self.t_e_tf] = np.random.uniform(0, 1, size=[2 * len(idx), 1]) 

                tf_dict[self.t_tf] =  np.ones_like( self.Xs ) *  self.T[idx_t]
                tf_dict[self.imt_tf] = self.imt[idx_t, :, :, :]

                idx_t = (idx_t + 1) % self.frames 

                self.sess.run(train_op_Adam_NeoHook, tf_dict)
                
                # Print
                iter = it + ep*idx_global.shape[0]//size
                if iter % 100 == 0:

                    loss_value = self.sess.run(loss_NeoHook, tf_dict)
                    loss_value_L1 = self.sess.run(loss_L1, tf_dict)
                    loss_value_NH = self.sess.run(reg_NeoHook, tf_dict)
                    self.lossit_value.append(loss_value)
                    self.lossit_MSE.append(loss_value_L1)
                    self.lossit_NeoHook.append(loss_value_NH)

                    elapsed = time.time() - start_time
                    print('Epoch: %d, It: %d, Loss: %.3e, Loss L1: %.3e, Loss NeoHook: %.3e, Time: %.2f' % 
                            (ep, iter, loss_value, loss_value_L1, loss_value_NH, elapsed))
                    start_time = time.time() 
            seed += 1
            
    def train_BFGS_L1_NeoHook(self, mu_NeoHook, crop=False): 
        reg_NeoHook = tf.reduce_mean(self.W_NH_pred) 
        
        if self.ffm:
            if crop:
                loss_L1 = self.loss_L1_crop
                loss_NeoHook = loss_L1 + mu_NeoHook * reg_NeoHook
                tf_dict = {self.cos_tf: self.cos_FF_crop, self.sin_tf: self.sin_FF_crop,
                            self.imt_crop_tf: self.imt_crop, self.imr_crop_tf: self.imr_crop} 
            else:
                loss_L1 = self.loss_L1
                loss_NeoHook = loss_L1 + mu_NeoHook * reg_NeoHook
                tf_dict = {self.cos_tf: self.cos_FF, self.sin_tf: self.sin_FF,
                            self.imt_tf: self.imt, self.imr_tf: self.imr}
        else:
            if crop:
                loss_L1 = self.loss_L1_crop
                loss_NeoHook = loss_L1 + mu_NeoHook * reg_NeoHook
                tf_dict = {self.x_tf: self.Xs_crop, self.y_tf: self.Ys_crop, self.z_tf: self.Zs_crop,
                            self.imt_crop_tf: self.imt_crop, self.imr_crop_tf: self.imr_crop} 
            else:
                loss_L1 = self.loss_L1
                loss_NeoHook = loss_L1 + mu_NeoHook * reg_NeoHook
                tf_dict = {self.x_tf: self.Xs, self.y_tf: self.Ys, self.z_tf: self.Zs,
                            self.imt_tf: self.imt, self.imr_tf: self.imr} 

        tf_dict[self.x_e_tf] = np.vstack((self.segm_sc[:,0:1], self.bg_sc[:,0:1]))
        tf_dict[self.y_e_tf] = np.vstack((self.segm_sc[:,1:2], self.bg_sc[:,1:2]))
        tf_dict[self.z_e_tf] = np.vstack((self.segm_sc[:,2:3], self.bg_sc[:,2:3]))

        # Call SciPy's L-BFGS otpimizer
        self.optimizer_L1 = self.optimize_LBFGS(loss_NeoHook)

        self.optimizer_L1.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [loss_NeoHook, loss_L1, reg_NeoHook], 
                                loss_callback = self.callback_MSE_NeoHook)

    ### Prediction ###
    def predict(self, idx_t):
        """ 
        Performs the registration from imt[idx_t] to imr. idx_t indicates the frame to be registered.
        Performs the prediction on voxel points
        Outputs: 
        - im_star: warped template image (should be close to imr)
        - u1_star, u2_star, u3_star: components of deformation field predicted
        - u1x_pred, u1y_pred, u1z_pred, u2x_pred, u2y_pred, u2z_pred, u3x_pred, u3y_pred, u3z_pred: predicted derivatives
        - J_pred: predicted jacobian
        """
        if self.ffm:
            tf_dict = {self.cos_tf: self.cos_FF, self.sin_tf: self.sin_FF, 
                        self.x_e_tf: self.Xs, 
                        self.y_e_tf: self.Ys, 
                        self.z_e_tf: self.Zs,  
                        self.t_tf: np.ones_like(self.Xs) * self.T[idx_t],  
                        self.t_e_tf: np.ones_like(self.Xs) * self.T[idx_t],
                        self.imt_tf: self.imt[idx_t, :, :, :]}
        else:
            tf_dict = {self.x_tf: self.Xs, self.y_tf: self.Ys, self.z_tf: self.Zs, 
                        self.x_e_tf: self.Xs, 
                        self.y_e_tf: self.Ys, 
                        self.z_e_tf: self.Zs,
                        self.t_tf: np.ones_like(self.Xs) * self.T[idx_t],  
                        self.t_e_tf: np.ones_like(self.Xs) * self.T[idx_t],
                        self.imt_tf: self.imt[idx_t, :, :, :]}

        u1_star = self.sess.run(self.u1_pred, tf_dict) / self.sc_factor[0]
        u2_star = self.sess.run(self.u2_pred, tf_dict) / self.sc_factor[1]
        u3_star = self.sess.run(self.u3_pred, tf_dict) / self.sc_factor[2]

        im_star = self.sess.run(self.imr_pred, tf_dict)

        u1x_pred = self.sess.run(self.u1x_pred, tf_dict) 
        u1y_pred = self.sess.run(self.u1y_pred, tf_dict)
        u1z_pred = self.sess.run(self.u1z_pred, tf_dict)

        u2x_pred = self.sess.run(self.u2x_pred, tf_dict) 
        u2y_pred = self.sess.run(self.u2y_pred, tf_dict)
        u2z_pred = self.sess.run(self.u2z_pred, tf_dict) 

        u3x_pred = self.sess.run(self.u3x_pred, tf_dict) 
        u3y_pred = self.sess.run(self.u3y_pred, tf_dict)
        u3z_pred = self.sess.run(self.u3z_pred, tf_dict) 

        J_pred = self.sess.run(self.J_pred, tf_dict)

        return im_star, u1_star, u2_star, u3_star, \
                u1x_pred, u1y_pred, u1z_pred, \
                u2x_pred, u2y_pred, u2z_pred, \
                u3x_pred, u3y_pred, u3z_pred, \
                J_pred

    def surface_deformation(self, surf_mesh, idx_t):
        """ 
        Performs the deformation of surf_mesh at idx_t. idx_t indicates the frame to be registered.
        - surf_mesh: mesh coordinates (for instance, points of the left ventricle)
        Outputs: 
        - XYZ_mesh: new location of points in surf_mesh.
        - u1, u2, u3: components of deformation field predicted
        - u1x_pred, u1y_pred, u1z_pred, u2x_pred, u2y_pred, u2z_pred, u3x_pred, u3y_pred, u3z_pred: predicted derivatives
        - J_pred: predicted jacobian
        """
        
        surf_sc = self.lim + (surf_mesh - self.lb_coords) * self.sc_factor

        if self.ffm:
            # Fourier Feature Mapping with new coordinates
            XYZ_FF = np.matmul(surf_sc, self.B)        
            cos_FF = np.cos(XYZ_FF)
            sin_FF = np.sin(XYZ_FF)

            tf_dict = {self.cos_tf: cos_FF, self.sin_tf: sin_FF,
                    self.x_e_tf: surf_sc[:,0:1],
                    self.y_e_tf: surf_sc[:,1:2],
                    self.z_e_tf: surf_sc[:,2:3],
                    self.t_tf: np.ones_like(surf_sc[:,0:1]) * self.T[idx_t],
                    self.t_e_tf: np.ones_like(surf_sc[:,0:1]) * self.T[idx_t]}
        else:
            tf_dict = {self.x_tf: surf_sc[:,0:1], self.y_tf: surf_sc[:,1:2], self.z_tf: surf_sc[:,2:3],
                    self.x_e_tf: surf_sc[:,0:1],
                    self.y_e_tf: surf_sc[:,1:2],
                    self.z_e_tf: surf_sc[:,2:3],
                    self.t_tf: np.ones_like(surf_sc[:,0:1]) * self.T[idx_t],
                    self.t_e_tf: np.ones_like(surf_sc[:,0:1]) * self.T[idx_t]}

        u1 = self.sess.run(self.u1_pred, tf_dict) / self.sc_factor[0]
        u2 = self.sess.run(self.u2_pred, tf_dict) / self.sc_factor[1]
        u3 = self.sess.run(self.u3_pred, tf_dict) / self.sc_factor[2]

        XYZ_mesh = surf_mesh + np.concatenate([u1, u2, u3], 1)

        u1x_pred = self.sess.run(self.u1x_pred, tf_dict) 
        u1y_pred = self.sess.run(self.u1y_pred, tf_dict)
        u1z_pred = self.sess.run(self.u1z_pred, tf_dict)
        u2x_pred = self.sess.run(self.u2x_pred, tf_dict) 
        u2y_pred = self.sess.run(self.u2y_pred, tf_dict)
        u2z_pred = self.sess.run(self.u2z_pred, tf_dict) 
        u3x_pred = self.sess.run(self.u3x_pred, tf_dict) 
        u3y_pred = self.sess.run(self.u3y_pred, tf_dict)
        u3z_pred = self.sess.run(self.u3z_pred, tf_dict) 
        J_pred = self.sess.run(self.J_pred, tf_dict) 

        return XYZ_mesh, u1, u2, u3, \
                u1x_pred, u1y_pred, u1z_pred, \
                u2x_pred, u2y_pred, u2z_pred, \
                u3x_pred, u3y_pred, u3z_pred, J_pred 

    def lmks_deformation(self, lmks_mesh, idx_t):
        """ 
        Performs the deformation of lmks_mesh at idx_t. idx_t indicates the frame to be registered.
        - lmks_mesh: mesh coordinates of landmarks
        Outputs: 
        - XYZ_mesh: new location of landmarks in surf_mesh.
        """
        lmks_sc = self.lim + (lmks_mesh - self.lb_coords) * self.sc_factor

        if self.ffm:
            # Fourier Feature Mapping with new coordinates
            XYZ_FF = np.matmul(lmks_sc, self.B)        
            cos_FF = np.cos(XYZ_FF)
            sin_FF = np.sin(XYZ_FF)

            tf_dict = {self.cos_tf: cos_FF, self.sin_tf: sin_FF,
                    self.t_tf: np.ones_like(lmks_sc[:,0:1]) * self.T[idx_t]}
        else:
            tf_dict = {self.x_tf: lmks_sc[:,0:1], self.y_tf: lmks_sc[:,1:2], self.z_tf: lmks_sc[:,2:3],
                        self.t_tf: np.ones_like(lmks_sc[:,0:1]) * self.T[idx_t]}

        u1 = self.sess.run(self.u1_pred, tf_dict) / self.sc_factor[0]
        u2 = self.sess.run(self.u2_pred, tf_dict) / self.sc_factor[1]
        u3 = self.sess.run(self.u3_pred, tf_dict) / self.sc_factor[2]

        XYZ_mesh = lmks_mesh + np.concatenate([u1, u2, u3], 1)

        return XYZ_mesh

############ UNUSED METHODS #######################

    def predicted_jacobian(self, x, y, z):
        
        u1x, u1y, u1z, u2x, u2y, u2z, u3x, u3y, u3z  = self.net_u_der(x, y, z)
        F1 =  tf.stack([u1x, u1y, u1z])
        F2 =  tf.stack([u2x, u2y, u2z])
        F3 =  tf.stack([u3x, u3y, u3z])

        F = (tf.squeeze(tf.stack([F1,F2,F3])) + tf.expand_dims(tf.eye(3),-1))
        F = tf.transpose(F, [2,0,1]) #batch x 2 x 2

        J = tf.linalg.det(F)

        return J

    def deform_crop(self, imt, u1, u2, u3):

        new_coords = self.im_sc_crop + tf.concat([u1, u2, u3], axis=1)

        # Back scaling to mesh coords
        new_coords = (new_coords + 1) / self.sc_factor + self.lb_coords

        # Mesh coords to pixel coords
        new_coords = new_coords * np.array([-1/self.px, -1/self.py, 1/self.pz]) - self.crop

        X_new = tf.reshape(new_coords[:,0:1], [self.depth_crop, self.height_crop, self.width_crop])
        Y_new = tf.reshape(new_coords[:,1:2], [self.depth_crop, self.height_crop, self.width_crop])
        Z_new = tf.reshape(new_coords[:,2:3], [self.depth_crop, self.height_crop, self.width_crop])

        #indices
        X0 = tf.cast(tf.floor(X_new), dtype = 'float32')
        Y0 = tf.cast(tf.floor(Y_new), dtype = 'float32')
        Z0 = tf.cast(tf.floor(Z_new), dtype = 'float32')

        X1 = X0 + 1        
        Y1 = Y0 + 1
        Z1 = Z0 + 1

        depth = tf.cast(self.depth_crop, dtype='float32')
        height = tf.cast(self.height_crop, dtype='float32')
        width = tf.cast(self.width_crop, dtype='float32')
        zero = tf.zeros([], dtype='float32')

        X0 = tf.clip_by_value(X0, zero, width - 1)
        X1 = tf.clip_by_value(X1, zero, width - 1)
        Y0 = tf.clip_by_value(Y0, zero, height - 1)
        Y1 = tf.clip_by_value(Y1, zero, height - 1) 
        Z0 = tf.clip_by_value(Z0, zero, depth - 1)
        Z1 = tf.clip_by_value(Z1, zero, depth - 1) 

        X_new = tf.reshape(X_new, [-1])
        Y_new = tf.reshape(Y_new, [-1])
        Z_new = tf.reshape(Z_new, [-1])
        X0 = tf.reshape(X0, [-1])
        X1 = tf.reshape(X1, [-1])
        Y0 = tf.reshape(Y0, [-1])
        Y1 = tf.reshape(Y1, [-1])
        Z0 = tf.reshape(Z0, [-1])
        Z1 = tf.reshape(Z1, [-1])

        Xd = X_new - X0
        Yd = Y_new - Y0
        Zd = Z_new - Z0

        Xdd = X1 - X_new
        Ydd = Y1 - Y_new
        Zdd = Z1 - Z_new

        i000 = tf.cast(X0 + width*Y0 + width*height*Z0, dtype = 'int32') 
        i100 = tf.cast(X1 + width*Y0 + width*height*Z0, dtype = 'int32') 
        i001 = tf.cast(X0 + width*Y0 + width*height*Z1, dtype = 'int32') 
        i101 = tf.cast(X1 + width*Y0 + width*height*Z1, dtype = 'int32') 
        i010 = tf.cast(X0 + width*Y1 + width*height*Z0, dtype = 'int32') 
        i110 = tf.cast(X1 + width*Y1 + width*height*Z0, dtype = 'int32') 
        i011 = tf.cast(X0 + width*Y1 + width*height*Z1, dtype = 'int32') 
        i111 = tf.cast(X1 + width*Y1 + width*height*Z1, dtype = 'int32') 

        imr = tf.reshape(imt, [-1])

        im00 = tf.gather(imr, i000)*Xdd + tf.gather(imr, i100)*Xd
        im10 = tf.gather(imr, i010)*Xdd + tf.gather(imr, i110)*Xd
        im01 = tf.gather(imr, i001)*Xdd + tf.gather(imr, i101)*Xd
        im11 = tf.gather(imr, i011)*Xdd + tf.gather(imr, i111)*Xd

        im0 = im00*Zdd + im01*Zd
        im1 = im10*Zdd + im11*Zd

        return tf.reshape(im0*Ydd+im1*Yd, shape=[self.depth_crop, self.height_crop, self.width_crop])

    
    # MSE + NEO HOOKEAN using previous deformation field

    def train_Adam_MSE_NeoHook_prev(self, nEpoch, mu_NeoHook, mu_prev, u1_prev, u2_prev, u3_prev, size=100, crop=False): 
        
        reg_NeoHook = tf.reduce_mean(self.W_NH_pred) #tf.reduce_mean(tf.abs(self.W_NH_pred))
        reg_prev = tf.reduce_mean( tf.square(self.u1_pred - u1_prev) +  \
                                    tf.square(self.u2_pred - u2_prev) + \
                                    tf.square(self.u3_pred - u3_prev) )

        if self.ffm:
            if crop:
                loss_MSE = self.loss_MSE_crop
                loss_NeoHook = loss_MSE + mu_NeoHook * reg_NeoHook
                tf_dict = {self.cos_tf: self.cos_FF_crop, self.sin_tf: self.sin_FF_crop,
                            self.imt_crop_tf: self.imt_crop, self.imr_crop_tf: self.imr_crop} 
            else:
                loss_MSE = self.loss_MSE
                loss_NeoHook = loss_MSE + mu_NeoHook * reg_NeoHook
                tf_dict = {self.cos_tf: self.cos_FF, self.sin_tf: self.sin_FF,
                            self.imt_tf: self.imt, self.imr_tf: self.imr}
        else:
            if crop:
                loss_MSE = self.loss_MSE_crop
                loss_NeoHook = loss_MSE + mu_NeoHook * reg_NeoHook
                tf_dict = {self.x_tf: self.Xs_crop, self.y_tf: self.Ys_crop, self.z_tf: self.Zs_crop,
                            self.imt_crop_tf: self.imt_crop, self.imr_crop_tf: self.imr_crop} 
            else:
                loss_MSE = self.loss_MSE
                loss_NeoHook = loss_MSE + mu_NeoHook * reg_NeoHook + mu_prev * reg_prev
                tf_dict = {self.x_tf: self.Xs, self.y_tf: self.Ys, self.z_tf: self.Zs,
                            self.imt_tf: self.imt, self.imr_tf: self.imr}    

        train_op_Adam_NeoHook = self.optimizer_Adam.minimize(loss_NeoHook)

        start_time = time.time()

        #https://github.com/fsahli/EikonalNet/blob/master/models_tf.py
        idx_global = np.arange(self.segm_sc.shape[0])

        for ep in range(nEpoch):
            # Different shuffle for each epoch
            np.random.shuffle(idx_global)
            splits = np.array_split(idx_global, idx_global.shape[0]//size)
            for it, idx in enumerate(splits):
                
                tf_dict[self.x_e_tf] = np.vstack((self.segm_sc[idx,0:1], self.bg_sc[idx,0:1]))
                tf_dict[self.y_e_tf] = np.vstack((self.segm_sc[idx,1:2], self.bg_sc[idx,1:2]))
                tf_dict[self.z_e_tf] = np.vstack((self.segm_sc[idx,2:3], self.bg_sc[idx,2:3]))               

                self.sess.run(train_op_Adam_NeoHook, tf_dict)
                loss_value = self.sess.run(loss_NeoHook, tf_dict)
                loss_value_MSE = self.sess.run(loss_MSE, tf_dict)
                loss_value_NH = self.sess.run(reg_NeoHook, tf_dict)
                self.lossit_value.append(loss_value)
                self.lossit_MSE.append(loss_value_MSE)
                self.lossit_NeoHook.append(loss_value_NH)
                # Print
                iter = it + ep*idx_global.shape[0]//size
                if iter % 100 == 0:
                    elapsed = time.time() - start_time
                    print('Epoch: %d, It: %d, Loss: %.3e, Loss MSE: %.3e, Loss NeoHook: %.3e, Time: %.2f' % 
                            (ep, iter, loss_value, loss_value_MSE, loss_value_NH, elapsed))
                    start_time = time.time() 
    
    def train_BFGS_MSE_NeoHook_prev(self, mu_NeoHook, mu_prev, u1_prev, u2_prev, u3_prev, crop=False): 
        
        reg_NeoHook = tf.reduce_mean(self.W_NH_pred) 
        reg_prev = tf.reduce_mean( tf.square(self.u1_pred - u1_prev) +  \
                                    tf.square(self.u2_pred - u2_prev) + \
                                    tf.square(self.u3_pred - u3_prev) )
        if self.ffm:
            if crop:
                loss_MSE = self.loss_MSE_crop
                loss_NeoHook = loss_MSE + mu_NeoHook * reg_NeoHook
                tf_dict = {self.cos_tf: self.cos_FF_crop, self.sin_tf: self.sin_FF_crop,
                            self.imt_crop_tf: self.imt_crop, self.imr_crop_tf: self.imr_crop} 
            else:
                loss_MSE = self.loss_MSE
                loss_NeoHook = loss_MSE + mu_NeoHook * reg_NeoHook
                tf_dict = {self.cos_tf: self.cos_FF, self.sin_tf: self.sin_FF,
                            self.imt_tf: self.imt, self.imr_tf: self.imr}
        else:
            if crop:
                loss_MSE = self.loss_MSE_crop
                loss_NeoHook = loss_MSE + mu_NeoHook * reg_NeoHook
                tf_dict = {self.x_tf: self.Xs_crop, self.y_tf: self.Ys_crop, self.z_tf: self.Zs_crop,
                            self.imt_crop_tf: self.imt_crop, self.imr_crop_tf: self.imr_crop} 
            else:
                loss_MSE = self.loss_MSE
                loss_NeoHook = loss_MSE + mu_NeoHook * reg_NeoHook + mu_prev * reg_prev
                tf_dict = {self.x_tf: self.Xs, self.y_tf: self.Ys, self.z_tf: self.Zs,
                            self.imt_tf: self.imt, self.imr_tf: self.imr} 

        tf_dict[self.x_e_tf] = np.vstack((self.segm_sc[:,0:1], self.bg_sc[:,0:1]))
        tf_dict[self.y_e_tf] = np.vstack((self.segm_sc[:,1:2], self.bg_sc[:,1:2]))
        tf_dict[self.z_e_tf] = np.vstack((self.segm_sc[:,2:3], self.bg_sc[:,2:3]))

        # Call SciPy's L-BFGS otpimizer
        self.optimizer_MSE = self.optimize_LBFGS(loss_NeoHook)

        self.optimizer_MSE.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [loss_NeoHook, loss_MSE, reg_NeoHook], 
                                loss_callback = self.callback_MSE_NeoHook)

    # TUCKEYS BIWEIGHT + NEO HOOKEAN

    def train_BFGS_Tuckey_NeoHook(self, c, mu_NeoHook, crop=False): 

        error_sq = (self.imr - self.imr_pred)**2
        mask_below = tf.cast((error_sq <= c**2), tf.float32)

        rho_above = tf.cast((error_sq > c**2), tf.float32) * c**2 / 2
        rho_below = (c**2 / 2) * (1 - ((1 - ((error_sq * mask_below) / c**2)) ** 3))

        rho = rho_above + rho_below 

        loss_tuckey_biweight = tf.reduce_mean(rho)
        
        reg_NeoHook = tf.reduce_mean(self.W_NH_pred) #tf.reduce_mean(tf.abs(self.W_NH_pred)) 
        
        if self.ffm:
            if crop:
                loss_MSE = self.loss_MSE_crop
                loss_NeoHook = loss_MSE + mu_NeoHook * reg_NeoHook
                tf_dict = {self.cos_tf: self.cos_FF_crop, self.sin_tf: self.sin_FF_crop,
                            self.imt_crop_tf: self.imt_crop, self.imr_crop_tf: self.imr_crop} 
            else:
                loss_NeoHook = loss_tuckey_biweight + mu_NeoHook * reg_NeoHook
                tf_dict = {self.cos_tf: self.cos_FF, self.sin_tf: self.sin_FF,
                            self.imt_tf: self.imt, self.imr_tf: self.imr}
        else:
            if crop:
                loss_MSE = self.loss_MSE_crop
                loss_NeoHook = loss_MSE + mu_NeoHook * reg_NeoHook
                tf_dict = {self.x_tf: self.Xs_crop, self.y_tf: self.Ys_crop, self.z_tf: self.Zs_crop,
                            self.imt_crop_tf: self.imt_crop, self.imr_crop_tf: self.imr_crop} 
            else:
                loss_NeoHook = loss_tuckey_biweight + mu_NeoHook * reg_NeoHook
                tf_dict = {self.x_tf: self.Xs, self.y_tf: self.Ys, self.z_tf: self.Zs,
                            self.imt_tf: self.imt, self.imr_tf: self.imr} 

        tf_dict[self.x_e_tf] = np.vstack((self.segm_sc[:,0:1], self.bg_sc[:,0:1]))
        tf_dict[self.y_e_tf] = np.vstack((self.segm_sc[:,1:2], self.bg_sc[:,1:2]))
        tf_dict[self.z_e_tf] = np.vstack((self.segm_sc[:,2:3], self.bg_sc[:,2:3]))

        # Call SciPy's L-BFGS otpimizer
        self.optimizer_MSE = self.optimize_LBFGS(loss_NeoHook)

        self.optimizer_MSE.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [loss_NeoHook, loss_tuckey_biweight, reg_NeoHook], 
                                loss_callback = self.callback_Tuckey_NeoHook)

    def train_Adam_Tuckey_NeoHook(self, nEpoch, c, mu_NeoHook, size=100, crop=False): 
        
        error_sq = (self.imr - self.imr_pred)**2
        mask_below = tf.cast((error_sq <= c**2), tf.float32)

        rho_above = tf.cast((error_sq > c**2), tf.float32) * c**2 / 2
        rho_below = (c**2 / 2) * (1 - ((1 - ((error_sq * mask_below) / c**2)) ** 3))

        rho = rho_above + rho_below 

        loss_tuckey_biweight = tf.reduce_mean(rho)
        
        reg_NeoHook = tf.reduce_mean(self.W_NH_pred) #tf.reduce_mean(tf.abs(self.W_NH_pred))
        
        if self.ffm:
            if crop:
                loss_MSE = self.loss_MSE_crop
                loss_NeoHook = loss_tuckey_biweight + mu_NeoHook * reg_NeoHook
                tf_dict = {self.cos_tf: self.cos_FF_crop, self.sin_tf: self.sin_FF_crop,
                            self.imt_crop_tf: self.imt_crop, self.imr_crop_tf: self.imr_crop} 
            else:
                loss_NeoHook = loss_tuckey_biweight + mu_NeoHook * reg_NeoHook
                tf_dict = {self.cos_tf: self.cos_FF, self.sin_tf: self.sin_FF,
                            self.imt_tf: self.imt, self.imr_tf: self.imr}
        else:
            if crop:
                loss_MSE = self.loss_MSE_crop
                loss_NeoHook = loss_MSE + mu_NeoHook * reg_NeoHook
                tf_dict = {self.x_tf: self.Xs_crop, self.y_tf: self.Ys_crop, self.z_tf: self.Zs_crop,
                            self.imt_crop_tf: self.imt_crop, self.imr_crop_tf: self.imr_crop} 
            else:
                loss_NeoHook = loss_tuckey_biweight + mu_NeoHook * reg_NeoHook
                tf_dict = {self.x_tf: self.Xs, self.y_tf: self.Ys, self.z_tf: self.Zs,
                            self.imt_tf: self.imt, self.imr_tf: self.imr}    

        train_op_Adam_NeoHook = self.optimizer_Adam.minimize(loss_NeoHook)

        start_time = time.time()

        idx_global = np.arange(self.segm_sc.shape[0])

        for ep in range(nEpoch):
            # Different shuffle for each epoch
            np.random.shuffle(idx_global)
            splits = np.array_split(idx_global, idx_global.shape[0]//size)
            for it, idx in enumerate(splits):
                
                tf_dict[self.x_e_tf] = np.vstack((self.segm_sc[idx,0:1], self.bg_sc[idx,0:1]))
                tf_dict[self.y_e_tf] = np.vstack((self.segm_sc[idx,1:2], self.bg_sc[idx,1:2]))
                tf_dict[self.z_e_tf] = np.vstack((self.segm_sc[idx,2:3], self.bg_sc[idx,2:3]))               

                self.sess.run(train_op_Adam_NeoHook, tf_dict)
                loss_value = self.sess.run(loss_NeoHook, tf_dict)
                loss_value_tuckey = self.sess.run(loss_tuckey_biweight, tf_dict)
                loss_value_NH = self.sess.run(reg_NeoHook, tf_dict)
                self.lossit_value.append(loss_value)
                self.lossit_MSE.append(loss_value_tuckey)
                self.lossit_NeoHook.append(loss_value_NH)
                # Print
                iter = it + ep*idx_global.shape[0]//size
                if iter % 100 == 0:
                    elapsed = time.time() - start_time
                    print('Epoch: %d, It: %d, Loss: %.3e, Loss MSE: %.3e, Loss NeoHook: %.3e, Time: %.2f' % 
                            (ep, iter, loss_value, loss_value_tuckey, loss_value_NH, elapsed))
                    start_time = time.time()