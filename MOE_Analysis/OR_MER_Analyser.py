### Author: Bogdan Bintu (bbintu@ucsd.edu)

### This contains the functions necessary to perform the analysis of the MOE MERFISH dataset

### Language: Python 2.7
import cPickle as pickle
import time
import itertools
import sys,os,glob
import numpy as np
import scipy as sp
from PIL import Image
import matplotlib as mpl
from tqdm import tqdm_notebook as tqdm
#mpl.use('Agg') #for rc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import IOTools as io
import GeneralTools as gt

from scipy.ndimage import sum as ndi_sum
from scipy.ndimage import label
from scipy.ndimage.measurements import center_of_mass
import sklearn.cluster
from scipy.ndimage import gaussian_filter

import cPickle as pickle
from scipy import signal
from scipy import ndimage
import tifffile
import FittingTools as ft
import cv2
def getsorted_files(or_chr):
    files_ = np.array(or_chr.files)
    return files_[np.argsort([os.path.basename(fls[0]) for fls in files_])]
def register_files_smallrot_big(self,fl0,fl1,reg_frame=None,im_msk=None,sG=10,sg=2,th_cut= 2,ssq=256):
    """
    Given two dax files fl1 and fl2 this loads the reg_frame
    It normalizes the image files using two gaussin filters <sg size filter> /<sG size filter>
    To avoid hot pixel problems the normalized images are capped at +/-th_cut

    It tiles the images with squares of size ssq that have non-zero im_msk

    It then computes the registration between the frames using fourier cross-correlation.
    It returns x,y such that gt.t"""

    import cv2
    #from imreg_dft import imreg
    import AlignmentTools as at

    if im_msk is None:
        #self.loadMOEmasks()
        im_msk = self.im_MOEmask(self.fov_name)
    if reg_frame is None:
        ncols = 4 if '3col' in self.device else 3
        nfrs = io.readInfoFile(fl0)['number_frames']
        reg_frame = int(nfrs/ncols/2+1)*ncols-1

    if fl0.split('.')[-1]=='dax':
        im0 = io.DaxReader(fl0).loadAFrame(reg_frame).astype(np.float32)#laod first dapi frame
    elif '.dax' not in fl0:
        im0 = load_im_set(fl0,[reg_frame])[0].astype(np.float32)
    else:
        fl0_ = glob.glob(fl0.split('.dax')[0]+'_fr*')[-1]
        im0 = np.load(fl0_).T.astype(np.float32)
    if fl1.split('.')[-1]=='dax':
        im1 = io.DaxReader(fl1).loadAFrame(reg_frame).astype(np.float32)#laod first dapi frame
    elif '.dax' not in fl1:
        im1 = load_im_set(fl1,[reg_frame])[0].astype(np.float32)
    else:
        fl1_ = glob.glob(fl1.split('.dax')[0]+'_fr*')[-1]
        im1 = np.load(fl1_).T.astype(np.float32)
    
    #correct using blur
    im0_50 = cv2.GaussianBlur(im0,(0,0),sigmaX=sG)
    im1_50 = cv2.GaussianBlur(im1,(0,0),sigmaX=sG)

    im0_sc= cv2.GaussianBlur(im0,(0,0),sigmaX=sg)/im0_50
    im1_sc= cv2.GaussianBlur(im1,(0,0),sigmaX=sg)/im1_50
    im0_sc=im0_sc-im0_sc.mean()
    im1_sc=im1_sc-im1_sc.mean()

    im1_sc[im1_sc>th_cut]=th_cut
    im0_sc[im0_sc>th_cut]=th_cut
    im0_sc[im0_sc<-th_cut]=-th_cut
    im1_sc[im1_sc<-th_cut]=-th_cut

    lims = get_list_limits(self,ssq=ssq)
    ts = []
    Xs = []
    cors=[]
    for lim in lims:
        zm,zM,xm,xM,ym,yM = lim
        if np.sum(im_msk[xm:xM,ym:yM]>0)>0:
            #((xt,yt),success) = imreg.translation(im0_sc,im1_sc[xm:xM,ym:yM],normalized=False,plt_val=False)
            ((xt,yt),success) =at.fftalign_2d(im1_sc,im0_sc[xm:xM,ym:yM], center=[0, 0], max_disp=np.inf, plt_val=False,norm=False,return_cor=True)
            ts.append((-xt+xm,-yt+ym))
            Xs.append(((xm+xM)/2.,(ym+yM)/2.))
            cors.append(success)
    return ts,Xs,cors,im0_sc,im1_sc
def get_vmins_vmax(self,fr=1.25,fr_min=None):
    dic_min_max = {}
    #print(len(self.Xconvs))
    for iR in range(self.nRs):
        X2ds=[]
        im_ = self.ims_matrix[iR]
        mins=[]
        maxs=[]
        texts = []
        for conv in self.Xconvs:
            conv_code = conv.code-1
            if iR in conv_code:
                #print('here')
                X = get_points_inside(conv,res=20).astype(int)
                trace = im_[tuple(X.T)]
                mins.append(np.min(trace))
                maxs.append(np.max(trace))
                X2d = get_2d_line(conv,axs=[1,2])
                X2ds.append(X2d)
                txt = conv.olfr+'\n'+str(conv.code)
                texts.append(txt)
        if len(mins)>0:
            vmin,vmax=np.median(mins),np.median(maxs)
            if fr_min is None: fr_min=fr
            vmin,vmax = (vmin+vmax)/2-(vmax-vmin)/2*fr_min,(vmin+vmax)/2+(vmax-vmin)/2*fr
            dic_min_max[iR]=[vmin,vmax]
    self.dic_min_max=dic_min_max
def get_big_xyz_v2(self,X):
    X_red = np.array(X)
    spacing = np.array([self.spacing_xy,self.spacing_xy,self.spacing_z])
    pads = np.array([self.pad_xy,self.pad_xy,self.pad_z])
    maxs = np.max(X_red,0)
    mins = np.min(X_red,0)
    szb = (maxs-mins)*spacing+pads
    XB = np.indices(szb).reshape([3,-1]).T+mins*spacing
    Xred = X_red*spacing
    XB_ = XB/pads.astype(float)-(0.5-10**(-10))
    Xred_ = Xred/pads.astype(float)

    neigh = NearestNeighbors(n_neighbors=1,p=np.inf)
    neigh.fit(Xred_)
    dists,inds = neigh.kneighbors(XB_,1,return_distance=True)
    keep = dists[:,0]<=0.5
    XB = XB[keep]
    return XB.T[[2,0,1]]
def plot_cor_dec(im_diff,im_argmax_cd,mask_th_dec,mask_th_bk,cd_mask,X_red,X_list,title):
    ims = [im_diff,im_argmax_cd,mask_th_dec,mask_th_bk,cd_mask]
    titles = ['Image','max guess','abs cross thresh','background thresh','final']
    fig, ax_arr = plt.subplots(2, 3, sharex=True,sharey=True)
    sz_t,sx_t,sy_t = ims[0].shape
    for i,ax in enumerate(np.array(ax_arr).flatten()):
        if i<len(ims):
            cmap = cm.gray if i==0 else cm.jet 
            f = np.max if i==0 else np.mean
            im_plot = gt.minmax(f(ims[i],axis=0))
            sx_t,sy_t=im_plot.shape
            sx_t0,sy_t0=sx_t,sy_t
            im_plot = np.concatenate([im_plot,np.ones([sx_t,2]),gt.minmax(f(ims[i],axis=-1)).T],axis=1)
            sx_t,sy_t=im_plot.shape
            im_plot_ = np.concatenate([gt.minmax(f(ims[i],axis=-2)),np.ones([sz_t,sy_t-sy_t0])],axis=1)
            im_plot = np.concatenate([im_plot_[::-1,:],np.ones([2,sy_t]),im_plot],axis=0)
            sx_t,sy_t=im_plot.shape
            ax.imshow(im_plot,interpolation='nearest',cmap=cmap)
            ax.set_title(titles[i])
            ax.set_xlim([0,sx_t])
            ax.set_ylim([0,sy_t])
        elif i==len(ims):
            x_,y_,z_ = X_red.T
            ax.plot(y_,x_+sy_t-sy_t0,'k+')
            for iX,X_ in enumerate(X_list):
                x_,y_,z_=X_.T
                ax.plot(y_,x_+sy_t-sy_t0,'o')
                ax.text(np.mean(y_),np.mean(x_+sy_t-sy_t0),str(iX))

            plt.axis('equal')
            ax.set_xlim([0,sx_t])
            ax.set_ylim([0,sy_t])
            pass
        ax.set_adjustable('box-forced')
    fig.suptitle(title)
    return fig

def set_bad_to_0(im_,X_t_):
    X_t,X_bad = X_t_
    if type(X_t_[0]) is int:
        tx,ty = X_t_
        if tx<0:im_[:,tx:,:]=0
        else:im_[:,:tx,:]=0
        if ty<0:im_[:,:,ty:]=0
        else:im_[:,:,:ty]=0
        return im_
    else:
        im_0=[]
        for im__ in im_:
            im__[X_bad[:,0],X_bad[:,1]]=0
            im_0.append(im__)
    return np.array(im_0)
def apply_X_t(im_,X_t_):
    X_t,X_bad = X_t_
    im_t = np.array([im__[X_t[:,0],X_t[:,1]] for im__ in im_]).reshape(im_.shape)
    #pisici
    return set_bad_to_0(im_t,X_t_)#im_t
def get_best_rot(X1,X2,plt_val=False):
    PA = np.array(X1)
    PB = np.array(X2)
    cA = np.mean(PA,0)
    cB = np.mean(PB,0)
    H=np.dot((PA-cA).T,(PB-cB))
    U,S,V = np.linalg.svd(H)
    R = np.dot(V,U.T).T
    #if np.linalg.det(R)<0:
    #    R[:,-1]*=-1
    t = -np.dot(cA,R)+cB
    PAT =  np.dot(PA,R)+t
    if plt_val:
        plt.figure()
        plt.plot(PA[:,0],PA[:,1],'r.')
        plt.plot(PB[:,0],PB[:,1],'bx')
        plt.plot(PAT[:,0],PAT[:,1],'gx')
        plt.axis('equal')
        #plt.title(np.round(np.mean(np.abs(np.round(PAT)-PB)),np.mean(np.abs(np.round(PA)-PB)))
    return R,t
def refine(X1,X2,keep,fr=0.95,target_distance=0.5):
    X1_ = X1[keep]
    X2_ = X2[keep]
    R,t = get_best_rot(X2_,X1_)
    X2T = np.dot(X2_,R)+t
    scores = np.linalg.norm(X2T-X1_,axis=1)
    #print(np.mean(scores))
    if np.mean(scores)>target_distance:
        keep_i = np.argsort(scores)[:int(len(scores)*fr)]
        keep_ = keep[keep_i]
    else:
        keep_ = keep
    return keep_,R,t

def get_points_inside(conv,res=np.inf,add_verts=False):
    """
    Ginve a a convex hul <conv> this returns an np.array of points inside the conv
    """
    
    sz = (conv.M-conv.m).astype(int)
    sz_ = np.min([sz,[res]*len(sz)],0).astype(int)
    resc_ = sz/sz_.astype(float)
    Ii = (np.indices(sz_).reshape([3,-1]).T*resc_+conv.m)
    if len(Ii)==0:
        return Ii
    Ii = Ii[in_hullF(Ii,conv)]
    if add_verts:
        return np.concatenate([Ii,conv.points[conv.vertices]])
    return Ii
def compare(conv1,conv2,th_fr=0.75,res=10):
    """
    This checks if conv1 in conv2 (>th_fr intersection using resolution res x res x res smaple points).
    If conv1 in conv2, compare their correlation score and decided whether to remove conv1 by adding conv1.remove=True
    """
    if (not getattr(conv1,'remove',False)) and (not getattr(conv2,'remove',False)):
        if intersect_box(conv1,conv2):
            Xi1 = get_points_inside(conv1,res=res,add_verts=True)
            Xi1_in2 = Xi1[in_hullF(Xi1,conv2)]
            fr_inter = len(Xi1_in2)/float(len(Xi1))
            if fr_inter>th_fr and conv1.cor<conv2.cor:
                conv1.remove=True
def compare_volume(conv1,conv2,th_fr=0.75,res=10):
    """
    This checks if conv1 in conv2 (>th_fr intersection using resolution res x res x res sample points).
    If conv1 in conv2, compare their volume (favoring bigger volume) and decided whether to remove conv1 by adding conv1.remove=True
    """
    if (not getattr(conv1,'remove',False)) and (not getattr(conv2,'remove',False)):
        if intersect_box(conv1,conv2):
            Xi1 = get_points_inside(conv1,res=res,add_verts=True)
            Xi1_in2 = Xi1[in_hullF(Xi1,conv2)]
            fr_inter = len(Xi1_in2)/float(len(Xi1))
            if fr_inter>th_fr and conv1.volume<conv2.volume:
                conv1.remove=True                
from sklearn.neighbors import NearestNeighbors
def filter_X(X,nneigh,dist):
    """ Given a list of points X (N x ndim), keep only the points that have nneigh a distance dist away"""
    if len(X)<nneigh:
        return []
    X = np.array(X)
    neigh = NearestNeighbors(n_neighbors=1,p=2)
    neigh.fit(X)
    dists,inds = neigh.kneighbors(X,nneigh,return_distance=True)
    keep = np.sum(dists<dist,-1)>=nneigh
    return X[keep]

def X_to_XBkeep(self,X,code_,npts):
    XB = self.get_big_xyz(X[:,::-1])
    keep = np.sum(self.im_thresholds[code_][...,XB[0],XB[1],XB[2]],0)>=npts
    XBkeep = np.array(XB).T[keep,:]
    return XBkeep
def get_2d_line(conv1,axs=[1,2]):
    if len(conv1.points)==0:
        return np.zeros([0,len(axs)])
    c1_2d = ConvexHull_(conv1.points[conv1.vertices][:,axs])
    Xc1_2d = c1_2d.points[list(c1_2d.vertices)+[c1_2d.vertices[0]]]
    return Xc1_2d
def display_convs_singleR(self,Xconvs,R=1):
    iR=R-1
    X2ds=[]
    for conv in Xconvs:
        conv_code = conv.code-1
        if iR in conv_code:
            X2d = get_2d_line(conv,axs=[1,2])
            X2ds.append(X2d)

    f = plt.figure()
    plt.imshow(np.max(self.ims_matrix[iR],0),cmap='gray',vmax=3)
    for X2d in X2ds:
        plt.plot(X2d[:,1],X2d[:,0],'b')
    return f
def display_convs_singleR_multicolor(self,Xconvs_list,R=1):
    iR=R-1
    colors= ['r','g','b','c','y','m']
    X2ds_list = []
    for Xconvs in Xconvs_list:
        X2ds=[]
        for conv in Xconvs:
            conv_code = conv.code-1
            if iR in conv_code:
                X2d = get_2d_line(conv,axs=[1,2])
                X2ds.append(X2d)
        X2ds_list.append(X2ds)
    f = plt.figure()
    plt.imshow(np.max(self.ims_matrix[iR],0),cmap='gray',vmax=3)
    for ic,X2ds in enumerate(X2ds_list):
        for X2d in X2ds:
            plt.plot(X2d[:,1],X2d[:,0],colors[ic%len(colors)])
    return f
def getConvs_inDisplay(self,f,Xconvs,R):
    ax = f.get_axes()[0]
    xm,xM = np.sort(ax.get_ylim())
    ym,yM = np.sort(ax.get_xlim())
    elems = []
    iR=R-1
    codes = self.codes-1
    for icd,cd in enumerate(codes):
        if iR in cd:
            for conv in Xconvs[icd]:
                if np.all((conv.c[1:]<[xM,yM])&(conv.c[1:]>[xm,ym])):
                    elems.append(conv)
    return elems


import numpy as np
from scipy.spatial import ConvexHull 
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

from sklearn.cluster import MiniBatchKMeans as KMeans#MiniBatchKMeans as KMeans#KMeans#
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist
def get_list_limits(self,ssq = 256):
    sz,sx,sy = self.sz_image,self.sx_image,self.sy_image #get dimensions of full image
    nsqs = [int(np.ceil(sc/float(ssq))) for sc in [sz,sx,sy]]
    limits_s = []
    for iz in range(nsqs[0]):
        for ix in range(nsqs[1]):
            for iy in range(nsqs[2]):
                limits_s.append([iz*ssq,np.min([(iz+1)*ssq,sz]),ix*ssq,np.min([(ix+1)*ssq,sx]),iy*ssq,np.min([(iy+1)*ssq,sy])])
    return limits_s
def get_traces(self,limits,im_msk=None):
    zm,zM,xm,xM,ym,yM=limits
    if im_msk is None:
        im_msk = self.im_MOEmask(self.fov_name)[xm:xM,ym:yM] #get the mask
        im_msk = np.array([im_msk]*(zM-zm),dtype=bool)
    _isgood = np.any(im_msk)
    if _isgood:
        ims_norm = get_normed_subimages(self,limits,s_norm=50) # get the normalized subimages
        #coordinates
        XA = np.indices([zM-zm,xM-xm,yM-ym])+np.array([zm,xm,ym])[:,np.newaxis,np.newaxis,np.newaxis]
        XA = np.array([XA_[im_msk] for XA_ in XA]).T
        #traces
        traces = np.array([im_norm[im_msk] for im_norm in ims_norm]).T
        return traces,XA
    else:
        return [],[]
def get_list_limits(self,ssq = 256):
    sz,sx,sy = self.sz_image,self.sx_image,self.sy_image #get dimensions of full image
    nsqs = [int(np.ceil(sc/float(ssq))) for sc in [sz,sx,sy]]
    limits_s = []
    for iz in range(nsqs[0]):
        for ix in range(nsqs[1]):
            for iy in range(nsqs[2]):
                limits_s.append([iz*ssq,np.min([(iz+1)*ssq,sz]),ix*ssq,np.min([(ix+1)*ssq,sx]),iy*ssq,np.min([(iy+1)*ssq,sy])])
    return limits_s
def mean_filter(self,im,recomp=True,recalc_lens=False):
    import pyfftw
    from scipy import fftpack
    fft = pyfftw.interfaces.numpy_fft.rfftn
    ifft = pyfftw.interfaces.numpy_fft.irfftn
    axes = [0,1,2]
    nthread = 100
    if (not hasattr(self,'fftker')) or recomp:
        #get kernel
        ker = getattr(self,'ker',np.ones([self.pad_z,self.pad_xy,self.pad_xy],dtype=np.float32))
        ker = ker/np.sum(ker)
        self.ker=ker
        kz,kx,ky = (np.array(self.ker.shape)/2).astype(int)
        fshape = np.array(im.shape)+[2*kz,2*kx,2*ky]
        if recalc_lens:
            fshape = [fftpack.next_fast_len(fshape[a]) for a in axes]
        sp2 = fft(ker, fshape, axes=axes,threads=nthread)
        self.fftker = sp2
        self.ker=ker
        
    kz,kx,ky = (np.array(self.ker.shape)/2).astype(int)
    sp2 = self.fftker
    #print(self.ker.shape)
    im_pd = np.pad(im,((kz,kz),(kx,kx),(ky,ky)),mode='reflect')
    #print(im_pd.shape,self.ker.shape)
    
    fshape = im_pd.shape
    shape = im_pd.shape
    if recalc_lens:
        fshape = [fftpack.next_fast_len(fshape[a]) for a in axes]
    sp1 = fft(im_pd, fshape, axes=axes,threads=nthread)
    im_prod = sp1 * sp2
    #print(im_pd.shape,self.ker.shape,im_prod.shape)
    ret = ifft(im_prod, fshape, axes=axes,threads=nthread)
    if recalc_lens:
        fslice = tuple([slice(sz) for sz in shape])
        ret = ret[fslice]
    
    ret = ret[2*kz:,2*kx:,2*ky:]
    #print(im_pd.shape,self.ker.shape)
    
    return ret    
def set_default_gaussian_kernel(self,sz = [15,56,56],resc_pads = 0.25):
    #pad_z,pad_x,pad_y = self.pad_z,self.pad_xy,self.pad_xy
    pad_z,pad_x,pad_y = np.array(sz)*resc_pads
    X = np.indices(np.array(sz).astype(int))
    X = X-np.array(X.shape[1:])[:,np.newaxis,np.newaxis,np.newaxis]/2.
    X = np.exp(-np.sum(X*X/np.array([pad_z,pad_x,pad_y])[:,np.newaxis,np.newaxis,np.newaxis]**2,0))
    X = X/np.sum(X)
    self.ker = X
def get_normed_subimages(self,limits,s_norm=50):
    #assume self has: ims_matrix, nRs, sz_image...
    nRs = self.nRs #number of 
    sz,sx,sy = self.sz_image,self.sx_image,self.sy_image #get dimensions of full image
    #pad the limits
    zm,zM,xm,xM,ym,yM=limits
    pad = int(s_norm*2)
    zm_,xm_,ym_= np.max([0,zm-pad]),np.max([0,xm-pad]),np.max([0,ym-pad])
    zM_,xM_,yM_= np.min([sz,zM+pad]),np.min([sx,xM+pad]),np.min([sy,yM+pad])
    #append normalized subimages
    ims_norm = []
    for iR in range(nRs):
        im_sm = self.ims_matrix[iR][zm_:zM_,xm_:xM_,ym_:yM_]
        
        #im_dif = im_sm-oma.quick_norm3d(im_sm,s_norm)
        #im_norm = im_dif/np.sqrt(oma.quick_norm3d(im_dif*im_dif,s_norm))
        
        im_dif = im_sm-quick_norm(im_sm,s_norm*4)
        im_norm = im_dif/np.sqrt(quick_norm(im_dif*im_dif,s_norm*4))
        
        #recrop to original dimensions
        im_norm_ = im_norm[(zm-zm_):(zM-zm_),(xm-xm_):(xM-xm_),(ym-ym_):(yM-ym_)]
        ims_norm.append(im_norm_)
    ims_norm = np.array(ims_norm)
    return ims_norm
def quick_norm_v2(im3d,s=200):
    import cv2
    im3d_ =np.mean(im3d,0).astype(np.float32)
    return cv2.blur(im3d_,(s,s))[np.newaxis,...]
def quick_fullnorm(im_sm,s_norm=50):
    im_sm = im_sm.astype(np.float32)
    im_dif = im_sm-quick_norm_v2(im_sm,s_norm*4)
    im_norm = im_dif/np.sqrt(quick_norm_v2(im_dif*im_dif,s_norm*4))
    return im_norm
def quick_norm(im3d,s=200):
    import cv2
    im3d_ =im3d.astype(np.float32)
    return np.array([cv2.blur(im2d,(s,s)) for im2d in im3d_])

def quick_norm3d(im,s=50):
    from scipy import ndimage
    import pyfftw
    input_ = fftf=pyfftw.interfaces.numpy_fft.fftn(im.astype(np.float32))
    result = ndimage.fourier_gaussian(input_, sigma=s)
    result = pyfftw.interfaces.numpy_fft.ifftn(result).real
    return result
def intersect_box(conv1,conv2):
    return np.all((conv2.M>conv1.m)&(conv1.M>conv2.m))
def get_mins_maxs(X): return np.min(X,0),np.max(X,0)
def get_DBSCANX(X,eps, min_samples,return_max=False,return_labels=False):
    if len(X)<min_samples: return []
    dbscan_ = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_.fit(X)
    labels_ = dbscan_.labels_
    if return_labels:
        return labels_
    ncells = np.max(labels_)+1
    Xs = [X[labels_==ic] for ic in range(ncells)]
    if return_max:
        if len(Xs)>0:
            lens = [len(X_) for X_ in Xs]
            return Xs[np.argmax(lens)]
        else:
            return []
    return Xs
def cellClustering(X,dbscan_corr_min_samples,dbscan_corr_eps,fr=1.5,performDBSCAN=True):
    if performDBSCAN:
        Xs = get_DBSCANX(X,eps=dbscan_corr_eps, min_samples=dbscan_corr_min_samples)
    else:
        Xs = [X]
    performX = np.ones(len(Xs),dtype=bool)
    while True:
        Xs_ = []
        performX_=[]
        for X_,split_ in zip(Xs,performX):
            if split_ and len(X_)>0:
                XT,split = XmeansClustering(X_,dbscan_corr_min_samples,dbscan_corr_eps*fr)
                split = [split]*len(XT)
            else:
                XT,split=[X_],[False]
            Xs_+=XT
            performX_+=split
        if len(Xs_)==len(Xs):
            break
        else:
            Xs=Xs_
            performX = performX_
    return Xs
def XmeansClustering(X,dbscan_corr_min_samples,cell_diam):
    
    Xm,XM = get_mins_maxs(X)
    dbim = np.max(XM-Xm)
    if dbim<cell_diam:
        return [X],False
    kno=1
    prev_labels = []
    while True:
        kno+=1

        kmeans = KMeans(kno)
        kmeans.fit(X)
        kmeanslabels_=np.array(kmeans.labels_,dtype=int)
        kmeanscluster_centers_=[]


        #if too few samples in a cluster ignore that cluster
        for iK in range(kno):
            investigate_small = kmeanslabels_==iK
            if np.sum(investigate_small)<dbscan_corr_min_samples:
                kmeanslabels_[investigate_small]=-1
            else:
                kmeanscluster_centers_.append(kmeans.cluster_centers_[iK])
        #if the min distance between any 2 non-ignored clusters is smaller than cell diamater flag break!
        if len(kmeanscluster_centers_)<2:
            break
        center_distances = pdist(kmeanscluster_centers_)
        if np.min(center_distances)<cell_diam:
            break
        prev_labels=kmeanslabels_
    if kno==2:
        return [X],False
    else:
        return [X[prev_labels==ik] for ik in range(kno-1) if np.sum(prev_labels==ik)>0],True

def denoise(X,sz=5,no_pts=5):
    
    tree = cKDTree(X)
    res = tree.query_ball_tree(tree,sz)
    lens = np.array(list(map(len,res)))
    return X[lens>no_pts]

class DeadConvexHull():
    def __init__(self):
        self.vertices = []
        self.points = np.array([])
        self.equations = []
        self.volume = 0
        self.m = np.array([0,0,0])
        self.M = self.m
        self.c = self.m
        self.L = 0
def ConvexHull_(elems):
    try:
        obj = ConvexHull(elems)
        obj.m = np.min(elems,0)
        obj.M = np.max(elems,0)
        obj.c = np.mean(elems,0)
        obj.L = np.max(obj.M-obj.m)
        return obj
    except:
        return DeadConvexHull()
def repopulate_egr1(dic_int_fl,egr1_fl=None,h_cutoff=0):
    dic_int = pickle.load(open(dic_int_fl,'rb'))
    if egr1_fl is None:
        egr1_fl = dic_int_fl.replace(r'\Decoded',r'\EGR1_fits').replace('__decoded_dic_int.pkl','_EGR1_zxyh.npy')
    zxyh = np.load(egr1_fl)
    zxyh_ = zxyh[:,zxyh[3,:]>h_cutoff].T
    for cell in dic_int:
        for tag_ in ['_loose','_tight']:
            X = dic_int[cell]['OR_convHull'+tag_]
            #print(X)
            if len(X):
                mins = np.min(X,0)
                maxs = np.max(X,0)
                in_box = np.all((zxyh_[:,:3]>=mins)&(zxyh_[:,:3]<=maxs),-1)
                zxyh__ = zxyh_[in_box]
                convH = ConvexHull_(X)
                inH = in_hullF(zxyh__[:,:3],convH)
                if len(zxyh__)>0:
                    zxyh__kp = zxyh__[inH]
                else:
                    zxyh__kp = zxyh__
                num_pts = len(zxyh__kp)
                dic_int[cell]['num_pts'+tag_],dic_int[cell]['zxyh_EGR1'+tag_]=num_pts,zxyh__kp
    return dic_int
def norm_sets(ims,norm_dapi=True,perc =80,nl = 30):
    normns = [1 for im_ in ims ]
    if norm_dapi and len(ims)>nl:
        normns = [np.percentile(im_,perc) for im_ in ims]        
        nll = int(nl/2)
        norms_ = list(normns[:nll][::-1])+list(normns)+list(normns[-nll:][::-1])
        norms_ = [np.median(norms_[il:il+nl]) for il in range(len(norms_)-nl)]
        normns = [e/norms_[0] for e in norms_]
    return normns
def save_tile_image_and_label_simple(self,ims_fls,folder,save_file,maxI=20,resc=2,pix_size_=0.153,
                        max_impose=False,verbose=True,rotation_angle=0,add_txt=False,norm_dapi=False):

    import tifffile
    
    pix_size=pix_size_*resc
    nims = len(np.load(ims_fls[0]))
    for iim in tqdm(range(nims)):
        ims,xys=[],[]
        
        for fl in tqdm(ims_fls):
            fov = os.path.basename(fl).replace('_ims.npy','.dax')
            dax = folder+os.sep+fov
            dic_inf = io.readInfoFile(dax.replace('.dax','.inf'))
            
            #Read appropriate frames
            ims_ = np.load(fl)[iim]
            ims_  = np.array(ims_)[1:-1:resc,1:-1:resc]
            
            #Consider adding illumination correction and better stitching
            xys.append([dic_inf['Stage X']/pix_size,dic_inf['Stage Y']/pix_size])
            ims.append(ims_)
            
        xys_=np.array(xys,dtype=int)
        xys_=xys_-np.expand_dims(np.min(xys_,axis=0), axis=0)


        sx,sy = ims[0].shape
        dim_base = list(np.max(xys_,axis=0)+np.array([sx+1,sy+1]))
    
    
        save_file_ = save_file.replace('.tif','_R'+str(iim+1)+'.tif')
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        im_base = np.zeros(dim_base,dtype=float)

        infos_filename = '.'.join(save_file_.split('.')[:-1])+'.infos'
        fid = open(infos_filename,'w')

        normns = norm_sets(ims[:],norm_dapi=(((iim+1)==nims) and norm_dapi))
        #print normns
        for i_,(im,(x,y),norm) in enumerate(zip(ims[:],xys_[:],normns)):
            im_ = np.swapaxes(im[::1,::-1,...],0,1)
            
            im_ = im_/norm
            
            im_base[x:sx+x,y:sy+y,...]=im_
            
            """
            im_cor_ = np.swapaxes(im_cor[::1,::-1,...],0,1)#[::-1,::-1]
            im_cor_ -= np.min(im_cor_)
            if rotation_angle!=0:
                im_ = rotate(im_, rotation_angle, center=None, scale=1.0)
                im_cor_ = rotate(im_cor_, rotation_angle, center=None, scale=1.0)
            if max_impose:
                im_base[x:sx+x,y:sy+y,...]=(im_compare[x:sx+x,y:sy+y,...]*im_base[x:sx+x,y:sy+y,...]+im_cor_*im_)/(im_compare[x:sx+x,y:sy+y,...]+im_cor_)
                im_compare[x:sx+x,y:sy+y,...]=(im_compare[x:sx+x,y:sy+y,...]*im_compare[x:sx+x,y:sy+y,...]+im_cor_*im_cor_)/(im_compare[x:sx+x,y:sy+y,...]+im_cor_)

                im_compare[x:sx+x,y:sy+y,...][np.isnan(im_compare[x:sx+x,y:sy+y,...])]=0
                im_base[x:sx+x,y:sy+y,...][np.isnan(im_base[x:sx+x,y:sy+y,...])]=0
                #im_sel[im_ind]#np.mean([im_base[x:sx+x,y:sy+y,...],im_],axis=0)
            else:
                im_base[x:sx+x,y:sy+y,...]=im_
            """
            fov = os.path.basename(ims_fls[i_]).replace('_ims.npy','.dax')
            if add_txt:
                txt = fov.split('_')[-1].split('.')[0]
                im_base = cv2.putText(im_base, txt, (y+100,x+100), cv2.FONT_HERSHEY_SIMPLEX ,  1, 10000, 2, cv2.LINE_AA) 
            save_pars = [ims_fls[i_],x,sx+x,y,sy+y,resc]
            fid.write("\t".join(map(str,save_pars))+'\n')
        fid.close()
        tifffile.imsave(save_file_,np.clip(im_base*2**16/maxI,0,2**16-1).astype(np.uint16))
        
def minmax(im,min_=None,max_=None,percmax=99.9,percmin=0.1):
    im_ = np.array(im,dtype=float)
    if min_ is None:
        min_ = np.min(im_)
        if percmin is not None:
            min_ = np.percentile(im_,percmin)
    if max_ is None:
        max_ = np.max(im_)
        if percmax is not None:
            max_ = np.percentile(im_,percmax)
    if (max_-min_)<=0:
        im_ = im_*0
    else:
        im_ = (im_-min_)/(max_-min_)
        im_[im_<0]=0
        im_[im_>1]=1
    return im_,min_,max_
def zoom_in(obj,col_row = 1.25,percmax=99.95,percmin=.5):
    ims=obj.ims
    ims_names = obj.image_names
    z_min,z_max,x_min,x_max,y_min,y_max =obj.get_limits()
    ims_sm = [im_[z_min:z_max,x_min:x_max,y_min:y_max] for im_ in ims]

    iimax = len(ims_sm)

    nrow = int(np.sqrt(iimax)/col_row)
    ncol = int(np.ceil(iimax/float(nrow)))
    ssz,ssx,ssy = ims_sm[0].shape
    imf = np.zeros([ssx*nrow,ssy*ncol])
    iim=0

    R_to_col={1:'750',2:'647',3:'561',4:'750',6:'647',5:'561',7:'750',8:'647',9:'561',10:'750',12:'647',11:'561',14:'750',15:'647',13:'561'}
    def nm_to_R(nm_sm,cols = ['561','647','750']):
        try:
            col = nm_sm.split('_')[-1]
            iRs = np.array(nm_sm[3:].split('_')[0].split(';')[0].split(','),dtype=int)
            cols_ = [R_to_col[iR] for iR in iRs]
            iR = str(iRs[cols_.index(col)])
            return 'R'+iR
        except:
            return nm_sm

    txts=[]
    for icol in range(ncol):
        for irow in range(nrow):
            if iim<len(ims_sm):
                ims_sm_ = ims_sm[iim]
                nm_sm = ims_names[iim]

                txt = nm_to_R(nm_sm)


                if 'dapi' in txt: ims_sm_=[ims_sm_[int(len(ims_sm_)/2)]]

                im_plt,min_,max_ = minmax(np.max(ims_sm_,0),percmax=percmax,percmin=percmin)
                imf[irow*ssx:(irow+1)*ssx,icol*ssy:(icol+1)*ssy]=im_plt
                #txt+=' - '+str([int(min_),int(max_)])
                txts.append([txt,irow*ssx,icol*ssy])
                iim+=1

    import matplotlib.pylab as plt
    fig = plt.figure(figsize=(12,7))
    for txt,xtxt,ytxt in txts:
        plt.text(ytxt,xtxt,txt,color='white',backgroundcolor='k',horizontalalignment='left',verticalalignment='top',fontsize=10)
    plt.imshow(imf,interpolation='nearest',cmap='gray')
    plt.axis('off')
    return fig,ims_sm
def save_tile_image_and_label(fls_iter,save_file,resc=2,custom_frms=None,pix_size_=0.162,
                        max_impose=True,verbose=False,im_cor__=None,
                              rotation_angle=0,add_txt=True,norm_dapi=False,invertX=False,invertY=False):

        import tifffile
        ims,xys=[],[]
        pix_size=pix_size_*resc
        fls_iter_ = fls_iter
        if verbose: fls_iter_ = tqdm(fls_iter)
        for dax in fls_iter_:
            

            dic_inf = io.readInfoFile(dax)
            
            #Read appropriate frames
            daxReader = io.DaxReader(dax)
            dapi_im = [daxReader.loadAFrame(frm) for frm in custom_frms]
            
            dapi_im_small  = np.array(dapi_im)[:,1:-1:resc,1:-1:resc]
            
            #Illumination correction:
            dapi_im_small = np.max(dapi_im_small,0)
            #Consider adding illumination correction and better stitching
            xys.append([dic_inf['Stage X']/pix_size,dic_inf['Stage Y']/pix_size])
            ims.append(dapi_im_small)

        if im_cor__ is None:
            im_cor = np.median(ims,axis=0)
            ims = [im_/im_cor*np.median(im_cor) for im_ in ims]
        else:
            im_cor = im_cor__[1:-1:resc,1:-1:resc].copy()
            ims = [im_/im_cor*np.median(im_cor) for im_ in ims]
        #return im_cor
        xys_=np.array(xys,dtype=int)
        xys_=xys_-np.expand_dims(np.min(xys_,axis=0), axis=0)

        if len(ims[0].shape)>2:
            sx,sy,sz = ims[0].shape
            dim_base = list(np.max(xys_,axis=0)+np.array([sx+1,sy+1]))
            dim_base+=[sz]
        else:
            sx,sy = ims[0].shape
            dim_base = list(np.max(xys_,axis=0)+np.array([sx+1,sy+1]))
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        im_base = np.zeros(dim_base,dtype=float)
        im_compare = np.zeros(dim_base,dtype=float)
        infos_filename = '.'.join(save_file.split('.')[:-1])+'.infos'
        fid = open(infos_filename,'w')
        

        normns = norm_sets(ims,norm_dapi=norm_dapi,perc =80,nl = 30)
        
        
        for i_,(im,(x,y),norm) in enumerate(zip(ims[:],xys_[:],normns)):
            Ix = int((1-2*invertX))
            Iy = int((1-2*invertY))
            im_ = np.swapaxes(im[::Ix,::Iy,...],0,1)
            im_cor_ = np.swapaxes(im_cor[::Ix,::Iy,...],0,1)#[::-1,::-1]
            im_cor_ = im_cor_.astype(np.float32)
            im_cor_ -= np.percentile(im_cor_,1)
            
            im_cor_[im_cor_<0]=0
            
            #im_ = im_[::-1,::-1]
            im_ = im_/norm
            if rotation_angle!=0:
                im_ = rotate(im_, rotation_angle, center=None, scale=1.0)
                im_cor_ = rotate(im_cor_, rotation_angle, center=None, scale=1.0)
            if max_impose:
                #im_base[x:sx+x,y:sy+y,...]=np.max([im_base[x:sx+x,y:sy+y,...],im_],axis=0)
                #im_ind = im_compare[x:sx+x,y:sy+y,...]>im_cor_
                #im_compare[x:sx+x,y:sy+y,...] = np.max([im_compare[x:sx+x,y:sy+y,...],im_cor_],axis=0)
                im_base[x:sx+x,y:sy+y,...]=(im_compare[x:sx+x,y:sy+y,...]*im_base[x:sx+x,y:sy+y,...]+im_cor_*im_)/(im_compare[x:sx+x,y:sy+y,...]+im_cor_)
                im_compare[x:sx+x,y:sy+y,...]=(im_compare[x:sx+x,y:sy+y,...]*im_compare[x:sx+x,y:sy+y,...]+im_cor_*im_cor_)/(im_compare[x:sx+x,y:sy+y,...]+im_cor_)
                
                im_compare[x:sx+x,y:sy+y,...][np.isnan(im_compare[x:sx+x,y:sy+y,...])]=0
                im_base[x:sx+x,y:sy+y,...][np.isnan(im_base[x:sx+x,y:sy+y,...])]=np.nanmedian(im_base[x:sx+x,y:sy+y,...])
                #im_sel[im_ind]#np.mean([im_base[x:sx+x,y:sy+y,...],im_],axis=0)
            else:
                im_base[x:sx+x,y:sy+y,...]=im_
            if add_txt:
            
                txt = os.path.basename(fls_iter[i_]).split('_')[-1].split('.')[0]
                im_base = cv2.putText(im_base, txt, (y+100,x+100), cv2.FONT_HERSHEY_SIMPLEX ,  1, 2**16-1, 2, cv2.LINE_AA) 
            save_pars = [fls_iter[i_],x,sx+x,y,sy+y,resc]
            fid.write("\t".join(map(str,save_pars))+'\n')
        fid.close()
        tifffile.imsave(save_file,np.clip(im_base,0,2**16-1).astype(np.uint16))
def get_new_name(dax_fl):
    dax_fl_ = dax_fl
    if not os.path.exists(dax_fl_):
        if dax_fl_.split('.')[-1]=='dax':
            dax_fl_ = dax_fl_.replace('.dax','.dax.zst')
        if dax_fl_.split('.')[-1]=='zst':
            dax_fl_ = dax_fl_.replace('.zst','')
    return dax_fl_
            
def save_tile_image_and_labelV2(fls_iter,save_file,resc=2,custom_frms=None,pix_size_=0.162,
                        max_impose=True,verbose=False,im_cor__=None,max_clip=False,tag_fl='none',
                              rotation_angle=0,add_txt=True,norm_dapi=False,invertX=False,invertY=False,transpose=False):

        import tifffile
        ims,xys=[],[]
        pix_size=pix_size_*resc
        fls_iter_ = fls_iter
        if verbose: fls_iter_ = tqdm(fls_iter)
        for dax in fls_iter_:
            
            dax_ = os.path.dirname(dax)+os.sep+os.path.basename(dax).split('.')[0]+'.dax'
            dic_inf = io.readInfoFile(dax_)
            
            if tag_fl=='npy':
                fr = custom_frms[0]
                dax_ = dax.split('.dax')[0]
                npy_fl = np.sort(glob.glob(dax_+'*.npy'))[fr]
                dapi_im_small = np.load(npy_fl).T
                dapi_im_small = dapi_im_small[1:-1:resc,1:-1:resc]
            else:
                dapi_im = load_im(dax,custom_frms=custom_frms)
                
                dapi_im_small  = np.array(dapi_im)[:,1:-1:resc,1:-1:resc]
                dapi_im_small = np.max(dapi_im_small,0)
            
            if transpose:
                dapi_im_small=dapi_im_small.T
            #Consider adding illumination correction and better stitching
            xys.append([dic_inf['Stage X']/pix_size,dic_inf['Stage Y']/pix_size])
            ims.append(dapi_im_small)
        #return ims
        if im_cor__ is None:
            #im_cor = np.median(ims,axis=0)
            #ims = [im_/im_cor*np.median(im_cor) for im_ in ims]
            
            ims = np.array(ims,dtype=np.float32)
            ims_sort = np.sort(ims,0)
            delta = 10
            ielems = (np.arange(delta,100-delta,delta)*len(ims)/100).astype(int)
            ims_sort_ = ims_sort[ielems]
            vec = np.median(ims_sort_.reshape(ims_sort_.shape[0],-1),-1)
            ima = ims_sort_[0]#np.median(ims_sort_,0)
            imb = (ims_sort_[-1]-ima)/(vec[-1]-vec[0])#[:,np.newaxis,np.newaxis])[-1]*vec[-1]
            im_cor = imb
            ims = [(im_-ima)/imb+np.min(ima/imb) for im_ in ims]
        else:
            im_cor = im_cor__[1:-1:resc,1:-1:resc].copy()
            ims = [im_/im_cor*np.median(im_cor) for im_ in ims]
        xys_=np.array(xys,dtype=int)
        xys_=xys_-np.expand_dims(np.min(xys_,axis=0), axis=0)

        if len(ims[0].shape)>2:
            sx,sy,sz = ims[0].shape
            dim_base = list(np.max(xys_,axis=0)+np.array([sx+1,sy+1]))
            dim_base+=[sz]
        else:
            sx,sy = ims[0].shape
            dim_base = list(np.max(xys_,axis=0)+np.array([sx+1,sy+1]))
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        im_base = np.zeros(dim_base,dtype=float)
        im_compare = np.zeros(dim_base,dtype=float)
        infos_filename = '.'.join(save_file.split('.')[:-1])+'.infos'
        fid = open(infos_filename,'w')
        

        normns = norm_sets(ims,norm_dapi=norm_dapi,perc =80,nl = 30)
        
        
        for i_,(im,(x,y),norm) in enumerate(zip(ims[:],xys_[:],normns)):
            Ix = int((1-2*invertX))
            Iy = int((1-2*invertY))
            im_ = np.swapaxes(im[::Ix,::Iy,...],0,1)
            im_cor_ = np.swapaxes(im_cor[::Ix,::Iy,...],0,1)#[::-1,::-1]
            im_cor_ = im_cor_.astype(np.float32)
            im_cor_ -= np.percentile(im_cor_,1)
            
            im_cor_[im_cor_<0]=0
            
            #im_ = im_[::-1,::-1]
            im_ = im_/norm
            if rotation_angle!=0:
                im_ = rotate(im_, rotation_angle, center=None, scale=1.0)
                im_cor_ = rotate(im_cor_, rotation_angle, center=None, scale=1.0)
            if max_impose:
                #im_base[x:sx+x,y:sy+y,...]=np.max([im_base[x:sx+x,y:sy+y,...],im_],axis=0)
                #im_ind = im_compare[x:sx+x,y:sy+y,...]>im_cor_
                #im_compare[x:sx+x,y:sy+y,...] = np.max([im_compare[x:sx+x,y:sy+y,...],im_cor_],axis=0)
                im_base[x:sx+x,y:sy+y,...]=(im_compare[x:sx+x,y:sy+y,...]*im_base[x:sx+x,y:sy+y,...]+im_cor_*im_)/(im_compare[x:sx+x,y:sy+y,...]+im_cor_)
                im_compare[x:sx+x,y:sy+y,...]=(im_compare[x:sx+x,y:sy+y,...]*im_compare[x:sx+x,y:sy+y,...]+im_cor_*im_cor_)/(im_compare[x:sx+x,y:sy+y,...]+im_cor_)
                
                im_compare[x:sx+x,y:sy+y,...][np.isnan(im_compare[x:sx+x,y:sy+y,...])]=0
                im_base[x:sx+x,y:sy+y,...][np.isnan(im_base[x:sx+x,y:sy+y,...])]=np.nanmedian(im_base[x:sx+x,y:sy+y,...])
                #im_sel[im_ind]#np.mean([im_base[x:sx+x,y:sy+y,...],im_],axis=0)
            else:
                im_base[x:sx+x,y:sy+y,...]=im_
            if add_txt:
            
                txt = os.path.basename(fls_iter[i_]).split('_')[-1].split('.')[0]
                im_base = cv2.putText(im_base, txt, (y+100,x+100), cv2.FONT_HERSHEY_SIMPLEX ,  1, 2**16-1, 2, cv2.LINE_AA) 
            save_pars = [fls_iter[i_],x,sx+x,y,sy+y,resc]
            fid.write("\t".join(map(str,save_pars))+'\n')
        fid.close()
        if max_clip:
            im_base = im_base/np.max(im_base)*(2**16-1)
        tifffile.imsave(save_file,np.clip(im_base,0,2**16-1).astype(np.uint16)) 
       
def save_tile_image_and_label_old(self,fls_iter,save_file,start_frame=1,resc=2,color_map=None,custom_frms=None,target_z=np.arange(-7.5,7.5,0.5),pix_size_=0.153,
                        max_impose=True,verbose=False,correction=True,correction_note='',rotation_angle=0,add_txt=True,norm_dapi=False):

        import tifffile
        ims,xys=[],[]
        pix_size=pix_size_*resc
        fls_iter_ = fls_iter
        if verbose: fls_iter_ = tqdm(fls_iter)
        for dax in fls_iter_:
            

            dic_inf = io.readInfoFile(dax.replace('.dax','.inf'))
            
            #Read appropriate frames
            daxReader = io.DaxReader(dax)
            dapi_im = [daxReader.loadAFrame(frm) for frm in custom_frms]
            
            dapi_im_small  = np.array(dapi_im)[:,1:-1:resc,1:-1:resc]
            
            #Illumination correction:
            im_cor = np.ones(dapi_im_small.shape)
            if correction:
                im_cor = self.correction_image(start_frame,set_fl=fls_iter,perc_=95,save_note=str(start_frame),save_file='auto',overwrite=False)
                im_cor = im_cor[1:-1:resc,1:-1:resc,...].astype(float)
                im_cor = im_cor/np.median(im_cor)
                while len(dapi_im_small.shape)!=len(im_cor.shape): 
                    im_cor = np.expand_dims(im_cor,0)
                dapi_im_small = dapi_im_small/im_cor
            dapi_im_small = np.max(dapi_im_small,0)
            im_cor = np.max(im_cor,0)
            #Consider adding illumination correction and better stitching
            xys.append([dic_inf['Stage X']/pix_size,dic_inf['Stage Y']/pix_size])
            ims.append(dapi_im_small)
            
        
        xys_=np.array(xys,dtype=int)
        xys_=xys_-np.expand_dims(np.min(xys_,axis=0), axis=0)

        if len(ims[0].shape)>2:
            sx,sy,sz = ims[0].shape
            dim_base = list(np.max(xys_,axis=0)+np.array([sx+1,sy+1]))
            dim_base+=[sz]
        else:
            sx,sy = ims[0].shape
            dim_base = list(np.max(xys_,axis=0)+np.array([sx+1,sy+1]))
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        im_base = np.zeros(dim_base,dtype=float)
        im_compare = np.zeros(dim_base,dtype=float)
        infos_filename = '.'.join(save_file.split('.')[:-1])+'.infos'
        fid = open(infos_filename,'w')
        

        normns = norm_sets(ims,norm_dapi=norm_dapi,perc =80,nl = 30)
        
        
        for i_,(im,(x,y),norm) in enumerate(zip(ims[:],xys_[:],normns)):
            #im_ = np.swapaxes(im[::-1,::1,...],0,1)
            im_ = np.swapaxes(im[::1,::-1,...],0,1)
            im_cor_ = np.swapaxes(im_cor[::1,::-1,...],0,1)#[::-1,::-1]
            im_cor_ -= np.min(im_cor_)
            if self.device == 'STORM6':
                im_ = im_[::-1,::-1]
            if self.device == 'STORM6_V2':
                im_ = im_[::-1,::-1]
            im_ = im_/norm
            if rotation_angle!=0:
                im_ = rotate(im_, rotation_angle, center=None, scale=1.0)
                im_cor_ = rotate(im_cor_, rotation_angle, center=None, scale=1.0)
            if max_impose:
                #im_base[x:sx+x,y:sy+y,...]=np.max([im_base[x:sx+x,y:sy+y,...],im_],axis=0)
                #im_ind = im_compare[x:sx+x,y:sy+y,...]>im_cor_
                #im_compare[x:sx+x,y:sy+y,...] = np.max([im_compare[x:sx+x,y:sy+y,...],im_cor_],axis=0)
                im_base[x:sx+x,y:sy+y,...]=(im_compare[x:sx+x,y:sy+y,...]*im_base[x:sx+x,y:sy+y,...]+im_cor_*im_)/(im_compare[x:sx+x,y:sy+y,...]+im_cor_)
                im_compare[x:sx+x,y:sy+y,...]=(im_compare[x:sx+x,y:sy+y,...]*im_compare[x:sx+x,y:sy+y,...]+im_cor_*im_cor_)/(im_compare[x:sx+x,y:sy+y,...]+im_cor_)
                
                im_compare[x:sx+x,y:sy+y,...][np.isnan(im_compare[x:sx+x,y:sy+y,...])]=0
                im_base[x:sx+x,y:sy+y,...][np.isnan(im_base[x:sx+x,y:sy+y,...])]=0
                #im_sel[im_ind]#np.mean([im_base[x:sx+x,y:sy+y,...],im_],axis=0)
            else:
                im_base[x:sx+x,y:sy+y,...]=im_
            if add_txt:
            
                txt = os.path.basename(fls_iter[i_]).split('_')[-1].split('.')[0]
                im_base = cv2.putText(im_base, txt, (y+100,x+100), cv2.FONT_HERSHEY_SIMPLEX ,  1, 10000, 2, cv2.LINE_AA) 
            save_pars = [fls_iter[i_],x,sx+x,y,sy+y,resc]
            fid.write("\t".join(map(str,save_pars))+'\n')
        fid.close()
        tifffile.imsave(save_file,np.clip(im_base,0,2**16-1).astype(np.uint16))
def in_hullF(pt,hull,offset=10**(-10)):
    """
    #checks whether a point pt is within the convex hull hull
    pts = np.random.random([100,3])
    conv = ConvexHull(pts)
    in_hull(pts[29:40],conv)
    """
    if len(hull.equations)==0:
        return np.array([False]*len(pt))
    else:
        surfs = hull.equations
        normal = surfs[:,:-1]
        offset_ = surfs[:,[-1]]
        t_ = np.dot(normal,pt.T)+offset_
        check = np.all(t_<offset,axis=0)
        return check
from tqdm import tqdm_notebook as tqdm
def get_dic_int(dec_fl,celli=None,ncols=3,h_cutoff=2.5,refit =True,plt_val=True):
    
    #load data necessary data
    fov = os.path.basename(dec_fl).split('__decoded')[0]
    data_folder = dec_fl.split(os.sep+'Decoded')[0]
    
    dic_dec = pickle.load(open(dec_fl,'rb'))
    
    dic_cells = dic_dec['cells']
    dic_paramaters = dic_dec['paramaters']
    dic_int={}
    cells = dic_cells.keys()
    
    
    
    if celli is None: celli = range(len(cells))
    for celli_ in celli:
        cell =cells[celli_%len(cells)]

        new_tag_cell = cell+'-__-'+os.path.basename(data_folder)+'-__-'+fov
        #get additional files
        egr1_fls_ = [data_folder+os.sep+'EGR1_fits'+os.sep+fov+'_EGR1_zxyh.npy',
                    data_folder+os.sep+'EGR1_fits'+os.sep+fov+'_cfos_zxyh.npy']
        egr1_fls = [fl for fl in egr1_fls_ if os.path.exists(fl)]
        egr_icols = [icol for icol,fl in zip([1,0],egr1_fls_) if os.path.exists(fl)]
        ##image file





        dic_cell = dic_cells[cell]
        code,scores = dic_cell['code'],dic_cell['scores']
        code= np.array(code)
        cd = code[np.argsort(scores[code-1])][-1]
        col = (cd+1)%2



        X_tight = dic_cells[cell]['zxy_signal_tight']
        X_loose = dic_cells[cell]['zxy_signal_loose']
        X_semiloose = dic_cells[cell].get('zxy_signal_semiloose',[])

        X = X_loose
        if len(X)>0:
            mins,maxs = np.min(X,axis=0),np.max(X,axis=0)+1

            if refit:
                file_signal = dic_paramaters['dic_name'][cd]
                tx,ty=-np.array(dic_paramaters['txys'][cd])
                im = io.DaxReader(file_signal).loadMap()
                im_rd = np.array(im[col::ncols][mins[0]:maxs[0],tx+mins[1]:tx+maxs[1],ty+mins[2]:ty+maxs[2]])

                file_EGR1 = dic_paramaters['dic_name'][0]
                im = io.DaxReader(file_EGR1).loadMap()
                zxyhs = []
                for iEGR in egr_icols:
                    im_EGR1 = np.array(im[iEGR::ncols][mins[0]:maxs[0],mins[1]:maxs[1],mins[2]:maxs[2]])


                    im_EGR1_ = im_EGR1/np.median(im_EGR1)-1
                    im_rd_ = im_rd/np.median(im_rd)-1
                    fr__ = 0
                    kp = im_rd_>1
                    if np.sum(kp)>0:
                        fr__ = np.median(im_EGR1_[kp])/np.median(im_rd_[kp])
                    im_EGR1__ = im_EGR1_ - im_rd_*fr__*1.5+1
                    zxyh_0 = get_local_max(im_EGR1__,th_fit=h_cutoff)
                    zxyh = np.array(zxyh_0)
                    zxyh_0 = np.array(zxyh_0)
                    zxyh[:3] = zxyh[:3]+mins[np.newaxis,:].T
                    zxyhs.append(zxyh)
                    if plt_val:
                        xyc = {}
                        for tag_,X_ in [('loose',X_loose),('tight',X_tight)]:
                            X__ = X_-mins
                            convH2D = ConvexHull_(X__[:,1:])
                            xc,yc = X__[:,1:][convH2D.vertices].T
                            if len(xc)>0:
                                xc=list(xc)+[xc[0]]
                                yc=list(yc)+[yc[0]]
                            xyc[tag_] = [yc,xc]
                        plt.figure()
                        plt.plot(xyc['loose'][0],xyc['loose'][-1],'r-')
                        plt.plot(xyc['tight'][0],xyc['tight'][-1],'b-')
                        plt.imshow(np.max(im_rd_+1,axis=0),vmax=3,cmap='gray')
                        plt.title('Readout: Drift x,y:'+str([tx,ty]))
                        plt.show()

                        plt.figure()
                        plt.plot(xyc['loose'][0],xyc['loose'][-1],'r-')
                        plt.plot(xyc['tight'][0],xyc['tight'][-1],'b-')
                        plt.imshow(np.max(im_EGR1_+1,axis=0),vmax=3,cmap='gray')
                        plt.title('Unmodified EGR1')
                        plt.show()

                        plt.figure()
                        plt.plot(xyc['loose'][0],xyc['loose'][-1],'r-')
                        plt.plot(xyc['tight'][0],xyc['tight'][-1],'b-')
                        plt.plot(zxyh_0[2],zxyh_0[1],'gx')
                        plt.imshow(np.max(im_EGR1__,axis=0)/np.median(im_EGR1__),vmax=3,cmap='gray')
                        plt.title('Subtracted EGR1 and fits')
                        plt.show()
            else:
                zxyhs = [np.load(egr1_fl) for egr1_fl in egr1_fls]
                
            zxyh__s=[]
            for zxyh in zxyhs:
                zxyh_ = zxyh[:,zxyh[3,:]>h_cutoff].T
                in_box = np.all((zxyh_[:,:3]>=mins)&(zxyh_[:,:3]<=maxs),-1)
                zxyh__ = zxyh_[in_box]
                zxyh__s.append(zxyh__)

            dic_int[new_tag_cell]={'fov':fov}

            for X,tag_ in [[X_loose,'_loose'],[X_tight,'_tight'],[X_semiloose,'_semiloose']]:
                if len(X)>0:
                    convH = ConvexHull_(X)
                    verts = X[convH.vertices]
                    dic_int[new_tag_cell]['OR_convHull'+tag_],dic_int[new_tag_cell]['OR_volume'+tag_]=verts,convH.volume
                    for i__,zxyh__ in enumerate(zxyh__s):
                        inH = in_hullF(zxyh__[:,:3],convH)
                        if len(zxyh__)>0:
                            zxyh__kp = zxyh__[inH]
                        else:
                            zxyh__kp = zxyh__
                        num_pts = len(zxyh__kp)
                        icol_EGR = egr_icols[i__]
                        tag_EGR = ['cfos','EGR1'][icol_EGR]
                        dic_int[new_tag_cell]['num_pts_'+tag_EGR+tag_],dic_int[new_tag_cell]['zxyh_'+tag_EGR+tag_]=num_pts,zxyh__kp
                    
                    
            extra_keys = ['code','olfr','sublibrary','mn_std_skew_kurt_nopix_bk','mn_std_skew_kurt_nopix','scores','MOE']
            for key in extra_keys:
                dic_int[new_tag_cell][key] = dic_cell[key]

            image_fl = data_folder+os.sep+r'Decoded'+os.sep+fov+os.sep+'good'+os.sep+dic_cell['olfr']+'_'+cell+'.png'
            dic_int[new_tag_cell]['fl_check'] = image_fl
            dic_int[new_tag_cell]['paramaters'] = dic_paramaters
            dic_int[new_tag_cell]['dec_fl'] = dec_fl
            dic_int[new_tag_cell]['celli'] = celli_
            if plt_val:
                pass
                #!{'"'+image_fl+'"'}
    return dic_int


def simplify_txys(txy_dic,ref_key):
    txy_dic_ = {}
    
    for key in txy_dic.keys():
        x1,x2 = txy_dic[key]
        if np.array(x1).ndim>0:
            txy_dic_[key] = np.median(x1,0)
        else:
            txy_dic_[key] = txy_dic[key]
            
    ref = txy_dic_[ref_key]#self.ref_fl
    for key in txy_dic_:
        txy_dic_[key]=txy_dic_[key]-ref
    return txy_dic_
    
def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]
    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)
    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    # return the rotated image
    return rotated
def flatten(list_):
    return [item for sublist in list_ for item in sublist]
def is_done(or_chr,ind):
    base_fl = os.path.basename(or_chr.files[ind][0]).replace('.dax','.finished')
    finish_file = or_chr.save_folder+os.sep+'Finished_mark'+os.sep+base_fl
    return os.path.exists(finish_file)
def gauss_div1(im_,gb=100):
    im_g2 = np.array([cv2.blur(im__,(gb,gb))for im__ in im_])
    im_ratio = im_/im_g2
    return im_ratio
def get_local_max_old(im_dif,th_fit):
    z,x,y = np.nonzero(im_dif>th_fit)
    zmax,xmax,ymax = im_dif.shape
    keep = (z>0)&(z<zmax-1)&(x>0)&(x<xmax-1)&(y>0)&(y<ymax-1)
    z,x,y = z[keep],x[keep],y[keep]
    in_im = im_dif[z,x,y]
    keep = (in_im>im_dif[z+1,x,y])&(in_im>im_dif[z-1,x,y])
    z,x,y = z[keep],x[keep],y[keep]
    in_im = in_im[keep]
    keep = (in_im>im_dif[z,x+1,y])&(in_im>im_dif[z,x-1,y])
    z,x,y = z[keep],x[keep],y[keep]
    in_im = in_im[keep]
    keep = (in_im>im_dif[z,x,y+1])&(in_im>im_dif[z,x,y-1])
    z,x,y = z[keep],x[keep],y[keep]
    in_im = in_im[keep]
    return z,x,y,in_im
def get_local_max_old_v2(im_dif,th_fit,delta=2):
    z,x,y = np.nonzero(im_dif>th_fit)
    zmax,xmax,ymax = im_dif.shape
    #keep = (z>delta)&(z<zmax-delta)&(x>delta)&(x<xmax-delta)&(y>delta)&(y<ymax-delta)
    #z,x,y = z[keep],x[keep],y[keep]
    in_im = im_dif[z,x,y]
    keep = np.ones(len(x))>0
    for d1 in range(-delta,delta+1):
        for d2 in range(-delta,delta+1):
            for d3 in range(-delta,delta+1):
                keep &= (in_im>=im_dif[(z+d1)%zmax,(x+d2)%xmax,(y+d3)%ymax])
    z,x,y = z[keep],x[keep],y[keep]
    in_im = in_im[keep]
    return z,x,y,in_im
    
def get_local_max(im_dif,th_fit,delta=2,delta_fit=0,dbscan=True):
    """Given a 3D image <im_dif> as numpy array, get the local maxima in cube -<delta>_to_<delta> in 3D.
    Optional a dbscan can be used to couple connected pixels with the same local maximum. 
    This is important if saturating the camera values.
    
    Returns: Xh - a list of z,x,y and brightness of the local maxima
    """
    z,x,y = np.nonzero(im_dif>th_fit)
    zmax,xmax,ymax = im_dif.shape
    in_im = im_dif[z,x,y]
    keep = np.ones(len(x))>0
    for d1 in range(-delta,delta+1):
        for d2 in range(-delta,delta+1):
            for d3 in range(-delta,delta+1):
                keep &= (in_im>=im_dif[(z+d1)%zmax,(x+d2)%xmax,(y+d3)%ymax])
    z,x,y = z[keep],x[keep],y[keep]
    h = in_im[keep]
    Xh = np.array([z,x,y,h]).T
    if dbscan and len(x)>0:
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=1, min_samples=1, metric='euclidean')
        X = np.array([z,x,y]).T
        db.fit(X)
        l = db.labels_
        Xh = []
        for l_ in np.unique(l):
            keep = l==l_
            imax = np.argmax(h[keep])
            Xh.append([z[keep][imax],x[keep][imax],y[keep][imax],h[keep][imax]])
        Xh = np.array(Xh)
    
    if delta_fit!=0:
        z,x,y,h = Xh.T
        z,x,y = z.astype(int),x.astype(int),y.astype(int)
        im_centers = [[],[],[],[]]
        for d1 in range(-delta_fit,delta_fit+1):
            for d2 in range(-delta_fit,delta_fit+1):
                for d3 in range(-delta_fit,delta_fit+1):
                    if (d1*d1+d2*d2+d3*d3)<=(delta_fit*delta_fit):
                        im_centers[0].append((z+d1))
                        im_centers[1].append((x+d2))
                        im_centers[2].append((y+d3))
                        im_centers[3].append(im[(z+d1)%zmax,(x+d2)%xmax,(y+d3)%ymax])

        im_centers_ = np.array(im_centers)
        im_centers_[-1] -= np.min(im_centers_[-1],axis=0)
        zc = np.sum(im_centers_[0]*im_centers_[-1],axis=0)/np.sum(im_centers_[-1],axis=0)
        xc = np.sum(im_centers_[1]*im_centers_[-1],axis=0)/np.sum(im_centers_[-1],axis=0)
        yc = np.sum(im_centers_[2]*im_centers_[-1],axis=0)/np.sum(im_centers_[-1],axis=0)
        Xh = np.array([zc,xc,yc,h]).T
    return Xh.T
def apply_warp(im2,warp_matrix):
    im2_aligned = cv2.warpAffine(im2, warp_matrix, (im2.shape[1],im2.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return im2_aligned
def get_best_rigid_im(im10,im20,gb=25,number_of_iterations = 100,termination_eps = 1e-5,plt_val=False):
    im1 = cv2.blur(im10,(1,1))-cv2.blur(im10,(gb,gb))#[:1024,:1024]
    im2 = cv2.blur(im20,(1,1))-cv2.blur(im20,(gb,gb))#[:1024,:1024]
    im1/=np.std(im1)
    im2/=np.std(im2)
    
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    # Define the motion model
    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1,im2,warp_matrix, warp_mode, criteria,None,1)
    if plt_val:
        im2_aligned = apply_warp(im2,warp_matrix)
        
        f,axs = plt.subplots(1,2,sharex=True,sharey=True)
        axs[0].imshow(im1,vmin=-2,vmax=2)
        axs[1].imshow(im2_aligned,vmin=-2,vmax=2)
        plt.show()
    return warp_matrix
    
    
def plot_decoded_cell_3d_v2(self,conv,std_th=20,transpose=False,draw_cont=True,rearange=None,save_file=None,**imshow_vars):
    """This plots the decoded image across bits extractiong information from self.ims_matrix
    Use as: self.plot_decoded_cell(self.decoded_dic['cells'][somekey],std_th=20,interpolation='nearest',cmap=cm.hot)"""

    
    olfr = conv.olfr
    cd = conv.code
    zm,xm,ym = conv.m.astype(int)
    zM,xM,yM = conv.M.astype(int)
    
    if not hasattr(self,'dic_min_max'):
        get_vmins_vmax(self,fr=1.25)
    
    ims_plt=[]
    edge_col = 0.25
    def xyz_project(im_3d,func=np.max):
        if func is None:
            im_plt_xy = im_3d[int(im_3d.shape[0]/2),...]
            im_plt_xz = im_3d[:,int(im_3d.shape[1]/2),...]
            #im_plt_xy=im_3d[int(np.mean(z_big)-zm),...]
            #im_plt_xz=im_3d[:,int(np.mean(y_big)-ym),...]
        else:
            im_plt_xy = func(im_3d,axis=0)
            im_plt_xz = func(im_3d,axis=1)
        #rep = np.arange(0,im_plt_norm.shape[0],self.nm_per_pixel_xy/float(self.nm_per_pixel_z))
        #rep = np.array(rep,dtype=int)
        #im_plt_xz=im_plt_xz[rep,...]
        pad_t = np.ones([1,im_plt_xz.shape[1]])*edge_col
        #print [im_plt_xy.shape,pad_t.shape,im_plt_xz.shape],im_3d.shape
        im_plt = np.concatenate([im_plt_xy,pad_t,im_plt_xz],axis=0)
        return im_plt
    for i in range(len(self.ims_matrix)):
        base_im = np.array(self.ims_matrix[i][zm:zM:,xm:xM,ym:yM])#.copy()
        if transpose:
            base_im = np.swapaxes(base_im,1,2)
        vminmax = self.dic_min_max.get(i,None)
        if vminmax is None:
            im_plt = base_im-np.mean(base_im)
            im_plt_norm = im_plt/np.std(im_plt)
            im_plt_norm = gt.minmax(im_plt_norm,max_=std_th)
        else:
            vmin,vmax=vminmax
            im_plt_norm =np.clip((base_im-vmin)/(vmax-vmin),0,1)
        im_plt = xyz_project(im_plt_norm,func=np.max)
        im_plt = gt.pad_im(im_plt,pad_=3,pad_val_=1 if (i+1 in cd) else edge_col)
        ims_plt.append(im_plt)
    #im_plt = xyz_project(im_mask[:-1,:-1,:-1],func=np.mean)
    #ims_plt.append(gt.pad_im(im_plt,pad_=3,pad_val_=edge_col))
    if 'DAPI0' in self.dic_fov:
        im_plt_3d = gt.minmax(self.dic_fov['DAPI0'][zm:zM:,xm:xM,ym:yM])
        im_plt = xyz_project(im_plt_3d,func=None)
        ims_plt.append(gt.pad_im(im_plt,pad_=3,pad_val_=edge_col))
    #return ims_plt
    ims_plt2 = [np.dstack([(img*255).astype(np.uint8)]*3) for img in ims_plt]
    if draw_cont:
        best_code=(conv.code-1)[np.argmax(conv.scores[conv.code-1])]
        X1 = get_2d_line(conv,axs=[1,2])-conv.m[[1,2]]+3
        X2 = get_2d_line(conv,axs=[2,0])-conv.m[[2,0]]+[3,np.max(X1[:,0])]
        cv2.polylines(ims_plt2[best_code],np.array([X1[:,::-1]],dtype=np.int32),True,color=[0,255,0])
        cv2.polylines(ims_plt2[best_code],np.array([X2[:,::1]],dtype=np.int32),True,color=[0,255,0]);
    
    fig = plt.figure(figsize=(12,12))
    
    if rearange is None:
        plt.imshow(gt.stitch(ims_plt2),**imshow_vars)
    else:
        plt.imshow(gt.stitch(np.array(ims_plt2)[rearange]),**imshow_vars)
    
    plt.title(olfr+'_'+str(cd))
    if save_file is None:
        plt.show()
        #plt.close(fig)
    else:
        fig.savefig(save_file)
        plt.close(fig)
        
def load_im_set(fl,custom_frms=None,dims=(3200,3200)):
    import blosc
    fls = fl_to_zstset(fl)
    if custom_frms is None: fls_ = fls
    else: fls_ = fls[custom_frms]
    im = np.array([np.frombuffer(blosc.decompress(open(fl,'rb').read()),dtype=np.uint16).reshape(list(dims)) 
         for fl in fls_])
    print(im.shape)
    return im
def fl_to_zstset(fl):
    fls_set = glob.glob(os.path.dirname(fl)+os.sep+os.path.basename(fl).split('.')[0]+'*.dax.zst*')
    fls_set = np.array([fl for fl in fls_set if fl.split('.dax.zst')[-1]])
    return fls_set[np.argsort([int(fl.split('.dax.zst')[-1]) for fl in fls_set])]
def load_im(dax_fl,custom_frms=None,dims=(2048,2048),cast=True):
    dax_fl_ = dax_fl
    if dax_fl_.split('.')[-1]=='npy':
        return np.load(dax_fl_)#np.swapaxes(np.load(dax_fl_),-1,-2)
    
    if '.dax' not in dax_fl_:
        return load_im_set(dax_fl_,custom_frms,dims=(3200,3200))
    #glob.glob(fl.split('.')[0]+'*.dax*')[0::4]

    
    dax_fl_ = get_new_name(dax_fl)
            
    
    ### patch missed frames
    master_folder = os.path.dirname(dax_fl)
    bucket_fl =master_folder+os.sep+'buckets.npy'
            
    if dax_fl_.split('.')[-1]=='dax':
        if custom_frms is not None and not os.path.exists(bucket_fl):
            daxReader = io.DaxReader(dax_fl_)
            im = np.array([daxReader.loadAFrame(frm) for frm in custom_frms])
        else:
            im = io.DaxReader(dax_fl).loadMap()
            #im = np.swapaxes(np.frombuffer(open(dax_fl_,'rb').read(),dtype=np.uint16).reshape([-1]+list(dims)),1,2)
    else:
        import blosc
        #im = np.swapaxes(np.frombuffer(blosc.decompress(open(dax_fl_,'rb').read()),dtype=np.uint16).reshape([-1]+list(dims)),1,2)
        im = np.frombuffer(blosc.decompress(open(dax_fl_,'rb').read()),dtype=np.uint16).reshape([-1]+list(dims))
        if custom_frms is not None and not os.path.exists(bucket_fl):
            im = np.array([im[frm] for frm in custom_frms if frm<len(im)])
    
    
    
    ### patch missed frames
    if os.path.exists(bucket_fl):
        buckets = np.load(bucket_fl)
        buckets = [list(b) for b in buckets]
        ires = 10
        for iim in range(len(im)):
            cors = [np.corrcoef(im[iim,::ires,::ires].ravel(),im[b[-1],::ires,::ires].ravel())[0,1] for b in buckets]
            best = np.argmax(cors)
            if iim>buckets[best][-1]:
                buckets[best]+=[iim]
        #buckets = [b[1:]for b in buckets]
        len_ = np.min([len(b) for b in buckets])
        im = im[[b[l] for l in range(len_) for b in buckets]]
        if custom_frms is not None:
            im = im[[frn for frm in custom_frms if frm<len(im)]]
    return im   
class OR_cropper:
    """
    
    Notes:
    Press A,D to navigate between fields of view. 
    
    """
    def internalize_paramaters(self):
        """This internalizes paramaters into self from paramater_dic keeping names"""
        if not hasattr(self,'paramater_dic'):
            self.paramater_dic = {'device':'STORM6',#misc paramaters
                                  'cell_diameter':55,#diamater of cell in xy camera pixels
                                  'nm_per_pixel_xy':153,#pixel size in nm
                                  'nm_per_pixel_z':500,#z step size of stage in nm
                                  'perc_baseline':0.1,'bleed_factor':10,'hybeindex':[7,7,2,2,3,3,4,4,5,5,6,6,1,1,0,0],#paramaters for bleedthrough
                                  'ref_fl':0,"reg_frame":3,"local_mean_size":50,#drif correction paramaters -> self.correct_drift
                                  'lib_remap':[1,2,3,4,5,7,8,9,10,6],#this is to correct a naming error -> self.cross_corrs_decode
                                  'lib_fl':r'C:\Data\Genomes\MouseLibraries\SI8.fasta',#library used to decode cells #change this to csv format
                                  'nRs':14,'nOn':4,'pad_xy':25,'pad_z':8,'spacing_xy':10,'spacing_z':3,'target_z':np.arange(-7.5,7.5,0.5),#correlation paramaters -> self.cross_corrs_decode
                                  'th_bk':3.5, #units of standard deviation above the mean to get called as a valuable pixel in (decoded - background) -> self.decoded_main
                                  'th_dec':0.6, #min cross-corr -> self.decoded_main
                                  'dbscan_corr_eps':4, #around the cell diamater in reduced coordinates -> self.decoded_main
                                  'dbscan_corr_min_samples':20,#patches to accept clusters -> self.decoded_main
                                  'background_threshold':0.4,#threshold background -> self.decoded_background
                                  'EGR1_start_frame':1,'EGR1_rep_frame':3,#indicate how to extract the EGR1 signal from the dax file provided in EGR1_fit
                                  'EGR1_DBSCAN_eps':27, 'EGR1_DBSCAN_min_samples':7, #Dbscan paramaters after fit from _alist.bin
                                  'EGR1_h_min':4000, #Threshold localisations to keep only the true EGR1 singals befor clustering
                                  'EGR1_h_plot':12000,#Plotting threshold - not required for analysis
                                  'EGR1_h_max_filter':30000,'EGR1_dist_cut_filter':20,'EGR1_fr_cut_filter':0.5,'EGR1_nbad_filter':500,#Paramaters used to filter out super-hot speckles.
                                  'EGR1_min_samples_Xmeans':3,'EGR1_dist_cut_Xmeans':60,#Xmeans clustering paramaters to segment cells.
                                  'EGR1_correction':True,#Apply flat field image correction
                                  'EGR1_refit':True,#Force refit
                                  'EGR1_force_redo':False}#Force redo analysis
        for key in self.paramater_dic.keys():
            setattr(self,key,self.paramater_dic[key])
    def load_data(self,data_folder,save_folder=None,force_remap = False):
        """This populates self.files with a list of lists grouped by fov"""
        self.internalize_paramaters()
        self.data_folder=data_folder
        if save_folder is None:
            data_folder_ = data_folder
            if type(data_folder)!=str: data_folder_ = data_folder[0]
            self.save_folder=os.path.dirname(data_folder_)+os.sep+'Analysis'
        else:
            self.save_folder=save_folder
         #make save_folder if it does not exist
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        #Deal with mapping files
        files_file = self.save_folder+os.sep+'files_map.pkl'
        
        if os.path.exists(files_file):
            self.files = pickle.load(open(files_file,'rb'))
            max_len = np.max(map(len,self.files))#patch missing pieces
            self.files = [fls for fls in self.files if len(fls)==max_len]
            #check the existence of first and last file
            if not os.path.exists(self.files[0][0]) or not os.path.exists(self.files[-1][-1]):
                force_remap = True
        else:
            force_remap = True
        if force_remap:
            files = []
            data_folders = [data_folder] if type(data_folder)==str else data_folder
            for data_folder_ in data_folders:
                #files += list(gt.listDax(data_folder_,'H*'))
                files +=glob.glob(data_folder_+os.sep+r'H*/*.dax*')
            if 'zst1' in [os.path.basename(fl).split('.')[-1] for fl in files]:
                files = np.unique([os.path.dirname(fl)+os.sep+os.path.basename(fl).split('.')[0] for fl in files])
            map_ = map(os.path.basename,files)
            map_ = [os.path.basename(fl).split('.')[0]for fl in files]
            self.files = gt.partition_map(files,map_)
            #max_len = np.max(map(len,self.files))#patch missing pieces
            max_len = len(np.unique([os.path.dirname(fl) for fl in files]))
            self.files = [fls for fls in self.files if len(fls)==max_len]
            pickle.dump(self.files,open(files_file,'wb'),protocol=pickle.HIGHEST_PROTOCOL)
        def sortHs(fls):
            tags = [os.path.basename(os.path.dirname(fl)) for fl in fls]
            return np.array(fls)[np.argsort(tags)]
        self.files = [sortHs(fls) for fls in self.files]
        #initialize paramaters
        #self.index_fov=0
        
    def load_index_rough(self):
        """Modified load to deal with rough alignment"""
        print "Running rough alignment."
        self.get_rough_alignment(box_cutoff=50,fine_align=50,neigh_cutoff=300,plt_val=False)
        self.internalize_paramaters()
        self.fov_name = self.basenames[self.index_fov].replace('.zst','')
        self.dic_name = {}
        self.drif_corrected = True
        dic={}

        for name_base in self.dirnames:
            print "Dealing with: "+name_base
            ref_slices_cor = self.rough_alignment_dics[name_base]['ref_slices_cor']
            target_slices_cor = self.rough_alignment_dics[name_base]['target_slices_cor']
            fls_inters = self.rough_alignment_dics[name_base]['fls_inters']
            sx,sy,sz = io.DaxReader(fls_inters[0]).filmSize()
            reconstr_im = np.zeros([sz,sx,sy])
            for ref_slice,target_slice,fl in zip(ref_slices_cor,target_slices_cor,fls_inters):
                (minx,maxx),(miny,maxy) = ref_slice
                (minx_,maxx_),(miny_,maxy_) = target_slice
                im_temp = io.DaxReader(fl).loadAll()
                reconstr_im[:,minx:maxx,miny:maxy]=im_temp[:,minx_:maxx_,miny_:maxy_]
            im_cy5 = reconstr_im[1::3]
            im_cy3 = reconstr_im[2::3]
            im_dapi = reconstr_im[3::3]
            namesR = name_base.split(';')[0].split(',')
            if namesR[0][0]=='R': 
                cy3_i = int(namesR[0][1:])
                cy5_i = int(namesR[1])
                dic[cy3_i]=im_cy3
                dic[cy5_i]=im_cy5
                dic['DAPI'+str(cy3_i)]=im_dapi
                dic['DAPI'+str(cy5_i)]=im_dapi
                self.dic_name[cy3_i]=fl
                self.dic_name[cy5_i]=fl
            elif namesR[0]=='EGR1':
                dic['background0']=im_cy3
                dic[0]=im_cy5
                dic['DAPI0']=im_dapi
                self.dic_name[0]=fl
                self.dic_name['background0']=fl
        self.dic_fov = dic

    def load_index(self,memmap=False,correct_drift=True): 
        """
        loads the files self.files[self.index_fov]
        Construct a dictionary where dic[int_] is the image of a field of view corresponding to bit int_ (from 1-14)
        Additionaly it also populates dic[0] as EGR1 and dic[dapi<i>] as the dapi image and dic[background0].
        """
        self.internalize_paramaters()
        if getattr(self,'rough_align',False):
            self.load_index_rough()
            return None
        fls = self.files[self.index_fov]
        self.fov_name = os.path.basename(fls[0]).replace('.zst','')
        self.dic_name = {}
        self.drif_corrected = False
        dic={}
        for fl in fls:
            if fl.split('.')[-1]=='dax':
                if memmap:
                    im = np.swapaxes(io.DaxReader(fl).loadMap(),1,2)
                else:
                    im = io.DaxReader(fl).loadAll()
            elif '.dax' not in fl:
                im = load_im_set(fl)
                im = np.swapaxes(im,1,2)
            elif fl.split('.')[-1]=='zst':
                info_fl = fl.replace('.dax.zst','.inf').replace('.dax','.inf')
                dims = [ln[:-1].split('=')[-1].split('x') 
                for ln in open(info_fl,'r') if 'dimensions' in ln][0]
                dims = np.array(dims,dtype=int)
                import blosc
                im = np.frombuffer(blosc.decompress(open(fl,'rb').read()),dtype=np.uint16).reshape([-1]+list(dims))
                im = np.swapaxes(im,1,2)
                #np.from blosc.decompress(open(fl,'rb').read())
            
            
            print "Loading: "+fl
            
            if '3col' in self.device.lower():
                if 'megafish' in self.device.lower():
                    print('Here-mega')
                    im_750 = im[0::4]
                    im_cy5 = im[1::4]
                    im_cy3 = im[2::4]
                    im_dapi = im[3::4]

                else:
                    im_750 = im[2::4]
                    im_cy5 = im[1::4]
                    im_cy3 = im[0::4]
                    im_dapi = im[3::4]
            else:
                if self.device.upper() == 'STORM6':
                    #STORM6 - standard
                    max_ = int((len(im)-10)/3)
                    im_cy5 = im[1::3][4:max_]
                    im_cy3 = im[2::3][4:max_]
                    im_dapi = im[3::3][4:max_]
                elif self.device.upper() == 'STORM6_BLANK':
                    max_ = int((len(im)-10)/4)
                    im_cy5 = im[0::4][4:max_]
                    im_cy3 = im[1::4][4:max_]
                    im_dapi = im[2::4][4:max_]
                elif self.device.upper() == 'STORM6_V2':
                    im_cy5 = im[1::3]
                    im_cy3 = im[0::3]
                    im_dapi = im[2::3]
                elif self.device.upper() == 'STORM3':
                    #STORM3 - standard
                    im_cy5 = im[0::3,:256,:256]
                    im_cy3 = im[1::3,:256,256:]
                    im_dapi = im[2::3,256:,256:]
                elif self.device.upper() == 'STORM65':
                    #STORM6.5 - standard
                    im_cy5 = im[1::3]
                    im_cy3 = im[0::3]
                    im_dapi = im[2::3]
                elif self.device.upper() == 'MERFISH3':
                    im_cy3 = im[:48]
                    im_cy5 = im[51:99][::-1]
                    im_dapi = np.array([im[101]]*48)
            #decide on how to populate the dictionary
            
            name_base = fl.split(os.sep)[-2]
            if 'H' in name_base: name_base=name_base[2:]
            namesR = name_base.split(';')[0].split(',')
            if '3col' in self.device.lower():
                R_to_col={1:'750',2:'647',3:'561',4:'750',6:'647',5:'561',7:'750',8:'647',9:'561',10:'750',12:'647',11:'561',14:'750',15:'647',13:'561'}
                if getattr(self,'replaceR14',False):
                    R_to_col={1:'750',2:'647',3:'561',4:'750',6:'647',5:'561',7:'750',8:'647',9:'561',10:'750',12:'647',11:'561',14:'647',15:'647',13:'561'}
                
                
                if 'EGR1' in name_base.upper() and 'R15' in name_base.upper():
                    self.replaceR14=True
                    print("here")
                    dic['background0']=im_cy3
                    dic[0]=im_750
                    dic['DAPI0']=im_dapi
                    self.dic_name[0]=fl
                    self.dic_name['background0']=fl
                    iR=15
                    dic['DAPI'+str(iR)]=im_dapi
                    dic[int(iR)]=im_cy5
                    self.dic_name[int(iR)]=fl
                elif 'EGR1' in namesR[0]: 
                    dic['background0']=im_cy3
                    dic[0]=im_cy5
                    dic['DAPI0']=im_dapi
                    self.dic_name[0]=fl
                    self.dic_name['background0']=fl
                elif 'cfos' in namesR[0].lower():
                    print('tdtom')
                    if 'megafish' in self.device.lower():
                        dic['background0']=im_cy3
                        dic[0]=im_cy5
                        dic['DAPI0']=im_dapi
                        self.dic_name[0]=fl
                        self.dic_name['background0']=fl
                    else:
                        dic['background0']=im_cy3
                        dic[0]=im_750
                        dic['DAPI0']=im_dapi
                        self.dic_name[0]=fl
                        self.dic_name['background0']=fl
                else:
                    name_base = name_base.replace('B','-1')
                    rinds = np.array(name_base[1:].split('_')[0].split(';')[0].split(','),dtype=int)
                    cols = ['561','647','750']
                    ims = [im_cy3, im_cy5, im_750]
                    for iR in rinds:
                        if iR>0:
                            dic['DAPI'+str(iR)]=im_dapi
                            icol = cols.index(R_to_col[iR])
                            dic[int(iR)]=ims[icol]
                            self.dic_name[int(iR)]=fl
            else:
                
                if namesR[0][0]=='R' and namesR[0][1].isdigit() and len(namesR)>1: 
                    cy3_i = int(namesR[0][1:])
                    cy5_i = int(namesR[1])
                    dic[cy3_i]=im_cy3
                    dic[cy5_i]=im_cy5
                    dic['DAPI'+str(cy3_i)]=im_dapi
                    dic['DAPI'+str(cy5_i)]=im_dapi
                    self.dic_name[cy3_i]=fl
                    self.dic_name[cy5_i]=fl
                if namesR[0][0]=='B' and len(namesR)>1: 
                    cy5_i = int(namesR[1][1:])
                    dic[cy5_i]=im_cy5
                    dic['DAPI'+str(cy5_i)]=im_dapi
                    self.dic_name[cy5_i]=fl
                elif namesR[0][0]=='R' and namesR[0][1].isdigit() and len(namesR)==1: 
                    cy5_i = int(namesR[0][1:])
                    dic[cy5_i]=im_cy5
                    dic['DAPI'+str(cy5_i)]=im_dapi
                    self.dic_name[cy5_i]=fl
                elif namesR[0]=='EGR1' or namesR[0]=='RP2':
                    dic['background0']=im_cy3
                    dic[0]=im_cy5
                    dic['DAPI0']=im_dapi
                    self.dic_name[0]=fl
                    self.dic_name['background0']=fl
                elif namesR[0]=='blank':
                    dic[15]=im_cy3
                    dic[16]=im_cy5
                    dic['DAPI15']=im_dapi
                    dic['DAPI16']=im_dapi
                    self.dic_name[15]=fl
                    self.dic_name[16]=fl
        self.dic_fov = dic
        print self.dic_fov.keys()
        sz,sx,sy = self.dic_fov[self.dic_fov.keys()[0]].shape
        self.sz_image,self.sx_image,self.sy_image = sz,sx,sy
        if correct_drift:
            self.correct_drift_smallrot() #
        else:
            self.txys = {}
        print "Loaded and drift corrected images!"
        
    def correct_drift_smallrot(self):
        """Corrects the drift and updates self.dic_fov"""
        fl_ref = self.dic_name[1]
        fls = np.unique([self.dic_name[key] for key in self.dic_name])
        Xt_dic = {fl:self.register_files_smallrot(fl_ref,fl) for fl in tqdm(fls)}
        self.txys={}
        for key in tqdm(self.dic_name):#dic_fov
            R=key
            if 'DAPI' in str(key): R = int(key[4:])
            fl = self.dic_name[R]
            X_t_ = Xt_dic[fl]
            self.txys[key] = X_t_
            self.dic_fov[key] = apply_X_t(self.dic_fov[key],X_t_)
            self.applied_transformation=True
    def register_files_smallrot(self,fl0,fl1,reg_frame=None,im_msk=None,sG=10,sg=2,th_cut= 2,ssq=300,
                                niter = 20,fr=0.95, target_distance=0.5,
                                verbose=False):
        """
        Given two dax files fl1 and fl2 this loads the reg_frame
        It normalizes the image files using two gaussin filters <sg size filter> /<sG size filter>
        To avoid hot pixel problems the normalized images are capped at +/-th_cut
        
        It tiles the images with squares of size ssq that have non-zero im_msk
        
        It then computes the registration between the frames using fourier cross-correlation.
        It returns x,y such that gt.translate(im2,[xt,yt]) is aligned with im1
        """
        import cv2
        from imreg_dft import imreg
        
        if im_msk is None:
            #self.loadMOEmasks()
            im_msk = self.im_MOEmask(self.fov_name)
        if reg_frame is None:
            ncols = 4 if '3col' in self.device else 3
            nfrs = io.readInfoFile(fl0)['number_frames']
            reg_frame = int(nfrs/ncols/2+1)*ncols-1
        
        
            if fl0.split('.')[-1]=='dax':
                im0 = io.DaxReader(fl0).loadAFrame(reg_frame).astype(np.float32)#laod first dapi frame
            elif '.dax' not in fl0:
                im0 = load_im_set(fl0,[reg_frame])[0].astype(np.float32)
            else:
                fl0_ = glob.glob(fl0.split('.dax')[0]+'_fr*')[-1]
                im0 = np.load(fl0_).T.astype(np.float32)
            if fl1.split('.')[-1]=='dax':
                im1 = io.DaxReader(fl1).loadAFrame(reg_frame).astype(np.float32)#laod first dapi frame
            elif '.dax' not in fl1:
                im1 = load_im_set(fl1,[reg_frame])[0].astype(np.float32)
            else:
                fl1_ = glob.glob(fl1.split('.dax')[0]+'_fr*')[-1]
                im1 = np.load(fl1_).T.astype(np.float32)
        #correct using blur
        im0_50 = cv2.GaussianBlur(im0,(0,0),sigmaX=sG)
        im1_50 = cv2.GaussianBlur(im1,(0,0),sigmaX=sG)
        
        im0_sc= cv2.GaussianBlur(im0,(0,0),sigmaX=sg)/im0_50
        im1_sc= cv2.GaussianBlur(im1,(0,0),sigmaX=sg)/im1_50
        im0_sc=im0_sc-im0_sc.mean()
        im1_sc=im1_sc-im1_sc.mean()
        
        im1_sc[im1_sc>th_cut]=th_cut
        im0_sc[im0_sc>th_cut]=th_cut
        im0_sc[im0_sc<-th_cut]=-th_cut
        im1_sc[im1_sc<-th_cut]=-th_cut

        lims = get_list_limits(self,ssq=ssq)
        ts = []
        Xs = []
        cors=[]
        for lim in lims:
            zm,zM,xm,xM,ym,yM = lim
            if np.sum(im_msk[xm:xM,ym:yM]>0)>0:
                ((xt,yt),success) = imreg.translation(im0_sc[xm:xM,ym:yM],im1_sc[xm:xM,ym:yM],normalized=False,plt_val=False)
                ts.append((xt,yt))
                Xs.append(((xm+xM)/2.,(ym+yM)/2.))
                cors.append(success)
        ts = np.array(ts)
        keep = np.arange(len(Xs))
        X1 = np.array(Xs)[keep]
        X2 = (np.array(Xs)+ts)[keep]
        
        for iiter in range(niter):
            keep,R,t = refine(X1,X2,keep,fr=fr,target_distance=target_distance)
        #print(len(keep))

        X_ = np.indices([self.sx_image,self.sy_image]).reshape([2,-1]).T
        X_t = np.round(np.dot(X_,R)+t).astype(int)
        smaller = X_t<0
        X_t[smaller]=0
        bigger = X_t>=[self.sx_image,self.sy_image]
        X_t[bigger[:,0],0]=self.sx_image-1
        X_t[bigger[:,1],1]=self.sy_image-1
        X_t_bad = X_[np.any((smaller|bigger),-1)]
        X_t1= (X_t,X_t_bad)
        im1_scT1 = im1_sc[X_t[:,0],X_t[:,1]].reshape(im1_sc.shape)
        
        argbest_cors = np.argsort(cors)[-5:]
        X_t = (X_-np.median(ts[argbest_cors],0)).astype(int)
        smaller = X_t<0
        X_t[smaller]=0
        bigger = X_t>=[self.sx_image,self.sy_image]
        X_t[bigger[:,0],0]=self.sx_image-1
        X_t[bigger[:,1],1]=self.sy_image-1
        X_t_bad = X_[np.any((smaller|bigger),-1)]
        X_t2= (X_t,X_t_bad)
        im1_scT2 = im1_sc[X_t[:,0],X_t[:,1]].reshape(im1_sc.shape)
        
        score1,score2 = np.sum(im0_sc*im1_scT1*(im_msk>0)),np.sum(im0_sc*im1_scT2*(im_msk>0))
        
        if verbose:
            im1_scT = im1_sc[X_t[:,0],X_t[:,1]].reshape(im1_sc.shape)
            f,(ax1,ax2,ax3,ax4)=plt.subplots(1,4,sharex=True,sharey=True)
            ax1.imshow(im0_sc,vmax=np.percentile(im0_sc,99),vmin=np.percentile(im0_sc,1))
            ax2.imshow(im1_scT1,vmax=np.percentile(im1_sc,99),vmin=np.percentile(im1_sc,1))
            ax3.imshow(im1_scT2,vmax=np.percentile(im1_sc,99),vmin=np.percentile(im1_sc,1))
            ax4.imshow(im1_sc,vmax=np.percentile(im1_sc,99),vmin=np.percentile(im1_sc,1))
            #return X_t,im0_sc,im1_sc,im1_scT1,im_msk,ts
        if score1>score2:
            return X_t1
        else:
            return X_t2
    def correct_drift(self):
        """Corrects the drift and updates self.dic_fov"""
        #internalize variables
        dic_reg = self.paramater_dic#grab necessary paramaters from here
        
        reference_file = self.files[self.index_fov][dic_reg['ref_fl']]
        if not self.drif_corrected:
            self.txys={}
        for key_i in self.dic_fov.keys():
            if (type(key_i) is int) and not self.drif_corrected:
                #get drift file and thus drift coords
                drift_file = self.file_to_driftfile(self.dic_name[key_i])
                warp_file = self.save_folder+os.sep+'warp_hybes.pkl'
                warp_matrix = None
                if os.path.exists(warp_file):
                    dic_warp = pickle.load(open(warp_file,'rb'))
                    hname = os.path.basename(os.path.dirname(self.dic_name[key_i]))
                    warp_matrix = dic_warp[hname]
                if os.path.exists(drift_file):
                    tx,ty = pickle.load(open(drift_file,'rb'))
                else:
                    print 'Registering: '+self.dic_name[key_i]
                    tx,ty=self.register_files(reference_file,self.dic_name[key_i],warp_matrix=warp_matrix)
                #correct all possible keys
                for temp_key in [key_i,'DAPI'+str(key_i),'background'+str(key_i)]:
                    if self.dic_fov.has_key(temp_key):
                        im_ = self.dic_fov[temp_key].copy()
                        if warp_matrix is not None:
                            im_ = np.array([apply_warp(im__,warp_matrix) for im__ in im_],dtype=np.uint16)
                        self.dic_fov[temp_key] = gt.translate(im_,[0,tx,ty])
                        self.txys[temp_key]=[tx,ty]
        self.drif_corrected = True
        
        
    def register_set_files(self,fls):
        """Uses register_files to register a set of files to a reference specified by fls[dic_reg['ref_fl']]"""
        #internalize variables
        dic_reg = self.paramater_dic#grab necessary paramaters from here
        ref_fl = dic_reg['ref_fl']
        #proceed
        fl0 = fls[ref_fl]
        for fl1 in fls:
            txy = self.register_files(fl0,fl1,verbose=False)
    def get_best_rigid(self,nfovs=10):
        """This computes the rotation matrix for a few fovs for each hybe and saves the output"""
        fls = np.array(self.files )
        fls_sel = fls[np.random.choice(range(len(fls)),nfovs,replace=False)]
        reg_frame  = self.reg_frame
        ref = self.ref_fl
        warps_full = []
        for fls_ in tqdm(fls_sel):
            ims = np.array([io.DaxReader(fl).loadAFrame(reg_frame) for fl in fls_],dtype=np.float32)
            warps = []
            for im in tqdm(ims):
                try:
                    warp = get_best_rigid_im(ims[ref],im)
                except:
                    warp = np.ones([2,3])+np.nan
                warps.append(warp)
            warps_full.append(np.array(warps))
        warps_full = np.array(warps_full)
        med_warps = np.nanmedian(warps_full,0)
        dic_warp = {os.path.basename(os.path.dirname(fl)):warp_matrix for fl,warp_matrix in zip(fls_sel[0],med_warps)}
        warp_file = self.save_folder+os.sep+'warp_hybes.pkl'
        pickle.dump(dic_warp,open(warp_file,'wb'))
        return warps_full,fls_sel
        
    def register_files(self,fl0,fl1,use_cor=True,verbose=False,warp_matrix=None):
        """
        Given two dax files fl1 and fl2 this loads the reg_frame (the first dapi image - frame 3 by default)
        It normalizes the image files using local median division with window <local_mean_size>
        It then computes the registration between the frames using fourier cross-correlation.
        It returns x,y such that gt.translate(im2,[xt,yt]) is aligned with im1
        """
        
        #local imports
        from cv2 import blur
        from imreg_dft import imreg
        #internalize variables
        dic_reg = self.paramater_dic#grab necessary paramaters from here
        reg_frame = dic_reg['reg_frame']
        local_mean_size = dic_reg['local_mean_size']
        #proceed
        #load data
        if fl0.split('.')[-1]=='dax':
            im0 = io.DaxReader(fl0).loadAFrame(reg_frame).astype(np.float32)#laod first dapi frame
        else:
            fl0_ = glob.glob(fl0.split('.dax')[0]+'_fr*')[-1]
            im0 = np.load(fl0_).T.astype(np.float32)
        if fl1.split('.')[-1]=='dax':
            im1 = io.DaxReader(fl1).loadAFrame(reg_frame).astype(np.float32)#laod first dapi frame
        else:
            fl1_ = glob.glob(fl1.split('.dax')[0]+'_fr*')[-1]
            im1 = np.load(fl1_).T.astype(np.float32)
        if use_cor: 
            cor_fl = self.save_folder+os.sep+'FlatField'+os.sep+'im_cor_fr2.pkl'
            im_cor = pickle.load(open(cor_fl,'rb'))
            im0 = (im0/im_cor).astype(np.float32)
            im1 = (im1/im_cor).astype(np.float32)
        #normalize by generous ~50 pixel local mean
        #blur cv2 is very fast
        im0_50 = blur(im0,(local_mean_size,local_mean_size))
        im1_50 = blur(im1,(local_mean_size,local_mean_size))
        #correct using blur
        im0_sc=im0.astype(float)/im0_50
        im1_sc=im1.astype(float)/im1_50
        im0_sc=im0_sc-im0_sc.mean()
        im1_sc=im1_sc-im1_sc.mean()
        if warp_matrix is not None:
            im1_sc = apply_warp(im1_sc,warp_matrix)
        #perform fast correlation and do unnormalized version
        ((xt,yt),success) = imreg.translation(im0_sc,im1_sc,normalized=False,plt_val=verbose)
        #save to file
        pickle.dump((xt,yt),open(self.file_to_driftfile(fl1),'wb'))
        
        if verbose:
            # Two subplots:
            f, (ax1, ax2) = plt.subplots(1, 2, sharex=True,sharey=True)
            ax1.imshow(im0_sc,interpolation='nearest',cmap=cm.gray)
            ax1.set_title('Original')
            ax2.imshow(gt.translate(im1_sc,[xt,yt]),interpolation='nearest',cmap=cm.gray)
            ax2.set_title('Translated')
            ## Two color overlay:
            #plt.figure()
            #plt.imshow(np.dstack(map(gt.minmax,[im0_sc,gt.translate(im1_sc,[xt,yt]),im0_sc])))
            plt.show()
        return (xt,yt)

    def file_to_driftfile(self,fl):
        """Decide how to save drift file"""
        save_base = '--'.join(fl.split(os.sep)[-2:])
        save_folder_drift = self.save_folder+os.sep+'Drifts'
        if not os.path.exists(save_folder_drift):
            os.makedirs(save_folder_drift)
        return save_folder_drift+os.sep+save_base+'.drft'
        
    def ini_plot(self,load_index=True):
        """Starts a plot for quick visualization
        Some plot specific functions are in here"""
        self.DAPI_on = False
        self.max_project = False
        self.ax_imshs=[]
        self.f, ax_arr = plt.subplots(3, 5, sharex=True, sharey=True)
        if load_index:
            self.load_index()
        dic = self.dic_fov
        self.z=0
        for i_ax,ax in enumerate(gt.flatten(ax_arr)):
            key_i = i_ax
            if key_i in dic.keys():
                im_ = gt.mat2gray(dic[key_i][self.z],perc_max=99.99)
                ax_imsh_ = ax.imshow(im_,cmap=cm.Greys_r,interpolation='nearest')
                self.ax_imshs.append(ax_imsh_)
                ax.set_title(str(key_i))
            
            ax.set_xlim([0,im_.shape[0]])
            ax.set_ylim([0,im_.shape[1]])
            ax.set_adjustable('box-forced')
        self.update_plt()
        self.min_max()
        def onclick(event):
            if event.button==3:
                list_axes = [im_sh_.get_axes() for im_sh_ in self.ax_imshs]
                code = list_axes.index(event.inaxes)#+1
                #print code
                if code not in self.code: 
                    self.code.append(code)
                else:
                    self.code.pop(self.code.index(code))
                self.f.suptitle(self.fov_name+str(self.code))
                self.f.canvas.draw()
        cid2 = self.f.canvas.mpl_connect('button_press_event', onclick)
        def press(event):
            if event.key == 'delete':
                self.save_list_dic.pop(-1)
                self.code=[]
            if event.key == 'q':
                if self.DAPI_on:
                    self.DAPI_on = False
                else:
                    self.DAPI_on = True
                self.update_plt()
                self.min_max()
            if event.key == 'e':
                if self.max_project:
                    self.max_project = False
                else:
                    self.max_project = True
                self.update_plt()
                self.min_max()
            if event.key == '.':
                self.z+=1
                dic = self.dic_fov
                max_z=min([len(dic[key]) for key in dic.keys()])
                if self.z>=max_z: self.z-=1
                self.update_plt()
            if event.key == ',':
                self.z-=1
                if self.z<0: self.z+=1
                self.update_plt()
            if len(self.code):
                self.add_cell()
                self.save()
                self.code=[]
            if event.key == 'd':
                self.index_fov+=1
                if self.index_fov>len(self.files)-1: self.index_fov=len(self.files)-1
                self.load_index()
                self.update_plt()
                self.min_max()
            if event.key == 'a':
                self.index_fov-=1
                if self.index_fov<0: self.index_fov=0
                self.load_index()
                self.update_plt()
                self.min_max()
            if event.key == 'x':
                self.min_max()
        cid = self.f.canvas.mpl_connect('key_press_event', press)
        plt.show()
    def update_plt(self):
        """plotfunction - updates the plot initiated with self.ini_plot"""
        dic = self.dic_fov
        for i_ax,ax in enumerate(self.ax_imshs):
            key_i = i_ax
            if key_i in dic.keys():
                #im_ = gt.mat2gray(dic[key_i][self.z],perc_max=99.999)
                if self.DAPI_on:
                    im__ = dic['DAPI'+str(key_i)]   
                else:
                    im__ = dic[key_i]
                if self.max_project:
                    im_=np.max(im__,axis=0)
                else:
                    im_ = im__[self.z]
                ax.set_data(im_)
        #self.min_max()
        self.f.suptitle(self.fov_name+str(self.code))
        self.plot_cells()
        self.f.canvas.draw()
    def min_max(self):
        """plotfunction - rescales min-max in zoomed ROI"""
        self.min_maxes=[]
        for ax_imsh in self.ax_imshs:
            axs = ax_imsh.get_axes()
            ylims,xlims = axs.get_xlim(),axs.get_ylim()
            xlims = np.array(xlims,dtype=int)
            ylims = np.array(ylims,dtype=int)
            cropped_im = ax_imsh.get_array()[xlims[0]:xlims[1],ylims[0]:ylims[1]]
            vmin,vmax = np.min(cropped_im),np.max(cropped_im)
            self.min_maxes.append([vmin,vmax])
            ax_imsh.set_clim(vmin,vmax)
        self.f.canvas.draw()
    def add_cell(self):
        """plotfunction - rescales min-max in zoomed ROI"""
        dic_cell = {}
        axs = self.ax_imshs[0].get_axes()
        ylims,xlims = axs.get_xlim(),axs.get_ylim()
        xlims = np.array(xlims,dtype=int)
        ylims = np.array(ylims,dtype=int)
        dic_cell['xylims'] = xlims,ylims
        dic_cell['ims'] = {}
        for key_ in self.dic_fov.keys():
            dic_cell['ims'][key_]=self.dic_fov[key_][self.z][xlims[0]:xlims[1],ylims[0]:ylims[1]]
        dic_cell['code']=self.code
        dic_cell['index_fov']=self.index_fov
        dic_cell['name_fov']=self.fov_name
        self.save_list_dic.append(dic_cell)
        self.plot_cells()
    def plot_cells(self):
        """plotfunction - plots lines where cells are selected"""
        for lines in self.lines:
            l=lines.pop(0)
            l.remove()
            #del l
        self.lines=[]
        dics_update = [dic for dic in self.save_list_dic if dic['index_fov']==self.index_fov]
        cols = ['r','g','b','m','y']
        for idic,dic_update in enumerate(dics_update):
            (ymin,ymax),(xmin,xmax) = dic_update['xylims']
            for ax_i,ax_imsh in enumerate(self.ax_imshs):
                if ax_i in dic_update['code']:
                    ax_ = ax_imsh.get_axes()
                    icol = np.mod(idic,len(cols))
                    lines = ax_.plot([xmin,xmax,xmax,xmin,xmin],[ymin,ymin,ymax,ymax,ymin],cols[icol])
                    self.lines.append(lines)
        self.f.canvas.draw()
    def save(self):
        """plotfunction - save cells are selected"""
        pickle_file = self.save_folder+os.sep+'cell_dic_temp.pkl'
        self.save_dic={'cells':self.save_list_dic,'utils':{'current_fov':self.index_fov}}
        pickle.dump(self.save_dic,open(pickle_file,'wb'),pickle.HIGHEST_PROTOCOL)
    def cross_corr(self,im0_,im1_):
        """Given two images this computes the normalized crosscorrelations between them relative to their medians"""
        im0=im0_-np.median(im0_)
        im1=im1_-np.median(im1_)
        im0_norm = np.sum(im0*im0)
        im1_norm = np.sum(im1*im1)
        im01 = np.sum(im0*im1)
        return float(im01)/np.sqrt(im1_norm*im0_norm)
    def cross_corr_fast(self,im0_,im1_,im0_norm_sqrt,im1_norm_sqrt):
        #im0=im0_-np.median(im0_)
        #im1=im1_-np.median(im1_)
        #im0_norm = np.sum(im0*im0)
        #im1_norm = np.sum(im1*im1)
        if (im0_norm_sqrt*im1_norm_sqrt==0):
            return 0
        else:
            im01 = np.sum(im0_*im1_)
            return float(im01)/im0_norm_sqrt/im1_norm_sqrt
    def get_cross_cors(self,ims_,return_pairs=False):
        "Given a set of images this computes the crosscor between every pair of them."
        n_ims=len(ims_)
        pairs_ = gt.flatten([[(i,j) for i in range(n_ims) if i<j] for j in range(n_ims)])#changed i>j to i<j
        #cross_cors = [self.cross_corr(ims_[cd[0]],ims_[cd[1]]) for cd in pairs_]
        ims__ = [im_-np.median(im_) for im_ in ims_]
        ims_norm = [np.sqrt(np.sum(im_*im_)) for im_ in ims__]
        cross_cors = [self.cross_corr_fast(ims__[cd[0]],ims__[cd[1]],ims_norm[cd[0]],ims_norm[cd[1]]) for cd in pairs_]
        if return_pairs:
            return cross_cors,pairs_
        return cross_cors
    def best_stage_vals(self,dax_fl,target_stage,start = 1):
        stage_ = gt.readOffset(dax_fl)['stage-z'][start:-1:3]#there is some issue droping the last frame
        #keep inner values
        arr=gt.flatten(np.where(np.diff(stage_-stage_[0])<-2))
        if len(arr)==1:
            stage_keep = stage_[arr[0]+1:]-stage_[0]
        elif len(arr)>=2:
            stage_keep = stage_[arr[0]+1:arr[1]+1]-stage_[0]
        else:
            stage_keep = stage_-stage_[0]
            arr=[-1]
        frames = np.array([arr[0]+1+np.argmin(np.abs(stage_keep - tg_val)) for tg_val in target_stage])
        #Apply an ugly correction. I no longer trust the stage values as absolute. Keep the first and last repetitions and interpolate between them
        min_non_rep,max_non_rep=0,len(frames)-2
        if np.sum(np.diff(frames)!=0)>0:        
            min_non_rep=np.min(np.where(np.diff(frames)!=0))
            max_non_rep=np.max(np.where(np.diff(frames)!=0))
        frames[min_non_rep],frames[max_non_rep+1]
        new_frames = np.array(list(frames[:min_non_rep])+list(np.linspace(frames[min_non_rep],frames[max_non_rep+1],2+max_non_rep-min_non_rep))+list(frames[max_non_rep+2:]),dtype=int)
        return new_frames
    def apply_matrix_convention_noEGR1(self):
        """This goes from the fov dictionary to the matrix conventison.
        We need matrices to compute fast corrrelations.
        The matrix current convention is: [R1,R2,...,Rn,backgroundEGR1(cy3),EGR1] , Reven is cy5 frm1,4.. , Rodd is cy3 frm2,5.., DAPI frm3,6..
        This also masks out bad data in self.mask_corr
        """
        self.internalize_paramaters()#make sure we have the latest version of paramaters
        self.dtype_mat = np.float32 #this needs to handshake the cython correlation library

        # decide how to reorganize data
        self.keys_fov = [i+1 for i in range(self.nRs)]# Add background and EGR1 to the end
        if '3col' not in self.device.lower():
            self.starts = [1+i%2 for i in range(self.nRs)]# Add background and EGR1 to the end
            if self.nRs==15:
                self.starts[self.nRs-1]=2#R15 is cy5
        else:
            R_to_col={1:'750',2:'647',3:'561',4:'750',6:'647',5:'561',7:'750',8:'647',9:'561',10:'750',12:'647',11:'561',14:'750',15:'647',13:'561'}
            cols = ['561','647','750']
            self.starts = [1+cols.index(R_to_col[i+1]) for i in range(self.nRs)]
        #deal with z, getting to frames to target (effective cropping) - handled by best_stage_vals
        self.ims_dax_names = [self.dic_name[key] for key in self.keys_fov]
        #self.stages_z = [self.best_stage_vals(dax_nm,self.target_z,start=start) for dax_nm,start in zip(self.ims_dax_names,self.starts)]
        #self.ims_matrix = np.array([self.dic_fov[key][stage_z] for key,stage_z in zip(self.keys_fov,self.stages_z)],dtype=self.dtype_mat)
        keys = list(self.dic_fov.keys())
        if 'DAPI0' not in self.dic_fov:
            self.dic_fov['DAPI0']=self.dic_fov['DAPI1']
        for key in keys: 
            if 'DAPI' in str(key) and 'DAPI0'!=key:
                self.dic_fov.pop(key)
        self.ims_matrix = np.array([self.dic_fov.pop(key) for key in self.keys_fov],dtype=self.dtype_mat)

        #Deal with masks, this depends on the self.correct_drift function. Subject to change.
        sz,sx,sy = self.ims_matrix[0].shape
        self.sz_image,self.sx_image,self.sy_image = sz,sx,sy

    def apply_matrix_convention(self):
        """This goes from the fov dictionary to the matrix conventison.
        We need matrices to compute fast corrrelations.
        The matrix current convention is: [R1,R2,...,Rn,backgroundEGR1(cy3),EGR1] , Reven is cy5 frm1,4.. , Rodd is cy3 frm2,5.., DAPI frm3,6..
        This also masks out bad data in self.mask_corr
        """
        self.internalize_paramaters()#make sure we have the latest version of paramaters
        self.dtype_mat = np.float32 #this needs to handshake the cython correlation library

        # decide how to reorganize data
        self.keys_fov = [i+1 for i in range(self.nRs)]+['background0',0]# Add background and EGR1 to the end
        if '3col' not in self.device.lower():
            self.starts = [1+i%2 for i in range(self.nRs)]+[1,2]# Add background and EGR1 to the end
            if self.nRs==15:
                self.starts[self.nRs-1]=2#R15 is cy5
        else:
            R_to_col={1:'750',2:'647',3:'561',4:'750',6:'647',5:'561',7:'750',8:'647',9:'561',10:'750',12:'647',11:'561',14:'750',15:'647',13:'561'}
            cols = ['561','647','750']
            self.starts = [1+cols.index(R_to_col[i+1]) for i in range(self.nRs)]+[1,2]
        #deal with z, getting to frames to target (effective cropping) - handled by best_stage_vals
        self.ims_dax_names = [self.dic_name[key] for key in self.keys_fov]
        #self.stages_z = [self.best_stage_vals(dax_nm,self.target_z,start=start) for dax_nm,start in zip(self.ims_dax_names,self.starts)]
        #self.ims_matrix = np.array([self.dic_fov[key][stage_z] for key,stage_z in zip(self.keys_fov,self.stages_z)],dtype=self.dtype_mat)
        keys = list(self.dic_fov.keys())
        for key in keys: 
            if 'DAPI' in str(key) and 'DAPI0'!=key:
                self.dic_fov.pop(key)
        self.ims_matrix = np.array([self.dic_fov.pop(key) for key in self.keys_fov],dtype=self.dtype_mat)

        #Deal with masks, this depends on the self.correct_drift function. Subject to change.
        sz,sx,sy = self.ims_matrix[0].shape
        self.sz_image,self.sx_image,self.sy_image = sz,sx,sy
        self.mask_corr = np.ones([sz,sx,sy],dtype=np.int32)
        if getattr(self,'rough_align',False):
            for im in self.ims_matrix:
                self.mask_corr*=(im>0)
        else:
            for key in self.keys_fov:
                self.mask_corr = set_bad_to_0(self.mask_corr,self.txys.get(key,(0,0)))
        self.apply_MOE_mask()
    def apply_MOE_mask(self):
        """This applies MOE mask"""
        self.maskMOEfov = self.im_MOEmask(self.fov_name)
        self.mask_corr *= np.expand_dims(self.maskMOEfov>0,0)
        #self.ims_matrix *= np.expand_dims(np.expand_dims(self.maskMOEfov>0,0),0)
    def apply_bleedthrough(self):
        """This applies the a bleedthrough correction - flags the potential bleed pixels setting them to 0."""
        #ims_matrix_bleed = self.ims_matrix.copy()
        hybeindex = np.array(self.hybeindex)
        n_ims = len(self.ims_matrix)
        for i_im in range(n_ims):
            select_max = hybeindex<=i_im
            select_max[i_im]=False
            current_im = self.ims_matrix[i_im].copy()
            #currrent_baseline = np.percentile(current_im,self.perc_baseline)
            current_bleed = np.max(self.ims_matrix[select_max],axis=0)
            #current_im[current_im>=2**16-1]=0
            #current_im[current_bleed>=2**16-1]=0
            current_im -= current_bleed/self.bleed_factor
            #current_im -= currrent_baseline
            #current_im[current_im<=0]=0
            #self.ims_matrix[i_im][current_im<=0]=0
            self.ims_matrix[i_im] = current_im
    def load_corrections(self,num_cols = 4):
        self.im_cors={}
        for frm in range(num_cols):
            fl = self.save_folder+os.sep+'FlatField'+os.sep+'im_cor_fr'+str(frm)+'.pkl'
            if os.path.exists(fl):
                im_cor = pickle.load(open(fl,'rb'))
                self.im_cors[frm+1]=im_cor
    def apply_flatfield(self,cor_ilum=True):
        """This uses self.correction_image to compute or load appropirate correction images 
        and then it applies them to the self.ims_matrix"""
        self.load_corrections()
        for i_im,frm in enumerate(self.starts):
            im_cor = np.array(self.im_cors[frm],dtype=np.float32)#/np.mean(self.im_cors[frm])
            self.ims_matrix[i_im]=self.ims_matrix[i_im]/np.expand_dims(im_cor,0)
            if cor_ilum:
                im_mds = np.array([np.median(im) for im in self.ims_matrix[i_im]])
                im_mds /=np.median(im_mds)
                self.ims_matrix[i_im]=self.ims_matrix[i_im]/np.expand_dims(np.expand_dims(im_mds,-1),-1)
    def correction_image(self,frm,set_fl=0,perc_=95,save_note='',save_file='auto',overwrite=False,use_tqdm=True):
        """
        Given a set of dax files set_fl(or an index which extracts all the files from self.files) 
        this extracts the frame frm and will create an image with the percentile perc_ which can be used to correct.
        
        *By defaults with save_file='auto' this saves to the analysis folder in subfolder FlatField
        """
        #Decide where to save:
        if save_file=='auto':
            save_folder = self.save_folder+os.sep+'FlatField'
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_file = save_folder+os.sep+'im_cor_fr'+save_note+'.pkl'
        else:
            if save_file is not None:
                save_folder = os.path.dirname(save_file)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
        #Decide how to handle the im_cor matrix
        if overwrite or (not os.path.exists(save_file)) or (save_file is None):
            if hasattr(set_fl,'__getitem__'): files_=set_fl #more flexible
            else: files_=[fls[set_fl] for fls in self.files] #ensures backwards compatibility
            ims = []
            print files_[0]
            if use_tqdm: files_ = tqdm(files_)
            
            for fl in files_:
                try:
                    if fl.split('.')[-1]=='dax':
                        ims.append(io.DaxReader(fl).loadAFrame(frm))
                    elif fl.split('.')[-1]=='npy':
                        ims.append(np.load(fl).T)
                    else:
                        ims.extend(load_im_set(fl,custom_frms=[frm]))
                except:
                    pass
            im_cor=np.percentile(ims,perc_,axis=0)
            self.im_cor=im_cor
            if save_file is not None:
                f, ax = plt.subplots()
                ax.imshow(im_cor,interpolation='nearest',cmap=cm.gray)
                f.savefig(save_file.replace('.pkl','.png'))
                plt.close(f)
                pickle.dump(im_cor,open(save_file,'wb'),protocol=pickle.HIGHEST_PROTOCOL)
        else:
            im_cor=pickle.load(open(save_file,'rb'))
        return im_cor
                

    
    #-----------------decoding functions start here
    def decoded_main_v2(self,lib_keep=None,plt_val=False,save_data=False):
        """This handles the main decoding
        lib_keep is a list with the indexes of library used. By convention the self.data_folder ends with libx,y
        This puts all info in decoded_dic
        Requires: self.load_library() - if this was not run it will run it automatically
                  self.cross_cors_3D_patch()
                  self.cross_corrs_decode()
        #---------------decoding paramaters
        #threshold paramaters
        self.th_bk = 3.5 #units of standard deviation above the mean to get called as a valuable spot in (decoded - background)
        self.th_dec = 0.6 #min cross-corr in decoded
        #dbscan_paramaters
        self.dbscan_corr_eps = 4 #cell diamater in reduced coordinates
        self.dbscan_corr_min_samples = 20 #patches to accept clusters
        self.dbscan_fr = 2 #split clusters until the disttance between clusters is <self.dbscan_corr_eps*self.dbscan_fr
        """
        self.internalize_paramaters()
        parms = self.paramater_dic.copy()
        sz_reduced,sx_reduced,sy_reduced,_=self.im_cross_cors.shape
        _,sz,sx,sy=self.ims_matrix.shape
        dax_reference = self.files[self.index_fov][self.ref_fl]
        dic_inf = io.readInfoFile(dax_reference.replace('.dax','.inf'))
        aux_paramaters = {'stage_x':dic_inf['Stage X'],'stage_y':dic_inf['Stage Y'],'fov_name':self.fov_name,'sz':sz,'sx':sx,'sy':sy,'sz_reduced':sz_reduced,'sx_reduced':sx_reduced,'sy_reduced':sy_reduced,#dimension
                          'stages_z':self.stages_z,'txys':self.txys,'dic_name':self.dic_name,
                          'data_folder':self.data_folder,'save_folder':self.save_folder}
        parms.update(aux_paramaters)
        decoded_dic = {'paramaters':parms,'cells':{}}
                
        keep_comb_index = self.keep_comb_index
        im_argmax = self.im_argmax
        n_ims,n_onbits = self.nRs,self.nOn
        spacing = np.array([self.spacing_z,self.spacing_xy,self.spacing_xy])
        combs = map(list,list(itertools.combinations(range(n_ims),n_onbits)))#all codes
        Xconvs = []
        X_sfs = []
        for no_index,keep_index in enumerate(tqdm(keep_comb_index)):

            #####set the current code to check
            current_code = tuple(np.array(combs[keep_index])+1)

            if True:#current_code==tuple(np.sort([1,8,10,14])):#[2,4,6,8],[1,8,10,14]
                im_argmax_cd = im_argmax == keep_index #
                im_dec = self.im_decoded[...,keep_index] #local decoded image
                #im_bk = self.im_background[...,keep_index] #local background image
                im_bk = self.im_med_bk_corr #local "background" image
                im_diff = im_dec-im_bk #local difference (decoded-background)


                mask_th_dec=im_dec>self.th_dec
                im_nonzeros=im_dec>=0
                mask_th_bk = im_diff>np.mean(im_diff[im_nonzeros])+self.th_bk*np.std(im_diff[im_nonzeros])

                cd_mask = im_argmax_cd*mask_th_bk*mask_th_dec
                in_library_index = self.codes_lib_p1.index(list(current_code))
                sublibrary = self.index_lib[in_library_index]#*save
                olfr_nm = self.olfr_lib[in_library_index]#*save
                z,x,y = np.where(cd_mask)
                
                
                X_red = np.array(np.where(cd_mask)).T[:,[1,2,0]]
                X_list = cellClustering(X_red, self.dbscan_corr_min_samples, self.dbscan_corr_eps, 
                                            fr=self.dbscan_fr, performDBSCAN=True)
                for label,coords_reduced in enumerate(X_list):
                    center_reduced = np.mean(coords_reduced,axis=0)
                    X_red_ = coords_reduced[:,[2,0,1]]
                    mean_crosscor = np.mean(im_dec[X_red_[:,0],X_red_[:,1],X_red_[:,2]])
                    mean_bk = np.mean(im_bk[X_red_[:,0],X_red_[:,1],X_red_[:,2]])
                    cell_id = str(current_code)+'_'+str(label)
                    decoded_dic['cells'][cell_id]={'code':current_code,'code_index':keep_index,'olfr':olfr_nm,
                                                           'sublibrary':sublibrary,'coords_reduced':coords_reduced,
                                                           'center_reduced':center_reduced,'mean_crosscor':mean_crosscor,'mean_bk':mean_bk}
        
                if plt_val:
                    fig = plot_cor_dec(im_diff,im_argmax_cd,mask_th_dec,mask_th_bk,cd_mask,X_red,X_list,title=self.library[in_library_index])
                    if save_data:
                        save_folder = self.save_folder+os.sep+'Decoded'+os.sep+self.fov_name.replace('.dax','')+os.sep+'decoded_main'
                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder)
                        try:#fixes a NaN
                            fig.savefig(save_folder+os.sep+str(current_code)+'.png')
                        except:
                            print "Failed to save the figure!"
                    else:
                        plt.show()
                    plt.close(fig)
        self.decoded_dic=decoded_dic
    def decoded_main(self,lib_keep=None,plt_val=False,save_data=False):
        """This handles the main decoding
        lib_keep is a list with the indexes of library used. By convention the self.data_folder ends with libx,y
        This puts all info in decoded_dic
        Requires: self.load_library() - if this was not run it will run it automatically
                  self.cross_cors_3D_patch()
                  self.cross_corrs_decode()
        #---------------decoding paramaters
        #threshold paramaters
        self.th_bk = 3.5 #units of standard deviation above the mean to get called as a valuable spot in (decoded - background)
        self.th_dec = 0.6 #min cross-corr in decoded
        #dbscan_paramaters
        self.dbscan_corr_eps = 4 #cell diamater in reduced coordinates
        self.dbscan_corr_min_samples = 20 #patches to accept clusters
        """
        #specific libraries
        from scipy.spatial.distance import pdist
        #from sklearn.cluster import KMeans#MiniBatchKMeans as KMeans#KMeans#
        from sklearn.cluster import DBSCAN
        from skimage.morphology import ball

        self.internalize_paramaters()
        self.cell_diameter_red=int(np.round(float(self.cell_diameter)/self.spacing_xy))
        #save paramaters to decoded_dic
        parms = self.paramater_dic.copy()
        sz_reduced,sx_reduced,sy_reduced,_=self.im_cross_cors.shape
        _,sz,sx,sy=self.ims_matrix.shape
        dax_reference = self.files[self.index_fov][self.ref_fl]
        dic_inf = io.readInfoFile(dax_reference.replace('.dax','.inf'))
        aux_paramaters = {'stage_x':dic_inf['Stage X'],'stage_y':dic_inf['Stage Y'],'fov_name':self.fov_name,'sz':sz,'sx':sx,'sy':sy,'sz_reduced':sz_reduced,'sx_reduced':sx_reduced,'sy_reduced':sy_reduced,#dimension
                          'stages_z':self.stages_z,'txys':self.txys,'dic_name':self.dic_name,
                          'data_folder':self.data_folder,'save_folder':self.save_folder}
        parms.update(aux_paramaters)
        decoded_dic = {'paramaters':parms,'cells':{}}
        #####set the current code to check to keep_index and iterate
        keep_comb_index = self.keep_comb_index
        im_argmax = self.im_argmax
        n_ims,n_onbits = self.nRs,self.nOn
        combs = map(list,list(itertools.combinations(range(n_ims),n_onbits)))#all codes
        for no_index,keep_index in enumerate(keep_comb_index):

            #####set the current code to check
            current_code = tuple(np.array(combs[keep_index])+1)

            if True:#current_code==tuple([4,5,7,13]):
                im_argmax_cd = im_argmax == keep_index #
                im_dec = self.im_decoded[...,keep_index] #local decoded image
                #im_bk = self.im_background[...,keep_index] #local background image
                im_bk = self.im_med_bk_corr #local "background" image
                im_diff = im_dec-im_bk #local difference (decoded-background)


                mask_th_dec=im_dec>self.th_dec
                im_nonzeros=im_dec>=0
                mask_th_bk = im_diff>np.mean(im_diff[im_nonzeros])+self.th_bk*np.std(im_diff[im_nonzeros])

                cd_mask = im_argmax_cd*mask_th_bk*mask_th_dec
                in_library_index = self.codes_lib_p1.index(list(current_code))
                sublibrary = self.index_lib[in_library_index]#*save
                olfr_nm = self.olfr_lib[in_library_index]#*save
                z,x,y = np.where(cd_mask)
                X_list=[]
                if len(x):
                    X=np.array(zip(x,y,z))
                    dbscan_ = DBSCAN(eps=self.dbscan_corr_eps, min_samples=self.dbscan_corr_min_samples)
                    dbscan_.fit(X)
                    #labels = dbscan_.labels_
                    labels = np.array(dbscan_.labels_,dtype=int)
                    self.temp_labels = labels
                    if np.sum(labels>=0)>0:
                        #save to dictionary
                        X_list = []
                        labels_copy = labels.copy()
                        max_lab = np.max(labels_copy)
                        for label in range(max_lab+1):
                            if label>=0:
                                keep = labels==label
                                coords_reduced = X[keep]
                                #decide whether to split - use X-means clustering-------------------------------------
                                #Increase the number of clusters succesively until the min distance between clusters is less than cell diamater 
                                kno=1
                                while True:
                                    kno+=1
                                    break_=False

                                    kmeans = KMeans(kno)
                                    kmeans.fit(coords_reduced)
                                    kmeanslabels_=np.array(kmeans.labels_,dtype=int)
                                    kmeanscluster_centers_=[]


                                    #if too few samples in a cluster ignore that cluster
                                    for iK in range(kno):
                                        investigate_small = kmeanslabels_==iK
                                        if np.sum(investigate_small)<self.dbscan_corr_min_samples:
                                            kmeanslabels_[investigate_small]=-1
                                        else:
                                            kmeanscluster_centers_.append(kmeans.cluster_centers_[iK])
                                    #if the min distance between any 2 non-ignored clusters is smaller than cell diamater flag break!
                                    if len(kmeanscluster_centers_)<2:
                                        break
                                    center_distances = pdist(kmeanscluster_centers_)
                                    if np.min(center_distances)<self.cell_diameter_red:
                                        break
                                    if break_:
                                        break
                                    prev_labels=kmeanslabels_
                                if kno>2:
                                    def remap_label(lab_):
                                        if lab_==0:
                                            return label
                                        if lab_>0:
                                            return lab_+max_lab
                                        if lab_<0:
                                            return -1
                                    labels[keep]=map(remap_label,prev_labels)#modify labels to add splits
                                    max_lab=np.max(labels)

                        for label in range(np.max(labels)+1):
                            if label>=0:
                                keep = labels==label
                                if len(keep)>0:
                                    coords_reduced = X[keep]#*save
                                    center_reduced = np.mean(coords_reduced,axis=0)
                                    X_list.append(coords_reduced)
                                    #decide how to split here
                                    mean_crosscor = im_dec[cd_mask][keep]#*save
                                    mean_bk = im_bk[cd_mask][keep]#*save
                                    cell_id = str(current_code)+'_'+str(label)
                                    decoded_dic['cells'][cell_id]={'code':current_code,'code_index':keep_index,'olfr':olfr_nm,'sublibrary':sublibrary,'coords_reduced':coords_reduced,
                                                                   'center_reduced':center_reduced,'mean_crosscor':mean_crosscor,'mean_bk':mean_bk}


                if plt_val:
                    #------plotting
                    ims = [im_diff,im_argmax_cd,mask_th_dec,mask_th_bk,cd_mask]
                    titles = ['Image','max guess','abs cross thresh','background thresh','final']
                    fig, ax_arr = plt.subplots(2, 3, sharex=True,sharey=True)
                    sz_t,sx_t,sy_t = ims[0].shape
                    for i,ax in enumerate(np.array(ax_arr).flatten()):
                        if i<len(ims):
                            cmap = cm.gray if i==0 else cm.jet 
                            f = np.max if i==0 else np.mean
                            im_plot = gt.minmax(f(ims[i],axis=0))
                            sx_t,sy_t=im_plot.shape
                            sx_t0,sy_t0=sx_t,sy_t
                            im_plot = np.concatenate([im_plot,np.ones([sx_t,2]),gt.minmax(f(ims[i],axis=-1)).T],axis=1)
                            sx_t,sy_t=im_plot.shape
                            im_plot_ = np.concatenate([gt.minmax(f(ims[i],axis=-2)),np.ones([sz_t,sy_t-sy_t0])],axis=1)
                            im_plot = np.concatenate([im_plot_[::-1,:],np.ones([2,sy_t]),im_plot],axis=0)
                            sx_t,sy_t=im_plot.shape
                            ax.imshow(im_plot,interpolation='nearest',cmap=cmap)
                            ax.set_title(titles[i])
                            ax.set_xlim([0,sx_t])
                            ax.set_ylim([0,sy_t])
                        elif i==len(ims):
                            ax.plot(y,x+sy_t-sy_t0,'k+')
                            for iX,X_ in enumerate(X_list):
                                x_,y_,z_=X_.T
                                ax.plot(y_,x_+sy_t-sy_t0,'o')
                                ax.text(np.mean(y_),np.mean(x_+sy_t-sy_t0),str(iX))
                            ax.set_xlim([0,sx_t])
                            ax.set_ylim([0,sy_t])
                        ax.set_adjustable('box-forced')
                    fig.suptitle(self.library[in_library_index])
                    #if len(X_list)>0:
                    #    plt.show()
                    if save_data:
                        save_folder = self.save_folder+os.sep+'Decoded'+os.sep+self.fov_name.replace('.dax','')+os.sep+'decoded_main'
                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder)
                        try:#fixes a NaN
                            fig.savefig(save_folder+os.sep+str(current_code)+'.png')
                        except:
                            print "Failed to save the figure!"
                    else:
                        plt.show()
                    plt.close(fig)
        #filter out the 0 elements
        for cell_id in decoded_dic['cells'].keys():
            if len(decoded_dic['cells'][cell_id]['coords_reduced'])==0:
                del decoded_dic['cells'][cell_id]
        self.decoded_dic = decoded_dic
    #post analysis

    #check for overlap:
    def decoded_overlap(self):
        """Given a decoded_dic created by decoded_main, this deals with 
        overlapped decoded cells (majority of cases caused by bleedthrough)
        It checks for proximity (cell radius cutoff) and compares the corr-background
        to select only one cell. The original cells are moved to decoded_dic['overlap']
        Requires:
        self.decoded_dic - the dictionary created by self.decoded_main
        self.im_med_bk_corr - cross-cor values for the code created by self.
        self.im_med_bk_corr - background cross-cor created by self.
        """
        from scipy.spatial.distance import pdist 
        decoded_dic = self.decoded_dic
        dic_cells = decoded_dic['cells']
        cell_keys = dic_cells.keys()
        decoded_dic['overlap']={}
        centers = [dic_cells[key]['center_reduced']for key in cell_keys]
        codes = [dic_cells[key]['code']for key in cell_keys]
        if len(centers)>1:
            #check for distances
            pairs_c=[]
            pairs_icode=[]#record the number of intersections
            for ic in range(len(centers)):
                for jc in range(len(centers)):
                    if ic<jc:
                        pairs_c.append((ic,jc))
                        pairs_icode.append(len(np.intersect1d(codes[ic],codes[jc])))
            pairs_c = np.array(pairs_c)
            pairs_icode = np.array(pairs_icode)
            keep = pdist(centers)<self.cell_diameter_red/2.
            keep = keep|((pdist(centers)<self.cell_diameter_red)&(pairs_icode>=self.nOn-1))
            pairs_c=pairs_c[keep]
            #iterate through proximal cells
            for pr_i,pr_j in pairs_c:
                key_i,key_j = cell_keys[pr_i],cell_keys[pr_j]
                if dic_cells.has_key(key_i) and dic_cells.has_key(key_j):
                    coords_i = dic_cells[key_i]['coords_reduced']
                    coords_j = dic_cells[key_j]['coords_reduced']
                    cdi_i = dic_cells[key_i]['code_index']
                    cdi_j = dic_cells[key_j]['code_index']
                    coords_both=np.concatenate([coords_i,coords_j],axis=0)
                    x,y,z = coords_both.T
                    #create scores
                    im_dec_t = self.im_decoded[z,x,y]
                    im_med_bk_corr_t = self.im_med_bk_corr[z,x,y]
                    mn_score_i = np.mean(im_dec_t[...,cdi_i]-im_med_bk_corr_t)
                    mn_score_j = np.mean(im_dec_t[...,cdi_j]-im_med_bk_corr_t)
                    #move cells to overlap
                    if not decoded_dic['overlap'].has_key(key_i):
                        decoded_dic['overlap'][key_i]=dic_cells[key_i].copy()
                    if not decoded_dic['overlap'].has_key(key_j):
                        decoded_dic['overlap'][key_j]=dic_cells[key_j].copy()

                    #compare scores
                    if mn_score_i>mn_score_j:
                        key_i,key_j=key_j,key_i
                        cdi_i,cdi_j=cdi_j,cdi_i
                    #will keep key_j by modifying it in the original 
                    dic_cells[key_j]['coords_reduced']=coords_both
                    dic_cells[key_j]['center_reduced']=np.mean(coords_both,axis=0)
                    dic_cells[key_j]['mean_crosscor']=im_dec_t[...,cdi_j]
                    dic_cells[key_j]['mean_bk']=im_med_bk_corr_t
                    overlap_keys_i = dic_cells[key_i].get('overlap_keys',set())
                    overlap_keys_j = dic_cells[key_j].get('overlap_keys',set())
                    dic_cells[key_j]['overlap_keys']=overlap_keys_i.union(overlap_keys_j)
                    dic_cells[key_j]['overlap_keys'].add(key_i)
                    dic_cells[key_j]['overlap_keys'].add(key_j)
                    del dic_cells[key_i]
        self.decoded_dic=decoded_dic
    def decoded_unoverlap(self):
        """This undos the self.decoded_overlap"""
        decoded_dic = self.decoded_dic
        dic_cells = decoded_dic['cells']
        dic_overlap = decoded_dic['overlap']
        dic_cell_keys = dic_cells.keys()
        for key in dic_cell_keys:
            overlap_keys = dic_cells[key].get('overlap_keys',set())
            if len(overlap_keys)>0:
                del dic_cells[key]
                for key_ in overlap_keys:
                    dic_cells[key_]=dic_overlap[key_]
                decoded_dic['overlap']={}
        self.decoded_dic = decoded_dic
    def decoded_pixelIntensities_v2(self):
        """Supposing self.decoded_dic and self.ims_matrix this finds mean,std,skewness,kurtosis and volume(in pixels) 
        of the intensities in the cell and arround it (background)
        # Usefull to get rough estimates of intensity of RNA signal
        """
        from scipy.ndimage.morphology import binary_dilation
        from skimage.morphology import ball
        self.cell_diameter_red=int(np.round(float(self.cell_diameter)/self.spacing_xy))
        self.decoded_dic['background']={}
        decoded_dic=self.decoded_dic
        sz_reduced=decoded_dic['paramaters']['sz_reduced']
        sx_reduced=decoded_dic['paramaters']['sx_reduced']
        sy_reduced=decoded_dic['paramaters']['sy_reduced']
        mask_olfr = np.zeros([sz_reduced,sx_reduced,sy_reduced],dtype=np.uint8)
        dic_cells = decoded_dic['cells']#deepcopy(decoded_dic['cells'])
        dic_cells_keys = dic_cells.keys()
        centers=[]
        for key in dic_cells_keys:
            X = dic_cells[key]['coords_reduced']
            x,y,z=X.T
            mask_olfr[z,x,y]=1

        background_mask=binary_dilation(mask_olfr,iterations=1)==0
        background_mask[self.im_med_bk_corr==0]=0#only look inside valid region

        background_kernel = ball(int(self.cell_diameter_red*1.))
        background_kernel = np.array(np.where(background_kernel)).T
        background_kernel = background_kernel-np.expand_dims(np.mean(background_kernel,axis=0),0)

        for i_key,key in enumerate(tqdm(dic_cells_keys)):
            code = dic_cells[key]['code']
            code_py = np.array(code)-1
            X = dic_cells[key]['coords_reduced']
            x,y,z=X.T

            center = dic_cells[key]['center_reduced']
            background_ball = np.array(background_kernel+np.expand_dims(center,0),dtype=int)
            bkx,bky,bkz = background_ball.T
            keep = (bkx<sx_reduced)&(bky<sy_reduced)&(bkz<sz_reduced)
            keep = keep&(bkx>=0)&(bky>=0)&(bkz>=0)
            bkx,bky,bkz=bkx[keep],bky[keep],bkz[keep]
            keep = background_mask[bkz,bkx,bky]
            bkx,bky,bkz=bkx[keep],bky[keep],bkz[keep]
            #deal with empty background
            if len(bkx)==0:
                z_big_bk,x_big_bk,y_big_bk=[],[],[]
            else:
                z_big_bk,x_big_bk,y_big_bk = self.get_big_xyz(zip(bky,bkx,bkz))#flip
            z_big,x_big,y_big = self.get_big_xyz(zip(y,x,z))#flip
            
            #z_big,x_big,y_big = self.get_big_xyz(dic_cell['coords_reduced'][:,[1,0,2]])
            keeps = []
            for icd in code_py:
                values = self.ims_matrix[icd][z_big,x_big,y_big]
                values_bk = self.ims_matrix[icd][z_big_bk,x_big_bk,y_big_bk]
                mn_,std_ = np.mean(values_bk),np.std(values_bk)
                keeps.append(values>mn_+2.5*std_)
            keeps = np.array(keeps,dtype=np.uint8)
            keep_sum = np.sum(keeps,axis=0)
            keep = keep_sum>=4
            zxy_tight = np.array([z_big[keep],x_big[keep],y_big[keep]]).T
            keep = keep_sum>=3
            zxy_semiloose = np.array([z_big[keep],x_big[keep],y_big[keep]]).T
            keep = keep_sum>=2
            zxy_loose = np.array([z_big[keep],x_big[keep],y_big[keep]]).T
            dic_cells[key]['zxy_signal_tight'] = zxy_tight
            dic_cells[key]['zxy_signal_loose'] = zxy_loose
            dic_cells[key]['zxy_signal_semiloose'] = zxy_semiloose
        self.decoded_dic['cells']=dic_cells
    def decoded_pixelIntensities(self):
        """Supposing self.decoded_dic and self.ims_matrix this finds mean,std,skewness,kurtosis and volume(in pixels) 
        of the intensities in the cell and arround it (background)
        # Usefull to get rough estimates of intensity of RNA signal
        Suggest doing:
        self.apply_matrix_convention()
        self.apply_flatfield()
        """
        from scipy.stats import kurtosis
        from scipy.stats import skew
        from scipy.ndimage.morphology import binary_dilation
        from skimage.morphology import ball
        self.cell_diameter_red=int(np.round(float(self.cell_diameter)/self.spacing_xy))
        self.decoded_dic['background']={}
        decoded_dic=self.decoded_dic
        sz_reduced=decoded_dic['paramaters']['sz_reduced']
        sx_reduced=decoded_dic['paramaters']['sx_reduced']
        sy_reduced=decoded_dic['paramaters']['sy_reduced']
        mask_olfr = np.zeros([sz_reduced,sx_reduced,sy_reduced],dtype=np.uint8)
        dic_cells = decoded_dic['cells']
        dic_cells_keys = dic_cells.keys()
        centers=[]
        for key in dic_cells_keys:
            X = dic_cells[key]['coords_reduced']
            x,y,z=X.T
            mask_olfr[z,x,y]=1

        background_mask=binary_dilation(mask_olfr,iterations=1)==0
        background_mask[self.im_med_bk_corr==0]=0#only look inside valid region

        background_kernel = ball(int(self.cell_diameter_red*1.))
        background_kernel = np.array(np.where(background_kernel)).T
        background_kernel = background_kernel-np.expand_dims(np.mean(background_kernel,axis=0),0)

        for i_key,key in enumerate(tqdm(dic_cells_keys)):
            #for i_key,key in enumerate([str((1,3,6,12))+'_0']):
            X = dic_cells[key]['coords_reduced']
            code = dic_cells[key]['code']
            code_py = np.array(code)-1
            x,y,z=X.T
            center = dic_cells[key]['center_reduced']
            background_ball = np.array(background_kernel+np.expand_dims(center,0),dtype=int)
            bkx,bky,bkz = background_ball.T
            keep = (bkx<sx_reduced)&(bky<sy_reduced)&(bkz<sz_reduced)
            keep = keep&(bkx>=0)&(bky>=0)&(bkz>=0)
            bkx,bky,bkz=bkx[keep],bky[keep],bkz[keep]
            keep = background_mask[bkz,bkx,bky]
            bkx,bky,bkz=bkx[keep],bky[keep],bkz[keep]
            #deal with empty background
            if len(bkx)==0:
                z_big_bk,x_big_bk,y_big_bk=[],[],[]
            else:
                z_big_bk,x_big_bk,y_big_bk = self.get_big_xyz(zip(bky,bkx,bkz))#flip
            z_big,x_big,y_big = self.get_big_xyz(zip(y,x,z))#flip
            mn_std_skew_kurt_nopix = []
            mn_std_skew_kurt_nopix_bk = []
            signal_rgs=[]
            for cd in range(self.nRs):#span regions
                pixel_values_bk = self.ims_matrix[cd][z_big_bk,x_big_bk,y_big_bk]
                pixel_values = self.ims_matrix[cd][z_big,x_big,y_big]
                mn_std_skew_kurt_nopix.append([np.mean(pixel_values),np.std(pixel_values),
                                              skew(pixel_values),kurtosis(pixel_values),len(pixel_values)])
                if len(pixel_values_bk)==0:
                    mn_std_skew_kurt_nopix_bk.append([np.nan,np.nan,np.nan,np.nan,np.nan])
                else:
                    mn_std_skew_kurt_nopix_bk.append([np.mean(pixel_values_bk),np.std(pixel_values_bk),
                                                      skew(pixel_values_bk),kurtosis(pixel_values_bk),len(pixel_values_bk)])
                if cd in code_py:
                    bk_subtract = mn_std_skew_kurt_nopix[-1][0]
                    weights = pixel_values-bk_subtract
                    keep = weights>0
                    weights_kp = weights[keep]
                    x_kp,y_kp,z_kp = x_big[keep],y_big[keep],z_big[keep]
                    x_kp = x_kp-np.average(x_kp,weights=weights_kp)
                    y_kp = y_kp-np.average(y_kp,weights=weights_kp)
                    z_kp = z_kp-np.average(z_kp,weights=weights_kp)
                    dist_kp = x_kp*x_kp+y_kp*y_kp+z_kp*z_kp*(self.nm_per_pixel_xy/self.nm_per_pixel_z)**2
                    signal_rg = np.sqrt(np.average(dist_kp,weights=weights_kp))
                    signal_rgs.append(signal_rg)

            dic_cells[key]['signal_rgs']=signal_rgs
            mn_std_skew_kurt_nopix=np.array(mn_std_skew_kurt_nopix)
            mn_std_skew_kurt_nopix_bk=np.array(mn_std_skew_kurt_nopix_bk)
            scores = (mn_std_skew_kurt_nopix[...,0]-mn_std_skew_kurt_nopix_bk[...,0])/mn_std_skew_kurt_nopix_bk[...,1]
            dic_cells[key]['mn_std_skew_kurt_nopix']=mn_std_skew_kurt_nopix
            dic_cells[key]['mn_std_skew_kurt_nopix_bk']=mn_std_skew_kurt_nopix_bk
            dic_cells[key]['scores']=scores
            
            
            #z_big,x_big,y_big = self.get_big_xyz(dic_cell['coords_reduced'][:,[1,0,2]])
            keeps = []
            code_ = np.array(code) - 1
            for ibest in code_[np.argsort(scores[code_])]:
                values = self.ims_matrix[ibest][z_big,x_big,y_big]
                mn_,std_ = mn_std_skew_kurt_nopix_bk[ibest][:2]
                keeps.append(values>mn_+2.5*std_)
            keep = np.all(keeps[-4:],axis=0)
            zxy_tight = np.array([z_big[keep],x_big[keep],y_big[keep]]).T
            keep = np.all(keeps[-3:],axis=0)
            zxy_semiloose = np.array([z_big[keep],x_big[keep],y_big[keep]]).T
            keep = np.all(keeps[-2:],axis=0)
            zxy_loose = np.array([z_big[keep],x_big[keep],y_big[keep]]).T
            dic_cells[key]['zxy_signal_tight'] = zxy_tight
            dic_cells[key]['zxy_signal_loose'] = zxy_loose
            dic_cells[key]['zxy_signal_semiloose'] = zxy_semiloose
        self.decoded_dic['cells']=dic_cells
    def decoded_background_v2(self):
        """Supposing self.decoded_dic and self.ims_matrix this decides whether a cross-cor decoded cell is background
        Requires self.decoded_pixelIntensities()
        Uppon subtraction this normalized score (mn_cd-mn_no) is thresholded by: self.background_threshold
        
        """
        self.internalize_paramaters()
        self.decoded_dic['background']={}
        decoded_dic=self.decoded_dic
        dic_cells = decoded_dic['cells']
        dic_cells_keys = dic_cells.keys()
        for i_key,key in enumerate(dic_cells_keys):
            dic_cell = dic_cells[key]
            code = dic_cell['code']
            code_py = np.array(code)-1
            scores = dic_cell['scores']
            
            non_code_py = np.setdiff1d(np.arange(len(scores)),code_py)
            mn_no = np.mean(scores[non_code_py])
            mn_cd = np.mean(scores[code_py])
            
            #if no bits are clearly above background (0.5 std above mean) then suspect background cross_corr
            #if np.sum(np.isnan(scores_code)|(scores_code>0.5))==0:
            if mn_cd-mn_no<self.background_threshold:#This has been obtained empirically from training set.
                self.decoded_dic['background'][key]=dic_cells.pop(key)
        self.decoded_dic['cells']=dic_cells
    def decoded_background(self):
        """Supposing self.decoded_dic and self.ims_matrix this decides whether a cross-cor decoded cell is background
        Requires self.decoded_pixelIntensities()
        *It uses the scores previously provided by self.decoded_pixelIntensities. These are per bit = (mean singnal - mean_background)/stv_background
        *It normalizes these scores by subtracting wcy3*mean(over_all_cy3_bitscores)+wcy5*mean(over_all_cy5_bitscores) 
        #wcy3,wcy5 are the weight for each code for cy3/cy5. For example: wcy3,wcy5=0.5,0.5 for code (1,2,3,4), while wcy3,wcy5=1,0 for code (1,3,5,9)
        Uppon subtraction this normalized score (mn_cd-mn_no) is thresholded by: self.background_threshold
        
        """
        self.internalize_paramaters()
        self.decoded_dic['background']={}
        decoded_dic=self.decoded_dic
        dic_cells = decoded_dic['cells']
        dic_cells_keys = dic_cells.keys()
        for i_key,key in enumerate(dic_cells_keys):
            dic_cell = dic_cells[key]
            code = dic_cell['code']
            code_py = np.array(code)-1
            scores = dic_cell['scores']
            
            non_code_py = np.setdiff1d(np.arange(len(scores)),code_py)
            non_code_py_cy3 = non_code_py[non_code_py%2==0]
            non_code_py_cy5 = non_code_py[non_code_py%2==1]
            mn_no_cy3 = np.mean(scores[non_code_py_cy3])
            mn_no_cy5 = np.mean(scores[non_code_py_cy5])
            mn_no = np.mean([mn_no_cy3 if cd%2==0 else mn_no_cy5 for cd in code_py])
            mn_cd = np.mean(scores[code_py])
            
            #if no bits are clearly above background (0.5 std above mean) then suspect background cross_corr
            #if np.sum(np.isnan(scores_code)|(scores_code>0.5))==0:
            if mn_cd-mn_no<self.background_threshold:#This has been obtained empirically from training set.
                self.decoded_dic['background'][key]=dic_cells.pop(key)
        self.decoded_dic['cells']=dic_cells

    def decoded_unbackground(self):
        """This undos the self.decoded_background"""
        decoded_dic = self.decoded_dic
        dic_cells = decoded_dic['cells']
        dic_background = decoded_dic.get('background',{})
        for key in dic_background.keys():
            dic_cells[key]=dic_background[key]
            del dic_background[key]
        self.decoded_dic = decoded_dic
    def decoded_MOE(self):
        self.maskMOEfov = self.im_MOEmask(self.fov_name)
        decoded_dic=self.decoded_dic
        dic_cells = decoded_dic['cells']
        dic_cells_keys = dic_cells.keys()
        for i_key,key in enumerate(dic_cells_keys):
            center = dic_cells[key]['center_reduced']
            xc,yc,zc=map(int,center)
            z,x,y =  map(int,map(np.mean,self.get_big_xyz([[yc,xc,zc]])))
            dic_cells[key]['MOE']=self.maskMOEfov[x,y]
            
    def decoded_overlap_v2(self):
        dic_cells = self.decoded_dic['cells']
        cells = list(dic_cells.keys())
        maskMOEfov = self.im_MOEmask(self.fov_name)
        from copy import deepcopy
        ### Construct convex hulls as main cell objects and populate them with usefull info
        parm_dic = deepcopy(self.decoded_dic['paramaters'])
        parm_dic['txys'] = simplify_txys(parm_dic['txys'],self.ref_fl)
        for cell in cells:
            dic_ = dic_cells[cell]
            X = dic_['zxy_signal_semiloose']
            conv = ConvexHull_(X)
            #populate
            conv.paramaters = parm_dic
            conv.code = np.array(dic_['code'])
            conv.fov = self.fov_name.split('.')[0]
            conv.cell = cell
            data_fld = self.data_folder
            if type(data_fld) is not str: data_fld = data_fld[0]
            conv.cell_name = conv.cell+'-__-'+os.path.basename(data_fld)+'-__-'+conv.fov
            conv.cor = dic_['mean_crosscor']
            for key in ['mean_crosscor','mean_bk','scores','zxy_signal_loose',
                        'zxy_signal_tight','mn_std_skew_kurt_nopix_bk','mn_std_skew_kurt_nopix','olfr','sublibrary']:
                setattr(conv,key,dic_.get(key,None))
            
            conv.MOE = maskMOEfov[tuple(conv.c.astype(int)[1:])]
            if len(X)>self.semiloosepoints_min and conv.volume>self.volume_min:
                conv.remove=False
            else:
                conv.remove=True
            
            dic_['conv']=conv
        
        ### Check for intersections and if intersect, choose best correlation score
        print("Checking overlap...")
        Xconvs = [dic_cells[cell]['conv'] for cell in cells]    
        for conv1 in tqdm(Xconvs):
            for conv2 in Xconvs:
                if not np.all(conv1.code==conv2.code):
                    compare(conv1, conv2, th_fr=0.5, res=10) 
        Xconvs = [conv for conv in Xconvs if (not conv.remove)]
        ### Check for remaining intersections and if intersect, choose biggest volume
        for conv1 in tqdm(Xconvs):
            for conv2 in Xconvs:
                if not np.all(conv1.code==conv2.code):
                    compare_volume(conv1, conv2, th_fr=0.5, res=10)
        Xconvs = [conv for conv in Xconvs if (not conv.remove)]
        self.Xconvs = Xconvs
    def decoded_refine(self):
        Xconvs = self.Xconvs
        from scipy.ndimage.morphology import binary_dilation,binary_erosion
        from skimage.morphology import ball

        im_cells = np.zeros([self.sz_image,self.sx_image,self.sy_image],np.uint16)
        for iconv,conv in enumerate(tqdm(Xconvs)):
            X = get_points_inside(conv).astype(int)
            conv.pts_inside = X
            im_cells[X[:,0],X[:,1],X[:,2]] = iconv+1
        im_bk = (1-binary_dilation(im_cells>0,iterations=5)).astype(bool)
        im_mask = self.im_MOEmask(self.fov_name)>0
        im_bk = im_bk*im_mask[np.newaxis]

        im_bk = (1-binary_dilation(im_cells>0,iterations=5)).astype(bool)
        maskMOEfov = self.im_MOEmask(self.fov_name)
        im_mask = maskMOEfov>0
        im_mask = binary_erosion(im_mask,iterations=10)
        im_bk = im_bk*im_mask[np.newaxis]
        background_kernel = ball(int(self.cell_diameter*1.25))
        background_kernel = np.array(np.where(background_kernel)).T
        background_kernel = background_kernel-np.expand_dims(np.mean(background_kernel,axis=0),0)

        for iconv,conv in enumerate(tqdm(Xconvs)):
            code=conv.code-1
            Xi = conv.pts_inside

            #get background
            center = np.mean(conv.points,0)
            background_ball = np.array(background_kernel+center,dtype=int)
            sz = np.array([self.sz_image,self.sx_image,self.sy_image])
            keep = np.all((background_ball<sz)&(background_ball>0),-1)
            background_ball = background_ball[keep]
            keep = im_bk[background_ball[:,0],background_ball[:,1],background_ball[:,2]]>0
            Xbk = background_ball[keep]

            traces = self.ims_matrix[:self.nRs,Xi[:,0],Xi[:,1],Xi[:,2]].copy()
            traces_n = traces - np.mean(traces,-1)[:,np.newaxis]
            traces_n = traces_n / np.std(traces_n,-1)[:,np.newaxis]
            conv.cor = np.mean([np.mean(traces_n[i]*traces_n[j]) for i in code for j in code if i>j])
            set_ = np.arange(len(traces_n))
            conv.cor_average = np.mean([np.mean(traces_n[i]*traces_n[j]) for i in set_ for j in set_ if i>j])
            conv.mean_traces = np.mean(traces,-1)
            conv.std_traces = np.std(traces,-1)
            

            traces = self.ims_matrix[:self.nRs,Xbk[:,0],Xbk[:,1],Xbk[:,2]].copy()
            traces_n = traces - np.mean(traces,-1)[:,np.newaxis]
            traces_n = traces_n / np.std(traces_n,-1)[:,np.newaxis]
            conv.cor_bk = np.mean([np.mean(traces_n[i]*traces_n[j]) for i in code for j in code if i>j])
            set_ = np.arange(len(traces_n))
            conv.cor_average_bk = np.mean([np.mean(traces_n[i]*traces_n[j]) for i in set_ for j in set_ if i>j])
            conv.mean_traces_bk = np.mean(traces,-1)
            conv.std_traces_bk = np.std(traces,-1)
            conv.scores = conv.mean_traces/conv.std_traces_bk
        for conv in Xconvs: 
            del conv.pts_inside
        self.Xconvs=Xconvs
    def display_convs_singleR(self,Rs=[1],save=True,fr=1.25):
        if Rs is None:
            iRs = range(self.nRs)
        else:
            iRs=np.array(Rs)-1
        for iR in iRs:
            X2ds=[]
            im_ = self.ims_matrix[iR]
            mins=[]
            maxs=[]
            texts = []
            for conv in self.Xconvs:
                conv_code = conv.code-1
                if iR in conv_code:
                    X = get_points_inside(conv,res=20).astype(int)
                    trace = im_[tuple(X.T)]
                    mins.append(np.min(trace))
                    maxs.append(np.max(trace))
                    X2d = get_2d_line(conv,axs=[1,2])
                    X2ds.append(X2d)
                    txt = conv.olfr+'\n'+str(conv.code)
                    texts.append(txt)
            vmin,vmax=np.median(mins),np.median(maxs)
            vmin,vmax = (vmin+vmax)/2-(vmax-vmin)/2*fr,(vmin+vmax)/2+(vmax-vmin)/2*fr
            f = plt.figure(figsize=(20,20))
            plt.imshow(np.max(im_,0),cmap='gray',vmin=vmin,vmax=vmax)#vmax=3)
            for iconv,X2d in enumerate(X2ds):
                plt.plot(X2d[:,1],X2d[:,0],'b',alpha=0.5)
                plt.text(np.mean(X2d[:,1]),np.mean(X2d[:,0]),texts[iconv],color='r')

            if save:
                save_folder = self.save_folder+os.sep+'Decoded'+os.sep+self.fov_name.replace('.dax','')
                if not os.path.exists(save_folder): os.makedirs(save_folder)
                save_file = save_folder+os.sep+'R'+str(iR+1).zfill(2)+'.png'
                f.savefig(save_file)
                plt.close(f)
    def get_dic_int_v2(self,h_cutoff = 1.35,refit =True):
        #load from file if does not have Xconvs
        if not hasattr(self,'Xconvs'):
            save_folder = self.save_folder+os.sep+'Decoded'
            save_file = save_folder+os.sep+self.fov_name.replace('.dax','__decoded_dic_v2.npy')
            self.Xconvs = np.load(save_file)
        
        for conv in tqdm(self.Xconvs):
            pad=3
            m,M = conv.m,conv.M
            m=m.astype(int)-pad
            M=M.astype(int)+pad
            m[m<0]=0

            iEGRs = [self.nRs,self.nRs+1]
            EGRtags = ['cfos','EGR1']

            egr1_cfos_tag = os.path.basename(os.path.dirname(conv.paramaters['dic_name'][0]))
            if 'cfos' not in egr1_cfos_tag.lower():
                iEGRs = [iEGRs[-1]]
                EGRtags = [EGRtags[-1]]
            ibest_score = (conv.code-1)[np.argmax(conv.scores[conv.code-1])]
            for iEGR,EGRtag in zip(iEGRs,EGRtags):
                im_EGR1 = self.ims_matrix[iEGR,m[0]:M[0],m[1]:M[1],m[2]:M[2]]
                im_EGR1__ = im_EGR1
                if refit:
                    im_rd = self.ims_matrix[ibest_score,m[0]:M[0],m[1]:M[1],m[2]:M[2]]
                    im_EGR1_ = im_EGR1/np.median(im_EGR1)-1
                    im_rd_ = im_rd/np.median(im_rd)-1
                    fr__ = 0
                    z_kp,x_kp,y_kp = (conv.points.astype(int)-m).T
                    fr__ = np.median(im_EGR1_[z_kp,x_kp,y_kp])/np.median(im_rd_[z_kp,x_kp,y_kp])
                    if fr__<0: fr__=0
                    im_EGR1__ = im_EGR1_ - im_rd_*fr__*1.5+1
                zxyh = get_local_max(im_EGR1__,th_fit=h_cutoff)
                zxyh[:3] = zxyh[:3]+m[:,np.newaxis]
                keep_in = in_hullF(zxyh[:3].T,conv)>-1
                zxyh = zxyh[:,keep_in].T
                setattr(conv,'zxyh_'+EGRtag+'_semiloose',zxyh)
    def decoded_save_v2(self,save_fov=False,save_cells=False,save_dic=True):
        #save self.Xconvs -- main save
        save_folder = self.save_folder+os.sep+'Decoded'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_file = save_folder+os.sep+self.fov_name.replace('.dax','')+'__decoded_dic_v2.npy'
        np.save(save_file,self.Xconvs)
        
        #save self.decoded_dic
        dic_cells = self.decoded_dic['cells']
        for cell in dic_cells:
            if "conv" in dic_cells[cell]:
                del dic_cells[cell]["conv"]
                
        if save_dic:
            save_file = save_folder+os.sep+self.fov_name.replace('.dax','')+'__decoded_dic.pkl'
            pickle.dump(self.decoded_dic,open(save_file,'wb'),protocol=pickle.HIGHEST_PROTOCOL)
        
        if save_fov:
            self.display_convs_singleR(Rs=None,save=True,fr=1.25)
        if save_cells:
            save_folder = self.save_folder+os.sep+'Decoded\cells'
            if not os.path.exists(save_folder):os.makedirs(save_folder)
            get_vmins_vmax(self)
            for conv in self.Xconvs:
                save_file = save_folder+os.sep+self.fov_name.replace('.dax','')+'__'+conv.olfr+'__'+conv.cell
                conv.fl_check = save_file
                plot_decoded_cell_3d_v2(self,conv,std_th=20,save_file=save_file,cmap='gray')
    def decoded_save(self,save_fov=False,save_cells=False,plot_3d=True):
        save_folder = self.save_folder+os.sep+'Decoded'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_file = save_folder+os.sep+self.fov_name.replace('.dax','__decoded_dic.pkl')
        pickle.dump(self.decoded_dic,open(save_file,'wb'),protocol=pickle.HIGHEST_PROTOCOL)
        if save_fov:
            save_folder = self.save_folder+os.sep+'Decoded'+os.sep+self.fov_name.replace('.dax','')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_file = save_folder+os.sep+'fov.png'
            self.plot_decoded_fov(self.decoded_dic,save_file=save_file)
        if save_cells:
            #save_good_cells
            save_folders = [self.save_folder+os.sep+'Decoded'+os.sep+self.fov_name.replace('.dax','')+os.sep+'good',
                            self.save_folder+os.sep+'Decoded'+os.sep+self.fov_name.replace('.dax','')+os.sep+'noMOE']
            for save_folder in save_folders:
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
            dic_cells = self.decoded_dic['cells']
            for key in dic_cells.keys():
                dic_cell = dic_cells[key]
                save_folder = save_folders[0] if dic_cell.get('MOE',1)>0 else save_folders[1]
                save_file=save_folder+os.sep+dic_cell['olfr']+'_'+key+'.png'
                if plot_3d:
                    self.plot_decoded_cell_3d(dic_cell,save_file=save_file,std_th=20,interpolation='nearest',cmap=cm.gray)
                else:
                    self.plot_decoded_cell_2d(dic_cell,save_file=save_file,std_th=20,interpolation='nearest',cmap=cm.gray)
            #save_background_cells
            save_folder = self.save_folder+os.sep+'Decoded'+os.sep+self.fov_name.replace('.dax','')+os.sep+'background'
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            dic_cells = self.decoded_dic.get('background',{})
            for key in dic_cells.keys():
                dic_cell = dic_cells[key]
                save_file=save_folder+os.sep+dic_cell['olfr']+'_'+key+'.png'
                if plot_3d:
                    self.plot_decoded_cell_3d(dic_cell,save_file=save_file,std_th=20,interpolation='nearest',cmap=cm.gray)
                else:
                    self.plot_decoded_cell_2d(dic_cell,save_file=save_file,std_th=20,interpolation='nearest',cmap=cm.gray)
            #save_overlaped_cells
            save_folder = self.save_folder+os.sep+'Decoded'+os.sep+self.fov_name.replace('.dax','')+os.sep+'overlap'
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            dic_cells = self.decoded_dic.get('overlap',{})
            for key in dic_cells.keys():
                if not self.decoded_dic['cells'].has_key(key):
                    dic_cell = dic_cells[key]
                    save_file=save_folder+os.sep+dic_cell['olfr']+'_'+key+'.png'
                    if plot_3d:
                        self.plot_decoded_cell_3d(dic_cell,save_file=save_file,std_th=20,interpolation='nearest',cmap=cm.gray)
                    else:
                        self.plot_decoded_cell_2d(dic_cell,save_file=save_file,std_th=20,interpolation='nearest',cmap=cm.gray)       
    def plot_decoded_fov(self,decoded_dic,save_file=None):
        """This plots the decoded image of the field of view extractiong information from self.ims_matrix"""
        sz_reduced=decoded_dic['paramaters']['sz_reduced']
        sx_reduced=decoded_dic['paramaters']['sx_reduced']
        sy_reduced=decoded_dic['paramaters']['sy_reduced']
        mask_olfr = np.zeros([sz_reduced,sx_reduced,sy_reduced],dtype=np.uint8)
        dic_cells = decoded_dic['cells']
        dic_cells_keys = dic_cells.keys()
        centers=[]
        for key in dic_cells_keys:
            X = dic_cells[key]['coords_reduced']
            x,y,z=X.T
            mask_olfr[z,x,y]=1
            centers.append(dic_cells[key]['center_reduced'])
        
        fig = plt.figure(figsize=(12,12))
        if len(centers)>0:
            x,y,z = zip(*centers)
            for x_,y_,key in zip(x,y,dic_cells_keys):
                plt.text(y_,x_,key,bbox={'facecolor':'white', 'alpha':0.5, 'pad':0})
        
        plt.imshow(np.mean(mask_olfr,axis=0),cmap=cm.hot)
        if save_file is None:
            plt.show()
        else:
            fig.savefig(save_file)
            plt.close(fig)
    
    def plot_decoded_cell_3d(self,cell_dic,std_th=20,save_file=None,**imshow_vars):
        """This plots the decoded image across bits extractiong information from self.ims_matrix
        Use as: self.plot_decoded_cell(self.decoded_dic['cells'][somekey],std_th=20,interpolation='nearest',cmap=cm.hot)"""
        yt,xt,zt = cell_dic['coords_reduced'].T
        cd = cell_dic['code']
        olfr = cell_dic['olfr']
        z_big,x_big,y_big = self.get_big_xyz(zip(xt,yt,zt))
        xm,xM = np.min(x_big),np.max(x_big)
        ym,yM = np.min(y_big),np.max(y_big)
        zm,zM = np.min(z_big),np.max(z_big)
        im_mask=np.zeros([zM-zm+1,xM-xm+1,yM-ym+1])
        range_mask_z = np.array(z_big)
        for zmsk,xmsk,ymsk in zip(np.array(z_big)-zm,np.array(x_big)-xm,np.array(y_big)-ym):
            im_mask[zmsk,xmsk,ymsk]=1
        """
        z_in_xy = max(int(self.nm_per_pixel_z/self.nm_per_pixel_xy),1)
        kernel = ball(self.ball_kernel_sz)[::z_in_xy,:,:]
        im_mask_=fftconvolve(im_mask,kernel)>np.sum(kernel)*self.ball_th_per
        im_mask_=im_mask_[kernel.shape[0]/2:-kernel.shape[0]/2,kernel.shape[1]/2:-kernel.shape[1]/2,kernel.shape[2]/2:-kernel.shape[2]/2]
        """
        ims_plt=[]
        edge_col = 0.25
        def xyz_project(im_3d,func=np.max):
            if func is None:
                im_plt_xy = im_3d[im_3d.shape[0]/2,...]
                im_plt_xz = im_3d[:,im_3d.shape[1]/2,...]
                #im_plt_xy=im_3d[int(np.mean(z_big)-zm),...]
                #im_plt_xz=im_3d[:,int(np.mean(y_big)-ym),...]
            else:
                im_plt_xy = func(im_3d,axis=0)
                im_plt_xz = func(im_3d,axis=1)
            #rep = np.arange(0,im_plt_norm.shape[0],self.nm_per_pixel_xy/float(self.nm_per_pixel_z))
            #rep = np.array(rep,dtype=int)
            #im_plt_xz=im_plt_xz[rep,...]
            pad_t = np.ones([1,im_plt_xz.shape[1]])*edge_col
            #print [im_plt_xy.shape,pad_t.shape,im_plt_xz.shape],im_3d.shape
            im_plt = np.concatenate([im_plt_xy,pad_t,im_plt_xz],axis=0)
            return im_plt
        for i in range(len(self.ims_matrix)):
            base_im = np.array(self.ims_matrix[i][zm:zM:,xm:xM,ym:yM])#.copy()
            im_plt = base_im-np.mean(base_im)
            im_plt_norm = im_plt/np.std(im_plt)
            im_plt_norm = gt.minmax(im_plt_norm,max_=std_th)
            im_plt = xyz_project(im_plt_norm,func=np.max)
            im_plt = gt.pad_im(im_plt,pad_=3,pad_val_=1 if (i+1 in cd) else edge_col)
            ims_plt.append(im_plt)
        im_plt = xyz_project(im_mask[:-1,:-1,:-1],func=np.mean)
        ims_plt.append(gt.pad_im(im_plt,pad_=3,pad_val_=edge_col))
        if 'DAPI0' in self.dic_fov:
            im_plt_3d = gt.minmax(self.dic_fov['DAPI0'][zm:zM:,xm:xM,ym:yM])
            im_plt = xyz_project(im_plt_3d,func=None)
            ims_plt.append(gt.pad_im(im_plt,pad_=3,pad_val_=edge_col))
        fig = plt.figure(figsize=(12,12))
        plt.imshow(gt.stitch(ims_plt),**imshow_vars)
        plt.title(olfr+'_'+str(cd))
        if save_file is None:
            plt.show()
            plt.close(fig)
        else:
            fig.savefig(save_file)
            plt.close(fig)
    def plot_decoded_cell_2d(self,cell_dic,std_th=15,save_file=None,**imshow_vars):
        """This plots the decoded image across bits extractiong information from self.ims_matrix
        Use as: self.plot_decoded_cell(self.decoded_dic['cells'][somekey],std_th=20,interpolation='nearest',cmap=cm.hot)"""
        yt,xt,zt = cell_dic['coords_reduced'].T
        cd = cell_dic['code']
        olfr = cell_dic['olfr']
        z_big,x_big,y_big = self.get_big_xyz(zip(xt,yt,zt))
        xm,xM = np.min(x_big),np.max(x_big)
        ym,yM = np.min(y_big),np.max(y_big)
        zm,zM = np.min(z_big),np.max(z_big)
        im_mask=np.zeros([zM-zm+1,xM-xm+1,yM-ym+1])
        range_mask_z = np.array(z_big)
        for zmsk,xmsk,ymsk in zip(np.array(z_big)-zm,np.array(x_big)-xm,np.array(y_big)-ym):
            im_mask[zmsk,xmsk,ymsk]=1
        """
        z_in_xy = max(int(self.nm_per_pixel_z/self.nm_per_pixel_xy),1)
        kernel = ball(self.ball_kernel_sz)[::z_in_xy,:,:]
        im_mask_=fftconvolve(im_mask,kernel)>np.sum(kernel)*self.ball_th_per
        im_mask_=im_mask_[kernel.shape[0]/2:-kernel.shape[0]/2,kernel.shape[1]/2:-kernel.shape[1]/2,kernel.shape[2]/2:-kernel.shape[2]/2]
        """
        ims_plt=[]
        for i in range(len(self.ims_matrix)):
            base_im = np.array(self.ims_matrix[i][zm:zM:,xm:xM,ym:yM])#.copy()
            im_plt = base_im-np.mean(base_im)
            im_plt_norm = im_plt/np.std(im_plt)
            
            im_plt = np.max(gt.minmax(im_plt_norm,max_=std_th),axis=0)
            im_plt = gt.pad_im(im_plt,pad_=3,pad_val_=(i+1 in cd))
            ims_plt.append(im_plt)
        im_plt = np.mean(im_mask[:-1,:-1,:-1],axis=0)
        ims_plt.append(gt.pad_im(im_plt,pad_=3,pad_val_=1))
        fig = plt.figure(figsize=(12,12))
        plt.imshow(gt.stitch(ims_plt),**imshow_vars)
        plt.title(olfr+'_'+str(cd))
        if save_file is None:
            plt.show()
        else:
            fig.savefig(save_file)
            plt.close(fig)
    def get_big_xyz_v2(self,X):
        X_red = np.array(X)
        spacing = np.array([self.spacing_xy,self.spacing_xy,self.spacing_z])
        pads = np.array([self.pad_xy,self.pad_xy,self.pad_z])
        maxs = np.max(X_red,0)
        mins = np.min(X_red,0)
        szb = (maxs-mins)*spacing+pads
        XB = np.indices(szb).reshape([3,-1]).T+mins*spacing
        Xred = X_red*spacing
        XB_ = XB/pads.astype(float)-(0.5-10**(-10))
        Xred_ = Xred/pads.astype(float)

        neigh = NearestNeighbors(n_neighbors=1,p=np.inf)
        neigh.fit(Xred_)
        dists,inds = neigh.kneighbors(XB_,1,return_distance=True)
        keep = dists[:,0]<=0.5
        XB = XB[keep]
        return XB.T[[2,0,1]]
    def get_big_xyz(self,xyz):
        """This takes a list of (x,y,z) and transforms it to a x,y,z in the big coordinates"""
        if not hasattr(self,'rangei_'):
            #do this once
            _,self.sz_,self.sx_,self.sy_ = self.ims_matrix.shape #big sizes
            self.rangei_ = np.arange(0,self.sx_-self.pad_xy+1,self.spacing_xy)
            self.rangej_ = np.arange(0,self.sy_-self.pad_xy+1,self.spacing_xy)
            self.rangek_ = np.arange(0,self.sz_-self.pad_z+1,self.spacing_z)
        maxs = np.max(xyz,0)
        mins = np.min(xyz,0)
        sz0 = (maxs-mins+1)*[self.pad_xy,self.pad_xy,self.pad_z]+1
        mask=np.zeros(sz0,dtype=bool)
        for ix,iy,iz in xyz:
            stx = self.rangei_[ix]-self.rangei_[mins[0]]
            sty = self.rangej_[iy]-self.rangej_[mins[1]]
            stz = self.rangek_[iz]-self.rangek_[mins[2]]
            xlist=np.arange(stx,stx+self.pad_xy)
            ylist=np.arange(sty,sty+self.pad_xy)
            zlist=np.arange(stz,stz+self.pad_z)
            mask[np.ix_(xlist,ylist,zlist)]=True
        x,y,z = np.where(mask)
        x,y,z = x+self.rangei_[mins[0]],y+self.rangej_[mins[1]],z+self.rangek_[mins[2]]
        return z,x,y
    def get_big_dimensions(self,i_=0,dim='x',tag = 'min'):
        """This takes a single index i_ and transforms it to the big coordinates"""
        if not hasattr(self,'rangei_'):
            #do this once
            self.sz_,self.sx_,self.sy_ = self.mask_corr.shape #big sizes
            self.rangei_ = range(0,self.sx_-self.pad_xy+1,self.spacing_xy)
            self.rangej_ = range(0,self.sy_-self.pad_xy+1,self.spacing_xy)
            self.rangek_ = range(0,self.sz_-self.pad_z+1,self.spacing_z)
        if dim=='x':
            pad = self.pad_xy
            spacing = self.spacing_xy
            range_ = self.rangei_
        elif dim=='y':
            pad = self.pad_xy
            spacing = self.spacing_xy
            range_ = self.rangej_
        elif dim=='z':
            pad = self.pad_z
            spacing = self.spacing_z
            range_ = self.rangek_
        if tag=='min': 
            return range_[i_]
        if tag=='max': 
            return range_[i_]+pad
        if tag=='range': 
            return range(range_[i_],range_[i_]+pad)            
    def cross_cors_3D_patch(self,save_file='auto',savePickle=True,overwrite=False):
        """Computes cross corelations for current field of view"""
        import cross_correlations2
        #Decide where to save:
        if save_file=='auto':
            save_folder = self.save_folder+os.sep+'CrossCorr'
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_file = save_folder+os.sep+self.fov_name.replace('.dax','.pkl')
        else:
            if save_file is not None:
                save_folder = os.path.dirname(save_file)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
        #Decide how to compute/load self.im_cross_cors
        if overwrite or (not os.path.exists(save_file)) or (save_file is None):
            start = time.time()
            self.im_cross_cors = cross_correlations2.cross_cors_3D_patch_fast(self.ims_matrix,self.mask_corr,
                    pad_xy=self.pad_xy,spacing_xy=self.spacing_xy,
                    pad_z=self.pad_z,spacing_z=self.spacing_z)
            if savePickle:
                pickle.dump(self.im_cross_cors,open(save_file,'wb'),protocol=pickle.HIGHEST_PROTOCOL)
            end = time.time()
            print "Time for 3D cross_cors: ",end - start
        else:
            self.im_cross_cors = pickle.load(open(save_file,'rb'))
        
    def cross_corrs_decode(self):
        """This decodes each pixel in the cross-cors matrix"""
        print "Please run cross_cors_2/3D_patch functions first."
        import cross_correlations2
        n_ims=self.nRs
        on_bits=self.nOn
        
        start = time.time()
        #Create indexes for pairwize combinations
        combs = list(itertools.combinations(range(n_ims),on_bits))
        pairs_ = cross_correlations2.get_pairs(n_ims)
        pairs__=[tuple(pr) for pr in pairs_]
        def pair_ind(val_): return pairs__.index(val_)
        ind_cmbs=[map(pair_ind,list(itertools.combinations(cmb,2))) for cmb in combs]
        self.im_decoded = np.mean(self.im_cross_cors[...,ind_cmbs],axis=-1)
        
        #####set the libraries to focus on
        if not hasattr(self,'library'):
            self.load_library(lib_keep=None)

        lib_keep = self.lib_keep
        #number of hybes in merfish and number of bits (default is 14,4 or 15,4)
        n_ims,n_onbits = self.nRs,self.nOn
        combs = map(list,list(itertools.combinations(range(n_ims),n_onbits)))#all codes
        keep_comb_index = [i_cmb for i_cmb,cmb in enumerate(combs) if cmb in self.codes_lib 
                   if self.index_lib[self.codes_lib.index(cmb)] in lib_keep]

        ##### end - now we got keep_comb_index and combs, the indexes of the codes in the libraries supplied and all the codes respectively

        #This is for only the libraries added
        im_argmax = np.argmax(self.im_decoded[...,keep_comb_index],axis=-1)#better not subtract background
        im_argmax = np.array(keep_comb_index)[im_argmax]
        self.im_argmax = im_argmax
        self.keep_comb_index = keep_comb_index
        self.im_med_bk_corr = np.mean(self.im_decoded[...],axis=-1)
        
        """
        ibackground = n_ims+2
        pairs_ =cross_correlations2.get_pairs(ibackground)
        pairs__ = map(tuple,pairs_)
        def pair_ind(val_): return pairs__.index(val_)
        ind_cmbs=[map(pair_ind,[(elem,ibackground-1) for elem in cmb]) for cmb in combs]
        self.im_background = np.mean(self.im_cross_cors[...,ind_cmbs],axis=-1)
        """
        end = time.time()
        print "Time for decoding: ",end - start
    def save_decoded(self):
        dic_save = {'im_cross_cors':getattr(self,'im_cross_cors'),'im_maxs':getattr(self,'im_maxs'),'im_mins':getattr(self,'im_mins'),'stages_z':getattr(self,'stages_z'),'target_z':getattr(self,'target_z')}
        pickle.dump(dic_save,open(self.save_folder+os.sep+self.fov_name+'.decoded','wb'),protocol=pickle.HIGHEST_PROTOCOL)
    def compute_drift_local(self):
        fl0 = self.dic_name[0]
        fls = np.setdiff1d(self.dic_name.values(),[fl0])
        for fl1 in fls:
            if not os.path.exists(self.save_folder+os.sep+'--'.join(fl1.split(os.sep)[-2:])+'.drft'):
                self.drift_corr_fl(fl0,fl1)
        plt.close('all')
    def compute_drift_batch(self,batch_size_=30,files=None):
        "This computes the registration of images within the dataset using the OR-MER_registration function which lives in functions."
        folder_functions = os.sep.join(gt.__file__.split(os.sep)[:-1])
        registration_script = folder_functions+os.sep+'OR-MER_Registration.py'
        if not os.path.exists(registration_script):
            print "Registation file not found in: "+folder_functions
            return None
        str_run_base = r'python '+registration_script+' '
        if files is None:
            files=self.files
        str_runs = gt.flatten([[str_run_base+fls[0]+' '+fl_t+' STORM6 '+self.save_folder for fl_t in fls[1:] if not os.path.exists(self.save_folder+os.sep+'--'.join(fl_t.split(os.sep)[-2:])+'.drft')] for fls in files])
        gt.batch_command(str_runs,batch_size=batch_size_,verbose=False)
        return str_runs
    
    def decode_batch(self,batch_size_=30,file_indexes=None):
        "This computes the decoding of images within the dataset using the OR-MER_Decoding function which lives in functions."
        folder_functions = os.sep.join(gt.__file__.split(os.sep)[:-1])
        script = folder_functions+os.sep+r'OR_MER_STD.py'
        if not os.path.exists(script):
            print "Decoding file not found at: "+script
            return None
        str_run_base = r'python '+script
        if file_indexes is None: file_indexes=range(len(self.files))
        str_runs = [str_run_base+' "'+self.data_folder+'" '+str(index_fov) for index_fov in file_indexes]
        if mock:
            return str_runs
        gt.batch_command(str_runs,batch_size=batch_size_,verbose=verbose)
        
    def EGR1_fit_batch(self,batch_size_=20,mock=False,verbose=False):
        "This computes the decoding of images within the dataset using the OR-MER_Decoding function which lives in functions."
        folder_functions = os.sep.join(gt.__file__.split(os.sep)[:-1])
        script = folder_functions+os.sep+r'OR-MER_EGR1fit.py'
        if not os.path.exists(script):
            print "EGR1 fitting file not found in: "+folder_functions
            return None
        str_run_base = r'python '+script+' '
        str_runs = [str_run_base+self.data_folder+' '+str(index_fov) for index_fov in range(len(self.files))]
        #str_runs = [str_run_base+self.data_folder+' '+self.save_folder+' '+self.device+' '+str(index_fov) for index_fov in range(len(self.files))]
                   #if not os.path.exists(self.save_folder+os.sep+str(index_fov)+'-completed_EGR1_batch.txt')]
        if mock:
            return str_runs
        else:
            gt.batch_command(str_runs,batch_size=batch_size_,verbose=verbose)
        

    def load_library(self,lib_keep=None):
        lib_fl = self.lib_fl
        lines = [ln for ln in open(lib_fl,'r')]
        lib_fls = [[ln for ln in lines if ln.count('_indexAB:'+str([2*i+2,2*i+1]))] for i in list(range(10))+[20]]
        import GeneralTools as gt
        import numpy as np
        names_simpl=[[gt.extract_flag(str_nm,'>','_')+'_'+str(list(np.array(eval(gt.extract_flag(str_nm,'_code','_').replace('(Steven):','').replace(':','')))+1))+'_'+str(i+1) 
                    for str_nm in lib_fls[i]] 
                        for i in range(len(lib_fls))]
        names_simpl = np.unique(gt.flatten(names_simpl))
        self.library = names_simpl
        self.codes_lib = [list(np.array(eval(nm.split('_')[1]))-1) for nm in self.library]
        self.codes_lib_p1 = [eval(nm.split('_')[1]) for nm in self.library]
        self.index_lib =[int(nm.split('_')[-1]) for nm in self.library]
        self.olfr_lib =[nm.split('_')[0] for nm in self.library]
        if lib_keep is None:
            # extract the index
            #std convention for getting the index of libraries used
            data_folder = self.data_folder
            if type(data_folder)!=str: data_folder = data_folder[0]
            if 'mer_lib' in os.path.basename(data_folder).lower():
                lib_keep = np.array(os.path.basename(data_folder).lower().split('mer_lib')[-1].split('_')[0].split(','),dtype=int)
            elif 'gfp' in os.path.basename(data_folder).lower():
                lib_keep = np.array([8])
            elif 'p2' in os.path.basename(data_folder).lower():
                lib_keep = np.array([4])
            #deal with naming mistake during synthesis
            remap_lib = np.array(self.lib_remap)
            lib_keep = remap_lib[lib_keep-1]
        self.lib_keep = lib_keep
        
        llib = [e for e in self.library if int(e.split('_')[-1]) in self.lib_keep]
        codes = np.array([eval(e.split('_')[1])for e in llib])
        Nnm = self.nRs
        codes_bin = []
        for cd in codes:
            blank = np.zeros(Nnm)
            blank[cd-1]=1
            codes_bin.append(blank)
        codes_bin = np.array(codes_bin)
        self.codes = codes
        self.codes_bin = codes_bin
    def drift_corr_fl(self,fl0,fl1,im_sz=[512,512],max_drft=100,med_filt_sz=50,save_image=True):
        im0 = io.DaxReader(fl0).loadAFrame(3)#laod first dapi frame
        im1 = io.DaxReader(fl1).loadAFrame(3)#laod first dapi frame
         
        im_=im0
        pairs_ij = gt.flatten([[(i_sz,j_sz)for i_sz in range(int(im_.shape[0]/im_sz[0]))] for j_sz in range(int(im_.shape[1]/im_sz[1]))])
        get_tile = lambda ij_sz: im_[ij_sz[0]*im_sz[0]:(ij_sz[0]+1)*im_sz[0],ij_sz[1]*im_sz[1]:(ij_sz[1]+1)*im_sz[1]]
        ims_tile = map(get_tile, pairs_ij)
        tile_i = np.argmax(map(np.mean,ims_tile))
        im0 =ims_tile[tile_i]
        im_ = im1
        im1 =get_tile(pairs_ij[tile_i])
        im0_50 = ndimage.filters.median_filter(im0, size=med_filt_sz)
        im1_50 = ndimage.filters.median_filter(im1, size=med_filt_sz)
        print "Computed median filters!"
        im0_sc=im0.astype(float)/im0_50
        im1_sc=im1.astype(float)/im1_50
        im0_sc=im0_sc-im0_sc.mean()
        im1_sc=im1_sc-im1_sc.mean()

        corr = signal.correlate2d(im0_sc, im1_sc[max_drft:-max_drft,max_drft:-max_drft], boundary='fill', mode='valid')
        print "Computed correlation!"
        y, x = np.unravel_index(np.argmax(corr), corr.shape)
        
        
        xt,yt=np.round(-np.array(corr.shape)/2.+[y,x]).astype(int)
        save_base = '--'.join(fl1.split(os.sep)[-2:])
        
        #save correlation image
        if save_image:
            f=plt.figure(figsize=(10,10))
            plt.imshow(corr,cmap=cm.hot)
            plt.plot(x,y,'ko')
            plt.title('DAPI-Crosscorrelation: \n'+str([xt,yt]))
            plt.savefig(self.save_folder+os.sep+save_base+'.png')
        #save correlation info
        fid=open(self.save_folder+os.sep+save_base+'.drft','w')
        fid.write(str(xt)+','+str(yt))
        fid.close()
    def to_red_coords(self,t,pad,spacing):
        sn_t = int(np.sign(t))
        t_=abs(t)
        if t_==0: return 0
        if t_>0 and t_<pad/2.: return sn_t*1
        if t_==pad/2.: return sn_t*2
        if t_>pad/2.: return sn_t*int(np.ceil((t_-pad/2.)/spacing)+1)
        
    def save_olfr_decoding(self,lib_keep=None,size_limit = 27,th_back=2.,th_corr=0.3,n_ims=14,n_onbits=4,pad=25,spacing=10,z_to_xy_pix=3):
        """depricated"""
        if lib_keep is None:
            lib_keep = np.array(gt.extract_flag(os.path.basename(self.data_folder),'lib','').split(','),dtype=int)
            remap_lib = np.array([1,2,3,4,5,7,8,9,10,6])
            lib_keep = remap_lib[lib_keep-1]
        combs = map(list,list(itertools.combinations(range(n_ims),n_onbits)))
        keep_comb_index = [i_cmb for i_cmb,cmb in enumerate(combs) if int(self.library[self.codes_lib.index(cmb)].split('_')[-1]) in lib_keep]
        
        #This is for all >1000 codes
        #im_argmax = np.argmax(self.im_decoded,axis=-1)
        #cds,cts=np.unique(im_argmax,return_counts=True) 
        
        #This is for only the libraries added
        im_argmax = np.argmax(self.im_decoded[...,keep_comb_index],axis=-1)
        im_argmax = np.array(keep_comb_index)[im_argmax]
        
        
        
        cds,cts=np.unique(im_argmax,return_counts=True)
        
        
        cds_cand = cds[np.argsort(cts)][::-1]
        run_stop = np.sum(cts>size_limit)
        cds_cand = [cd_ for cd_ in cds_cand[:run_stop] if cd_ in keep_comb_index]
        #loop through identified codes
        for comb_index in cds_cand:
            comb_val = combs[comb_index]
            name_probe = self.library[self.codes_lib.index(comb_val)]
            print name_probe
            library_ = int(name_probe.split('_')[-1])
            im_bw=im_argmax==comb_index
            
            #background filter
            n_ims_corr=n_ims+2
            pairs_ = gt.flatten([[(i,j) for i in range(n_ims_corr) if i<j] for j in range(n_ims_corr)])
            im_cross_back = np.mean(self.im_cross_cors[...,[pairs_.index((vl,n_ims_corr-2))for vl in comb_val]],axis=-1)
            im_corr_mask = self.im_decoded[...,comb_index]>th_corr
            im_bw = im_bw*(im_cross_back*(n_onbits-1)*n_onbits/2+th_back<self.im_decoded[...,comb_index])*im_corr_mask
            
            label_,n_label = label(im_bw)
            
            #pad the edges to remove drift effect
            ts_ = map(lambda t: self.to_red_coords(t,pad=pad,spacing=spacing),list(np.min(self.txys,axis=0))+list(np.max(self.txys,axis=0)))
            axis_ = [1,2,1,2]
            mask_drift = gt.mask_pad(np.ones(label_.shape),axis_,ts_,pad_val=0)
            label_ = label_*mask_drift
            
            #filter on size
            indexes = np.array(range(1,n_label+1))
            indexes_keep = indexes[ndi_sum(im_bw,label_,index=indexes)>size_limit]
            label_keep = np.sum([label_*0]+[(label_==ind_keep)*ind_keep for ind_keep in indexes_keep],axis=0)
            indexes = np.array(range(1,n_label+1))
            indexes_keep = indexes[ndi_sum(im_bw,label_,index=indexes)>size_limit]
            label_keep = np.sum([label_*0]+[(label_==ind_keep)*ind_keep for ind_keep in indexes_keep],axis=0)

            #plt.imshow(np.max(label_keep,axis=0))
            #plt.show()

            ims_ = [self.dic_fov[i+1] for i in range(14)]+[self.dic_fov[0]]
            background_im = self.dic_fov['background0']

            for ind_check in indexes_keep:#loop
                cm_z,cm_x,cm_y = center_of_mass(np.max(self.im_decoded,axis=-1)*(label_keep==ind_check))
                spacing_z = spacing/z_to_xy_pix
                pad_z = pad/z_to_xy_pix
                
                cx,cy,cz = int(cm_x*spacing+pad/2),int(cm_y*spacing+pad/2),int(cm_z*spacing_z+pad_z/2)
                #cz = int(1.*(max_z_range-min_z_range)/label_.shape[0]*cm_z)
                
                
                label_keep_ind = label_keep==ind_check
                mean_corrs_codes = np.mean(self.im_decoded[...,keep_comb_index][label_keep_ind],axis=0)
                height_ = np.sort(mean_corrs_codes)[-1]
                conf_ = np.sort(mean_corrs_codes)[-1]-np.sort(mean_corrs_codes)[-2]
                
                mean_corrs = np.mean(self.im_cross_cors[label_keep_ind],axis=0)
                
                median_background_pix = np.mean([self.im_cross_cors[:,:,:,n_ims*(n_ims-1)/2+i_cd][label_keep_ind] for i_cd in comb_val])
                median_EGR1_pix = np.mean([self.im_cross_cors[:,:,:,n_ims*(n_ims+1)/2+i_cd][label_keep_ind] for i_cd in comb_val])
                maxs_pixs = [np.median(self.im_maxs[:,:,:,i_cd][label_keep_ind]) for i_cd in comb_val]
                mins_pixs = [np.median(self.im_mins[:,:,:,i_cd][label_keep_ind]) for i_cd in comb_val]
                
                MOEmsk = self.im_MOEmask(self.dic_name[0])
                MOE_val = MOEmsk[cy,cx]
                mean_cors_dic = {pr:mean_corr for pr,mean_corr in zip(pairs_,mean_corrs)}
                keep_analyzed_dic = {'code':comb_val,'mean_cors_dic':mean_cors_dic,"MOE":MOE_val,"label":label_keep==ind_check,'index':ind_check,'zxy':[cz,cy,cx],'median_EGR1_pix':median_EGR1_pix,
                                     'median_background_pix':median_background_pix,'median_background':None,'maxs_pixs':maxs_pixs,'mins_pixs':mins_pixs,
                                     'area':np.sum(label_keep_ind),'mean_corrs_codes':mean_corrs_codes,'corr_height':height_,'corr_confidence':conf_,'target_z':self.target_z}
                decode_folder = self.save_folder+os.sep+'decoded'
                
                if not os.path.exists(decode_folder):
                    os.mkdir(decode_folder)
                olfr_pb_,cd_pb_,lib_pb_=name_probe.split('_')
                base_save_name = self.fov_name+'_cd'+cd_pb_+'_On'+olfr_pb_+'_lib'+lib_pb_+'_czxy'+str([cz,cy,cx])+'_i'+str(ind_check)+'_moe'+str(MOE_val)
                pickle.dump(keep_analyzed_dic,open(decode_folder+os.sep+base_save_name+'.olfr','wb'),protocol=pickle.HIGHEST_PROTOCOL)

                #Plotting: 
                further_saves = self.save_folder+os.sep+'prediction-auto', self.save_folder+os.sep+'background-auto'
                for further_save in further_saves:
                    if not os.path.exists(further_save):
                        os.mkdir(further_save)
                further_save=further_saves[0]#put everything in prediction-auto for now
                
                ims_f = ims_+[background_im]
                f = plt.figure(figsize=(20,20))
                minmax_ = lambda x: gt.mat2gray(x,perc_max=99.95)
                ims_adj = [gt.pad_im(minmax_(np.max(gt.grab_block(im__,[cz,cy,cx],[200,150,150]),axis=0)),pad_=10,pad_val_=i_im in combs[comb_index]) 
                           for i_im,im__ in enumerate(ims_f)]
                plt.imshow(gt.stitch(ims_adj),cmap=cm.hot)
                #plt.show()
                f.savefig(further_save+os.sep+base_save_name+'_xy.png')
                plt.close('all')

                f = plt.figure(figsize=(20,20))
                minmax_ = lambda x: gt.mat2gray(x,perc_max=99.95)
                minz_all = np.min(map(len,ims_f))
                ims_adj = [gt.pad_im(minmax_(np.max(gt.grab_block(im__[:minz_all],[cz,cy,cx],[200,50,150]),axis=1)),pad_=10,pad_val_=i_im in combs[comb_index]) 
                           for i_im,im__ in enumerate(ims_f)]
                plt.imshow(gt.stitch(ims_adj),cmap=cm.hot)
                f.savefig(further_save+os.sep+base_save_name+'_z.png')
                plt.close('all')
                #[min_z_range:max_z_range]

    def statistics_decoded_cells(self,corr_confidence_th=0.25,corr_height_th=2.5,corr_back_th=0.75,save_stats=True,reload_data=False):
        folder_decoded = self.save_folder+os.sep+'decoded'
        decoded_stats_fl = folder_decoded+os.sep+'decoded_stats.pkl'
        if os.path.exists(decoded_stats_fl) and not reload_data:
            dic_decoded_stats = pickle.load(open(decoded_stats_fl,'rb'))
            names_,x,y,area_,background_ = dic_decoded_stats['decoded_fnames'],dic_decoded_stats['corr_height'],dic_decoded_stats['corr_confidence'],dic_decoded_stats['area'],dic_decoded_stats['corr_background']
        else:
            #Load individual data
            decoded_files = glob.glob(folder_decoded+os.sep+'*.olfr')
            saves_ = []
            for dec_fl in decoded_files:
                dic_olfr = pickle.load(open(dec_fl,'rb'))
                saves_.append([dec_fl,dic_olfr['corr_height'],dic_olfr['corr_confidence'],dic_olfr['area'],dic_olfr['median_background']])
            #Historicl cumbersome deconstruction
            x,y=np.array([q[1] for q in saves_]),np.array([q[2] for q in saves_])
            names_ = np.array([q[0] for q in saves_])
            area_ = np.array([q[3] for q in saves_])
            background_ = np.array([q[4] for q in saves_])
        
        #Decide what to keep
        keep = (y>corr_confidence_th)&(x>corr_height_th)&(background_<corr_back_th)
        f=plt.figure(figsize=(15,15))
        plt.scatter(x[keep==False],y[keep==False],alpha=0.05)
        plt.scatter(x[keep],y[keep],alpha=0.05,color='r')
        plt.xlim([0,np.max(x)])
        plt.ylim([0,np.max(y)])
        plt.show()
        if save_stats:
            f.savefig(folder_decoded+os.sep+'Thresholded_image.png')
            dic_decoded_stats = {'decoded_fnames':names_,'corr_height':x,'corr_confidence':y,'area':area_,'corr_background':background_,'corr_confidence_th':corr_confidence_th,'corr_height_th':corr_height_th}
            pickle.dump(dic_decoded_stats,open(decoded_stats_fl,'wb'),protocol=pickle.HIGHEST_PROTOCOL)
            
            decoded_files = glob.glob(folder_decoded+os.sep+'*.olfr')
            for nm_dec in decoded_files:
                for xyz_tag in ['_xy.png','_z.png']:
                    nm_png = nm_dec.replace('.olfr',xyz_tag).replace('decoded','background-auto')
                    if not os.path.exists(nm_png):
                        nm_png = nm_png.replace('background-auto','prediction-auto')
                    if keep[list(names_).index(nm_dec)]:
                        os.rename(nm_png,nm_png.replace('background','prediction'))
                    else:
                        os.rename(nm_png,nm_png.replace('prediction','background'))
    def load_decoded_crosscor(self):
        fl_decoded = self.save_folder+os.sep+self.fov_name+'.decoded'
        print "Loading for fl. no. "+str(self.index_fov)+': '+fl_decoded
        dic_load = pickle.load(open(fl_decoded,'rb'))
        self.im_cross_cors = dic_load['im_cross_cors']
        self.cross_corrs_decode(n_ims=14,on_bits=4)
        self.im_mins = dic_load.get('im_mins',None)
        self.im_maxs = dic_load.get('im_maxs',None)
        self.stages_z = dic_load.get('stages_z',None)
        self.target_z = dic_load.get('target_z',None)
    def load_stage_xy_pix(self,fls_iter,pix_size=0.153):
        xys=[]
        for dax in fls_iter:
            dic_inf = io.readInfoFile(dax.replace('.dax','.inf'))
            xys.append([dic_inf['Stage X']/pix_size,dic_inf['Stage Y']/pix_size])
        return np.array(xys)
    #new functions for rough alignment
    def get_rough_alignment(self,box_cutoff=50,fine_align=50,neigh_cutoff=300,plt_val=False):
        """This computes rough alignment and stores it in Drift_rough subforlder of self.save_data and in self.rough_alignment_dics"""
        #load stage values
        ind_ref = self.index_fov
        self.dirnames,self.basenames,self.xys_2d = self.get_stage_xy_pix()
        self.rough_alignment_dics={}
        for target in self.dirnames:
            print 'Aligning: '+target
            save_file = self.save_folder+os.sep+'Drift_rough'+os.sep+target+'__'+self.basenames[ind_ref].replace('.dax','.drift')
            if os.path.exists(save_file):
                self.rough_alignment_dics[target]=pickle.load(open(save_file,'rb'))
            else:
                dapi_i = 2
                if '3col' in self.device.lower(): dapi_i = 3
                im_cor = self.correction_image(dapi_i,set_fl=-1,perc_=95,extra_note='',save_file='auto',overwrite=False)
                im_flat = im_cor/im_cor.mean()
                center_fl = self.save_folder+os.sep+'center_estimates.pkl'

                if os.path.exists(center_fl):
                     #load and internalize center_estimates paramaters
                    center_estimates = pickle.load(open(center_fl,'rb'))

                    ref = center_estimates['ref']
                    rough_align=center_estimates[target+'_cut']['rough_cut']
                    use_fft = center_estimates[target+'_cut']['use_fft']
                    center_align_dic = center_estimates[target]
                    inds_saved = np.array(center_align_dic.keys())
                    ind_saved = inds_saved[np.argmin(np.abs(inds_saved-ind_ref))]
                    center_align = center_align_dic[ind_saved]
                else:
                    print "Did not find center_estimates file at:"+center_fl
                    center_estimates={}
                    ref = self.dirnames[0]
                    rough_align = 40
                    use_fft = False
                    center_align = [0,0]
                #apply dapi aligment of center images
                
                im1_fl = self.data_folder+os.sep+ref+os.sep+self.basenames[ind_ref]
                im2_fl = self.data_folder+os.sep+target+os.sep+self.basenames[ind_ref]
                im1 = io.DaxReader(im1_fl).loadAFrame(3)/im_flat
                im2 = io.DaxReader(im2_fl).loadAFrame(3)/im_flat

                txy = self.fftalign_guess(im1,im2,center=center_align,max_disp=rough_align,use_fft=use_fft,normalized=True,plt_val=plt_val)

                center_estimates[target][ind_ref]=txy


                xys = self.xys_2d[self.dirnames.index(target)]

                sx,sy = im_flat.shape
                center_ref = xys[ind_ref]
                centers_neighs = np.array([[-sx,-sy],[0,-sy],[sx,-sy],[sx,0],[sx,sy],[0,sy],[-sx,sy],[-sx,0],[0,0]])
                centers_all = xys-np.expand_dims(center_ref,0)
                ind_neighs,xy_neighs=[],[]
                for center_neigh in centers_neighs:
                    dists = np.sum(np.abs(centers_all-np.expand_dims(center_neigh,0)),-1)
                    ind_neigh = np.argmin(dists)
                    if dists[ind_neigh]<neigh_cutoff:
                        ind_neighs.append(ind_neigh)
                        xy_neighs.append(centers_all[ind_neigh]+np.array(txy))
                #we have ind_neighs and xy_neighs
                def cent_to_box(center):
                    minx,miny=center
                    maxx,maxy=minx+sx,miny+sy
                    return [[minx,maxx],[miny,maxy]]
                box_ref = cent_to_box([0,0])
                boxes = map(cent_to_box,xy_neighs)
                box_inters = [gt.rect_intersect(box_ref,box) for box in boxes]

                keep_box=[]
                for box in box_inters:
                    box_x,box_y = box
                    keep=False
                    if len(box_x)>0 and len(box_y)>0:
                        if (np.max(box_x)-np.min(box_x))>box_cutoff and (np.max(box_y)-np.min(box_y))>box_cutoff:
                            keep=True
                    keep_box.append(keep)

                box_inters_ref = np.array([box for box,keep in zip(box_inters,keep_box) if keep],dtype=int)
                ind_inters = np.array([ind for ind,keep in zip(ind_neighs,keep_box) if keep])
                fls_inters = [self.data_folder+os.sep+target+os.sep+self.basenames[ind]for ind in ind_inters]

                boxes_inters = np.array([box for box,keep in zip(boxes,keep_box) if keep],dtype=int)

                def ensure_overlap(ref_slice,target_slice,ref_shift=[0,0],target_shift=[0,0]):
                    ref_slice_=np.array(ref_slice)
                    target_slice_=np.array(target_slice)

                    #apply shifts:
                    (minx,maxx),(miny,maxy) = ref_slice_
                    minx-=ref_shift[0]
                    maxx-=ref_shift[0]
                    miny-=ref_shift[1]
                    maxy-=ref_shift[1]
                    ref_slice_ = np.array([(minx,maxx),(miny,maxy)])
                    (minx,maxx),(miny,maxy) = target_slice_
                    minx-=target_shift[0]
                    maxx-=target_shift[0]
                    miny-=target_shift[1]
                    maxy-=target_shift[1]
                    target_slice_ = np.array([(minx,maxx),(miny,maxy)])


                    minx_ref,miny_ref = np.min(ref_slice_,axis=-1)
                    minx_target,miny_target = np.min(target_slice_,axis=-1)
                    #bounds
                    (minx,maxx),(miny,maxy)=ref_slice_
                    ref_slice_ = np.array([[min(max(0,minx),sx),min(max(0,maxx),sx)],
                                           [min(max(0,miny),sy),min(max(0,maxy),sy)]])
                    (minx,maxx),(miny,maxy)=target_slice_
                    target_slice_ = np.array([[min(max(0,minx),sx),min(max(0,maxx),sx)],
                                           [min(max(0,miny),sy),min(max(0,maxy),sy)]])
                    (minx,maxx),(miny,maxy)=target_slice_
                    shift_x,shift_y=minx_target-minx_ref,miny_target-miny_ref
                    target_slice_ref = np.array([(minx-shift_x,maxx-shift_x),(miny-shift_y,maxy-shift_y)])
                    ref_slice_ = np.array(gt.rect_intersect(ref_slice_,target_slice_ref))
                    minmaxx,minmaxy = ref_slice_
                    if len(minmaxx)==0 or len(minmaxy)==0:
                        return np.array([[0,0],[0,0]]),np.array([[0,0],[0,0]])
                    (minx,maxx),(miny,maxy)=ref_slice_
                    target_slice_ = np.array([(minx+shift_x,maxx+shift_x),(miny+shift_y,maxy+shift_y)])
                    return ref_slice_,target_slice_

                ref_slices,target_slices=[],[]
                for box_ref,box_int in zip(box_inters_ref,boxes_inters):
                    (minx,maxx),(miny,maxy) = box_ref
                    (minx_,maxx_),(miny_,maxy_) = box_int
                    ref_slice = np.array(box_ref)
                    target_slice = [[minx-minx_,maxx-minx_],
                                    [miny-miny_,maxy-miny_]]
                    ref_slice_,target_slice_=ensure_overlap(ref_slice,target_slice)
                    ref_slices.append(ref_slice_)
                    target_slices.append(target_slice_)
                #we have fls_inters,ref_slices,target_slices
                ref_slices_cor,target_slices_cor=[],[]
                ims2 = [io.DaxReader(fl).loadAFrame(3)/im_flat for fl in fls_inters]
                reconstr_im = np.zeros([sx,sy])
                for ref_slice,target_slice,im in zip(ref_slices,target_slices,ims2):
                    (minx,maxx),(miny,maxy) = ref_slice
                    (minx_,maxx_),(miny_,maxy_) = target_slice
                    im_ref = im1[minx:maxx,miny:maxy]
                    im_target = im[minx_:maxx_,miny_:maxy_]
                    txy = self.fftalign_guess(im_ref,im_target,center=[0,0],max_disp=fine_align,use_fft=False,normalized=False,plt_val=plt_val)
                    ref_slice_,target_slice_=ensure_overlap(ref_slice,target_slice,ref_shift=[0,0],target_shift=np.array(txy))
                    ref_slices_cor.append(ref_slice_)
                    target_slices_cor.append(target_slice_)
                    (minx,maxx),(miny,maxy) = ref_slice_
                    (minx_,maxx_),(miny_,maxy_) = target_slice_
                    #print txy
                    reconstr_im[minx:maxx,miny:maxy]=im[minx_:maxx_,miny_:maxy_]
                if plt_val:
                    plt.close('all')
                    f, ax_arr = plt.subplots(1,2, sharex=True, sharey=True)
                    axs = np.ravel(ax_arr)
                    axs[0].imshow(im1,interpolation='nearest',cmap=cm.gray)
                    axs[1].imshow(reconstr_im,interpolation='nearest',cmap=cm.gray)
                    plt.show()
                save_dic = {'target':target,'center_estimates':center_estimates,'ind_ref':ind_ref,
                    'ref_slices_cor':ref_slices_cor,'target_slices_cor':target_slices_cor,'fls_inters':fls_inters}
                self.rough_alignment_dics[target]=save_dic
                save_folder = os.path.dirname(save_file)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                pickle.dump(save_dic,open(save_file,'wb'))
    def get_stage_xy_pix(self):
        """
        Returns dirnames,basenames,xys_2d with the corrected xy values.
        dirnames,basenames,xys_2d = get_stage_xy_pix(or_chr)
        """
        xys_file = self.save_folder+os.sep+'xys.pkl'
        if os.path.exists(xys_file):
            dirnames,basenames,xys_2d = pickle.load(open(xys_file,'rb'))
            dirnames=[os.path.basename(dirname) for dirname in dirnames]
            pickle.dump((dirnames,basenames,xys_2d),open(xys_file,'wb'))
            return dirnames,basenames,xys_2d
        else:
            files = flatten(self.files)
            dirnames = list(np.unique([os.path.basename(os.path.dirname(fl)) for fl in files]))
            basenames = list(np.unique([os.path.basename(fl) for fl in files]))

            xys = self.load_stage_xy_pix(files,pix_size=0.153)
            xys = xys*np.expand_dims([1,-1],0)
            xys = xys[:,::-1]

            xys_2d=np.zeros([len(dirnames),len(basenames),2])+np.inf
            for fl,xy in zip(files,xys):
                dirname=os.path.basename(os.path.dirname(fl))
                basename=os.path.basename(fl)
                xys_2d[dirnames.index(dirname),basenames.index(basename),:]=xy
            pickle.dump((dirnames,basenames,xys_2d),open(xys_file,'wb'))
            return (dirnames,basenames,xys_2d)
    def fftalign_guess(self,im1,im2,center=[0,0],max_disp=50,use_fft=False,normalized=True,plt_val=False):
        """
        Inputs: 2 images im1, im2 and a maximum displacement max_disp.
        This computes the cross-cor between im1 and im2 using fftconvolve (fast) and determines the maximum
        """
        
        if not use_fft:
            from scipy.signal import fftconvolve
            im2_=np.array(im2[::-1,::-1],dtype=float)
            im2_-=np.mean(im2_)
            im1_=np.array(im1,dtype=float)
            im1_-=np.mean(im1_)
            im_cor = fftconvolve(im1_,im2_, mode='full')
        else:
            from numpy import fft
            im2_=np.array(im2,dtype=float)
            im2_-=np.mean(im2_)
            im1_=np.array(im1,dtype=float)
            im1_-=np.mean(im1_)
            f0, f1 = [fft.fft2(arr) for arr in (im1_,im2_)]
            # spectrum can be filtered, so we take precaution against dividing by 0
            eps = abs(f1).max() * 1e-15
            # cps == cross-power spectrum of im0 and im2
            #cps = abs(fft.ifft2((f0 * f1.conjugate()) ))
            if normalized:
                cps = abs(fft.ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1) + eps)))
            else:
                cps = abs(fft.ifft2((f0 * f1.conjugate()) ))
            # scps = shifted cps
            im_cor = fft.fftshift(cps)
        """
        im1__ = np.ones_like(im1)
        im2__ = np.ones_like(im2_)
        im_cor_norm = fftconvolve(im1__,im2__, mode='full')
        im_cor = im_cor/im_cor_norm
        """
        sx_cor,sy_cor = im_cor.shape
        
        center_ = np.array(center)+np.array([sx_cor,sy_cor])/2.
        
        x_min = int(min(max(center_[0]-max_disp,0),sx_cor))
        x_max = int(min(max(center_[0]+max_disp,0),sx_cor))
        y_min = int(min(max(center_[1]-max_disp,0),sy_cor))
        y_max = int(min(max(center_[1]+max_disp,0),sy_cor))
        
        im_cor0=np.zeros_like(im_cor)
        im_cor0[x_min:x_max,y_min:y_max]=1
        im_cor = im_cor*im_cor0
           
        y, x = np.unravel_index(np.argmax(im_cor), im_cor.shape)
        if np.sum(im_cor>0)>0:
            im_cor[im_cor==0]=np.min(im_cor[im_cor>0])
        else:
            im_cor[im_cor==0]=0
        if plt_val:
            plt.figure()
            plt.plot([x],[y],'k+')
            plt.imshow(im_cor,interpolation='nearest')
            plt.show()
        xt,yt=np.round(-np.array(im_cor.shape)/2.+[y,x]).astype(int)
        return xt,yt
    def neighbours(self,fls_iter,pix_size_=0.153,dist_cut=1024):
        import scipy.spatial.distance as dist
        xys = self.load_stage_xy_pix(fls_iter,pix_size=pix_size_)
        dists_=dist.pdist(xys)
        dists_=dist.squareform(dists_)
        dist_links = map(np.squeeze,map(np.where,dists_<dist_cut))
        return dist_links
    def return_intersection(self,fls_iter,ij,frms_keep=slice(2,None,3),pix_size_=0.153):
        i,j=ij
        if type(frms_keep) is slice:
            im_i=io.DaxReader(fls_iter[i]).loadAll()
            im_j=io.DaxReader(fls_iter[j]).loadAll()
            im_i=im_i[frms_keep]
            im_j=im_j[frms_keep]
        else:
            im_i=np.array([io.DaxReader(fls_iter[i]).loadAFrame(fr) for fr in frms_keep])
            im_j=np.array([io.DaxReader(fls_iter[i]).loadAFrame(fr) for fr in frms_keep])
        #im_i=np.max(im_i[frms_keep],axis=0)
        #im_j=np.max(im_j[frms_keep],axis=0)

        def adj_im(im): return np.swapaxes(im[:,1:-1:-1,1:-1:1,...],0,1)
        ims=map(adj_im,[im_i,im_j])
        xys_ = self.load_stage_xy_pix([fls_iter[i]for i in ij],pix_size=pix_size_)
        xys_=xys_-np.expand_dims(np.min(xys_,axis=0), axis=0)
        box_i = zip(xys_[0],xys_[0]+ims[0].shape[1:])
        box_j = zip(xys_[1],xys_[1]+ims[1].shape[1:])
        coords_int_i = np.round(np.array(gt.rect_intersect(box_i,box_j))-np.expand_dims(xys_[0],axis=1))
        coords_int_j = np.round(np.array(gt.rect_intersect(box_i,box_j))-np.expand_dims(xys_[1],axis=1))
        im_int_i = ims[0][:,coords_int_i[0][0]:coords_int_i[0][1],coords_int_i[1][0]:coords_int_i[1][1]]
        im_int_j = ims[1][:,coords_int_j[0][0]:coords_int_j[0][1],coords_int_j[1][0]:coords_int_j[1][1]]
        
        return [im_int_i,im_int_j]
    
    
    def save_tile_function(self,function_,fls_iter,save_file,resc=1,max_impose=True,verbose=True,pix_size_=0.153):
        """For EGR1 verif use as:
        or_chr.save_tile_function(or_chr.EGR1_verif_image,[fls[0] for fls in or_chr.files],or_chr.save_folder+os.sep+'EGR1_auto_mask.tiff',resc=2,max_impose=True,verbose=True)
        """
        import tifffile
        ims,xys=[],[]
        pix_size=pix_size_*resc

        for dax in fls_iter:
            if verbose: print dax
            dic_inf = io.readInfoFile(dax.replace('.dax','.inf'))
            xys.append([dic_inf['Stage X']/pix_size,dic_inf['Stage Y']/pix_size])
            ims.append(function_(dax)[::resc,::resc,...])
            
        xys_=np.array(xys,dtype=int)
        xys_=xys_-np.expand_dims(np.min(xys_,axis=0), axis=0)

        if len(ims[0].shape)>2:
            sx,sy,sz = ims[0].shape
            dim_base = list(np.max(xys_,axis=0)+np.array([sx+1,sy+1]))
            dim_base+=[sz]
        else:
            sx,sy = ims[0].shape
            dim_base = list(np.max(xys_,axis=0)+np.array([sx+1,sy+1]))

        im_base = np.zeros(dim_base,dtype=ims[0].dtype)
        infos_filename = ".".join(save_file.split('.')[:-1])+'.infos'
        fid = open(infos_filename,'w')
        for i_,(im,(x,y)) in enumerate(zip(ims[:],xys_[:])):
            im_ = np.swapaxes(im[::-1,::1,...],0,1)
            if max_impose:
                im_base[x:sx+x,y:sy+y,...]=np.max([im_base[x:sx+x,y:sy+y,...],im_],axis=0)
            else:
                im_base[x:sx+x,y:sy+y,...]=im_
            save_pars = [fls_iter[i_],x,sx+x,y,sy+y,resc]
            fid.write("\t".join(map(str,save_pars))+'\n')
        fid.close()
        self.im_tile = im_base
        tifffile.imsave(save_file,im_base)
    def save_tile_image(self,fls_iter,save_file,start_frame=1,resc=2,color_map=cm.brg,custom_frms=None,target_z=np.arange(-7.5,7.5,0.5),pix_size_=0.153,
                        max_impose=True,verbose=False,correction=True,correction_note='',rotation_angle=0):
        """
        This is intended as a tiler. Given a set of files <fls_iter>, it will create and save a tile image as 16bit png.
        #Save Dapi image:
        or_chr.save_tile_image([fls[0] for fls in or_chr.files],
                        or_chr.save_folder+os.sept+MOEMask+os.sep+'dapi_100x_resc4.tiff',
                        start_frame=3,resc=4,color_map=None,custom_frms=[3],target_z=None,pix_size_=0.153,max_impose=True,verbose=True,correction=True)
        #Save EGR1 image:
        or_chr.save_tile_image([fls[0] for fls in or_chr.files],
                        os.path.dirname(or_chr.data_folder)+os.sep+'EGR1_100x_resc2.tiff',
                        start_frame=1,resc=2,color_map=cm.brg,custom_frms=None,target_z=np.arange(-7.5,7.5,0.5),pix_size_=0.153,max_impose=True,verbose=True,correction=True)
        """
        import tifffile
        ims,xys=[],[]
        pix_size=pix_size_*resc
        fls_iter_ = fls_iter
        if verbose: fls_iter_ = tqdm(fls_iter)
        for dax in fls_iter_:
            

            dic_inf = io.readInfoFile(dax.replace('.dax','.inf'))
            
            #Read appropriate frames
            daxReader = io.DaxReader(dax)
            
            if custom_frms is None: 
                frms = self.best_stage_vals(dax, target_z, start=start_frame)
                dapi_im = daxReader.loadAll()
                dapi_im = dapi_im[start_frame::3][frms]
            ## faster way to load this above
            else:
                dapi_im = [daxReader.loadAFrame(frm) for frm in custom_frms]
            
            
            dapi_im_small  = np.array(dapi_im)[:,1:-1:resc,1:-1:resc]
            
            #Illumination correction:
            if correction:
                im_cor = self.correction_image(start_frame,set_fl=fls_iter,perc_=95,save_note=str(start_frame),save_file='auto',overwrite=False)
                im_cor = im_cor[1:-1:resc,1:-1:resc,...].astype(float)
                im_cor = im_cor/np.median(im_cor)
                while len(dapi_im_small.shape)!=len(im_cor.shape): 
                    im_cor = np.expand_dims(im_cor,0)
                dapi_im_small = dapi_im_small/im_cor
                
            #Apply color
            if color_map is None:
                dapi_im_small = np.max(dapi_im_small,axis=0)
            else:
                dapi_im_rep = np.tile(dapi_im_small,[3,1,1,1])
                col_arr =np.array([color_map(fr)[:3] for fr in np.linspace(0,1,len(dapi_im_small))]).T
                col_arr = np.expand_dims(np.expand_dims(col_arr,-1),-1)
                dapi_im_small = np.dstack(np.max(dapi_im_rep*col_arr,axis=1))
            
            #Consider adding illumination correction and better stitching
            xys.append([dic_inf['Stage X']/pix_size,dic_inf['Stage Y']/pix_size])
            ims.append(dapi_im_small)
            

        xys_=np.array(xys,dtype=int)
        xys_=xys_-np.expand_dims(np.min(xys_,axis=0), axis=0)

        if len(ims[0].shape)>2:
            sx,sy,sz = ims[0].shape
            dim_base = list(np.max(xys_,axis=0)+np.array([sx+1,sy+1]))
            dim_base+=[sz]
        else:
            sx,sy = ims[0].shape
            dim_base = list(np.max(xys_,axis=0)+np.array([sx+1,sy+1]))
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        im_base = np.zeros(dim_base,dtype=float)
        infos_filename = '.'.join(save_file.split('.')[:-1])+'.infos'
        fid = open(infos_filename,'w')
        for i_,(im,(x,y)) in enumerate(zip(ims[:],xys_[:])):
            #im_ = np.swapaxes(im[::-1,::1,...],0,1)
            im_ = np.swapaxes(im[::1,::-1,...],0,1)
            if self.device == 'STORM6':
                im_ = im_[::-1,::-1]
            if self.device == 'STORM6_V2':
                im_ = im_[::-1,::-1]
            if rotation_angle!=0:
                im_ = rotate(im_, rotation_angle, center=None, scale=1.0)
            if max_impose:
                im_base[x:sx+x,y:sy+y,...]=np.max([im_base[x:sx+x,y:sy+y,...],im_],axis=0)
            else:
                im_base[x:sx+x,y:sy+y,...]=im_
            save_pars = [fls_iter[i_],x,sx+x,y,sy+y,resc]
            fid.write("\t".join(map(str,save_pars))+'\n')
        fid.close()
        
        tifffile.imsave(save_file,np.clip(im_base,0,2**16-1).astype(np.uint16))
    
    def EGR1_verif_image(self,fl):
        if not hasattr(self,'list_cells'):
            self.list_cells = np.array(pickle.load(open(self.save_folder+os.sep+'EGR1_filter_cells.pkl','rb')))
        ind_file = np.where(np.array([cell['fl'] for cell in self.list_cells])==fl)[0]
        sxy = gt.frame_dimensions(fl)
        im_base = np.zeros(list(sxy)+[3],dtype=np.uint8)
        if len(ind_file):
            list_cells_file = self.list_cells[ind_file]
            for cell in list_cells_file:
                x_,y_,z_=cell['x'],cell['y'],cell['z']
                if len(x_)>=3:
                    arr_,start_=self.hull_array(zip(x_,y_),resc=1)
                    col=np.array([0,255,0])
                    if not cell['filter']:
                        col+=[255,0,0]
                    if not cell['MOE']:
                        col+=[0,0,255]
                    for col_i in range(3):
                        im_base[start_[0]:start_[0]+arr_.shape[0],start_[1]:start_[1]+arr_.shape[1],col_i]+=arr_*col[col_i]
        return im_base
    def hull_array(self,coords,resc=1):
        coords_=np.array(coords)
        if coords_.shape[-1] !=2:
            from scipy.spatial import ConvexHull
            import scipy.ndimage as ndim
            
            hull = ConvexHull(coords)
            coords_hull = hull.points
            mins_ = np.min(coords_hull,axis=0).astype(int)
            maxs_ = np.max(coords_hull,axis=0).astype(int)
            arr_ = np.mgrid[[slice(min_,max_,resc) for min_,max_ in zip(mins_,maxs_)]]
            
            #bottleneck is the function bellow
            def in_hull(p,hull_=hull):
                new_points = np.append(hull_.points, [p], axis=0)
                new_hull = ConvexHull(new_points)
                if list(hull_.points) == list(new_hull.points):
                    return True
                else:
                    return False
            arr_small = np.apply_along_axis(in_hull,0,arr_)
            #bottleneck
            
            arr_f = ndim.zoom(arr_small, resc, order=0)
            return arr_f,mins_
        if coords_.shape[-1]==2:
            from cv2 import convexHull
            result = convexHull(coords_)
            x_hull,y_hull=zip(*np.squeeze(result))
            coords_hull=np.array(zip(x_hull,y_hull))
            #above will change for 3D
            
            mins_ = np.min(coords_hull,axis=0).astype(int)
            maxs_ = np.max(coords_hull,axis=0).astype(int)
            arr_ = np.mgrid[[slice(min_,max_,resc) for min_,max_ in zip(mins_,maxs_)]]
            
            #the bellow function will change for 3D
            
            def in_path(points_arr,coords_hull_=coords_hull):
                import matplotlib.path as path
                path_ = path.Path(coords_hull_)
                s_ = points_arr.shape
                points_=np.reshape(points_arr,[s_[0],np.prod(s_[1:])]).T
                return np.reshape(path_.contains_points(points_),s_[1:])
            arr_small = in_path(arr_)
            import scipy.ndimage as ndim
            arr_f = ndim.zoom(arr_small, resc, order=0)
            return arr_f,mins_

    
    def filter_EGR1fits(self,h_max = 30000,dist_cut = 20,fr_cut = 0.5,folder=None):
        """"
        This is intended after running the fits to gather and cleanup the dbscan clusters and decide the MOE of each cell.
        It will look for pixels higher than h_max in the dax_fit. If the coordinates of fr_cut percent of a group of smFISH signals are too close to 'hot' pixels
        tag as a bad cell.
        
        Returns a dictionary with MOE filed and filter fields added
        Uses filter_EGR1fit for single field of view and troubleshooting puposes.
        """
        def filter_(fl): return self.filter_EGR1fit(fl,h_max = h_max,dist_cut = dist_cut,fr_cut = fr_cut,folder=folder)
        if folder is None: folder = self.save_folder
        files_ = glob.glob(folder+os.sep+'*.cellfits')
        dics_cells = map(filter_,files_)
        dics_cells = gt.flatten(dics_cells)
        pickle.dump(dics_cells,open(self.save_folder+os.sep+'EGR1_filter_cells.pkl','wb'),protocol=pickle.HIGHEST_PROTOCOL)
        return dics_cells

    def loadMOEmasks(self,img_filenames=None,infos_filename=None,dil=10):
        """This assumes the analysis folder has a subfolder MOEMask with files *mask_1.png,*mask_2.png,..."""
        from cv2 import imread,dilate
        if img_filenames is None:
            print "Assuming standard naming for masks."
            img_filenames = glob.glob(self.save_folder+os.sep+'MOEMask'+os.sep+'*mask_*')
        #dilation kernel
        y,x = np.ogrid[-dil:dil, -dil:dil]
        kernel = x*x + y*y < dil*dil
        MOE_masks = []

        for i_,img_filename in enumerate(img_filenames):
            img = imread(img_filename,0)
            img_dil = dilate(img,kernel.astype(np.uint8),iterations = 1)
            MOE_masks.append((img_dil>0)*(i_+1))
        dims = np.min([msk.shape for msk in MOE_masks],0)
        MOE_masks = np.max([msk[:dims[0],:dims[1]] for msk in MOE_masks],axis=0)
        if infos_filename is None:
            print "Assuming standard naming for infofile."
            infos_filename = glob.glob(self.save_folder+os.sep+'MOEMask'+os.sep+'*.infos')[0]
        self.MOE_masks = MOE_masks
        self.MOE_masks_infofl = infos_filename
        
        fl_moe_split = self.save_folder+os.sep+'MOEMask'+os.sep+"MOE_fov_split.tif"
        name,x_min,x_max,y_min,y_max,resc = self.load_coords_dax(self.MOE_masks_infofl)
        im_mask_all = []
        for index_nm in range(len(name)):
            sx_t,sy_t = self.sy_image,self.sx_image
            #load the image
            im_mask_ = self.MOE_masks[int(x_min[index_nm]):int(x_max[index_nm]),int(y_min[index_nm]):int(y_max[index_nm])]
            #rescale
            from cv2 import resize,INTER_NEAREST
            im_mask = resize(im_mask_,(sx_t,sy_t),interpolation = INTER_NEAREST)
            im_mask = im_mask[::1,::-1].T
            if 'STORM65' in self.device :
                im_mask = im_mask[::-1,::-1]
            if 'STORM6_BLANK' in self.device.upper():
                im_mask = im_mask[::1,::1]
            im_mask_all.append(im_mask)
        im_mask_all = np.array(im_mask_all,dtype=np.uint8)
        tifffile.imwrite(fl_moe_split,im_mask_all)
    def im_MOEmask(self,nm):
        """Get the MOE mask for the nm field of view"""
        #load the MOE mask if does not exist
        fl_moe_split = self.save_folder+os.sep+'MOEMask'+os.sep+"MOE_fov_split.tif"
        self.fl_moe_split = fl_moe_split
        if not os.path.exists(fl_moe_split):
            self.loadMOEmasks()
            #this populates self.MOE_masks and self.MOE_masks_infofl
        #load the infofile
        self.MOE_masks_infofl = glob.glob(self.save_folder+os.sep+'MOEMask'+os.sep+'*.infos')[0]
        #Decide the target image size
        name,x_min,x_max,y_min,y_max,resc = self.load_coords_dax(self.MOE_masks_infofl)
        name_simple = [os.path.basename(nm_).split('.')[0] for nm_ in name]
        
        index_nm = name_simple.index(os.path.basename(nm).split('.')[0])
        im_mask = tifffile.imread(fl_moe_split,key=index_nm)
        
        return im_mask
    def load_coords_dax(self,filename):
        """
        Read info file of a saved tile.
        Returns list of names, coordinates for each name in the savetile image and rescale.
        """
        fid = open(filename,'r')
        txt_arr = np.array([ln.split('\t') for ln in fid])
        fid.close()
        name,x_min,x_max,y_min,y_max,resc = txt_arr.T
        x_min = np.array(x_min,dtype=float)
        x_max = np.array(x_max,dtype=float)
        y_min = np.array(y_min,dtype=float)
        y_max = np.array(y_max,dtype=float)
        resc = np.array(resc,dtype=float)
        return name,x_min,x_max,y_min,y_max,resc

    def im_mask_return(self,nms,maskfl,infofl,return_im=True):
        import tifffile
        mask = tifffile.imread(maskfl)
        name,x_min,x_max,y_min,y_max,resc = self.load_coords_dax(infofl)

        name_simpl = list(map(os.path.basename,name))
        nms_simpl = map(os.path.basename,nms)

        index_nms = [name_simpl.index(nm_simpl) for nm_simpl in nms_simpl]

        #load the image
        im_masks_ = [mask[int(x_min[index_nm]):int(x_max[index_nm]),int(y_min[index_nm]):int(y_max[index_nm])] for index_nm in index_nms]
        #rescale
        if return_im is False:
            return [np.sum(im_mask_)>0 for im_mask_ in im_masks_]
        else:
            from cv2 import resize,INTER_NEAREST
            #Decide the target image size
            im_masks=[]
            for index_nm,im_mask_ in zip(index_nms,im_masks_):
                sx_t = int((x_max[index_nm]-x_min[index_nm])*resc[0])
                sy_t = int((y_max[index_nm]-y_min[index_nm])*resc[0])
                im_mask = resize(im_mask_,(sx_t,sy_t),interpolation = INTER_NEAREST)
                im_masks.append(im_mask[::1,::-1].T)
            return im_masks
    def filter_EGR1fit(self,fl_,h_max = 30000,dist_cut = 20,fr_cut = 0.5,nbad=500,folder=None,plot_val=False):
        """This filters the EGR1 fits
        This loads first the file outputed fl_ by self.EGR1_fit. 
        It then seaches very hot regions (>h_max) in the normalized EGR1 dax.
        If a points in a cluster is closer than dist_cut to these hot pixels, they are maked as bad.
        If there are more than nbad bad points in the cluster or more than fr_cut% are bad, that cluster is rejected.
        """
        print "Dealing with "+fl_
        list_cells = pickle.load(open(fl_,'rb'))
        
        if len(list_cells):
            cell_init = list_cells[0]
            #load fitted dax and find very hot pixels
            dax_fl = cell_init['fit_fl']
            im_EGR1=io.DaxReader(dax_fl).loadAll()
            z_max,x_max,y_max = np.where(im_EGR1>h_max)
            rescz = float(self.nm_per_pixel_xy)/self.nm_per_pixel_z
            z_max=z_max*rescz #rescale to fit to standard z
            
            from scipy import spatial
            if len(z_max)>0: tree = spatial.KDTree(zip(z_max,x_max,y_max))
            else: tree=None
                
            # decide whether the cell is good (if too close to high pixels return False)
            def good_cell(cell,tree=tree,dist_cut = dist_cut,fr_cut = fr_cut,nbad=nbad):
                cell_val = True
                if tree is not None:
                    x_,y_,z_ = cell['x'],cell['y'],cell['z']
                    pts_ = zip(z_,x_,y_)
                    neighs = tree.query(pts_,distance_upper_bound=dist_cut)
                    ngood = np.sum(1./neighs[0]==0)
                    if (len(x_)-ngood)>nbad:
                        cell_val = False
                    if float(ngood)/len(x_)<fr_cut:
                        cell_val = False
                return cell_val
            
            list_good = map(good_cell,list_cells)
            for dic,good in zip(list_cells,list_good): 
                dic['filter']=good
                
            # decide where in the MOE the cell is.
            im_mask = self.im_MOEmask(cell_init['fl'])
            for dic in list_cells: 
                x_MOE,y_MOE = int(np.median(dic['x'])),int(np.median(dic['y']))
                dic['MOE']= im_mask[x_MOE,y_MOE]
            #save to file
            pickle.dump(list_cells,open(fl_,'wb'))
        if plot_val:
            for i_,cell in enumerate(list_cells):
                plt.text(np.median(cell['y']),np.median(cell['x']),str(i_)+'_'+str(cell['MOE']),color='w')
                if cell['filter']:
                    plt.plot(cell['y'],cell['x'],'b.')
                else:
                    plt.plot(cell['y'],cell['x'],'r.')
            #load base
            im_ = io.DaxReader(cell_init['fl']).loadAll()

            im_egr1 = gt.minmax(np.max(im_[1::3],axis=0),max_=8000)
            im_dapi = gt.minmax(im_[3::3],max_=30000)
            im_plt = np.dstack([im_egr1,im_dapi[0],0.2*im_mask])
            plt.imshow(im_plt)
            plt.plot(y_max,x_max,'g.')
            plt.show()
        return list_cells
    def Olfr_counts_expr(self,pltMOE=False,fpkm_file=None):
        """
        After decoding the data, use this to calculate the number of neuros decoded and the aproximate expression/cell.
        fpkm_file=r'C:\Data\Transcriptomes\mouse\OlfrLoganData.csv'
        This stores data in: self.olfrs_names,self.MOE_counts,self.MOE_counts_split,self.exprs_approx
        
        """
        if not hasattr(self,'MOE_counts_split'):
            decoded_files = glob.glob(self.save_folder+os.sep+'Decoded'+os.sep+'*decoded_dic.pkl')
            dic_counts={}
            dic_signal={}
            for decoded_file in decoded_files:
                decoded_dic = pickle.load(open(decoded_file,'rb'))
                dic_cells = decoded_dic['cells']
                keys_cells = dic_cells.keys()
                for key in keys_cells:
                    cell = dic_cells[key]
                    MOE = cell['MOE']
                    olfr = cell['olfr']
                    if not dic_counts.has_key(MOE):
                        dic_counts[MOE]={}
                    dic_counts[MOE][olfr]=dic_counts[MOE].get(olfr,0)+1

                    code = np.array(cell['code'])-1
                    signal,background,area = cell['mn_std_skew_kurt_nopix'][code,0],cell['mn_std_skew_kurt_nopix_bk'][code,0],cell['mn_std_skew_kurt_nopix'][code,-1]
                    if dic_signal.has_key(olfr):
                        dic_signal[olfr]['signal'].append(signal)
                        dic_signal[olfr]['background'].append(background)
                        dic_signal[olfr]['area'].append(area)
                    else:
                        dic_signal[olfr]={'code':code,'signal':[],'background':[],'area':[]}
            MOEs = dic_counts.keys()
            olfrs=[]
            for MOE in MOEs:
                olfrs = np.union1d(dic_counts[MOE].keys(),olfrs)
            MOE_counts = []
            for i,MOE in enumerate(MOEs):
                MOE_counts.append([])
                for olfr in olfrs:
                    MOE_counts[i].append(dic_counts[MOE].get(olfr,0))
            self.MOE_counts_split = MOE_counts
            self.olfrs_names = olfrs
            self.MOE_counts = np.sum(MOE_counts,axis=0)
        if pltMOE:
            plt.figure()
            plt.loglog(self.MOE_counts_split[0],self.MOE_counts_split[1],'o')
            plt.xlabel("Counts in MOE 1")
            plt.ylabel("Counts in MOE 2")
            plt.show()
        
        if not hasattr(self,'exprs_approx'):
            #calculate approximate single cell expression - probably needs work
            exprs_approx=[]
            for olfr in olfrs:
                arr = np.array(dic_signal[olfr]['signal'])/np.array(dic_signal[olfr]['background'])-1
                if len(arr)==0:
                    exprs_approx.append((0,0))
                else:
                    exprs_approx.append((max(np.nanmean(arr,axis=0)),np.std(np.nanmean(arr,axis=0))))
                    #print max(np.nanmean(arr,axis=0)),np.std(np.nanmean(arr,axis=0),axis=0)
            self.exprs_approx = np.array(exprs_approx)
        if fpkm_file is not None:
            fid = open(fpkm_file,'r')
            lines = [ln for ln in fid]
            fid.close()
            expr_dic = {ln.split(',')[1]:float(ln.split(',')[-3]) for ln in lines[2:]}
            fpkm = [expr_dic.get(olfr,0) for olfr in self.olfrs_names]
            plt.loglog(self.MOE_counts*self.exprs_approx[:,0],fpkm,'o')
            plt.xlabel("Estimated total expression")
            plt.ylabel("FPKM")
            plt.show()
    def save_local_maxima_EGR1(self,th_fit = 1.5):
        im_EGR1 =  self.ims_matrix[-1].copy()
        im_bad = im_EGR1==0
        im_EGR1[im_EGR1==0]=np.median(im_EGR1[im_EGR1!=0])
        im_EGR1_ = gauss_div1(im_EGR1,gb=150)
        zxyh_0 = np.array(get_local_max(im_EGR1_,th_fit = th_fit))
        save_folder_ = self.save_folder+os.sep+'EGR1_fits'
        if not os.path.exists(save_folder_):
            os.makedirs(save_folder_)
        save_file_ = save_folder_+os.sep+self.fov_name.replace('.dax','_EGR1_zxyh.npy')
        np.save(save_file_,zxyh_0)
        if 'cfos' in os.path.basename(os.path.dirname(self.dic_name[0])).lower():
            im_EGR1 =  self.ims_matrix[-2].copy()
            im_bad = im_EGR1==0
            im_EGR1[im_EGR1==0]=np.median(im_EGR1[im_EGR1!=0])
            im_EGR1_ = gauss_div1(im_EGR1,gb=150)
            zxyh_0 = np.array(get_local_max(im_EGR1_,th_fit = th_fit))
            save_folder_ = self.save_folder+os.sep+'EGR1_fits'
            if not os.path.exists(save_folder_):
                os.makedirs(save_folder_)
            save_file_ = save_folder_+os.sep+self.fov_name.replace('.dax','_cfos_zxyh.npy')
            np.save(save_file_,zxyh_0)
    def EGR1_Olfr_intersection_statistics(self,n_inters_cutoff=1,plt_val=True):
        "Computes intersection statistics and plots a fancy bar plot"
        save_folder = self.save_folder+os.sep+'EGR1_Intersection'
        inters_files = glob.glob(save_folder+os.sep+'*.pkl')
        dic_intersection = {}
        for inters_file in inters_files:
            if not os.path.exists(save_folder+os.sep+'Bad'+os.sep+os.path.basename(inters_file).replace('.pkl','.png')):
                inters_dic = pickle.load(open(inters_file,'r'))
                olfr = inters_dic['dic_olfr']['olfr']
                nEGR1 = len(inters_dic['dic_EGR1']['x'])
                if dic_intersection.has_key(olfr):
                    dic_intersection[olfr]['counts']+=1
                    dic_intersection[olfr]['nsEGR1'].append(nEGR1)
                else:
                    dic_intersection[olfr]={'counts':1,'nsEGR1':[nEGR1]}
            else:
                pass
        for olfr in dic_intersection:
            dic_intersection[olfr]['nEGR1']=np.mean(dic_intersection[olfr]['nsEGR1'])
        tot_counts,inters_counts,nEGR1s,olfrs,stdEGR1s = [],[],[],[],[]
        if not hasattr(self,'MOE_counts'):
            self.Olfr_counts_expr(pltMOE=False,fpkm_file=None)
        for olfr in dic_intersection.keys():
            tot_count = self.MOE_counts[list(self.olfrs_names).index(olfr)]
            inters_count = dic_intersection[olfr]['counts']
            nEGR1 = dic_intersection[olfr]['nEGR1']
            stdEGR1 = np.std(dic_intersection[olfr]['nsEGR1'])
            tot_counts.append(tot_count)
            inters_counts.append(inters_count)
            nEGR1s.append(nEGR1)
            stdEGR1s.append(stdEGR1)
            olfrs.append(olfr)
        inds = np.argsort(inters_counts)[::-1]
        keep = np.sort(inters_counts)[::-1]>n_inters_cutoff
        tot_counts=np.array(tot_counts)[inds][keep]
        inters_counts=np.array(inters_counts)[inds][keep]
        nEGR1s=np.array(nEGR1s)[inds][keep]
        stdEGR1s=np.array(stdEGR1s)[inds][keep]
        olfrs=np.array(olfrs)[inds][keep]
        if plt_val:
            N = len(tot_counts)
            ind = np.arange(N)  # the x locations for the groups
            width = 0.2       # the width of the bars

            fig, ax = plt.subplots()
            rects1 = ax.bar(ind, inters_counts, width, color='r')
            rects2 = ax.bar(ind + width, tot_counts, width, color='g')#,yerr=women_std)
            rects3 = ax.bar(ind + 2*width, nEGR1s, width, color='b',yerr=stdEGR1s)
            rects = [rects1,rects2,rects3]
            # add some text for labels, title and axes ticks
            ax.set_ylabel('Counts')
            ax.set_title('EGR1-Olfr Intersection')
            ax.set_xticks(ind + 2*width / 3.)
            ax.set_xticklabels(olfrs)

            ax.legend((rects1[0], rects2[0],rects3[0]), ('Intersection counts', 'Total counts', 'EGR1 number'))


            def autolabel(rects):
                """
                Attach a text label above each bar displaying its height
                """
                for rect in rects:
                    height = rect.get_height()
                    ax.text(rect.get_x() + 2*rect.get_width()/3., height+2,
                            '%d' % int(height),
                            ha='center', va='bottom')

            autolabel(rects1)
            autolabel(rects2)
            autolabel(rects3)

            plt.show()
    def EGR1_Olfr_intersection_filter(self):
        """This uses gt.imshow_pngs to load and clasify PNGs in self.save_folder+os.sep+'EGR1_Intersection'. 
        Post clasification the good/bad pngs are copied to Good/Bad subfolder"""
        inters_pngs = glob.glob(self.save_folder+os.sep+'EGR1_Intersection'+os.sep+'*.png')
        save_file = self.save_folder+os.sep+'EGR1_Intersection'+os.sep+'EGR1_intersection.train'
        imshow_pngs_ = gt.imshow_pngs(inters_pngs, index_good=set(), index_bad=set(),save_file=save_file, use_cv2=True, screen_size=1000)
        
        save_folder_good = self.save_folder+os.sep+'EGR1_Intersection'+os.sep+'Good'
        save_folder_bad = self.save_folder+os.sep+'EGR1_Intersection'+os.sep+'Bad'
        
        print "Saving png files to Good/Bad subfolders"
        for save_folder in [save_folder_good,save_folder_bad]:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
        import shutil
        for ind_good in imshow_pngs_.index_good:
            shutil.copy2(imshow_pngs_.files[ind_good],save_folder_good)
        for ind_bad in imshow_pngs_.index_bad:
            shutil.copy2(imshow_pngs_.files[ind_bad],save_folder_bad)
    def EGR1_Olfr_intersection(self,EGR1_approx='interpolation',type_intersection='2d',intersection_dist_cutoff = 2.,plt_val=True,saveTOfile=True):
        """
        This loads the decoded file and the EGR1 clustered file and finds the intersection by thresholding the inter-distacnce with intersection_dist_cutoff
        type_intersection='2d' or '3d'. 2d is recommended as the z is inacurate
        EGR1_approx='interpolation' uses the interpolation approximation to compute from EGR1 fits to reduced coordinates - self.to_red_coords
        EGR1_approx='extension' uses the extension to compute from EGR1 fits to reduced coordinates - internal
        If saveTOfile this saves the intersection info in a dictionary with keys: 'area_EGR1','area_olfr','area_overlap','distance','dic_EGR1','dic_olfr'
        Use as:
        import OR_MER_Analyser as oma
        reload(oma)
        or_chr = oma.OR_cropper()
        or_chr.load_data(r'X:\09_19_2016-OR-MER\SI8-lib2,3')
        for or_chr.index_fov in range(len(or_chr.files)):
            or_chr.EGR1_Olfr_intersection(intersection_dist_cutoff = 2.5,EGR1_approx='interpolation',type_intersection='2d',plt_val=False,saveTOfile=True)
        """
        self.fov_name = os.path.basename(self.files[self.index_fov][0]).replace('.zst','')
        #internalize paramters
        self.EGR1_approx = EGR1_approx
        self.type_intersection = type_intersection
        self.intersection_dist_cutoff=intersection_dist_cutoff
        #load decoded file
        
        decoded_fl = self.save_folder+os.sep+'Decoded'+os.sep+self.fov_name.replace('.dax','__decoded_dic.pkl')
        prev_decoded_fl = getattr(self,'decoded_fl','')
        if prev_decoded_fl!=decoded_fl:
            self.decoded_fl=decoded_fl
            if os.path.exists(decoded_fl):
                self.decoded_dic = pickle.load(open(decoded_fl,'rb'))
            else:
                self.decoded_dic=None
        #load EGR1 file
        EGR1_file = self.save_folder+os.sep+'EGR1Analysis'+os.sep+'EGR1'+self.fov_name.replace('.dax','.cellfits')
        prev_EGR1_file = getattr(self,'decoded_fl','')
        if prev_EGR1_file!=EGR1_file:
            if os.path.exists(EGR1_file):
                self.list_EGR1 = pickle.load(open(EGR1_file,'rb'))
            else:
                self.list_EGR1=None
        if self.decoded_dic is None or self.list_EGR1 is None:
            print "Could not find decoded file or EGR1 file. Make sure to analyze those."
            return None
        dic_paramaters = self.decoded_dic['paramaters']
        sx_reduced,sy_reduced,sz_reduced=dic_paramaters['sx_reduced'],dic_paramaters['sy_reduced'],dic_paramaters['sz_reduced']
        dic_cells = self.decoded_dic['cells']
        cell_keys = dic_cells.keys()
        cells_centrs,cells_reds = [],[]
        for key in cell_keys:
            dic_cell = dic_cells[key]
            olfr = dic_cell['olfr']
            x_cell,y_cell,z_cell = zip(*dic_cell['coords_reduced'])
            cells_reds.append(dic_cell['coords_reduced'])
            cells_centrs.append(dic_cell['center_reduced'])
        cells_centrs = np.array(cells_centrs)

        def to_red_single(t,pad,spacing):
            index_right = int(t/spacing)+1
            index_left = int(np.ceil(float(t-pad)/spacing))
            if index_left<0: index_left=0
            return np.arange(index_left,index_right)
        def to_red(xyz,pad_xy=10,spacing_xy=25,pad_z=8,spacing_z=3):
            xf,yf,zf=[],[],[]
            for x,y,z in xyz:
                xs = to_red_single(x,pad_xy,spacing_xy)
                ys = to_red_single(y,pad_xy,spacing_xy)
                zs = to_red_single(z,pad_z,spacing_z)
                xs,ys,zs = map(np.ravel,np.meshgrid(xs,ys,zs))
                xf.extend(xs)
                yf.extend(ys)
                zf.extend(zs)
            return zip(xf,yf,zf)
        rescz = float(dic_paramaters['nm_per_pixel_xy'])/float(dic_paramaters['nm_per_pixel_z'])
        EGR1_centrs,EGR1_reds=[],[]
        list_EGR1_keep=[]
        for EGR1_dic in self.list_EGR1:
            if EGR1_dic['MOE']>0 and EGR1_dic['filter']:
                if self.EGR1_approx=='interpolation':
                    x_red=[self.to_red_coords(t,dic_paramaters['pad_xy'],dic_paramaters['spacing_xy']) for t in EGR1_dic['y']]
                    y_red=[self.to_red_coords(t,dic_paramaters['pad_xy'],dic_paramaters['spacing_xy']) for t in EGR1_dic['x']]
                    z_red=[self.to_red_coords(t,dic_paramaters['pad_z'],dic_paramaters['spacing_z']) for t in EGR1_dic['z']/rescz]
                    xyz_red_EGR1=zip(x_red,y_red,z_red)
                else:
                    xyz_EGR1 = zip(EGR1_dic['y'],EGR1_dic['x'],EGR1_dic['z']/rescz)
                    xyz_red_EGR1 = to_red(xyz_EGR1,pad_xy=dic_paramaters['pad_xy'],spacing_xy=dic_paramaters['spacing_xy'],
                                         pad_z=dic_paramaters['pad_z'],spacing_z=dic_paramaters['spacing_z'])
                EGR1_reds.append(xyz_red_EGR1)
                EGR1_centrs.append(np.mean(xyz_red_EGR1,axis=0))
                list_EGR1_keep.append(EGR1_dic)
        EGR1_centrs = np.array(EGR1_centrs)
        if plt_val:
            plt.figure()
            for EGR1_red in EGR1_reds:
                x_red,y_red,z_red = zip(*EGR1_red)
                plt.plot(np.array(x_red),np.array(y_red),'r.')
            for centr in EGR1_centrs:
                plt.plot([centr[0]],[centr[1]],'ro')
            
            for key in cell_keys:
                dic_cell = dic_cells[key]
                olfr = dic_cell['olfr']
                if True:#olfr=='Olfr56':
                    x_cell,y_cell,z_cell = zip(*dic_cell['coords_reduced'])
                    plt.plot(x_cell,y_cell,'g+')
            for centr in cells_centrs:
                plt.plot([centr[0]],[centr[1]],'go')
            im_plt=np.ones([sx_reduced,sy_reduced])
            im_plt[0,0]=0
            plt.imshow(im_plt,cmap=cm.gray)
            plt.show()
        
        # cross-distance and intersection
        from scipy.spatial import KDTree
        import shutil
        #decide whether to do xy searching, z might not be accurate enough
        inds_EGR1,inds_cells=[],[]
        if len(cells_centrs)>0 and len(EGR1_centrs)>0:
            end_dim = int(self.type_intersection.replace('d',''))
            tree = KDTree(cells_centrs[:,:end_dim])
            dists_,inds_ = tree.query(EGR1_centrs[:,:end_dim])
            dist_cutoff = self.intersection_dist_cutoff
            inds_EGR1,inds_cells = np.arange(len(EGR1_centrs))[dists_<dist_cutoff],inds_[dists_<dist_cutoff]
        
        save_folder = self.save_folder+os.sep+'EGR1_Intersection'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if len(inds_EGR1)>0:
            for ind_EGR1,ind_cell,distance in zip(inds_EGR1,inds_cells,dists_[dists_<dist_cutoff]):
                cell_strs = map(lambda xyz: '_'.join(np.array(xyz[:end_dim],dtype=str)), cells_reds[ind_cell])
                EGR1_strs = map(lambda xyz: '_'.join(np.array(xyz[:end_dim],dtype=str)), EGR1_reds[ind_EGR1])
                
                print dic_cells[cell_keys[ind_cell]]['olfr']+" intersected in "+self.fov_name
                cell_inter_key = cell_keys[ind_cell]
                olfr_inter = dic_cells[cell_keys[ind_cell]]['olfr']
                if saveTOfile:
                    save_image_file = self.save_folder+os.sep+'Decoded'+os.sep+self.fov_name.replace('.dax','')+os.sep+'good'+os.sep+olfr_inter+'_'+cell_inter_key+'.png'
                    save_target_image_file = save_folder+os.sep+self.fov_name.replace('.dax','')+"__"+os.path.basename(save_image_file)
                    save_dic_file = save_target_image_file.replace('.png','.pkl')
                    
                    pickle.dump({'area_EGR1':len(np.unique(EGR1_strs)),'area_olfr':len(np.unique(cell_strs)),'area_overlap':len(np.intersect1d(cell_strs,EGR1_strs)),
                                 'distance':distance,
                                 'dic_EGR1':list_EGR1_keep[ind_EGR1],'dic_olfr':dic_cells[cell_keys[ind_cell]]},open(save_dic_file,'wb'))
                    if os.path.exists(save_image_file):
                        shutil.copy2(save_image_file, save_target_image_file)
        return None
    def EGR1_fit(self,fl):
        """
        This performs: 
        EGR1 extract from raw dax file given by 'fl' according to self.target_z
        Corrects the z stack for flat field
        Saves a new dax file in self.save_folder+os.sep+'EGR1Analysis'
        Performs dao_STORM fitting of this file with a paramater from EGR1Conv_zscan_pars.xml in the python functions
        Clusters the fits from _alist.bin
        Decides to throw away super-hot speckles.
        Decides wheter to subdivide DBSCAN clusters into cells according to Xmeans clustering
        Checks to see if in MOE mask.
        Save in EGR1Analysis folder in a .cellfits file. This contains list_cells a list of dictionaries.
        {'fl':original dax file,'fit_fl':the processed EGR1 dax file,'x','y',
        'z':rescaled by multiplying frame with rescz = float(self.nm_per_pixel_xy)/self.nm_per_pixel_z,
        'h':height,'MOE':MOE index-0 if outside MOE masks,'filter':True/False if good/bad cell}
        """
        start = time.time()
        self.internalize_paramaters()
        save_folder = self.save_folder+os.sep+'EGR1Analysis'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        dax_fl = save_folder+os.sep+'EGR1'+os.path.basename(fl)
        print os.path.exists(dax_fl.replace('.dax','.cellfits')),dax_fl.replace('.dax','.cellfits')
        start_ = not os.path.exists(dax_fl.replace('.dax','.cellfits')) or self.EGR1_force_redo
        print "good to go:",start_
        if start_:
            
            if not os.path.exists(dax_fl):
                im = io.DaxReader(fl).loadAll()
                frames = self.best_stage_vals(fl,target_stage=self.target_z,start = self.EGR1_start_frame)
                source = im[self.EGR1_start_frame::self.EGR1_rep_frame][frames]
                if self.EGR1_correction:
                    im_cor = self.correction_image(self.EGR1_start_frame,set_fl=0,perc_=95,extra_note='',save_file='auto',overwrite=False)
                    im_cor = im_cor.astype(float)
                    im_cor = im_cor/np.median(im_cor)
                    if len(source.shape)!=len(im_cor.shape): im_cor=np.expand_dims(im_cor,0)
                    source = 1./im_cor*source
                source = (gt.minmax(source,max_=2**16-1)*(2**16-1)).astype(np.uint16)
                self.sz_image,self.sx_image,self.sy_image = source.shape
                #saa.writeDax(dax_fl,source)
            else:
                source = io.DaxReader(dax_fl).loadAll()
                self.sz_image,self.sx_image,self.sy_image = source.shape
            
            rescz = float(self.nm_per_pixel_xy)/self.nm_per_pixel_z    
            fit_fl = dax_fl.replace('.dax','_mlist.pkl')
            if self.EGR1_refit is True or not os.path.exists(fit_fl):
                centers = ft.get_seed_points_base(source,gfilt_size=0.75,filt_size=3,th_seed=500,hot_pix_th=5)
                pfits = ft.fast_local_fit(source,centers.T,radius=3,width_zxy=[1,1,1])
                pickle.dump(pfits,open(fit_fl,'wb'))
            else:
                pfits=pickle.load(open(fit_fl,'rb'))
                print "loading fit"
            keep = pfits[:,0]>self.EGR1_h_min
            keep = keep&(pfits[:,0]/pfits[:,4]>1.)
            z,x,y,h =  pfits[keep,1]*rescz,pfits[keep,2],pfits[keep,3],pfits[keep,0]
            #DBScanner
            dbscan_ = sklearn.cluster.DBSCAN(eps=self.EGR1_DBSCAN_eps,min_samples=self.EGR1_DBSCAN_min_samples)
            
            if len(x)>0:
                dbscan_.fit(np.array(zip(x,y,z)))
                labels_ = np.array(dbscan_.labels_,dtype=int)                
                labels_ = self.Xmeans_clustering(zip(x,y,z),labels_,min_samples_Xmeans=self.EGR1_min_samples_Xmeans,dist_cut_Xmeans=self.EGR1_dist_cut_Xmeans,plt_val=False)

                labels_keep = np.unique(labels_)
                labels_keep = labels_keep[labels_keep!=-1]
                list_cells = [{'fl':fl,'fit_fl':dax_fl,'x':x[labels_==i],'y':y[labels_==i],'z':z[labels_==i],'h':h[labels_==i]} for i in labels_keep]
            else:
                list_cells = []
            #filtering....................
            hot_clusters = False
            
            if len(list_cells):
                list_good = [True]*len(list_cells)
                if hot_clusters:
                    #load fitted dax and find very hot pixels
                    im_EGR1=source
                    z_max,x_max,y_max = np.where(im_EGR1>self.EGR1_h_max_filter)
                    rescz = float(self.nm_per_pixel_xy)/self.nm_per_pixel_z
                    z_max=z_max*rescz #rescale to fit to standard z
                    
                    from scipy import spatial
                    if len(z_max)>0: tree = spatial.KDTree(zip(z_max,x_max,y_max))
                    else: tree=None
                        
                    # decide whether the cell is good (if too close to high pixels return False)
                    def good_cell(cell,tree=tree,dist_cut = self.EGR1_dist_cut_filter,fr_cut = self.EGR1_fr_cut_filter,nbad=self.EGR1_nbad_filter):
                        cell_val = True
                        if tree is not None:
                            x_,y_,z_ = cell['x'],cell['y'],cell['z']
                            pts_ = zip(z_,x_,y_)
                            neighs = tree.query(pts_,distance_upper_bound=dist_cut)
                            ngood = np.sum(1./neighs[0]==0)
                            if (len(x_)-ngood)>nbad:
                                cell_val = False
                            if float(ngood)/len(x_)<fr_cut:
                                cell_val = False
                        return cell_val
                    
                    list_good = map(good_cell,list_cells)
                for dic,good in zip(list_cells,list_good): 
                    dic['filter']=good
                    
                # decide where in the MOE the cell is.
                im_mask = self.im_MOEmask(fl)
                for dic in list_cells: 
                    x_MOE,y_MOE = int(np.median(dic['x'])),int(np.median(dic['y']))
                    if x_MOE<im_mask.shape[0] and y_MOE<im_mask.shape[0]:
                        dic['MOE']= im_mask[x_MOE,y_MOE]
                    else:
                        dic['MOE']=0
            
            
            #Plotting
            f=plt.figure(figsize=(20,20))
            for i_cell,dic in enumerate(list_cells):
                x,y=dic['x'],dic['y']
                if dic['MOE']>0 and dic['filter']:
                    plt.plot(y,x,'.')
                    plt.text(np.mean(y),np.mean(x),str(i_cell+1),color='g')
                else:
                    plt.plot(y,x,'r+')
            plt.imshow(gt.minmax(np.max(source,axis=0),max_=self.EGR1_h_plot),cmap=cm.Greys_r)
            f.savefig(dax_fl.replace('.dax','_cluster.png'))
            plt.close(f)
            
            pickle.dump(list_cells,open(dax_fl.replace('.dax','.cellfits'),'wb'),protocol=pickle.HIGHEST_PROTOCOL)
        
        end = time.time()
        print "Elapsed time for EGR1 processing: "+str(end-start)
    def Xmeans_clustering(self,X,labels,min_samples_Xmeans=3,dist_cut_Xmeans=60,plt_val=False):
        """Given a list of coordinates X and a list of labels as f.e. computed by DBSCAN,
        this attemps to subdivide the clusters in ever increasing numbers until 
        the minimum distance between any 2 clusters is less than dist_cut_Xmeans.
        CLusters smaller than min_samples_Xmeans will be ignored.
        Testing:
        X=np.random.random([100,3])*30
        X[30:36,:]+=100
        X[60:,:]+=30
        labels = np.zeros(len(X))
        Xmeans_clustering(None,X,labels,min_samples_Xmeans=5,dist_cut_Xmeans=50,plt_val=True)
        """
        from scipy.spatial.distance import pdist
        #from sklearn.cluster import KMeans#MiniBatchKMeans as KMeans#KMeans#
        labels_copy = np.array(labels).astype(int)
        X_ = np.array(X)
        max_lab = np.max(labels_copy)
        for label in range(max_lab+1):
            if label>=0:
                keep = labels_copy==label
                coords_reduced = X_[keep]
                #decide whether to split - use X-means clustering-------------------------------------
                #Increase the number of clusters succesively until the min distance between clusters is less than cell diamater 
                kno=1
                while True:
                    kno+=1
                    break_=False
                    kmeans = KMeans(kno)
                    kmeans.fit(coords_reduced)
                    kmeanslabels_=np.array(kmeans.labels_,dtype=int)
                    kmeanscluster_centers_=[]

                    #if too few samples in a subcluster ignore that subcluster
                    for iK in range(kno):
                        investigate_small = kmeanslabels_==iK
                        if np.sum(investigate_small)<min_samples_Xmeans:
                            kmeanslabels_[investigate_small]=-1
                        else:
                            kmeanscluster_centers_.append(kmeans.cluster_centers_[iK])
                    #if the min distance between any 2 non-ignored clusters is smaller than cell diamater flag break!
                    if len(kmeanscluster_centers_)<2:
                        break
                    center_distances = pdist(kmeanscluster_centers_)
                    if np.min(center_distances)<dist_cut_Xmeans:
                        break
                    if break_:
                        break
                    prev_labels=kmeanslabels_
                #print kno
                if kno>2:
                    def remap_label(lab_):
                        if lab_==0:
                            return label
                        if lab_>0:
                            return lab_+max_lab
                        if lab_<0:
                            return -1
                    labels_copy[keep]=map(remap_label,prev_labels)#modify labels to add splits
                    max_lab=np.max(labels_copy)
        labels_unk = np.unique(labels_copy)
        
        for i_unk,unk in enumerate(labels_unk):
            subtr = 0 if -1 not in labels_unk else 1
            if unk>=0:
                labels_copy[labels_copy==unk]=i_unk-subtr
        if plt_val:
            if X_.shape[-1]>=2:
                plt.figure(figsize=(10,10))
                for label in range(np.max(labels_copy)+1):
                    keep = labels_copy==label
                    plt.plot(X_[keep,0],X_[keep,1],'o')
                    plt.text(np.mean(X_[keep,0]),np.mean(X_[keep,1]),str(label+1),color='g')
                plt.plot(X_[:,0],X_[:,1],'+')
                plt.show()
        return labels_copy
    def ORtrainer(self,files__,notes='',save_val=True):
        if files__ is None:
            files__ = glob.glob(self.save_folder+os.sep+'prediction-auto'+os.sep+'*.png')
        reload(gt)
        trainer = gt.imshow_pngs(files__)
        if save_val: 
            self.save_training_set(trainer,filename=None,notes=notes)
    def save_training_set(self,trainer,filename=None,notes=''):
        if filename is None:
            save_folder = self.save_folder
            prev_trainings = glob.glob(save_folder+os.sep+'ORtraining*')
            indx_existent = np.array([gt.extract_flag(trnm,'training_','.')for trnm in prev_trainings],dtype=int)
            indx_curr = '0'
            if len(indx_existent)>0:
                indx_curr = str(np.max(indx_existent)+1)
            filename = save_folder+os.sep+'ORtraining_'+indx_curr+'.trng'
        if trainer.index_good or trainer.index_bad:
            dic_trainer={'files_good':np.array(trainer.files)[trainer.index_good],'files_bad':np.array(trainer.files)[trainer.index_bad],'notes':notes}
            pickle.dump(dic_trainer,open(filename,'wb'),protocol=pickle.HIGHEST_PROTOCOL)
    def slurm_STD(self,n_array=6,time_='0-03:00',ranges_test=None):
        import time
        Nfiles = len(self.files)
        ranges = [[j for j in range(i,i+n_array) if j<Nfiles] for i in range(0,Nfiles,n_array)]
        
        python_file_base = os.path.dirname(gt.__file__)+os.sep+'OR_MER_STD.py'
        if not os.path.exists(python_file_base):
            print "Standard file not found at: "+python_file_base
            return None
        if ranges_test is not None:
            ranges = ranges_test 
        print "Requesting no. of jobs: "+str(len(ranges))
        for i,range_ in enumerate(ranges):
            python_file = python_file_base+' "'+self.data_folder+'" '+str(range_).replace(" ","")
            time.sleep(1)
            slurm_file = self.save_folder+os.sep+'Scripts'+os.sep+'slurm_'+str(i)+'.sh'
            self.slurm_python(python_file,slurm_file,t=time_)
    def slurm_python(self,python_file,slurm_script,n=1,N=1,t='0-03:00',p='serial_requeue',mem=32000,
                     err_file=None,out_file=None,job_index=False):
        "Given python file and slurm specs this launches a python file in sbatch."
        import subprocess,time
        save_folder = os.path.dirname(slurm_script)
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        base_name = slurm_script.replace('.sh','')
        if job_index: job_str = '_%j'
        else: job_str = ''
        if err_file is None: err_file=base_name
        if out_file is None: out_file=base_name
        string = """#!/bin/bash
#SBATCH -n """+str(n)+"""                    # Number of cores
#SBATCH -N """+str(N)+"""                    # Ensure that all cores are on one machine
#SBATCH -t """+str(t)+"""              # Runtime in D-HH:MM
#SBATCH -p """+str(p)+"""       # Partition to submit to
#SBATCH --mem="""+str(mem)+"""               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o """+out_file+job_str+""".out      # File to which STDOUT will be written
#SBATCH -e """+err_file+job_str+""".err      # File to which STDERR will be written


. new-modules.sh && module load fftw && module load python/2.7.6-fasrc01 && source activate python27rc

python """+python_file+"""
"""
        fid = open(slurm_script,'w')
        fid.write(string)
        fid.close()
        time.sleep(1)
        subprocess.check_output("sbatch "+slurm_script,shell=True)
