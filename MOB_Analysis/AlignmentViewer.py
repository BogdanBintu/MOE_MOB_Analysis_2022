from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pylab as plt
import numpy as np
import os,sys,glob
import tifffile
import cv2
import pickle

from scipy.spatial.distance import cdist
def saveManual_X1_X2(X1t_,X2t_,save_fl_align):
    dic_ = {'coords':[(list(X1t_[:,1]),list(X1t_[:,0])),(list(X2t_[:,1]),list(X2t_[:,0]))],'names':['Image 1', 'Image 2']}
    pickle.dump(dic_,open(save_fl_align,'wb'))
    return save_fl_align
def get_txyth(X1,X2,res=[10,10,1.5],plt_val=False,return_X2T=False):
    X1i,X1j = [],[]
    X2i,X2j = [],[]
    for i in range(len(X1)):
        for j in range(i):
            X1i.append(X1[i])
            X1j.append(X1[j])
            X2i.append(X2[i])
            X2j.append(X2[j])
    X1i,X1j = np.array(X1i),np.array(X1j)
    X2i,X2j = np.array(X2i),np.array(X2j)
    dX1 = X1i-X1j
    dX2 = X2i-X2j
    costh = np.sum(dX1*dX2,-1)/np.linalg.norm(dX1,axis=-1)/np.linalg.norm(dX2,axis=-1)
    sinth = np.sqrt(1-costh*costh)*np.sign(np.sum(dX1*(dX2[:,::-1]*[-1,1]),-1))
    X1c = (X1i+X1j)/2.
    X2c = (X2i+X2j)/2.
    X2cT = np.array([costh*X2c[:,0]-sinth*X2c[:,1],costh*X2c[:,1]+sinth*X2c[:,0]]).T
    txys = X1c - X2cT
    ths = np.arctan2(sinth,costh)
    difs = np.concatenate([txys,ths[:,np.newaxis]],axis=-1)
    res = np.array(res)*[1,1,2*np.pi/360]
    
    
    bins = [np.arange(m,M+res_*2*1.01,res_*2) for m,M,res_ in zip(np.nanmin(difs,0),np.nanmax(difs,0),res)]
    hhist,bins_ = np.histogramdd(difs,bins)
    max_i = np.unravel_index(np.argmax(hhist),hhist.shape)
    if plt_val:
        plt.figure()
        plt.imshow(np.max(hhist,-1))
    center_ = [(bin_[iM_]+bin_[iM_+1])/2. for iM_,bin_ in zip(max_i,bins_)]
    keep = np.all(np.abs(difs-center_)<=res,-1)
    center_ = np.mean(difs[keep],0)
    for i in range(5):
        keep = np.all(np.abs(difs-center_)<=res,-1)
        center_ = np.mean(difs[keep],0)
        
    ### apply transf
    tx,ty,th = center_
    R = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]]).T
    X2_ = np.dot(X2,R)+[tx,ty]
    
    dists_nn = np.sort(cdist(X1,X2_),-1)[:,0]
    med_score = np.mean(dists_nn)
    if plt_val:
        plt.figure()
        plt.plot(X1[:,0],X1[:,1],'o')
        plt.plot(X2_[:,0],X2_[:,1],'x')
    if return_X2T:
        return center_,med_score,X2_
    return center_,med_score

def get_txyth_pm(X1,X2,res=[10,10,1.5],plt_val=False):
    (txm,tym,thm),med_scorem,X2Tm = get_txyth(X1,X2*[-1,1],res=res,plt_val=plt_val,return_X2T=True)
    (txp,typ,thp),med_scorep,X2Tp = get_txyth(X1,X2*[1,1],res=res,plt_val=plt_val,return_X2T=True)
    if med_scorem<med_scorep: tx,ty,th,tm,med_score,X2T=txm,tym,thm,-1,med_scorem,X2Tm
    else: tx,ty,th,tm,med_score,X2T=txp,typ,thp,1,med_scorep,X2Tp
    return tx,ty,th,tm,med_score,X2T
def align_im(im2,txtyth,dtX2_=None,sh = [3600,3600],resc=10,bk=np.exp(-6),dglom=70):
    
    tx,ty,th,tm=txtyth
    R = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]]).T
    
    if dtX2_ is None:
    
        Xi0 = np.indices(sh).reshape([2,-1]).T
        Xi = Xi0
        Xii = np.round(np.dot(Xi-[tx,ty],R.T)).astype(int)
        Xii = Xii*[tm,1]
        Xii_copy = Xii.copy()
        imsh = im2[0].shape if type(im2) is list else im2.shape
        Xii = Xii%imsh
        Xii_bad = Xi0[np.any(Xii!=Xii_copy,axis=-1)]
        
        if type(im2) is list:
            im2Ts = []
            for im2_ in im2:
                im2__ = im2_[Xii[:,0],Xii[:,1]].reshape(sh)
                im2__[Xii_bad[:,0],Xii_bad[:,1]]=0
                im2Ts.append(im2__)
            return im2Ts
        else:
            im2__ = im2[Xii[:,0],Xii[:,1]].reshape(sh)
            im2__[Xii_bad[:,0],Xii_bad[:,1]]=0
            return im2__
    else:
        dts,X2_ = dtX2_
        Xi = np.indices(sh)[:,::resc,::resc]
        sh_ = Xi.shape[1:]
        Xi = Xi.reshape([2,-1]).T
        Xi0 = np.indices(sh).reshape([2,-1]).T
        eW = np.exp(-cdist(Xi,X2_)**2/(2*(1*dglom)**2))
        eW = np.concatenate([eW,bk+np.zeros([len(Xi),1])],axis=-1)###
        dts = np.concatenate([dts,[[0,0]]])
        eW = eW/np.sum(eW,-1)[:,np.newaxis]
        Xi_ = Xi+np.dot(eW,dts)

        Xii = np.round(np.dot(Xi_-[tx,ty],R.T)).astype(int)
        from scipy.ndimage import zoom
        Xii1 = np.ravel(zoom(Xii[:,0].reshape(sh_),np.array(sh)/sh_,order=1))
        Xii2 = np.ravel(zoom(Xii[:,1].reshape(sh_),np.array(sh)/sh_,order=1))
        Xii = np.array([Xii1,Xii2]).T.astype(np.int)*[tm,1]
        Xii_copy = Xii.copy()
        imsh = im2[0].shape if type(im2) is list else im2.shape
        Xii = Xii%imsh
        Xii_bad = Xi0[np.any(Xii!=Xii_copy,axis=-1)]
        
        if type(im2) is list:
            im2Ts = []
            for im2__ in im2:
                im2_ = im2__[Xii[:,0],Xii[:,1]].reshape(sh)
                im2_[Xii_bad[:,0],Xii_bad[:,1]]=0
                im2Ts.append(im2_)
            return im2Ts
        else: 
            im2_ = im2[Xii[:,0],Xii[:,1]].reshape(sh)
            im2_[Xii_bad[:,0],Xii_bad[:,1]]=0
            return im2_

def applt_txtyth_dT(Xi,txtyth,dtX2_=None,resc=1,bk=np.exp(-6),dglom=7):
    tx,ty,th,tm=txtyth
    R = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]]).T
    Xi_ = np.array(Xi)/resc
    Xii = np.dot(Xi_*[tm,1],R)+[tx,ty]
    Xii_ = Xii 
    if dtX2_ is not None:
        dts,X2_ = dtX2_
        eW = np.exp(-cdist(Xii,X2_)**2/(2*(1*dglom)**2))
        eW = np.concatenate([eW,bk+np.zeros([len(eW),1])],axis=-1)###
        dts_ = np.concatenate([dts,[[0,0]]])
        eW = eW/np.sum(eW,-1)[:,np.newaxis]
        Xii_ = Xii+np.dot(eW,dts_)
    return Xii_*resc

def loadManual_X1_X2(fl):
    if not os.path.exists(fl):
        return None,None
    dic_ = pickle.load(open(fl,'rb'))
    X1,X2 = dic_['coords']
    X1=np.array(X1).T[:,::-1]
    X2=np.array(X2).T[:,::-1]
    return X1,X2
    
def glomPos_to_im(XOB_,im_dapi,pad = 0.1,plt_val=True):
    xm = np.min(XOB_,0).astype(int)
    xM = np.max(XOB_,0).astype(int)
    pad = int(np.max(xM-xm)*pad)
    xm = xm-pad
    xM = xM+pad
    xm[xm<0]=0
    im_sh = np.array(im_dapi.shape)[::-1]
    xM[xM>=im_sh]=im_sh[xM>=im_sh]-1
    
    im_ = im_dapi[xm[1]:xM[1],xm[0]:xM[0]]
    #print(xm,xM,im_dapi.shape)
    perc = np.percentile(im_[im_!=0],10)
    im_[im_==0]=perc
    im_ = np.array(im_,dtype=np.float32)/perc
    if plt_val:
        plt.figure()
        plt.imshow(im_,cmap='gray',vmax=np.percentile(im_,99))
    return im_,xm,xM
def get_glom_info(folder):
    fl = folder+r'\dapi_resc8.tif'
    fl_mask = fl.replace('.tif','_mask_manual.png')
    #im_msk = cv2.imread(fl_mask)[:,:,0]
    fl_glom = fl_mask.replace('.png','.npy')
    im_labf,limits = np.load(fl_glom)
    im_dapi = tifffile.imread(fl)
    XOB_s = np.load(folder+os.sep+'glomeruli_order.npy')
    return im_dapi,XOB_s
    
class imshow_2d:
    def __init__(self,im,fig=None,save_file=None):
        """
        This is a class which controls an interactive maplotlib figure.
        Intended for navigating and interacting with 'spot'-like data that is spread across multiple images <ims>.
        Two max projection images are shown: xy and xz. By zooming into a region, the zoomed in region gets re-maxprojected.
        
        Right click to add a 'seed' point.
        Shift+Right click  to remove the closest 'seed' point
        
        Press 'x' to automatically adjust contrast to min-max in the zoomed in region.
        
        Optional features:
        Can provide a list of color 3d images (or functions that produce color 3d images) as markers (i.e. DAPI or geminin)
        
        """

        self.save_file = save_file
        
        #define extra vars
        self.draw_x1,self.draw_y1 = [],[]
        self.coords1 = list(zip(self.draw_x1,self.draw_y1))
        self.delete_mode = False
        #load vars
        self.load_coords()
        #construct images
        self.im1=im
        self.matrix=None
        self.matrix_inv=None
        #setup plots
        self.f,self.ax1=plt.subplots(1,1,sharex=True,sharey=True,facecolor='k')
        self.l1,=self.ax1.plot(self.draw_x1, self.draw_y1, 'w-o',
                              markersize=12,markeredgewidth=1,markeredgecolor='w',markerfacecolor='None')
        #self.imshow1 = self.ax1.imshow(self.im1,interpolation='nearest',cmap='gray')
        #self.imshow1.set_clim(np.min(self.im1),np.max(self.im1))
        #connect zoom/pan
        #self.f.suptitle(self.image_names[self.index_im])
        #connect mouse and keyboard
        cid = self.f.canvas.mpl_connect('button_press_event', self.onclick)
        cid2 = self.f.canvas.mpl_connect('key_press_event', self.press)
        cid3 = self.f.canvas.mpl_connect('key_release_event', self.release)
        self.set_image()
        if fig is None:
            plt.show()
    def master_reset(self):
        #self.dic_min_max = {}
        self.class_ids = []
        self.draw_x1,self.draw_y1=[],[]
        self.draw_x2,self.draw_y2=[],[]
        #load vars
        self.load_coords()
        self.set_image()
    def onclick(self,event):
        if event.button==3:
            #print "click"
            self.mouse_pos = [event.xdata,event.ydata]
           
            if self.delete_mode:
                if event.inaxes is self.ax1:
                    
                    if len(self.draw_x1)>0:
                        X = np.array([self.draw_x1,self.draw_y1]).T
                        difs = X-np.array([self.mouse_pos])
                        ind_= np.argmin(np.sum(np.abs(difs),axis=-1))
                        #self.here='ax1',self.mouse_pos,ind_
                        self.draw_x1.pop(ind_)
                        self.draw_y1.pop(ind_)
            else:
                if event.xdata is not None and event.ydata is not None:
                    if event.inaxes is self.ax1:
                        self.draw_x1.append(self.mouse_pos[0])
                        self.draw_y1.append(self.mouse_pos[1])
            self.update_point_plot()
    def press(self,event):
        if event.key== 'x':
            self.auto_scale()

        if event.key == 'delete':
            if len(self.draw_x1)>1:
                self.draw_x1.pop(-1)
                self.draw_y1.pop(-1)
            self.update_point_plot()
        if event.key == 'shift':
            self.delete_mode = True
    def release(self, event):
        if event.key == 'shift':
            self.delete_mode = False
    def create_text(self):
        self.texts = []
        #i_ims = np.zeros(len(self.ims),dtype=int)
        for i1,(x1,y1) in enumerate(zip(self.draw_x1,self.draw_y1)):
            self.texts.append(self.ax1.text(x1,y1,str(i1+1),color='r'))

    def update_point_plot(self):
        self.l1.set_xdata(self.draw_x1)
        self.l1.set_ydata(self.draw_y1)
 
        self.save_coords()
        self.remove_text()
        self.create_text()
        self.f.canvas.draw()
    def remove_text(self):
        if not hasattr(self,'texts'): self.texts = []
        for txt in self.texts:
            txt.remove()
    def load_coords(self):
        save_file = self.save_file
        if save_file is not None and os.path.exists(save_file):
            save_dic = pickle.load(open(save_file,'rb'))
            (self.draw_x1,self.draw_y1) = save_dic['coords']
    def save_coords(self):
        save_file = self.save_file
        if save_file is not None:
            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))
            fid = open(save_file,'wb')
            save_dic = {'coords':(self.draw_x1,self.draw_y1)}
            pickle.dump(save_dic,fid)
            fid.close()
    def auto_scale(self):
        x_min,x_max,y_min,y_max = self.get_limits()
        im_chop = np.array(self.im1[x_min:x_max,y_min:y_max])
        min_,max_ = np.min(im_chop),np.max(im_chop)
        self.imshow1.set_clim(min_,max_)
        self.f.canvas.draw()
    def del_ext(self,str_):
        "Deletes extention"
        if os.path.basename(str_).count('.')>0:
            return '.'.join(str_.split('.')[:-1])
        else:
            return str_
    def set_image(self):
        self.update_point_plot()
        self.f.canvas.draw()
        
    def get_limits(self):
        y_min,y_max = np.sort(self.ax1.get_xlim())
        x_min,x_max = np.sort(self.ax1.get_ylim())
        x_min = max(int(x_min),0)
        x_max = min(int(x_max),self.im1.shape[0])
        y_min = max(int(y_min),0)
        y_max = min(int(y_max),self.im1.shape[1])
        return x_min,x_max,y_min,y_max


    
class imshow_2d_align:
    def __init__(self,ims,fig=None,image_names=None,save_file=None):
        """
        This is a class which controls an interactive maplotlib figure.
        Intended for navigating and interacting with 'spot'-like data that is spread across multiple images <ims>.
        Two max projection images are shown: xy and xz. By zooming into a region, the zoomed in region gets re-maxprojected.
        
        Right click to add a 'seed' point.
        Shift+Right click  to remove the closest 'seed' point
        
        Press 'x' to automatically adjust contrast to min-max in the zoomed in region.
        
        Optional features:
        Can provide a list of color 3d images (or functions that produce color 3d images) as markers (i.e. DAPI or geminin)
        
        """
        self.ims=ims #gray scale images
        self.order=1
        self.matrix=None
        if image_names is None:
            self.image_names = ['Image '+str(i+1) for i in range(len(ims))]
        else:
            self.image_names = image_names
        self.save_file = save_file
        
        #define extra vars
        self.dic_min_max = {} #kees record of the min-max for adjusting contrast for the grayscale images
        self.draw_x1,self.draw_y1 = [],[]
        self.draw_x2,self.draw_y2 = [],[]
        self.coords1 = list(zip(self.draw_x1,self.draw_y1))
        self.coords2 = list(zip(self.draw_x2,self.draw_y2))
        self.delete_mode = False
        #load vars
        self.load_coords()
        #construct images
        self.im1,self.im2 = self.ims[0],self.ims[1]
        if True:
            shape = np.max([self.im1.shape,self.im2.shape],0)
            im0 = np.zeros(shape,dtype=self.im1.dtype)
            im0[:self.im1.shape[0],:self.im1.shape[1]]=self.im1
            self.im1 = im0
            im0 = np.zeros(shape,dtype=self.im2.dtype)
            im0[:self.im2.shape[0],:self.im2.shape[1]]=self.im2
            self.im2 = im0
        self.matrix=None
        self.matrix_inv=None
        #setup plots
        self.f,(self.ax1,self.ax2)=plt.subplots(1,2,sharex=True,sharey=True)
        self.l1,=self.ax1.plot(self.draw_x1, self.draw_y1, 'o',
                              markersize=12,markeredgewidth=1,markeredgecolor='y',markerfacecolor='None')
        self.l2,=self.ax2.plot(self.draw_x2, self.draw_y2, 'o',
                      markersize=12,markeredgewidth=1,markeredgecolor='y',markerfacecolor='None')
        self.imshow1 = self.ax1.imshow(self.im1,interpolation='nearest',cmap='gray')
        self.imshow2 = self.ax2.imshow(self.im2,interpolation='nearest',cmap='gray')

        self.imshow1.set_clim(np.min(self.im1),np.max(self.im1))
        self.imshow2.set_clim(np.min(self.im2),np.max(self.im2))
        #connect zoom/pan
        #self.f.suptitle(self.image_names[self.index_im])
        #connect mouse and keyboard
        cid = self.f.canvas.mpl_connect('button_press_event', self.onclick)
        cid2 = self.f.canvas.mpl_connect('key_press_event', self.press)
        cid3 = self.f.canvas.mpl_connect('key_release_event', self.release)
        self.set_image()
        if fig is None:
            plt.show()
    def master_reset(self):
        #self.dic_min_max = {}
        self.class_ids = []
        self.draw_x1,self.draw_y1=[],[]
        self.draw_x2,self.draw_y2=[],[]
        #load vars
        self.load_coords()
        self.set_image()
    def onclick(self,event):
        if event.button==3:
            #print "click"
            self.mouse_pos = [event.xdata,event.ydata]
            if (event.inaxes is self.ax2) and (self.matrix is not None):
                self.mouse_pos = apply_colorcor(np.array([self.mouse_pos[::-1]]),m=self.matrix)[0][::-1]
            if self.delete_mode:
                if event.inaxes is self.ax1:
                    
                    if len(self.draw_x1)>0:
                        X = np.array([self.draw_x1,self.draw_y1]).T
                        difs = X-np.array([self.mouse_pos])
                        ind_= np.argmin(np.sum(np.abs(difs),axis=-1))
                        #self.here='ax1',self.mouse_pos,ind_
                        self.draw_x1.pop(ind_)
                        self.draw_y1.pop(ind_)
                elif event.inaxes is self.ax2:
                    if len(self.draw_x2)>0:
                        X = np.array([self.draw_x2,self.draw_y2]).T
                        difs = X-np.array([self.mouse_pos])
                        ind_= np.argmin(np.sum(np.abs(difs),axis=-1))
                        self.draw_x2.pop(ind_)
                        self.draw_y2.pop(ind_)
            else:
                if event.xdata is not None and event.ydata is not None:
                    if event.inaxes is self.ax1:
                        self.draw_x1.append(self.mouse_pos[0])
                        self.draw_y1.append(self.mouse_pos[1])
                    if event.inaxes is self.ax2:
                        self.draw_x2.append(self.mouse_pos[0])
                        self.draw_y2.append(self.mouse_pos[1])
            self.update_point_plot()
    def press(self,event):
        if event.key== 'x':
            self.auto_scale()
        if event.key== 't':
            self.calculate_transf()
        if event.key== 'u':
            self.calculate_transf()

        if event.key== '[':
            self.order-=1
            if self.order<0:self.order=0
        if event.key== ']':
            self.order+=1
            
        if event.key == 'delete':
            if len(self.draw_x1)>1:
                self.draw_x1.pop(-1)
                self.draw_y1.pop(-1)
            if len(self.draw_x2)>1:
                self.draw_x2.pop(-1)
                self.draw_y2.pop(-1)
            self.update_point_plot()
        if event.key == 'shift':
            self.delete_mode = True
    def release(self, event):
        if event.key == 'shift':
            self.delete_mode = False
    def create_text(self):
        self.texts = []
        i_ims = np.zeros(len(self.ims),dtype=int)
        for i1,(x1,y1) in enumerate(zip(self.draw_x1,self.draw_y1)):
            self.texts.append(self.ax1.text(x1,y1,str(i1+1),color='r'))
        for i2,(x2,y2) in enumerate(zip(self.draw_x2T,self.draw_y2T)):
            self.texts.append(self.ax2.text(x2,y2,str(i2+1),color='r'))
    def uncalculate_transf(self):
        self.matrix = None
        self.matrix_inv = None
        self.imshow2.set_data(self.im2)
        self.update_point_plot()
    def calculate_transf(self):
        X1 = np.array([self.draw_y1,self.draw_x1]).T
        X2 = np.array([self.draw_y2,self.draw_x2]).T
        m = calc_color_matrix(X1,X2,order=self.order)
        self.matrix = m
        self.matrix_inv = calc_color_matrix(X2,X1,order=self.order)
        X_ = np.indices(self.im1.shape).reshape([2,-1]).T
        print(self.matrix)
        X__ = apply_colorcor(X_,m=self.matrix).astype(int)
        X__[X__<0]=0
        X__[X__[:,0]>=self.im2.shape[0],0]=self.im2.shape[0]-1
        X__[X__[:,1]>=self.im2.shape[1],1]=self.im2.shape[1]-1
        self.im2T = self.im2[X__[:,0],X__[:,1]].reshape(self.im1.shape)
        self.imshow2.set_data(self.im2T)
        self.update_point_plot()
    def update_point_plot(self):
        self.l1.set_xdata(self.draw_x1)
        self.l1.set_ydata(self.draw_y1)
        if self.matrix is None:
            self.draw_x2T = self.draw_x2[:]
            self.draw_y2T = self.draw_y2[:]
        else:
            self.draw_y2T,self.draw_x2T = apply_colorcor(np.array([self.draw_y2,self.draw_x2]).T,m=self.matrix_inv).T
            self.draw_x2T,self.draw_y2T = list(self.draw_x2T),list(self.draw_y2T)
        self.l2.set_xdata(self.draw_x2T)
        self.l2.set_ydata(self.draw_y2T)
        self.save_coords()
        self.remove_text()
        self.create_text()
        self.f.canvas.draw()
    def remove_text(self):
        if not hasattr(self,'texts'): self.texts = []
        for txt in self.texts:
            txt.remove()
    def load_coords(self):
        save_file = self.save_file
        if save_file is not None and os.path.exists(save_file):
            save_dic = pickle.load(open(save_file,'rb'))
            (self.draw_x1,self.draw_y1),(self.draw_x2,self.draw_y2) = save_dic['coords']
    def save_coords(self):
        save_file = self.save_file
        if save_file is not None:
            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))
            fid = open(save_file,'wb')
            save_dic = {'coords':[(self.draw_x1,self.draw_y1),(self.draw_x2,self.draw_y2)]}
            save_dic['names']=self.image_names
            pickle.dump(save_dic,fid)
            fid.close()
    def auto_scale(self):
        x_min,x_max,y_min,y_max = self.get_limits()
        im_chop = np.array(self.im1[x_min:x_max,y_min:y_max])
        min_,max_ = np.min(im_chop),np.max(im_chop)
        self.imshow1.set_clim(min_,max_)
        im_chop = np.array(self.im2[x_min:x_max,y_min:y_max])
        min_,max_ = np.min(im_chop),np.max(im_chop)
        self.imshow2.set_clim(min_,max_)
        self.f.canvas.draw()
    def del_ext(self,str_):
        "Deletes extention"
        if os.path.basename(str_).count('.')>0:
            return '.'.join(str_.split('.')[:-1])
        else:
            return str_
    def set_image(self):
        self.update_point_plot()
        self.f.canvas.draw()
        
    def get_limits(self):
        y_min,y_max = np.sort(self.ax1.get_xlim())
        x_min,x_max = np.sort(self.ax1.get_ylim())
        x_min = max(int(x_min),0)
        x_max = min(int(x_max),self.im1.shape[0])
        y_min = max(int(y_min),0)
        y_max = min(int(y_max),self.im1.shape[1])
        return x_min,x_max,y_min,y_max
def calc_color_matrix(x,y,order=2):
    """This gives a quadratic color transformation (in matrix form)
    x is Nx3 vector of positions in the reference channel (typically cy5)
    y is the Nx3 vector of positions in another channel (i.e. cy7)
    return m_ a 3x7 matrix which when multipled with x,x**2,1 returns y-x
    This m_ is indended to be used with apply_colorcor
    """ 
    x_ = np.array(x)# ref zxy
    y_ = np.array(y)
    # get a list of exponents
    exps = []
    for p in range(order+1):
        for i in range(p+1):
            exps.append([i,p-i])
    # construct A matrix
    A = np.zeros([len(x_),len(exps)])
    for iA,(ix,iy) in enumerate(exps):
        s = (x_[:,0]**ix*x_[:,1]**iy)
        A[:,iA]=s
    m_ = [np.linalg.lstsq(A, y_[:,iy])[0] for iy in range(len(x_[0]))]
    m_=np.array(m_)
    return m_
def apply_colorcor(x,m=None):
    """This applies chromatic abberation correction to order 2
    x is a Nx3 vector of positions (typically 750(-->647))
    m is a matrix computed by function calc_color_matrix
    y is the corrected vector in another channel"""
    if m is None:
        return x[:]
    exps = []
    order_max=10
    for p in range(order_max+1):
        for i in range(p+1):
            exps.append([i,p-i])
    #find the order
    mx,my = m.shape
    order = int((my-1)/mx)
    assert(my<len(exps))
    x_ = np.array(x)
    # construct A matrix
    exps = exps[:my]
    A = np.zeros([len(x_),len(exps)])
    for iA,(ix,iy) in enumerate(exps):
        s = (x_[:,0]**ix*x_[:,1]**iy)
        A[:,iA]=s
    diff = [np.dot(A,m_) for m_ in m]
    return np.array(diff).T