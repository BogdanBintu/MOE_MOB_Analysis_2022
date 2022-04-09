
#
import numpy as np
import glob,os,sys
sys.path.append(r'C:\Users\Bogdan\Dropbox\ChromatinImagingV2\CommonTools')
sys.path.append(r'E:\Bogdan\Dropbox\ChromatinImagingV2\CommonTools')
import IOTools_py3 as io
import MaxViewer_py3 as mv
import Fitting_v4 as ft
from scipy.signal import fftconvolve
import tifffile
import numpy as np
import glob,os,sys
import cv2
from tqdm import tqdm_notebook as tqdm
from sklearn.cluster import KMeans
from imp import reload
import matplotlib.pylab as plt
import pickle
from scipy.spatial import cKDTree
reload(io)
def get_neighbours(coords_dic,key_ob,Nmax = 50):
    """Return the neighbors of OB <key_ob> """
    isl = coords_dic[key_ob]['iSliceAbs']
    OBs_isl = np.array([ob for ob in coords_dic 
                        if coords_dic[ob]['iSliceAbs']==isl 
                        if int(ob/100000000)==int(key_ob/100000000)
                        if 'npts_pm' in coords_dic[ob]])
    X0 = coords_dic[key_ob]['xyzT_aligned2']
    X_ = np.array([coords_dic[ob]['xyzT_aligned2']for ob in OBs_isl])

    keep = np.argsort(np.linalg.norm(X_-X0,axis=-1))[:Nmax]
    OBs_kp = OBs_isl[keep]
    #X_ = np.array([coords_dic[ob]['xyzT_aligned2']for ob in OBs_kp])
    return OBs_kp


def update_npts_cor(coords_dic_final,nneigh = 50):
    keysOB = list(coords_dic_final.keys())
    for key_ob in tqdm(keysOB):
        ob_dic = coords_dic_final[key_ob]
        if 'npts_pm' in ob_dic:
            if 'npts_pm_cor' not in ob_dic:
                OBs_neigh = get_neighbours(coords_dic_final,key_ob,Nmax = nneigh)

                As = np.array([coords_dic_final[ob]['area'] for ob in OBs_neigh])
                A = ob_dic['area']
                A_exp = (A/As)[:,np.newaxis]
                #return A_exp,key_ob
                npts = np.array([coords_dic_final[ob]['npts_pm'] for ob in OBs_neigh])
                #return npts*A_exp
                median_ = np.median(npts*A_exp,axis=0)

                npts_cor = (ob_dic['npts_pm']-median_)#/median_
                npts_pm = npts_cor
                coords_dic_final[key_ob]['npts_pm_cor'] = npts_pm

                npts = np.array([coords_dic_final[ob]['npts_multi'] for ob in OBs_neigh])
                median_ = np.median(npts*A_exp,axis=0)
                npts_cor = (ob_dic['npts_multi']-median_)#/median_
                npts_multi = npts_cor

                coords_dic_final[key_ob]['npts_multi_cor'] = npts_multi
def get_OB_pair(coords_dic_final,id_1,id_2,is_med):
    icodes = list(np.arange(499))+list(1000+np.arange(622))
    dic_set1 = {}
    dic_set2 = {}
    for OB in coords_dic_final:
        if coords_dic_final[OB]['is_med']==is_med:
            icode = coords_dic_final[OB].get('best_code2',[-1,0,0])[0]
            if coords_dic_final[OB]['id']==id_1:
                dic_set1[icode]=dic_set1.get(icode,[])+[OB]
            elif coords_dic_final[OB]['id']==id_2:
                dic_set2[icode]=dic_set2.get(icode,[])+[OB]
    dic_pair_OB = {}
    from scipy.spatial.distance import cdist

    for icode in icodes:
        OBs1 = dic_set1.get(icode,[])
        OBs2 = dic_set2.get(icode,[])
        if len(OBs1)>0 and len(OBs2)>0:
            X1 = [coords_dic_final[OB]['xyzT_aligned2'] for OB in OBs1]
            X2 = [coords_dic_final[OB]['xyzT_aligned2'] for OB in OBs2]
            M = cdist(X1,X2)
            bdist = np.min(M)
            best_pr = np.array(np.where(M==bdist))[:,0]
            dic_pair_OB[icode]=[OBs1[best_pr[0]],OBs2[best_pr[1]],bdist]
    return dic_pair_OB
def get_best_dic(coords_dic_final,tag='_cor',tag_best='',sc_th=7.5,n_th=5,restart=True,local_enrich=False):
    """updates coords_dic with best_code value"""
    keysOB = list(coords_dic_final.keys())
    for key_ob in tqdm(keysOB):
        ob_dic = coords_dic_final[key_ob]
        if 'npts_pm' in ob_dic:
            bcode = ob_dic.get('best_code'+tag_best,[-1])[0]
            if (bcode==-1) or restart:
            
                enrichment_pm = ob_dic['enrichment_pm'].copy()
                enrichment_multi = ob_dic['enrichment_multi'].copy()

                if local_enrich:
                    enrichment_pm = ob_dic['npts_pm_cor']/(ob_dic['npts_pm']-ob_dic['npts_pm_cor'])
                    enrichment_pm = ob_dic['npts_multi_cor']/(ob_dic['npts_multi']-ob_dic['npts_multi_cor'])
                #enrichment_pm = 
                npts_pm = ob_dic['npts_pm'+tag].copy()
                npts_multi = ob_dic['npts_multi'+tag].copy()

                npts_pm[enrichment_pm<sc_th]=0
                npts_multi[enrichment_multi<sc_th]=0
                
                npts_multi = np.max([npts_pm,npts_multi],0)
                ibc = np.argmax(npts_multi)
                max_pt = np.max(npts_multi)
                enrichment_multi = np.array([enrichment_pm,enrichment_multi])[np.argmax([npts_pm,npts_multi],0),np.arange(len(enrichment_pm))]
                
                best_code = ibc+1000*ob_dic['is_lib2'] if max_pt>=n_th else -1

                coords_dic_final[key_ob]['best_code'+tag_best]=(best_code,enrichment_multi[ibc],np.max(npts_multi))
                
def get_dic_pairs(coords_dic_final,delta = 500,ids_ = [0,1,2,3],icodes = list(np.arange(499))+list(1000+np.arange(622))):                
    dic_pairs = {}
    
    for icode_ in tqdm(icodes[:]):
        OBsC =[]
        for id_ in ids_:
            OB = np.array([OB for OB in coords_dic_final 
                              if coords_dic_final[OB].get('best_code',[-1,0])[0]==icode_ and coords_dic_final[OB]['id']==id_])
            OBsC.append(OB)
        from scipy.spatial.distance import cdist
        pairs = []
        for i1 in range(len(OBsC)):
            for i2 in range(i1):
                X1 = [coords_dic_final[OB]['xyzT_aligned2'] for OB in OBsC[i1]]
                X2 = [coords_dic_final[OB]['xyzT_aligned2'] for OB in OBsC[i2]]
                if len(X1)>0 and len(X2)>0:
                    M = cdist(X1,X2)

                    i1s,i2s = np.where(M<delta)

                    for ob1,ob2 in zip(OBsC[i1][i1s],OBsC[i2][i2s]):
                        pairs.append([ob1,ob2])

        #OB_kp = np.unique(pairs)
        dic_pairs[icode_] = pairs
    return dic_pairs
def map_on_surface(XR,CM,D_max=5000,Dres=50,D=200,n = 500,plt_val=True):

    #Fibonacci sampling

    goldenRatio = (1 + 5**0.5)/2
    i = np.arange(0, n)
    theta = 2 *np.pi * i / goldenRatio
    phi = np.arccos(1 - 2*(i+0.5)/n)
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);
    dir_ = np.array([x, y, z]).T

    from scipy.spatial.distance import cdist
    ts = np.arange(0,D_max,Dres)
    XSurf = []
    for dir__ in tqdm(dir_):
        pts_ = np.array([CM+t_*dir__ for t_ in ts])
        M_kp = cdist(pts_,XR)<D
        npts = np.sum(M_kp,-1)
        best_pos = np.argmax(npts)
        Npt = np.max(npts)
        Xsel = XR[M_kp[best_pos]]
        if Npt>10:
            XSurf.append(np.mean(Xsel,0))
        else:
            XSurf.append([np.nan]*3)
    XSurf = np.array(XSurf)
    if plt_val:
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        #ax.scatter(Xsel[::,0],Xsel[::,1],Xsel[::,2])#,s=1) 
        ax.scatter(XR[::,0],XR[::,1],XR[::,2],s=0.04)
        ax.scatter(XSurf[::,0],XSurf[::,1],XSurf[::,2],s=20)
    return XSurf
def align_ims(im1,im2,angle=None,plt_val=True,pad=1000,return_im=False):
    def get_im(im,xm,xM,pad = 0 ):
        xm,xM = np.array(xm,dtype=int)-pad,np.array(xM,dtype=int)+pad
        sh = xM-xm
        sh = list(sh)+list(im.shape[2:])
        im_ = np.zeros(sh,dtype=im.dtype)
        xm_ = xm.copy()
        xm_[xm_<0]=0
        im_send = im[xm_[0]:xM[0],xm_[1]:xM[1]]
        sh_ = im_send.shape
        xm__ = xm_-xm
        im_[xm__[0]:+xm__[0]+sh_[0],xm__[1]:+xm__[1]+sh_[1]]=im_send
        return im_
    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result
    if angle is None:
        imfftms = []
        angles = np.arange(0,360,2)
        for angle in angles:
            imfftm = np.max(fftconvolve(rotate_image(im1[::-1,::-1], angle),im2,mode='full'))
            imfftms.append(imfftm)
        angle = angles[np.argmax(imfftms)]

        for _ in range(3):
            imfftms=[]
            dangle = np.abs(angles[0]-angles[1])
            angles = np.linspace(angle-2*dangle,angle+2*dangle,10)
            for angle in angles:
                imfftm = np.max(fftconvolve(rotate_image(im1[::-1,::-1], angle),im2,mode='full'))
                imfftms.append(imfftm)
            angle = angles[np.argmax(imfftms)]
    if type(angle) is list:
        #print('here')
        imfftms = []
        angles = list(angle)
        for angle in angles:
            imfftm = np.max(fftconvolve(rotate_image(im1[::-1,::-1], angle),im2,mode='full'))
            imfftms.append(imfftm)
        angle = angles[np.argmax(imfftms)]

        for _ in range(3):
            imfftms=[]
            dangle = np.abs(angles[0]-angles[1])
            angles = np.linspace(angle-2*dangle,angle+2*dangle,10)
            for angle in angles:
                imfftm = np.max(fftconvolve(rotate_image(im1[::-1,::-1], angle),im2,mode='full'))
                imfftms.append(imfftm)
            angle = angles[np.argmax(imfftms)]
    fftim = fftconvolve(rotate_image(im1, angle)[::-1,::-1],im2,mode='full')
    score = np.max(fftim)
    if plt_val:
        plt.figure()
        plt.imshow(fftim)
    xm = np.array(np.unravel_index(np.argmax(fftim),fftim.shape))-im1.shape
    sh = np.array(im1.shape)
    xM = xm+sh
    #1000
    im2T = get_im(im2,xm,xM,pad = pad )
    im2T = rotate_image(im2T, -angle)[pad:-pad,pad:-pad]
    
    im2T_ = (im2T-np.min(im2T))/(np.max(im2T)-np.min(im2T))
    im2_ = (im2-np.min(im2))/(np.max(im2)-np.min(im2))
    im1_ = (im1-np.min(im1))/(np.max(im1)-np.min(im1))
    #score = 2*np.sum(im2T_*im1)/(np.sum(im1_)+np.sum(im2_))
    #score = np.mean(im2T*im1)/np.sqrt(np.mean(im1*im1)*np.mean(im2T*im2T))
    score = np.corrcoef(im2T.ravel(),im1.ravel())[0,1]
    if plt_val:
        f,(ax1,ax2)=plt.subplots(1,2,sharex=True,sharey=True)
        ax1.imshow(im1,cmap='gray')
        ax2.imshow(im2T,cmap='gray')
        plt.show()
    if return_im:
        return angle,xm[0],xm[1],score,im2T
    return angle,xm[0],xm[1],score

def get_im2D(im,xm,xM,pad = 0 ,const=0):
    xm,xM = np.array(xm,dtype=int)-pad,np.array(xM,dtype=int)+pad
    sh = xM-xm
    sh = list(sh)#+list(im.shape[2:])
    #print(sh)
    im_ = np.zeros(sh,dtype=im.dtype)+const
    xm_ = xm.copy()
    xm_[xm_<0]=0
    im_send = im[xm_[0]:xM[0],xm_[1]:xM[1]]
    sh_ = im_send.shape
    xm__ = xm_-xm
    im_[xm__[0]:+xm__[0]+sh_[0],xm__[1]:+xm__[1]+sh_[1]]=im_send
    return im_
def get_im(im,xm,xM,pad = 0 ,const=0):
    xm,xM = np.array(xm,dtype=int)-pad,np.array(xM,dtype=int)+pad
    sh = xM-xm
    sh = list(sh)#+list(im.shape[2:])
    #print(sh)
    im_ = np.zeros(sh,dtype=im.dtype)+const
    xm_ = xm.copy()
    xm_[xm_<0]=0
    im_send = im[xm_[0]:xM[0],xm_[1]:xM[1],xm_[2]:xM[2]]
    sh_ = im_send.shape
    xm__ = xm_-xm
    im_[xm__[0]:+xm__[0]+sh_[0],xm__[1]:+xm__[1]+sh_[1],xm__[2]:+xm__[2]+sh_[2]]=im_send
    return im_
def get_new_res_org(X_fl,nRs = 15,cor_th = 0.33,dinstance_th = 2, save_cts=True,nth=np.inf,th_dapis=[1.15,1.35,1.5,2,3,5],tag_OB = None,etag=''):
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import pdist
    import numpy as np
    import glob,os,sys
    import cv2
    from tqdm import tqdm_notebook as tqdm
    #from sklearn.cluster import KMeans
    from imp import reload
    import matplotlib.pylab as plt
    import pickle
    import time
    
    res_org_fl = X_fl.replace('_Xsf.npy','_res_org.pkl')
    if not os.path.exists(res_org_fl):
    
        analysis_folder = os.path.dirname(os.path.dirname(X_fl))
        Xsff = np.load(X_fl)
        cd_to_olfr,olfr_to_cd,codes_valid1,codes_valid2 = pickle.load(open(r'\\mcclintock\mcclintock_5\Bogdan-OB-analysis\cd_to_olfr_codes_valid.pkl','rb'))
        codes = [codes_valid1,codes_valid2][int('6,7,8,9' in analysis_folder)]



        
        Xsf = Xsff
        XOB = Xsf[:,9].astype(int)
        Xdapi = Xsf[:,7]
        XRs = Xsf[:,6].astype(int)
        Xcor = Xsf[:,5]
        Xhabs = Xsf[:,4]
        Xhnorm = Xsf[:,3]


        
        #cutoff brightness/correlation and inside OB
        #keep = (XOB>0)&(Xcor>cor_th)
        keep = (Xcor>cor_th)
        #Xsf = Xsf[keep]
        #self.set_X_vals(Xsf)
        # Find all intersections
        Xs = Xsf[keep,:3]
        if len(Xs)>0:
            Ts = cKDTree(Xs)
            res = Ts.query_ball_tree(Ts,dinstance_th)
            keep_i = np.where(keep)[0]
            res=[keep_i[r_] for r_ in res]
            #resS = np.array([np.array(eval(e)) for e in np.unique([str(list(np.sort(r))) for r in res])])#make unique

        # ignore R>15:
        resk = [r[XRs[r]<nRs] for r in res]

        visited = np.zeros([len(codes),len(Xcor)])
        visited_multi = np.zeros([len(codes),len(Xcor)])
        res_org = [[],[],[[] for cd in codes],[[] for cd in codes]]
        for r in tqdm(resk):
            if len(r)==4:
                iorg = 2
                R_ = XRs[r]
                icds = np.where(np.all(R_==codes,-1))[0]
                if len(icds)>0:
                    icd = icds[0]
                    if np.sum(visited[icd,r])==0:#not visited
                        visited[icd,r]=1
                        res_org[iorg][icd].append(r)

        for r in tqdm(resk):
            if len(r)>4: #multipoint
                iorg = 3
                #if np.sum(visited[r])==0:
                #if np.sum(visited_multi[0][r])==0:
                #visited_multi[0][r]=1
                X_ = Xsff[r,:3]#.copy()
                R_ = XRs[r]
                #np.in1d(R_,cd)
                codes_in_R = np.in1d(codes,R_,assume_unique=True).reshape(codes.shape)
                cds = np.all(codes_in_R,-1)
                if np.sum(cds)>0:
                    #visited[r]=1
                    icds = np.where(cds)[0]
                    scores= []

                    for icd_,cd in zip(icds,codes[icds]):
                        X_ = Xsff[r][np.in1d(R_,cd)]
                        #metric is distance from center
                        x_ = X_[:,:3]
                        #score = np.mean(np.abs(np.mean(x_,0)-x_))
                        score = np.mean(pdist(x_))
                        scores.append(score)
                    icd_save = icds[np.argmin(scores)] #best index of code
                    if np.sum(visited[icd_save,r])==0 and np.sum(visited_multi[icd_save,r])==0:
                        visited_multi[icd_save,r] = 1
                        res_org[iorg][icd_save].append(r)

        pickle.dump(res_org,open(res_org_fl,'wb'))            
        
    if save_cts:
        #import time
        #start = time.time()
        if os.path.exists(res_org_fl):
            res_org = pickle.load(open(res_org_fl,'rb'))
            #print(start-time.time())            
            analysis_folder = os.path.dirname(os.path.dirname(X_fl))
            Xsff = np.load(X_fl)
            #print(start-time.time())     
            cd_to_olfr,olfr_to_cd,codes_valid1,codes_valid2 = pickle.load(open(r'\\mcclintock\mcclintock_5\Bogdan-OB-analysis\cd_to_olfr_codes_valid.pkl','rb'))
            codes = [codes_valid1,codes_valid2][int('6,7,8,9' in analysis_folder)]
            
            Xsf = Xsff
            XOB = Xsf[:,9].astype(int)
            Xdapi = Xsf[:,7]
            
        infos = np.array([ln[:-1].split('\t') for ln in open(analysis_folder+os.sep+'dapi_resc8.infos','r')])
        #resc_mos = int(infos[0][-1])
        fovs_mos = [os.path.basename(dax).split('.')[0] for dax in infos[:,0]]
        fov = os.path.basename(X_fl).split('final_fits__')[-1].split('_Xsf')[0].split('.')[0]
        if fov in fovs_mos:
            fov_index = fovs_mos.index(fov)
            if tag_OB is not None:
            
                file_tag=tag_OB#'glom3d_new.tif'
                filename=analysis_folder+os.sep+file_tag
                im_OB = tifffile.imread(filename,key=fov_index)[::-1]
                sh = [2048,2048]
                im_OB = resize(im_OB,sh)
                X,Y = Xsf[:,1:3].astype(int).T
                sh = im_OB.shape
                X[X>=sh[0]]=sh[0]-1
                Y[Y>=sh[1]]=sh[1]-1
                X[X<0]=0
                Y[Y<0]=0
                Xsf[:,9]=im_OB[X,Y]
                XOB = Xsf[:,9].astype(int)
            OBs,AOBs,aOBs = np.load(X_fl.replace('_Xsf.npy','_AOBf.npy'),allow_pickle=True)
            for th_dapi in th_dapis:
                A_dic,OB_fov_dic = update_OBarea_fov_dic_simple(OBs,AOBs,aOBs,XOB,Xdapi,fov_index,A_dic={},OB_fov_dic={},im_OB=None,th_dapi=th_dapi)
                ct_dic_4 = update_OBcount_dic_simple(res_org[2],XOB,Xdapi,ct_dic={},n_th=nth,th_dapi=th_dapi)
                ct_dic_b4 = update_OBcount_dic_simple(res_org[3],XOB,Xdapi,ct_dic={},n_th=nth,th_dapi=th_dapi)
                save_fl_OB_cts = X_fl.replace('_Xsf.npy','_cts_dics'+etag+str(th_dapi)+'.pkl')
                pickle.dump([ct_dic_4,ct_dic_b4,A_dic,OB_fov_dic],open(save_fl_OB_cts,'wb'))
            #print(start-time.time())
        
        
def update_OBarea_fov_dic_simple(OBs,AOBs,aOBs,XOB,Xdapi,fov_index,A_dic={},OB_fov_dic={},im_OB=None,th_dapi=1.35):
    if im_OB is not None:
        OBs,AOBs = np.unique(im_OB,return_counts=True)
        aOBs = np.zeros_like(OBs)
    for ob_ in OBs:
        if ob_ not in OB_fov_dic:
            OB_fov_dic[ob_]=[fov_index]
        else:
            OB_fov_dic[ob_]+=[fov_index]
    Xoutcell = Xdapi>th_dapi
    for ob_,AOB,aOB in zip(OBs,AOBs,aOBs):
        if ob_ not in A_dic:
            A_dic[ob_]=np.array([0,0,0,0],dtype=float)
        COB,cOB=0,0
        if True:
            COB = np.sum((XOB==ob_))
            cOB = np.sum((XOB==ob_)&(Xoutcell))
        A_dic[ob_]+=np.array([aOB,AOB,cOB,COB],dtype=float)
    return A_dic,OB_fov_dic
def update_OBcount_dic_simple(res_org,XOB,Xdapi,ct_dic={},n_th=np.inf,th_dapi=1.35):
    # 0,1,2->2,3,4 point intersection, 3->multi-point best, 4->all multipoint
    Xoutcell=Xdapi<th_dapi
    ncodes = len(res_org)
    for icd,res_org_ in enumerate(res_org): 
        obs,cts_ = np.unique([XOB[e[0]]for e in res_org_ if Xoutcell[e[0]] and len(e)<=n_th],return_counts=True)
        for ob_,ct_ in zip(obs,cts_):
            if ob_ not in ct_dic:
                ct_dic[ob_]=np.zeros(ncodes)
            ct_dic[ob_][icd]+=ct_
    return ct_dic

def plot_image_glom_test(analysis_folder,OB_,icode,plot_multi=True,th_dapi = 1.35,notOB_only=True,nneigh = 9):
    import tifffile
    imglom = tifffile.imread(analysis_folder+os.sep+'glom3d.tif')
    OB_ifovs = np.where([np.any(im_==OB_) for im_ in imglom])[0]

    hfolders,fovs,htags = np.load(analysis_folder+os.sep+'files.npy',allow_pickle=True)
    fovs = np.array([fov.split('.dax')[0]for fov in fovs])
    infos = np.array([ln[:-1].split('\t') for ln in open(analysis_folder+os.sep+'dapi_resc8.infos','r')])
    resc_mos = int(infos[0][-1])
    fovs_mos = [os.path.basename(dax).split('.dax')[0] for dax in infos[:,0]]
    limits_mos = infos[:,1:5].astype(int)
    cms_mos = np.array([np.mean(limits_mos[:,:2],axis=-1),np.mean(limits_mos[:,2:],axis=-1)]).T
    ifos_mos = [fovs_mos.index(fov) for fov in fovs[OB_ifovs]]

    #expand with n(=9) neighbors
    from scipy.spatial.distance import cdist
    ifovs_mos_extended = np.unique(np.argsort(cdist(cms_mos[ifos_mos],cms_mos),axis=-1)[:,:nneigh])##############9
    limits_mos_extended = limits_mos[ifovs_mos_extended]

    xmmos,xMmos,ymmos,yMmos=[np.min(limits_mos_extended[:,0]),np.max(limits_mos_extended[:,1]),
                            np.min(limits_mos_extended[:,2]),np.max(limits_mos_extended[:,3])]
    #reextend to rectangle
    pad = 100
    keep  =(limits_mos[:,0]>=xmmos-pad)&(limits_mos[:,1]<=xMmos+pad)&(limits_mos[:,2]>=ymmos-pad)&(limits_mos[:,3]<=yMmos+pad)
    ifovs_mos_extended = np.where(keep)[0]
    limits_mos_extended = limits_mos[ifovs_mos_extended]
    xmmos,xMmos,ymmos,yMmos=[np.min(limits_mos_extended[:,0]),np.max(limits_mos_extended[:,1]),
                            np.min(limits_mos_extended[:,2]),np.max(limits_mos_extended[:,3])]

    im_dapi = tifffile.imread(analysis_folder+os.sep+'dapi_resc8.tif')
    #im_labf = tifffile.imread(analysis_folder+os.sep+'dapi_resc8--imlabf_new.tif')
    im_labf,limits = np.load(analysis_folder+os.sep+'dapi_resc8_glomLayer.tif.npy',allow_pickle=True)


    im_dapi = im_dapi[xmmos:xMmos,ymmos:yMmos]
    im_labf = im_labf[xmmos:xMmos,ymmos:yMmos]


    fovs_mos_extended = np.array(fovs_mos)[ifovs_mos_extended]
    fovs_list = list(fovs)
    ifovsF,limsF = zip(*[(fovs_list.index(fov_),lims_) for fov_,lims_ in zip(fovs_mos_extended,limits_mos_extended) if fov_ in fovs_list])
    #fovs[ifovsF]
    fovsF = fovs[list(ifovsF)]
    ifovsF




    Xs2d_all,Xs2d,Xs2d_e = [],[],[]
    for ifv,fov in enumerate(tqdm(fovsF)):
        Xf_fl = analysis_folder+os.sep+'Decoded'+os.sep+'final_fits__'+fov.replace('.dax','')+'_Xsf.npy'
        if not os.path.exists(Xf_fl):
            Xf_fl = analysis_folder+os.sep+'Decoded'+os.sep+'final_fits__'+fov.replace('.dax','')+'.zst'+'_Xsf.npy'
        if os.path.exists(Xf_fl):
            Xsff = np.load(Xf_fl)
            res_org_fl = Xf_fl.replace('_Xsf.npy','_res_org.pkl')
            if not os.path.exists(res_org_fl):
                res_org_fl = Xf_fl.replace('_Xsf.npy','_res_org.npy')
                res_org = np.load(res_org_fl,allow_pickle=True)
                res_org = remove_seq_inters(res_org,Rs,nRs=15)
            else:
                res_org = pickle.load(open(res_org_fl,'rb'))
            Rs = Xsff[:,6]
            
            Xsff_ = np.array([Xsff[r_[0]] for r_ in res_org[2][icode] if (notOB_only or Xsff[r_[0],9]>0)])
            if len(Xsff_)>0: Xs2d.extend((Xsff_[:,[2,1,7]]*[1,-1,1]+[0,2048,0])/[resc_mos,resc_mos,1]+[limsF[ifv][2]-ymmos,limsF[ifv][0]-xmmos,0])
            Xsff_ = np.array([Xsff[r_[0]] for r_ in res_org[3][icode] if ((len(r_)<=6) and (notOB_only or Xsff[r_[0],9]>0))])
            if len(Xsff_)>0: Xs2d_e.extend((Xsff_[:,[2,1,7]]*[1,-1,1]+[0,2048,0])/[resc_mos,resc_mos,1]+[limsF[ifv][2]-ymmos,limsF[ifv][0]-xmmos,0])
            #if len(Xsff)>0: Xs2d_all.extend((Xsff[:,[2,1,7]]*[1,-1,1]+[0,2048,0])/[resc_mos,resc_mos,1]+[limsF[ifv][2]-ymmos,limsF[ifv][0]-xmmos,0])#/resc_mos+[limsF[ifv][0]-xmmos,limsF[ifv][2]-ymmos][::-1])
    Xs2d = np.array(Xs2d)
    Xs2d_all = np.array(Xs2d_all)
    Xs2d_e = np.array(Xs2d_e)
    """
    plt.figure()
    plt.imshow(im_dapi,cmap='gray')
    plt.contour(im_labf>0,[0.5],colors='b')
    plt.contour(im_labf==OB_,[0.5],colors='r')
    #plt.plot(Xs2d_all[::30,0],Xs2d_all[::30,1],'y.',alpha=0.0075)
    plt.plot(Xs2d_e[:,0],Xs2d_e[:,1],'yx')
    plt.plot(Xs2d[:,0],Xs2d[:,1],'y.')
    """
    
    fig = plt.figure(figsize=(10,10))
    plt.title(str([OB_,icode]))
    plt.imshow(im_dapi,cmap='gray',vmax=np.percentile(im_dapi[im_dapi!=0],99.9))
    plt.contour(im_labf>0,[0.5],colors='b')
    plt.contour(im_labf==OB_,[0.5],colors='orange')
    #plt.plot(Xs2d_all[::30,0],Xs2d_all[::30,1],'y.',alpha=0.0075)
    #print(Xs2d[:,-1])
    keep = Xs2d[:,-1]<th_dapi
    plt.plot(Xs2d[keep,0],Xs2d[keep,1],'r.')
    if plot_multi and len(Xs2d_e)>0:#OB_dic_final[OB_][-2]!=1:
        keep = Xs2d_e[:,-1]<th_dapi
        plt.plot(Xs2d_e[keep,0],Xs2d_e[keep,1],'rx')
    plt.imshow(im_labf,alpha=0.)
    return fig
#might need to do: pip install opencv-python
def get_OB_dic_final(ct_dic_4,ct_dic_b4,A_dic,th_n=5,th_pval = -7.5):
    nlim=th_n
    zeros = ct_dic_b4[0]*0
    ndec = {OB_:ct_dic_4[OB_]+ct_dic_b4.get(OB_,zeros)for OB_ in ct_dic_4}
    scores_all,enrichment,enrichment_std,M_cts,iOBs = ndec_to_scores(ndec,A_dic)
    scores_all[M_cts.T<nlim]=0
    best_codes = np.argmin(scores_all,0)
    scores_all_sorted = np.min(scores_all,0)
    enrichment_ = [enrichment[iob,icd] for iob,icd in enumerate(best_codes)]
    OB_dic_finalV2 = {iOB:[icdb,sc,enrc,ndec[iOB][icdb]]  for iOB,icdb,sc,enrc in zip(iOBs,best_codes,scores_all_sorted,enrichment_)}
    OB_dic_finalV2

    ndec = ct_dic_4
    scores_all,enrichment,enrichment_std,M_cts,iOBs = ndec_to_scores(ndec,A_dic)
    scores_all[M_cts.T<nlim]=0
    best_codes = np.argmin(scores_all,0)
    scores_all_sorted = np.min(scores_all,0)
    enrichment_ = [enrichment[iob,icd] for iob,icd in enumerate(best_codes)]
    OB_dic_finalV3 = {iOB:[icdb,sc,enrc,ndec[iOB][icdb]]  for iOB,icdb,sc,enrc in zip(iOBs,best_codes,scores_all_sorted,enrichment_)}
    OB_dic_finalV4 = {iOB:OB_dic_finalV2[iOB]+[2] if OB_dic_finalV3[iOB][1]>OB_dic_finalV2[iOB][1] else OB_dic_finalV3[iOB]+[1]
     for iOB in OB_dic_finalV3}

    OB_dic_finalV5 = {O:OB_dic_finalV4[O] for O in OB_dic_finalV4 if OB_dic_finalV4[O][1]<th_pval}
    return OB_dic_finalV5
def ndec_to_scores(ndec,A_dic,iA=1,nexcl=5):
    iOBs = np.sort([OB_ for OB_ in ndec if OB_>0])
    all_cts = np.sum([ndec[OB_]for OB_ in iOBs],0)
    all_area = np.array([A_dic[OB_][iA]for OB_ in iOBs])
    AT = np.sum(all_area)
    medA = np.median(all_area)
    exp_dens_cts = all_cts/AT
    M = np.array([ndec[OB_]/A_dic[OB_][iA] for OB_ in iOBs])*medA#-exp_dens_cts
    M_ = np.sort(M,0)[:-nexcl]
    #M_[M_==0]=np.nan
    enrichment = M/np.nanmean(M_,0)
    enrichment_std = (M-np.nanmean(M_,0))/np.nanstd(M_,0)
    from scipy.stats import gamma
    scores_all = []
    M_cts = np.array([ndec[OB_] for OB_ in iOBs])
    for icd in range(M.shape[-1]):
        Xo = M[:,icd].copy()
        #if np.mean(Xo>0)>0.75:
        X=Xo#[Xo!=0]
        #X=Xo+1
        X = np.sort(X)[:-nexcl]
        E,V = np.mean(X),np.var(X)
        alpha = E**2/V
        beta = E/V
        scores = gamma.logsf(Xo, alpha,scale=1/beta)
        scores_all.append(scores)

        if False:
            x = np.linspace(gamma.ppf(0.2, alpha,scale=1/beta),
                        gamma.ppf(0.98, alpha,scale=1/beta), 100)
            plt.figure()
            plt.title(icd)
            plt.plot(x, gamma.pdf(x, alpha,scale=1/beta))
            #plt.figure()
            plt.hist(X,bins=50,density=True)
            plt.show()
    scores_all = np.array(scores_all)
    return scores_all,enrichment,enrichment_std,M_cts,iOBs
def remove_seq_inters(res_org,Rs,nRs=15):
    res_org_4 = [list(e) for e in res_org[2]]
    res_org_multi = res_org[3]
    res_org_multi_ = []
    res_org_4 = [list(e) for e in res_org[2]]
    for icd,res in enumerate(res_org_multi):
        res_org_multi_.append([])
        for cmb in res:
            Rs_ = Rs[cmb]
            keep = Rs_<nRs
            cmb_ = cmb[keep]
            if len(cmb_)==4:
                res_org_4[icd].append(cmb_)
            else:
                res_org_multi_[-1].append(cmb_)
    res_org[2] = res_org_4
    res_org[3] = res_org_multi_
    return res_org
def get_dice(X1,X2,xmax=10000):
    nint = float(len(np.intersect1d(np.dot(X1,[xmax,1]),np.dot(X2,[xmax,1]))))
    dice = 2*nint/(len(X1)+len(X2))
    fr1,fr2 = nint/len(X1),nint/len(X2)
    return dice,fr1,fr2,np.max([fr1,fr2])
def get_closest_gloms(dic_glomeruli,OB_,delta=1,fr_th=0.6,ath = 1000,dglom = 70):
    dic_glom = dic_glomeruli[OB_]
    islice = dic_glom['islice']
    cm = dic_glom['cm']
    islice_ = islice+delta
    OBs_ = [OB__ for OB__ in dic_glomeruli if dic_glomeruli[OB_]['area']>ath 
            if dic_glomeruli[OB__]['islice']==(islice+delta)
            if np.linalg.norm(dic_glomeruli[OB__]['cm']-cm)<3*dglom
            if get_dice(dic_glomeruli[OB_]['coords'],dic_glomeruli[OB__]['coords'])[-1]>fr_th]
    return OBs_
def plt_image_centered_glomeruli(OB_dic_final,OB_ = 1310049,icode=None,th_dapi=1.15,nneigh=9,color='r'):
    if icode is None:
        icode = int(OB_dic_final[OB_][0])#########################################
    

    dic_analysis_folder = pickle.load(open(r'//meitner/f/Bogdan/glomeruli_alignment/dic_analysis_folder.pkl','rb'))

    analysis_folder,OB_ifovs = dic_analysis_folder[OB_]
    codes_valid,code_dic = get_code(analysis_folder)
    olfr_nm = code_dic[tuple(codes_valid[icode])]
    hfolders,fovs,htags = np.load(analysis_folder+os.sep+'files.npy',allow_pickle=True)
    fovs = [fov.split('.dax')[0]for fov in fovs]
    infos = np.array([ln[:-1].split('\t') for ln in open(analysis_folder+os.sep+'dapi_resc8.infos','r')])
    resc_mos = int(infos[0][-1])
    fovs_mos = [os.path.basename(dax).split('.dax')[0] for dax in infos[:,0]]
    limits_mos = infos[:,1:5].astype(int)
    cms_mos = np.array([np.mean(limits_mos[:,:2],axis=-1),np.mean(limits_mos[:,2:],axis=-1)]).T
    ifos_mos = [fovs_mos.index(fov) for fov in fovs[OB_ifovs]]

    #expand with n(=9) neighbors
    from scipy.spatial.distance import cdist
    ifovs_mos_extended = np.unique(np.argsort(cdist(cms_mos[ifos_mos],cms_mos),axis=-1)[:,:nneigh])###9
    limits_mos_extended = limits_mos[ifovs_mos_extended]

    xmmos,xMmos,ymmos,yMmos=[np.min(limits_mos_extended[:,0]),np.max(limits_mos_extended[:,1]),
                            np.min(limits_mos_extended[:,2]),np.max(limits_mos_extended[:,3])]
    #reextend to rectangle
    pad = 100
    keep  =(limits_mos[:,0]>=xmmos-pad)&(limits_mos[:,1]<=xMmos+pad)&(limits_mos[:,2]>=ymmos-pad)&(limits_mos[:,3]<=yMmos+pad)
    ifovs_mos_extended = np.where(keep)[0]
    limits_mos_extended = limits_mos[ifovs_mos_extended]
    xmmos,xMmos,ymmos,yMmos=[np.min(limits_mos_extended[:,0]),np.max(limits_mos_extended[:,1]),
                            np.min(limits_mos_extended[:,2]),np.max(limits_mos_extended[:,3])]

    im_dapi = tifffile.imread(analysis_folder+os.sep+'dapi_resc8.tif')
    im_labf = tifffile.imread(analysis_folder+os.sep+'dapi_resc8--imlabf_new.tif')
    im_dapi = im_dapi[xmmos:xMmos,ymmos:yMmos]
    im_labf = im_labf[xmmos:xMmos,ymmos:yMmos]


    fovs_mos_extended = np.array(fovs_mos)[ifovs_mos_extended]
    fovs_list = list(fovs)
    ifovsF,limsF = zip(*[(fovs_list.index(fov_),lims_) for fov_,lims_ in zip(fovs_mos_extended,limits_mos_extended) if fov_ in fovs_list])
    #fovs[ifovsF]
    fovsF = fovs[list(ifovsF)]
    ifovsF


    res_orggs=[]

    Xs2d_all,Xs2d,Xs2d_e = [],[],[]
    for ifv,fov in enumerate(fovsF):
        Xf_fl = analysis_folder+os.sep+'Decoded'+os.sep+'final_fits__'+fov.replace('.dax','')+'_Xsf.npy'
        if os.path.exists(Xf_fl):
            Xsff = np.load(Xf_fl)
            res_org = np.load(Xf_fl.replace('_Xsf.npy','_res_org.npy'),allow_pickle=True)

            Xsff_ = np.array([Xsff[r_[0]] for r_ in res_org[-3][icode]])
            if len(Xsff_)>0: Xs2d.extend((Xsff_[:,[2,1,7]]*[1,-1,1]+[0,2048,0])/[resc_mos,resc_mos,1]+[limsF[ifv][2]-ymmos,limsF[ifv][0]-xmmos,0])

            #XRs = Xsff[:,6].astype(int)
            #cd = codes_valid[icode]
            #res_org_ = set([tuple(e[np.in1d(XRs[e],cd)]) for e in res_org[-2][icode] if len(e)<=6])
            #res_orggs.append(res_org_)
            res_org_ = res_org[-2][icode]
            Xsff_ = np.array([Xsff[r_[0]] for r_ in res_org_])
            if len(Xsff_)>0: Xs2d_e.extend((Xsff_[:,[2,1,7]]*[1,-1,1]+[0,2048,0])/[resc_mos,resc_mos,1]+[limsF[ifv][2]-ymmos,limsF[ifv][0]-xmmos,0])
            if len(Xsff)>0: Xs2d_all.extend((Xsff[:,[2,1,7]]*[1,-1,1]+[0,2048,0])/[resc_mos,resc_mos,1]+[limsF[ifv][2]-ymmos,limsF[ifv][0]-xmmos,0])#/resc_mos+[limsF[ifv][0]-xmmos,limsF[ifv][2]-ymmos][::-1])
    Xs2d = np.array(Xs2d)
    Xs2d_all = np.array(Xs2d_all)
    Xs2d_e = np.array(Xs2d_e)
    """
    plt.figure()
    plt.imshow(im_dapi,cmap='gray')
    plt.contour(im_labf>0,[0.5],colors='b')
    plt.contour(im_labf==OB_,[0.5],colors='r')
    #plt.plot(Xs2d_all[::30,0],Xs2d_all[::30,1],'y.',alpha=0.0075)
    plt.plot(Xs2d_e[:,0],Xs2d_e[:,1],'yx')
    plt.plot(Xs2d[:,0],Xs2d[:,1],'y.')
    """
    fig = plt.figure()
    plt.title(str([OB_,icode,olfr_nm])+'\n'+str(OB_dic_final.get(OB_,'Not in')))
    plt.imshow(im_dapi,cmap='gray',vmax=np.percentile(im_dapi[im_dapi!=0],99.9))
    plt.contour(im_labf>0,[0.5],colors='b')
    plt.contour(im_labf==OB_,[0.5],colors='orange')
    #plt.plot(Xs2d_all[::30,0],Xs2d_all[::30,1],'y.',alpha=0.0075)
    keep = Xs2d[:,-1]<th_dapi
    plt.plot(Xs2d[keep,0],Xs2d[keep,1],'.',color=color)#######################o
    if True:#OB_dic_final[OB_][-2]!=1:
        keep = Xs2d_e[:,-1]<th_dapi
        plt.plot(Xs2d_e[keep,0],Xs2d_e[keep,1],'.',color=color)
    plt.imshow(im_labf%10000,alpha=0)
    return fig

def print_top_statistics_OB(OB_dic_final,OB_ = 1310051 ,tag_files=('1p15u_area_neigh50','1p15u'),min_score=5,min_no=2):
    dic_analysis_folder = pickle.load(open(r'//meitner/f/Bogdan/glomeruli_alignment/dic_analysis_folder.pkl','rb'))
    analysis_folder,ifovs = dic_analysis_folder[OB_]
    print(OB_,OB_dic_final.get(OB_,'Not in'))
    print(analysis_folder)
    print()
    
    
    
    stats_intersections_fl = analysis_folder+os.sep+'FinalDecoding'+os.sep+'stats_intersections'+tag_files[0]+'.npy' 
    stats_intersections = np.load(stats_intersections_fl,allow_pickle=True)
    ct_A_OBfov2_new_fl = analysis_folder+os.sep+'FinalDecoding'+os.sep+'ct_A_OBfov'+tag_files[1]+'_new.npy' 
    ct_dic_4,ct_dic_b4,A_dic,OB_fov_dic = np.load(ct_A_OBfov2_new_fl,allow_pickle=True)

    for OB__,(best_icds,scores_icds,ncts),(best_icds_e,scores_icds_e,ncts_e) in stats_intersections:
        if OB_==OB__:
            keep = (scores_icds>min_score)&(ncts>min_no)
            print(best_icds[keep])
            print(np.round(scores_icds[keep],0))
            print(ncts[keep])
            print()
            keep = (scores_icds_e>min_score)&(ncts_e>min_no)
            print(best_icds_e[keep])
            print(np.round(scores_icds_e[keep],0))
            print(ncts_e[keep])



def load_OBcount_dic_Xsf(self,Xsf_dic={},ires_org = 2,set_unique = None,n_th=np.inf,th_dapi=None,save=None):
    res_org = self.res_org[ires_org]# 0,1,2->2,3,4 point intersection, 3->multi-point best, 4->all multipoint
    if th_dapi is not None: self.Xoutcell = self.Xdapi<th_dapi
    obs_u,cts_ob = np.unique(self.XOB[self.Xoutcell],return_counts=True)
    for ob_ in obs_u:
        if ob_ not in Xsf_dic:
            Xsf_dic[ob_]=[[] for _ in range(len(self.codes_valid))]
    for icd,cd in enumerate(self.codes_valid): 
        res_org_ = res_org[icd]
        res_org_ = [e for e in res_org_ if self.Xoutcell[e[0]] and len(e)<=n_th]
        if set_unique is not None:
            visited = list(np.unique(self.res_org[set_unique][icd])) #all the visited points
            res_org__ = []
            for e in res_org_:
                egood = e[np.in1d(self.XRs[e],cd)]
                if not np.any([e_ in visited for e_ in egood]):
                    visited.extend(egood)
                    res_org__.append(e)
            res_org_ = res_org__
        for e in res_org_:
            Xsf_ = self.Xsff[e]
            Xsf_dic[self.XOB[e[0]]][icd] += [Xsf_]
    if save is not None:
        fl_save = self.analysis_folder+os.sep+'FinalDecoding'+os.sep+'Xsf_dic_ifov'+str(str(self.fov_index))+'_org'+str(ires_org)+'.pkl'
        pickle.dump(Xsf_dic,open(fl_save,'wb'))
    return Xsf_dic

def get_code(analysis_folder,fl = r'SI8_final.fasta'):
    if 'ORN' in os.path.abspath(fl):
        lines = [ln for ln in open(fl,'r') if '_indexAB:' in ln and 'newORLib_32bit' in ln ]
        code_dic = {}
        for ln in lines:
            lib_id = int("['fwd_W1B11', 'rev_W1B12']" not in ln)#int(eval(ln.split('_indexAB:')[-1].split('_')[0])[0]/2)
            #if lib_id<=10:
            code = [cd+1 for cd in eval(ln.split('_code:')[-1].split('_')[0])]
            olfr = ln[1:].split('_')[0]
            code_dic[str(code)]=olfr+'_'+str(lib_id)
            #all_codes.append(code)
        codes = np.array(list(map(eval,code_dic.keys())))
    else:
        lines = [ln for ln in open(fl,'r')][::2]
        code_dic = {tuple(eval(ln.split('_code(Steven):')[-1].split('_')[0])):ln.split('_')[0][1:] 
                    for ln in lines if '_code(Steven):' in ln}
        codes = np.array([ln for ln in code_dic.keys()])



        lines = [ln for ln in open(fl,'r') if '_indexAB:' in ln and '_code(Steven):' in ln]
        code_dic = {}
        for ln in lines:
            lib_id = int(eval(ln.split('_indexAB:')[-1].split('_')[0])[0]/2)
            if lib_id<=10:
                code = [cd for cd in eval(ln.split('_code(Steven):')[-1].split('_')[0])]
                olfr = ln[1:].split('_')[0]
                code_dic[tuple(code)]=olfr+'_'+str(lib_id)

        lines_extra_lib = [ln for ln in open(fl,'r') if '_indexAB:[42, 41]'in ln]
        for ln in lines_extra_lib:
            lib_id = 11
            code = [cd for cd in eval(ln.split('_code:')[-1].split('_')[0])]
            olfr = ln[1:].split('_')[0]
            code_dic[tuple(code)]=olfr+'_'+str(lib_id)
        codes = np.array([ln for ln in code_dic.keys()])

    #code_dic =code_dic
    #codes =codes
    libs = eval(os.path.basename(analysis_folder).split('_lib')[-1].split('_')[0])
    codes_valid = np.array([cd for cd in code_dic if int(code_dic[cd].split('_')[-1]) in libs])
    return codes_valid,code_dic
def to_uint8_3col(im):
    im_ = np.array(im,dtype=np.float32)
    min_,max_ = im_.min(),im_.max()
    im_ = (im_-min_)/(max_-min_)
    im_ = (im_*255).astype(np.uint8)
    return np.dstack([im_,im_,im_])
    
def get_standard_analysis_per_slide_v2(islice,min_RNA_per_glom=5,th_score=10,fraction=0.9,area_th=1000,
                                    fld_dic=None,save=True,show_statistics=True,tag_files=('1p15u','1p15u')):
    if fld_dic is None:
        fldrs = np.array(glob.glob(r'\\meitner\f\Bogdan\*-AnalysisOB'))
        ob_ind = [int(os.path.basename(fld).split('_OB')[1].split('-')[0].split('_')[0])for fld in fldrs]
        fldrs[np.argsort(ob_ind)]
        fld_dic = {ob_:fld for ob_, fld in zip(ob_ind,fldrs)}
    
    #print('here1')
    
    #Load the info
    analysis_folder = fld_dic[islice]
    
    stats_intersections_fl = analysis_folder+os.sep+'FinalDecoding'+os.sep+'stats_intersections'+tag_files[0]+'.npy' 
    stats_intersections = np.load(stats_intersections_fl,allow_pickle=True)
    ct_A_OBfov2_new_fl = analysis_folder+os.sep+'FinalDecoding'+os.sep+'ct_A_OBfov'+tag_files[1]+'_new.npy' 
    ct_dic_4,ct_dic_b4,A_dic,OB_fov_dic = np.load(ct_A_OBfov2_new_fl,allow_pickle=True)
    
    #run
    OBs_all = []
    OB_dic_final = {}
    area_obs={}
    if area_th>0:
        area_obs = pickle.load(open(r'\\meitner\f\Bogdan\glomeruli_alignment\slide_alignment\non_rigid_gaussian\area_obs.pkl','rb'))
    for OB_,(best_icds,scores_icds,ncts),(best_icds_e,scores_icds_e,ncts_e) in stats_intersections:
        if area_obs.get(OB_,1)>area_th:
            OBs_all.append(OB_)
            combs = [(icd,sc,nct,1) for icd,sc,nct in zip(best_icds,scores_icds,ncts)]
            combs += [(icd,sc,nct,2) for icd,sc,nct in zip(best_icds_e,scores_icds_e,ncts_e)]
            combs = np.array(combs)
            scores,counts = combs[:,1].copy(),combs[:,2]
            scores[counts<min_RNA_per_glom]=0
            max_sc = np.max(scores)
            if max_sc>=th_score:
                keep = scores>=(max_sc*fraction)
                cand_inds = np.where(keep)[0]
                besti = cand_inds[np.argmax(counts[cand_inds])]
                best_comb = combs[besti]
                #if best_comb[1]>th_score:
                OB_dic_final[OB_] = best_comb
            
            
    tag_analysis = os.path.basename(analysis_folder)
    libs = eval(tag_analysis.split('_lib')[1].split('_')[0])
    for OB_ in OB_dic_final:
        OB_dic_final[OB_] = list(OB_dic_final[OB_])+[libs]
        
        
    if save:
        pickle.dump(OB_dic_final,open(analysis_folder+os.sep+'FinalDecoding'+os.sep+'OB_dic_final.npy','wb'))
    #print('here5')
    if show_statistics:
        print(tag_analysis)
        OBs_decoded = list(OB_dic_final.keys())

        print("Number of glomeruli decoded:",len(OBs_decoded),"; Fraction decoded from total: ",
              np.round(1.*len(OBs_decoded)/len(OBs_all),2))
        As_decoded = [A_dic[OB_][-2] for OB_ in OBs_decoded]
        As_all = [A_dic[OB_][-2] for OB_ in OBs_all]
        Am,AM = np.percentile(As_decoded,5),np.percentile(As_decoded,95)
        #print(Am)
        print("Detection efficiency:",np.round(1.*len(OBs_decoded)/np.sum(np.array(As_all)>Am),2))

        cds_,ncds_ = np.unique([OB_dic_final[OB_][0] for OB_ in OB_dic_final],return_counts=True)
        print("Mean number of glomeruli per gene:",np.mean(ncds_))
        print("Number of genes decoded:",len(cds_))
        #np.array(list(zip(cds_[np.argsort(ncds_)],np.sort(ncds_))))
    return OB_dic_final    
def get_standard_analysis_per_slide_v3(islice,min_RNA_per_glom=5,th_score=10,fraction=0.9,area_th=1000,
                                    fld_dic=None,save=True,show_statistics=True,tag_files=('1p15u','1p15u')):
    if fld_dic is None:
        fldrs = np.array(glob.glob(r'\\meitner\f\Bogdan\*-AnalysisOB'))
        ob_ind = [int(os.path.basename(fld).split('_OB')[1].split('-')[0].split('_')[0])for fld in fldrs]
        fldrs[np.argsort(ob_ind)]
        fld_dic = {ob_:fld for ob_, fld in zip(ob_ind,fldrs)}
    
    #print('here1')
    
    #Load the info
    analysis_folder = fld_dic[islice]
    
    stats_intersections_fl = analysis_folder+os.sep+'FinalDecoding'+os.sep+'stats_intersections'+tag_files[0]+'.npy' 
    stats_intersections = np.load(stats_intersections_fl,allow_pickle=True)
    ct_A_OBfov2_new_fl = analysis_folder+os.sep+'FinalDecoding'+os.sep+'ct_A_OBfov'+tag_files[1]+'_new.npy' 
    ct_dic_4,ct_dic_b4,A_dic,OB_fov_dic = np.load(ct_A_OBfov2_new_fl,allow_pickle=True)
    
    #run
    OBs_all = []
    OB_dic_final = {}
    area_obs={}
    if area_th>0:
        area_obs = pickle.load(open(r'\\meitner\f\Bogdan\glomeruli_alignment\slide_alignment\non_rigid_gaussian\area_obs.pkl','rb'))
    for OB_,(best_icds,scores_icds,ncts),(best_icds_e,scores_icds_e,ncts_e) in stats_intersections:
        if area_obs.get(OB_,1)>area_th:
            OBs_all.append(OB_)
            
            combs = [(icd,sc,nct,2) for icd,sc,nct in zip(best_icds_e,scores_icds_e,ncts_e)]
            combs = np.array(combs)
            scores,counts = combs[:,1].copy(),combs[:,2]
            scores[counts<min_RNA_per_glom]=0
            max_sc = np.max(scores)
            if max_sc>=th_score:
                keep = scores>=(max_sc*fraction)
                cand_inds = np.where(keep)[0]
                besti = cand_inds[np.argmax(counts[cand_inds])]
                best_comb = combs[besti]
                if best_comb[1]>th_score:
                    OB_dic_final[OB_] = best_comb
            """
            else:
                combs = [(icd,sc,nct,1) for icd,sc,nct in zip(best_icds,scores_icds,ncts)]
                combs = np.array(combs)
                scores,counts = combs[:,1].copy(),combs[:,2]
                scores[counts<min_RNA_per_glom]=0
                max_sc = np.max(scores)
                if max_sc>=th_score:
                    keep = scores>=(max_sc*fraction)
                    cand_inds = np.where(keep)[0]
                    besti = cand_inds[np.argmax(counts[cand_inds])]
                    best_comb = combs[besti]
                    if best_comb[1]>th_score:
                        OB_dic_final[OB_] = best_comb
            """
    tag_analysis = os.path.basename(analysis_folder)
    libs = eval(tag_analysis.split('_lib')[1].split('_')[0])
    for OB_ in OB_dic_final:
        OB_dic_final[OB_] = list(OB_dic_final[OB_])+[libs]
        
        
    if save:
        pickle.dump(OB_dic_final,open(analysis_folder+os.sep+'FinalDecoding'+os.sep+'OB_dic_final.npy','wb'))
    #print('here5')
    if show_statistics:
        print(tag_analysis)
        OBs_decoded = list(OB_dic_final.keys())

        print("Number of glomeruli decoded:",len(OBs_decoded),"; Fraction decoded from total: ",
              np.round(1.*len(OBs_decoded)/len(OBs_all),2))
        As_decoded = [A_dic[OB_][-2] for OB_ in OBs_decoded]
        As_all = [A_dic[OB_][-2] for OB_ in OBs_all]
        Am,AM = np.percentile(As_decoded,5),np.percentile(As_decoded,95)
        #print(Am)
        print("Detection efficiency:",np.round(1.*len(OBs_decoded)/np.sum(np.array(As_all)>Am),2))

        cds_,ncds_ = np.unique([OB_dic_final[OB_][0] for OB_ in OB_dic_final],return_counts=True)
        print("Mean number of glomeruli per gene:",np.mean(ncds_))
        print("Number of genes decoded:",len(cds_))
        #np.array(list(zip(cds_[np.argsort(ncds_)],np.sort(ncds_))))
    return OB_dic_final       
def get_standard_analysis_per_slide(islice,min_RNA_per_glom=5,th_score1=10,th_score2s=(12.5,1,False),area_th=1000,
                                    fld_dic=None,save=True,show_statistics=True,tag_files=('1p15u','1p15u')):
    if fld_dic is None:
        fldrs = np.array(glob.glob(r'\\meitner\f\Bogdan\*-AnalysisOB'))
        ob_ind = [int(os.path.basename(fld).split('_OB')[1].split('-')[0].split('_')[0])for fld in fldrs]
        fldrs[np.argsort(ob_ind)]
        fld_dic = {ob_:fld for ob_, fld in zip(ob_ind,fldrs)}
    
    #print('here1')
    
    #Load the info
    analysis_folder = fld_dic[islice]
    
    stats_intersections_fl = analysis_folder+os.sep+'FinalDecoding'+os.sep+'stats_intersections'+tag_files[0]+'.npy' 
    stats_intersections = np.load(stats_intersections_fl,allow_pickle=True)
    ct_A_OBfov2_new_fl = analysis_folder+os.sep+'FinalDecoding'+os.sep+'ct_A_OBfov'+tag_files[1]+'_new.npy' 
    ct_dic_4,ct_dic_b4,A_dic,OB_fov_dic = np.load(ct_A_OBfov2_new_fl,allow_pickle=True)

    #print('here2')
    top_th_score_4=[]
    top_th_score_4_extended = []
    remember_ob={}

    nth=min_RNA_per_glom
    OBs_all = []
    area_obs={}
    if area_th>0:
        area_obs = pickle.load(open(r'\\meitner\f\Bogdan\glomeruli_alignment\slide_alignment\non_rigid_gaussian\area_obs.pkl','rb'))
    for OB_,(best_icds,scores_icds,ncts),(best_icds_e,scores_icds_e,ncts_e) in stats_intersections:
        if area_obs.get(OB_,1)>area_th:
            scores_icds[ncts<nth]=0
            keep = np.argsort(scores_icds)[::-1]
            best_icds,scores_icds,ncts = best_icds[keep],scores_icds[keep],ncts[keep]
            remember_ob[OB_] = [(np.round(scores_icds[:5],2),best_icds[:5],ncts[:5])]
            top_th_score_4.append([scores_icds[0],best_icds[0],ncts[0]])

            best_icds,scores_icds,ncts=best_icds_e,scores_icds_e,ncts_e
            scores_icds[ncts<nth]=0
            keep = np.argsort(scores_icds)[::-1]
            best_icds,scores_icds,ncts = best_icds[keep],scores_icds[keep],ncts[keep]

            top_th_score_4_extended.append([scores_icds[0],best_icds[0],ncts[0]])
            remember_ob[OB_]+=[(np.round(scores_icds[:5],2),best_icds[:5],ncts[:5])]
            OBs_all.append(OB_)
    #print('here3')
    th_score1=th_score1    
    th_score2,nth1_2,use_multi = th_score2s
    #th_score1=7.5
    #th_score2=12.5
    top_th_score_4,top_th_score_4_extended = np.array(top_th_score_4),np.array(top_th_score_4_extended)
    

    
    OB_dic_final = {}
    
    for (sc1,icd1,ncd1),(sc2,icd2,ncd2),OB_ in zip(top_th_score_4,top_th_score_4_extended,OBs_all):
        
        if sc1>th_score1:
            OB_dic_final[OB_]=(icd1,sc1,ncd1,1)
        elif (sc2>th_score2 and ct_dic_4[OB_][int(icd2)]>=nth1_2) and use_multi:
            OB_dic_final[OB_]=(icd2,sc2,ncd2,2)
    
    tag_analysis = os.path.basename(analysis_folder)
    libs = eval(tag_analysis.split('_lib')[1].split('_')[0])
    for OB_ in OB_dic_final:
        OB_dic_final[OB_] = list(OB_dic_final[OB_])+[libs]
    #dec_obj.OB_dic_final=OB_dic_final
    #print('here4')
    if save:
        pickle.dump(OB_dic_final,open(analysis_folder+os.sep+'FinalDecoding'+os.sep+'OB_dic_final.npy','wb'))
    #print('here5')
    if show_statistics:
        print(tag_analysis)
        OBs_decoded = list(OB_dic_final.keys())

        print("Number of glomeruli decoded:",len(OBs_decoded),"; Fraction decoded from total: ",np.round(1.*len(OBs_decoded)/len(OBs_all),2))
        As_decoded = [A_dic[OB_][-2] for OB_ in OBs_decoded]
        As_all = [A_dic[OB_][-2] for OB_ in OBs_all]
        Am,AM = np.percentile(As_decoded,5),np.percentile(As_decoded,95)
        #print(Am)
        print("Detection efficiency:",np.round(1.*len(OBs_decoded)/np.sum(np.array(As_all)>Am),2))

        cds_,ncds_ = np.unique([OB_dic_final[OB_][0] for OB_ in OB_dic_final],return_counts=True)
        print("Mean number of glomeruli per gene:",np.mean(ncds_))
        print("Number of genes decoded:",len(cds_))
        #np.array(list(zip(cds_[np.argsort(ncds_)],np.sort(ncds_))))
    return OB_dic_final
    


def get_limits_from_im_labf_old(im_labf):
    from scipy.ndimage import find_objects
    slices = find_objects(im_labf)
    islices,slices = zip(*[(isl+1,sl) for isl,sl in enumerate(slices) if sl is not None])
    limits = []
    for isl,sl in zip(islices,slices):
        xm,xM,ym,yM = sl[0].start,sl[0].stop,sl[1].start,sl[1].stop
        cm = np.mean(np.where(im_labf[xm:xM,ym:yM]==isl),-1)+[xm,ym]
        limits.append([xm,xM,ym,yM,cm,isl])
    limits = np.array(limits)
    return limits
def get_limits_from_im_labf(im_labf,area_th=1000):
    from scipy.ndimage import find_objects
    from scipy.spatial import ConvexHull
    slices = find_objects(im_labf)
    islices,slices = zip(*[(isl+1,sl) for isl,sl in enumerate(slices) if sl is not None])
    limits = []
    coords = []
    contours = []
    convex_hulls = []
    

    for isl,sl in zip(islices,slices):
        xm,xM,ym,yM = sl[0].start,sl[0].stop,sl[1].start,sl[1].stop
        im_ = im_labf[xm:xM,ym:yM]==isl
        coords_ = np.array(np.where(im_)).T+[xm,ym]
        area = len(coords_)
        if area>area_th:
            try:
                convex_hull = coords_[ConvexHull(coords_).vertices]
                contours_,hier = cv2.findContours(im_.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours_ = contours_[0][:,0,::-1]+[xm,ym]
            except:
                convex_hull = coords_
                contours_ = coords_
            
            cm = np.mean(coords_,0)
            
            coords.append(coords_)
            contours.append(contours_)
            convex_hulls.append(convex_hull)
            limits.append([xm,xM,ym,yM,cm,isl])
    limits = np.array(limits)
    return limits,coords,contours,convex_hulls
def check_glomeruli_decoding(dec_obj,icode=586,verbose=True,show_all=False,new_glom=False):
    if not hasattr(dec_obj,'contours'):
        if not new_glom:
            fl = dec_obj.analysis_folder+os.sep+'dapi_resc8_mask_manual.npy'
            if not os.path.exists(fl): fl = dec_obj.analysis_folder+os.sep+'dapi_resc8_glomLayer.tif.npy'
            dec_obj.im_labf,dec_obj.limits = np.load(fl,allow_pickle=True)
            
        else:
            fl = dec_obj.analysis_folder+os.sep+'dapi_resc8--imlabf_new.tif'
            dec_obj.im_labf = tifffile.imread(fl)
            dec_obj.limits = get_limits_from_im_labf(dec_obj.im_labf)
            dec_obj.iOBs = [lim[-1] for lim in dec_obj.limits]
        
        dec_obj.im_dapi = tifffile.imread(dec_obj.analysis_folder+os.sep+'dapi_resc8.tif')
        dec_obj.contours,hier = cv2.findContours(dec_obj.im_labf.astype(np.int32),cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_NONE)
    OB_dic_final = dec_obj.OB_dic_final
    OBs_t = [OB_ for OB_ in OB_dic_final if OB_dic_final[OB_][0]==icode]
    olfr_nm = dec_obj.code_dic[tuple(dec_obj.codes_valid[icode])]
    cd_ = tuple(dec_obj.codes_valid[icode])
    if verbose:
        print(([(OB_,OB_dic_final[OB_]) for OB_ in OBs_t],olfr_nm,cd_))
        
    fig = plt.figure(figsize=(6,6))
    im_dapi_ = dec_obj.im_dapi
    vmax = np.percentile(im_dapi_[im_dapi_!=0],99)
    plt.imshow(im_dapi_,vmax=vmax,cmap='gray')
    
    plt.title(olfr_nm.split('_')[0])
    for cont in dec_obj.contours:
        plt.plot(cont[:,0,0],cont[:,0,1],'r-',alpha=0.1)
    if not show_all:
        for OB_t in OBs_t:
            icd_=int(OB_dic_final[OB_t][0])
            olfr_nm = dec_obj.code_dic[tuple(dec_obj.codes_valid[icd_])]
            ilim = dec_obj.iOBs.index(OB_t) if new_glom else int(OB_t-1)
            lims = dec_obj.limits[ilim]
            if len(lims)>5: xm,xM,ym,yM,(cy,cx) = lims[:5]
            else: xm,xM,ym,yM,(cx,cy) = lims
            plt.plot([cx],[cy],'o',color=[0,1,0])
            plt.text(cx,cy,str(str(OB_t)),color='green',rotation=0)#olfr_nm.split('_')[0]+'_'+
    
    # plot all glomeruli
    else:
        for OB_t in OB_dic_final:
            icd_=int(OB_dic_final[OB_t][0])
            olfr_nm = dec_obj.code_dic[tuple(dec_obj.codes_valid[icd_])]
            ilim = dec_obj.iOBs.index(OB_t) if new_glom else int(OB_t-1)
            lims = dec_obj.limits[ilim]
            if len(lims)>5: xm,xM,ym,yM,(cy,cx) = lims[:5]
            else: xm,xM,ym,yM,(cx,cy) = lims
            plt.plot([cx],[cy],'o',color=[0,1,0])
            plt.text(cx,cy,str(olfr_nm.split('_')[0])+'\n'+str(int(OB_t))+'_'+str(icd_),color='red')#,rotation=90)
    return fig
def get_stats_ob(ndec_OB,A_dic,OB_ = 663,nth = 5,verbose=5,inorm=1,dic_glomeruli=None,use_neighbourhood=0):
    """
    dic_glomeruli is assumed to be of type:
    dic_glomeruli = {OB_:{'area':number,'coords':N x 2,'cm':2,'contour':N x 2,'convex_hull':N x 2,
                    'islice':slice_coordinate,'z-pos':z_coordinate(pixel_units)}}
    """
    
    OBs_compare = list(ndec_OB.keys())
    
    if dic_glomeruli is not None:
        for OB__ in A_dic:
            if OB__>0:
                A_dic[OB__][1] = dic_glomeruli[OB__]['area']
        
        if use_neighbourhood>0:
            islice_OB = dic_glomeruli[OB_]['islice']
            cm_OB = dic_glomeruli[OB_]['cm']
            cms = []
            OBs_compare_cands = []
            for OB__ in dic_glomeruli:
                if (dic_glomeruli[OB__]['islice']==islice_OB) and (dic_glomeruli[OB__]['area']>1000):
                    cms.append(dic_glomeruli[OB__]['cm'])
                    OBs_compare_cands.append(OB__)
            cms = np.array(cms)
            OBs_compare_cands = np.array(OBs_compare_cands)
            dists = np.linalg.norm(cms-cm_OB,axis=-1)
            OBs_compare = OBs_compare_cands[np.argsort(dists)[:use_neighbourhood]]

    
    
    Adec ={OB__:A_dic[OB__][inorm] for OB__ in A_dic}

    OBs_compare = [OB__ for OB__ in OBs_compare if OB__ in ndec_OB  and OB__ in A_dic and OB__>=1]
    #print(OBs_compare)
    ndec_OB[-1] = np.sum([ndec_OB[OB__] for OB__ in OBs_compare],0)
    Adec[-1] = np.sum([Adec.get(OB__,0) for OB__ in OBs_compare])
    #if OB_<10:print(len(OBs_compare))
    zero_cts = ndec_OB[-1]*0
    ob_cts = ndec_OB.get(OB_,zero_cts)
    observed = 1.*ob_cts/Adec[OB_]
    
    expected = 1.*(ndec_OB[-1]-ob_cts)/(Adec[-1]-Adec[OB_])
    #print("expected:",Adec[-1],ndec_OB[-1])
    obs_exp = observed/expected
    obs_exp[ob_cts<nth]=0
    obs_exp[ob_cts==0]=0
    order_ = np.argsort(obs_exp)[::-1]
    if verbose>0:
        print('score:',np.round(obs_exp[order_[:verbose]],2),
              'icode:',order_[:verbose],
              'npts:',ob_cts[order_][:verbose])
    return order_,obs_exp[order_],ob_cts[order_]
def dilate(img,dil=10):
    """dilates a binary image by dil pixels"""
    from cv2 import dilate
    #dilation kernel
    y,x = np.ogrid[-dil:dil, -dil:dil]
    kernel = (x*x + y*y < dil*dil).astype(np.uint8)
    return dilate(img.astype(np.uint8),kernel)
def check_glomeruli_figure(im_labf,im_msk):
    ncols=11
    vals = (np.arange(np.max(im_labf))%ncols+0.5)/float(ncols)
    cmap = plt.cm.colors.ListedColormap(plt.cm.Paired(vals))
    cmap.colors[0][:3]=0
    fig = plt.figure()
    plt.imshow(im_labf,cmap=cmap)
    plt.imshow(im_msk,alpha=0.25,cmap='gray')
    
    """
    im_ = tifffile.imread(fl)
    contours,hier = cv2.findContours(im_labf.astype(np.int32),cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_NONE) 
    plt.figure()
    plt.imshow(im_,vmax=10000)
    for cont in contours:
        plt.plot(cont[:,0,0],cont[:,0,1],'r-',alpha=0.1)
    plt.axis('equal')
    """
    
    return fig
def expand_glomeruli(labels,radius = 15,random_numbering=False):
    im_bw = (labels>0).astype(np.uint8)
    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(im_bw)
    from sklearn.neighbors import NearestNeighbors
    labels2 = np.zeros_like(labels)
    x_,y_ = (np.indices([2*radius+1]*2)-radius)/radius
    footprint = ((x_*x_+y_*y_)<1).astype(np.uint8)
    labs_ = np.arange(1,nb_components)
    if random_numbering:
        labs_ = np.random.permutation(labs_)
    limits = []
    for iob in range(1, nb_components):
        ym,xm,sy,sx,size =  stats[iob]
        rad = radius*2
        xm_,xM_ = xm-rad,(sx+xm)+rad
        ym_,yM_ = ym-rad,(sy+ym)+rad
        if xm_<0: xm_=0
        if ym_<0: ym_=0
        temp = labels[xm_:xM_,ym_:yM_]
        good = temp == iob
        bad = (temp != iob)&(temp>0)
        goodEx = cv2.morphologyEx(good.astype(np.uint8), cv2.MORPH_DILATE, footprint)
        Xi = np.array(np.where(good)).T
        Xo = np.array(np.where(bad)).T
        limits.append([xm_,xM_,ym_,yM_,centroids[iob]])
        if len(Xo)>0:
            nbrsi = NearestNeighbors(n_neighbors=1).fit(Xi)
            nbrso = NearestNeighbors(n_neighbors=1).fit(Xo)
            X = np.array(np.where(goodEx-good)).T
            distancesI, indices = nbrsi.kneighbors(X)
            distancesO, indices = nbrso.kneighbors(X)
            Xb = X[(distancesI>distancesO)[:,0]]
            goodEx[Xb[:,0],Xb[:,1]]=0
            good_ = goodEx>0
            labels2[xm_:xM_,ym_:yM_][good_] = labs_[iob-1]
    return labels2,limits
def label_glomeruli(im_msk,radius=5,min_size = 20,Am=200,fM=10,plt_val=False):
    if type(im_msk)==str:
        im_msk = cv2.imread(fl_mask)[:,:,0]
    x_,y_ = (np.indices([2*radius+1]*2)-radius)/radius
    footprint = ((x_*x_+y_*y_)<1).astype(np.uint8)
    im_msk_ = cv2.morphologyEx(im_msk, cv2.MORPH_CLOSE, footprint)
    im_msk_ = ((1-(im_msk>0))*255).astype(np.uint8)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(im_msk_)
    A = stats[:,-1]
    med_A = np.median(A[A>Am])
    AM = med_A*fM
    keep = (A>Am)&(A<AM)
    Ag = A[keep]
    if plt_val:
        print("Median number of glomeruli per bulb:",len(Ag)/6./2)
        print("Median diameter of glomeruli (um):",np.sqrt(np.median(Ag)/np.pi)*2*(0.108*8))
        plt.figure()
        plt.hist(np.sqrt(Ag/np.pi)*2*(0.108*8))
        plt.xlabel('Diameter (um)')
        plt.ylabel('Number of glomeruli')
    img = np.zeros(im_msk_.shape,dtype=np.uint32)
    nb_components = retval
    iobj = 0
    for i in range(1, nb_components):
        ym,xm,sy,sx,size =  stats[i]
        if keep[i]:
            iobj+=1
            good = labels[xm:(sx+xm),ym:(sy+ym)] == i
            img[xm:(sx+xm),ym:(sy+ym)][good] = iobj
    return img,Ag
def remove_small_components(img_,min_size = 20):
    img = img_.astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img)
    img2 = np.zeros((output.shape),dtype=bool)
    for i in range(1, nb_components):
        ym,xm,sy,sx,size =  stats[i]
        if size >= min_size:
            good = output[xm:(sx+xm),ym:(sy+ym)] == i
            img2[xm:(sx+xm),ym:(sy+ym)][good] = 1
    return img2
def perc_filter(im_,perc=5,resize=5,size=20):
    from scipy.ndimage import percentile_filter
    from scipy.ndimage import zoom
    x_,y_ = (np.indices([2*size+1]*2)-size)/size
    footprint = (x_*x_+y_*y_)<1
    im_filter = percentile_filter(im_[::resize,::resize], perc, footprint=footprint);
    x_int = np.linspace(0,im_filter.shape[0]-1,im_.shape[0]).astype(int)
    y_int = np.linspace(0,im_filter.shape[1]-1,im_.shape[1]).astype(int)
    im_filter_ = im_filter[x_int][:,y_int]
    return im_filter_
def resize(im,shape_ = [2048,2048]):
    x_int = np.round(np.linspace(0,im.shape[0]-1,shape_[0])).astype(int)
    y_int = np.round(np.linspace(0,im.shape[1]-1,shape_[1])).astype(int)
    return im[x_int][:,y_int]
def minmax(im,min_=None,max_=None,percmax=99.9,pecmin=0.1):
    im_ = np.array(im,dtype=float)
    if min_ is None:
        min_ = np.min(im_)
        if pecmin is not None:
            min_ = np.percentile(im_,pecmin)
    if max_ is None:
        max_ = np.max(im_)
        if percmax is not None:
            max_ = np.percentile(im_,percmax)
    #if (max_-min_)<=0:
    #    im_ = im_*0
    else:
        im_ = (im_-min_)/(max_-min_)
        im_[im_<0]=0
        im_[im_>1]=1
    return im_,min_,max_
from sklearn.cluster import DBSCAN
def get_tiles(im_3d,size=256):
    sz,sx,sy = im_3d.shape
    Mz = int(np.ceil(sz/float(size)))
    Mx = int(np.ceil(sx/float(size)))
    My = int(np.ceil(sy/float(size)))
    ims_dic = {}
    for iz in range(Mz):
        for ix in range(Mx):
            for iy in range(My):
                ims_dic[(iz,ix,iy)]=ims_dic.get((iz,ix,iy),[])+[im_3d[iz*size:(iz+1)*size,ix*size:(ix+1)*size,iy*size:(iy+1)*size]] 
    return ims_dic
from scipy.spatial.distance import cdist
def get_best_trans(Xh1,Xh2,th_h=1,th_dist = 2,return_pairs=False):
    mdelta = np.array([np.nan,np.nan,np.nan])
    if len(Xh1)==0 or len(Xh2)==0:
        if return_pairs:
            return mdelta,[],[]
        return mdelta
    X1,X2 = Xh1[:,:3],Xh2[:,:3]
    h1,h2 = Xh1[:,-1],Xh2[:,-1]
    i1 = np.where(h1>th_h)[0]
    i2 = np.where(h2>th_h)[0]
    if len(i1)==0 or len(i2)==0:
        if return_pairs:
            return mdelta,[],[]
        return mdelta
    i2_ = np.argmin(cdist(X1[i1],X2[i2]),axis=-1)
    i2 = i2[i2_]
    deltas = X1[i1]-X2[i2]
    dif_ = deltas
    bins = [np.arange(m,M+th_dist*2+1,th_dist*2) for m,M in zip(np.min(dif_,0),np.max(dif_,0))]
    hhist,bins_ = np.histogramdd(dif_,bins)
    max_i = np.unravel_index(np.argmax(hhist),hhist.shape)
    #plt.figure()
    #plt.imshow(np.max(hhist,0))
    center_ = [(bin_[iM_]+bin_[iM_+1])/2. for iM_,bin_ in zip(max_i,bins_)]
    keep = np.all(np.abs(dif_-center_)<=th_dist,-1)
    center_ = np.mean(dif_[keep],0)
    for i in range(5):
        keep = np.all(np.abs(dif_-center_)<=th_dist,-1)
        center_ = np.mean(dif_[keep],0)
    mdelta = center_
    keep = np.all(np.abs(deltas-mdelta)<=th_dist,1)
    if return_pairs:
        return mdelta,Xh1[i1[keep]],Xh2[i2[keep]]
    return mdelta
def get_uniform_points(im_raw,coords=None,sz_big = 21,sz_small=5,size=128,delta_fit=11,plt_val=False):
    """Normaly used on a dapi image to extract sub-pixel features. Returns Xh"""
    #normalize image
    
    im_raw = im_raw.astype(np.float32)
    sz=sz_big
    im_n = np.array([im_/cv2.GaussianBlur(im_,ksize= (sz*4+1,sz*4+1),sigmaX = sz,sigmaY = sz) for im_ in im_raw])
    sz=sz_small
    im_nn = np.array([cv2.GaussianBlur(im_,ksize= (sz*4+1,sz*4+1),sigmaX = sz,sigmaY = sz) for im_ in im_n])
    if coords is None:
        dic_ims = get_tiles(im_nn,size=size)
        coords = []
        for key in dic_ims:
            im_ = dic_ims[key][0]
            coords+=[np.unravel_index(np.argmax(im_),im_.shape)+np.array(key)*size]

    z,x,y = np.array(coords).T
    im_centers = [[],[],[],[]]
    zmax,xmax,ymax = im_nn.shape
    for d1 in range(-delta_fit,delta_fit+1):
        for d2 in range(-delta_fit,delta_fit+1):
            for d3 in range(-delta_fit,delta_fit+1):
                if (d1*d1+d2*d2+d3*d3)<=(delta_fit*delta_fit):
                    im_centers[0].append((z+d1))
                    im_centers[1].append((x+d2))
                    im_centers[2].append((y+d3))
                    im_centers[3].append(im_nn[(z+d1)%zmax,(x+d2)%xmax,(y+d3)%ymax])

    im_centers_ = np.array(im_centers)
    im_centers_[-1] -= np.min(im_centers_[-1],axis=0)
    zc = np.sum(im_centers_[0]*im_centers_[-1],axis=0)/np.sum(im_centers_[-1],axis=0)
    xc = np.sum(im_centers_[1]*im_centers_[-1],axis=0)/np.sum(im_centers_[-1],axis=0)
    yc = np.sum(im_centers_[2]*im_centers_[-1],axis=0)/np.sum(im_centers_[-1],axis=0)
    h = np.max(im_centers[-1],axis=0)
    Xh = np.array([zc,xc,yc,h]).T
    if plt_val:
        plt.figure()
        plt.imshow(np.max(im_nn,0),vmax=np.median(h)*1.5)
        plt.plot(yc,xc,'rx')
    return Xh
def get_local_max(im_dif,th_fit,delta=2,delta_fit=3,dbscan=True,return_centers=False,mins=None,psf_file=None):
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
    if dbscan and len(Xh)>0:
        from scipy import ndimage
        im_keep = np.zeros(im_dif.shape,dtype=bool)
        im_keep[z,x,y]=True
        lbl, nlbl = ndimage.label(im_keep,structure=np.ones([3]*3))
        l=lbl[z,x,y]#labels after reconnection
        ul = np.arange(1,nlbl+1)
        il = np.argsort(l)
        l=l[il]
        z,x,y,h = z[il],x[il],y[il],h[il]
        inds = np.searchsorted(l,ul)
        Xh = np.array([z,x,y,h]).T
        Xh_ = []
        for i_ in range(len(inds)):
            j_=inds[i_+1] if i_<len(inds)-1 else len(Xh)
            Xh_.append(np.mean(Xh[inds[i_]:j_],0))
        Xh=np.array(Xh_)
        z,x,y,h = Xh.T
    im_centers=[]
    if delta_fit!=0 and len(Xh)>0:
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
                        im_centers[3].append(im_dif[(z+d1)%zmax,(x+d2)%xmax,(y+d3)%ymax])

        im_centers_ = np.array(im_centers)
        im_centers_[-1] -= np.min(im_centers_[-1],axis=0)
        zc = np.sum(im_centers_[0]*im_centers_[-1],axis=0)/np.sum(im_centers_[-1],axis=0)
        xc = np.sum(im_centers_[1]*im_centers_[-1],axis=0)/np.sum(im_centers_[-1],axis=0)
        yc = np.sum(im_centers_[2]*im_centers_[-1],axis=0)/np.sum(im_centers_[-1],axis=0)
        Xh = np.array([zc,xc,yc,h]).T
        if psf_file is not None:
            if mins is None: mins = [0,0,0]#np.min([z,x,y],-1)
            im_psf = np.load(psf_file)
            pix = int(os.path.basename(psf_file).split('_pix')[-1].split('_')[0])
            coords = np.array([(z+mins[0])/pix,(x+mins[1])/pix,(y+mins[2])/pix],dtype=int).T
            im_psfs = np.array([im_psf[tuple(coord)] for coord in coords])
            
            hcent = np.array(im_centers_[-1],dtype=np.float32).T
            mid = int(hcent.shape[1]/2)
            hcent[:,mid]=np.nan
            hcent = (hcent-np.nanmean(hcent,axis=-1)[:,np.newaxis])/np.nanstd(hcent,axis=-1)[:,np.newaxis]
            scores = np.nanmean(hcent*im_psfs,-1)
            Xh = np.array([zc,xc,yc,h,scores]).T
    if return_centers:
        return Xh,np.array(im_centers)
    return Xh

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
def norm_sets(ims,norm_dapi=True,perc =80,nl = 30):
    normns = [1 for im_ in ims ]
    if norm_dapi and len(ims)>nl:
        normns = [np.percentile(im_,perc) for im_ in ims]        
        nll = int(nl/2)
        norms_ = list(normns[:nll][::-1])+list(normns)+list(normns[-nll:][::-1])
        norms_ = [np.median(norms_[il:il+nl]) for il in range(len(norms_)-nl)]
        normns = [e/norms_[0] for e in norms_]
    return normns
    
def get_new_name(dax_fl):
    dax_fl_ = dax_fl
    if not os.path.exists(dax_fl_):
        if dax_fl_.split('.')[-1]=='dax':
            dax_fl_ = dax_fl_.replace('.dax','.dax.zst')
        if dax_fl_.split('.')[-1]=='zst':
            dax_fl_ = dax_fl_.replace('.zst','')
    return dax_fl_
    


def load_im(dax_fl,custom_frms=None,dims=(2048,2048),cast=True):
    dax_fl_ = dax_fl
    if dax_fl_.split('.')[-1]=='npy':
        return np.load(dax_fl_)#np.swapaxes(np.load(dax_fl_),-1,-2)
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
def save_tile_image_and_label(fls_iter,save_file,resc=2,custom_frms=None,pix_size_=0.162,
                        max_impose=True,verbose=False,im_cor__=None,
                              rotation_angle=0,add_txt=True,norm_dapi=False,invertX=False,invertY=False):

        import tifffile
        ims,xys=[],[]
        pix_size=pix_size_*resc
        fls_iter_ = fls_iter
        if verbose: fls_iter_ = tqdm(fls_iter)
        for dax in fls_iter_:
            
            xml_fl = os.path.dirname(dax)+os.sep+os.path.basename(dax).split('_fr')[0].split('.dax')[0]+'.xml'
            pos = np.array([ln.split('>')[1].split('<')[0].split(',') for ln in open(xml_fl,'r') if 'stage_position' in ln][0],dtype=np.float32)
            dapi_im = load_im(dax,custom_frms=custom_frms)
            
            dapi_im_small  = np.array(dapi_im)[:,1:-1:resc,1:-1:resc]
            
            #Illumination correction:
            dapi_im_small = np.max(dapi_im_small,0)
            if '.zst' in dax: dapi_im_small = dapi_im_small.T
            #Consider adding illumination correction and better stitching
            xys.append(pos/pix_size)
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
        tifffile.imsave(save_file,np.clip(im_base,0,2**16-1).astype(np.uint16))
def clean_within(res_org_):
    res_org_F = []

    for es in res_org_:
        res_org_F_ =[]
        visited = set()
        for e in es:
            visited_ = [e_ in visited for e_ in e]
            if not np.any(visited_):
                visited.update(e)
                res_org_F_.append(e)
        res_org_F.append(res_org_F_)
    return res_org_F
def clean_res_org_3(res_org,ires):
    if ires!=2:
        res_org_3 = []
        for es_4,es_m in zip(res_org[2],res_org[ires]):
            u_ = np.unique(es_4)
            res_org_3.append([e_ for e_ in es_m if not np.any(np.isin(e_,u_))])
        res_org[ires] = res_org_3
    return res_org

def dax_mmap_colors(dax_fl, cols = ['561','647','405']):
    num_col = len(cols)
    dax_fl = get_new_name(dax_fl)
    #if dax_fl.split('.')[-1]=='dax':
        #im_full = io.DaxReader(dax_fl).loadMap()
        #ims = [im_full[icol::num_col] for icol in range(num_col)]
    #else:
        #ims = [(dax_fl,icol,num_col) for icol in range(num_col)]    
    ims = [(dax_fl,icol,num_col) for icol in range(num_col)]
    base_name = os.path.basename(os.path.dirname(dax_fl))
    ims_names = [base_name+'_'+col for col in cols]
    return ims,ims_names
def load_im_col(dax_fl_icol_num_col,cast=True):
    if len(dax_fl_icol_num_col)>3:
        return dax_fl_icol_num_col
    dax_fl,icol,num_col = dax_fl_icol_num_col
    return load_im(dax_fl,cast=cast)[icol::num_col]
def load_im_col_single(dax_fl_icol_num_col):
    if len(dax_fl_icol_num_col)>3:
        im_ = dax_fl_icol_num_col
        return im_[int(len(im_)/2)]
    dax_fl,icol,num_col = dax_fl_icol_num_col
    dax_fl = get_new_name(dax_fl)
    if dax_fl.split('.')[-1]=='zst':
        npy_fls = np.sort(glob.glob(dax_fl.split('.dax')[0]+'_fr*'))
        return load_im(npy_fls[icol])
    else:
        im_ = load_im_col(dax_fl_icol_num_col,cast=False)
        return np.array(im_[int(len(im_)/2)])
    
def str_to_int(nm,bad_ind=1000):
    try:
        return int(nm)
    except:
        return bad_ind
def nm_to_R(nm_sm,cols = ['561','647','750'],bad_ind=1000):
    if len(cols)==2:
        try:
            col = nm_sm.split('_')[-1]
            coli = cols.index(col)
            return 'R'+nm_sm[3:].split('_')[0].split(';')[0].split('T')[0].split(',')[coli]+'_'+col
        except:
            return nm_sm
    else:
        try:
            if 'cfos,r15' in nm_sm.lower() or 'r14,b' in nm_sm.lower():
                R_to_col={1000:'750',1:'750',2:'647',3:'561',4:'750',6:'647',5:'561',7:'750',8:'647',9:'561',10:'750',12:'647',11:'561',14:'647',15:'647',13:'561',16:'750',17:'561',18:'750'}
            else:
                R_to_col={1000:'750',1:'750',2:'647',3:'561',4:'750',6:'647',5:'561',7:'750',8:'647',9:'561',10:'750',12:'647',11:'561',14:'750',15:'647',13:'561',16:'750',17:'561',18:'561'}
            
            nmT = nm_sm.lower().replace('egr1,cfos','r16,17,18').replace('cfos,egr1','r16,17,18')
            nmT = nmT.replace('cfos,r15,egr1','r16,15,17')
            nmT = nmT.replace('r14,b,13','r14,18,13')
            col = nm_sm.split('_')[-1]
            iRs = [str_to_int(s_,bad_ind) for s_ in nmT[3:].split('_')[0].split(';')[0].split(',')]
            cols_ = [R_to_col.get(iR) for iR in iRs]
            iR = str(iRs[cols_.index(col)]) if col in cols_ else str(bad_ind)
            return 'R'+iR
        except:
            return nm_sm


def get_H(folder):
    htag = os.path.basename(folder)[1:]
    i=0
    try:
        while True:
            if htag[i].isdigit() is False: break
            i+=1
        return int(htag[:i])
    except:
        return 0

    

def htag_to_Rs(htag): return htag.split('R')[-1].split(';')[0].split(',')

def norm_im(im,sz=55):
    im_= np.array([im__-cv2.blur(im__,(sz,sz))for im__ in im])
    im_ = np.array([im__/cv2.blur(np.abs(im__),(sz,sz))for im__ in im_])
    return im_


class decoded_fov:
    def __init__(self,data_folder,analysis_folder=None):
        self.data_folder=data_folder
        #self.Hfolders = np.sort(glob.glob(self.data_folder+os.sep+'H*R*'))
        #self.fovs = [os.path.basename(fl) for fl in np.sort(glob.glob(self.Hfolders[0]+os.sep+'*.dax'))]
        if type(self.data_folder) is not list: self.data_folder= [self.data_folder] 
        if analysis_folder is None:
            self.analysis_folder = self.data_folder[0]+'-AnalysisOB'
        else:
            self.analysis_folder = analysis_folder+os.sep+os.path.basename(self.data_folder[0])+'-AnalysisOB'
        if not os.path.exists(self.analysis_folder): os.makedirs(self.analysis_folder)
        
    def get_files(self,htag=r'\H*R*',force=False):
        files_file = self.analysis_folder+os.sep+'files.npy'
        if os.path.exists(files_file) and not force:
            self.Hfolders,self.fovs,self.htags = np.load(files_file,allow_pickle=True)
        else:
            folders = [fld for data_folder in self.data_folder for fld in glob.glob(data_folder+htag) 
               if os.path.isdir(fld) if len(glob.glob(fld+os.sep+'*.dax*'))>0 if 'test' not in fld]
            self.Hfolders = np.array(folders)[np.argsort([get_H(folder) for folder in folders])]
            self.fovs = np.sort([os.path.basename(fov) for fov in glob.glob(folders[0]+os.sep+'*.dax*')])
            self.htags = [os.path.basename(fld) for fld in self.Hfolders]
            np.save(files_file,[self.Hfolders,self.fovs,self.htags])
    def is_good(self,ifov):
        self.fov_index=ifov
        self.fov = self.fovs[self.fov_index]
        for folder in self.Hfolders:
            dax_fl = folder+os.sep+self.fov
            dax_fl = dax_fl.split('.dax')[0]+'.dax'
            zst_fl = dax_fl+'.zst'
            if not os.path.exists(dax_fl) and not os.path.exists(zst_fl):
                return False
        return True
    def is_complete(self,ifov):
        self.fov_index=ifov
        self.fov = self.fovs[self.fov_index]
        self.final_fits_file = self.analysis_folder+os.sep+'Fits'+os.sep+'final_fits__'+self.fov.replace('.dax','')+'.pkl'
        self.final_file = self.final_fits_file.replace(os.sep+'Fits'+os.sep,os.sep+'Decoded'+os.sep).replace('.pkl','_AOBf.npy')
        return os.path.exists(self.final_file)
    def get_incomplete(self):
        ifovs = []
        for ifov in range(len(self.fovs)):
            fov = self.fovs[ifov]
            if not self.is_complete(ifov):
                ifovs.append(ifov)
        return ifovs
        
    
    def get_flatfield(self,force=False,cols = ['561','647','405'],ref=0):
        self.flat_field_file = self.analysis_folder+os.sep+'median_signals.npy'
        if not os.path.exists(self.flat_field_file) or force:
            ims_f = []
            for fov in tqdm(self.fovs): 
                dax_fl = self.Hfolders[ref]+os.sep+fov
                if os.path.exists(dax_fl):
                    ims,ims_names = dax_mmap_colors(dax_fl,cols=cols)
                    if len(ims[0])==3:
                        npy_fls = np.sort(glob.glob(dax_fl.split('.dax')[0]+'_fr*'))
                        ims_f.append(np.array([load_im(npy_fl) for npy_fl in npy_fls],dtype=np.float32))
                    else:
                        ims_f.append(np.array([im_[0] for im_ in ims],dtype=np.float32))
            ims_mean = np.median(ims_f,axis=0)
            self.ims_mean = ims_mean
            np.save(self.flat_field_file,ims_mean)
        else:
            self.ims_mean = np.load(self.flat_field_file)
    def get_mapped_ims(self,index,cols = ['561','647','750','405']):
        self.fov_index = index
        #set fov
        self.fov = self.fovs[self.fov_index]
        fov = self.fov
        ims,ims_names=[],[]
        im_dapis = []
        self.cols = cols
        self.ncols = len(cols)
        for folder in self.Hfolders:
            dax_fl = folder+os.sep+fov
            #if os.path.exists(dax_fl):
            ims_,ims_names_ = dax_mmap_colors(dax_fl, cols = cols)
            ims.extend(ims_[:-1])
            ims_names.extend(ims_names_[:-1])
            im_dapi = ims_[-1]#[int(len(ims_[-1])/2),:,:]
            im_dapis.append(im_dapi)
        self.ims_dapi = im_dapis
        self.ims = ims
        self.ims_names = ims_names
        
    def fit(self,iim,th_fit=4,th_cor=0.35,maxfev=10,better_fit=True,plt_val=False):
        self.iim =iim
        im = np.array(load_im_col(self.ims[self.iim]),dtype=np.float32)
        self.icol = self.iim%(self.ncols-1)
        if hasattr(self,'ims_mean'):
            im_mean = self.ims_mean[self.icol]
            im_norm = im/im_mean
            im_norm = np.array([im_/np.median(im_) for im_ in im_norm],dtype=np.float32)
        else:
            im_norm = np.array([im_/cv2.blur(im_,(256,256)) for im_ in im],dtype=np.float32)
        
        
        im_norm2 = norm_im(im_norm,sz=25)
        self.im_norm = im_norm
        self.im_norm2 = im_norm2
        self.zxyh=[]
        self.zxyh_all=[]
        self.zxyh,centers = get_local_max(im_norm2,th_fit,delta=2,delta_fit=3,dbscan=True,return_centers=True,mins=None,
                                        psf_file = self.psf_file)
        
        
        if len(self.zxyh)>0:
            z,x,y,h,cor = self.zxyh.T
            h0 = im_norm[z.astype(int)%im_norm.shape[0],x.astype(int)%im_norm.shape[1],y.astype(int)%im_norm.shape[2]]
            self.zxyh_all = np.array([z,x,y,h,h0,cor],dtype=np.float32).T
            keep = cor>th_cor
            self.zxyh = self.zxyh_all[keep]
            self.centers = np.swapaxes(centers[:,:,keep],0,-1)
            if better_fit:
                #pfits = ft.fast_fit_big_image(self.im_norm,self.zxyh[:,:3],
                #                             avoid_neigbors=False,radius_fit=3,better_fit=True,verbose=False)
                pfits = []
                for centers_ in tqdm(self.centers):
                    h_ = centers_[:,-1]
                    X_ = centers_[:,:3].T
                    fobj = ft.GaussianFit(h_,X_,delta_center=2.5)
                    fobj.fit(maxfev=maxfev)
                    pfits.append(fobj.p)
                pfits = np.array(pfits)
                self.zxyh[:,:3]=pfits[:,1:4]
        
        if plt_val:
            plt.figure(figsize=(10,10))
            plt.imshow(np.max(im_norm2,0),vmax=10)#,vmax=3)
            plt.plot(self.zxyh[:,2],self.zxyh[:,1],'rx')

        #return self.zxyh,self.zxyh_all
    def get_drift_3d(self,ref=1,sz_big = 21,sz_small=5,size=128,delta_fit=11,th_h=1.5,th_dist=2):
    
        self.npy_drift = self.analysis_folder+os.sep+'Drfits'+os.sep+self.fov.split('.dax')[0]+'_drift.npy'
        if os.path.exists(self.npy_drift):
            self.Ds,self.DEs,self.Tzxys=np.load(self.npy_drift,allow_pickle=True)
            
            return None
        ims_align = self.ims_dapi
        #Xhs = [get_uniform_points(im_,sz_big = sz_big,sz_small=sz_small,size=size,delta_fit=delta_fit,plt_val=False)
        #  for im_ in tqdm(ims_align)]
        #self.Xhs_drift = Xhs
        ##get pairing and computer drift
        
        ims_align_ref = load_im_col(ims_align[ref])
        
        Xh2 = get_uniform_points(ims_align_ref,sz_big = sz_big,sz_small=sz_small,size=size,delta_fit=delta_fit,plt_val=False)#Xhs[ref]
        Ds,DEs = [],[]
        for iIm,im_ in enumerate(ims_align):
            im_ = load_im_col(im_)
            Xh1 = get_uniform_points(im_,sz_big = sz_big,sz_small=sz_small,size=size,delta_fit=delta_fit,plt_val=False)
            dtf = get_best_trans(Xh1,Xh2,th_h=th_h,th_dist=th_dist)
            if np.max(np.abs(dtf))>5:
                coords = (Xh2[:,:3]+dtf).astype(int)
                Xh1 = get_uniform_points(im_,coords,sz_big = sz_big,sz_small=sz_small,
                                        size=size,delta_fit=delta_fit,plt_val=False)
                dtf = get_best_trans(Xh1,Xh2,th_h=th_h,th_dist=th_dist)
            if np.isnan(dtf[0]):
                if not hasattr(self,'Txys_2d'): self.get_drift_2d(nsq=4, snorm=40)
                dtf = [0]+list(-self.Txys_2d[iIm])
                print(dtf)
                coords = (Xh2[:,:3]+dtf).astype(int)
                Xh1 = get_uniform_points(im_,coords,sz_big = sz_big,sz_small=sz_small,
                                        size=size,delta_fit=delta_fit,plt_val=True)
                dtf = get_best_trans(Xh1,Xh2,th_h=th_h,th_dist=th_dist)
            print(dtf)
            dt1 = get_best_trans(Xh1[0::2],Xh2[0::2],th_h=th_h,th_dist=th_dist)
            dt2 = get_best_trans(Xh1[1::2],Xh2[1::2],th_h=th_h,th_dist=th_dist)
            #dt1-dt2
            Ds.append(dtf)
            DEs.append(dt1-dt2)
        self.Ds=Ds
        self.DEs=DEs
        self.Tzxys = -np.array([tzxy for tzxy in self.Ds for icol in range(self.ncols-1)])
        if not os.path.exists(os.path.dirname(self.npy_drift)): os.makedirs(os.path.dirname(self.npy_drift))
        np.save(self.npy_drift,[self.Ds,self.DEs,self.Tzxys])
    def get_drift_2d(self,nsq = 4,snorm = 40):
        def norm_dapi(im_dapi):
            im_dapi_ = im_dapi.astype(np.float32)
            im_dapi_ = im_dapi_/cv2.blur(im_dapi_,(snorm,snorm))-1
            return im_dapi_
        self.ims_dapi_norm = [norm_dapi(load_im_col_single(im_)) for im_ in self.ims_dapi]    


        # There will be nsq*nsq tiles of images
        delta = int(2048/nsq)
        centers_ref = np.array([[int(delta/2)+delta*i,int(delta/2)+delta*j] for i in range(nsq) for j in range(nsq)])
        sz =int(delta/2)

        im_ref = self.ims_dapi_norm[1]
        Txys = []
        for iset in range(len(self.ims_dapi_norm)):
            im_t = self.ims_dapi_norm[iset]
            txys = [ft.fftalign_2d(im_ref[xc:xc+sz,yc:yc+sz], im_t[xc:xc+sz,yc:yc+sz],max_disp=sz) for (xc,yc) in centers_ref]
            #print(np.median(txys,axis=0),txys)
            Txys.append(np.median(txys,axis=0))
        self.Txys_2d = Txys
        self.Txys = np.array([txy for txy in Txys for icol in range(self.ncols-1)],dtype=int)
        self.txys = txys
        
    def get_points_codes(self,cor_th=0.45,dinstance_th = 1.5,nRs=15):
        """
        Given that .get_code is used, this goes through zxyhsf and looks for perfect intersections.
        It screens for cor_th first and then  for distance_th.
        """
        Xsf,Rs=[],[]

        for iR,zxy_ in enumerate(self.zxyhsf[:nRs]):
            Xsf.extend(zxy_)
            Rs.extend([iR]*len(zxy_))
        Rs=np.array(Rs)
        Xsf = np.concatenate([Xsf,Rs[:,np.newaxis]],axis=-1)
        
        #cutoff brightness/correlation
        cors = Xsf[:,5]
        keep = cors>cor_th
        Xsf = Xsf[keep]
        Rs = Rs[keep]
        self.Xsf=Xsf
        # Find all intersections
        Xs = Xsf[:,:3]
        Ts = cKDTree(Xs)
        res = Ts.query_ball_tree(Ts,dinstance_th)
        self.res=res
        #get perfect matches
        rints = np.array([r for r in res if len(r)==4])
        rints = np.array([np.array(eval(e)) for e in np.unique([str(list(np.sort(r))) for r in rints])]) # make unique
        rints_txt = np.array([str(Rs[r]) for r in rints])
        codes_txt = np.array([str(cd) for cd in np.array(self.codes_valid)])
        keeps = np.array([rints_txt==cd for cd in codes_txt])
        points_all_codes = [[Xsf[r] for r in rints[keep]] for keep in keeps]
        self.points_all_codes = points_all_codes
        #lens_= [len(pts) for pts in points_all_codes]
        #points_all_codes_ = np.array(points_all_codes)[np.argsort(lens_)[::-1]]
    def check_fits(self,pt,szxy=[4,5,5],normalize=False,proj_axis=0,plt_val=True,use_raw=False):
        ims_sm = []
        ims_names = self.ims_names
        for iim,im_ in enumerate(self.ims):
            if not hasattr(self,'Tzxys') or use_raw:
                xd,yd = self.Txys[iim]; zd = 0
            else:
                zd,xd,yd = self.Tzxys[iim]
                zd,xd,yd = int(np.round(zd)),int(np.round(xd)),int(np.round(yd))
            z_min,z_max = int(pt[0]-szxy[0]-zd),int(pt[0]+szxy[0]-zd)
            x_min,x_max = int(pt[1]-szxy[1]-xd),int(pt[1]+szxy[1]-xd)
            y_min,y_max = int(pt[2]-szxy[2]-yd),int(pt[2]+szxy[2]-yd)
            im_sm = im_[z_min:z_max,x_min:x_max,y_min:y_max]
            ims_sm.append(im_sm)
        ims_sm = [im_.astype(np.float32)for im_ in ims_sm]
        if normalize:
            sz=55
            ims_sm = [np.array([im__-cv2.blur(im__,(sz,sz))for im__ in im_]) for im_ in ims_sm]
            #ims_sm = [im_/np.std(im_) for im_ in ims_sm]
            ims_sm = [np.array([im__/cv2.blur(np.abs(im__),(sz,sz))for im__ in im_]) for im_ in ims_sm]

        self.ims_sm = ims_sm

        if plt_val:
            #construct figure

            iimax = len(ims_sm)
            col_row = 1.25
            nrow = self.ncols-1#int(np.sqrt(iimax)/col_row)
            ncol = int(np.ceil(iimax/float(nrow)))
            #nrow,ncol=ncol,nrow
            ssz,ssx,ssy = ims_sm[0].shape
            ssx,ssy = np.array([ssz,ssx,ssy])[np.setdiff1d([0,1,2],[proj_axis])]
            imf = np.zeros([ssx*nrow,ssy*ncol])
            iim=0
            txts=[]
            for icol in range(ncol):
                for irow in range(nrow):
                    if iim<len(ims_sm):
                        ims_sm_ = ims_sm[iim]
                        nm_sm = ims_names[iim]

                        txt = nm_to_R(nm_sm)


                        if 'dapi' in txt: ims_sm_=[ims_sm_[int(len(ims_sm_)/2)]]

                        im_plt,min_,max_ = minmax(np.max(ims_sm_,proj_axis),percmax=99.95,pecmin=.5)
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
    def get_fine_drift_postfit(self,icol=0,npts =200,dist=3.5):
        ncols = self.ncols-1
        nims = len(self.zxyhs)
        nsets = int(nims/ncols)
        isets = np.arange(icol,nims,ncols)
        Tzxys12 = []
        for iR in [icol,isets[int(len(isets)/2)]]:
            Xi = self.zxyhs[iR]
            Ti = cKDTree(Xi[:,:3]+([0]+list(self.Txys[iR])))
            Tzxys = []
            for jR in isets:
                Xj = self.zxyhs[jR]
                Tj = cKDTree(Xj[:,:3]+([0]+list(self.Txys[jR])))
                res = Ti.query_ball_tree(Tj,dist,p=1.)
                Ii,Ij = np.array([(ir,res_[0]) for ir,res_ in enumerate(res) if len(res_)>0]).T
                isort = np.argsort(Xi[Ii][:,3])[::-1][:npts]
                difs = Xi[Ii][isort,:3]-Xj[Ij][isort,:3]
                Tzxys.append(np.median(difs,0))
            Tzxys = Tzxys-Tzxys[0]
            Tzxys12.append(Tzxys)

        self.Tzxys = [np.mean(Tzxys12,axis=0)[iel] for iel in range(len(Tzxys)) for _ in range(ncols)]
        self.error_drift = Tzxys12[0]-Tzxys12[1]
    def get_fine_drift(self,icol=3,npts=50,szxy=[4,7,7],th=0.5):
        ncol = self.ncols-1
        ps_all = []
        nsets = int(len(self.ims)/ncol)
        
        if hasattr(self,'zxyhs'):
            zxyh_ = self.zxyhs[icol]
        else:
            zxyh_ = self.fit(icol,plt_val=False)
            
        for ipt in range(1,npts):
            try:

                pt = zxyh_[:,:3][np.argsort(zxyh_[:,-1])[-ipt]]

                self.check_fits(pt,szxy=szxy,plt_val=False,use_raw=True)
                ps = []
                for iset in range(nsets):
                    im_ = np.array(self.ims_sm[icol+ncol*iset])
                    im_ = im_/np.median(im_)
                    X_ = np.indices(im_.shape).reshape([3,-1])
                    fobj = ft.GaussianFit(im_[X_[0],X_[1],X_[2]],X_,delta_center=2.5)
                    fobj.fit()
                    ps.append(fobj.p[:4])
                ps_all.append(ps)
            except:
                pass

        ps_all = np.array(ps_all)
        self.ps_all = ps_all
        bad = ps_all[:,:,0]<th
        X_ = ps_all[:,:,1:]
        X_[bad]=np.nan
        drift = X_-X_[:,[0],:]
        drift_fine = np.nanmedian(drift[::],axis=0)
        error_drift = np.nanmedian(drift[1::2],axis=0)-np.nanmedian(drift[0::2],axis=0)
        self.drift_fine = [drft for drft in drift_fine for icol in range(ncol)]
        self.error_drift = [drft for drft in error_drift for icol in range(ncol)]
        self.Tzxys = []
        for iim in range(len(self.Txys)):
            xd,yd = self.Txys[iim]; zd = 0
            zdf,xdf,ydf = -self.drift_fine[iim][0],-self.drift_fine[iim][1]+xd,-self.drift_fine[iim][2]+yd
            self.Tzxys.append([zdf,xdf,ydf])
    
        self.Tzxys = np.array(self.Tzxys)
        self.Tzxys_pix = np.round(self.Tzxys).astype(int)
    def fitall(self,th_fit=4,th_cor=0.35,maxfev=10,overwrite=False,better_fit=True):
        self.zxyhs = []
        self.zxyhs_all = []
        fits_folder = self.analysis_folder+os.sep+'Fits'
        if not os.path.exists(fits_folder): os.makedirs(fits_folder)
        self.fits_files = []
        for iim in tqdm(np.arange(len(self.ims))):
            fit_file = fits_folder+os.sep+'Fit__ifov'+str(self.fov_index)+'__iim'+str(iim)+'.npy'
            fit_file_centers = fits_folder+os.sep+'Fit__ifov'+str(self.fov_index)+'__iim'+str(iim)+'_centers.npy'
            if os.path.exists(fit_file) and (not overwrite):
                self.zxyh,self.zxyh_all = np.load(fit_file,allow_pickle=True)
            else:
                self.fit(iim,th_fit=th_fit,th_cor=th_cor,maxfev=maxfev,better_fit=better_fit,plt_val=False)
                np.save(fit_file,[self.zxyh,self.zxyh_all])
                #if not better_fit: np.save(fit_file_centers,self.centers)
            self.fits_files.append(fit_file)
            self.zxyhs.append(self.zxyh)
            self.zxyhs_all.append(self.zxyh_all)
    def get_final_zxyhs(self,sort=True,overwrite=False):
        
        fits_folder = self.analysis_folder+os.sep+'Fits'
        if not os.path.exists(fits_folder): os.makedirs(fits_folder)
            
        self.final_fits_file = fits_folder+os.sep+'final_fits__'+self.fov.replace('.dax','')+'.pkl'
        
        dic_chr = pickle.load(open(self.chrom_file,'rb'))
        if not os.path.exists(self.final_fits_file) or overwrite:
            self.zxyhsf = []
            self.readout_names = []
            #get readout names and sort
            rnms = [nm_to_R(nm_sm,cols = self.cols[:-1])for nm_sm in self.ims_names]
            if sort:
                readouts_int = [int(nm_sm[1:].split('_')[0]) for nm_sm in rnms]
                _,readouts = np.unique(readouts_int[::-1],return_index=True)
                readouts = len(readouts_int)-readouts-1
                self.readouts = readouts#np.argsort([int(nm_sm[1:].split('_')[0]) for nm_sm in rnms])
            else: 
                self.readouts = range(len(rnms))
            ims_names =[] 
            for iim in self.readouts:
                self.readout_names.append(rnms[iim])
                col = self.ims_names[iim].split('_')[-1]
                
                zxyh_ = self.zxyhs[iim].copy()
                mcol = dic_chr.get('m_'+col+'_647',None)
                zxyh_[:,:3] = ft.apply_colorcor(zxyh_[:,:3],mcol)
                zxyh_[:,:3]+=self.Tzxys[iim]
                self.zxyhsf.append(zxyh_)
                ims_names.append(self.ims_names[iim])
            dic_save = {'zxyhsf':self.zxyhsf,
                    'Tzxys':self.Tzxys,'error_drift':self.DEs,
                   'readout_names':self.readout_names,
                       'image_names':ims_names}
            pickle.dump(dic_save,open(self.final_fits_file,'wb'))
        else:
            dic_save = pickle.load(open(self.final_fits_file,'rb'))
            for key in dic_save:
                setattr(self,key,dic_save[key])
    def get_Xsf(self,load=True):
        if load:
            ###Load data from file
            fits_folder = self.analysis_folder+os.sep+'Fits'
            self.final_fits_file = fits_folder+os.sep+'final_fits__'+self.fov.replace('.dax','')+'.pkl'
            assert(os.path.exists(self.final_fits_file))
            dic_save = pickle.load(open(self.final_fits_file,'rb'))
            for key in dic_save:
                setattr(self,key,dic_save[key])
        Xsf,Rs=[],[]
        for iR,zxy_ in enumerate(self.zxyhsf[:]):
            Xsf.extend(zxy_)
            Rs.extend([iR]*len(zxy_))
        Rs=np.array(Rs)
        self.Xsf = np.concatenate([Xsf,Rs[:,np.newaxis]],axis=-1)
    def get_code(self,fl = r'SIORN1.fasta'):
        if 'ORN' in os.path.abspath(fl):
            lines = [ln for ln in open(fl,'r') if '_indexAB:' in ln and 'newORLib_32bit' in ln ]
            code_dic = {}
            for ln in lines:
                lib_id = int("['fwd_W1B11', 'rev_W1B12']" not in ln)#int(eval(ln.split('_indexAB:')[-1].split('_')[0])[0]/2)
                #if lib_id<=10:
                code = [cd+1 for cd in eval(ln.split('_code:')[-1].split('_')[0])]
                olfr = ln[1:].split('_')[0]
                code_dic[str(code)]=olfr+'_'+str(lib_id)
                #all_codes.append(code)
            codes = np.array(list(map(eval,code_dic.keys())))
        else:
            lines = [ln for ln in open(fl,'r')][::2]
            code_dic = {tuple(eval(ln.split('_code(Steven):')[-1].split('_')[0])):ln.split('_')[0][1:] 
                        for ln in lines if '_code(Steven):' in ln}
            codes = np.array([ln for ln in code_dic.keys()])
            
            
            
            lines = [ln for ln in open(fl,'r') if '_indexAB:' in ln and '_code(Steven):' in ln]
            code_dic = {}
            for ln in lines:
                lib_id = int(eval(ln.split('_indexAB:')[-1].split('_')[0])[0]/2)
                if lib_id<=10:
                    code = [cd for cd in eval(ln.split('_code(Steven):')[-1].split('_')[0])]
                    olfr = ln[1:].split('_')[0]
                    code_dic[tuple(code)]=olfr+'_'+str(lib_id)
                    
            lines_extra_lib = [ln for ln in open(fl,'r') if '_indexAB:[42, 41]'in ln]
            for ln in lines_extra_lib:
                lib_id = 11
                code = [cd for cd in eval(ln.split('_code:')[-1].split('_')[0])]
                olfr = ln[1:].split('_')[0]
                code_dic[tuple(code)]=olfr+'_'+str(lib_id)
            codes = np.array([ln for ln in code_dic.keys()])
            
        self.code_dic =code_dic
        self.codes =codes
        self.libs = eval(os.path.basename(self.data_folder[0]).split('_lib')[-1].split('_')[0])
        self.codes_valid = np.array([cd for cd in self.code_dic if int(self.code_dic[cd].split('_')[-1]) in self.libs])
    
    def get_chromatic_pairing(self,pairs = [[4,10,9],[6,11,10],[7,9,6]],th=7.5,plt_val = True):
        dic_chr = {}
        chrom_save_fl = self.analysis_folder+os.sep+'final__'+self.fov.replace('.dax','')+'__chromAB.pkl'
        if os.path.exists(chrom_save_fl):
            self.dic_chr = pickle.load(open(chrom_save_fl,'rb'))
        if not os.path.exists(chrom_save_fl):
            for ip,jp,ibad in pairs:
                zi,xi,yi,hi,h0i = self.zxyhsf[ip].T
                zj,xj,yj,hj,h0j = self.zxyhsf[jp].T
                zbad,xbad,ybad,hbad,h0bad = self.zxyhsf[ibad].T
                keepi=hi>th
                keepj=hj>th


                Xi = np.array([zi[keepi],xi[keepi],yi[keepi]]).T
                Xj = np.array([zj[keepj],xj[keepj],yj[keepj]]).T
                Xbad = np.array([zbad,xbad,ybad]).T
                Ti = cKDTree(Xi*[3,1,1])
                Tj = cKDTree(Xj*[3,1,1])
                Tbad = cKDTree(Xbad*[3,1,1])
                res = Ti.query_ball_tree(Tj,6.,p=1.)
                res_bad = Ti.query_ball_tree(Tbad,6.,p=1.)
                Ii,Ij = np.array([[ei,ejs[0]] for ei,(ejs,ebad) in enumerate(zip(res,res_bad)) if (len(ejs)==1)and(len(ebad)==0)]).T

                #plt.figure();plt.plot(Xi[Ii,1],Xi[Ii,2],'x');plt.plot(Xj[Ij,1],Xj[Ij,2],'x')
                im_ = np.array(self.ims[ip],dtype=np.float32)
                psi = ft.fast_fit_big_image(im_,Xi[Ii]-self.Tzxys[ip],better_fit=True)
                im_ = np.array(self.ims[jp],dtype=np.float32)
                psj = ft.fast_fit_big_image(im_,Xj[Ij]-self.Tzxys[jp],better_fit=True)
                Xif = psi[:,1:4]+self.Tzxys[ip]
                Xjf = psj[:,1:4]+self.Tzxys[jp]
                hXif = np.concatenate([Xif,hi[keepi][Ii][:,np.newaxis],h0i[keepi][Ii][:,np.newaxis]],axis=-1)
                hXjf = np.concatenate([Xjf,hj[keepj][Ij][:,np.newaxis],h0j[keepj][Ij][:,np.newaxis]],axis=-1)
                #plt.figure();plt.plot(hXif[:,1],hXif[:,2],'x');plt.plot(hXjf[:,1],hXjf[:,2],'x')

                key = self.ims_names[ip]+'--'+self.ims_names[jp]
                dic_chr[key] = [hXif,hXjf]



                if plt_val:
                    fig = plt.figure(figsize=(16,8))
                    ax1 = plt.subplot2grid((2,4), (0,0), colspan=2,rowspan=2)
                    ax3 = plt.subplot2grid((2,4), (0,2))
                    ax2 = plt.subplot2grid((2,4), (1, 2))
                    ax5 = plt.subplot2grid((2,4), (0, 3))
                    ax4 = plt.subplot2grid((2,4), (1, 3))
                    ax1.plot(hXif[:,1],hXif[:,2],'x');ax1.plot(hXjf[:,1],hXjf[:,2],'x')
                    ax2.plot(hXif[:,1],hXif[:,2],'x');ax2.plot(hXjf[:,1],hXjf[:,2],'x');ax2.set_xlim([0,256]);ax2.set_ylim([0,256])
                    ax3.plot(hXif[:,1],hXif[:,2],'x');ax3.plot(hXjf[:,1],hXjf[:,2],'x');ax3.set_xlim([0,256]);ax3.set_ylim([2048-256,2048])
                    ax4.plot(hXif[:,1],hXif[:,2],'x');ax4.plot(hXjf[:,1],hXjf[:,2],'x');ax4.set_ylim([0,256]);ax4.set_xlim([2048-256,2048])
                    ax5.plot(hXif[:,1],hXif[:,2],'x');ax5.plot(hXjf[:,1],hXjf[:,2],'x');ax5.set_ylim([2048-256,2048]);ax5.set_xlim([2048-256,2048])
                    for ax in [ax1,ax2,ax3,ax4,ax5]: ax.set_xticks([]);ax.set_yticks([])
                    plt.tight_layout()
                    key_ = '--'.join([e.split('_')[-1]for e in key.split('--')])
                    fig.savefig(chrom_save_fl.replace('.pkl','__'+key_+'.png'))
            self.dic_chr = dic_chr
            pickle.dump(dic_chr,open(chrom_save_fl,'wb'))
    def complete_code(self,cd):
        if not hasattr(self,'codes'): self.get_code()
        return [cd_ for cd_ in self.code_dic.keys() 
             if np.all([c_ in eval(cd_) for c_ in cd])]
    
 
    def apply_colorcor(self,chr_fl = r'\\dolly\Analysis\Bogdan\7_20_2020__OB-MER_libB32-p2_SDS-Analysis\chr_aberration_7-24-2020.pkl'):
        dic_chr = pickle.load(open(chr_fl,'rb'))
        cols = [nm.split('_')[-1] for nm in self.readout_names]
        self.zxyhsc = []
        for icol,zxy_ in enumerate(self.zxyhsf):
            zxy__ = zxy_.copy()
            col = cols[icol]
            M = dic_chr.get('M_488-'+col)
            zxy_c = ft.apply_colorcor(zxy__[:,:3],M)
            zxy__[:,:3]=zxy_c
            self.zxyhsc.append(zxy__)
    def get_std_dic_prev_bleed(self,nbits=14,ncl=2):
        dic_prev = {}
        dic_bleed = {}
        for icd in range(nbits): 
            for jcd in range(nbits):
                dic_prev[(icd,jcd)] = icd<jcd and (icd%ncl)==(jcd%ncl)
                dic_bleed[(icd,jcd)] = (int(icd/ncl)==int(jcd/ncl)) and (icd!=jcd)
        self.dic_prev = dic_prev
        self.dic_bleed = dic_bleed
        
    def im_OBmask(self):
        """Get the OB mask for the nm field of view"""
        #load the OB mask if does not exist
        if not hasattr(self, 'OB_masks'):
            self.loadOBmasks() #this populates self.OB_masks and self.OB_masks_infofl
        #load the infofile
        name,x_min,x_max,y_min,y_max,resc = self.load_coords_dax(self.OB_masks_infofl)
        name_simple = [os.path.basename(nm_).split(os.sep)[0] for nm_ in name]

        nm = self.fov
        index_nm = name_simple.index(os.path.basename(nm))
        #Decide the target image size
        if not hasattr(self,'sx_image'):
            self.sz_image,self.sx_image,self.sy_image = self.ims[0].shape
        sx_t,sy_t = self.sy_image,self.sx_image
        #load the image
        im_mask_ = self.OB_masks[int(x_min[index_nm]):int(x_max[index_nm]),int(y_min[index_nm]):int(y_max[index_nm])]
        #rescale
        from cv2 import resize,INTER_NEAREST
        im_mask = resize(im_mask_,(sx_t,sy_t),interpolation = INTER_NEAREST)
        im_mask = im_mask[::1,::-1].T

        #scope specific flips
        if not hasattr(self,'device'): self.device = 'STORM65'
        if self.device == 'STORM65':
            im_mask = im_mask[::-1,::-1]
        
        self.im_mask = im_mask
    def imshow_glomeruli(self):
        if not hasattr(self,'OB_masks_infofl'): self.loadOBmasks()
        name,x_min,x_max,y_min,y_max,resc = self.load_coords_dax(self.OB_masks_infofl)
        name_simple = [os.path.basename(nm_) for nm_ in name]
        import tifffile
        im_dapi_full = tifffile.imread(self.OB_masks_infofl.replace('.infos','.tiff'))
        fig = plt.figure(figsize=(10,10))
        plt.imshow(im,cmap='gray',vmax=10000,alpha=0.75)
        for itxt,(x_,y_) in enumerate(zip((y_min+y_max)/2,(x_min+x_max)/2)):
            plt.text(x_,y_,name_simple[itxt].split('_')[-1].split('.dax')[0],color='w')
        plt.contour(self.OB_masks>0,[0.5],colors=['r'])
        return fig
    def get_im_OB(self,filename=None,file_tag='glom3d.tif'):
        self.im_OB = np.zeros([2048,2048])
        ob_index = self.fov_index
        if filename is None:
            filename=self.analysis_folder+os.sep+file_tag
            info_fl=self.analysis_folder+os.sep+'dapi_resc8.infos'
            ob_fovs = [os.path.basename(ln.split('\t')[0]).split('.dax')[0] for ln in open(info_fl,'r')]
            self.fov_ = self.fov.split('.dax')[0]
            if self.fov_ not in ob_fovs:
                ob_index = -1
            else:
                ob_index = ob_fovs.index(self.fov_)
        if ob_index>=0:
            self.im_OB = tifffile.imread(filename,key=ob_index)[::-1]
        sh = [2048,2048]
        #if hasattr(self,'ims_dapi'):
        #    sh = self.ims_dapi[0].shape[1:]
        self.im_OB = resize(self.im_OB,sh)
    def set_X_vals(self,Xsf):
        self.Xsfk = Xsf
        self.XOB = Xsf[:,9].astype(int)
        self.Xoutcell = Xsf[:,8].astype(bool)
        self.Xdapi = Xsf[:,7]
        self.XRs = Xsf[:,6].astype(int)
        self.Xcor = Xsf[:,5]
        self.Xhabs = Xsf[:,4]
        self.Xhnorm = Xsf[:,3]
    def clean_intersections(self,nRs=15):
        self.nRs=nRs
        Rs = self.XRs
        h_abs = self.Xhabs

        r_clean = []
        for r_ in self.res:
            if len(r_)>0:
                R_ = Rs[r_]
                remove=False
                if np.any(R_>nRs):
                    h_ = h_abs[r_]
                    remove = R_[np.argmax(h_)]>=nRs
                if not remove:
                    r_clean.append(r_[R_<nRs])
        self.resk=r_clean
    def organize_inters(self):
        import itertools
        codes = self.codes_valid
        self.res_org = []
        for nlen in [2,3,4]:
            r_to_cd={}
            for icd,cd in enumerate(codes):
                cd_ = tuple(cd)
                for e in itertools.combinations(cd,nlen):
                    r_to_cd[e]=r_to_cd.get(e,[])+[icd]

            res_ = [r_ for r_ in self.resk if len(r_)==nlen]
            Rs_ = [tuple(self.XRs[r_]) for r_ in res_]
            res_org = [[] for cd in codes]
            for r_,R_ in zip(res_,Rs_):
                key = r_to_cd.get(R_,[])
                for icd in key:
                    res_org[icd].append(r_)
            self.res_org.append(res_org)
            
        #Multi-point intersection - pick the best code based on distance
        from scipy.spatial.distance import pdist
        res_org=[[]for cd in codes]
        res_org_inclusive = [[]for cd in codes]# include all of the codes
        for r_ in self.resk: 
            if len(r_)>4:
                R_ = self.XRs[r_]
                codes_in_R = np.in1d(codes,R_,assume_unique=True).reshape(codes.shape)
                cds = np.all(codes_in_R,1)
                if np.sum(cds)>0:
                    icds = np.where(cds)[0]
                    scores= []
                    for icd_,cd in zip(icds,codes[icds]):
                        X_ = self.Xsff[r_][np.in1d(R_,cd)]
                        #metric is distance from center
                        x_ = X_[:,:3]
                        #score = np.mean(np.abs(np.mean(x_,0)-x_))
                        score = np.mean(pdist(x_))
                        scores.append(score)
                        res_org_inclusive[icd_].append(r_)
                    icd_save = icds[np.argmin(scores)] #best index of code
                    res_org[icd_save].append(r_)
                    
        self.res_org.append(res_org)
        self.res_org.append(res_org_inclusive)
    def update_OBarea_fov_dic(self,A_dic={},OB_fov_dic={},th_dapi=None,new_glom=False):
        if new_glom:
            self.OBs,self.AOBs = np.unique(self.im_OB,return_counts=True)
            self.aOBs = np.zeros_like(self.OBs)
        if th_dapi is not None: self.Xoutcell = self.Xdapi<th_dapi
        for ob_ in self.OBs:
            if ob_ not in A_dic:
                OB_fov_dic[ob_]=[self.fov_index]
            else:
                OB_fov_dic[ob_]+=[self.fov_index]
        for ob_,AOB,aOB in zip(self.OBs,self.AOBs,self.aOBs):
            if ob_ not in A_dic:
                A_dic[ob_]=np.array([0,0,0,0],dtype=float)
            
            COB = np.sum((self.XOB==ob_))
            cOB = np.sum((self.XOB==ob_)&(self.Xoutcell))
            A_dic[ob_]+=np.array([aOB,AOB,cOB,COB],dtype=float)
        return A_dic,OB_fov_dic
    

        
    def update_OBcount_dic_old(self,ct_dic={},ires_org = 2,set_unique = None,n_th=np.inf,th_dapi=None):
        self.res_org = clean_res_org_3(self.res_org,ires_org)
        
        res_org = clean_within(self.res_org[ires_org])# 0,1,2->2,3,4 point intersection, 3->multi-point best, 4->all multipoint
        if th_dapi is not None: self.Xoutcell = self.Xdapi<th_dapi
        obs_u,cts_ob = np.unique(self.XOB[self.Xoutcell],return_counts=True)
        for ob_ in obs_u:
            if ob_ not in ct_dic:
                ct_dic[ob_]=np.zeros(len(self.codes_valid))
        for icd,cd in enumerate(self.codes_valid): 
            res_org_ = res_org[icd]
            if set_unique is not None:
                res_org_compare = [tuple(e)for e in res_org[set_unique]]
                res_org_ = set([tuple(e[np.in1d(self.XRs[e],cd)]) for e in res_org[icd] if len(e)<=n_th])
                res_org_ = [e for e in res_org_ if e not in res_org_compare]
            obs,cts_ = np.unique([self.XOB[e[0]]for e in res_org_ if self.Xoutcell[e[0]] and len(e)<=n_th],return_counts=True)
            for ob_,ct_ in zip(obs,cts_):
                ct_dic[ob_][icd]+=ct_
        return ct_dic
        
    def update_OBcount_dic(self,ct_dic={},ires_org = 2,set_unique = None,n_th=np.inf,th_dapi=None):
        self.res_org = clean_res_org_3(self.res_org,ires_org)
        res_org = clean_within(self.res_org[ires_org])
        #res_org = self.res_org[ires_org]# 0,1,2->2,3,4 point intersection, 3->multi-point best, 4->all multipoint
        if th_dapi is not None: self.Xoutcell = self.Xdapi<th_dapi
        obs_u,cts_ob = np.unique(self.XOB[self.Xoutcell],return_counts=True)
        for ob_ in obs_u:
            if ob_ not in ct_dic:
                ct_dic[ob_]=np.zeros(len(self.codes_valid))
        for icd,cd in enumerate(self.codes_valid): 
            res_org_ = res_org[icd]
            res_org_ = [e for e in res_org_ if self.Xoutcell[e[0]] and len(e)<=n_th]
            if set_unique is not None:
                visited = list(np.unique(self.res_org[set_unique][icd])) #all the visited points
                res_org__ = []
                for e in res_org_:
                    egood = e[np.in1d(self.XRs[e],cd)]
                    if not np.any([e_ in visited for e_ in egood]):
                        visited.extend(egood)
                        res_org__.append(e)
                res_org_ = res_org__
            obs,cts_ = np.unique([self.XOB[e[0]]for e in res_org_],return_counts=True)
            for ob_,ct_ in zip(obs,cts_):
                ct_dic[ob_][icd]+=ct_
        return ct_dic
    def save_intersections(self):
        base_dec_file = self.final_fits_file.replace(os.sep+'Fits'+os.sep,os.sep+'Decoded'+os.sep)
        if not os.path.exists(os.path.dirname(base_dec_file)):
            os.makedirs(os.path.dirname(base_dec_file))
        fl = base_dec_file.replace('.pkl','_Xsf.npy')
        np.save(fl,self.Xsff)
        fl = base_dec_file.replace('.pkl','_ints.npy')
        np.save(fl,self.res)
        
        fl = base_dec_file.replace('.pkl','_res_prev_org.npy')
        np.save(fl,self.res_org[:2])
        self.res_org[:2] = [[],[]]
        fl = base_dec_file.replace('.pkl','_res_org.npy')
        np.save(fl,self.res_org)
        
        AOBf = np.array([self.OBs,self.AOBs,self.aOBs])
        fl = base_dec_file.replace('.pkl','_AOBf.npy')
        np.save(fl,AOBf)
        fl = base_dec_file.replace('.pkl','_flagV2.tmp')
        np.save(fl,[])
    def get_intersections(self,cor_th=0.45,dinstance_th = 1.5,nRs=15):
        """
        Given that .get_code is used, this goes through zxyhsf and looks for perfect intersections.
        It screens for cor_th first and then  for distance_th.
        """
        Xsf = self.Xsff.copy()
        self.set_X_vals(Xsf)
        #cutoff brightness/correlation and inside OB
        keep = (self.Xcor>cor_th)#&(self.XOB>0)
        Xsf = Xsf[keep]
        #self.set_X_vals(Xsf)
        # Find all intersections
        Xs = Xsf[:,:3]
        self.res=[]
        if len(Xs)>0:
            Ts = cKDTree(Xs)
            res = Ts.query_ball_tree(Ts,dinstance_th)
            keep_i = np.where(keep)[0]
            res=[keep_i[r_] for r_ in res]
            self.res = np.array([np.array(eval(e)) for e in np.unique([str(list(np.sort(r))) for r in res])])#make unique
    def replace_OB(self,Xsf,file_tag='glom3d_new.tif'):
        self.get_im_OB(filename=None,file_tag=file_tag)
        X,Y = Xsf[:,1:3].astype(int).T
        sh = self.im_OB.shape
        X[X>=sh[0]]=sh[0]-1
        Y[Y>=sh[1]]=sh[1]-1
        X[X<0]=0
        Y[Y<0]=0
        Xsf[:,9]=self.im_OB[X,Y]
        return Xsf
    def refine_OB_mask(self,th_dapi = 1.5,iref = 1,rescs=[2,8,8],radius=10,shape_ = [2048,2048],plt_val=False):
        self.get_im_OB()
        
        self.im_mask_OB = np.zeros(shape_,dtype=np.uint8)
        if np.any(self.im_OB>0) or plt_val:
            im_dapi = self.ims_dapi[iref] #mem mapped 3d image of dapi
            im_dapi=load_im_col(im_dapi)
            im_dapi_ = np.array(im_dapi[::rescs[0],::rescs[1],::rescs[2]]) #rescale
            im_dapi_ = im_dapi_/self.ims_mean[-1][::rescs[1],::rescs[2]] #normalize
            im_dapi__th = np.max(im_dapi_,0)>th_dapi #threshold
            im_dapi__th_ = resize(im_dapi__th,shape_ = im_dapi.shape[1:])
            im_dapi__th_ = dilate(im_dapi__th_,radius)
            self.im_dapi_th = im_dapi__th_
            self.im_norm_dapi = resize(np.max(im_dapi_,0),shape_ = im_dapi.shape[1:])
            
            self.im_mask_OB = self.im_OB*(self.im_dapi_th==0)
            
            self.OBs = np.unique(self.im_OB)
            self.AOBs = np.array([np.sum(self.im_OB==OB_) for OB_ in self.OBs])
            self.aOBs = np.array([np.sum(self.im_mask_OB==OB_) for OB_ in self.OBs])
            
            if hasattr(self,'Xsf'):
                ims_ = np.array([self.im_norm_dapi,1-self.im_dapi_th,self.im_OB])
                X,Y = self.Xsf[:,1:3].astype(int).T
                X[X>=ims_.shape[1]]=ims_.shape[1]-1
                Y[Y>=ims_.shape[2]]=ims_.shape[2]-1
                X[X<0]=0
                Y[Y<0]=0
                self.Xsff = np.concatenate([self.Xsf,ims_[:,X,Y].T],-1)
            
            if plt_val:
                fig = plt.figure()
                plt.imshow(self.im_norm_dapi,vmax=4,cmap='gray')
                for iB in np.unique(self.im_mask_OB):
                    if iB!=0:
                        im_ = self.im_mask_OB==iB
                        xB,yB = np.where(im_)
                        plt.contour(im_,[0.5],colors=['g'])
                        plt.text(np.mean(yB),np.mean(xB),str(iB),color=[0,1,0])
                #plt.contour(self.im_mask>0,[0.5],colors=['r'],alpha=0.33)
                return fig
    def get_OB(self,xy):
        y,x = xy
        x,y = int(x),int(y)
        sx,sy = self.im_mask_OB.shape
        if x<sx and x>0 and y<sy and y>0: return self.im_mask_OB[y,x]
        else: return 0