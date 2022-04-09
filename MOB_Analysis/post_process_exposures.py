### This assumes a infos_dic folder is loaded.

import numpy as np
from tqdm.notebook import tqdm
import os,glob,sys
import pickle
import matplotlib.pylab as plt
from tqdm.notebook import tqdm
from mock import Mock
from scipy.signal import fftconvolve   
import cv2

def get_im(im,xm,xM,pad = 0):
    xm,xM = np.array(xm,dtype=int)-pad,np.array(xM,dtype=int)+pad
    sh = xM-xm
    sh = list(sh)#+list(im.shape[2:])
    im_ = np.zeros(sh,dtype=im.dtype)
    xm_ = xm.copy()
    xm_[xm_<0]=0
    im_send = im[xm_[0]:xM[0],xm_[1]:xM[1],xm_[2]:xM[2]]
    sh_ = im_send.shape
    xm__ = xm_-xm
    im_[xm__[0]:+xm__[0]+sh_[0],xm__[1]:+xm__[1]+sh_[1],xm__[2]:+xm__[2]+sh_[2]]=im_send
    return im_
def loadMap(dax_fl,x_pixels=2048,y_pixels=2048):
    im = np.memmap(dax_fl,dtype='uint16',mode='r')
    im = im.reshape([-1,x_pixels,y_pixels]).swapaxes(1,2)
    return np.swapaxes(im,1,2)
    
def norm_im(im,max_=4):
    im_ = np.array(im).astype(np.float32)
    im_ = im_/cv2.blur(im_,(50,50))
    im_[im_>max_]=max_
    im_ =cv2.blur(im_,(2,2))
    im_ = (im_-np.mean(im_))/np.std(im_)
    return im_
def plot_ncells(obj,fr=0.1,ncells=10):

    save_folder = r'\\MCCLINTOCK\mcclintock_3\Bogdan_Bil_temporary\extra_files\NO_UPLOAD'
    top_o=[o for o in obj.dic_frA if obj.dic_frA[o][0]>fr]
    for OLFR in top_o:
        for icell in range(ncells):
            for isActive in [True,False]:
                if True:#try:
                    fig,tag = get_fig_olfr(obj,OLFR = OLFR,icell=icell,hth=None,nth=None,isActive=isActive,pad = [0,75,75])
                    etag = '__nth'+tag.split('_nth')[-1].split('_')[0]+'__hth'+tag.split('_hth')[-1].split('_')[0]
                    save_folder_ = save_folder+os.sep+obj.key+etag
                    if not os.path.exists(save_folder_): os.makedirs(save_folder_)
                    fig.savefig(save_folder_+os.sep+tag+'.png')
                else:#except:
                    pass
def get_cor_objs(obj,obj2):
    o1 = list(obj.dic_frA.keys())
    o2 = list(obj2.dic_frA.keys())
    o_ = np.union1d(o1,o2)
    fr1 = [obj.dic_frA.get(o,[0,0,0])[0] for o in o_]
    fr2 = [obj2.dic_frA.get(o,[0,0,0])[0] for o in o_]
    import matplotlib.pylab as plt
    plt.figure()
    plt.loglog(fr1,fr2,'o')
    lfr1,lfr2=np.log(fr1),np.log(fr2)
    good = ~(np.isnan(lfr1+lfr2)|np.isinf(lfr1+lfr2))
    plog =np.corrcoef(lfr1[good],lfr2[good])[0,1]
    plin = np.corrcoef(fr1,fr2)[0,1]
    from scipy.stats import spearmanr
    print(spearmanr(fr1,fr2))
    plt.title(obj.key+'--'+obj2.key+'\n'+'plog='+str(np.round(plog,2))+' plin='+str(np.round(plin,2)))
def get_abs_pos(cell):
    return cell.c[1:]*cell.paramaters['nm_per_pixel_xy']/1000.+[-cell.paramaters['stage_x'],cell.paramaters['stage_y']][::-1]
def get_info(cell):
    #cell_id,olfactory_receptor_gene,boundaryX,boundaryY,boundaryZ,absolute_center_x,absolute_center_y,fov_id,slice_id,number of EGR1 molecules
    um_xy = cell.paramaters['nm_per_pixel_xy']/1000.
    pix = [um_xy,um_xy,0.3]
    um_xy = cell.paramaters['nm_per_pixel_xy']/1000.
    pix = [um_xy,um_xy,0.3]
    boundary = pix*cell.points[cell.vertices]
    olfr = cell.olfr.split('-')[0].split(' ')[0]
    abs_pos = get_abs_pos(cell)
    txys = np.array([cell.paramaters['txys'].get(iR,[0,0]) for iR in range(16)])
    return [olfr,boundary,abs_pos,cell.fov,cell.MOE,cell.zxyh_EGR1_semiloose,cell.sublibrary,pix,cell.volume,cell.code,cell.cor,cell.cor_average,cell.mean_traces,cell.std_traces,cell.mean_traces_bk,cell.std_traces_bk,txys]
 
    
def get_fig_olfr(obj,OLFR = 'Olfr204',icell=0,hth=None,nth=None,isActive=None,pad = [0,100,100]):
    #print(obj.key)
    tag = sample_dic[obj.key]
    dt_flds = [fld for fld in all_data_folders if tag in fld]
    if hth is None: hth=obj.hth
    if nth is None: nth=obj.nth
    active_ = np.array([np.sum(hA>=hth) for hA in obj.hAs])>=nth

    if isActive is not None:
        if isActive: icells = np.where((obj.olfrs==OLFR)&active_)[0]
        else: icells = np.where((obj.olfrs==OLFR)&(~active_))[0]
    else:
        icells = np.where((obj.olfrs==OLFR))[0]
    icell = icells[icell]
    fov = obj.fovs[icell]
    X_ = obj.bds[icell].astype(int)
    code = obj.dic_olfr_code[OLFR]

    #def reconstruct_im_cell(fov,X_,dt_flds):
    Xm,XM = np.min(X_,0),np.max(X_,0)
    etag = 'set2' if 'zscan2' in fov else 'set1' if 'set' in dt_flds[0] else ''
    hflds=[fld_ for fld in dt_flds if etag in fld for fld_ in glob.glob(fld+os.sep+'H*')]
    dax_fls = [fld+os.sep+fov+'.dax' for fld in hflds]
    col3 = '3col' in hflds[0]
    ncol = 4 if col3 else 3
    def htag_to_Rs(fl):return [0] if 'egr1' in fl.lower().split(';')[0] else \
                    eval('['+os.path.basename(os.path.dirname(fl)).split('R')[1].split(';')[0].replace('B,','')+']')
    R_to_col3={0:'647',1:'750',2:'647',3:'561',4:'750',6:'647',5:'561',7:'750',8:'647',9:'561',10:'750',12:'647',11:'561',14:'750',15:'647',13:'561'}
    R_to_col2={0:'647',1:'561',2:'647',3:'561',4:'647',6:'647',5:'561',7:'561',8:'647',9:'561',10:'647',12:'647',11:'561',14:'647',15:'647',13:'561'}
    R_to_col = R_to_col3 if col3 else R_to_col2

    cols = ['561','647','750'][:ncol-1]
    htags = [os.path.basename(fld) for fld in hflds]
    dic_ims,dic_dapi,dic_nm = {},{},{}
    for dax_fl in dax_fls:
        Rs = htag_to_Rs(dax_fl)
        im = loadMap(dax_fl)
        dic_dapi[dax_fl]=im[ncol-1::ncol]
        for R in Rs:
            icol = cols.index(R_to_col[R])
            dic_ims[R] =im[icol::ncol]
            dic_nm[R]=dax_fl
    if True:
        dic_txy={}

        
        iref,ireffr = 0,20
        im_ref = norm_im(dic_dapi[dax_fls[iref]][ireffr])
        for dax_fl in tqdm(dax_fls):
            im_t = norm_im(dic_dapi[dax_fl][ireffr])

            im_cor = fftconvolve(im_t,im_ref[::-1,::-1],mode='full')
            tx_,ty_ = np.unravel_index(np.argmax(im_cor),im_cor.shape)
            tx,ty = np.array([tx_,ty_])-im_ref.shape+1
            dic_txy[dax_fl] = [tx,ty]
            

    rref = 0
    im_sms = []
    
    for R in list(code)+[0,0]:
        tx,ty = np.array(dic_txy[dic_nm[R]])-dic_txy[dic_nm[rref]]
        Xm_ = Xm+[0,ty,tx]
        XM_ = XM+[0,ty,tx]
        im = np.swapaxes(dic_ims[R],1,2)
        Xm_[0],XM_[0]=0,im.shape[0]
        im_sm = get_im(im,Xm_,XM_,pad=pad)
        im_sms.append(im_sm)
        
    from scipy.spatial import ConvexHull
    c1_2d = ConvexHull((X_-Xm+pad)[:,[1,2]])
    Xc1_2d = c1_2d.points[list(c1_2d.vertices)+[c1_2d.vertices[0]]]
    XA_ = obj.XAs[icell][:,:3]-Xm[:]+pad[:]

    fig = plt.figure(figsize=(5*6,5))
    tag = OLFR+'__'+str(icell)+'__nth'+str(nth)+'__hth'+str(hth)+'__'+str(isActive)+'__'+obj.key
    plt.title(tag)
    plt.plot(Xc1_2d[:,1],Xc1_2d[:,0],'r-',alpha=0.5)
    plt.imshow(np.concatenate([np.max([im_.astype(np.float32)/np.median(im_) for im_ in im_sm],0) for im_sm in im_sms],1),vmax=3.5,cmap='gray')
    keep = obj.XAs[icell][:,-1]>hth
    plt.scatter(im_sms[0].shape[2]*4+XA_[keep,2],XA_[keep,1], s=80, facecolors='none', edgecolors='r')
    return fig,tag
def bin_expand(im_, radius=5):
    x_,y_ = (np.indices([2*radius+1]*2)-radius)/radius
    footprint = ((x_*x_+y_*y_)<1).astype(np.uint8)
    imf = cv2.morphologyEx(im_.astype(np.uint8), cv2.MORPH_DILATE, footprint)
    return imf
    
def get_expr_dic(mean_traces_,mean_traces_bk_,std_traces_bk_,olfrs,dic_olfr_code,plt_val=False):
    mean_traces_ = np.array(mean_traces_)
    mean_traces_bk_ = np.array(mean_traces_bk_)
    std_traces_bk_ = np.array(std_traces_bk_)
    olfrs = np.array(olfrs)
    # get approximate values
    scores = (mean_traces_-mean_traces_bk_)/std_traces_bk_
    good = np.array([scores[io][dic_olfr_code[o]-1] for io,o in enumerate(olfrs)])
    bad = scores+np.nan
    for io,o in enumerate(olfrs):
        iind = np.setdiff1d(np.arange(len(scores[0])),dic_olfr_code[o]-1)
        bad[io][iind]=scores[io][iind]
    codes_ = np.array([dic_olfr_code[o]-1 for io,o in enumerate(olfrs)])
    red_codes_ = np.array([codes_[icd,cd][:2] for icd,cd in enumerate(np.argsort(good,axis=-1)[:,::-1])])
    #red_codes_ = np.sort(red_codes_,axis=-1)
    scores_raw = (mean_traces_-mean_traces_bk_)
    bad_med = np.nanmedian(bad,axis=0)
    red_scores_ = np.array([scores[icd,cd]-bad_med[cd] for icd,cd in enumerate(red_codes_)])
    red_scores_f = red_scores_.copy()

    all_pairs = {}
    nbits = len(scores[0])
    for ib in range(nbits):
        for jb in range(ib):
            pairs = np.array([[icd,np.where(cd==ib)[0][0],np.where(cd==jb)[0][0]] 
                              for icd,(cd,sc) in enumerate(zip(red_codes_,red_scores_f)) if jb in cd and ib in cd])
            all_pairs[(ib,jb)]=pairs
            
    red_scores_f = red_scores_.copy()
    for iit in range(20):
        keys_ = [list(all_pairs.keys())[i_] for i_ in np.random.permutation(len(all_pairs))]
        rs = []
        for pr in keys_:
            if len(all_pairs[pr])>0:
                H1 = red_scores_f[all_pairs[pr][:,0],all_pairs[pr][:,1]]
                H2 = red_scores_f[all_pairs[pr][:,0],all_pairs[pr][:,2]]
                ib,jb=pr
                h1,h2 = np.median(H1),np.median(H2)
                r1 = np.sqrt(h1*h2)/h1
                r2 = np.sqrt(h1*h2)/h2
                if np.isnan(r1):r1,r2 = 1,1
                if np.isnan(r2):r1,r2 = 1,1
                rs.append(h1/h2)
                red_scores_f[red_codes_==ib]*=r1
                red_scores_f[red_codes_==jb]*=r2
            #print(r1,r2)
        #print(np.std(rs))        
    olfrs = np.array(olfrs)
    os,inv = np.unique(olfrs,return_inverse=True)
    dic_expr1 = {}
    dic_expr2 = {}
    for iinv,o in enumerate(os):
        H1,H2 = red_scores_f[inv==iinv].T
        dic_expr1[o] = np.median(H1)
        dic_expr2[o] = np.median(H2)
    if plt_val:
        dic1,dic2 = dic_expr1,dic_expr2
        keys = np.intersect1d(list(dic1.keys()),list(dic2.keys()))
        X1 = np.array([dic1[o] for o in keys])
        X2 = np.array([dic2[o] for o in keys])
        X1,X2 = X1/np.sum(X1),X2/np.sum(X2)
        import matplotlib.pylab as plt
        plt.figure()
        plt.plot(X1,X2,'o')
    return dic_expr1
def get_fr_obj(infos_dic,key='mouse1_sample1',hth=2.5,nth=7,obj=None):
    
    infos = infos_dic[key]
    nctsA = []
    fovs = []
    hAs = []
    olfrs=[]
    MOEs = []
    bds = []
    dic_olfr_code={}
    XAs = []
    for info in tqdm(infos):
        [olfr,boundary,abs_pos,fov,MOE,zxyh_EGR1_semiloose,sublibrary,pix,volume,code,
        cor,cor_average,mean_traces,std_traces,mean_traces_bk,std_traces_bk] = info
        hA = zxyh_EGR1_semiloose[:,-1]
        hAs.append(hA)
        MOEs.append(MOE)
        ncts = np.sum(hA>hth)
        nctsA.append(ncts)
        fovs.append(fov)
        olfrs.append(olfr)
        bds.append(boundary/pix)
        dic_olfr_code[olfr]=code
        XAs.append(zxyh_EGR1_semiloose)
    olfrs = np.array(olfrs)
    bds = np.array(bds)
    nctsA=np.array(nctsA)
    fovs = np.array(fovs)
    hAs= np.array(hAs)
    MOEs= np.array(MOEs)
    XAs = np.array(XAs)
    #medH = np.median(hAs)
    #print(key,medH,medH+np.median(np.abs(hAs-medH))*5)
    O,C = np.unique(olfrs,return_counts=True)
    dic_counts = {o:c for o,c in zip(O,C)}
    O,C = np.unique(olfrs[nctsA>=nth],return_counts=True)
    dic_countsA = {o:c for o,c in zip(O,C)}
    dic_frA = {o:(float(dic_countsA.get(o,0))/dic_counts[o],dic_countsA.get(o,0),dic_counts[o]) for o in dic_counts}
    
    
    
    if obj is None: obj = Mock()
    obj.dic_olfr_code=dic_olfr_code
    obj.fovs=fovs
    obj.olfrs = olfrs
    obj.nctsA = nctsA
    obj.bds=bds
    obj.hAs = hAs
    obj.hth = hth
    obj.nth = nth
    obj.key = key
    obj.dic_frA = dic_frA
    obj.dic_countsA = dic_countsA
    obj.XAs=XAs
    #obj.MOEs = MOEs
    
    Xs,olfrs,OU,C = get_Xolfr(infos_dic,obj.key)
    obj.Xs = Xs
    obj.olfrsX = olfrs
    obj.OU = OU
    obj.C = C
    
    return obj
def get_sorted_fr(obj):
    frs,olfrs_=zip(*[(obj.dic_frA[olfr][0],olfr) for olfr in obj.dic_frA])
    olfrs_ = np.array(olfrs_)[np.argsort(frs)[::-1]]
    return [(o,obj.dic_frA[o]) for o in olfrs_]
def recalc_MOEs(obj,n_clusters=6,plt_val=False):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters,random_state=0)
    kmeans.fit(obj.Xs)
    lab = kmeans.labels_
    obj.MOEs=lab
    if plt_val:
        plt.figure()
        for iL in np.unique(lab):
            plt.plot(obj.Xs[iL==lab,0],obj.Xs[iL==lab,1],'.')
        plt.axis('equal')
def get_dic_fr(infos_dic,keys=['mouse1_sample1'],hths=[2.],nths=[8],return_cts=False):
    dic_frAf = {}
    nctsA_=[]
    for key,hth,nth in zip(keys,hths,nths):
        infos = infos_dic[key]
        nctsA = []
        fovs = []
        hAs = []
        olfrs=[]
        for info in tqdm(infos):
            [olfr,boundary,abs_pos,fov,MOE,zxyh_EGR1_semiloose,sublibrary,pix,volume,code,
            cor,cor_average,mean_traces,std_traces,mean_traces_bk,std_traces_bk] = info
            hA = zxyh_EGR1_semiloose[:,-1]
            hAs.extend(hA)
            ncts = np.sum(hA>hth)
            nctsA.append(ncts)
            fovs.append(fov)
            olfrs.append(olfr)
        olfrs = np.array(olfrs)
        nctsA=np.array(nctsA)
        fovs = np.array(fovs)
        hAs= np.array(hAs)
        medH = np.median(hAs)
        print(key,medH,medH+np.median(np.abs(hAs-medH))*5)
        O,C = np.unique(olfrs,return_counts=True)
        dic_counts = {o:c for o,c in zip(O,C)}
        O,C = np.unique(olfrs[nctsA>=nth],return_counts=True)
        dic_countsA = {o:c for o,c in zip(O,C)}
        dic_frA = {o:(float(dic_countsA.get(o,0))/dic_counts[o],dic_countsA.get(o,0),dic_counts[o]) for o in dic_counts}
        dic_frAf.update(dic_frA)
        nctsA_.extend(nctsA)
    if return_cts:
        return dic_frAf,np.array(nctsA_)
    return dic_frAf
def coef(es1,es2):
    return 2*np.sum(np.array(es1)*es2)/(np.sum(np.array(es1)*es1)+np.sum(np.array(es2)*es2))
def reencode(dic_EGR1_fr):
    dic_EGR1_fr_ = {key_:{key__.decode('ascii'):dic_EGR1_fr[key_][key__] for key__ in dic_EGR1_fr[key_]} for key_ in dic_EGR1_fr}
    return dic_EGR1_fr_
def combine(dic_fr1,dic_fr2):
    from copy import deepcopy
    dic_fr = deepcopy(dic_fr1)
    dic_fr.update(dic_fr2)
    return dic_fr
    
    

def combine_mean(dic_fr1,dic_fr2):
    dic_fr1_ = {e:dic_fr1[e][0] if hasattr(dic_fr1[e], '__iter__') else dic_fr1[e] for e in dic_fr1}
    dic_fr2_ = {e:dic_fr2[e][0] if hasattr(dic_fr2[e], '__iter__') else dic_fr2[e] for e in dic_fr2}
    from copy import deepcopy
    dic_fr_ = deepcopy(dic_fr1_)
    for o in dic_fr2_:
        dic_fr_[o]= dic_fr2_[o] if o not in dic_fr1_ else (dic_fr2_[o]+dic_fr1_[o])/2.
    return dic_fr_
def plot_activity_on_reference(infos_dic,dic_fr,key_ref = 'mouse5_sample2',color='r',ealpha=0.5,th_fr=0.1):
    dic_fr_ = {e:dic_fr[e][0] if hasattr(dic_fr[e], '__iter__') else dic_fr[e] for e in dic_fr}
    os = [o for o in dic_fr_ if dic_fr_[o]>th_fr]
    oFRs = [dic_fr_[o] for o in os]

    Xs,olfrs,OU,C = get_Xolfr(infos_dic,key_ref)

    import matplotlib.pylab as plt
    plt.figure()
    plt.plot(Xs[:,0],Xs[:,1],'.',color='gray')
    for o in os:
        X_ = Xs[olfrs==o] 
        plt.plot(X_[:,0],X_[:,1],'.',color=color,alpha=dic_fr_[o]*ealpha)
    plt.axis('equal')
from scipy.spatial.distance import cdist
def calc_dists(X1O,X2O,X2OT,O1,O2,save_fl_final_,nneigh=1):
    olfrs1O,olfrs2O=O1,O2
    dic_nneigh = {}
    dic_med_dist = {}
    
    X1OK,X2OTK,O1K,O2K = get_good_pairs_across_sections(X1O,X2OT,O1,O2,resc_im = 20,radius_int=5)
    res_all = [[X1O,X1O,olfrs1O,olfrs1O],[X2O,X2O,olfrs2O,olfrs2O],[X1OK,X2OTK,O1K,O2K]]
    
    
    
    for X1__,X2__,O1__,O2__ in res_all:
    
        OU1__ = np.unique(O1__)
        OU2__ = np.unique(O2__)
        for O1_ in tqdm(OU1__):
            for O2_ in OU2__:
                for O1_k,O2_k,XO_1,XO_2 in [[O1_,O2_,X1__[O1__==O1_],X2__[O2__==O2_]],
                                            [O2_,O1_,X2__[O2__==O2_],X1__[O1__==O1_]]]:
                    if (O1_k,O2_k) not in dic_nneigh:
                        nneigh_ = nneigh+1 if len(XO_2)>nneigh else len(XO_2)
                        distances = cdist(XO_1,XO_2)

                        distances = np.sort(distances,-1)
                        if O1_==O2_: distances_ = distances[:,1:nneigh_]
                        if O1_!=O2_ and nneigh_>nneigh: distances_ = distances[:,:nneigh_-1]
                        dic_nneigh[(O1_k,O2_k)]=(np.median(np.median(distances_,-1)),len(XO_1),len(XO_2))

                        distances_ = distances[:,1:] if O1_==O2_ else distances[:,:]
                        dic_med_dist[(O1_k,O2_k)] = (np.median(distances_),len(XO_1),len(XO_2))
            

    pickle.dump([dic_nneigh,dic_med_dist],open(save_fl_final_,'wb'))
    
    
def calc_dists_distr(X1O,X2O,X2OT,O1,O2,save_fl_final_):
    olfrs1O,olfrs2O=O1,O2
    dic_distr = {}
    
    X1OK,X2OTK,O1K,O2K = get_good_pairs_across_sections(X1O,X2OT,O1,O2,resc_im = 20,radius_int=5)
    res_all = [[X1O,X1O,olfrs1O,olfrs1O],[X2O,X2O,olfrs2O,olfrs2O],[X1OK,X2OTK,O1K,O2K]]
    
    
    
    for X1__,X2__,O1__,O2__ in res_all:
    
        OU1__ = np.unique(O1__)
        OU2__ = np.unique(O2__)
        for O1_ in tqdm(OU1__):
            for O2_ in OU2__:
                O1_k,O2_k,XO_1,XO_2 = O1_,O2_,X1__[O1__==O1_],X2__[O2__==O2_]
                distances = cdist(XO_1,XO_2)
                distances = np.sort(distances,-1)
                distances_ = distances[:,1:] if O1_==O2_ else distances[:,:]
                dic_distr[(O1_k,O2_k)] = np.histogram(distances_,np.arange(0,7000,20))[0]
                dic_distr[(O2_k,O1_k)] = dic_distr[(O1_k,O2_k)]

    pickle.dump(dic_distr,open(save_fl_final_,'wb'))
def get_good_pairs_across_sections(X1O,X2OT,O1,O2,resc_im = 20,radius_int=5,plt_val=False):
    X12 = np.concatenate([X1O,X2OT],axis=0)
    mins = np.min(X12,0)
    xyM = np.max(X12,0)

    im1 = plts_to_im(X1O,resc=resc_im,mins=mins,xyM = xyM)
    im2 = plts_to_im(X2OT,resc=resc_im,mins=mins,xyM = xyM)
    #print(im1.shape,im2.shape)
    im1_ = bin_expand(im2,radius=radius_int)*im1
    im2_ = bin_expand(im1,radius=radius_int)*im2

    #plt.figure()
    #plt.imshow(im1_+im2_*2,cmap='jet')

    X1red = ((X1O-mins)/resc_im).astype(int)
    X2red = ((X2OT-mins)/resc_im).astype(int)

    keep1 = im1_[X1red[:,0],X1red[:,1]]>0
    keep2 = im2_[X2red[:,0],X2red[:,1]]>0
    X1OK = X1O[keep1]
    X2OTK = X2OT[keep2]
    O1K = O1[keep1]
    O2K = O2[keep2]
    
    if plt_val:
        plt.figure()
        plt.plot(X1OK[:,0],X1OK[:,1],'r.')
        plt.plot(X2OTK[:,0],X2OTK[:,1],'b.')
        plt.plot(X1O[~keep1,0],X1O[~keep1,1],'.',color='orange')
        plt.plot(X2OT[~keep2,0],X2OT[~keep2,1],'.',color='cyan')
    return X1OK,X2OTK,O1K,O2K    
    
    
from sklearn.cluster import KMeans
def plts_to_im(cms,resc=10,mins=None,xyM = None):
    cms_ = (cms/resc).astype(int)
    if mins is None:
        mins=np.min(cms,0)
    mins = (mins/resc).astype(int)
    cms_ = cms_-mins
    if xyM is None:
        xyM = np.max(cms,0)
    xM,yM = (xyM/resc).astype(int)-mins+10
    im_ = np.zeros([xM,yM],dtype=np.float32)
    im_[cms_[:,0],cms_[:,1]]=1
    return im_
from scipy.signal import fftconvolve
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
def norm_im(im,gsz=7):
    im_=np.array(im,np.float32)
    im_ = cv2.blur(im_,(gsz,gsz))
    #im_ = im_-np.mean(im_)
    #im_ = im_/np.std(im_)
    return im_
def align_ims(im1,im2,plt_val=True):
        
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
            
    
    fftim = fftconvolve(rotate_image(im1, angle)[::-1,::-1],im2,mode='full')
    score = np.max(fftim)
    if plt_val:
        plt.figure()
        plt.imshow(fftim)
    xm = np.array(np.unravel_index(np.argmax(fftim),fftim.shape))-im1.shape
    sh = np.array(im1.shape)
    xM = xm+sh
    pad=1000#1000
    im2T = get_im(im2,xm,xM,pad = pad )
    im2T = rotate_image(im2T, -angle)[pad:-pad,pad:-pad]
    
    im2T_ = (im2T-np.min(im2T))/(np.max(im2T)-np.min(im2T))
    im2_ = (im2-np.min(im2))/(np.max(im2)-np.min(im2))
    im1_ = (im1-np.min(im1))/(np.max(im1)-np.min(im1))
    score = 2*np.sum(im2T_*im1)/(np.sum(im1_)+np.sum(im2_))
    if plt_val:
        f,(ax1,ax2)=plt.subplots(1,2,sharex=True,sharey=True)
        ax1.imshow(im1,cmap='gray')
        ax2.imshow(im2T,cmap='gray')
    return angle,xm[0],xm[1],score
def align_ims_refl(im1,im2,plt_val=False):
    thxys_scores = []
    for pm in [1,-1]:
        th,xt,yt,score = align_ims(im1,im2[::pm],plt_val=plt_val)
        thxys_scores.append([th,xt,yt,pm,im1.shape[0],im1.shape[1],
                             im2.shape[0],im2.shape[1],score])
    thxys_scores = np.array(thxys_scores)
    return thxys_scores[np.argmax(thxys_scores[:,-1])]

def get_coef(dic_fr1,dic_fr2):
    oints = np.intersect1d(list(dic_fr1.keys()),list(dic_fr2.keys()))
    es1 = np.array([dic_fr1[e][0] if hasattr(dic_fr1[e], '__iter__') else dic_fr1[e] for e in oints])
    es2 = np.array([dic_fr2[e][0] if hasattr(dic_fr2[e], '__iter__') else dic_fr2[e] for e in oints])
    return coef(es1,es2),es1,es2
def plot_olfr(obj,olfr):
    plt.figure()
    plt.title(olfr)
    plt.plot(obj.Xs[:,0],obj.Xs[:,1],'.',color='gray')
    plt.plot(obj.Xs[obj.olfrs==olfr,0],obj.Xs[obj.olfrs==olfr,1],'.',color='red')
    plt.axis('equal')
from scipy.spatial.distance import cdist,pdist

def save_obj(obj):
    import os,pickle
    
    save_fl = r'\\MCCLINTOCK\mcclintock_3\Bogdan_Bil_temporary\extra_files\NO_UPLOAD'+os.sep+obj.key+'_obj.pkl'
    pickle.dump(obj.__dict__,open(save_fl,'wb'))
def load_obj(key):
    import os,pickle
    save_fl = r'\\MCCLINTOCK\mcclintock_3\Bogdan_Bil_temporary\extra_files\NO_UPLOAD'+os.sep+key+'_obj.pkl'
    obj = Mock()
    obj.__dict__=pickle.load(open(save_fl,'rb'))
    return obj
def get_distance_mat(olfrs1O,X1O,MOE1=None,olfrs2O=None,X2O=None,X2OT=None,MOE2=None,nneigh=1):
    OU1 = np.unique(olfrs1O)
    if MOE1 is None: MOE1 = np.ones(len(olfrs1O))

    

    dic_nneigh = {}
    dic_nneigh_raw = {}
    for O1_ in tqdm(OU1):
        for O2_ in OU1:
            XO_1 = X1O[olfrs1O==O1_]
            XO_2 = X1O[olfrs1O==O2_]
            
            nneigh_ = nneigh+1 if len(XO_2)>nneigh else len(XO_2)
            distances = np.sort(cdist(XO_1,XO_2),-1)[:,:nneigh_]
            if O1_==O2_: distances = distances[:,1:]
            if O1_!=O2_ and nneigh_>nneigh: distances = distances[:,:-1]
            dic_nneigh[(O1_,O2_)]=(np.median(np.median(distances,-1)),len(XO_1),len(XO_2))
            dic_nneigh_raw[(O1_,O2_)]=np.median(distances,-1)
    if olfrs2O is not None:
        OU2 = np.unique(olfrs2O)
        for O1_ in tqdm(OU2):
            for O2_ in OU2:
                XO_1 = X2O[olfrs2O==O1_]
                XO_2 = X2O[olfrs2O==O2_]
                
                nneigh_ = nneigh+1 if len(XO_2)>nneigh else len(XO_2)
                distances = np.sort(cdist(XO_1,XO_2),-1)[:,:nneigh_]
                if O1_==O2_: distances = distances[:,1:]
                if O1_!=O2_ and nneigh_>nneigh: distances = distances[:,:-1]
                dic_nneigh[(O1_,O2_)]=(np.median(np.median(distances,-1)),len(XO_1),len(XO_2))
                dic_nneigh_raw[(O1_,O2_)]=np.median(distances,-1)
                
        for O1_ in tqdm(OU1):
            for O2_ in OU2:
                XO_1 = X1O[olfrs1O==O1_]
                XO_2 = X2OT[olfrs2O==O2_]
                nneigh_ = nneigh+1 if len(XO_2)>nneigh else len(XO_2)
                mat = cdist(XO_1,XO_2)
                distances = np.sort(mat,-1)[:,:nneigh_]
                if O1_==O2_: distances = distances[:,1:]
                if O1_!=O2_ and nneigh_>nneigh: distances = distances[:,:-1]
                dic_nneigh[(O1_,O2_)]=(np.median(np.median(distances,-1)),len(XO_1),len(XO_2))
                
                nneigh_ = nneigh+1 if len(XO_1)>nneigh else len(XO_1)
                distances = np.sort(mat.T,-1)[:,:nneigh_]
                if O1_==O2_: distances = distances[:,1:]
                if O1_!=O2_ and nneigh_>nneigh: distances = distances[:,:-1]
                dic_nneigh[(O2_,O1_)]=(np.median(np.median(distances,-1)),len(XO_2),len(XO_1))
                dic_nneigh_raw[(O1_,O2_)]=np.median(distances,-1)
    return dic_nneigh,dic_nneigh_raw         
def dic_nneigh_to_mat(dic_nneigh,list_olfr,min=True):
    mat = np.zeros([len(list_olfr)]*2)
    for io1,o1 in enumerate(list_olfr):
        for io2,o2 in enumerate(list_olfr):
            di,n1i,n2i =dic_nneigh.get((o1,o2),[0,0,0])
            dj,n1j,n2j =dic_nneigh.get((o2,o1),[0,0,0])
            #if n1i>n1j:
            #    mat[io1,io2]=di
            #else:
            #    mat[io1,io2]=di
            mat[io1,io2]=di
    if min: mat_ = np.nanmin([mat,mat.T],0)
    mat_[np.isnan(mat_)]=0   
    return mat
def get_Xolfr(infos_dic,key):
    infos = infos_dic[key]
    Xs,olfrs = [],[]
    for info in tqdm(infos):
        [olfr,boundary,abs_pos,fov,MOE,zxyh_EGR1_semiloose,sublibrary,pix,volume,code,
        cor,cor_average,mean_traces,std_traces,mean_traces_bk,std_traces_bk] = info
        
        Xs.append(abs_pos)
        olfrs.append(olfr)
    Xs=np.array(Xs)
    olfrs=np.array(olfrs)
    
    OU,C = np.unique(olfrs,return_counts=True)
    iU = np.argsort(C)[::-1]
    OU,C = OU[iU],C[iU]
    return Xs,olfrs,OU,C
def plt_activity_obj(infos_dic,obj,nth=None,alpha=0.05):
    
    Xs = obj.Xs
    olfrs = obj.olfrs
    
    plt.figure(figsize=(12,12))
    plt.plot(Xs[:,0],Xs[:,1],'.',color='gray')
    if nth is None:
        nth = obj.nth
    keep = obj.nctsA>=nth
    plt.plot(Xs[keep,0],Xs[keep,1],'r.',alpha=alpha)
    plt.axis('equal')
    
def plt_activity(infos_dic,key,hth=2.25,nth=10,alpha=0.05):
    Xs,olfrs,OU,C = get_Xolfr(infos_dic,key)
    dic_frAf_,nctsA_ = get_dic_fr(infos_dic,keys=[key],hths=[hth],nths=[nth],return_cts=True)
    
    plt.figure(figsize=(12,12))
    plt.plot(Xs[:,0],Xs[:,1],'.',color='gray')
    keep = nctsA_>=nth
    plt.plot(Xs[keep,0],Xs[keep,1],'r.',alpha=alpha)
    plt.axis('equal')
    
    
all_data_folders = ['\\\\mendel\\Chromatin2\\Bogdan\\9_21_2020__OR-MER_lib1,2,3,4,5____CD1male_toB6maleMOE6_3col_set1',
 '\\\\mendel\\Chromatin4\\Bogdan\\9_21_2020__OR-MER_lib1,2,3,4,5____CD1male_toB6maleMOE6_3col_set2',
 '\\\\mendel\\Chromatin5\\Bogdan\\9_15_2020__OR-MER_lib6,7,8,9,10,11____CD1male_toB6femaleMOE3_3col_set1',
 '\\\\mendel\\Chromatin6\\Bogdan\\9_15_2020__OR-MER_lib6,7,8,9,10,11____CD1male_toB6femaleMOE3_3col_set2',
 '\\\\mendel\\Chromatin7\\Bogdan\\8_30_2020__OR-MER_lib6,7,8,9,10,11__CD1male_toB6maleMOE7_set1',
 '\\\\mendel\\Chromatin7\\Bogdan\\8_30_2020__OR-MER_lib6,7,8,9,10,11__CD1male_toB6maleMOE7_set2',
 '\\\\mendel\\N\\Bogdan\\8_27_2020__OR-MER_lib1,2,3,4,5__CD1male_toB6maleMOE7_set1',
 '\\\\mendel\\N\\Bogdan\\8_27_2020__OR-MER_lib1,2,3,4,5__CD1male_toB6maleMOE7_set2',
 '\\\\mendel\\N\\Bogdan\\8_30_2020__OR-MER_lib6,7,8,9,10,11__CD1male_toB6maleMOE7_set1',
 '\\\\mendel\\N\\Bogdan\\8_30_2020__OR-MER_lib6,7,8,9,10,11__CD1male_toB6maleMOE7_set2',
 '\\\\mendel\\DSB1\\Bogdan\\8_10_2020__OR-MER_lib1,2,3,4,5__CD1female_toB6maleMOE11',
 '\\\\mendel\\DSB1\\Bogdan\\8_11_2020__OR-MER_lib1,2,3,4__CD1female_toB6maleMOE11_set2',
 '\\\\mendel\\DSB2\\Bogdan\\8_11_2020__OR-MER_lib1,2,3,4__CD1female_toB6maleMOE11_set1',
 '\\\\mendel\\DSB2\\Bogdan\\8_11_2020__OR-MER_lib1,2,3,4__CD1female_toB6maleMOE11_set2',
 '\\\\mendel\\DSB3\\8_12_2020__OR-MER_lib5,6,7,8__CD1female_toB6maleMOE11_STORM6',
 '\\\\mendel\\Y\\Bogdan\\9_6_2020__OR-MER_lib6,7,8,9,10,11____CD1male_toB6femaleMOE4_set1',
 '\\\\mendel\\Y\\Bogdan\\9_6_2020__OR-MER_lib6,7,8,9,10,11____CD1male_toB6femaleMOE4_set2',
 '\\\\morgan\\MorganData\\Bogdan\\9_8_2020__OR-MER_lib1,2,3,4,5____CD1male_toB6femaleMOE3_set1',
 '\\\\morgan\\MorganData\\Bogdan\\9_8_2020__OR-MER_lib1,2,3,4,5____CD1male_toB6femaleMOE3_set2',
 '\\\\morgan\\MorganData2\\Bogdan\\9_3_2020__OR-MER_lib1,2,3,4,5__CD1male_toB6femaleMOE4_set1',
 '\\\\morgan\\MorganData2\\Bogdan\\9_3_2020__OR-MER_lib1,2,3,4,5__CD1male_toB6femaleMOE4_set2',
 '\\\\morgan\\MorganData2\\Bogdan\\9_8_2020__OR-MER_lib1,2,3,4,5____CD1male_toB6femaleMOE3_set1',
 '\\\\morgan\\MorganData2\\Bogdan\\9_8_2020__OR-MER_lib1,2,3,4,5____CD1male_toB6femaleMOE3_set2',
 '\\\\morgan\\TSTORMdata2\\Bogdan\\9_6_2020__OR-MER_lib6,7,8,9,10,11____CD1male_toB6femaleMOE4_set1',
 '\\\\morgan\\TSTORMdata2\\Bogdan\\9_6_2020__OR-MER_lib6,7,8,9,10,11____CD1male_toB6femaleMOE4_set2',
 '\\\\mcclintock\\mcclintock_1\\Bogdan\\10_10_2020__OR-MER_lib6,7,8,9,10,11____CD1male_toCAT2_3col_40xStorm3',
 '\\\\mcclintock\\mcclintock_1\\Bogdan\\10_12_2020__OR-MER_lib1,2,3,4,5____CD1mom_toB6pups_3col_40xStorm3_OB41',
 '\\\\mcclintock\\mcclintock_1\\Bogdan\\10_12_2020__OR-MER_lib1,2,3,4,5____CD1mom_toB6pups_3col_40xStorm3_OB41_OB',
 '\\\\mcclintock\\mcclintock_1\\Bogdan\\10_13_2020__OR-MER_lib6,7,8,9,10,11____CD1mom_toB6pups_3col_40xStorm65_OB43',
 '\\\\mcclintock\\mcclintock_1\\Bogdan\\10_13_2020__OR-MER_lib6,7,8,9,10,11____CD1mom_toB6pups_3col_40xStorm65_OB43_OB',
 '\\\\mcclintock\\mcclintock_1\\Bogdan\\10_18_2020__OR-MER_lib1,2,3,4,5____CD1virginFem2_toB6pups_3col_40xStorm3_OB37_MOE',
 '\\\\mcclintock\\mcclintock_1\\Bogdan\\10_18_2020__OR-MER_lib1,2,3,4,5____CD1virginFem2_toB6pups_3col_40xStorm3_OB37_OB',
 '\\\\mcclintock\\mcclintock_1\\Bogdan\\10_18_2020__OR-MER_lib6,7,8,9,10,11____CD1virginFem2_toB6pups_3col_40xStorm65_OB38_MOE',
 '\\\\mcclintock\\mcclintock_1\\Bogdan\\10_18_2020__OR-MER_lib6,7,8,9,10,11____CD1virginFem2_toB6pups_3col_40xStorm65_OB38_OB',
 '\\\\mcclintock\\mcclintock_1\\Bogdan\\10_1_2020__OR-MER_lib1,2,3,4,5____CD1female_toB6femaleMOE12_3col_set1',
 '\\\\mcclintock\\mcclintock_1\\Bogdan\\10_1_2020__OR-MER_lib1,2,3,4,5____CD1female_toB6femaleMOE12_3col_set2',
 '\\\\mcclintock\\mcclintock_1\\Bogdan\\10_4_2020__OR-MER_lib6,7,8,9,10,11____CD1female_toB6femaleMOE12_3col_set1',
 '\\\\mcclintock\\mcclintock_1\\Bogdan\\10_4_2020__OR-MER_lib6,7,8,9,10,11____CD1female_toB6femaleMOE12_3col_set2',
 '\\\\mcclintock\\mcclintock_1\\Bogdan\\10_7_2020__OR-MER_lib1,2,3,4,5____CD1male_toCAT2_3col_40xStorm3',
 '\\\\mcclintock\\mcclintock_1\\Bogdan\\9_23_2020__OR-MER_lib6,7,8,9,10,11____CD1male_toB6maleMOE6_3col_set1',
 '\\\\mcclintock\\mcclintock_1\\Bogdan\\9_23_2020__OR-MER_lib6,7,8,9,10,11____CD1male_toB6maleMOE6_3col_set2',
 '\\\\mcclintock\\mcclintock_1\\Bogdan\\9_26_2020__OR-MER_lib1,2,3,4,5____CD1male_toB6femaleMOE3_3col_set1',
 '\\\\mcclintock\\mcclintock_1\\Bogdan\\9_26_2020__OR-MER_lib1,2,3,4,5____CD1male_toB6femaleMOE3_3col_set2',
 '\\\\mcclintock\\mcclintock_1\\Bogdan\\9_28_2020__OR-MER_lib6,7,8,9,10,11____CD1male_toB6femaleMOE3_3col_set1',
 '\\\\mcclintock\\mcclintock_1\\Bogdan\\9_28_2020__OR-MER_lib6,7,8,9,10,11____CD1male_toB6femaleMOE3_3col_set2',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\10_18_2020__OR-MER_lib6,7,8,9,10,11____CD1virginFem2_toB6pups_3col_40xStorm65_OB38_MOE',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\10_18_2020__OR-MER_lib6,7,8,9,10,11____CD1virginFem2_toB6pups_3col_40xStorm65_OB38_OB',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\10_20_2020__OR-MER_lib1,2,3,4,5____CD1virginFem2_toB6pups_3col_40xStorm3_OB26_MOE',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\10_20_2020__OR-MER_lib1,2,3,4,5____CD1virginFem2_toB6pups_3col_40xStorm3_OB26_OB',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\10_21_2020__OR-MER_lib6,7,8,9,10,11____CD1virginFem2_toB6pups_3col_40xStorm65_OB27_part1_MOE',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\10_21_2020__OR-MER_lib6,7,8,9,10,11____CD1virginFem2_toB6pups_3col_40xStorm65_OB27_part1_OB',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\10_22_2020__OR-MER_lib1,2,3,4,5____CD1male_toCAT3_3col_40xStorm3',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\10_22_2020__OR-MER_lib6,7,8,9,10,11____CD1male_toCatMOE3_3col_40xStorm3',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\10_29_2020__OR-MER_lib1,2,3,4,5____CD1virginMaleControl_3col_40xStorm3_OBC_MOE',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\10_29_2020__OR-MER_lib1,2,3,4,5____CD1virginMaleControl_3col_40xStorm3_OBC_OB',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\10_29_2020__OR-MER_lib6,7,8,9,10,11____CD1virginMaleControl_3col_40xStorm65_OBC_MOE',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\10_29_2020__OR-MER_lib6,7,8,9,10,11____CD1virginMaleControl_3col_40xStorm65_OBC_OB',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\11_05_2020__OR-MER_lib1,2,3,4,5____CD1virginFemale1_toB6pupsl_3col_40xStorm3_OB42_MOE',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\11_05_2020__OR-MER_lib1,2,3,4,5____CD1virginFemale1_toB6pupsl_3col_40xStorm3_OB42_OB',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\11_09_2020__OR-MER_lib1,2,3,4,5____CD1virginFemale1_toB6pupsl_3col_40xStorm3_OB18_MOE',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\11_09_2020__OR-MER_lib1,2,3,4,5____CD1virginFemale1_toB6pupsl_3col_40xStorm3_OB18_OB',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\11_16_2020__OR-MER_lib1,2,3,4,5____CD1mom3_toB6pups_3col_40xStorm3_OB34_MOE',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\11_16_2020__OR-MER_lib1,2,3,4,5____CD1mom3_toB6pups_3col_40xStorm3_OB34_OB',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\11_16_2020__OR-MER_lib6,7,8,9,10,11____CD1mom3_toB6pups_3col_40xStorm65_OB36',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\11_16_2020__OR-MER_lib6,7,8,9,10,11____CD1mom3_toB6pups_3col_40xStorm65_OB36_MOE',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\11_19_2020__OR-MER_lib1,2,3,4,5____CD1mom3_toB6pups_3col_40xStorm3_OB22_MOE',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\11_19_2020__OR-MER_lib1,2,3,4,5____CD1mom3_toB6pups_3col_40xStorm3_OB22_OB',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\11_19_2020__OR-MER_lib6,7,8,9,10,11____CD1mom3_toB6pups_3col_40xStorm65_OB24',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\11_19_2020__OR-MER_lib6,7,8,9,10,11____CD1mom3_toB6pups_3col_40xStorm65_OB24_MOE',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\11_21_2020__OR-MER_lib1,2,3,4,5____CD1mom3_toB6pups_3col_40xStorm3_OB26_MOE',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\11_21_2020__OR-MER_lib1,2,3,4,5____CD1mom3_toB6pups_3col_40xStorm3_OB26_OB',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\11_21_2020__OR-MER_lib6,7,8,9,10,11____CD1mom3_toB6pups_3col_40xStorm65_OB27',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\11_21_2020__OR-MER_lib6,7,8,9,10,11____CD1mom3_toB6pups_3col_40xStorm65_OB27_MOE',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\11_5_2020__OR-MER_lib6,7,8,9,10,11____CD1virginFemale1_toB6pups_3col_40xStorm65_OB44',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\11_5_2020__OR-MER_lib6,7,8,9,10,11____CD1virginFemale1_toB6pups_3col_40xStorm65_OB44_MOE',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\11_9_2020__OR-MER_lib6,7,8,9,10,11____CD1virginFemale1_toB6pups_3col_40xStorm65_OB20',
 '\\\\mcclintock\\mcclintock_4\\Bogdan\\11_9_2020__OR-MER_lib6,7,8,9,10,11____CD1virginFemale1_toB6pups_3col_40xStorm65_OB20_MOE',
 '\\\\dolly\\Raw_data_3\\Bogdan\\8_10_2020__OR-MER_lib1,2,3,4,5__CD1female_toB6maleMOE11-Analysis',
 '\\\\dolly\\Raw_data_3\\Bogdan\\8_11_2020__OR-MER_lib1,2,3,4__CD1female_toB6maleMOE11_set1-Analysis',
 '\\\\dolly\\Raw_data_3\\Bogdan\\8_12_2020__OR-MER_lib5,6,7,8__CD1female_toB6maleMOE11_STORM6-Analysis',
 '\\\\dolly\\Raw_data_3\\Bogdan\\8_14_2020__OR-MER_lib9,10,11__CD1female_toB6maleMOE11_25x_20x',
 '\\\\dolly\\Raw_data_3\\Bogdan\\8_14_2020__OR-MER_lib9,10,11__CD1female_toB6maleMOE11_set1',
 '\\\\dolly\\Raw_data_3\\Bogdan\\8_14_2020__OR-MER_lib9,10,11__CD1female_toB6maleMOE11_set1-Analysis',
 '\\\\dolly\\Raw_data_3\\Bogdan\\8_14_2020__OR-MER_lib9,10,11__CD1female_toB6maleMOE11_set2',
 '\\\\dolly\\Raw_data_3\\Bogdan\\8_14_2020__OR-MER_lib9,10,11__CD1female_toB6maleMOE11_set2-Analysis',
 '\\\\dolly\\Raw_data_3\\Bogdan\\8_17_2020__OR-MER_lib1,2,3,4,5__CD1female_toB6maleMOE15_set1',
 '\\\\dolly\\Raw_data_3\\Bogdan\\8_17_2020__OR-MER_lib1,2,3,4,5__CD1female_toB6maleMOE15_set1-Analysis',
 '\\\\dolly\\Raw_data_3\\Bogdan\\8_17_2020__OR-MER_lib1,2,3,4,5__CD1female_toB6maleMOE15_set2',
 '\\\\dolly\\Raw_data_3\\Bogdan\\8_17_2020__OR-MER_lib1,2,3,4,5__CD1female_toB6maleMOE15_set2-Analysis',
 '\\\\dolly\\Raw_data_3\\Bogdan\\8_19_2020__OR-MER_lib6,7,8,9,10,11__CD1female_toB6maleMOE15_set1',
 '\\\\dolly\\Raw_data_3\\Bogdan\\8_19_2020__OR-MER_lib6,7,8,9,10,11__CD1female_toB6maleMOE15_set1-Analysis',
 '\\\\dolly\\Raw_data_3\\Bogdan\\8_19_2020__OR-MER_lib6,7,8,9,10,11__CD1female_toB6maleMOE15_set2',
 '\\\\dolly\\Raw_data_3\\Bogdan\\8_22_2020__OR-MER_lib1,2,3,4,5__CD1female_toB6femaleMOE13_set1',
 '\\\\dolly\\Raw_data_3\\Bogdan\\8_22_2020__OR-MER_lib1,2,3,4,5__CD1female_toB6femaleMOE13_set2',
 '\\\\dolly\\Raw_data_3\\Bogdan\\8_24_2020__OR-MER_lib6,7,8,9,10,11__CD1female_toB6femaleMOE13_set1',
 '\\\\dolly\\Raw_data_3\\Bogdan\\8_24_2020__OR-MER_lib6,7,8,9,10,11__CD1female_toB6femaleMOE13_set2',
 '\\\\dolly\\Raw_data_3\\Bogdan\\8_27_2020__OR-MER_lib1,2,3,4,5__CD1male_toB6maleMOE7_set1',
 '\\\\dolly\\Raw_data_3\\Bogdan\\8_27_2020__OR-MER_lib1,2,3,4,5__CD1male_toB6maleMOE7_set2',
 '\\\\dolly\\Raw_data_4\\Bogdan\\11_21_2020__OR-MER_lib1,2,3,4,5____CD1mom3_toB6pups_3col_40xStorm3_OB26_MOE',
 '\\\\dolly\\Raw_data_4\\Bogdan\\11_21_2020__OR-MER_lib1,2,3,4,5____CD1mom3_toB6pups_3col_40xStorm3_OB26_OB',
 '\\\\dolly\\Raw_data_4\\Bogdan\\11_27_2020__OR-MER_lib1,2,3,4,5____CD1mom3_toB6pups_3col_40xStorm3_OB29_MOE',
 '\\\\dolly\\Raw_data_4\\Bogdan\\11_27_2020__OR-MER_lib1,2,3,4,5____CD1mom3_toB6pups_3col_40xStorm3_OB29_OB',
 '\\\\dolly\\Raw_data_4\\Bogdan\\11_27_2020__OR-MER_lib6,7,8,9,10,11____CD1mom3_toB6pups_3col_40xStorm65_OB31',
 '\\\\dolly\\Raw_data_4\\Bogdan\\11_27_2020__OR-MER_lib6,7,8,9,10,11____CD1mom3_toB6pups_3col_40xStorm65_OB31_MOE',
 '\\\\dolly\\Raw_data_4\\Bogdan\\11_30_2020__OR-MER_lib1,2,3,4,5____CD1mom3_toB6pups_3col_40xStorm3_OB18_MOE',
 '\\\\dolly\\Raw_data_4\\Bogdan\\11_30_2020__OR-MER_lib1,2,3,4,5____CD1mom3_toB6pups_3col_40xStorm3_OB18_OB',
 '\\\\dolly\\Raw_data_4\\Bogdan\\11_30_2020__OR-MER_lib1,2,3,4,5____CD1mom3_toB6pups_3col_40xStorm3_OB18_OB-Analysis',
 '\\\\dolly\\Raw_data_4\\Bogdan\\11_30_2020__OR-MER_lib6,7,8,9,10,11____CD1mom3_toB6pups_3col_40xStorm65_OB20',
 '\\\\dolly\\Raw_data_4\\Bogdan\\11_30_2020__OR-MER_lib6,7,8,9,10,11____CD1mom3_toB6pups_3col_40xStorm65_OB20_MOE',
 '\\\\dolly\\Raw_data_4\\Bogdan\\12_10_2020__OR-MER_lib6,7,8,9,10,11____CD1mom3_toB6pups_3col_40xStorm65_OB8',
 '\\\\dolly\\Raw_data_4\\Bogdan\\12_10_2020__OR-MER_lib6,7,8,9,10,11____CD1mom3_toB6pups_3col_40xStorm65_OB8_MOE',
 '\\\\dolly\\Raw_data_4\\Bogdan\\12_10_2020__OR-MER_libClassI____CD1_JAR-2-STORM65-40X',
 '\\\\dolly\\Raw_data_4\\Bogdan\\12_11_2020__OR-MER_lib1,2,3,4,5____CD1mom3_toB6pups_3col_40xStorm3_OB6',
 '\\\\dolly\\Raw_data_4\\Bogdan\\12_24_2020__OR-MER_libClassI___CD1femaleOld-CO2_40XStorm3',
 '\\\\dolly\\Raw_data_4\\Bogdan\\12_4_2020__OR-MER_lib1,2,3,4,5____CD1mom3_toB6pups_3col_40xStorm3_OB14_MOE',
 '\\\\dolly\\Raw_data_4\\Bogdan\\12_4_2020__OR-MER_lib1,2,3,4,5____CD1mom3_toB6pups_3col_40xStorm3_OB14_OB',
 '\\\\dolly\\Raw_data_4\\Bogdan\\12_4_2020__OR-MER_lib6,7,8,9,10,11____CD1mom3_toB6pups_3col_40xStorm65_OB16',
 '\\\\dolly\\Raw_data_4\\Bogdan\\12_4_2020__OR-MER_lib6,7,8,9,10,11____CD1mom3_toB6pups_3col_40xStorm65_OB16_MOE',
 '\\\\dolly\\Raw_data_4\\Bogdan\\12_6_2020__OR-MER_lib1,2,3,4,5____CD1mom3_toB6pups_3col_40xStorm3_OB10',
 '\\\\dolly\\Raw_data_4\\Bogdan\\12_6_2020__OR-MER_lib6,7,8,9,10,11____CD1mom3_toB6pups_3col_40xStorm65_OB12',
 '\\\\dolly\\Raw_data_4\\Bogdan\\12_6_2020__OR-MER_lib6,7,8,9,10,11____CD1mom3_toB6pups_3col_40xStorm65_OB12_MOE',
 '\\\\dolly\\Raw_data_4\\Bogdan\\1_20_2020__OR-MER_lib1,2,3,4,5____CD1mom_to_B6pups-CO2_3col_40XStorm3',
 '\\\\dolly\\Raw_data_4\\Bogdan\\1_20_2020__OR-MER_lib7,8,9,10,11____CD1mom_to_B6pups-CO2_3col_40XStorm65',
 '\\\\dolly\\Raw_data_4\\Bogdan\\1_22_2020__OR-MER_lib1,2,3,4,5____CD1mom_to_B6pups_rep-CO2_3col_40XStorm65',
 '\\\\dolly\\Raw_data_4\\Bogdan\\1_22_2020__OR-MER_lib7,8,9,10,11____CD1mom_to_B6pups_rep-CO2_3col_40XStorm3']+\
 ['\\\\dolly\\Raw_data_2\\Bogdan\\12_01_2019__OR-MER_lib1,2__CD1female_toB6maleMOE15',
 '\\\\dolly\\Raw_data_2\\Bogdan\\12_06_2019__OR-MER_lib3,4__CD1female_toB6maleMOE15',
 '\\\\dolly\\Raw_data_2\\Bogdan\\12_07_2019__OR-MER_lib5,6__CD1female_toB6maleMOE15',
 '\\\\dolly\\Raw_data_2\\Bogdan\\12_08_2019__OR-MER_lib7,8__CD1female_toB6maleMOE15',
 '\\\\dolly\\Raw_data_2\\Bogdan\\12_10_2019__OR-MER_lib9,10__CD1female_toB6maleMOE15',
 '\\\\dolly\\Raw_data_2\\Bogdan\\1_18_2020__OR-MER_lib1,2__CD1female_toB6maleMOE15_rep',
 '\\\\dolly\\Raw_data_2\\Bogdan\\1_26_2020__OR-MER_lib1,2__CD1male_toB6femaleMOE4',
 '\\\\dolly\\Raw_data_2\\Bogdan\\1_27_2020__OR-MER_lib3,4__CD1male_toB6femaleMOE4',
 '\\\\dolly\\Raw_data_2\\Bogdan\\2_5_2020__OR-MER_lib1,2__CD1femaleMOM_toB6pupsNEWMOE6',
 '\\\\dolly\\Raw_data_2\\Bogdan\\2_6_2020__OR-MER_lib3,4__CD1femaleMOM_toB6pupsNEWMOE6',
 '\\\\dolly\\Raw_data_2\\Bogdan\\2_7_2020__OR-MER_lib5,6__CD1femaleMOM_toB6pupsNEWMOE6',
 '\\\\dolly\\Raw_data_2\\Bogdan\\2_8_2020__OR-MER_lib7,8__CD1femaleMOM_toB6pupsNEWMOE6',
 '\\\\dolly\\Raw_data_2\\Bogdan\\2_9_2020__OR-MER_lib9,10__CD1femaleMOM_toB6pupsNEWMOE6',
 '\\\\dolly\\Raw_data_2\\Bogdan\\3_4_2020__OR-MER_lib1,2__CD1femaleMOM_toB6pupsNEWMOE6rep',
 '\\\\dolly\\Raw_data_2\\Bogdan\\3_5_2020__OR-MER_lib3,4__CD1femaleMOM_toB6pupsNEWMOE6rep',
 '\\\\dolly\\Raw_data_2\\Bogdan\\6_14_2020__OR-MER_lib3,4__CD1maleFATHER_toB6pupsNEWMOE5rep',
 '\\\\dolly\\Raw_data_2\\Bogdan\\6_15_2020__OR-MER_lib5,6__CD1maleFATHER_toB6pupsNEWMOE5rep',
 '\\\\dolly\\Raw_data_2\\Bogdan\\6_16_2020__OR-MER_lib7,8__CD1maleFATHER_toB6pupsNEWMOE5rep',
 '\\\\dolly\\Raw_data_2\\Bogdan\\6_17_2020__OR-MER_lib9,10__CD1maleFATHER_toB6pupsNEWMOE5rep',
 '\\\\dolly\\Raw_data_2\\Bogdan\\6_18_2020__OR-MER_lib5,6__CD1male_toB6femaleMOE4_rep',
 '\\\\dolly\\Raw_data\\Bogdan\\7_1_2019__OR-MER_lib3,4__AcetoExp1',
 '\\\\dolly\\Raw_data\\Bogdan\\7_28_2019__OR-MER_lib3,4__CD1male_toB6female',
 '\\\\dolly\\Raw_data\\Bogdan\\7_29_2019__OR-MER_lib1,2__CD1male_toB6female',
 '\\\\dolly\\Raw_data\\Bogdan\\7_2_2019__OR-MER_lib7,8__AcetoExp1',
 '\\\\dolly\\Raw_data\\Bogdan\\7_30_2019__OR-MER_lib5,6__CD1male_toB6female',
 '\\\\dolly\\Raw_data\\Bogdan\\7_31_2019__OR-MER_lib7,8__CD1male_toB6female',
 '\\\\dolly\\Raw_data\\Bogdan\\7_3_2019__OR-MER_lib1,2__AcetoExp1',
 '\\\\dolly\\Raw_data\\Bogdan\\7_3_2019__OR-MER_lib9,10__AcetoExp1',
 '\\\\dolly\\Raw_data\\Bogdan\\7_4_2019__OR-MER_lib5,6__AcetoExp1',
 '\\\\dolly\\Raw_data\\Bogdan\\8_01_2019__OR-MER_lib9,10__CD1male_toB6female',
 '\\\\dolly\\Raw_data\\Bogdan\\8_07_2019__OR-MER_lib9,10__CD1femalemom_toPups',
 '\\\\dolly\\Raw_data\\Bogdan\\8_08_2019__OR-MER_lib5,6__CD1femalemom_toPups',
 '\\\\dolly\\Raw_data\\Bogdan\\8_09_2019__OR-MER_lib7,8__CD1femalemom_toPups',
 '\\\\dolly\\Raw_data\\Bogdan\\8_10_2019__OR-MER_lib3,4__CD1femalemom_toPups',
 '\\\\dolly\\Raw_data\\Bogdan\\8_13_2019__OR-MER_lib1,2__CD1femalemom_toPups',
 '\\\\dolly\\Raw_data\\Bogdan\\9_01_2019__OR-MER_lib1,2__CD1female_toB6male',
 '\\\\dolly\\Raw_data\\Bogdan\\9_02_2019__OR-MER_lib3,4__CD1female_toB6male',
 '\\\\dolly\\Raw_data\\Bogdan\\9_05_2019__OR-MER_lib5,6__CD1female_toB6male',
 '\\\\dolly\\Raw_data\\Bogdan\\9_06_2019__OR-MER_lib7,8__CD1female_toB6male',
 '\\\\dolly\\Raw_data\\Bogdan\\9_07_2019__OR-MER_lib9,10__CD1female_toB6male',
 '\\\\dolly\\Analysis\\Bogdan\\10_04_2019__OR-MER_lib9,10__CD1male_toB6male',
 '\\\\dolly\\Analysis\\Bogdan\\10_05_2019__OR-MER_lib3,4__CD1male_toB6male',
 '\\\\dolly\\Analysis\\Bogdan\\10_06_2019__OR-MER_lib5,6__CD1male_toB6male',
 '\\\\dolly\\Analysis\\Bogdan\\10_07_2019__OR-MER_lib7,8__CD1male_toB6male',
 '\\\\dolly\\Analysis\\Bogdan\\10_09_2019__OR-MER_lib1,2__CD1male_toB6male',
 '\\\\dolly\\Analysis\\Bogdan\\6_18_2020__OR-MER_lib5,6__CD1male_toB6femaleMOE4_rep',
 '\\\\dolly\\Analysis\\Bogdan\\6_19_2020__OR-MER_lib1,2__CD1maleFATHER_toB6pupsNEWMOE5rep',
 '\\\\dolly\\Analysis\\Bogdan\\6_20_2020__OR-MER_lib1,2__CD1male_toB6femaleMOE4_rep',
 '\\\\dolly\\Analysis\\Bogdan\\6_21_2020__OR-MER_lib3,4__CD1male_toB6femaleMOE4_rep',
 '\\\\dolly\\Analysis\\Bogdan\\6_22_2020__OR-MER_lib9,10__CD1male_toB6femaleMOE4_rep',
 '\\\\dolly\\Analysis\\Bogdan\\6_23_2020__OR-MER_lib7,8__CD1male_toB6femaleMOE4_rep',
 '\\\\dolly\\Analysis\\Bogdan\\6_27_2020__OR-MER_lib5,8__CD1maleFATHER_toB6pupsNEWMOE5rep',
 '\\\\dolly\\Analysis\\Bogdan\\6_29_2020__OR-MER_lib6,10__CD1maleFATHER_toB6pupsNEWMOE5rep',
 '\\\\dolly\\Analysis\\Bogdan\\6_30_2020__OR-MER_lib3,4__CD1maleFATHER_toB6pupsNEWMOE5rep',
 '\\\\dolly\\Analysis\\Bogdan\\7_04_2020__OR-MER_lib9,10__CD1DfemaleMOM_toB6pupsNEWMOE2',
 '\\\\dolly\\Analysis\\Bogdan\\7_04_2020__OR-MER_lib9,10__CD1DfemaleMOM_toB6pupsNEWMOE2_20x',
 '\\\\dolly\\Analysis\\Bogdan\\7_06_2020__OR-MER_libB32-p1__CD1DfemaleMOM_toB6pupsNEWMOE2',
 '\\\\dolly\\Analysis\\Bogdan\\9_22_2019__OR-MER_lib1,2__CD1female_toB6female',
 '\\\\dolly\\Analysis\\Bogdan\\9_23_2019__OR-MER_lib7,8__CD1female_toB6female',
 '\\\\dolly\\Analysis\\Bogdan\\9_25_2019__OR-MER_lib5,6__CD1female_toB6female',
 '\\\\dolly\\Analysis\\Bogdan\\9_26_2019__OR-MER_lib3,4__CD1female_toB6female',
 '\\\\dolly\\Analysis\\Bogdan\\9_27_2019__OR-MER_lib9,10__CD1female_toB6female']
th_dic={'mouse1_sample1': 2.5,
 'mouse1_sample2': 2.5,
 'mouse2_sample1': 2.75,
 'mouse2_sample2': 2.75,
 'mouse3_sample1': 2.65,
 'mouse3_sample2': 2.75,
 'mouse4_sample1': 2.25,
 'mouse4_sample2': 2.5,
 'mouse4_sample3': 2.75,
 'mouse5_sample1': 2,
 'mouse5_sample2': 2,
 'mouse6_sample1': 2,
 'mouse6_sample2': 2.5,
 'mouse7_sample1': 2.75,
 'mouse7_sample2': 2.75,
 'mouse8_sample1': 2.75,
 'mouse8_sample2': 2.75,
 'mouse9_sample1': 2.5,
 'mouse9_sample2': 2.5,
 'mouse10_sample1': 2.5,
 'mouse10_sample2': 2.5}
sample_dic = {'mouse1_sample1': '9_3_2020__OR-MER_lib1,2,3,4,5__CD1male_toB6femaleMOE4',
 'mouse1_sample2': '9_6_2020__OR-MER_lib6,7,8,9,10,11____CD1male_toB6femaleMOE4',
 'mouse2_sample1': '9_26_2020__OR-MER_lib1,2,3,4,5____CD1male_toB6femaleMOE3_3col',
 'mouse2_sample2': '9_28_2020__OR-MER_lib6,7,8,9,10,11____CD1male_toB6femaleMOE3_3col',
 'mouse3_sample1': '8_17_2020__OR-MER_lib1,2,3,4,5__CD1female_toB6maleMOE15',
 'mouse3_sample2': '8_19_2020__OR-MER_lib6,7,8,9,10,11__CD1female_toB6maleMOE15',
 'mouse4_sample1': '8_10_2020__OR-MER_lib1,2,3,4,5__CD1female_toB6maleMOE11',
 'mouse4_sample2': '8_12_2020__OR-MER_lib5,6,7,8__CD1female_toB6maleMOE11_STORM6',
 'mouse4_sample3': '8_14_2020__OR-MER_lib9,10,11__CD1female_toB6maleMOE11',
 'mouse5_sample1': '10_7_2020__OR-MER_lib1,2,3,4,5____CD1male_toCAT2_3col_40xStorm3',
 'mouse5_sample2': '10_10_2020__OR-MER_lib6,7,8,9,10,11____CD1male_toCAT2_3col_40xStorm3',
 'mouse6_sample1': '10_22_2020__OR-MER_lib1,2,3,4,5____CD1male_toCAT3_3col_40xStorm3',
 'mouse6_sample2': '10_22_2020__OR-MER_lib6,7,8,9,10,11____CD1male_toCatMOE3_3col_40xStorm3',
 'mouse7_sample1': '8_27_2020__OR-MER_lib1,2,3,4,5__CD1male_toB6maleMOE7',
 'mouse7_sample2': '8_30_2020__OR-MER_lib6,7,8,9,10,11__CD1male_toB6maleMOE7',
 'mouse8_sample1': '9_21_2020__OR-MER_lib1,2,3,4,5____CD1male_toB6maleMOE6_3col',
 'mouse8_sample2': '9_23_2020__OR-MER_lib6,7,8,9,10,11____CD1male_toB6maleMOE6_3col',
 'mouse9_sample1': '8_22_2020__OR-MER_lib1,2,3,4,5__CD1female_toB6femaleMOE13',
 'mouse9_sample2': '8_24_2020__OR-MER_lib6,7,8,9,10,11__CD1female_toB6femaleMOE13',
 'mouse10_sample1': '10_1_2020__OR-MER_lib1,2,3,4,5____CD1female_toB6femaleMOE12_3col',
 'mouse10_sample2': '10_4_2020__OR-MER_lib6,7,8,9,10,11____CD1female_toB6femaleMOE12_3col'}
sample_dic.update({'mouse11_sample1':'7_29_2019__OR-MER_lib1,2__CD1male_toB6female',
                   'mouse11_sample2':'7_28_2019__OR-MER_lib3,4__CD1male_toB6female',
                   'mouse11_sample3':'7_30_2019__OR-MER_lib5,6__CD1male_toB6female',
                   'mouse11_sample4':'7_31_2019__OR-MER_lib7,8__CD1male_toB6female',
                   'mouse11_sample5':'8_01_2019__OR-MER_lib9,10__CD1male_toB6female',
                   
                   'mouse12_sample1':'6_20_2020__OR-MER_lib1,2__CD1male_toB6femaleMOE4_rep',
                   'mouse12_sample2':'6_21_2020__OR-MER_lib3,4__CD1male_toB6femaleMOE4_rep',
                   'mouse12_sample3':'6_18_2020__OR-MER_lib5,6__CD1male_toB6femaleMOE4_rep',
                   'mouse12_sample4':'6_23_2020__OR-MER_lib7,8__CD1male_toB6femaleMOE4_rep',
                   'mouse12_sample5':'6_22_2020__OR-MER_lib9,10__CD1male_toB6femaleMOE4_rep'})