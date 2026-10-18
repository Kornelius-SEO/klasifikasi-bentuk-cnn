"""Synthetic shape classification with a small NumPy CNN."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

LABELS = ['lingkaran', 'persegi', 'segitiga']
SIZE = 16

def make_image(label, rng):
    im = Image.new('L', (64, 64), 0)
    d = ImageDraw.Draw(im)
    x, y = int(rng.integers(12, 24)), int(rng.integers(12, 24))
    w, h = int(rng.integers(26, 39)), int(rng.integers(26, 39))
    if label == 0: d.ellipse((x,y,x+w,y+h), fill=255)
    elif label == 1: d.rectangle((x,y,x+w,y+h), fill=255)
    else: d.polygon([(x+w//2,y),(x+w,y+h),(x,y+h)], fill=255)
    im = im.rotate(float(rng.uniform(-12,12)), resample=Image.Resampling.BILINEAR)
    im = im.filter(ImageFilter.GaussianBlur(float(rng.uniform(0,0.8))))
    a = np.asarray(im.resize((SIZE,SIZE), Image.Resampling.LANCZOS), dtype=np.float32)/255
    a = np.clip(a + rng.normal(0,0.025,a.shape),0,1)
    return a

def dataset(seed, n_per_class):
    rng=np.random.default_rng(seed)
    xs=[]; ys=[]
    for label in range(3):
        for _ in range(n_per_class): xs.append(make_image(label,rng)); ys.append(label)
    order=rng.permutation(len(ys))
    return np.array(xs,dtype=np.float32)[order], np.array(ys)[order]

def patches(x):
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(x,(3,3),axis=(1,2)).reshape(len(x),196,9)

class CNN:
    def __init__(self, filters=8, dropout=0.0, seed=1):
        rng=np.random.default_rng(seed)
        self.wc=(rng.normal(0,0.22,(9,filters))).astype('f')
        self.bc=np.zeros(filters,dtype='f')
        self.wd=(rng.normal(0,0.15,(filters,3))).astype('f')
        self.bd=np.zeros(3,dtype='f')
        self.dropout=dropout; self.rng=rng
    def forward(self,x,training=False):
        p=patches(x); z=p@self.wc+self.bc
        a=np.maximum(z,0)
        # Global average pooling keeps the model compact and translation tolerant.
        pooled=a.mean(axis=1)
        mask=np.ones_like(pooled)
        if training and self.dropout:
            mask=(self.rng.random(pooled.shape)>=self.dropout)/(1-self.dropout)
        h=pooled*mask
        logits=h@self.wd+self.bd
        e=np.exp(logits-logits.max(axis=1,keepdims=True)); probs=e/e.sum(axis=1,keepdims=True)
        return probs,(p,z,pooled,mask,h)
    def train_epoch(self,x,y,lr=0.05,batch=32):
        order=self.rng.permutation(len(y)); losses=[]
        for ids in np.array_split(order,max(1,int(np.ceil(len(y)/batch)))):
            xb=x[ids]; yb=y[ids]; n=len(ids)
            prob,(p,z,pooled,mask,h)=self.forward(xb,True)
            losses.append(-np.log(np.maximum(prob[np.arange(n),yb],1e-9)).mean())
            dl=prob.copy();dl[np.arange(n),yb]-=1;dl/=n
            gwd=h.T@dl; gbd=dl.sum(axis=0)
            dh=(dl@self.wd.T)*mask
            dz=(dh[:,None,:]/196)*(z>0)
            gwc=np.einsum('npi,npf->if',p,dz); gbc=dz.sum(axis=(0,1))
            self.wd-=lr*gwd;self.bd-=lr*gbd
            self.wc-=lr*gwc;self.bc-=lr*gbc
        return float(np.mean(losses))
    def predict(self,x): return self.forward(x)[0]
    def save(self,path): np.savez(path,wc=self.wc,bc=self.bc,wd=self.wd,bd=self.bd)
    @classmethod
    def load(cls,path):
        z=np.load(path); m=cls(z['wc'].shape[1]);
        for k in ['wc','bc','wd','bd']: setattr(m,k,z[k])
        return m

def preprocess_image(im):
    im=im.convert('L').resize((SIZE,SIZE),Image.Resampling.LANCZOS)
    a=np.asarray(im,dtype=np.float32)/255
    # Accept either bright shape on dark background or the reverse.
    if a.mean()>0.5: a=1-a
    return a[None,:,:]
