import numpy as np
import sys
def peakdet(v, delta, x = None):

    maxtab = []
    mintab = []
       
    if x is None:
        x = np.arange(len(v))
    
    v = np.asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    
    lookformax = True
    i=0
    while (i < len(v)):
        
        
        this = v[i]
        if this > mx:
            
            mx = this
            mxpos = x[i]
            
        if this < mn:
            mn = this
            mnpos = x[i]
            
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                
                lookformax = True
        i=i+1
    maxtab = np.array(maxtab)

    maxtab_n=[]
    for i in range(len(maxtab)):
        flag=1
        mxpos=maxtab[i,0]
        mx=maxtab[i,1]
        if i>0:
            if mxpos-maxtab[i-1,0]<100:
                flag=0
        if (flag==1):
            for j in range(i,len(maxtab)):
                if (maxtab[j,0] - maxtab[i,0])<100:
                    if (maxtab[j,1] > mx):
                        mxpos=maxtab[j,0]
                        mx=maxtab[j,1]
            maxtab_n.append((mxpos,mx))
                
    maxtab_n=np.array(maxtab_n)  

    return np.array(maxtab_n)
