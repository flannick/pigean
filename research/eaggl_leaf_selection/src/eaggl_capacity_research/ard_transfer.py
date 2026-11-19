"""Research-only factor contribution matching for ARD transfer validation."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from eaggl_mechanism_research.objectives import union_predict
MATCH_COLUMNS=['factor_A','factor_B','cosine','amplitude_A_to_B','amplitude_B_to_A','stable','shape_pass']
def stable_match(cosine,forward,reverse):
 return bool(cosine>=.9 and .8<=forward<=1.25 and .8<=reverse<=1.25)
def state(W,b0):
 W=np.asarray(W,dtype=float);n=len(W);tri=np.triu_indices(n,1);E=(W*W).sum(0);active=np.flatnonzero(E>b0);P=union_predict(W);np.fill_diagonal(P,0);Q=np.empty((len(tri[0]),len(active)),np.float32)
 for j,k in enumerate(active):
  x=W[tri[0],k]*W[tri[1],k];q=np.divide(x*(1-P[tri]),1-x,out=np.zeros_like(x),where=1-x>1e-14);bad=1-x<=1e-14
  if bad.any():
   keep=np.arange(W.shape[1])!=k;q[bad]=x[bad]*np.prod(1-W[tri[0][bad]][:,keep]*W[tri[1][bad]][:,keep],axis=1)
  Q[:,j]=q
 norms=np.sqrt(np.einsum('ij,ij->j',Q,Q,dtype=np.float64));Q/=np.maximum(norms,1e-30)
 return dict(W=W,P=P,Q=Q,norm=norms,active=active,E=E,tri=tri)
def match(A,B):
 if not len(A['active']) or not len(B['active']):return pd.DataFrame(columns=MATCH_COLUMNS)
 cosine=A['Q'].T@B['Q'];ii,jj=linear_sum_assignment(-cosine);rows=[]
 for i,j in zip(ii,jj):
  c=float(np.clip(cosine[i,j],-1,1));ab=c*float(B['norm'][j]/max(A['norm'][i],1e-30));ba=c*float(A['norm'][i]/max(B['norm'][j],1e-30));rows.append(dict(factor_A=int(A['active'][i]+1),factor_B=int(B['active'][j]+1),cosine=c,amplitude_A_to_B=ab,amplitude_B_to_A=ba,stable=stable_match(c,ab,ba),shape_pass=bool(c>=.9)))
 return pd.DataFrame(rows,columns=MATCH_COLUMNS)
def structural(S):
 V=S['W'][:,S['active']];e=(V*V).sum(0)
 if not len(e):return dict(retained=0,unused=S['W'].shape[1],q90_membership=0.,max_support=0.,median_support=0.,anchors=0,near_duplicates=0,prediction_mass=0.)
 support=V.sum(0)**2/e;membership=V.sum(1)**2/np.maximum((V*V).sum(1),1e-30);unit=V/np.sqrt(e);cos=unit.T@unit;np.fill_diagonal(cos,0)
 return dict(retained=len(e),unused=S['W'].shape[1]-len(e),q90_membership=float(np.quantile(membership,.9)),max_support=float(support.max()),median_support=float(np.median(support)),anchors=int((V.max(0)**2/e>.5).sum()),near_duplicates=int((cos>=.95).sum()//2),prediction_mass=float(S['P'][S['tri']].sum()))
def panel(A,B):
 m=match(A,B);stable=m[m.stable.astype(bool)].copy()
 if not len(stable):return m,pd.DataFrame(columns=['reference_partition','reference_factor','pair_id','weak','strong','mean_energy'])
 stable['mean_energy']=[float((A['E'][int(r.factor_A)-1]+B['E'][int(r.factor_B)-1])/2) for r in stable.itertuples()];stable=stable.sort_values(['mean_energy','factor_A','factor_B']).reset_index(drop=True);nw=int(np.ceil(.25*len(stable)));ns=min(10,len(stable));rows=[]
 for i,r in stable.iterrows():
  for side in [0,1]:rows.append(dict(reference_partition=side,reference_factor=int(r['factor_A' if side==0 else 'factor_B']),pair_id=i,weak=i<nw,strong=i>=len(stable)-ns,mean_energy=float(r.mean_energy)))
 return m,pd.DataFrame(rows)
