"""Train/evaluate a variable-horizon action-conditioned failure model."""
from __future__ import annotations
import argparse, json, random, re
from pathlib import Path
import h5py, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

HORIZONS=(1,2,4,8,16,32,64)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_files(root, task_filter=None):
    rows=[]
    for p in sorted(root.glob('*.hdf5')):
        m=re.search(r'task=(\d+).*success=(True|False)',p.name)
        if not m: continue
        task=int(m.group(1)); success=m.group(2)=='True'
        if task_filter is not None and task != task_filter: continue
        with h5py.File(p,'r') as f:
            rows.append({'path':str(p),'task':task,'label':float(not success),
                         'obs':np.asarray(f['proprio'],np.float32),'act':np.asarray(f['actions'],np.float32)})
    if not rows: raise RuntimeError(f'no HDF5 files in {root}')
    return rows

def split_rollouts(rows,seed):
    rng=np.random.default_rng(seed); out={'train':[],'val':[],'test':[]}
    for task in sorted({r['task'] for r in rows}):
        for label in (0.,1.):
            group=[r for r in rows if r['task']==task and r['label']==label]; rng.shuffle(group)
            n=len(group); nv=max(1,round(.2*n)) if n>=3 else 0; nt=max(1,round(.2*n)) if n>=3 else 0
            out['val']+=group[:nv]; out['test']+=group[nv:nv+nt]; out['train']+=group[nv+nt:]
    return out

def normalize(splits):
    obs=np.concatenate([r['obs'] for r in splits['train']]); act=np.concatenate([r['act'] for r in splits['train']])
    om,os=obs.mean(0),obs.std(0); am,ass=act.mean(0),act.std(0); os[os<1e-6]=1; ass[ass<1e-6]=1
    for part in splits.values():
        for r in part: r['obs']=(r['obs']-om)/os; r['act']=(r['act']-am)/ass

class Windows(Dataset):
    def __init__(self,rows,horizon=None,shuffle_actions=False):
        self.rows=rows; self.horizon=horizon; self.shuffle_actions=shuffle_actions; self.idx=[]
        # Keep the same anchor timestamps for every evaluation horizon.
        maxh=max(HORIZONS)
        for i,r in enumerate(rows):
            for t in range(2,len(r['obs'])-maxh): self.idx.append((i,t))
    def __len__(self): return len(self.idx)
    def __getitem__(self,j):
        i,t=self.idx[j]; r=self.rows[i]; h=self.horizon or random.choice(HORIZONS)
        actions=r['act'][t:t+h]
        if self.shuffle_actions: actions=actions[np.random.permutation(len(actions))]
        padded=np.zeros((max(HORIZONS),actions.shape[1]),np.float32); padded[:h]=actions
        return (torch.from_numpy(r['obs'][t-2:t+1]),torch.from_numpy(padded),h,r['task'],
                torch.tensor(r['label'],dtype=torch.float32),torch.from_numpy(r['obs'][t+h]))

class Model(nn.Module):
    def __init__(self,obs_dim=9,act_dim=7,latent=64):
        super().__init__(); self.obs=nn.Sequential(nn.Flatten(),nn.Linear(obs_dim*3,128),nn.ReLU(),nn.Linear(128,latent))
        self.action=nn.GRU(act_dim,64,batch_first=True); self.task=nn.Embedding(10,16); self.horizon=nn.Embedding(65,16)
        self.fuse=nn.Sequential(nn.Linear(latent+64+16+16,128),nn.ReLU(),nn.Linear(128,latent),nn.ReLU())
        self.future=nn.Linear(latent,obs_dim); self.failure=nn.Linear(latent,1)
    def forward(self,o,a,h,task):
        z=self.obs(o); lengths=h.cpu(); packed=nn.utils.rnn.pack_padded_sequence(a,lengths,batch_first=True,enforce_sorted=False)
        _,ah=self.action(packed); x=torch.cat([z,ah[-1],self.task(task),self.horizon(h)],-1); z=self.fuse(x)
        return self.future(z),self.failure(z).squeeze(-1)

def auc(labels,scores):
    labels=np.asarray(labels); scores=np.asarray(scores); pos=scores[labels==1]; neg=scores[labels==0]
    if not len(pos) or not len(neg): return float('nan')
    return float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg)))

def evaluate(model,rows,device,batch,shuffle_actions=False):
    result={}
    model.eval()
    for h in HORIZONS:
        ys=[]; ss=[]; residual=[]
        dl=DataLoader(Windows(rows,h,shuffle_actions),batch_size=batch,num_workers=4)
        with torch.no_grad():
            for o,a,hh,t,y,target in dl:
                pred,logit=model(o.to(device),a.to(device),hh.to(device),t.to(device)); ys.extend(y.numpy()); ss.extend(logit.cpu().numpy()); residual.extend(((pred-target.to(device))**2).mean(1).cpu().numpy())
        ya=np.asarray(ys); sa=np.asarray(ss); ra=np.asarray(residual)
        result[str(h)]={'roc_auc':auc(ya,sa),'logit_gap':float(sa[ya==1].mean()-sa[ya==0].mean()),
                        'residual_gap':float(ra[ya==1].mean()-ra[ya==0].mean()),'n':len(ys)}
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task',type=int); p.add_argument('--seed',type=int,default=0); p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=1024); p.add_argument('--shuffle-actions',action='store_true'); args=p.parse_args()
    seed_all(args.seed); rows=load_files(args.data,args.task); splits=split_rollouts(rows,args.seed); normalize(splits); device='cuda' if torch.cuda.is_available() else 'cpu'
    model=Model().to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    pos=sum(r['label'] for r in splits['train']); neg=len(splits['train'])-pos; bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/max(pos,1),device=device))
    train=Windows(splits['train'],shuffle_actions=args.shuffle_actions); dl=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=4,pin_memory=True)
    history=[]
    for epoch in range(args.epochs):
        model.train(); total=0.; count=0
        for o,a,h,t,y,target in dl:
            o,a,h,t,y,target=[x.to(device,non_blocking=True) for x in (o,a,h,t,y,target)]
            pred,logit=model(o,a,h,t); loss=torch.nn.functional.smooth_l1_loss(pred,target)+bce(logit,y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); total+=float(loss)*len(y); count+=len(y)
        history.append(total/count); print(f'epoch={epoch+1} loss={history[-1]:.5f}',flush=True)
    result={'seed':args.seed,'task':args.task,'shuffle_actions':args.shuffle_actions,'rollouts':{k:len(v) for k,v in splits.items()},
            'epochs':args.epochs,'history':history,'val':evaluate(model,splits['val'],device,args.batch_size,args.shuffle_actions),
            'test':evaluate(model,splits['test'],device,args.batch_size,args.shuffle_actions)}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)); torch.save(model.state_dict(),args.output.with_suffix('.pt'))
    print(json.dumps(result['test'],indent=2))
if __name__=='__main__': main()
