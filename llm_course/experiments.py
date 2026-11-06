"""可复查的因果/缓存、SFT loss mask 和局部 LoRA 实验。"""
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model import Config, Decoder
from .train import seed_everything

def sft_targets(sequence,answer_start):
    """answer_start 是原始序列中首个答案 token 的下标；EOS 可属于答案。"""
    if not 1<=answer_start<len(sequence): raise ValueError('需要 prompt 与至少一个答案 token')
    return sequence[:-1],sequence[1:],[i>=answer_start for i in range(1,len(sequence))]

def masked_ce(logits,targets,mask):
    losses=F.cross_entropy(logits,targets,reduction='none')
    if not bool(mask.any()): raise ValueError('没有纳入 loss 的答案目标')
    return losses[mask].mean()

class LoRALinear(nn.Module):
    def __init__(self,input_size,output_size,rank=2,alpha=2):
        super().__init__()
        if rank<1: raise ValueError('rank 必须为正')
        self.base=nn.Linear(input_size,output_size,bias=False)
        self.base.requires_grad_(False)
        self.A=nn.Parameter(torch.randn(rank,input_size)*.1)
        self.B=nn.Parameter(torch.zeros(output_size,rank))
        self.scale=alpha/rank
    def forward(self,x):
        return self.base(x)+F.linear(F.linear(x,self.A),self.B)*self.scale

def demonstration():
    seed_everything(42); model=Decoder(Config(11,16,4,2,16)).eval()
    ids=torch.tensor([[2,3,4,5,6]])
    with torch.no_grad():
        full,_=model(ids); changed=ids.clone(); changed[0,-1]=7
        other,_=model(changed)
        prefix_error=float((full[:,:-1]-other[:,:-1]).abs().max())
        cache=None; incremental=[]
        for i in range(ids.shape[1]):
            out,cache=model(ids[:,i:i+1],cache); incremental.append(out)
        cache_error=float((full-torch.cat(incremental,dim=1)).abs().max())
    x,y,mask=sft_targets([2,3,4,5,1],3)
    logits=torch.zeros(len(y),6)
    sft_loss=float(masked_ce(logits,torch.tensor(y),torch.tensor(mask)))
    adapter=LoRALinear(3,2); before={k:v.detach().clone() for k,v in adapter.state_dict().items()}
    optimizer=torch.optim.SGD([p for p in adapter.parameters() if p.requires_grad],lr=.2)
    features=torch.tensor([[1.,2.,3.],[-1.,0.,2.]])
    for _ in range(5):
        optimizer.zero_grad(); loss=F.mse_loss(adapter(features),torch.tensor([[1.,0.],[0.,1.]])); loss.backward(); optimizer.step()
    changes={k:not torch.equal(before[k],v) for k,v in adapter.state_dict().items()}
    assert prefix_error<1e-6 and cache_error<2e-6
    assert not changes['base.weight'] and changes['A'] and changes['B']
    return {'causal_prefix_max_error':prefix_error,'cache_max_error':cache_error,
            'sft':{'sequence':[2,3,4,5,1],'input':x,'shifted_targets':y,'answer_start':3,'loss_mask':mask,
                   'target_count':sum(mask),'uniform_logits_vocab':6,'loss':sft_loss,'expected_loss':math.log(6)},
            'lora':{'parameter_changed':changes,'final_mse':float(loss.detach()),'rank':2,'alpha':2,'steps':5},
            'limitation':'SFT 是移位和 loss mask 算术，LoRA 是独立局部矩阵更新；未运行完整模型指令微调或对齐。'}
if __name__=='__main__': print(json.dumps(demonstration(),ensure_ascii=False,indent=2))
