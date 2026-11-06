"""Pre-LN decoder，绝对位置，显式缓存与上下文边界。"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn

@dataclass(frozen=True)
class Config:
    vocab_size: int
    d_model: int = 32
    heads: int = 4
    layers: int = 2
    context: int = 64
    def __post_init__(self):
        if min(self.vocab_size, self.d_model, self.heads, self.layers, self.context) < 1:
            raise ValueError('配置值必须为正整数')
        if self.d_model % self.heads:
            raise ValueError('d_model 必须整除 heads')
    def to_dict(self):
        return asdict(self)

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.width = c.heads, c.d_model // c.heads
        self.qkv = nn.Linear(c.d_model, 3*c.d_model)
        self.proj = nn.Linear(c.d_model, c.d_model)
    def forward(self, x, past=None):
        b, t, d = x.shape
        q, k, v = [a.reshape(b,t,self.heads,self.width).transpose(1,2)
                   for a in self.qkv(x).chunk(3, dim=-1)]
        offset = 0 if past is None else past[0].shape[2]
        if past is not None:
            k = torch.cat((past[0],k),dim=2)
            v = torch.cat((past[1],v),dim=2)
        scores = q @ k.transpose(-2,-1) / math.sqrt(self.width)
        allowed = torch.arange(k.shape[2],device=x.device)[None,:] <= (offset+torch.arange(t,device=x.device))[:,None]
        weights = scores.masked_fill(~allowed, float('-inf')).softmax(dim=-1)
        y = (weights @ v).transpose(1,2).contiguous().reshape(b,t,d)
        return self.proj(y), (k,v)

class Block(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.attn = Attention(c)
        self.mlp = nn.Sequential(nn.Linear(c.d_model,4*c.d_model),nn.GELU(),nn.Linear(4*c.d_model,c.d_model))
    def forward(self,x,past=None):
        a,cache = self.attn(self.ln1(x),past)
        x = x+a
        return x+self.mlp(self.ln2(x)),cache

class Decoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size,config.d_model)
        self.position = nn.Embedding(config.context,config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model,config.vocab_size,bias=False)
    def forward(self,ids,cache=None):
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError('ids 必须为非空 [batch,time]')
        offset = 0
        if cache is not None:
            if self.training or torch.is_grad_enabled():
                raise ValueError('缓存仅允许 eval + no_grad 推理')
            if len(cache) != len(self.blocks):
                raise ValueError('缓存层数不符')
            offset = cache[0][0].shape[2]
            for k,v in cache:
                expected=(ids.shape[0],self.config.heads,offset,self.config.d_model//self.config.heads)
                if tuple(k.shape)!=expected or tuple(v.shape)!=expected:
                    raise ValueError('缓存形状不符')
        if offset+ids.shape[1] > self.config.context:
            raise ValueError('超出 context；请显式重置缓存并开始新序列，不自动裁剪或重编号')
        x = self.token(ids)+self.position(torch.arange(offset,offset+ids.shape[1],device=ids.device))
        result=[]
        for i,block in enumerate(self.blocks):
            x,c = block(x,None if cache is None else cache[i]); result.append(c)
        return self.head(self.norm(x)),tuple(result)

@torch.no_grad()
def generate(model,prefix,max_new_tokens=80,eos_id=1):
    if max_new_tokens < 0 or not prefix:
        raise ValueError('需要非空 prefix 和非负生成预算')
    model.eval()
    ids=list(prefix)
    if len(ids)>model.config.context:
        raise ValueError('prompt 超出 context')
    if ids[-1]==eos_id:
        return {'ids':ids,'stop_reason':'eos'}
    logits,cache=model(torch.tensor([ids],dtype=torch.long,device=next(model.parameters()).device))
    for _ in range(max_new_tokens):
        if len(ids)>=model.config.context:
            return {'ids':ids,'stop_reason':'context_limit'}
        nxt=int(logits[0,-1].argmax()); ids.append(nxt)
        if nxt==eos_id:
            return {'ids':ids,'stop_reason':'eos'}
        logits,cache=model(torch.tensor([[nxt]],device=logits.device),cache)
    return {'ids':ids,'stop_reason':'token_budget'}
