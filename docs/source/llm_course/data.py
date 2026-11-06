"""先划分原始资料，再拟合词表与构建移位目标。"""
import hashlib
import itertools
import random
from collections import Counter, defaultdict
from pathlib import Path
import math

class Tokenizer:
    def __init__(self,chars):
        self.tokens=['<UNK>','<EOS>']+sorted(set(chars))
        self.index={s:i for i,s in enumerate(self.tokens)}
    def encode(self,text):
        return [self.index.get(c,0) for c in text]
    def decode(self,ids):
        return ''.join(self.tokens[i] for i in ids)

def toy_groups():
    """原创有限语法：完整序列唯一，但各组共享模板，非自然语言。"""
    return [''.join('服务正常。' if x==0 else '服务超时，重试。' for x in pattern)
            for n in range(2,8) for pattern in itertools.product((0,1),repeat=n)]

def prepare(dataset,seed=42,data_dir=None):
    root=Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[1]/'data'
    if dataset=='toy':
        groups=toy_groups(); random.Random(seed).shuffle(groups)
        a,b=int(.8*len(groups)),int(.9*len(groups))
        splits=dict(zip(('train','validation','test'),(groups[:a],groups[a:b],groups[b:])))
        source='作者原创有限语法；完整文本组不相交，模板共享'
        raw='\n'.join(toy_groups()).encode()
    elif dataset=='real':
        path=root/'real.txt'
        if not path.is_file():
            raise FileNotFoundError(f'缺少 {path}：请先按 data 中来源与许可说明提供合法真实文本；不会用 toy 冒充 real。')
        raw=path.read_bytes(); text=raw.decode('utf-8')
        a,b=int(.8*len(text)),int(.9*len(text))
        splits=dict(zip(('train','validation','test'),([text[:a]],[text[a:b]],[text[b:]])))
        if min(len(s[0]) for s in splits.values())<3:
            raise ValueError('真实文本过短，三个字符分区均至少需要 3 字符')
        source='data/real.txt，连续原始字符 80/10/10 切分，窗口不跨分区'
    else:
        raise ValueError('dataset 仅允许 toy/real')
    tokenizer=Tokenizer(''.join(splits['train']))
    return splits,tokenizer,{'source':source,'sha256':hashlib.sha256(raw).hexdigest()}

def windows(groups,tokenizer,context):
    if context<1: raise ValueError('context 必须为正')
    rows=[]
    for group in groups:
        ids=tokenizer.encode(group)+[1]
        # stride=context：每个组内从第二字符到 EOS 恰好一个目标。
        for start in range(0,len(ids)-1,context):
            part=ids[start:start+context+1]
            rows.append((part[:-1],part[1:]))
    return rows

class Counts:
    def __init__(self,train_groups,tokenizer,alpha=1.):
        if alpha<=0: raise ValueError('alpha 必须为正')
        self.vocab_size=len(tokenizer.tokens); self.alpha=alpha
        self.unigram=Counter(); self.bigram=defaultdict(Counter)
        for group in train_groups:
            ids=tokenizer.encode(group)+[1]
            self.unigram.update(ids[1:])
            for a,b in zip(ids,ids[1:]): self.bigram[a][b]+=1
    def score(self,rows,order=1):
        loss=0.; count=0
        for x,y in rows:
            for a,b in zip(x,y):
                counts=self.unigram if order==1 else self.bigram.get(a,{})
                p=(counts.get(b,0)+self.alpha)/(sum(counts.values())+self.alpha*self.vocab_size)
                loss-=math.log(p); count+=1
        if count==0: raise ValueError('没有评价目标')
        return {'ce':loss/count,'nll_sum':loss,'target_count':count}
