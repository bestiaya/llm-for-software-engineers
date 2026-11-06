"""二维→2 个 ReLU 隐藏单元→3 类：手算反传、数值梯度、autograd 与训练。"""
import copy
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from .train import seed_everything

# PyTorch Linear 的 [out,in] 存储口径；样本是行向量，前向用 x @ W.T + b。
INITIAL = {'W1': [[.2,.1],[.1,.3]], 'b1': [.1,.2],
           'W2': [[.1,.2],[.3,-.1],[-.2,.4]], 'b2': [0.,.1,-.1]}


def forward_math(parameters, x, target):
    """标准库前向；用于独立数值差分，不借助 autograd。"""
    a=[sum(w*v for w,v in zip(row,x))+b for row,b in zip(parameters['W1'],parameters['b1'])]
    h=[max(0.,v) for v in a]
    z=[sum(w*v for w,v in zip(row,h))+b for row,b in zip(parameters['W2'],parameters['b2'])]
    m=max(z); total=sum(math.exp(v-m) for v in z)
    p=[math.exp(v-m)/total for v in z]
    return {'pre_relu':a,'hidden':h,'logits':z,'probabilities':p,
            'loss':m+math.log(total)-z[target]}


def analytic_gradients(parameters, x, target):
    trace=forward_math(parameters,x,target)
    dz=[p-(i==target) for i,p in enumerate(trace['probabilities'])]
    dh=[sum(dz[k]*parameters['W2'][k][j] for k in range(3)) for j in range(2)]
    da=[d if a>0 else 0. for d,a in zip(dh,trace['pre_relu'])]
    return {'W1':[[d*v for v in x] for d in da], 'b1':da,
            'W2':[[d*h for h in trace['hidden']] for d in dz], 'b2':dz}


def finite_difference(parameters,x,target,eps=1e-5):
    result=copy.deepcopy(parameters)
    for key,value in parameters.items():
        for i,row in enumerate(value):
            columns=range(len(row)) if isinstance(row,list) else [None]
            for j in columns:
                plus,minus=copy.deepcopy(parameters),copy.deepcopy(parameters)
                if j is None:
                    plus[key][i]+=eps; minus[key][i]-=eps
                else:
                    plus[key][i][j]+=eps; minus[key][i][j]-=eps
                gradient=(forward_math(plus,x,target)['loss']-forward_math(minus,x,target)['loss'])/(2*eps)
                if j is None: result[key][i]=gradient
                else: result[key][i][j]=gradient
    return result


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.first=nn.Linear(2,2,dtype=torch.float64)
        self.second=nn.Linear(2,3,dtype=torch.float64)
        with torch.no_grad():
            for parameter,key in [(self.first.weight,'W1'),(self.first.bias,'b1'),
                                  (self.second.weight,'W2'),(self.second.bias,'b2')]:
                parameter.copy_(torch.tensor(INITIAL[key],dtype=torch.float64))
    def forward(self,x):
        return self.second(F.relu(self.first(x)))
    def named_matrices(self):
        return {'W1':self.first.weight,'b1':self.first.bias,'W2':self.second.weight,'b2':self.second.bias}


def max_difference(left,right):
    return max(float((torch.tensor(left[k],dtype=torch.float64)-torch.tensor(right[k],dtype=torch.float64)).abs().max()) for k in left)


def gradient_demo():
    x=[1.,2.]; target=1; lr=.1
    analytic=analytic_gradients(INITIAL,x,target)
    numeric=finite_difference(INITIAL,x,target)
    model=TinyMLP(); features=torch.tensor([x],dtype=torch.float64)
    loss=F.cross_entropy(model(features),torch.tensor([target])); loss.backward()
    automatic={k:p.grad.tolist() for k,p in model.named_matrices().items()}
    before=forward_math(INITIAL,x,target)
    optimizer=torch.optim.SGD(model.parameters(),lr=lr); optimizer.step()
    after_parameters={k:p.detach().tolist() for k,p in model.named_matrices().items()}
    return {'input':x,'target':target,'initial_parameters':INITIAL,'before':before,
            'analytic_gradients':analytic,'finite_difference_gradients':numeric,'autograd_gradients':automatic,
            'max_difference':{'analytic_vs_finite_difference':max_difference(analytic,numeric),
                              'analytic_vs_autograd':max_difference(analytic,automatic),
                              'finite_difference_vs_autograd':max_difference(numeric,automatic)},
            'learning_rate':lr,'updated_parameters':after_parameters,'after':forward_math(after_parameters,x,target)}


def text_windows(documents):
    """字符编码 A=0/B=1；每个长为2的窗口分为含0/1/2个B三类，不预测下一个字符。"""
    inputs=[]; targets=[]; labels=[]
    for text in documents:
        if any(c not in 'AB' for c in text): raise ValueError('本实验仅接受 A/B 字符')
        for i in range(len(text)-1):
            window=text[i:i+2]; values=[float(c=='B') for c in window]
            inputs.append(values); targets.append(int(sum(values))); labels.append(window)
    if not inputs: raise ValueError('没有长为2的文本窗口')
    return torch.tensor(inputs,dtype=torch.float64),torch.tensor(targets),labels


@torch.no_grad()
def classification_score(model,x,y):
    logits=model(x)
    return {'ce':float(F.cross_entropy(logits,y)), 'accuracy':float((logits.argmax(-1)==y).double().mean()),
            'window_count':len(y)}


def train_demo(steps=300):
    if steps<1: raise ValueError('steps 必须为正')
    seed_everything(42); model=TinyMLP()
    train_documents=['AABABB','BBAABA','ABBAAA']
    evaluation_documents=['BBABAA','ABAABB']
    x,y,_=text_windows(train_documents); test_x,test_y,labels=text_windows(evaluation_documents)
    before=classification_score(model,x,y)
    initial={k:p.detach().clone() for k,p in model.named_matrices().items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=.03)
    for _ in range(steps):
        optimizer.zero_grad(); loss=F.cross_entropy(model(x),y); loss.backward(); optimizer.step()
    return {'task':'将2字符窗口分类为含0/1/2个B，标签由人工规则给出；不是 next-token LM',
            'encoding':{'A':0,'B':1}, 'train_documents':train_documents,'evaluation_documents':evaluation_documents,
            'steps':steps,'optimizer':'Adam','learning_rate':.03,'seed':42,
            'train_before':before,'train_after':classification_score(model,x,y),
            'evaluation':classification_score(model,test_x,test_y),
            'evaluation_windows':labels,'evaluation_labels':test_y.tolist(),
            'evaluation_predictions':model(test_x).argmax(-1).tolist(),
            'parameter_tensors_changed':{k:not torch.equal(initial[k],p) for k,p in model.named_matrices().items()},
            'limitation':'完整文档分开，但两边共享AA/AB/BA/BB全部4种窗口；这是优化闭环演示，不是独立泛化证据。'}


def demonstration():
    return {'gradient_check':gradient_demo(),'text_window_classifier':train_demo(),
            'boundary':'全连接小网络，15个参数；固定单样本 ReLU 在正区间，中心差分不跨折点。没有 attention、位置向量或完整语言模型训练。'}

if __name__=='__main__':
    print(json.dumps(demonstration(),ensure_ascii=False,indent=2,allow_nan=False))
