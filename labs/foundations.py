"""计数、交叉熵、数值梯度、一步更新与链式法则；仅用标准库。"""
import argparse
import json
import math
from collections import Counter

def softmax(z):
    m=max(z); ex=[math.exp(x-m) for x in z]; total=sum(ex)
    return [x/total for x in ex]

def cross_entropy(z,target):
    # log-sum-exp 避免极小概率 log(0)。
    m=max(z)
    return m+math.log(sum(math.exp(x-m) for x in z))-z[target]

def finite_difference(z,target,eps=1e-5):
    result=[]
    for i in range(len(z)):
        plus=list(z); minus=list(z); plus[i]+=eps; minus[i]-=eps
        result.append((cross_entropy(plus,target)-cross_entropy(minus,target))/(2*eps))
    return result

def counting_probabilities(strings,vocabulary,previous,alpha=1.):
    counts=Counter()
    for s in strings:
        counts.update(b for a,b in zip(s,s[1:]) if a==previous)
    denominator=sum(counts.values())+alpha*len(vocabulary)
    return {s:(counts[s]+alpha)/denominator for s in vocabulary}

def demonstration():
    z=[0.,0.,0.]; target=1; p=softmax(z)
    gradient=[x-(i==target) for i,x in enumerate(p)]
    updated=[x-.3*g for x,g in zip(z,gradient)]
    # x=2,w=3,b=1 -> u=wx+b=7 -> L=u²=49。
    chain={'x':2,'w':3,'b':1,'u':7,'loss':49,'dL_du':14,'du_dw':2,'dL_dw':28,
           'numeric_dL_dw':(((3+1e-5)*2+1)**2-((3-1e-5)*2+1)**2)/(2e-5)}
    return {'counting_A':counting_probabilities(['AB','AB','AC'],['A','B','C'],'A'),
            'logits':z,'probabilities':p,'ce':cross_entropy(z,target),'finite_difference':finite_difference(z,target),
            'analytic_gradient':gradient,'learning_rate':.3,'updated_logits':updated,
            'updated_probabilities':softmax(updated),'updated_ce':cross_entropy(updated,target),'chain_rule':chain,
            'boundary':'手设的三维 logits 和标量链式法则演示；不等于已训练完整语言模型。'}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--json',action='store_true'); args=p.parse_args()
    result=demonstration()
    if args.json: print(json.dumps(result,ensure_ascii=False,indent=2))
    else:
        for key,value in result.items(): print(f'{key}: {value}')
