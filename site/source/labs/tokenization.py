"""确定性的玩具 BPE：保留空白，不跨空白合并；不是现代 tokenizer 的完整实现。"""
from collections import Counter
import json

def pieces(text):
    # 空白本身保留为 token；非空白之间才允许 merge。
    return list(text)

def apply_merge(tokens,pair):
    result=[]; i=0
    while i<len(tokens):
        if i+1<len(tokens) and (tokens[i],tokens[i+1])==pair:
            result.append(tokens[i]+tokens[i+1]); i+=2
        else: result.append(tokens[i]); i+=1
    return result

def learn_bpe(documents,num_merges=4):
    rows=[pieces(s) for s in documents]; merges=[]; trace=[]
    for _ in range(num_merges):
        counts=Counter((a,b) for row in rows for a,b in zip(row,row[1:])
                       if not any(c.isspace() for c in a+b))
        if not counts: break
        pair=min(counts,key=lambda p:(-counts[p],p))
        merges.append(pair); trace.append({'pair':list(pair),'count':counts[pair]})
        rows=[apply_merge(row,pair) for row in rows]
    return merges,rows,trace

def encode(text,merges):
    row=pieces(text)
    for pair in merges: row=apply_merge(row,pair)
    return row

def demonstration():
    training=['low low lower','new newer low']; merges,rows,trace=learn_bpe(training)
    return {'training':training,'trace':trace,'training_tokens':rows,'test':'lowest new',
            'test_tokens':encode('lowest new',merges),'whitespace':'空格作为字符 token 保留，merge 不跨空白；未见字符保留为字符。',
            'limitation':'这是字符起点的局部 BPE 过程，不含字节编码、特殊 token、预切词正则或生产词表。'}
if __name__=='__main__': print(json.dumps(demonstration(),ensure_ascii=False,indent=2))
