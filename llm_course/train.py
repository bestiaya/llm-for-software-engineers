"""CPU 完整参数训练：validation 选 checkpoint，最后才评 test。"""
import argparse
import copy
import json
import platform
import random
import time
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from .data import Counts, prepare, windows
from .model import Config, Decoder, generate

def seed_everything(seed):
    random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)

@torch.no_grad()
def evaluate(model,rows):
    model.eval(); total=0.; count=0
    for x,y in rows:
        logits,_=model(torch.tensor([x],dtype=torch.long))
        total+=float(F.cross_entropy(logits.reshape(-1,model.config.vocab_size),torch.tensor(y),reduction='sum'))
        count+=len(y)
    if not count: raise ValueError('没有评价目标')
    return {'ce':total/count,'nll_sum':total,'target_count':count}

def load_checkpoint(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=Decoder(Config(**saved['config']))
    model.load_state_dict(saved['state_dict'],strict=True); model.eval()
    return model,saved['tokens']

def run(args):
    if min(args.steps,args.batch_size,args.eval_every)<1 or args.lr<=0 or args.max_new_tokens<0:
        raise ValueError('steps/batch-size/eval-every/lr 必须为正；生成预算不能为负')
    out=Path(args.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('输出路径已存在且不是空目录；为保留已有报告和模型，请换一个新的 --out 路径。')
    started=time.perf_counter(); seed_everything(args.seed)
    splits,tok,source=prepare(args.dataset,args.seed)
    rows={key:windows(value,tok,args.context) for key,value in splits.items()}
    config=Config(len(tok.tokens),args.d_model,args.heads,args.layers,args.context)
    model=Decoder(config)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    buckets=defaultdict(list)
    for row in rows['train']: buckets[len(row[0])].append(row)
    rng=random.Random(args.seed); history=[]
    best_loss=float('inf'); best=None; best_step=0
    for step in range(1,args.steps+1):
        model.train()
        # 按训练窗口频率选长度桶；桶内有放回采样，无 PAD，无丢弃短尾。
        length=len(rng.choice(rows['train'])[0])
        batch=rng.choices(buckets[length],k=args.batch_size)
        x=torch.tensor([r[0] for r in batch]); y=torch.tensor([r[1] for r in batch])
        optimizer.zero_grad(set_to_none=True)
        logits,_=model(x)
        loss=F.cross_entropy(logits.reshape(-1,config.vocab_size),y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        if step%args.eval_every==0 or step==args.steps:
            validation=evaluate(model,rows['validation'])
            history.append({'step':step,'batch_train_ce':float(loss.detach()),'validation_ce':validation['ce']})
            if validation['ce']<best_loss:
                best_loss=validation['ce']; best=copy.deepcopy(model.state_dict()); best_step=step
    changed={k:bool(not torch.equal(v,initial[k])) for k,v in model.state_dict().items()}
    model.load_state_dict(best); model.eval()
    # test 从未用于上面的参数更新或 checkpoint 选择。
    selected_validation=evaluate(model,rows['validation'])
    test=evaluate(model,rows['test'])
    baselines=Counts(splits['train'],tok)
    prompt_text=args.prompt or splits['train'][0][:3]
    prefix=tok.encode(prompt_text)
    sample=generate(model,prefix,args.max_new_tokens)
    sample['text']=tok.decode(sample['ids'])
    sample.update(prompt_text=prompt_text,prompt_ids=prefix,prompt_decoded=tok.decode(prefix),
                  generated_ids=sample['ids'][len(prefix):],generated_text=tok.decode(sample['ids'][len(prefix):]))
    report={'schema':1,'dataset':args.dataset,'seed':args.seed,'config':config.to_dict(),
            'training':{'steps':args.steps,'batch_size':args.batch_size,'lr':args.lr,'optimizer':'AdamW',
                        'weight_decay':0.01,'gradient_clip':1.0,'eval_every':args.eval_every,
                        'length_buckets':'window-frequency length selection; replacement within bucket'},
            'parameter_count':sum(p.numel() for p in model.parameters()),'source':source,
            'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu','threads':1},
            'splits':{k:{'groups':len(splits[k]),'raw_characters':sum(map(len,splits[k])),
                        'windows':len(rows[k]),'targets':sum(len(y) for _,y in rows[k]),
                        'unknown_characters':sum(tok.encode(g).count(0) for g in splits[k])} for k in splits},
            'tokens':tok.tokens,'history':history,'selected_step':best_step,'validation':selected_validation,
            'test':{'model':test,'unigram':baselines.score(rows['test'],1),'bigram':baselines.score(rows['test'],2)},
            'parameter_tensors_changed_at_final_step':changed,'sample':sample,
            'duration_seconds':time.perf_counter()-started,
            'limitations':['有限数据和单 seed；不证明语言理解或工程应用有效。','toy 完整组不同，但共享语法；real 相邻切分仍可能有重复表达。',
                           '开发期间可见的固定 test 不是永远未触碰的盲测；不要据此反复调参。',
                           '计数 baseline 仅使用前一个 token 或全局频率；与 decoder 的上下文长度不同。',
                           '无 PAD；字符而非 BPE 训练；无跨窗口记忆；绝对位置按窗口重置。',
                           '生成使用贪心，EOS/context/token budget 停止；不保证语义、正确性或安全性。']}
    out.mkdir(parents=True,exist_ok=True)
    torch.save({'config':config.to_dict(),'tokens':tok.tokens,'state_dict':best},out/'checkpoint.pt')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (out/'sample.txt').write_text(sample['text']+'\n停止原因：'+sample['stop_reason']+'\n')
    lines=['# 真实运行报告','',f"dataset={args.dataset}，seed={args.seed}，参数={report['parameter_count']}，选择第 {best_step} 步。",'',
           '| 模型 | test 每目标 CE | 目标数 |','|---|---:|---:|']
    for name,score in report['test'].items(): lines.append(f"| {name} | {score['ce']:.6f} | {score['target_count']} |")
    lines+=['',f"validation CE={best_loss:.6f}；总耗时 {report['duration_seconds']:.3f} 秒。",'',
            '## 配置与资料', '',
            f"d_model={config.d_model}，heads={config.heads}，layers={config.layers}，context={config.context}，vocab={config.vocab_size}。",
            f"AdamW，lr={args.lr}，weight_decay=0.01，梯度裁剪=1.0，batch_size={args.batch_size}，steps={args.steps}，每 {args.eval_every} 步及最后一步检查 validation。",
            'CPU 单线程；窗口频率选长度桶，桶内有放回采样；无 PAD。',
            source['source'],f"语料 SHA-256：{source['sha256']}",'',
            '| 分区 | 组数 | 原字符数 | 窗口数 | 目标数 | 未知字符数 |',
            '|---|---:|---:|---:|---:|---:|']
    for name,counts in report['splits'].items():
        lines.append(f"| {name} | {counts['groups']} | {counts['raw_characters']} | {counts['windows']} | {counts['targets']} | {counts['unknown_characters']} |")
    lines+=['','模型按 validation 最低 CE 选定后才评价 test；完整训练轨迹与环境版本见 report.json。',
            '', '## 生成', '', '**输入提示**', '```text',sample['prompt_text'],'```','**模型新增内容**','```text',sample['generated_text'],'```',f"停止原因：{sample['stop_reason']}",'', '## 限制', '']
    lines+=['- '+s for s in report['limitations']]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return report

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['toy','real'],default='toy'); p.add_argument('--steps',type=int,default=200)
    p.add_argument('--out',default='outputs/toy'); p.add_argument('--seed',type=int,default=42)
    for name,default in [('d-model',32),('heads',4),('layers',2),('context',64),('batch-size',16),('eval-every',25),('max-new-tokens',80)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--lr',type=float,default=.003); p.add_argument('--prompt',default='')
    return p

def main():
    p=parser(); args=p.parse_args()
    try: report=run(args)
    except (ValueError,FileNotFoundError) as error: p.error(str(error))
    print(json.dumps({'out':args.out,'selected_step':report['selected_step'],'test':report['test'],'sample':report['sample']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
