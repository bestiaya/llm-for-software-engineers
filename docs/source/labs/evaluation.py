"""固定公开教学资料的检索与证据评分；不调用或假装运行 LLM。"""
import json

DOCUMENTS={
    'timeout':'请求超时后最多重试2次；每次重试间隔1秒。',
    'backup':'系统每天凌晨2点备份，保留7天。',
    'deploy':'发布前必须通过自动测试，失败时回滚到上一个版本。',
}
CASES=[{'query':'超时后重试几次？','relevant':{'timeout'},'claim':'最多重试2次','citation':'timeout'},
       {'query':'备份保留几天？','relevant':{'backup'},'claim':'保留7天','citation':'backup'},
       {'query':'测试失败如何发布？','relevant':{'deploy'},'claim':'失败时回滚','citation':'deploy'}]

def retrieve(query,k=1):
    if k<1: raise ValueError('k 必须为正')
    terms=set(query)-set('？。，！的后前如何几多少')
    return sorted(DOCUMENTS,key=lambda key:(-len(terms & set(DOCUMENTS[key])),key))[:k]

def evidence_supported(claim,citation):
    # 可检验的连续原文包含关系，不是语义蕴含判定。
    return bool(claim) and citation in DOCUMENTS and claim in DOCUMENTS[citation]

def evaluate(cases=CASES):
    if not cases: raise ValueError('评价集不能为空')
    rows=[]
    for case in cases:
        hits=retrieve(case['query'])
        rows.append({'query':case['query'],'retrieved':hits,
                     'recall_at_1':len(set(hits)&case['relevant'])/len(case['relevant']),
                     'exact_quote_supported':evidence_supported(case['claim'],case['citation'])})
    return {'cases':rows,'mean_recall_at_1':sum(r['recall_at_1'] for r in rows)/len(rows),
            'quote_support_rate':sum(r['exact_quote_supported'] for r in rows)/len(rows),
            'unsupported_example':{'claim':'超时后无限重试','citation':'timeout','supported':evidence_supported('无限重试','timeout')},
            'limitation':'作者自编3条固定文档及预写 claim，无 LLM 生成。原文包含不证明回答全面、正确或系统泛化。'}
if __name__=='__main__': print(json.dumps(evaluate(),ensure_ascii=False,indent=2))
