"""Build a fully offline course site from Markdown and generated reports."""
from __future__ import annotations
import argparse
import html
import json
import re
import shutil
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parents[1]
FILES = ['01-models','02-math','03-counting','04-parameters','05-gradients','06-backprop','07-training','08-tokenization','09-context','10-attention','11-position-heads','12-decoder','13-next-token','14-toy-training','15-real-text','16-post-training','17-inference','18-evaluation']
GROUPS = [('必要基础',FILES[:3]),('神经网络',FILES[3:7]),('文本与上下文',FILES[7:9]),('Transformer',FILES[9:12]),('训练与生成',FILES[12:15]),('工程与验收',FILES[15:])]
EXTRA_CSS = '''.bookmark{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:20px 0}.book-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}.book-group{border:1px solid var(--line);border-radius:10px;padding:22px;background:white}.book-group h2{border:0;margin:0 0 16px;padding:0;font-size:23px}.book-group li{margin:10px 0}.book-group ol{padding-left:22px}.book-home{max-width:1180px;margin:auto;padding:0 24px 50px}.chapter-links{display:flex;justify-content:space-between;gap:16px;margin:30px 0}.primary-link{display:inline-block;background:var(--accent);color:white;padding:9px 16px;text-decoration:none;border-radius:7px}.progress-note{font-size:14px;color:var(--muted)}.book-toc{border-bottom:1px solid var(--line);margin-bottom:14px;padding-bottom:14px}.book-toc a{padding:8px}.hero-facts{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}.hero-facts span{background:var(--soft);border:1px solid var(--line);border-radius:6px;padding:5px 10px;font-size:14px}.lead{font-size:21px}.book-intro{padding:26px;background:white;border:1px solid var(--line);border-radius:12px;margin:22px 0}.book-intro h2{margin-top:0;border:0;padding:0}@media(max-width:850px){.book-grid{grid-template-columns:1fr}.book-home{padding:0 14px 30px}.chapter-links{font-size:14px}.lead{font-size:18px}}'''
NAV_JS = '''(()=>{const KEY='llm-engineers-read-v1';let read=[];try{read=JSON.parse(localStorage.getItem(KEY)||'[]');if(!Array.isArray(read))read=[]}catch{}const current=document.body.dataset.chapter,check=document.getElementById('mark-read');if(check){check.checked=read.includes(current);check.addEventListener('change',()=>{read=read.filter(x=>x!==current);if(check.checked)read.push(current);try{localStorage.setItem(KEY,JSON.stringify(read))}catch{document.getElementById('progress-message').textContent='当前浏览器不能保存记录；正文仍可正常阅读。'}})}const count=document.getElementById('read-count');if(count)count.textContent=read.filter(x=>/^\\d\\d-/.test(x)).length+' / 18';const hs=[...document.querySelectorAll('main h2[id],main h3[id]')],as=[...document.querySelectorAll('nav .toc a')];let scheduled=false;function update(){scheduled=false;let cur=hs[0];for(const h of hs)if(h.getBoundingClientRect().top<150)cur=h;for(const a of as){if(cur&&decodeURIComponent(a.hash.slice(1))===cur.id){a.setAttribute('aria-current','location');const n=a.closest('nav'),b=n.getBoundingClientRect(),v=a.getBoundingClientRect();if(innerWidth>850&&(v.top<b.top||v.bottom>b.bottom))a.scrollIntoView({block:'nearest'})}else a.removeAttribute('aria-current')}}addEventListener('scroll',()=>{if(!scheduled){scheduled=true;requestAnimationFrame(update)}},{passive:true});update()})();'''

def title_of(source):
    return re.search(r'^# (.+)',source,re.M).group(1)

def convert(source):
    md=markdown.Markdown(extensions=['extra','toc'],extension_configs={'toc':{'toc_depth':'2-3'}})
    body=md.convert(source)
    body=re.sub(r'<h1[^>]*>.*?</h1>','',body,count=1,flags=re.S)
    body=body.replace('<svg','<div class="figure"><svg').replace('</svg>','</svg></div>')
    body=body.replace('<table>','<div class="table-wrap"><table>').replace('</table>','</table></div>')
    for name in FILES:body=body.replace('href="'+name+'.md','href="'+name+'.html').replace('href="../chapters/'+name+'.md','href="'+name+'.html')
    body=body.replace('href="../reports/toy.md','href="reports-toy.html').replace('href="../reports/real.md','href="reports-real.html')
    body=body.replace('href="../SETUP.md','href="setup.html')
    for folder in ['labs','llm_course','data','reports']:
        body=body.replace('href="../'+folder+'/','href="source/'+folder+'/')
    body=body.replace('href="../site/','href="')
    for name in ['toy','real']:body=body.replace('href="'+name+'.json','href="source/reports/'+name+'.json')
    body=body.replace('href="reports/toy.md','href="reports-toy.html').replace('href="reports/real.md','href="reports-real.html')
    return body,md.toc

def shell(title,content,css,js,chapter=''):
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="面向软件工程师的中文大语言模型课程：必要基础、神经网络、Transformer与可复现实验。"><title>'+html.escape(title)+' · 给软件工程师的大语言模型课</title><style>'+css+'</style></head><body data-chapter="'+chapter+'">'+content+'<script>'+js+'</script></body></html>'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');args=ap.parse_args()
    missing=[f for f in FILES if not (ROOT/'chapters'/f'{f}.md').exists()]
    if missing and not args.preview:raise SystemExit('Missing chapters: '+', '.join(missing))
    out=ROOT/'site';out.mkdir(exist_ok=True)
    titles={f:title_of((ROOT/'chapters'/f'{f}.md').read_text()) for f in FILES if f not in missing}
    css=(ROOT/'tools/theme.css').read_text()+EXTRA_CSS
    for i,name in enumerate(FILES):
        if name in missing:continue
        source=(ROOT/'chapters'/f'{name}.md').read_text();body,toc=convert(source)
        prev=FILES[i-1] if i else None;next_=FILES[i+1] if i+1<len(FILES) else None
        links='<div class="chapter-links">'+(f'<a href="{prev}.html">← 上一章</a>' if prev and prev in titles else '<a href="index.html">← 课程首页</a>')+(f'<a href="{next_}.html">下一章 →</a>' if next_ and next_ in titles else '<a href="index.html">返回课程路线 →</a>')+'</div>'
        header='<header><div class="eyebrow">LLM FOR SOFTWARE ENGINEERS · '+str(i+1).zfill(2)+' / 18</div><h1>'+html.escape(titles[name])+'</h1><a href="index.html">课程路线</a> · <a href="source/chapters/'+name+'.md">阅读源文</a> · <a href="setup.html">环境与运行</a></header>'
        body+=links+'<div class="bookmark"><label><input id="mark-read" type="checkbox"> 我已完成本章练习</label><span id="progress-message" class="progress-note">只记你的勾选，不自动判断掌握。</span></div>'
        js=NAV_JS
        if name=='10-attention':body+=(ROOT/'tools/attention-widget.html').read_text();js+=(ROOT/'tools/attention-widget.js').read_text()
        content=header+'<div class="layout"><nav aria-label="章节目录"><div class="book-toc"><a href="index.html">← 全部18章</a><a href="setup.html">运行与排查</a></div>'+toc+'</nav><main>'+body+'</main></div>'
        (out/f'{name}.html').write_text(shell(titles[name],content,css,js,name))
    for slug,file in [('setup','SETUP.md'),('reports-toy','reports/toy.md'),('reports-real','reports/real.md')]:
        p=ROOT/file
        if p.exists():
            body,toc=convert(p.read_text());title=title_of(p.read_text());(out/f'{slug}.html').write_text(shell(title,'<header><a href="index.html">← 课程首页</a><h1>'+html.escape(title)+'</h1></header><div class="layout"><nav>'+toc+'</nav><main>'+body+'</main></div>',css,NAV_JS))
    cards=''
    for group,names in GROUPS:
        cards+='<section class="book-group"><h2>'+group+'</h2><ol start="'+str(FILES.index(names[0])+1)+'">'
        for name in names:cards+='<li>'+(f'<a href="{name}.html">{html.escape(titles[name])}</a>' if name in titles else name+'（编写中）')+'</li>'
        cards+='</ol></section>'
    preview='<p>当前为制作预览，部分章节尚在编写。</p>' if missing else ''
    hero='<header><div class="eyebrow">LLM FOR SOFTWARE ENGINEERS</div><h1>从一个参数，走到一段回答</h1><p class="lead">给会编程、没系统学过机器学习的软件工程师：把重要原理算清楚，把实验跑起来，再解释结果和局限。</p><div class="hero-facts"><span>中文 · 18章</span><span>手算 + Python + PyTorch</span><span>CPU主线 · 无付费API前置</span><span>离线可读</span></div>'+preview+'<a class="primary-link" href="01-models.html">从第一章开始</a> <a href="10-attention.html#attention-playground">先试注意力交互</a> · <a href="setup.html">环境与运行</a><p class="progress-note">你标记完成练习：<strong id="read-count">0 / 18</strong>。记录仅存当前浏览器地址。</p></header>'
    intro='<section class="book-intro"><h2>你会完成什么</h2><p>先用计数预测下一个符号，再理解可训练参数、损失和反向传播；随后逐步拼出Transformer，训练两个小模型，并用证据评价一次问答。数学在需要时解释，注意力与训练都不跳过。</p><p><strong>三层证据：</strong>给定小数字验证计算；有限语法验证训练循环；英文原作小语料观察真实文本预测。模型很小，生成结果可能粗糙；这些实验不等于通用助手能力。</p><p>已有ML基础可从第08章进入，遇到梯度/矩阵卡点回查第02、05、06章。所有读者都应读数据切分与结果边界。</p><p><a href="reports-toy.html">有限语法实测报告</a> · <a href="reports-real.html">自然语言实测报告</a></p></section>'
    (out/'index.html').write_text(shell('课程路线',hero+'<div class="book-home">'+intro+'<div class="book-grid">'+cards+'</div><footer>内容与实验随版本维护。材料完成和测试通过，不代表读者已经掌握。<br><a href="https://github.com/bestiaya/llm-for-software-engineers">GitHub源代码与勘误</a></footer></div>',css,NAV_JS))
    # Package sources needed by links inside the offline site; omit checkpoints/caches.
    dst=out/'source';dst.mkdir(exist_ok=True)
    for folder in ['chapters','labs','llm_course','reports','data']:
        src=ROOT/folder
        if src.exists():shutil.copytree(src,dst/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pt','IMPLEMENTATION.md'))
    (out/'.nojekyll').touch()
    shutil.copytree(out,ROOT/'docs',dirs_exist_ok=True)
    print(f'Built {len(titles)} chapters and course home; docs/ mirrors site/ for Pages')

if __name__=='__main__':main()
