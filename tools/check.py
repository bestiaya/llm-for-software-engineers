"""Check standalone publication paths, chapter structure and generated assets."""
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote,urlsplit
import hashlib
import json
import re
import sys
from build import FILES
ROOT=Path(__file__).resolve().parents[1]
class HTML(HTMLParser):
    def __init__(self,s):
        super().__init__();self.ids=[];self.links=[];self.assets=[];self.feed(s)
    def handle_starttag(self,t,a):
        d=dict(a)
        if d.get('id'):self.ids.append(d['id'])
        if t=='a' and d.get('href'):self.links.append(d['href'])
        if t in ['script','link','img','iframe'] and (d.get('src') or d.get('href')):self.assets.append(d.get('src') or d.get('href'))
def main():
    errors=[];links=0;chapters=[]
    for name in FILES:
        f=ROOT/'chapters'/f'{name}.md'
        if not f.exists():errors.append(f'missing chapter {name}');continue
        s=f.read_text();chapters.append(name)
        if s.count('<details')<1 or (s.count('<details')<3 and s.count('<li>')<3):errors.append(f'{name}: missing folded answers for three exercises')
        if '<svg' not in s and not re.search(r'!\[[^\]]*\]\(\.\./assets/[^)]+\.svg\)',s):errors.append(f'{name}: no diagram')
        if not re.search(r'^## ',s,re.M):errors.append(f'{name}: no sections')
        if any(x in s for x in ['/Users/','TODO','待补充正文']):errors.append(f'{name}: private path or unfinished text')
    parsed={p:HTML(p.read_text()) for p in (ROOT/'site').glob('*.html')}
    for p,h in parsed.items():
        if len(h.ids)!=len(set(h.ids)):errors.append(f'{p.name}: duplicate ids')
        if h.assets:errors.append(f'{p.name}: non-inline runtime assets')
        for a in h.links:
            u=urlsplit(a)
            if u.scheme or u.netloc:continue
            links+=1;t=(p.parent/unquote(u.path)).resolve() if u.path else p
            if not t.is_relative_to(ROOT):errors.append(f'{p.name}: leaves repo {a}')
            elif not t.exists():errors.append(f'{p.name}: missing {a}')
            elif t.suffix=='.html' and u.fragment:
                target=parsed.get(t) or HTML(t.read_text())
                if unquote(u.fragment) not in target.ids:errors.append(f'{p.name}: missing anchor {a}')
    for f in [ROOT/'README.md',ROOT/'README.en.md',ROOT/'SETUP.md',ROOT/'CONTRIBUTING.md',ROOT/'NOTICE.md',*(ROOT/'chapters').glob('*.md')]:
        for a in re.findall(r'\]\(([^)]+)\)',f.read_text()):
            u=urlsplit(a)
            if u.scheme or u.netloc:continue
            t=(f.parent/unquote(u.path)).resolve()
            if not t.is_relative_to(ROOT) or not t.exists():errors.append(f'{f.name}: missing local source link {a}')
    meta=json.loads((ROOT/'data/real-provenance.json').read_text())
    if hashlib.sha256((ROOT/'data/real.txt').read_bytes()).hexdigest()!=meta['prepared_sha256']:errors.append('corpus hash mismatch')
    result={'chapters':len(chapters),'html_pages':len(parsed),'html_local_links':links,'errors':errors}
    print(json.dumps(result,ensure_ascii=False,indent=2));return bool(errors)
if __name__=='__main__':sys.exit(main())
