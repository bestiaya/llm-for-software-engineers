"""Build offline reading pages from the course manifest (v2/course.json)."""

from __future__ import annotations

import html
from itertools import groupby
import json
import re
from os.path import relpath
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
from xml.etree import ElementTree

import markdown
from markdown.extensions import Extension
from markdown.extensions.toc import slugify_unicode
from markdown.treeprocessors import Treeprocessor


ROOT = Path(__file__).resolve().parents[1]
COURSE = ROOT / "v2"
EXTRA_CSS = """
.lesson-floating-toc { align-self: start; }
.lesson-floating-toc .toc > ul { padding-left: 0; }
.lesson-floating-toc a.is-active { color: var(--ink); }
.figure { display: block; }
.course-home { max-width: 1180px; padding: 0 24px 50px; margin: auto; }
.course-units { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 22px; }
.course-unit { padding: 20px 24px; background: var(--paper); border: 1px solid var(--line); border-radius: 12px; }
.course-unit h2 { padding: 0; margin: 0 0 14px; border: 0; font-size: 20px; }
.course-unit ol { padding-left: 0; list-style: none; }
.course-unit li { padding-left: 3px; }
.chapter-links { display: flex; justify-content: space-between; gap: 20px; margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--line); }
@media (max-width: 850px) {
  .course-home { padding: 0 14px 30px; }
  .course-units { grid-template-columns: 1fr; }
}
"""

SCROLLSPY = """
(() => {
  const nav = document.querySelector('[data-lesson-floating-toc]');
  const links = [...nav.querySelectorAll('.toc a')];
  const sections = links.map(link => ({
    link,
    heading: document.getElementById(decodeURIComponent(link.hash.slice(1)))
  })).filter(section => section.heading);
  const desktop = matchMedia('(min-width: 851px)');
  let active;
  let scheduled = false;

  function update() {
    scheduled = false;
    if (!sections.length) return;
    let current = sections[0].link;
    for (const { link, heading } of sections) {
      if (heading.getBoundingClientRect().top > 135) break;
      current = link;
    }
    if (scrollY > 0 && innerHeight + scrollY >= document.documentElement.scrollHeight - 2) {
      current = sections.at(-1).link;
    }
    if (current !== active) {
      active?.removeAttribute('aria-current');
      active?.classList.remove('is-active');
      current.setAttribute('aria-current', 'location');
      current.classList.add('is-active');
      active = current;
    }
    const viewport = nav.getBoundingClientRect();
    const item = current.getBoundingClientRect();
    if (desktop.matches && viewport.top >= 0 && viewport.bottom <= innerHeight
        && (item.top < viewport.top || item.bottom > viewport.bottom)) {
      current.scrollIntoView({ block: 'nearest', inline: 'nearest', container: 'nearest' });
    }
  }

  function schedule() {
    if (!scheduled) {
      scheduled = true;
      requestAnimationFrame(update);
    }
  }

  addEventListener('scroll', schedule, { passive: true });
  addEventListener('resize', schedule);
  addEventListener('hashchange', schedule);
  update();
})();
"""


def course_path(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to(COURSE):
        raise ValueError(f"课程文件须保存在 v2 目录中（{relative}）")
    return path


class LessonMarkup(Treeprocessor):
    def __init__(self, md, directory):
        super().__init__(md)
        self.directory = directory

    def run(self, root):
        first = root.find('h1')
        if first is not None:
            root.remove(first)
        for heading in root.iter('h2'):
            if heading.get('id'):
                heading.set('data-concept', heading.get('id'))

        for picture in root.iter('img'):
            location = urlsplit(picture.get('src', ''))
            path = Path(unquote(location.path))
            if location.scheme or location.netloc or path.is_absolute() or path.suffix.lower() != '.svg':
                raise ValueError(f"课件图片须引用 v2 目录中的 SVG（{picture.get('src')}）")
            asset = course_path(self.directory, path)
            diagram = asset.read_text(encoding='utf-8')
            svg = ElementTree.fromstring(diagram)
            if svg.tag not in ('{http://www.w3.org/2000/svg}svg', 'svg'):
                raise ValueError(f"图片文件须包含 SVG 图（{path}）")
            # Keep the SVG as authored, starting at its root element.
            # The wrapper supplies the Markdown alt text as an accessible name.
            diagram = diagram[diagram.index('<svg'):]
            picture.tag = 'span'
            alt = picture.attrib.pop('alt', '')
            del picture.attrib['src']
            picture.set('class', 'figure')
            if alt:
                picture.set('role', 'img')
                picture.set('aria-label', alt)
            picture.text = self.md.htmlStash.store(diagram)

        # Sources link to other chapters and to the guide as Markdown files, which
        # GitHub can follow. In the built pages those links go to the HTML pages.
        pages = {'lesson.md': 'lesson.html', 'ABOUT.md': 'about.html'}
        for link in root.iter('a'):
            location = urlsplit(link.get('href', ''))
            name = Path(unquote(location.path)).name
            if location.scheme or location.netloc or name not in pages:
                continue
            target = location.path[:len(location.path) - len(name)] + pages[name]
            link.set('href', target + ('#' + location.fragment if location.fragment else ''))

        for table in list(root.iter('table')):
            inner = ElementTree.Element('table', table.attrib)
            inner.extend(table)
            table[:] = [inner]
            table.tag = 'div'
            table.attrib.clear()
            table.set('class', 'table-wrap')


class LessonExtension(Extension):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory

    def extendMarkdown(self, md):
        # Run after inline markup and attr_list, ahead of the table of contents.
        md.treeprocessors.register(LessonMarkup(md, self.directory), 'lesson-markup', 6)


def shell(title, content, styles, script=''):
    script_tag = f'<script>{script}</script>' if script else ''
    return (
        '<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{html.escape(title)}</title><style>{styles}</style></head>\n'
        f'<body>{content}{script_tag}</body></html>\n'
    )


def relative_url(target, start):
    relative = Path(relpath(target, start)).as_posix()
    return quote(relative)


def chapter_code(course, chapter):
    return f"{course['code']}-{chapter['number']:02d}"


def chapter_title(course, chapter):
    return f"{chapter_code(course, chapter)} {chapter['title']}"


def chapter_page(course, chapter, source, styles, *, is_guide=False):
    md = markdown.Markdown(
        extensions=['extra', 'toc', LessonExtension(source.parent)],
        extension_configs={'toc': {'toc_depth': '2-3', 'slugify': slugify_unicode}},
    )
    text = source.read_text(encoding='utf-8')
    # Explicit HTML anchors work in GitHub Markdown. Attach the same IDs to
    # headings for the offline table of contents, avoiding duplicate anchors.
    text = re.sub(
        r'^<a id="([A-Za-z][A-Za-z0-9_-]*)"></a>\n\n(#{2,3}) (.+)$',
        lambda match: f'{match[2]} {match[3]} {{#{match[1]}}}',
        text,
        flags=re.MULTILINE,
    )
    body = md.convert(text)
    title = f"{course['code']} {chapter['title']}" if is_guide else chapter_title(course, chapter)
    home_url = relative_url(COURSE / 'index.html', source.parent)
    source_url = relative_url(source, source.parent)
    guide_url = relative_url(COURSE / 'about.html', source.parent)
    guide_link = '' if is_guide else f'<a href="{guide_url}">课程导读</a> · '
    source_label = '本文源文' if is_guide else '本章源文'
    toc_label = '导读目录' if is_guide else f'{chapter_code(course, chapter)} 本章目录'
    header = (
        '<header>'
        f'<p>{guide_link}<a href="{home_url}">课程目录</a> · <a href="{source_url}">{source_label}</a></p>'
        f'<h1>{html.escape(title)}</h1>'
        '</header>'
    )
    nav = (
        f'<nav class="lesson-floating-toc" data-lesson-floating-toc aria-label="{toc_label}">'
        f'<strong>{toc_label}</strong>{md.toc}</nav>'
    )
    siblings = course['chapters']
    position = -1 if is_guide else siblings.index(chapter)
    previous = siblings[position - 1] if position > 0 else None
    next_chapter = siblings[position + 1] if position + 1 < len(siblings) else None
    navigation = []
    if position == 0:
        navigation.append(f'<a href="{guide_url}">课程导读</a>')
    for adjacent, label in ((previous, '上一章'), (next_chapter, '下一章')):
        if adjacent and adjacent['source']:
            adjacent_source = course_path(COURSE, adjacent['source'])
            if adjacent_source.is_file():
                url = relative_url(adjacent_source.with_name('lesson.html'), source.parent)
                navigation.append(f'<a href="{url}">{label} · {html.escape(chapter_title(course, adjacent))}</a>')
    navigation.append(f'<a href="{home_url}">课程目录</a>')
    body += '<div class="chapter-links" aria-label="章节切换">' + ''.join(navigation) + '</div>'
    page_title = title + ' · ' + course['title']
    return shell(page_title, f'{header}<div class="layout">{nav}<main>{body}</main></div>', styles, SCROLLSPY)


def course_page(course, published, styles):
    units = []
    for unit, chapters in groupby(course['chapters'], key=lambda chapter: chapter['unit']):
        first_number = None
        items = []
        for chapter in chapters:
            number = chapter['number']
            if first_number is None:
                first_number = number
            title = html.escape(chapter_title(course, chapter))
            if number in published:
                url = relative_url(published[number], COURSE)
                title = f'<a href="{url}">{title}</a>'
            items.append(f'<li value="{number}">{title}</li>')
        units.append(
            f'<section class="course-unit"><h2>{html.escape(unit)}</h2>'
            f'<ol start="{first_number}">{"".join(items)}</ol></section>'
        )
    title = f"{course['code']} {course['title']}"
    header = (
        f'<header><h1>{html.escape(title)}</h1>'
        '<p>从一次请求开始，理解机器学习和大语言模型的原理，再用实验判断它们能承担哪些工程任务。</p>'
        '<p><a href="about.html">先读课程导读：目标读者、初衷、设计理念与取舍</a></p>'
        f'<p>全课共 {len(course["chapters"])} 章，正文逐章更新。带链接的章节可打开阅读。</p></header>'
    )
    content = header + f'<div class="course-home"><div class="course-units">{"".join(units)}</div></div>'
    return shell('课程目录 · ' + title, content, styles)


def main():
    course = json.loads((COURSE / 'course.json').read_text(encoding='utf-8'))
    styles = (ROOT / 'tools' / 'theme.css').read_text(encoding='utf-8') + EXTRA_CSS
    published = {}
    for chapter in course['chapters']:
        if chapter['source'] is None:
            continue
        source = course_path(COURSE, chapter['source'])
        if not source.is_file():
            continue
        destination = source.with_name('lesson.html')
        destination.write_text(chapter_page(course, chapter, source, styles), encoding='utf-8')
        published[chapter['number']] = destination

    (COURSE / 'index.html').write_text(course_page(course, published, styles), encoding='utf-8')
    guide = COURSE / 'ABOUT.md'
    (COURSE / 'about.html').write_text(
        chapter_page(course, {'title': '课程导读：写给应用工程师'}, guide, styles, is_guide=True),
        encoding='utf-8',
    )
    print(f'已构建课程导读、{len(published)} 章和 {len(course["chapters"])} 章的课程目录。')


if __name__ == '__main__':
    main()
