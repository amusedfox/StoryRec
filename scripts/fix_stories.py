import sys
sys.path.append('.')
from bs4 import BeautifulSoup
import os
from story.base import BaseStory
import time
import datetime

for subdir, dirs, files in os.walk('../Stories'):
    for file in files:
        filepath = os.path.join(subdir, file)
        day = datetime.datetime.strptime(time.ctime(os.path.getmtime(filepath)),
                                         "%a %b %d %H:%M:%S %Y").day

        if day >= 27:
            continue

        if filepath.endswith(".html"):
            print(filepath)
            with open(filepath) as in_file:
                bs = BeautifulSoup(in_file.read(), 'lxml')

                if bs.find('author') is None:
                    continue

                title = bs.find('h1').text
                author = bs.find('author').text
                date_published = bs.find('date').text
                summary = bs.find('desc').text
                link = bs.find_all('a')[-1]['href']
                tag_list = []
                soup_tags = bs.find('tag')
                if soup_tags:
                    for tag in soup_tags:
                        tag_list.append(tag)
                content = bs.find('article')

                p_list = content.text.split('\n')
                p_list = [f'<p>{p}</p>' for p in p_list if p]

                content = '\n\n'.join(p_list)

                base = BaseStory('../Stories')
                base.title = title
                base.author = author
                base.published_date = date_published
                base.summary = summary
                base.link = link
                base.tags = tag_list
                base.chapter_names.append('1. Chapter 1')
                base.num_chapters = 1
                base.chapters.append(content)
                base.category = subdir.split('/')[-1]
                base.story_path = f'{base.category}/{author} - {title}.html'
                base.story_id = base.story_path
                base.story_dir = '../Stories'
                base.is_complete = True
                base.word_count = len(content.split())
                base.publisher = 'literotica.com'
                base.rating = 'M'
            os.remove(filepath)

            # base.write()


