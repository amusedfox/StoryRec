import pandas as pd
import sys
import os
from bs4 import BeautifulSoup, SoupStrainer
from tqdm import tqdm
sys.path.append('.')

from story_stats.story_stats import STAT_NAME_FUNCS

assert len(sys.argv) == 2

df = pd.read_csv(sys.argv[1], index_col=0)

index_labels = df.index

title_list = [t.split('-')[-1][:-5].strip() for t in index_labels]
print(title_list[0])
df['title'] = title_list

author_list = [a.split('/')[1].split()[0] for a in index_labels]
print(author_list[0])
df['author'] = author_list

category_list = [c.split('/')[0] for c in index_labels]
print(category_list[0])
df['category'] = category_list

word_count_list = []
link_list = []
strainer = SoupStrainer(True, {'id': ['title', 'word_count']})
for index in tqdm(index_labels):
    index = os.path.join('../Stories', index)
    with open(index) as in_file:
        soup = BeautifulSoup(in_file.read(), 'lxml', parse_only=strainer)
    word_count = soup.find(id='word_count').text
    link = soup.find(id='title')['href']
    word_count_list.append(word_count)
    link_list.append(link)

df['word_count'] = word_count_list
df['link'] = link_list

df.to_csv('temp/test.csv',
          columns=['title', 'author', 'category', 'link', 'word_count'] +
                  [s[0] for s in STAT_NAME_FUNCS],
          float_format='%.6f')

print(df)
