import sys
sys.path.append('.')
from bs4 import BeautifulSoup, SoupStrainer
import os
from story.base import BaseStory
import time
import datetime
import pandas as pd
from tqdm import tqdm


df = pd.read_csv('out/lit_story_values.csv', index_col=0)
index_labels = df.index
author_title_list = [a.split('/')[1] for a in index_labels]
author_list = []
strainer = SoupStrainer(id='author')
for i, file_name in tqdm(enumerate(author_title_list)):
    is_split = file_name.split(' - ')
    author_list.append(is_split[0])
df['author'] = author_list

df.to_csv('temp.csv', float_format='%.6f')
