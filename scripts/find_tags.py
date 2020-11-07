from tqdm import tqdm
import os
from bs4 import BeautifulSoup, SoupStrainer
import re

strainer = SoupStrainer('tag')

tag_dict = {}
for subdir, dirs, files in tqdm(list(os.walk('../Stories'))):
    for file in tqdm(files):
        if not file.endswith('.html'):
            continue

        file_path = os.path.join(subdir, file)
        story_path = os.path.join(subdir.split('/')[-1], file)

        with open(file_path) as in_file:
            soup = BeautifulSoup(in_file.read(), 'lxml', parse_only=strainer)

        tag_list = [t.text for t in soup.find_all('tag')]
        for tag in tag_list:
            if tag not in tag_dict:
                tag_dict[tag] = []
            tag_dict[tag].append(story_path)

if not os.path.isdir('../Stories/tags'):
    os.mkdir('../Stories/tags')

for tag in tag_dict:
    tag_file = tag.replace('/', '-')

    with open(f'../Stories/tags/{tag_file}.txt', 'w') as out_file:
        for story_path in tag_dict[tag]:
            out_file.write(story_path + '\n')
