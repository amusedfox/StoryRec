import os
import numpy as np

import spacy
import sys
import pandas as pd
from tqdm import tqdm

sys.path.append('.')
from story.base import BaseStory

POS_TAGS = [
    'ADJ',
    'ADV',
    'INTJ',
    'NOUN',
    'PROPN',
    'VERB',
    'ADP',
    'AUX',
    'CCONJ',
    'DET',
    'PART',
    'PRON',
    'SCONJ'
]

nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])

stories_values_df = pd.read_csv('out/lit_story_values.csv', index_col=0)

author_stats = {}

for author in tqdm(['GigglingGoblin', 'justincbenedict', 'en_extase', 'Samuelx', 'Sean Renaud']):

    author_df = stories_values_df[stories_values_df['author'] == author]

    if os.path.exists(f'temp/{author}.csv'):
        author_stats[author] = pd.read_csv(f'temp/{author}.csv', index_col=0)
        continue

    if len(author_df.index) == 0:
        print(f'{author} not found')
        continue

    df = pd.DataFrame(columns=[f'pos_{pos}' for pos in POS_TAGS])
    for story_path in tqdm(author_df.index):
        base = BaseStory(story_dir='../Stories', story_path=story_path)
        doc = nlp(base.content)
        pos_dict = {f'pos_{pos}': 0 for pos in POS_TAGS}
        for token in doc:
            pos_tag = 'pos_' + token.pos_
            if pos_tag in pos_dict:
                pos_dict[pos_tag] += 1

                tag_tag = 'tag_' + token.tag_
                if tag_tag not in pos_dict:
                    pos_dict[tag_tag] = 0
                pos_dict[tag_tag] += 1

                tag_dep = 'dep_' + token.dep_
                if tag_dep not in pos_dict:
                    pos_dict[tag_dep] = 0
                pos_dict[tag_dep] += 1

        for tag in pos_dict:
            if tag not in df:
                df[tag] = 0
        df.loc[story_path] = pos_dict

    for index in df.index:
        df.loc[index] = df.loc[index] / sum(df.loc[index])

    author_stats[author] = df

    author_stats[author].to_csv(f'temp/{author}.csv')
    print(f'Wrote {author} to temp file')

# for author in author_stats:
#     print(author, author_stats[author].mean())
