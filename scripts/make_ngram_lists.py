import sys
import os
from tqdm import tqdm

sys.path.append('.')
from story_stats.util import get_lemm_list, get_ngram_list

if len(sys.argv) != 3:
    raise ValueError('Arguments: story_text_dir ngram_list_dir')

story_text_dir = sys.argv[1]
ngram_list_dir = sys.argv[2]

if not os.path.isdir(ngram_list_dir):
    print(f'Making directory: {ngram_list_dir}')
    os.mkdir(ngram_list_dir)

for root, dirs, files in tqdm(list(os.walk(story_text_dir))):
    for file_name in tqdm(files):
        if not file_name.endswith('.txt'):
            continue

        category = root.split('/')[-1]
        if not os.path.isdir(f'{ngram_list_dir}/{category}'):
            os.mkdir(f'{ngram_list_dir}/{category}')

        with open(os.path.join(root, file_name)) as in_file:
            text = in_file.read()

        lemm_list = get_lemm_list(text)
        ngram_list = get_ngram_list(lemm_list)
        out_str = ""
        for ngram in ngram_list:
            out_str += ngram + '\n'
        with open(f'{ngram_list_dir}/{category}/{file_name}', 'w') as out_file:
            out_file.write(out_str)
        break
