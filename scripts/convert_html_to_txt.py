import os
from pathlib import Path

from tqdm import tqdm
import sys
from bs4 import BeautifulSoup, SoupStrainer
import plac


@plac.pos('story_html_dir', 'Directory with story .html files', type=Path)
@plac.pos('story_text_dir', "Output directory with .txt files", type=Path)
@plac.opt('n_files', "Number of files to convert", type=int)
def main(story_html_dir, story_text_dir, n_files=-1):
    """Convert specifically formatted html files into text files"""

    file_count = 0
    strainer = SoupStrainer('div', {'class': 'chapter'})

    if not os.path.isdir(story_text_dir):
        os.mkdir(story_text_dir)

    for subdir, dirs, files in tqdm(list(os.walk(story_html_dir))):
        for file_name in tqdm(files):
            file_path = os.path.join(subdir, file_name)
            if not file_path.endswith('.html'):
                continue

            pre, ext = os.path.splitext(file_name)
            new_file_path = os.path.join(story_text_dir, pre + '.txt')

            if os.path.exists(new_file_path):
                continue

            with open(file_path) as in_file:
                soup = BeautifulSoup(in_file.read(), 'lxml', parse_only=strainer)

            text = '\n\n'.join([t.text for t in soup.find_all('div', {'class': 'chapter'})])
            with open(new_file_path, 'w') as out_file:
                out_file.write(text)

            file_count += 1
            if file_count == n_files:
                return


if __name__ == '__main__':
    plac.call(main)
