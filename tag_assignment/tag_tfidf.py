from pathlib import Path
import os
import plac

from tqdm import tqdm
import sys
from bs4 import BeautifulSoup, SoupStrainer

sys.path.append('.')
from tag_assignment.util import search_dir, dict_to_file
from story.base import ILLEGAL_FILE_CHARS


def associate_tags_stories(story_html_dir, tag_stories_dir, tfidf_dir,
                           n_stories_to_use=-1):

    tag_stories_dict = {}

    strainer = SoupStrainer('tag')

    story_count = 0
    for file_path in tqdm(search_dir(story_html_dir, '.html')):
        file_name = file_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]

        if not os.path.isfile(os.path.join(tfidf_dir, f'{file_name}.txt')):
            continue

        with open(file_path) as in_file:
            soup = BeautifulSoup(in_file.read(), 'lxml', parse_only=strainer)

        tags = soup.find_all('tag')
        for tag in tags:
            tag = ILLEGAL_FILE_CHARS.sub('_', tag.text)
            if tag not in tag_stories_dict:
                tag_stories_dict[tag] = []
            tag_stories_dict[tag].append(file_name)

        story_count += 1
        if story_count == n_stories_to_use:
            break

    if not os.path.isdir(tag_stories_dir):
        os.mkdir(tag_stories_dir)

    for tag, file_name_list in tag_stories_dict.items():
        with open(os.path.join(tag_stories_dir, f'{tag}.txt'), 'w') as out_file:
            for file_name in file_name_list:
                out_file.write(f'{file_name}\n')

    print(f'Found {story_count} stories')

    return tag_stories_dict


def find_tag_ngram_tfidf(tfidf_dir, tag_tfidf_dir, tag_stories_dict,
                         min_n_stories=100):
    tag_count = 0

    for tag, file_name_list in tqdm(tag_stories_dict.items()):

        if len(file_name_list) < min_n_stories:
            continue

        comb_ngram_tfidf = {}
        for file_name in file_name_list:
            with open(os.path.join(tfidf_dir, f'{file_name}.txt')) as in_file:
                for line in in_file:
                    data = line.rsplit(' ', 1)
                    ngram = data[0]
                    tfidf = float(data[1])

                    if ngram not in comb_ngram_tfidf:
                        comb_ngram_tfidf[ngram] = 0

                    comb_ngram_tfidf[ngram] += tfidf

        for ngram in comb_ngram_tfidf:
            comb_ngram_tfidf[ngram] /= len(file_name_list)

        if not os.path.isdir(tag_tfidf_dir):
            os.mkdir(tag_tfidf_dir)

        dict_to_file(os.path.join(tag_tfidf_dir, f'{tag}.txt'),
                     comb_ngram_tfidf)

        tag_count += 1

    print(f'Found {tag_count} tags')


@plac.pos('story_html_dir', 'Directory with story .html files', type=Path)
@plac.pos('tfidf_dir', "Directory with TF-IDF of stories", type=Path)
@plac.pos('tag_stories_dir', "Output directory for story list for tags", type=Path)
@plac.pos('tag_tfidf_dir', "Output directory for TF-IDF of tags", type=Path)
@plac.opt('n_stories_to_use', "Number of stories to use in corpus", type=int)
@plac.opt('min_n_stories', "Min number of stories per tag", type=int)
def main(story_html_dir, tfidf_dir, tag_stories_dir, tag_tfidf_dir,
         n_stories_to_use=-1, min_n_stories=100):
    """Find the n-gram frequencies for each tag"""

    print('Associating tags with stories')
    tag_stories_dict = associate_tags_stories(story_html_dir, tag_stories_dir,
                                              tfidf_dir, n_stories_to_use)

    print('Finding combined TF-IDF for each tag')
    find_tag_ngram_tfidf(tfidf_dir, tag_tfidf_dir, tag_stories_dict,
                         min_n_stories)


if __name__ == '__main__':
    plac.call(main)
