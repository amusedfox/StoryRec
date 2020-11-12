from pathlib import Path
import os
import plac
import pandas as pd
from pandasgui import show

from tqdm import tqdm
import numpy as np
import sys

from sklearn.metrics.pairwise import cosine_similarity

sys.path.append('.')
from tag_assignment.util import search_dir, dict_to_file
from story.base import get_tags


def associate_tags_stories(story_html_dir: str, tag_stories_dir: str,
                           min_stories_per_tag: int = 100,
                           overwrite_tag_stories: bool = False):

    tag_stories_dict = {}
    story_count = 0

    if os.path.isdir(tag_stories_dir) and not overwrite_tag_stories:
        for tag_file in os.listdir(tag_stories_dir):
            tag_name = os.path.splitext(os.path.basename(tag_file))[0]
            with open(os.path.join(tag_stories_dir, tag_file)) as in_file:
                tag_list = in_file.read().strip().split('\n')
                tag_stories_dict[tag_name] = tag_list
                story_count += len(tag_list)

        print(f'Using previously computed tag-story associations')
        print(f'Found {story_count} stories and {len(tag_stories_dict)} tags')

        return tag_stories_dict

    for rel_file_path in tqdm(
            search_dir(story_html_dir, '.html', abs_path=False)):
        abs_file_path = os.path.join(story_html_dir, rel_file_path)

        tags = get_tags(abs_file_path)

        for tag in tags:
            if tag not in tag_stories_dict:
                tag_stories_dict[tag] = []
            tag_stories_dict[tag].append(os.path.splitext(rel_file_path)[0])

        story_count += 1

    tag_stories_dict = {k: v for k, v in tag_stories_dict.items()
                        if len(v) > min_stories_per_tag}

    os.makedirs(tag_stories_dir, exist_ok=True)
    for tag, file_name_list in tag_stories_dict.items():
        with open(os.path.join(tag_stories_dir, f'{tag}.txt'), 'w') as out_file:
            for file_name in file_name_list:
                out_file.write(f'{file_name}\n')

    print(f'Found {story_count} stories and {len(tag_stories_dict)} tags')

    return tag_stories_dict


def find_tag_ngram_tfidf(tfidf_dir: str, tag_tfidf_dir: str,
                         tag_stories_dict: dict, master_ngram_set: set,
                         min_n_stories: int = 100):

    total_ngram_tfidf = {k: 0 for k in master_ngram_set}
    for tag, file_name_list in tqdm(tag_stories_dict.items()):

        if len(file_name_list) < min_n_stories:
            raise ValueError(
                f'Given file_name_list with length {len(file_name_list)}')

        for file_name in file_name_list:
            with open(os.path.join(tfidf_dir, f'{file_name}.txt')) as in_file:
                for line in in_file:
                    data = line.rsplit(' ', 1)
                    ngram = data[0]

                    if ngram not in master_ngram_set:
                        raise ValueError(
                            f'{file_name}: {ngram} not found in master set')

                    tfidf = float(data[1])
                    total_ngram_tfidf[ngram] += tfidf

        file_name_list_len = len(file_name_list)
        for ngram in total_ngram_tfidf:
            total_ngram_tfidf[ngram] /= file_name_list_len

        dict_to_file(os.path.join(tag_tfidf_dir, f'{tag}.txt'),
                     total_ngram_tfidf, write_zeros=False)


def test_cosine_similarity(story_tfidf_dir, tag_stories_dir, tag_tfidf_dir,
                           story_html_dir, assets_dir, master_ngram_set: set):
    tag_tfidf_list = []
    story_tfidf_list = []
    total_story_list = []
    tag_list = []
    for abs_tag_path in tqdm(search_dir(tag_stories_dir, '.txt')):

        with open(abs_tag_path) as in_file:
            story_list = in_file.read().strip().split('\n')
        tag_file = os.path.basename(abs_tag_path)
        tag_list.append(os.path.splitext(tag_file)[0])

        master_ngram_dict = {k: i for i, k in enumerate(master_ngram_set)}
        tag_tfidf = read_tfidf(os.path.join(tag_tfidf_dir, tag_file),
                               master_ngram_dict)
        tag_tfidf_list.append(tag_tfidf)

        for rel_story_path in story_list:
            abs_story_txt_path = os.path.join(story_tfidf_dir,
                                              f'{rel_story_path}.txt')
            story_tfidf = read_tfidf(abs_story_txt_path, master_ngram_dict)
            story_tfidf_list.append(story_tfidf)

        total_story_list += story_list

    tag_dict = {t: i for i, t in enumerate(tag_list)}
    master_tag_set = set(tag_list)

    results = cosine_similarity(story_tfidf_list, tag_tfidf_list)
    error = np.empty_like(results)
    for story_i, rel_story_path in enumerate(total_story_list):
        abs_story_path = os.path.join(story_html_dir, f'{rel_story_path}.html')
        found_tags = set(get_tags(abs_story_path))
        not_found_tags = master_tag_set - found_tags

        # Result should be 1
        for tag in found_tags:
            tag_i = tag_dict[tag]
            error[story_i][tag_i] = 1 - results[story_i][tag_i]

        # Result should be 0
        for tag in not_found_tags:
            tag_i = tag_dict[tag]
            error[story_i][tag_i] = -1 * results[story_i][tag_i]

    df = pd.DataFrame(error, total_story_list, tag_list)
    mse = (np.square(error)).mean(axis=None)
    print('Square root Mean squared error: ', np.math.sqrt(mse))
    df.to_csv(os.path.join(assets_dir, 'cosine_sim_error.csv'),
              columns=tag_list, float_format='%.6f')


def read_tfidf(file_path, master_ngram_dict):
    tfidf_vector = [0 for _ in range(len(master_ngram_dict))]

    with open(file_path) as in_file:
        for line in in_file:
            data = line.rsplit(' ', 1)
            ngram = data[0]
            tfidf = data[1]

            ngram_i = master_ngram_dict[ngram]
            tfidf_vector[ngram_i] = tfidf

    return tfidf_vector


@plac.pos('story_html_dir', 'Directory with story .html files', Path)
@plac.pos('story_tfidf_dir', "Directory with TF-IDF of stories", Path)
@plac.pos('tag_stories_dir', "Output directory for story list for tags", Path)
@plac.pos('tag_tfidf_dir', "Output directory for TF-IDF of tags", Path)
@plac.pos('assets_dir', "Directory with misc files", Path)
@plac.pos('master_ngram_set_file', "File path to Set of n-grams to use", Path)
@plac.opt('min_n_stories', "Min number of stories per tag", int)
@plac.opt('overwrite_tag_stories', 'Whether to overwrite tag_stories', bool)
def main(story_html_dir, story_tfidf_dir, tag_stories_dir, tag_tfidf_dir,
         assets_dir, master_ngram_set_file,
         min_n_stories=100, overwrite_tag_stories=False):
    """Find the n-gram frequencies for each tag"""

    print('Associating tags with stories')
    tag_stories_dict = associate_tags_stories(story_html_dir, tag_stories_dir,
                                              min_n_stories,
                                              overwrite_tag_stories)

    with open(master_ngram_set_file) as in_file:
        master_ngram_set = set(in_file.read().strip().split('\n'))

    print('Finding combined TF-IDF for each tag')
    find_tag_ngram_tfidf(story_tfidf_dir, tag_tfidf_dir, tag_stories_dict,
                         master_ngram_set, min_n_stories)

    # print('Testing cosine similarity between stories and tags')
    # test_cosine_similarity(story_tfidf_dir, tag_stories_dir, tag_tfidf_dir,
    #                        story_html_dir, assets_dir, master_ngram_set)


if __name__ == '__main__':
    plac.call(main)
