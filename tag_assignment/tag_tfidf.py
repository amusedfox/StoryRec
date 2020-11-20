from collections import defaultdict
from pathlib import Path
import os
import plac
import pandas as pd
# from pandasgui import show

from tqdm import tqdm
import numpy as np
import sys

from sklearn.metrics.pairwise import cosine_similarity

sys.path.append('.')
from tag_assignment.util import search_dir, dict_to_file
from story.base import get_story_tags


def associate_tags_stories(story_html_dir: str, tag_stories_dir: str,
                           min_stories_per_tag: int,
                           overwrite_tag_stories: bool = False,
                           use_ratio: float = 0.8):
    """Return list of stories in each tag to train with"""

    print(f'Associating tags with stories with {use_ratio} as training')
    tag_stories_dict = {}
    story_count = 0

    if os.path.isdir(tag_stories_dir) and not overwrite_tag_stories:
        for tag_file in os.listdir(tag_stories_dir):
            tag_name = os.path.splitext(os.path.basename(tag_file))[0]
            if not tag_name.endswith('_TRAIN'):
                continue
            tag_name = tag_name.rsplit('_', 1)[0]
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

        tags = get_story_tags(abs_file_path)

        for tag in tags:
            if tag not in tag_stories_dict:
                tag_stories_dict[tag] = []
            tag_stories_dict[tag].append(os.path.splitext(rel_file_path)[0])

        story_count += 1

    tag_stories_dict = {k: v for k, v in tag_stories_dict.items()
                        if len(v) > min_stories_per_tag}

    os.makedirs(tag_stories_dir, exist_ok=True)
    for tag, file_name_list in tag_stories_dict.items():
        n_files = len(file_name_list)
        train_list = file_name_list[:int(n_files * use_ratio)]
        valid_list = file_name_list[int(n_files * use_ratio):]
        with open(os.path.join(tag_stories_dir, f'{tag}_TRAIN.txt'), 'w') \
                as out_file:
            for file_name in train_list:
                out_file.write(f'{file_name}\n')

        with open(os.path.join(tag_stories_dir, f'{tag}_VALID.txt'), 'w') \
                as out_file:
            for file_name in valid_list:
                out_file.write(f'{file_name}\n')

    print(f'Found {story_count} stories and {len(tag_stories_dict)} tags')

    return {k: v[:int(len(v) * use_ratio)] for k, v in tag_stories_dict.items()}


def find_tag_ngram_tfidf(tfidf_dir: str, tag_stories_dict: dict):
    """Find the combined TF-IDF stats of each tag

    Combine all story TF-IDFs together for each tag and find statistics from
    them.
    """

    print('Finding n-gram TF-IDF stats for each tag')

    tag_stats = {}
    for tag, file_name_list in tqdm(tag_stories_dict.items()):
        ngram_list = defaultdict(list)

        for file_name in file_name_list:
            with open(os.path.join(tfidf_dir, f'{file_name}.txt')) as in_file:
                for line in in_file:
                    data = line.rsplit(' ', 1)
                    ngram = data[0]

                    tfidf = float(data[1])
                    ngram_list[ngram].append(tfidf)

        n_files = len(file_name_list)
        # to_delete = []
        # # Delete ngrams that have little relevance to the tag
        for ngram, tfidf_list in ngram_list.items():
            # if len(tfidf_list) < 0.05 * n_files:
            #     to_delete.append(ngram)
            pass
        # for ngram in to_delete:
        #     del total_ngram_tfidf[ngram]

        for ngram, tfidf_list in ngram_list.items():
            while len(tfidf_list) < n_files:
                tfidf_list.append(0)

        ngram_stats = {
            ngram: {
                'mean': np.mean(tfidf_list),
                # 'median': np.median(tfidf_list),
                # 'std': np.std(tfidf_list),
                'n': len([_ for _ in tfidf_list if _ != 0])
            } for ngram, tfidf_list in ngram_list.items()
        }

        # avg_ngram_tfidf = {ngram: np.mean(tfidf_list)
        #                    for ngram, tfidf_list in total_ngram_tfidf.items()}
        # dict_to_file(os.path.join(tag_tfidf_dir, f'{tag}.txt'),
        #              avg_ngram_tfidf, write_zeros=False)

        tag_stats[tag] = ngram_stats

    return tag_stats


def test_cosine_similarity(story_tfidf_dir, tag_stories_dir, tag_tfidf_dir,
                           story_html_dir, assets_dir, master_ngram_set: set):
    print('Testing cosine similarity between stories and tags')
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
        found_tags = set(get_story_tags(abs_story_path))
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


def find_important_ngrams(tag_stats_dir, tag_stats, tag_n_stories):
    """Get a list of important ngrams, weights, means, stds per tag

    n-gram important depends on the max difference between tags divided by the
    median. Higher values will indicate that the n-gram is important in
    identifying a tag. Tags that are far from the median in important n-grams
    will affect the probability the story has a tag.
    """
    os.makedirs(tag_stats_dir, exist_ok=True)

    # Get n-gram TF-IDF for each tag
    tags_ngram_list = defaultdict(list)
    for ngram_tfidf_dict in tag_stats.values():
        for ngram, stat_dict in ngram_tfidf_dict.items():
            tags_ngram_list[ngram].append(stat_dict['mean'])

    # Median of n-gram values from each tag
    # Used to find which n-grams are important to which tag
    median_values = {}
    n_tags = len(tag_stats)
    for ngram, stat_dict in tags_ngram_list.items():
        while len(stat_dict) < n_tags:
            stat_dict.append(0)
        median_values[ngram] = np.median(stat_dict)

    df = pd.DataFrame.from_dict(median_values, orient='index',
                                columns=['median'])

    # Get stats for each n-gram within each tag's n-gram list
    # tag_dfs = {'overall': df}
    for tag, ngram_tfidf_dict in tag_stats.items():
        # tag_df = pd.DataFrame()
        ngram_weight = {}
        ngram_mean = {}
        n_ngram = {}
        for ngram, stat_dict in ngram_tfidf_dict.items():
            # If value is far from the median, increase the weight
            # Relatively low weight for values close to median
            # Higher weight for defining features
            # ngram_weight[ngram] = abs(value - median_values[ngram]) / \
            #     median_values[ngram]

            ngram_weight[ngram] = stat_dict['mean'] - median_values[ngram]
            ngram_mean[ngram] = stat_dict['mean']
            n_ngram[ngram] = stat_dict['n']

        # tag_df['weight'] = pd.Series(ngram_weight)
        # tag_df['mean'] = pd.Series(ngram_mean)
        # tag_df['n_stories'] = pd.Series(n_ngram)
        # tag_dfs[tag] = tag_df
        # df[tag] = pd.Series(ngram_mean)

        # Write statistics to file
        with open(os.path.join(tag_stats_dir, tag + '.txt'), 'w') as out_file:
            out_file.write('ngram\tweight\tmean\n')
            for ngram, weight in sorted(ngram_weight.items(),
                                        key=lambda l: l[1], reverse=True)[:1000]:
                out_file.write(f'{ngram}\t{weight}\t{ngram_mean[ngram]}\n')

    # show(**tag_dfs)
    # df.to_csv()


@plac.pos('story_html_dir', 'Directory with story .html files', Path)
@plac.pos('story_tfidf_dir', "Directory with TF-IDF of stories", Path)
@plac.pos('tag_stories_dir', "Output directory for story list for tags", Path)
@plac.pos('tag_stats_dir', "Directory for TF-IDF and stats of tags", Path)
@plac.opt('min_n_stories', "Min number of stories per tag", int)
@plac.opt('overwrite_tag_stories', 'Whether to overwrite tag_stories', bool)
def main(story_html_dir, story_tfidf_dir, tag_stories_dir, tag_stats_dir,
         min_n_stories=100, overwrite_tag_stories=False):
    """Find the n-gram frequencies for each tag"""

    tag_stories_dict = associate_tags_stories(story_html_dir, tag_stories_dir,
                                              min_n_stories,
                                              overwrite_tag_stories)

    tag_stats = find_tag_ngram_tfidf(story_tfidf_dir, tag_stories_dict)

    find_important_ngrams(tag_stats_dir, tag_stats,
                          {tag: len(stories_list) for tag, stories_list
                           in tag_stories_dict.items()})

    # test_cosine_similarity(story_tfidf_dir, tag_stories_dir, tag_tfidf_dir,
    #                        story_html_dir, assets_dir, master_ngram_set)


if __name__ == '__main__':
    plac.call(main)
