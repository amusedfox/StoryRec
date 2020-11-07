import sys
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

sys.path.append('.')
from story.base import BaseStory
from story_stats.stat_func import *
from story_stats.util import load_short_story_set, MIN_STORY_WORD_COUNT

np.seterr(all='raise')

STAT_NAME_FUNCS = [
    ('lex_density', get_lexical_density),
    ('avg_sent_len', get_avg_sent_len),
    ('avg_word_len', get_avg_word_len),
    ('avg_word_len_lex', get_avg_word_len_lex),
    ('quote_density', get_quote_density),
    ('commas_per_word', get_commas_per_word),
    ('avg_word_syl_num', get_avg_word_syl_num)
]
META_INFO_LIST = ['title', 'author', 'category', 'link', 'word_count']


def get_stories_stats(parent_dir: str,
                      short_story_file_path: str = None,
                      output_file_path: str = None,
                      found_stories_stats_pd: pd.DataFrame = None,
                      stories_stats_pd_file_path: str = None,
                      update_col=False):
    total_num_stories = 0
    num_short_stories = 0

    if found_stories_stats_pd is not None:
        stories_stats_pd = found_stories_stats_pd
    elif stories_stats_pd_file_path is not None and \
            os.path.exists(stories_stats_pd_file_path):
        stories_stats_pd = pd.read_csv(stories_stats_pd_file_path,
                                       index_col=0)
    else:
        stories_stats_pd = pd.DataFrame(
            columns=
            META_INFO_LIST + [s[0] for s in STAT_NAME_FUNCS])
        stories_stats_pd.index.name = 'story_label'
    for stat_name, stat_func in STAT_NAME_FUNCS:
        if stat_name not in stories_stats_pd:
            stories_stats_pd[stat_name] = np.nan
    for info_name in META_INFO_LIST:
        if info_name not in stories_stats_pd:
            stories_stats_pd[info_name] = 'N/A'

    check_short_story = False
    short_story_set = None
    short_story_list_file = None
    if short_story_file_path:
        check_short_story = True
        short_story_set = load_short_story_set(short_story_file_path)
        short_story_list_file = open(short_story_file_path, 'a+')

    story_count = 0
    for subdir, dirs, files in os.walk(parent_dir):

        if 'pytest' in subdir and 'pytest' not in parent_dir:
            continue

        # for file in files:
        for file in tqdm(files):

            if not file.endswith('.html'):
                continue

            story_count += 1
            total_num_stories += 1

            story_label = os.path.join(subdir.split('/')[-1], file)
            if not update_col and story_label in stories_stats_pd.index:
                continue

            if story_count % 10000 == 0:
                try:
                    stories_stats_pd.to_csv(
                        f'temp/temp_story_values_{story_count}.csv',
                        columns=META_INFO_LIST + [s[0] for s in STAT_NAME_FUNCS],
                        float_format='%.6f')
                except KeyboardInterrupt:
                    print('ERROR: Interrupted while printing to file')
                    os.remove(f'temp/temp_story_values_{story_count}.csv')
                    raise KeyboardInterrupt

                try:
                    os.remove(
                        f'temp/temp_story_values_{story_count - 10000}.csv')
                except FileNotFoundError:
                    pass

            if check_short_story and story_label in short_story_set:
                num_short_stories += 1
                continue

            story = BaseStory(parent_dir, story_path=story_label)
            if story is None:
                continue

            if check_short_story and story.word_count < MIN_STORY_WORD_COUNT:
                num_short_stories += 1
                short_story_list_file.write(story_label + '\n')
                continue

            story_stats_series = pd.Series(dtype='float64')
            if story_label in stories_stats_pd.index:
                story_stats_series = stories_stats_pd.loc[story_label].copy()

            get_story_stats(story, story_stats_series)
            stories_stats_pd.loc[story_label] = story_stats_series

    if check_short_story:
        short_story_list_file.close()

    # Write to csv file
    if output_file_path:
        try:
            stories_stats_pd.to_csv(
                output_file_path,
                columns=META_INFO_LIST + [s[0] for s in STAT_NAME_FUNCS],
                float_format='%.6f'
            )
            print(f'Wrote to story_stats_pd to {output_file_path}')
        except KeyboardInterrupt:
            print(f'ERROR: Interrupted while printing to file')
            os.remove(output_file_path)
            raise KeyboardInterrupt

    print(f'Numb Short Stories: {num_short_stories}')
    print(f'Total Numb Stories: {total_num_stories}')
    print(f'Stats Found Stories: {len(stories_stats_pd.index)}')

    # for k, v in sorted(word_freq_dict.items(), key=lambda item: item[1], reverse=False):
    #     if v > 3:
    #         print(k, v)

    return stories_stats_pd


def get_story_stats(story: BaseStory, story_stat_series: pd.Series) -> None:
    story.preprocess()
    for stat_name, stat_func in STAT_NAME_FUNCS:
        if stat_name not in story_stat_series or np.isnan(
                story_stat_series[stat_name]):
            story_stat_series[stat_name] = stat_func(story)


def remove_stat(stat_name: str, stories_stats_pd: pd.DataFrame):
    stories_stats_pd.drop(stat_name, axis=1, inplace=True)


def remove_stat_from_file(stat_name: str, pd_file_path: str):
    df = pd.read_csv(pd_file_path, index_col=0)
    remove_stat(stat_name, df)
    stat_names = list(df.columns)
    df.to_csv(
        pd_file_path,
        columns=stat_names,
        float_format='%.6f'
    )
