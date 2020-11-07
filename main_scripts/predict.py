import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import pandas as pd
import numpy as np

from story_stats.util import remove_outliers

bad_story_list = []
good_story_list = []
author_story_paths_dict = {}

WARNING_COLOR = '\033[93m'
ERROR_COLOR = '\033[91m'
NORMAL_COLOR = '\033[0m'
GREEN_COLOR = '\033[92m'

np.random.seed(0)
COLORS = [
    'azure',
    'beige',
    'black',
    'brown',
    'chocolate',
    'cyan',
    'gold',
    'fuchsia',
    'indigo',
    'khaki',
    'lavender',
    'orange'
]


def load_stories(story_type: str, stories_stats_df: pd.DataFrame):
    index_list = stories_stats_df.index
    with open(f'test_set/{story_type}/good_stories.txt') as in_file:
        for story in in_file:
            good_story_list.append(story.strip())
            if story.strip() not in index_list:
                print(f"WARNING: Story not found: {story.strip()}")

    with open(f'test_set/{story_type}/bad_stories.txt') as in_file:
        for story in in_file:
            bad_story_list.append(story.strip())
            if story.strip() not in index_list:
                print(f"WARNING: Story not found: {story.strip()}")


def find_author_stories(story_type: str, story_stats_pd: pd.DataFrame):
    with open(f'test_set/{story_type}/authors.txt') as in_file:
        for author in in_file:
            author = author.strip()

            story_paths = []
            for story_path in story_stats_pd.index:
                if author in story_path:
                    story_paths.append(story_path)

            if len(story_paths) == 0:
                raise ValueError(f'{author}\'s stories could not be found')

            author_story_paths_dict[author] = story_paths


def plot_stats(story_stats_pd, folder_path='graphs'):
    color_index = 0

    stat_names = story_stats_pd.columns

    print(f'Plotting for {folder_path} ...')
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    good_story_stats = \
        story_stats_pd.loc[story_stats_pd.index.isin(good_story_list)]
    bad_story_stats = \
        story_stats_pd.loc[story_stats_pd.index.isin(bad_story_list)]
    author_story_stats_dict = {}
    author_color_dict = {}
    for author in author_story_paths_dict:
        author_story_stats_dict[author] = \
            story_stats_pd.loc[story_stats_pd.index.str.contains(author)]
        author_color_dict[author] = COLORS[color_index]
        color_index += 1

        if color_index == len(COLORS):
            raise ValueError('Too many authors and not enough colors')

    num_good_stories = len(good_story_stats.index)
    num_bad_stories = len(bad_story_stats.index)
    if num_good_stories == 0:
        print(f'{WARNING_COLOR}Number of good stories in test set: '
              f'{num_good_stories}{NORMAL_COLOR}')
    else:
        print(f'Number of good stories in test set: {num_good_stories}')
    if num_bad_stories == 0:
        print(f'{WARNING_COLOR}Number of bad stories in test set: '
              f'{num_bad_stories}{NORMAL_COLOR}')
    else:
        print(f'Number of bad stories in test set: {num_bad_stories}')

    for i in range(len(stat_names) - 1):
        for j in range(i + 1, len(stat_names), 1):
            x_stat_name = stat_names[i]
            y_stat_name = stat_names[j]
            plot_title = f'{x_stat_name}_v_{y_stat_name}'
            print(f"Plotting {plot_title} ...")

            fig, ax = plt.subplots()

            ax.scatter(story_stats_pd[x_stat_name].array,
                       story_stats_pd[y_stat_name].array)

            # Plot random colors for each author
            for author in author_story_stats_dict:
                story_stats = author_story_stats_dict[author]
                ax.scatter(story_stats[x_stat_name].array,
                           story_stats[y_stat_name].array,
                           # color=[author_color_dict[author]
                           #        for _ in range(len(story_stats.index))],
                           c=author_color_dict[author],
                           label=author)
            ax.scatter(good_story_stats[x_stat_name].array,
                       good_story_stats[y_stat_name].array,
                       color='green', label='good')
            ax.scatter(bad_story_stats[x_stat_name].array,
                       bad_story_stats[y_stat_name].array,
                       color='red', label='bad')

            plt.title(plot_title)
            plt.xlabel(stat_names[i])
            plt.ylabel(stat_names[j])
            ax.legend()
            ax.grid(True)
            plt.savefig(os.path.join(folder_path, plot_title), dpi=600)
            plt.close()


def main():
    # k_means = quality.get_kmeans('lit')

    if len(sys.argv) != 2 or (sys.argv[1] != 'ff' and sys.argv[1] != 'lit'):
        print(f'{ERROR_COLOR}Need argument story_type: ff lit{NORMAL_COLOR}')
        sys.exit(1)

    stories_stats_pd = pd.read_csv(f'out/{sys.argv[1]}_story_values.csv',
                                 index_col=0)
    load_stories(sys.argv[1], stories_stats_pd)
    find_author_stories(sys.argv[1], stories_stats_pd)

    # stories_stats_pd = stories_stats_pd[stories_stats_pd['quote_density'] > 0]
    remove_outliers(df=stories_stats_pd)

    # author_predicts = {}
    # for author in author_story_paths:
    #     author_story_stats_list = []
    #     for story_path in author_story_paths[author]:
    #         author_story_stats_list.append(story_stats_pd[story_path])
    #     author_predicts[author] = k_means.fit_predict(author_story_stats_list)

    plot_stats(stories_stats_pd, f'graphs/{sys.argv[1]}')


if __name__ == '__main__':
    main()
