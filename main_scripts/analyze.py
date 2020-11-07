import sys
import time
import os

from sklearn.cluster import KMeans
import pandas as pd
from tqdm import tqdm

from story.base import BaseStory
from story_stats.story_stats import get_story_stats, STAT_NAME_FUNCS, \
    get_stories_stats
from story_stats.util import remove_outliers, word_freq_dict

story_types = {
    'lit': '../Stories',
    'ff': '../fanfic',
    'test': 'test_set/test_stories'
}
STORY_SAMPLE_SIZE = 8000
WARNING_COLOR = '\033[93m'
ERROR_COLOR = '\033[91m'
NORMAL_COLOR = '\033[0m'
GREEN_COLOR = '\033[92m'


def get_k_means(story_type: str) -> KMeans:

    if story_type not in story_types:
        raise ValueError(f'{ERROR_COLOR}{story_type} is not a valid story '
                         f'type{NORMAL_COLOR}')

    stories_stats_pd = get_stories_stats(
        story_types[story_type],
        f'out/{story_type}_short_story_list.txt',
        output_file_path=f'out/{story_type}_story_values.csv',
        stories_stats_pd_file_path=f'out/{story_type}_story_values.csv'
    )

    return None

    # Remove stories with no quotes
    stories_stats_pd = stories_stats_pd[stories_stats_pd['quote_density'] > 0]

    # Remove story outliers
    remove_outliers(df=stories_stats_pd)

    k_means = KMeans(n_clusters=2, random_state=0).fit(stories_stats_pd)
    return k_means


# Split each text into smaller ones of size at least min_sample_Len
# Find the relevant statistics of that
# Get the average statistic for each text
# Perform K-means (k=2) to separate good and bad stories
def main():

    if len(sys.argv) != 2:
        print(f'{ERROR_COLOR}Need to provide a type of story: lit, ff, test')
        return

    start_time = time.time()
    k_means = get_k_means(sys.argv[1])
    print('Time: ', time.time() - start_time)

    # if sys.argv[1] == 'test':
    #     os.remove('out/test_short_story_list.txt')
    #     os.remove('out/test_story_values.csv')

    # results = kmeans.fit_predict(cleaned_story_values)
    # type0_count = 0
    # type1_count = 0
    # for result in results:
    #     if result == 0:
    #         type0_count += 1
    #     elif result == 1:
    #         type1_count += 1
    #     else:
    #         print(f'{WARNING_COLOR}WARNING: Unknown result: {result}{NORMAL_COLOR}')
    #
    # print(f'Type0: {type0_count}')
    # print(f'Type1: {type1_count}')


if __name__ == "__main__":
    main()
