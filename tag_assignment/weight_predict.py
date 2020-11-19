import os
from pathlib import Path

import pandas as pd
import numpy as np
import plac
import sys

from tqdm import tqdm
from pandasgui import show
from sklearn.preprocessing import scale

sys.path.append('.')
from story.base import get_story_tags
from tag_assignment.util import *


def get_input_vectors(rel_story_list, input_dir, save_input_vector_dir,
                      df, n_input_samples, stories_to_use=None):

    print(f'Finding input vectors')

    if os.path.isdir(save_input_vector_dir):
        print(f'WARNING: Using previously computed values in '
              f'{save_input_vector_dir}')

    ngram_index_dict = {}
    for ngram, row in df.iterrows():
        row_index = row['row_index']
        ngram_index_dict[ngram] = int(row_index)

    n_inputs = 0
    for rel_html_path in rel_story_list:
        rel_path = os.path.splitext(rel_html_path)[0]

        story_name = os.path.basename(os.path.splitext(rel_html_path)[0])
        if stories_to_use and story_name not in stories_to_use:
            yield story_name, None

        else:
            rel_story_txt_file = rel_path + '.txt'
            story_txt_file = story_name + '.txt'

            # Get input
            # TODO: Choose ngram_freq_dir or story_tfidf_dir
            abs_input_path = os.path.join(input_dir, rel_story_txt_file)
            abs_save_path = os.path.join(save_input_vector_dir, story_txt_file)
            yield story_name, get_input_vector(ngram_index_dict, abs_input_path,
                                               abs_save_path)

        if n_inputs == n_input_samples:
            break
        n_inputs += 1


def get_tag_story_labels(abs_story_list, master_tag_list, n_input_samples):
    """Create and return a vector for each tag

    For each story index, its corresponding index for each tag will denote if
    the story includes that tag.
    """

    print('Generating y vectors for each tag')

    tag_story_labels = {}
    for tag in master_tag_list:
        tag_story_labels[tag] = []

    n_inputs = 0

    for abs_html_path in tqdm(abs_story_list):

        story_tag_set = get_story_tags(abs_html_path)
        for tag in master_tag_list:
            if tag not in story_tag_set:
                tag_story_labels[tag].append(0)
            else:
                tag_story_labels[tag].append(1)

        if n_inputs == n_input_samples:
            break
        n_inputs += 1

    return tag_story_labels


def test_weights(tag_text, df, x_test_vectors, train_story_set,
                 valid_story_set, results_dir, n_ngrams=500):
    print(f'Testing weights for {tag_text}')

    # df should be sorted by weight
    weight_vector = np.array(df['weight'])[:n_ngrams]
    sum_weight_vector = sum(weight_vector)

    os.makedirs(results_dir, exist_ok=True)
    train_file = open(os.path.join(results_dir, f'{tag_text}_train_results.txt'), 'w')
    valid_file = open(os.path.join(results_dir, f'{tag_text}_valid_results.txt'), 'w')
    train_error = 0
    valid_error = 0
    avg_value = 0
    for story_name, x_vector in x_test_vectors:
        if story_name in train_story_set:
            x_test_vector = x_vector[:n_ngrams]
            value = np.dot(weight_vector, x_test_vector) / sum_weight_vector
            train_file.write(f'{story_name} {value}\n')
            # print('Train:', story_name, value)
            train_error += 0 if value > 1 else (1 - value)
            df[story_name] = x_vector
            avg_value += value
        elif story_name in valid_story_set:
            x_test_vector = x_vector[:n_ngrams]
            value = np.dot(weight_vector, x_test_vector) / sum_weight_vector
            valid_file.write(f'{story_name} {value}\n')
            # print('Validation:', story_name, value)
            valid_error += 0 if value > 1 else (1 - value)
            avg_value += value
        else:
            assert x_vector is None, story_name
    train_file.close()
    valid_file.close()

    if tag_text == 'bdsm':
        show(df)

    print(f'Train Error: {train_error / len(train_story_set)}')
    print(f'Valid Error: {valid_error / len(valid_story_set)}')
    print(f'Avg Value: {avg_value / (len(train_story_set) + len(valid_story_set))}')
    print()


def predict_tag(tag_text, df, x_vectors, y_vector, results_dir, n_ngrams=500):
    print(f'Predicting stories that have {tag_text}')

    # df should be sorted by weight
    weight_vector = np.array(df['weight'])[:n_ngrams]
    sum_weight_vector = sum(weight_vector)

    os.makedirs(results_dir, exist_ok=True)
    predict_file = open(os.path.join(results_dir, f'{tag_text}_predictions.txt'), 'w')
    avg_value = 0
    for story_vector, has_tag in zip(x_vectors, y_vector):
        if has_tag:
            continue

        story_name, x_vector = story_vector
        x_test_vector = x_vector[:n_ngrams]
        value = np.dot(weight_vector, x_test_vector) / sum_weight_vector
        if value > 1:
            predict_file.write(f'{story_name} {value}\n')
            print(story_name, value)
            df[story_name] = x_vector
        avg_value += value
    if tag_text == 'anal':
        show(df)


@plac.pos('story_html_dir', 'Directory with story .html files', Path)
@plac.pos('input_dir', "Directory with values corresponding to n-grams", Path)
@plac.pos('tag_stats_dir', 'Directory with tag TF-IDFs and other stats', Path)
@plac.pos('tag_stories_dir', 'Directory with list of stories per tag', Path)
@plac.pos('save_vectors_dir', 'Directory to save input vectors', Path)
@plac.pos('results_dir', 'Directory to output results', Path)
@plac.opt('n_input_samples', 'Number of input samples to use', int)
def main(story_html_dir, input_dir, tag_stats_dir, tag_stories_dir,
         save_vectors_dir, results_dir, n_input_samples=-1):
    master_tag_list = get_tags_to_predict(tag_stats_dir)

    # List of all story paths in html
    rel_html_list = search_dir(story_html_dir, '.html', abs_path=False)
    abs_html_list = \
        [os.path.join(story_html_dir, path) for path in rel_html_list]

    # tag_story_labels = get_tag_story_labels(abs_html_list, master_tag_list,
    #                                         n_input_samples)

    for tag_text in master_tag_list:
        print(f'Testing tag: {tag_text}')

        # Directory to save input vectors to avoid making them again
        save_tag_vector_dir = os.path.join(save_vectors_dir, tag_text)
        os.makedirs(save_tag_vector_dir, exist_ok=True)

        df = pd.read_table(os.path.join(tag_stats_dir, tag_text + '.txt'),
                           index_col='ngram')
        df.name = tag_text
        df['row_index'] = [x for x in range(len(df))]
        df = df.infer_objects()

        # Get set of stories with this tag
        tag_file = tag_text + '_TRAIN.txt'
        train_stories_set = get_tag_stories(os.path.join(tag_stories_dir,
                                                         tag_file))
        tag_file = tag_text + '_VALID.txt'
        valid_stories_set = get_tag_stories(os.path.join(tag_stories_dir,
                                                         tag_file))

        # Remove the alphabetical folder separation prefix
        train_stories_set = set([story[2:] for story in train_stories_set])
        valid_stories_set = set([story[2:] for story in valid_stories_set])

        x_test_vectors = \
            get_input_vectors(rel_html_list, input_dir, save_tag_vector_dir, df,
                              n_input_samples,
                              valid_stories_set.union(train_stories_set))

        test_weights(tag_text, df, x_test_vectors, train_stories_set,
                     valid_stories_set, results_dir)

        # x_vectors = \
        #     get_input_vectors(rel_html_list, input_dir, save_tag_vector_dir, df,
        #                       n_input_samples)
        # predict_tag(tag_text, df, x_vectors, tag_story_labels[tag_text],
        #             results_dir)


if __name__ == '__main__':
    plac.call(main)
