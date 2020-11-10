import os
import sys
import plac
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np

import math
import string

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

sys.path.append('.')
from tag_assignment.util import search_dir
from story.base import get_tags


def get_used_tags(tag_dir) -> list:
    """Get tags that have at least some number of stories"""

    tag_list = []
    for tag_file in os.scandir(tag_dir):
        tag_list.append(os.path.splitext(tag_file.name)[0])
    tag_list.sort()
    return tag_list


def get_input_vector(master_ngram_dict, abs_input_path, abs_save_path):
    """Return the input vector for a given story"""

    if os.path.isfile(abs_save_path):
        with open(abs_save_path) as in_file:
            ngram_values = \
                [float(v) for v in in_file.read().strip().split('\n')]
            ngram_vector = np.array(ngram_values)
    else:
        ngram_vector = np.zeros(len(master_ngram_dict))

        # TODO: Choose ngram_freq_dir or story_tfidf_dir
        with open(abs_input_path) as in_file:
            for line in in_file:
                ngram, value = line.rsplit(' ', 1)

                if ngram not in master_ngram_dict:
                    continue  # When using ngram_freq_dir

                    # When using story_tfidf_dir
                    # raise ValueError(f'{ngram} not in master_ngram')

                ngram_i = master_ngram_dict[ngram]
                ngram_vector[ngram_i] = float(value)

        with open(abs_save_path, 'w') as out_file:
            for value in ngram_vector:
                out_file.write(f'{value}\n')

    if len(ngram_vector) != 10000:
        raise ValueError(f'{len(ngram_vector)} {abs_input_path}')

    return ngram_vector


def get_vectors(story_html_dir, input_dir, master_tag_list,
                master_ngram_dict, save_input_vector_dir):
    if os.path.isdir(save_input_vector_dir):
        print(f'WARNING: Using previously computed values in '
              f'{save_input_vector_dir}')

    vector_count = 0

    for char in string.ascii_lowercase:
        os.makedirs(os.path.join(save_input_vector_dir, char), exist_ok=True)

    x_vectors = []

    stories_tag_list = []
    for rel_html_path in tqdm(
            search_dir(story_html_dir, '.html', abs_path=False)):
        abs_html_path = os.path.join(story_html_dir, rel_html_path)
        story_name = \
            os.path.splitext(rel_html_path)[0] + '.txt'

        story_tag_list = get_tags(abs_html_path)
        stories_tag_list.append(story_tag_list)

        # Get input
        # TODO: Choose ngram_freq_dir or story_tfidf_dir
        abs_file_path = os.path.join(input_dir, story_name)
        ngram_vector = get_input_vector(master_ngram_dict, abs_file_path,
                                        os.path.join(save_input_vector_dir,
                                                     story_name))
        x_vectors.append(ngram_vector)

        # if vector_count == 50:
        #     break
        vector_count += 1

    mlb = MultiLabelBinarizer()
    mlb.fit([master_tag_list])
    print(mlb.classes_)
    # Edited line 994 of /home/hyrial/.local/lib/python3.8/site-packages/sklearn/preprocessing/_label.py
    # to ignore the warning
    y_vectors = mlb.transform(stories_tag_list)

    return np.array(x_vectors), np.array(y_vectors)


def find_accuracy_per_tag(truths, predictions, tp_weight: int = 100):
    """Find the accuracy of the predictions

    Parameters
    ----------
    tp_weight
        Weigh labelled positives much more since they are very infrequent
    """

    assert truths.shape == predictions.shape, \
        f'{truths.shape} {predictions.shape}'

    BER_values = np.zeros((truths.shape[1], 4))
    TP_i = 0
    FP_i = 1
    TN_i = 2
    FN_i = 3
    n_pos = 0
    n_neg = 0
    tag_successes = np.zeros(truths.shape[1])
    for i in range(len(truths)):
        truth = truths[i]
        prediction = predictions[i]
        for j in range(len(truth)):
            if truth[j]:
                n_pos += 1
                tag_successes[j] += prediction[j] * tp_weight
                if prediction[j]:
                    BER_values[j][TP_i] += 1
            else:
                n_neg += 1
                tag_successes[j] += 1 - prediction[j]
                if not prediction[j]:
                    BER_values[j][TN_i] += 1
                    tag_successes[j] += 1

    print("Total true positives:", n_pos)
    print("Positive success rate:",
          sum(values[TP_i] for values in BER_values) / n_pos / tp_weight)
    print("Total true negatives:", n_neg)
    print("Negative success rate:",
          sum(values[TN_i] for values in BER_values) / n_neg)
    return tag_successes / (n_pos * tp_weight + n_neg)


@plac.pos('story_html_dir', 'Directory with story .html files', Path)
@plac.pos('input_dir', "Directory with values corresponding to n-grams", Path)
@plac.pos('tag_dir', 'Directory with used tags', Path)
@plac.pos('results_dir', 'Directory with results', Path)
@plac.pos('input_vector_dir', 'Directory to save input vectors', type=Path)
@plac.pos('master_ngram_set_file', "File path to Set of n-grams to use", Path)
def main(story_html_dir, input_dir, tag_dir, results_dir,
         save_input_vector_dir, master_ngram_set_file):
    master_tag_list = get_used_tags(tag_dir)

    with open(master_ngram_set_file) as in_file:
        master_ngram_list = in_file.read().strip().split('\n')
    master_ngram_dict = {tag: i for i, tag in enumerate(master_ngram_list)}

    # Get x,y for training/testing logistic model
    print("Generating x, y vectors")
    x_vectors, y_vectors = get_vectors(story_html_dir, input_dir,
                                       master_tag_list, master_ngram_dict,
                                       save_input_vector_dir)
    x_vectors = preprocessing.scale(x_vectors)
    print('Vector shapes:', x_vectors.shape, y_vectors.shape, '\n')
    x_train, x_test, y_train, y_test = train_test_split(x_vectors, y_vectors,
                                                        random_state=1)

    models = [
        # ("BinaryRelevance w/ GaussianNB", BinaryRelevance(GaussianNB())),
        # ('Decision Tree', DecisionTreeClassifier(random_state=1)),
        ('Multi-Layer Perceptron', MLPClassifier(max_iter=500, random_state=1)),
        ('Random Forest', RandomForestClassifier(random_state=1, n_jobs=-1)),
        # ('Ridge w/ CV', RidgeClassifierCV())
    ]

    while len(models) > 0:
        name, model = models.pop()

        print(f'Training {name}')

        start_time = time.time()
        model.fit(x_train, y_train)
        train_predicts = model.predict_proba(x_train)
        val_predicts = model.predict_proba(x_test)
        if name == 'Random Forest' or name == 'Decision Tree':
            # Outputs predictions in (n_labels, n_samples)
            val_predicts = \
                np.array([[1 - feature[i][0] for feature in val_predicts]
                          for i in range(len(x_test))])
            train_predicts = \
                np.array([[1 - feature[i][0] for feature in train_predicts]
                          for i in range(len(x_train))])
        elif name == 'Multi-Layer Perceptron':
            # Outputs predictions in (n_samples, n_labels) for 0
            val_predicts = (np.array(val_predicts) - 1) * -1
            train_predicts = (np.array(train_predicts) - 1) * -1

        print(f'Model: {name}')
        print(f'Regression Time: {time.time() - start_time}\n')
        print('Training results:')
        train_results = find_accuracy_per_tag(y_train, train_predicts)
        print(f'Training score: {np.mean(train_results)}')
        print(train_results)

        print()

        print('Validation results:')
        test_results = find_accuracy_per_tag(y_test, val_predicts)
        print(f'Validation score: {np.mean(test_results)}')
        print(test_results)

        # print('\nw/ log')
        # train_predictions = np.log(train_predictions)
        # test_predictions = np.log(test_predictions)
        # train_results = find_accuracy_per_tag(y_train, train_predictions)
        # test_results = find_accuracy_per_tag(y_test, test_predictions)
        # print(f'Training score: {np.mean(train_results)}')
        # print(train_results)
        # print(f'Validation score: {np.mean(test_results)}')
        # print(test_results)
        print('\n-----------------------------------------------------------\n')


if __name__ == '__main__':
    plac.call(main)
