import os
import sys
import plac
from pathlib import Path

from tqdm import tqdm
import time
import numpy as np
import pandas as pd

import math
import string

from sklearn.preprocessing import MultiLabelBinarizer, scale
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE

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


def get_input_vector(tag_ngram_dict, abs_input_path, abs_save_path):
    """Return the input vector for a given story"""

    if os.path.isfile(abs_save_path):
        with open(abs_save_path) as in_file:
            ngram_values = \
                [float(v) for v in in_file.read().strip().split('\n')]
            ngram_vector = np.array(ngram_values)
    else:
        ngram_vector = np.zeros(len(tag_ngram_dict))

        # TODO: Choose ngram_freq_dir or story_tfidf_dir
        with open(abs_input_path) as in_file:
            for line in in_file:
                ngram, value = line.rsplit(' ', 1)

                if ngram not in tag_ngram_dict:
                    continue

                ngram_i = tag_ngram_dict[ngram]
                ngram_vector[ngram_i] = float(value)

        with open(abs_save_path, 'w') as out_file:
            for value in ngram_vector:
                out_file.write(f'{value}\n')

    return ngram_vector


def get_vectors(story_html_dir, input_dir, tag_ngram_dict, tag_stories_set,
                save_input_vector_dir, n_input_samples=-1):
    if os.path.isdir(save_input_vector_dir):
        print(f'WARNING: Using previously computed values in '
              f'{save_input_vector_dir}')

    os.makedirs(save_input_vector_dir, exist_ok=True)

    x_vectors = []
    y_vectors = []
    vector_count = 0

    story_txt_files = []
    for rel_story_path_html in tqdm(
            search_dir(story_html_dir, '.html', abs_path=False)):
        rel_path = os.path.splitext(rel_story_path_html)[0]

        story_name = os.path.basename(os.path.splitext(rel_story_path_html)[0])
        rel_story_txt_file = rel_path + '.txt'
        story_txt_file = story_name + '.txt'
        story_txt_files.append(story_txt_file)

        # Get input
        # TODO: Choose ngram_freq_dir or story_tfidf_dir
        abs_file_path = os.path.join(input_dir, rel_story_txt_file)
        ngram_vector = get_input_vector(tag_ngram_dict, abs_file_path,
                                        os.path.join(save_input_vector_dir,
                                                     story_txt_file))
        x_vectors.append(ngram_vector)
        if rel_path not in tag_stories_set:
            y_vectors.append(0)
        else:
            y_vectors.append(1)

        if vector_count == n_input_samples:
            break
        vector_count += 1

    # mlb = MultiLabelBinarizer()
    # mlb.fit([master_tag_list])
    # print(mlb.classes_)
    # Edited line 994 of /home/hyrial/.local/lib/python3.8/site-packages/sklearn/preprocessing/_label.py
    # to ignore the warning
    # y_vectors = mlb.transform(stories_tag_list)

    return np.array(x_vectors), np.array(y_vectors), story_txt_files


def get_input_vectors(story_html_dir, input_dir, save_input_vector_dir,
                      master_ngram_dict, n_input_samples=-1):
    if os.path.isdir(save_input_vector_dir):
        print(f'WARNING: Using previously computed values in '
              f'{save_input_vector_dir}')

    vector_count = 0

    for char in string.ascii_lowercase:
        os.makedirs(os.path.join(save_input_vector_dir, char), exist_ok=True)

    for rel_html_path in tqdm(
            search_dir(story_html_dir, '.html', abs_path=False)):
        story_name = \
            os.path.splitext(rel_html_path)[0] + '.txt'

        # Get input
        # TODO: Choose ngram_freq_dir or story_tfidf_dir
        abs_file_path = os.path.join(input_dir, story_name)
        yield get_input_vector(master_ngram_dict, abs_file_path,
                               os.path.join(save_input_vector_dir, story_name))

        if vector_count == n_input_samples:
            break
        vector_count += 1


def get_label_vectors(story_html_dir, master_tag_list):
    stories_tag_list = []
    for rel_html_path in tqdm(
            search_dir(story_html_dir, '.html', abs_path=False)):
        abs_html_path = os.path.join(story_html_dir, rel_html_path)
        story_name = \
            os.path.splitext(rel_html_path)[0] + '.txt'

        story_tag_list = get_tags(abs_html_path)
        stories_tag_list.append(story_tag_list)

    mlb = MultiLabelBinarizer()
    mlb.fit([master_tag_list])
    print(mlb.classes_)
    # Edited line 994 of /home/hyrial/.local/lib/python3.8/site-packages/sklearn/preprocessing/_label.py
    # to ignore the warning
    y_vectors = mlb.transform(stories_tag_list)

    return np.array(y_vectors)


def find_accuracy_per_tag(truths, predictions, story_names=None):
    """Find the accuracy of the predictions"""

    n_labels = 1
    n_samples = len(truths)
    if predictions.ndim == 2:
        n_labels = predictions.shape[1]
        assert truths.shape == predictions.shape, \
            f'{truths.shape} {predictions.shape}'

    BER_values = np.zeros((n_labels, 4))
    TP_i = 0
    FP_i = 1
    TN_i = 2
    FN_i = 3
    n_pos = 0
    n_neg = 0
    for i in range(n_samples):
        truth = truths[i]
        prediction = predictions[i]
        for j in range(n_labels):
            tag_truth = truth if n_labels == 1 else truth[j]
            tag_pred = prediction if n_labels == 1 else prediction[j]
            tag_pred = 1 if tag_pred >= 0.5 else 0
            if tag_truth:
                n_pos += 1
                if tag_pred:
                    BER_values[j][TP_i] += 1
                else:
                    if story_names:
                        print(story_names[i])
                    BER_values[j][FN_i] += 1
            else:
                n_neg += 1
                if not tag_pred:
                    BER_values[j][TN_i] += 1
                else:
                    BER_values[j][FP_i] += 1

    total_TP = int(sum(values[TP_i] for values in BER_values))
    total_FP = int(sum(values[FP_i] for values in BER_values))
    total_TN = int(sum(values[TN_i] for values in BER_values))
    total_FN = int(sum(values[FN_i] for values in BER_values))

    # Use when minimizing false positives
    try:
        precision = total_TP / (total_TP + total_FP)
    except ZeroDivisionError:
        precision = np.nan

    # Use when minimizing false negatives
    try:
        recall = total_TP / (total_TP + total_FN)
    except ZeroDivisionError:
        recall = np.nan

    f_scores = []
    for values in BER_values:
        TP = values[TP_i]
        FP = values[FP_i]
        FN = values[FN_i]

        f_scores.append(TP / (TP + 0.5 * (FP + FN)))

    print('f-scores', f_scores)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    assert n_pos + n_neg == n_samples * n_labels
    print(f"Number of samples: {n_samples}")
    print(f'Number of labels: {n_labels}')
    print(f"Number of predictions: {n_pos + n_neg}")
    print(f'Number of negatives: {n_neg}')
    print(f'Number of positives: {n_pos}')

    try:
        combined_f_score = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        combined_f_score = np.nan
    print(f'Total f-scores: {combined_f_score}')


def test_model(x_vectors, y_vectors, story_names, model):
    print('Vector shapes:', x_vectors.shape, '\n')
    x_train, x_test, y_train, y_test, story_names_train, story_names_test = \
        train_test_split(x_vectors, y_vectors, story_names, random_state=1)

    sm = SMOTE(random_state=2)
    x_train_res, y_train_res = sm.fit_sample(x_train, y_train)

    # class_weight = [{0: 1, 1: 100} for _ in range(len(master_tag_list))]

    start_time = time.time()
    model.fit(x_train_res, y_train_res)
    if hasattr(model, 'predict_proba'):
        train_predicts = model.predict_proba(x_train_res)
        val_predicts = model.predict_proba(x_test)

        print(val_predicts.shape)
        print(model.classes_)

        # Outputs predictions in (n_samples, 2) when outputting only 1 var
        val_predicts = \
            np.array([1 - feature[0] for feature in val_predicts])
        train_predicts = \
            np.array([1 - feature[0] for feature in train_predicts])
    else:
        train_predicts = model.predict(x_train_res)
        val_predicts = model.predict(x_test)

        print(val_predicts.shape)
        print(val_predicts[:5])

        # if name in ('Random Forest', 'Logistic Regression'):
        #    # Outputs predictions in (n_labels, n_samples)
        #    # val_predicts = \
        #    #    np.array([[1 - feature[i][0] for feature in val_predicts]
        #    #               for i in range(len(x_test))])
        #    # train_predicts = \
        #    #     np.array([[1 - feature[i][0] for feature in train_predicts]
        #    #               for i in range(len(x_train))])
        # elif name == 'Multi-Layer Perceptron':
        #    # Outputs predictions in (n_samples, n_labels) for 0
        #    # val_predicts = (np.array(val_predicts) - 1) * -1
        #    # train_predicts = (np.array(train_predicts) - 1) * -1

    # x_train = list(x_train)
    # for i, predict in enumerate(train_predicts):
    #     x_train[i] = np.append(x_train[i], predict)
    # x_train = np.array(x_train)
    #
    # x_test = list(x_test)
    # for i, predict in enumerate(val_predicts):
    #     x_test[i] = np.append(x_test[i], predict)
    # x_test = np.array(x_test)

    print(f'Regression Time: {time.time() - start_time}\n')
    print('Training results:')
    find_accuracy_per_tag(y_train_res, train_predicts)

    print()

    print('Validation results:')
    find_accuracy_per_tag(y_test, val_predicts, story_names_test)

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


def test_weights(x_vectors, y_vectors, story_names, tag_name, tag_stats_dir):
    with open(os.path.join(tag_stats_dir, tag_name + '.txt')) as in_file:
        df = pd.read_table(os.path.join(tag_stats_dir, tag_name + '.txt'))
        


@plac.pos('story_html_dir', 'Directory with story .html files', Path)
@plac.pos('input_dir', "Directory with values corresponding to n-grams", Path)
@plac.pos('tag_stats_dir', 'Directory with tag TF-IDFs and other stats', Path)
@plac.pos('tag_stories_dir', 'Directory with list of stories per tag', Path)
@plac.pos('input_vector_dir', 'Directory to save input vectors', Path)
@plac.opt('n_input_samples', 'Number of input samples to use', int)
def main(story_html_dir, input_dir, tag_stats_dir, tag_stories_dir,
         save_vector_dir, n_input_samples=-1):
    master_tag_list = get_used_tags(tag_stats_dir)

    for tag_name in master_tag_list:
        print(f'Testing tag: {tag_name}')
        tag_file = tag_name + '.txt'

        with open(os.path.join(tag_stats_dir, tag_file)) as in_file:
            tag_ngram_list = in_file.read().strip().split('\n')
        tag_ngram_dict = \
            {line.split('\t', 1)[0]: i for i, line in enumerate(tag_ngram_list)}

        with open(os.path.join(tag_stories_dir, tag_file)) as in_file:
            tag_stories_list = in_file.read().strip().split('\n')
        tag_stories_set = set(tag_stories_list)

        save_tag_dir = os.path.join(save_vector_dir, tag_name)

        print("Generating x, y vectors")
        orig_x_vectors, orig_y_vector, story_names = \
            get_vectors(story_html_dir, input_dir, tag_ngram_dict,
                        tag_stories_set, save_tag_dir, n_input_samples)
        orig_x_vectors = scale(orig_x_vectors)

        test_weights(orig_x_vectors, orig_y_vector, story_names, tag_name,
                     tag_stats_dir)
        return

        model_list = [
            # ('Adaboost Classifier', AdaBoostClassifier(random_state=1)),
            # ('LinearSVC', LinearSVC(random_state=1)),
            # ('Random Forest', RandomForestClassifier(random_state=1, n_jobs=-1)),
            ("Logistic Regression", LogisticRegression(n_jobs=-1)),
            # ('SGDClassifier', SGDClassifier(random_state=1)),
            # ('BenoulliNB', BernoulliNB()),
            # ('GaussianNB', GaussianNB()),
            # ('Multi-Layer Perceptron', MLPClassifier(max_iter=500, random_state=1)),
        ]

        while len(model_list) > 0:
            model_name, model = model_list.pop()
            print(f'Testing model: {model_name}')

            test_model(orig_x_vectors, orig_y_vector, story_names, model)


if __name__ == '__main__':
    plac.call(main)
