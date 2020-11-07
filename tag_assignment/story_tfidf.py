import os
import math
import sys
from pathlib import Path

from spacy.tokens import Doc
from spacy.util import minibatch
from joblib import Parallel, delayed
from functools import partial
from tqdm import tqdm
import plac
import time
import shutil

sys.path.append('.')
from story_stats.util import *
from tag_assignment.util import *


def batch_func(nlp, batch_id, batch, ngram_freq_dir, lemm_list_dir):

    print('Batch ID:', batch_id)

    for rel_story_path, story_text in batch:
        lemm_list_path = os.path.join(lemm_list_dir, rel_story_path)

        if os.path.isfile(lemm_list_path):
            with open(lemm_list_path) as in_file:
                lemm_str = in_file.read()
                lemm_list = lemm_str.split('\n')

        else:
            story_text = expand_contractions(story_text)
            text_list = [story_text]
            if len(story_text) > MAX_STORY_LEN:
                text_list = split_text(story_text, MAX_STORY_LEN)

            lemm_list = []
            for text in text_list:  # Some texts need to be split for spacy
                doc = nlp(text)
                lemm_list.extend(get_lemm_list(doc))

            with open(lemm_list_path, 'w') as out_file:
                for lemm in lemm_list:
                    out_file.write(f'{lemm}\n')

        ngram_list = get_ngram_list(lemm_list)
        ngram_freq = get_ngram_freq(ngram_list)

        dict_to_file(os.path.join(ngram_freq_dir, rel_story_path), ngram_freq)


def get_ngram_freq(ngram_list: List[str]):
    ngram_freq = {}
    for ngram in ngram_list:
        if ngram not in ngram_freq:
            ngram_freq[ngram] = 0
        ngram_freq[ngram] += 1
    return ngram_freq


def find_ngram_freqs(story_text_dir, ngram_freq_dir, lemm_list_dir,
                     n_files_to_use=-1, batch_size=5000, n_jobs=4):
    if not os.path.isdir(story_text_dir):
        raise NotADirectoryError(story_text_dir)

    if not os.path.isdir(ngram_freq_dir):
        os.mkdir(ngram_freq_dir)

    if not os.path.isdir(lemm_list_dir):
        os.mkdir(lemm_list_dir)

    start_time = time.time()
    print('Creating batches')
    partitions = minibatch(load_dir(story_text_dir, '.txt', n_files_to_use),
                           size=batch_size)
    executor = Parallel(n_jobs=n_jobs, backend="multiprocessing",
                        prefer="processes")
    do = delayed(partial(batch_func, spacy_nlp))

    print('Creating tasks')
    tasks = (do(i, batch, ngram_freq_dir, lemm_list_dir)
             for i, batch in enumerate(partitions))

    print('Executing tasks')
    executor(tasks)
    print('Total time (hours):', (time.time() - start_time) / 3600)


def get_ngram_doc_freq(ngram_freq_dir):

    ngram_doc_freq = {}
    n_docs = 0
    for file_path in tqdm(search_dir(ngram_freq_dir, '.txt')):

        n_docs += 1

        ngram_set = set()
        with open(file_path) as in_file:
            for line in in_file:
                data = line.rsplit(' ', 1)  # Since n-grams have multiple words
                assert len(data) == 2, data

                ngram = data[0]
                ngram_set.add(ngram)

            for ngram in ngram_set:
                if ngram not in ngram_doc_freq:
                    ngram_doc_freq[ngram] = 0

                ngram_doc_freq[ngram] += 1

    if not os.path.isdir('out'):
        os.mkdir('out')

    dict_to_file('out/ngram_doc_freq.txt', ngram_doc_freq)

    return ngram_doc_freq, n_docs


def calculate_tfidf(ngram_freq_dir: str, tf_idf_dir: str,
                    ngram_doc_freq: dict, n_docs: int):

    if not os.path.isdir(tf_idf_dir):
        os.mkdir(tf_idf_dir)

    for file_path in tqdm(search_dir(ngram_freq_dir, '.txt')):
        file_name = file_path.split('/')[-1]
        ngram_freq = {}
        n_ngrams = 0

        tfidf = {}
        with open(file_path) as in_file:
            for line in in_file:
                data = line.rsplit(' ', 1)
                ngram = data[0]
                freq = int(data[1])

                assert ngram not in ngram_freq

                n_ngrams += freq
                ngram_freq[ngram] = freq

        for ngram, freq in ngram_freq.items():
            tf = freq / n_ngrams
            idf = math.log(n_docs / ngram_doc_freq[ngram])
            tfidf[ngram] = tf * idf

        dict_to_file(os.path.join(tf_idf_dir, file_name), tfidf)


@plac.pos('story_text_dir', "Directory with .txt files", Path)
@plac.pos('ngram_freq_dir', "Directory with n-gram frequencies", Path)
@plac.pos('lemm_list_dir', "Directory with list of lemmatized words", Path)
@plac.pos('tfidf_dir', "Output directory for TF-IDF of stories", Path)
@plac.opt('n_files_to_use', "Number of files to use in corpus", int)
@plac.opt('batch_size', "Number of files in each batch", int)
@plac.opt('n_jobs', "Number of jobs", int, 'j')
@plac.opt('overwrite', "Option to overwrite all previous values", bool)
def main(story_text_dir, ngram_freq_dir, lemm_list_dir, tfidf_dir,
         n_files_to_use=-1, batch_size=5000, n_jobs=4, overwrite=False):
    """Calculate the TF-IDF of given text files"""

    if overwrite:
        confirm_overwrite = input(f'Confirm deletion of "{lemm_list_dir}"? ')
        if confirm_overwrite == 'y':
            if os.path.isdir(lemm_list_dir):
                print(f'Deleting directory {lemm_list_dir}')
                shutil.rmtree(lemm_list_dir)
            else:
                print(f'WARNING: {lemm_list_dir} does not exist')
        else:
            return
    else:
        print(f'Using previous computed values in {lemm_list_dir}')

    print('Finding n-gram frequencies')
    find_ngram_freqs(story_text_dir, ngram_freq_dir, lemm_list_dir,
                     n_files_to_use, batch_size, n_jobs)

    print('Finding n-gram doc frequencies')
    ngram_doc_freq, n_docs = get_ngram_doc_freq(ngram_freq_dir)

    print('Calculating TF-IDF for each text')
    calculate_tfidf(ngram_freq_dir, tfidf_dir, ngram_doc_freq, n_docs)


if __name__ == '__main__':
    plac.call(main)
