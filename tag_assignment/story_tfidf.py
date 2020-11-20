import sys
from pathlib import Path

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


def ngram_freqs_batch_func(nlp, batch_id, batch, ngram_freq_dir, lemm_list_dir):
    print('Batch ID:', batch_id)

    for rel_story_path, story_text in batch:
        lemm_list_path = os.path.join(lemm_list_dir, rel_story_path)

        if os.path.isfile(lemm_list_path):
            with open(lemm_list_path) as in_file:
                lemm_str = in_file.read().strip()
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


def find_ngram_freqs(story_text_dir, lemm_list_dir, ngram_freq_dir,
                     n_files_to_use=-1, batch_size=1000, n_jobs=-1):
    if not os.path.isdir(story_text_dir):
        raise NotADirectoryError(story_text_dir)

    start_time = time.time()
    partitions = minibatch(load_dir(story_text_dir, '.txt', n_files_to_use),
                           size=batch_size)
    executor = Parallel(n_jobs=n_jobs, backend="multiprocessing",
                        prefer="processes")
    do = delayed(partial(ngram_freqs_batch_func, spacy_nlp))

    tasks = (do(i, batch, ngram_freq_dir, lemm_list_dir)
             for i, batch in enumerate(partitions))

    print('Finding n-gram frequencies')
    executor(tasks)
    print('Total time (hours):', (time.time() - start_time) / 3600)


def get_combined_freq(ngram_freq_dir, ngram_doc_freq_file):

    print('Finding n-gram combined frequencies')
    ngram_doc_freq = {}
    min_doc_freq = 3
    n_docs = 0
    for file_path in tqdm(search_dir(ngram_freq_dir, '.txt')):
        n_docs += 1

        with open(file_path) as in_file:
            for line in in_file:
                data = line.rsplit(' ', 1)  # Since n-grams have multiple words
                ngram = data[0]

                # Avoid n-grams of over length 3 for now
                if len(ngram.split()) > 3:
                    continue

                try:
                    if ngram not in ngram_doc_freq:
                        ngram_doc_freq[ngram] = 0
                except MemoryError as e:
                    print("Size of ngram_doc_freq:", len(ngram_doc_freq))
                    print(e)
                    sys.exit()

                ngram_doc_freq[ngram] += 1

        # Dictionaries use about 1.2 GB / 10mil items
        # 10GB / 1.2 = 8
        if len(ngram_doc_freq) > 60000000:
            to_delete = []
            for ngram, doc_freq in ngram_doc_freq.items():
                if doc_freq <= min_doc_freq:
                    to_delete.append(ngram)
            for ngram in to_delete:
                del ngram_doc_freq[ngram]
            min_doc_freq += 1
            print(len(ngram_doc_freq))

    # Use arbitrary number to filter out obviously low frequency n-grams to keep
    # file size small
    min_n = 0.01 * n_docs
    ngram_doc_freq = {k: v for k, v in ngram_doc_freq.items() if v >= min_n}
    dict_to_file(os.path.join(ngram_doc_freq_file), ngram_doc_freq)

    return ngram_doc_freq, n_docs


def calculate_tfidf_batch_func(rel_story_paths, ngram_freq_dir, tf_idf_dir,
                               ngram_doc_freq, n_docs, master_ngram_set):
    pass


def calculate_tfidf(ngram_freq_dir: str, tf_idf_dir: str, ngram_doc_freq: dict,
                    n_docs: int):
    print('Calculating TF-IDF for each text')

    # partitions = minibatch(search_dir(ngram_freq_dir, '.txt', abs_path=False))
    # executor = Parallel(n_jobs=n_jobs, backend="multiprocessing",
    #                     prefer="processes")

    for rel_story_path in tqdm(
            search_dir(ngram_freq_dir, '.txt', abs_path=False)):
        abs_file_path = os.path.join(ngram_freq_dir, rel_story_path)
        ngram_freq = {}

        story_tfidf = {}
        total_freq = 0
        with open(abs_file_path) as in_file:
            for line in in_file:
                data = line.rsplit(' ', 1)
                ngram = data[0]

                if ngram not in ngram_doc_freq:
                    continue

                ngram_len = len(ngram.split())

                term_freq = int(data[1]) * (1.5 ** (ngram_len - 1))
                total_freq += term_freq
                ngram_freq[ngram] = term_freq

        most_freq_ngram_count = max(ngram_freq.values())

        for ngram, term_freq in ngram_freq.items():
            # ngram_len = len(ngram.split())

            # Prevent a bias towards larger documents (Source: Wikipedia)
            # tf = 0.5 + 0.5 * freq / most_freq_ngram_count
            # tf = term_freq / total_freq
            # Longer stories should have more tags
            tf = term_freq / most_freq_ngram_count
            idf = math.log(n_docs / ngram_doc_freq[ngram])
            story_tfidf[ngram] = tf * idf

        dict_to_file(os.path.join(tf_idf_dir, rel_story_path), story_tfidf,
                     write_zeros=False)


@plac.pos('story_text_dir', "Directory with .txt files", Path)
@plac.pos('lemm_list_dir', "Directory with list of lemmatized words", Path)
@plac.pos('ngram_freq_dir', "Directory with n-gram frequencies", Path)
@plac.pos('story_tfidf_dir', "Output directory for TF-IDF of stories", Path)
@plac.pos('ngram_doc_freq_file', "File path to document freq of n-grams", Path)
@plac.opt('n_files_to_use', "Number of files to use in corpus", int, 'f')
@plac.opt('batch_size', "Number of files in each batch", int)
@plac.opt('n_jobs', "Number of jobs", int, 'j')
@plac.opt('overwrite', "Option to overwrite previous lemm_list_dir", bool)
@plac.opt('skip_find_ngram_freqs', "Option to skip finding n-gram freqs", bool)
def main(story_text_dir, lemm_list_dir, ngram_freq_dir, story_tfidf_dir,
         ngram_doc_freq_file, n_files_to_use=-1, batch_size=1000, n_jobs=-1,
         overwrite=False, skip_find_ngram_freqs=False):
    """Calculate the TF-IDF of given text files"""

    if not skip_find_ngram_freqs:
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

        find_ngram_freqs(story_text_dir, lemm_list_dir, ngram_freq_dir,
                         n_files_to_use, batch_size, n_jobs)

    ngram_doc_freq, n_docs = \
        get_combined_freq(ngram_freq_dir, ngram_doc_freq_file)

    calculate_tfidf(ngram_freq_dir, story_tfidf_dir, ngram_doc_freq, n_docs)


if __name__ == '__main__':
    plac.call(main)
