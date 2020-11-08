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


def batch_func(nlp, batch_id, batch, ngram_freq_dir, lemm_list_dir):
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


def find_ngram_freqs(story_text_dir, ngram_freq_dir, lemm_list_dir,
                     n_files_to_use=-1, batch_size=1000, n_jobs=4):
    if not os.path.isdir(story_text_dir):
        raise NotADirectoryError(story_text_dir)

    start_time = time.time()
    partitions = minibatch(load_dir(story_text_dir, '.txt', n_files_to_use),
                           size=batch_size)
    executor = Parallel(n_jobs=n_jobs, backend="multiprocessing",
                        prefer="processes")
    do = delayed(partial(batch_func, spacy_nlp))

    tasks = (do(i, batch, ngram_freq_dir, lemm_list_dir)
             for i, batch in enumerate(partitions))

    print('Executing tasks')
    executor(tasks)
    print('Total time (hours):', (time.time() - start_time) / 3600)


def get_combined_freq(ngram_freq_dir, ngram_doc_freq_file,
                      ngram_total_freq_file, master_ngram_set_file, n_ngrams):
    ngram_doc_freq = {}
    ngram_total_freq = {}
    n_docs = 0
    for file_path in tqdm(search_dir(ngram_freq_dir, '.txt')):
        n_docs += 1

        with open(file_path) as in_file:
            for line in in_file:
                data = line.rsplit(' ', 1)  # Since n-grams have multiple words

                ngram = data[0]
                freq = int(data[1])

                if ngram not in ngram_total_freq:
                    ngram_total_freq[ngram] = 0
                ngram_total_freq[ngram] += freq

                if ngram not in ngram_doc_freq:
                    ngram_doc_freq[ngram] = 0
                ngram_doc_freq[ngram] += 1

    # Use arbitrary number to filter out obviously low frequency n-grams to keep
    # file size small
    ngram_doc_freq = {k: v for k, v in ngram_doc_freq.items() if v > 3}
    dict_to_file(os.path.join(ngram_doc_freq_file), ngram_doc_freq)

    ngram_total_freq = \
        {k: v for k, v in sorted(ngram_total_freq.items(), key=lambda x: x[1],
                                 reverse=True)[:n_ngrams]}
    dict_to_file(os.path.join(ngram_total_freq_file), ngram_total_freq)
    with open(os.path.join(master_ngram_set_file), 'w') as out:
        for ngram in ngram_total_freq.keys():
            out.write(f'{ngram}\n')

    return ngram_doc_freq, n_docs, set(ngram_total_freq.keys())


def calculate_tfidf(ngram_freq_dir: str, tf_idf_dir: str, ngram_doc_freq: dict,
                    n_docs: int, master_ngram_set: set):
    for rel_story_path in tqdm(
            search_dir(ngram_freq_dir, '.txt', abs_path=False)):
        abs_file_path = os.path.join(ngram_freq_dir, rel_story_path)
        ngram_freq = {}

        story_tfidf = {k: 0 for k in master_ngram_set}
        total_freq = 0
        with open(abs_file_path) as in_file:
            for line in in_file:
                data = line.rsplit(' ', 1)
                ngram = data[0]

                if ngram not in master_ngram_set:
                    continue

                term_freq = int(data[1])
                total_freq += term_freq
                ngram_freq[ngram] = term_freq

        # most_freq_ngram_count = max(ngram_freq.values())

        for ngram, term_freq in ngram_freq.items():
            # Prevent a bias towards larger documents (Source: Wikipedia)
            # tf = 0.5 + 0.5 * freq / most_freq_ngram_count
            tf = term_freq / total_freq
            idf = math.log(n_docs / ngram_doc_freq[ngram])
            story_tfidf[ngram] = tf * idf

        dict_to_file(os.path.join(tf_idf_dir, rel_story_path), story_tfidf)


@plac.pos('story_text_dir', "Directory with .txt files", Path)
@plac.pos('ngram_freq_dir', "Directory with n-gram frequencies", Path)
@plac.pos('lemm_list_dir', "Directory with list of lemmatized words", Path)
@plac.pos('story_tfidf_dir', "Output directory for TF-IDF of stories", Path)
@plac.pos('ngram_doc_freq_file', "File path to document freq of n-grams", Path)
@plac.pos('ngram_total_freq_file', "File path to total freq of n-grams", Path)
@plac.pos('master_ngram_set_file', "File path to Set of n-grams to use", Path)
@plac.opt('n_files_to_use', "Number of files to use in corpus", int, 'f')
@plac.opt('batch_size', "Number of files in each batch", int)
@plac.opt('n_jobs', "Number of jobs", int, 'j')
@plac.opt('n_ngrams', "Number of n-grams to use", int, 'n')
@plac.opt('overwrite', "Option to overwrite previous lemm_list_dir", bool)
def main(story_text_dir, ngram_freq_dir, lemm_list_dir, story_tfidf_dir,
         ngram_doc_freq_file, ngram_total_freq_file, master_ngram_set_file,
         n_files_to_use=-1, batch_size=1000, n_jobs=4, n_ngrams=10000,
         overwrite=False):
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
    ngram_doc_freq, n_docs, master_ngram_set = \
        get_combined_freq(ngram_freq_dir, ngram_doc_freq_file,
                          ngram_total_freq_file, master_ngram_set_file,
                          n_ngrams)

    print('Calculating TF-IDF for each text')
    calculate_tfidf(ngram_freq_dir, story_tfidf_dir, ngram_doc_freq, n_docs,
                    master_ngram_set)


if __name__ == '__main__':
    plac.call(main)
