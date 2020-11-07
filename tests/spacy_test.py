#!/usr/bin/env python
# coding: utf8
"""Example of multi-processing with Joblib. Here, we're exporting
part-of-speech-tagged, true-cased, (very roughly) sentence-separated text, with
each "sentence" on a newline, and spaces between tokens. Data is loaded from
the IMDB movie reviews dataset and will be loaded automatically via Thinc's
built-in dataset loader.

Compatible with: spaCy v2.0.0+
Last tested with: v2.1.0
Prerequisites: pip install joblib
"""
from __future__ import print_function, unicode_literals

from pathlib import Path
from joblib import Parallel, delayed
from functools import partial
import thinc.extra.datasets
import plac
import spacy
from spacy.util import minibatch
import sys
import time


@plac.annotations(
    model=("Model name (needs tagger)", "positional", None, str),
    n_jobs=("Number of workers", "option", "n", int),
    batch_size=("Batch-size for each process", "option", "b", int),
    limit=("Limit of entries from the dataset", "option", "l", int),
)
def main(model="en_core_web_sm", n_jobs=4, batch_size=500, limit=10000):
    nlp = spacy.load(model, disable=['ner', 'parser'])  # load spaCy model
    print("Loaded model '%s'" % model)
    # load and pre-process the IMBD dataset
    print("Loading IMDB data...")
    data, _ = thinc.extra.datasets.imdb()
    texts, _ = zip(*data[-limit:])
    texts = texts[0:3000]
    print("Processing texts...")
    start_time = time.time()
    partitions = minibatch(texts, size=batch_size)
    executor = Parallel(n_jobs=n_jobs, backend="multiprocessing", prefer="processes")
    do = delayed(partial(find_lemma, nlp))
    tasks = (do(i, batch) for i, batch in enumerate(partitions))
    a = executor(tasks)
    print(a)
    print('Total time:', time.time() - start_time)


def find_lemma(nlp, batch_id, texts):
    print('Batch ID:', batch_id)
    lemma_list = []
    for doc in nlp.pipe(texts):
        for token in doc:
            lemma_list.append(token.lemma_.lower())
    return lemma_list


if __name__ == "__main__":
    plac.call(main)
