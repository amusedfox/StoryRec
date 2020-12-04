import sys
from nltk.tokenize import sent_tokenize
import numpy as np
import re

sys.path.append('.')
from story_stats.util import *
from story.base import BaseStory

SINGLE_QUOTE = re.compile(r'(?:\'(.*?)\')')
DOUBLE_QUOTE = re.compile('(?:"(.*?)")')


def get_quote_density(story: BaseStory):
    text = story.content

    total_in_quote = 0
    text_len = len(text)
    lines = [t.strip() for t in text.split('\n') if t.strip()]

    single_quote_count = story.content.count(r'\'')
    double_quote_count = story.content.count('"')
    if single_quote_count > double_quote_count:
        print('single:', double_quote_count / single_quote_count)
        re_to_use = SINGLE_QUOTE
    elif double_quote_count > single_quote_count:
        re_to_use = DOUBLE_QUOTE
    elif double_quote_count == single_quote_count and double_quote_count > 0:
        raise ValueError(
            f'Equal number of double quotes and single quotes for '
            f'{story.story_html_path}: {single_quote_count}')
    else:
        return 0

    quote_len_list = []
    for line in lines:
        for quote in re_to_use.findall(line):
            quote_len_list.append(len(quote))

    if len(quote_len_list) == 0:
        return 0

    remove_outliers(quote_len_list)
    for quote_len in quote_len_list:
        total_in_quote += quote_len

    return total_in_quote / text_len


# Sentence length in number of characters
def get_avg_sent_len(story: BaseStory):
    text = story.content

    sent_list = sent_tokenize(text)

    sent_len_list = []
    for sent in sent_list:
        sent_len_list.append(len(sent))

    remove_outliers(sent_len_list)

    return np.mean(sent_len_list)


def get_commas_per_word(story: BaseStory):
    return story.content.count(',') / story.n_words


def get_avg_word_len(story: BaseStory):
    story.load_word_list()
    # text = remove_punct(story.content)
    # word_list = word_tokenize(text)

    word_len_list = [len(word) for word in story.word_list]

    remove_outliers(word_len_list)

    return np.mean(word_len_list)


def get_lexical_density(story: BaseStory):
    story.load_word_list()

    raise RuntimeError('Problem with get_lexical_density due to change in lemmatization')

    word_lem_dict = get_lemm_dict(story.word_list)

    paragraph_lex_density_list = []
    for word_list_sample in get_sample_word_lists(story.word_list):
        num_words = len(word_list_sample)
        filtered_word_set = get_filtered_word_set(word_list_sample)

        filtered_word_list = [word_lem_dict[word] for word in filtered_word_set]
        word_set = set(filtered_word_list)
        num_unique_words = len(word_set)

        paragraph_lex_density_list.append(num_unique_words / num_words)

    remove_outliers(paragraph_lex_density_list)

    return np.mean(paragraph_lex_density_list)


def get_avg_word_len_lex(story: BaseStory):
    story.load_word_list()

    word_len_list = []
    if len(story.filtered_word_list) == 0:
        story.filtered_word_list = get_filtered_word_list(story.word_list)

    for word in story.filtered_word_list:
        word_len_list.append(len(word))

    remove_outliers(word_len_list)

    return np.mean(word_len_list)


def get_avg_word_syl_num(story: BaseStory):
    story.load_word_list()

    syl_list = []
    for word in story.word_list:
        syl_list.append(syllables(word))

    remove_outliers(syl_list)
    return np.mean(syl_list)
