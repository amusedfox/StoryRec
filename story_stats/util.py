import string
import os
import re
from typing import List, Dict, Set
import math

from scipy import stats
import pandas as pd
from nltk.corpus import cmudict
import spacy

from spacy.tokens import Doc

cList = {
    "aren't": "are not",
    "can't": "cannot",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it had",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'  am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "so's": "so is",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "y'all": "you all",
    "y'alls": "you alls",
    "you'd": "you had",
    "you'll": "you you will",
    "you're": "you are",
    "you've": "you have"
}
lower_contractions = list(cList.items())
for key, item in lower_contractions:
    cList[key.capitalize()] = item.capitalize()

MIN_STORY_WORD_COUNT = 2000
MAX_STORY_LEN = 1000000  # In characters

stop_word_list = []
with open('assets/stopwords.txt') as g_in_file:
    for g_line in g_in_file:
        g_word = g_line.strip()
        stop_word_list.append(g_word)

STOP_WORD_SET = set(stop_word_list)
CONTRACTION_RE = re.compile('(%s)' % '|'.join(cList.keys()))
REGEX_NUMBERS = re.compile(r'\d+')
ONLY_LETTERS_SPACES = re.compile(r'[^A-Za-z\s]+')

UNWANTED_POS = {'PUNCT', 'PROPN', 'SYM', 'X', 'SPACE'}

# nltk_lem = WordNetLemmatizer()
spacy_nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
spacy_nlp.max_length = MAX_STORY_LEN

word_freq_dict = {}
syllable_dict = cmudict.dict()


def get_ngram_list(lemm_list: List[str], ngram_max_size: int = 3) -> List[str]:
    ngram_list = []

    lemm_list = [ngram for ngram in lemm_list if ngram != "'s"]
    n_lemm = len(lemm_list)

    for word_index, ngram in enumerate(lemm_list):
        # Don't include boundary points set in get_lemm_list
        if ngram == '.' or len(ngram) == 1:
            continue

        # # Don't include n-grams that start and end with a stop word
        # start_with_stop = False
        # if ngram in STOP_WORD_SET:
        #     start_with_stop = True
        # else:
        #     # Include 1-gram strings that are not stop words
        #     ngram_list.append(ngram)
        ngram_list.append(ngram)

        # Include up to ngram_max_size-gram strings
        for i in range(word_index + 1, word_index + ngram_max_size):
            if i >= n_lemm:
                break

            # Don't include strings with only punctuation in any n-gram
            if lemm_list[i] == '.':
                break

            ngram += " " + lemm_list[i]

            # # Don't include n-grams that start and end with a stop word
            # if lemm_list[i] in STOP_WORD_SET and start_with_stop:
            #     continue

            if len(lemm_list[i]) == 1:
                continue

            # # Don't include bigrams with any stopwords
            # if start_with_stop and ngram_max_size == 2 and \
            #         lemm_list[i] in STOP_WORD_SET:
            #     continue

            ngram_list.append(ngram)

    return ngram_list


# Split text into a list when it is too long
def split_text(content: str, max_content_len: int):
    content_len = len(content)
    n_to_split = math.ceil(content_len / max_content_len)

    paragraph_list = content.strip().split('\n')
    n_paragraphs = len(paragraph_list)

    paragraphs_per = int(n_paragraphs / n_to_split) + 1
    content_list = []
    for i in range(n_to_split):
        paragraph = paragraph_list[i * paragraphs_per:(i + 1) * paragraphs_per]
        content_list.append(' '.join([p.strip() for p in paragraph]))

    return content_list


def syllables(word):
    word = word.lower()
    if word in syllable_dict:
        syllable_list = syllable_dict[word][0]
        return len([s for s in syllable_list if s[-1].isdigit()])

    # Referred from stackoverflow.com/questions/14541303/count-the-number-of-syllables-in-a-word
    count = 0
    vowels = set('aeiouy')
    if word[0] in vowels:
        count += 1

    for i in range(1, len(word)):
        curr_char = word[i]
        next_char = word[i - 1]
        if curr_char in vowels and next_char not in vowels:
            count += 1

    if word.endswith('e'):
        count -= 1

    if word.endswith('le'):
        count += 1

    if count == 0:
        count += 1

    return count


def expand_contractions(text):
    def replace(match):
        return cList[match.group(0)]

    return CONTRACTION_RE.sub(replace, text)


def remove_punct(text: str) -> str:
    # return "".join([char for char in text if char not in string.punctuation])
    return ONLY_LETTERS_SPACES.sub('', text).strip()
    # new_text = text.translate(str.maketrans("", "", string.punctuation))
    # new_text = re.sub('[\W]+', '', text) # Needs to go after lemmatize contractions


def get_filtered_word_list(word_list: List[str]) -> List[str]:
    return [word for word in word_list if word not in STOP_WORD_SET]


def get_filtered_word_set(word_list: List[str]) -> Set[str]:
    filtered_word_list = get_filtered_word_list(word_list)
    filtered_word_set = set(filtered_word_list)

    return filtered_word_set


def get_lemm_list(doc: Doc) -> List[str]:
    word_list = []
    # pos_list = []
    for token in doc:
        if token.pos_ in UNWANTED_POS:
            if token.text == '-':
                continue
            word_list.append('.')  # Find boundaries for n-grams
            continue

        lemma = token.lemma_
        if lemma == '-PRON-':
            word = token.text.lower().strip('-')
            if len(word) > 1:
                word_list.append(word)
            continue

        word = lemma.lower().replace('\202f', '').strip('-')
        if len(word) > 1:
            word_list.append(word)
        # pos_list.append(token.pos_)

    return word_list


def find_outliers(n_list) -> List[int]:
    try:
        z_score_list = stats.zscore(n_list)
    except FloatingPointError:
        return []

    to_remove = []
    for i in range(len(n_list)):
        if abs(z_score_list[i]) > 3:
            to_remove.append(i)

    # if len(to_remove) > 0:
    #     print(n_list)
    #     print(z_score_list)
    #     print(to_remove)

    return to_remove


def remove_outliers(float_list: List[float] = None,
                    df: pd.DataFrame = None) -> None:
    if float_list is not None:
        to_remove = find_outliers(float_list)
        to_remove.sort(reverse=True)

        for index in to_remove:
            del float_list[index]
    elif df is not None:
        to_remove_set = set()
        for column_name, column_data in df.iteritems():
            if len(column_data.values) == 0:
                print("WARNING: Vector length is 0")
                print(df)
                continue

            to_remove = find_outliers(column_data.values)
            to_remove_set.update(to_remove)

        df.drop(
            [df.index[i] for i in to_remove_set], inplace=True)
    else:
        print('WARNING: Did not give any parameter to remove_outliers')


# Randomly choose a place in the story to start a sample
def get_sample_word_lists(
        word_list: List[str],
        min_sample_len: int = MIN_STORY_WORD_COUNT) -> List[List[str]]:
    n_words = len(word_list)
    n_samples = int(n_words / min_sample_len)
    if n_samples == 0:
        n_samples = 1

    sample_word_lists = []

    for i in range(n_samples):
        start_i = i * min_sample_len
        end_i = (i + 1) * min_sample_len
        sample_word_lists.append(word_list[start_i:end_i])

    return sample_word_lists


def load_short_story_set(short_story_file_path: str) -> set:
    short_story_set = set()

    if not os.path.exists(short_story_file_path):
        return short_story_set

    with open(short_story_file_path) as in_file:
        for row in in_file:
            short_story_set.add(row.strip())

    return short_story_set
