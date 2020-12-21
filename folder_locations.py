import os


STORY_REC = os.environ.get('STORY_REC')
STORY_TFIDF_DIR = f'{STORY_REC}/story_tfidfs'
STORY_TFIDF_SPARSE_DIR = f'{STORY_REC}/story_sparse_tfidfs'
STORY_HTML_DIR = f'{STORY_REC}/story_html'
STORY_TXT_DIR = f'{STORY_REC}/story_txt'
TAG_STORIES_DIR = f'{STORY_REC}/tag_stories'
TAG_STATS_DIR = f'{STORY_REC}/tag_stats'
LEMM_LIST_DIR = f'{STORY_REC}/lemm_list'
ASSETS_DIR = f'{STORY_REC}/assets'
NGRAM_FREQ_DIR = f'{STORY_REC}/ngram_freq'
RESULTS_DIR = f'{STORY_REC}/results'
SAVE_VECTORS_DIR = f'{STORY_REC}/save_vectors'
MODELS_DIR = f'{STORY_REC}/models'

