from bs4 import BeautifulSoup, SoupStrainer
import requests
import os
from typing import List, Set
import datetime
import re
import time

from story_stats.util import expand_contractions, remove_punct, REGEX_NUMBERS

WARNING_COLOR = '\033[93m'
ERROR_COLOR = '\033[91m'
NORMAL_COLOR = '\033[0m'
GREEN_COLOR = '\033[92m'

ILLEGAL_FILE_CHARS = re.compile(r'[<>:"\\/|*?\n]')
TAG_STRAINER = SoupStrainer('tag')


def get_soup(url: str, sleep_time=2) -> BeautifulSoup:
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.content, "lxml")

    time.sleep(sleep_time)

    return soup


def get_story_tags(file_path) -> Set[str]:
    assert file_path.endswith('.html'), f'{file_path} is not an .html file'

    with open(file_path) as in_file:
        soup = BeautifulSoup(in_file.read(), 'lxml', parse_only=TAG_STRAINER)

    tags = soup.find_all('tag')
    return set(
        ILLEGAL_FILE_CHARS.sub('_', t.text) for t in tags if t.text[0] != '-')


def get_prefix_folder(file_name):
    first_char = file_name[0]
    first_char_i = 1
    while not first_char.isalpha():
        first_char = file_name[first_char_i]
        first_char_i += 1

    return first_char.lower()


class BaseStory:
    """Base class for story objects"""

    def __init__(self, story_html_path: str = None, story_url: str = None,
                 save_html_dir: str = None, save_txt_dir: str = None):
        """Creates a BaseStory instance

        Has multiple usages:
            Download a story
            Read a story from disk

        Parameters
        ----------
        story_html_path
            Path to story file to read
        story_url
            Url of story to download
        save_html_dir
            Directory to download story html file to
        save_txt_dir
            Directory to download story html file to
        """
        self.story_txt_path = None
        self.story_html_path = story_html_path
        self.story_url = story_url

        self.title: str = "Unknown"
        self.author: str = "Unknown"
        self.author_link: str = "Unknown"
        self.tags: List[str] = []
        self.is_complete: bool = False
        self.n_chapters: int = 0
        self.n_words: int = 0
        self.origin: str = "Unknown"
        self.summary: str = "Unknown"
        self.n_comments: int = 0
        self.n_views: int = 0
        self.n_favorites: int = 0

        self.n_chapters: int = 0
        self.chapters: List[str] = []
        self.chapter_names: List[str] = []
        self.content: str = ""
        self.word_list: List[str] = []
        self.filtered_word_list: List[str] = []
        self.load_success = False
        self.download_success = False
        self.is_preprocessed = False

        if story_url:
            try:
                self.download_story(save_html_dir, save_txt_dir)
            except Exception as e:
                print(f'ERROR: Could not download {story_url}')
                print(e)
                time.sleep(60 * 60)
                return

        elif story_html_path:
            # Read story from disk

            success = self.load_story()
            if not success:
                to_remove = input(f'ERROR: Error with loading '
                                  f'{self.story_html_path}. '
                                  f'Remove? ')
                if to_remove == 'y':
                    print('Removed')
                    os.remove(self.story_html_path)

    def download_story(self, save_html_dir, save_txt_dir):
        if not save_html_dir:
            raise ValueError(f'Not given a folder to download to')

        soup = get_soup(self.story_url)
        self.find_story_metadata(soup)
        self.title = ILLEGAL_FILE_CHARS.sub('', self.title)
        self.author = ILLEGAL_FILE_CHARS.sub('', self.author)
        self.author = re.sub('^-*', '_', self.author)

        file_name = f'{self.author} - {self.title}'
        first_char = get_prefix_folder(file_name)

        self.story_html_path = os.path.join(save_html_dir, first_char,
                                            f'{file_name}.html')
        self.story_txt_path = os.path.join(save_txt_dir, first_char,
                                           f'{file_name}.txt')

        if os.path.exists(self.story_html_path):
            return

        self.download_story_chapters(soup)
        self.download_success = True

    def load_story(self):
        with open(self.story_html_path) as in_file:
            soup = BeautifulSoup(in_file.read(), 'lxml')

            self.content = ""
            for text in soup.find_all('div', {'class': 'chapter'}):
                self.content += text.get_text()

            title_soup = soup.find(id='title')
            if title_soup is None:
                return False

            self.title = title_soup.text
            self.author = soup.find(id='author').text
            self.n_words = \
                int(soup.find(id='word_count').text.replace(',', ''))
            self.summary = soup.find(id='summary').text

        return True

    def prepare_to_save(self):
        self.n_words = sum(len(chapter.replace('<br/>', '').split()) for
                           chapter in self.chapters)

        for i in range(len(self.tags)):
            self.tags[i] = ILLEGAL_FILE_CHARS.sub('_', self.tags[i])
            self.tags[i] = re.sub('^-*', '_', self.tags[i])

        for i in range(len(self.chapters)):
            self.chapters[i] = self.chapters[i].replace('“', '"')
            self.chapters[i] = self.chapters[i].replace('”', '"')
            self.chapters[i] = self.chapters[i].replace('‘', "'")
            self.chapters[i] = self.chapters[i].replace('’', "'")
            self.chapters[i] = self.chapters[i].replace('\202f', '')

    def save(self, write_to_disk=True):

        if not self.download_success:
            return None, None

        self.prepare_to_save()

        # open a file with title as name
        output = \
            f'<!DOCTYPE html>\n<html lang="en-US">\n' \
            f'<title>{self.title}</title>\n' \
            f'<meta charset="utf-8">\n' \
            f'<h1><a id="title" href="{self.story_url}">{self.title}</a> by ' \
            f'<a id="author" href="{self.author_link}">{self.author}</a></h1>\n' \
            f'<table id="meta">\n' \
            f'<tr><td><b>IsComplete:</b></td><td id="is_complete">{self.is_complete}</td></tr>\n' \
            f'<tr><td><b>View Count:</b></td><td id="n_views">{self.n_views}</td></tr>\n' \
            f'<tr><td><b>Comment Count:</b></td><td id="n_comments">{self.n_comments}</td></tr>\n' \
            f'<tr><td><b>Favorite Count:</b></td><td id="n_favorites">{self.n_favorites}</td></tr>\n' \
            f'<tr><td><b>Word Count:</b></td><td id="n_words">{self.n_words}</td></tr>\n' \
            f'<tr><td><b>Origin:</b></td><td id="origin">{self.origin}</td></tr>\n' \
            f'<tr><td><b>Summary:</b></td><td id="summary">{self.summary}</td></tr>\n' \
            f'</table>\n' \
            f'<a><h2>Table of Contents</h2></a>\n' \
            f'<p>\n'

        for i in range(self.n_chapters):
            output += f'<a href=#section{format(i + 1, "04d")}>{self.chapter_names[i]}</a><br/>\n'
        output += f'</p>\n' \
                  f'<div id="story_content">\n\n'
        for i in range(self.n_chapters):
            output += \
                f'<a name="section{format(i + 1, "04d")}">' \
                f'<h2 class="chapter_name">{self.chapter_names[i]}</h2></a>\n' \
                f'<div class="chapter_content"><p>{self.chapters[i]}</p></div>\n\n'

        output += f'</div>\n' \
                  f'<br/>\n' \
                  f'<tags><b>Tags:</b><br/>\n'
        for tag in self.tags:
            output += f'<tag>{tag}</tag><br/>\n'
        output += f'</tags>\n</html>\n'

        if write_to_disk:
            os.makedirs(os.path.dirname(self.story_html_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.story_txt_path), exist_ok=True)

            try:
                with open(self.story_html_path, 'w') as f:
                    f.write(output)

                with open(self.story_txt_path, 'w') as f:
                    f.write(self.content)
            except:
                # Cleanup
                if os.path.exists(self.story_html_path):
                    os.remove(self.story_html_path)
                if os.path.exists(self.story_txt_path):
                    os.remove(self.story_txt_path)

        return output, self.content

    def add_tag(self, tag_name: str):
        tag_name = ILLEGAL_FILE_CHARS.sub('_', tag_name)
        self.tags.append(tag_name.lower())

    def preprocess(self):
        if self.content is None:
            raise ValueError

        self.content = self.content.lower()
        self.content = expand_contractions(self.content)
        self.content = REGEX_NUMBERS.sub('', self.content)

        self.is_preprocessed = True

        # text = remove_punct(self.content)
        # self.word_list = word_tokenize(text)

    def load_word_list(self):

        if not self.is_preprocessed:
            self.preprocess()

        if len(self.word_list) == 0:
            text = remove_punct(self.content)
            self.word_list = text.split()

    def find_story_metadata(self, soup: BeautifulSoup):
        """Find story title and author

        May find other metadata as well if applicable
        """
        raise AttributeError(f'Cannot download story info into BaseStory')

    def download_story_chapters(self, soup: BeautifulSoup):
        raise AttributeError(f'Cannot download story into BaseStory')

    def __str__(self):
        return self.story_html_path
