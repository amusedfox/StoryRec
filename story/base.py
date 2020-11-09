from bs4 import BeautifulSoup, SoupStrainer
import requests
import os
from typing import List
import datetime
import re

from story_stats.util import expand_contractions, remove_punct, REGEX_NUMBERS

WARNING_COLOR = '\033[93m'
ERROR_COLOR = '\033[91m'
NORMAL_COLOR = '\033[0m'
GREEN_COLOR = '\033[92m'

ILLEGAL_FILE_CHARS = re.compile(r'[<>:"\\/|*?\n]')
TAG_STRAINER = SoupStrainer('tag')


def get_soup(story_url: str) -> BeautifulSoup:
    r = requests.get(story_url)
    soup = BeautifulSoup(r.content, "lxml")

    return soup


def get_tags(file_path) -> List[str]:
    assert file_path.endswith('.html'), f'{file_path} is not an .html file'

    with open(file_path) as in_file:
        soup = BeautifulSoup(in_file.read(), 'lxml', parse_only=TAG_STRAINER)

    tags = soup.find_all('tag')
    return \
        [ILLEGAL_FILE_CHARS.sub('_', t.text) for t in tags if t.text[0] != '-']


class BaseStory:
    """Base class for story objects"""

    def __init__(self, story_dir: str = None, story_path: str = None,
                 story_id: str = None):
        """Creates a BaseStory instance

        Has multiple usages:
            Download a story
            Read a story from disk

        Parameters
        ----------
        story_dir
        story_path
            Path to story file to read
        story_id
            Used to download a story?
        """
        self.story_path = story_path
        self.story_id = story_id
        self.story_dir = story_dir

        self.title: str = "Unknown"
        self.author: str = "Unknown"
        self.author_link: str = "Unknown"
        self.category: str = "Unknown"
        self.tags: List[str] = []
        self.is_complete: bool = False
        time_now = datetime.datetime.now()
        self.published_date: str = "Unknown"
        self.updated_date: str = "Unknown"
        self.downloaded_date: str = time_now.strftime('%Y-%m-%d')
        self.rating: str = "Unknown"
        self.num_chapters: int = 0
        self.word_count: int = 0
        self.publisher: str = "Unknown"
        self.summary: str = "Unknown"
        self.link: str = "Unknown"

        self.chapters: List[str] = []
        self.chapter_names: List[str] = []
        self.content: str = "Unknown"
        self.word_list: List[str] = []
        self.filtered_word_list: List[str] = []
        self.load_success = False
        self.is_preprocessed = False

        if story_id and story_path:
            print(f'{ERROR_COLOR}ERROR: Given both an id and a file_path{NORMAL_COLOR}')
            return

        if story_id:
            if not story_dir:
                print(f'{ERROR_COLOR}ERROR: Not given a folder to download to{NORMAL_COLOR}')
                return

            soup = self.get_story_info()
            self.title = ILLEGAL_FILE_CHARS.sub('', self.title)
            self.author = ILLEGAL_FILE_CHARS.sub('', self.author)
            self.category = ILLEGAL_FILE_CHARS.sub('', self.category)
            self.story_path = f'{self.category}/{self.author} - {self.title}.html'

            # TODO: Fix tags for already downloaded stories
            for i in range(len(self.tags)):
                self.tags[i] = ILLEGAL_FILE_CHARS.sub('_', self.tags[i])

            if os.path.exists(os.path.join(self.story_dir, self.story_path)):
                return

            success = self.download_story(soup)
            if not success:
                return

            # print(self.chapters[0].replace('</p>', '</p>\n\n'))
            for i in range(len(self.chapters)):
                self.chapters[i] = self.chapters[i].replace('“', '"')
                self.chapters[i] = self.chapters[i].replace('”', '"')
                self.chapters[i] = self.chapters[i].replace('‘', "'")
                self.chapters[i] = self.chapters[i].replace('’', "'")

            self.write()
        elif story_path:
            if not story_dir:
                print(f'{ERROR_COLOR}ERROR: Not given a folder to load story from{NORMAL_COLOR}')
                return
            success = self.load_story()
            if not success:
                os.remove(os.path.join(self.story_dir, self.story_path))

    def load_story(self):
        with open(os.path.join(self.story_dir, self.story_path)) as in_file:
            soup = BeautifulSoup(in_file.read(), 'lxml')

            self.content = ""
            for text in soup.find_all('div', {'class': 'chapter'}):
                self.content += text.get_text()

            title_soup = soup.find(id='title')
            if title_soup is None:
                return False

            self.title = title_soup.text
            self.author = soup.find(id='author').text
            self.story_id = soup.find(id='story_id').text
            self.word_count = \
                int(soup.find(id='word_count').text.replace(',', ''))
            self.summary = soup.find(id='summary').text
            self.category = soup.find(id='category').text

        return True

    def write(self):
        # open a file with title as name
        output = \
            f'<!DOCTYPE html>\n<html lang="en-US">\n<title>{self.title}</title>\n' \
            f'<meta charset="utf-8">\n' \
            f'<h1><a id="title" href="{self.link}">{self.title}</a> by ' \
            f'<a id="author" href="{self.author_link}">{self.author}</a></h1>\n' \
            f'<table>\n' \
            f'<tr><td><b>Category:</b></td><td id="category">{self.category}</td></tr>\n' \
            f'<tr><td><b>IsComplete:</b></td><td id="is_complete">{self.is_complete}</td></tr>\n' \
            f'<tr><td><b>Published:</b></td><td id="published_date">{self.published_date}</td></tr>\n' \
            f'<tr><td><b>Updated:</b></td><td id="updated_date">{self.updated_date}</td></tr>\n' \
            f'<tr><td><b>Downloaded:</b></td><td id="downloaded_date">{self.downloaded_date}</td></tr>\n' \
            f'<tr><td><b>Rating:</b></td><td id="rating">{self.rating}</td></tr>\n' \
            f'<tr><td><b>Chapters:</b></td><td id="num_chapters">{self.num_chapters}</td></tr>\n' \
            f'<tr><td><b>Word Count:</b></td><td id="word_count">{self.word_count}</td></tr>\n' \
            f'<tr><td><b>Publisher:</b></td><td id="publisher">{self.publisher}</td></tr>\n' \
            f'<tr><td><b>Summary:</b></td><td id="summary">{self.summary}</td></tr>\n' \
            f'<tr><td><b>ID:</b></td><td id="story_id">{self.story_id}</td></tr>\n' \
            f'</table>\n' \
            f'<a><h2>Table of Contents</h2></a>\n' \
            f'<p>\n'
        for i in range(self.num_chapters):
            output += f'<a href=#section{format(i + 1, "04d")}>{self.chapter_names[i]}</a><br/>\n'
        output += f'</p>\n' \
                  f'<div id="story_content">\n\n'
        for i in range(self.num_chapters):
            output += \
                f'<a name="section{format(i + 1, "04d")}">' \
                f'<h2>{self.chapter_names[i]}</h2></a>\n' \
                f'<div class="chapter">{self.chapters[i]}</div>\n\n'

        output += f'</div>\n' \
                  f'<br/><tags><b>Tags:</b><br/>\n'
        for tag in self.tags:
            output += f'<tag>{tag}</tag><br/>\n'
        output += f'</tags>\n</html>\n'

        story_dir = os.path.join(self.story_dir, self.category)
        if not os.path.exists(story_dir):
            os.mkdir(story_dir)

        with open(os.path.join(self.story_dir, self.story_path), 'w') as f:
            f.write(output)

        print(f'{self}')

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

    def get_story_info(self):
        raise AttributeError(f'Cannot download story info into BaseStory')

    def download_story(self, soup: BeautifulSoup):
        raise AttributeError(f'Cannot download story into BaseStory')

    def __str__(self):
        return self.story_path
