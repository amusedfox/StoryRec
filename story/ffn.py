import sys

from bs4 import BeautifulSoup
import time
import os
from tqdm.auto import tqdm

sys.path.append('.')
from story.base import BaseStory, get_file_path, get_cf_soup
from folder_locations import *

FFN_URL = "https://www.fanfiction.net/"
FFN_DEFAULT_PARAM = '/?&srt=3&lan=1&r=10'
WARNING_COLOR = '\033[93m'
ERROR_COLOR = '\033[91m'
NORMAL_COLOR = '\033[0m'
GREEN_COLOR = '\033[92m'

GENRE_LIST = [
    'Adventure',
    'Angst',
    'Crime',
    'Drama',
    'Family',
    'Fantasy',
    'Friendship',
    'General',
    'Horror',
    'Humor',
    'Hurt/Comfort',
    'Mystery',
    'Parody',
    'Poetry',
    'Romance',
    'Sci-Fi',
    'Spiritual',
    'Supernatural',
    'Suspense',
    'Tragedy',
    'Western'
]


def get_num_chapters(metadata_list) -> int:
    try:
        chapter_index = metadata_list.index('Chapters:')
    except ValueError:
        return 1
    return int(metadata_list[chapter_index + 1])


def get_story_content(soup: BeautifulSoup) -> (str, str):
    content = soup.find('div', {'id': 'storytext'})
    if content is None:
        raise ValueError('Story content of page was not found')

    def check_selected(tag):
        return tag.has_attr('selected')

    chapter_tag = soup.find(check_selected)
    chapter_name = '1. Chapter 1'
    if chapter_tag:
        chapter_name = chapter_tag.text

    p_content = ""
    for p in content.find_all('p'):
        p_content += p.text + '\n<br/><br/>\n'

    return chapter_name, p_content.strip()


def download_chapter(story_id, chapter_num) -> (str, str):
    story_url = FFN_URL + '/s/' + str(story_id) + f"/{chapter_num}/"
    soup = get_cf_soup(story_url)
    return get_story_content(soup)


class FFNStory(BaseStory):

    def __init__(self, story_id, file_path):
        story_url = f'{FFN_URL}/s/{story_id}'
        super().__init__(story_url=story_url, save_html_dir=FFN_STORY_HTML_DIR,
                         save_txt_dir=FFN_STORY_TXT_DIR)

    def find_story_metadata(self, soup):

        metadata_div = soup.find('span', class_='xgray xcontrast_txt')

        if ' English ' not in metadata_div.text:
            return False

        self.title = soup.find('b', {'class': 'xcontrast_txt'}).text
        self.author = soup.find_all('a', {'class': 'xcontrast_txt'})[-3].text
        self.author_link = FFN_URL + soup.find_all(
            'a', {'class': 'xcontrast_txt'}
        )[-3]['href']

        self.rating = metadata_div.find('a', {'target': 'rating'}).text.split()[-1]
        if self.rating == 'M':
            self.rating = 'E'
        self.is_complete = ' Status: Complete ' in str(metadata_div)
        self.n_comments = int(metadata_div.find_all('a')[-1].text.replace(',', ''))
        metadata_parts = metadata_div.text.split()
        n_favorites_i = metadata_parts.index('Favs:')
        assert n_favorites_i >= 0
        self.n_favorites = int(metadata_parts[n_favorites_i + 1].replace(',', ''))

        self.summary = soup.find('div', {'class': 'xcontrast_txt'}).text

        self.tags.append(soup.find_all('a', {'class': 'xcontrast_txt'})[-4].text)
        for genre in GENRE_LIST:
            if genre in metadata_div.text:
                self.tags.append(genre)

        n_words_i = metadata_parts.index('Words:')
        assert n_words_i >= 0
        self.n_words = int(metadata_parts[n_words_i + 1].replace(',', ''))

        n_chapters_i = metadata_parts.index('Chapters:')
        assert n_chapters_i >= 0
        self.n_chapters = int(metadata_parts[n_chapters_i + 1].replace(',', ''))

        self.origin = 'fanfiction.net'

        return True

    def download_story_chapters(self, soup: BeautifulSoup):
        chapter_name, chapter_content = get_story_content(soup)
        self.chapter_names.append(chapter_name)
        self.chapters.append(chapter_content)

        story_id = self.story_url.split('/')[-1]

        for i in tqdm(range(2, self.n_chapters + 1)):
            chapter_name, chapter_content = download_chapter(story_id, i)
            self.chapter_names.append(chapter_name)
            self.chapters.append(chapter_content)

        return True


def get_sub_categories(category):
    category_url = FFN_URL + category
    soup = get_cf_soup(category_url, 0)
    subcategories = soup.find_all('a', title=True)
    subcategories = [x['href'] for x in subcategories]

    if len(subcategories) == 0:
        raise ValueError(f'No subcategories were found for {category}')

    return subcategories


def get_story_list(soup):
    story_blocks = soup.find_all('a', {'class': 'stitle'})

    story_list = []
    for story_block in story_blocks:
        parent = story_block.parent
        author_block = parent.find_all('a')[-2]
        author = author_block.text
        title = story_block.text

        file_path = get_file_path(title, author, FFN_STORY_HTML_DIR)

        if not os.path.isfile(file_path):
            ffn_story_id = story_block['href'].split('/')[2]

            try:
                int(ffn_story_id)
            except ValueError:
                print(f'{ffn_story_id} is not a valid ffn story id')
                continue

            story_list.append((ffn_story_id, file_path))

    return story_list


def download_story(story_id, file_path):
    story = FFNStory(story_id, file_path)
    story.save()


def main():
    subcategories = get_sub_categories('book')
    print(subcategories[:5])
    for subcategory in subcategories:
        subcategory_story_list_url = FFN_URL + subcategory + FFN_DEFAULT_PARAM
        story_list_soup = get_cf_soup(subcategory_story_list_url)
        story_list = get_story_list(story_list_soup)
        for story_id, file_path in story_list:
            download_story(story_id, file_path)
            sys.exit()


if __name__ == '__main__':
    main()
