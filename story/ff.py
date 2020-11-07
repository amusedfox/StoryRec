from story.base import BaseStory, get_soup, ILLEGAL_FILE_CHARS
from bs4 import BeautifulSoup
import time
import os
from tqdm import tqdm

FF_URL = "https://www.fanfiction.net"
WARNING_COLOR = '\033[93m'
ERROR_COLOR = '\033[91m'
NORMAL_COLOR = '\033[0m'
GREEN_COLOR = '\033[92m'


def story_exists(story_soup):
    errors = story_soup.find_all("span", attrs={"class": "gui_warning"})
    if len(errors) >= 1:
        return False
    else:
        return True


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
    for p in content:
        p_content += str(p) + '\n\n'

    return chapter_name, p_content.strip()


def download_chapter(story_id, chapter_num) -> (str, str):
    story_url = FF_URL + '/s/' + str(story_id) + f"/{chapter_num}/"
    soup = get_soup(story_url)
    return get_story_content(soup)


def get_story_links():
    pass


class FFStory(BaseStory):

    def __init__(self, story_path=None, story_dir=None, story_id=None):
        super().__init__(story_dir, story_path, story_id)

    def get_meta(self, soup) -> bool:
        meta = soup.find("span", attrs={"class": "xgray xcontrast_txt"}).text
        meta_list = meta.split()
        self.num_chapters = get_num_chapters(meta_list)
        self.rating = meta_list[2]
        word_count_index = meta_list.index("Words:") + 1
        self.word_count = meta_list[word_count_index]

        # Get dates
        # publish_index = meta_list.index("Published:") + 1
        # self.published_date = meta_list[publish_index]
        #
        # last_update_index = meta_list.index("Updated:") + 1
        # self.updated_date = meta_list[last_update_index]
        #
        # if 'h' in self.updated_date:

        return True

    def get_story_info(self):
        story_url = FF_URL + '/s/' + str(self.story_id) + "/1/"
        soup = get_soup(story_url)

        if not story_exists(soup):
            print(f'{ERROR_COLOR}ERROR: Not a valid story{NORMAL_COLOR}')
            return

        self.title = soup.find('b', {'class': 'xcontrast_txt'}).text
        self.author = soup.find_all('a', {'class': 'xcontrast_txt'})[-3].text
        self.category = soup.find_all('a', {'class': 'xcontrast_txt'})[-4].text

        self.author_link = FF_URL + soup.find_all(
            'a', {'class': 'xcontrast_txt'}
        )[-3]['href']

        self.summary = soup.find('div', {'class': 'xcontrast_txt'}).text
        self.publisher = 'fanfiction.net'
        self.link = story_url

        return soup

    def download_story(self, soup: BeautifulSoup):

        success = self.get_meta(soup)
        if not success:
            return False

        chapter_name, chapter_content = get_story_content(soup)
        self.chapter_names.append(chapter_name)
        self.chapters.append(chapter_content)
        for i in tqdm(range(2, self.num_chapters + 1)):
            time.sleep(2)
            chapter_name, chapter_content = download_chapter(self.story_id, i)
            self.chapter_names.append(chapter_name)
            self.chapters.append(chapter_content)

        return True
