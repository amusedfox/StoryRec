from bs4 import BeautifulSoup
import os
import plac
from pathlib import Path
import requests
import re
from functools import partial

import sys
sys.path.append('.')
from story.base import BaseStory

BASE_URL = 'https://www.literotica.com/stories/'
NEW_STORIES_URL = 'new_submissions.php?type=story&page='


class LitStory(BaseStory):

    def find_story_metadata(self, soup):
        meta = soup.find('div', {'class': 'b-story-header'})

        self.title = meta.h1.text
        self.author = meta.span.a.text

        self.author_link = meta.span.a['href']

        # Sometimes author is not available
        if self.author is None:
            self.author = 'Unknown'

        if self.title is None:
            raise ValueError(f'Could not find title')

        self.summary = soup.find('meta', {'name': "description"})['content']

        story_stats = soup.find('span', {'class': 'b-story-stats'}).text.split()
        self.n_comments = story_stats[0]
        self.n_views = story_stats[2]
        self.n_favorites = story_stats[4]

        self.origin = 'Literotica'

    def download_story_chapters(self, soup):

        n_pages, first_page_soup = get_page_count(self.story_url, suffix='')

        for i in range(1, n_pages + 1):
            if i > 1:
                soup = get_soup(self.story_url + f'?page={i}')
            else:
                soup = first_page_soup

            if i == n_pages:
                tag_list = soup.find('div', {'class': 'b-s-story-tag-list'})
                if tag_list:
                    for tag in tag_list.find_all('li'):
                        self.tags.append(tag.text.replace('\xa0–', '').strip())

            story_content = soup.find('div', {'class': 'b-story-body-x'}).div
            if not story_content:
                raise ValueError('Missing story_content')

            self.chapters.append(str(story_content))
            self.chapter_names.append(f'Page {i}')
            self.content += story_content.text + '\n'

        self.n_chapters = n_pages
        self.is_complete = True

        return True


def get_soup(url=BASE_URL):
    content = requests.get(url).content
    return BeautifulSoup(content, "lxml")


def get_page_count(url, suffix):
    soup = get_soup(url + suffix)
    links = soup.find_all('div', {'class': 'b-pager-pages'})

    if 'option' in str(links):
        return int(re.findall(r'<option[^>]*>([^<]+)</option>', str(links))[-1]), soup
    else:
        return int(1), soup


# Get all categories
def get_category_links(url=BASE_URL):
    soup = get_soup(url)
    categories = {}
    for item in soup.find_all('a'):
        if '/c/' in item.get('href'):
            name = item.text

            name = name.replace('/', ' ')

            categories[name] = item.get('href')

    return categories


def scrape_story(story_url, category, story_html_dir, story_txt_dir):
    story = LitStory(story_url=story_url, save_html_dir=story_html_dir,
                     save_txt_dir=story_txt_dir)
    story.add_tag(category)
    story.save()


def download_stories(page_links, category, story_html_dir, story_txt_dir):
    print(f'Downloading stories from {category} with {len(page_links)} pages ...')
    for link in page_links:
        print(f'Finding stories in {link} ...')
        soup = get_soup(link)

        story_list = soup.findAll('a', {'class': 'r-34i'})
        story_url_list = [x['href'] for x in story_list]

        scrape_cat = partial(scrape_story, category=category,
                             story_html_dir=story_html_dir,
                             story_txt_dir=story_txt_dir)

        # Sometimes hangs here
        main_pool.map(scrape_cat, story_url_list)
        # try:
        #     main_pool.map(scrape_cat, story_info_list)
        # except AttributeError:
        #     print(story_info_list)


def download(story_html_dir, story_txt_dir):
    categories = get_category_links()

    # for name in PRIORITY_CATEGORIES + list(categories.keys()):
    for category_name, link in categories.items():

        # get count of pages
        n_pages, _ = get_page_count(link, '/1-page')
        print(f"{n_pages} pages for {category_name}")

        # get links to all pages
        page_links = [f'{link}/{i}-page' for i in range(1, n_pages + 1)]

        # get all Story objects
        download_stories(page_links, category_name, story_html_dir,
                         story_txt_dir)


@plac.pos('story_html_dir', "Directory with .html files", Path)
@plac.pos('story_txt_dir', "Directory with .txt files", Path)
def main(story_html_dir, story_txt_dir):
    download(story_html_dir, story_txt_dir)
    main_pool.close()
    main_pool.join()


if __name__ == '__main__':
    import multiprocessing

    main_pool = multiprocessing.Pool(
        processes=multiprocessing.cpu_count() * 2
    )
    plac.call(main)
