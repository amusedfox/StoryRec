import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool
from functools import partial
import re
import os
import time
import sys

BASE_URL = 'https://www.literotica.com/stories/'
NEW_STORIES_URL = 'new_submissions.php?type=story&page='
NUMB_NEW_STORY_PAGES = 10

STORY_DIR = '../../Stories/'
CURRENT_CATEGORY = None

small_story_set = set()

UNWANTED_CATEGORIES = [
    'Adult Comics',
    'Erotic Art',
    'Erotic Poetry',
    'How To',
    'Humor & Satire',
    'Illustrated',
    'Illustrated Poetry',
    'Letters & Transcripts',
    'Non-English',
    'Non-Erotic',
    'Non-Erotic Poetry',
    'Poetry With Audio',
    'Reviews & Essays',
    'Text With Audio',
    'Audio'
]

FINISHED_CATEGORIES = [
    'BDSM',
    'Erotic Horror',
    'NonConsent Reluctance',
    'Mind Control',
    'NonHuman',
    'Sci-Fi & Fantasy',
    'Toys & Masturbation',
    'Fetish',
    'Transgender & Crossdressers',
    'Erotic Couplings',
    'First Time',
    'Gay Male',
    'Group Sex',

]

PRIORITY_CATEGORIES = [
    'BDSM',
    'Erotic Horror',
    'NonConsent Reluctance',
    'Mind Control',
    'NonHuman',
    'Sci-Fi & Fantasy',
    'Toys & Masturbation',
    'Fetish',
    'Transgender & Crossdressers',
    'Erotic Couplings',

    'First Time',
    'Gay Male',
    'Group Sex',
    'Exhibitionist & Voyeur',
    'Chain Stories',
    'Anal',
    'Celebrities & Fan Fiction'
]

SPECIAL_CHARS = re.compile(r'[<>:"\\/|*\n]')
OLD_SPECIAL_CHARS = re.compile(r'[@_!#$%^&*()<>?\\|}{~:\n\"\']')

MIN_STORY_SIZE = 10000

DESC_PREFIX = '\xa0-\xa0'
WARNING_COLOR = '\033[93m'
NORMAL_COLOR = '\033[0m'


class ScrapedStory:

    def __init__(self, tag, category):
        rating = meta.find('span', {'class': 'b-sli-rating'})
        if rating:
            self.rating = rating.string
        else:
            self.rating = '0'

        # fetch content
        self.content = ''
        self.tags = []
        self.fetch_content()
        self.write()

    def __str__(self):
        return '{0} by {1} posted, rated {2}'.format(self.title, self.author, self.rating)

    def fetch_content(self):
        page_count, first_page = get_max_pages(self.link, suffix='')

        for i in range(1, page_count + 1):
            if i > 1:
                soup = get_soup(self.link + '?page={}'.format(i))
            else:
                soup = first_page

            if i == page_count:
                tag_list = soup.find('div', {'class': 'b-s-story-tag-list'})
                if tag_list:
                    for tag in tag_list.find_all('li'):
                        self.tags.append(tag.text.replace('\xa0–', '').strip())
                else:
                    print(f'{WARNING_COLOR}WARNING: No tags for {self.link}?page={i}{NORMAL_COLOR}')

            div = soup.find('div', {'class': 'b-story-body-x'})
            if not div:
                break

            self.content += str(div.find('div'))

    def write(self):
        # open a file with title as name
        output = HTML_HEADER
        output += f'<h1>{self.title}</h1>\n' \
                  f'<info>by <author>{self.author}</author> on <date>{self.date}</date>' \
                  f'<br/><br/>Rating: <rating>{self.rating}</rating></info>\n' \
                  f'<br/><br/><desc>{self.description}</desc>\n' \
                  f'<br/><br/><article>\n{self.content}\n</article><br/>\n<tags>Tags:<br/>\n'
        for tag in self.tags:
            output += f'<tag>{tag}</tag><br/>\n'
        output += f'</tags>\n<a href={self.link}>Read at Literotica</a>\n</html>'
        # if len(output) < MIN_STORY_SIZE:
        #     print(f'{self} is too small: {len(output)}')
        #     self.is_small = True
        #     return

        with open(f'{STORY_DIR}{self.file_path}', 'w') as f:
            f.write(output)

        print('{0}'.format(self))


def get_soup(url=BASE_URL):
    content = requests.get(url).content
    return BeautifulSoup(content, "lxml")


# Get all categories
def get_category_links(url=BASE_URL):
    soup = get_soup(url)
    categories = {}
    for item in soup.find_all('a'):
        if '/c/' in item.get('href'):
            name = item.text
            if name in UNWANTED_CATEGORIES:
                continue

            name = name.replace('/', ' ')

            categories[name] = item.get('href')

    return categories


# For each category, find max number of pages
def get_max_pages(url, suffix='/1-page'):
    soup = get_soup(url + suffix)
    links = soup.find_all('div', {'class': 'b-pager-pages'})
    if 'option' in str(links):
        return int(re.findall(r'<option[^>]*>([^<]+)</option>', str(links))[-1]), soup
    else:
        return int(1), soup


# Get page links in each category
def util_get_pages(url, max_page):
    print('Getting links to all pages')
    return ['{}/{}-page'.format(url, i) for i in range(1, max_page + 1)]


def scrape_story(story_info, category):
    story_info = BeautifulSoup(story_info, 'lxml')
    ScrapedStory(story_info, category)
    # if story.is_small:
    #     return story.file_path
    return None


# Get Story objects
def download_stories(page_links, category):
    print(f'Downloading stories from {category} with {len(page_links)} pages ...')
    for link in page_links:
        print(f'Finding stories in {link} ...')
        soup = get_soup(link)

        # find all matching tags
        story_info_list = soup.findAll('div', {'class': 'b-sl-item-r'})
        story_info_list = [x.__str__() for x in story_info_list]

        scrape_cat = partial(scrape_story, category=category)

        # Sometimes hangs here
        main_pool.map(scrape_cat, story_info_list)
        # try:
        #     main_pool.map(scrape_cat, story_info_list)
        # except AttributeError:
        #     print(story_info_list)


# def read_small_story_set():
#     with open(f'small_stories.txt') as in_file:
#         for line in in_file:
#             small_story_set.add(line.strip())
#
#
# def write_small_story_set():
#     with open(f'small_stories.txt', 'w') as out_file:
#         for story in small_story_set:
#             out_file.write(story + '\n')


def download_categories():
    categories = get_category_links()

    # create folders for each category
    print('Getting directories for categories')
    for name in categories:
        print(name)
        if not os.path.exists(STORY_DIR + name):
            os.makedirs(STORY_DIR + name)

    # for name in PRIORITY_CATEGORIES + list(categories.keys()):
    for name in PRIORITY_CATEGORIES:
        if name in FINISHED_CATEGORIES:
            continue

        # get count of pages
        max_page, temp = get_max_pages(categories[name])
        print(f"{max_page} pages for {name}")

        # get links to all pages
        page_links = util_get_pages(categories[name], max_page)

        # get all Story objects
        download_stories(page_links, name)


# Not working
def download_new():
    urls = []
    for i in range(NUMB_NEW_STORY_PAGES):
        url = f'{BASE_URL}{NEW_STORIES_URL}{i}'
        urls.append(url)
    download_stories(urls)


def main():
    download_categories()
    main_pool.close()
    main_pool.join()


if __name__ == '__main__':
    main_pool = Pool(16)
    main()
