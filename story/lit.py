from story.base import BaseStory
from bs4 import BeautifulSoup
import os


class LitStory(BaseStory):

    def download_story(self):
        pass
        # meta = tag.find('span', {'class': 'b-sli-meta'})
        #
        # self.title = tag.find('h3').string.replace('/', '')
        # self.author = meta.find('span', {'class': 'b-sli-author'}).find('a').string
        #
        # if self.author is None:
        #     self.author = 'None'
        #
        # try:
        #     self.title = SPECIAL_CHARS.sub('', self.title)
        #     self.author = SPECIAL_CHARS.sub('', self.author)
        # except TypeError:
        #     print('title:', self.title)
        #     print("author:", self.author)
        #     sys.exit()
        #
        # self.file_path = f'{category}/{self.author} - {self.title}.html'
        #
        # # Fix old mistake of deleting important characters from file names
        # old_path = OLD_SPECIAL_CHARS.sub('', self.file_path)
        # if old_path != self.file_path and \
        #         os.path.exists(f'{STORY_DIR}{old_path}'):
        #     os.remove(f'{STORY_DIR}{old_path}')
        #     print(f"{WARNING_COLOR}Deleted {old_path}{NORMAL_COLOR}")
        #
        # if os.path.exists(f"{STORY_DIR}{self.file_path}") or \
        #         f'{self.file_path}' in small_story_set:
        #     return
        #
        # self.date = meta.find('span', {'class': 'b-sli-date'}).string
        #
        # self.link = tag.find('h3').find('a')['href']
        # if self.link[:6] != 'https:':
        #     self.link = 'https:' + self.link
        #
        # self.description = tag.find('span', {'class': 'b-sli-description'}).text.replace(DESC_PREFIX, '')
        #
        # page_count, first_page = get_max_pages(self.link, suffix='')
        #
        # for i in range(1, page_count + 1):
        #     if i > 1:
        #         soup = get_soup(self.link + '?page={}'.format(i))
        #     else:
        #         soup = first_page
        #
        #     if i == page_count:
        #         tag_list = soup.find('div', {'class': 'b-s-story-tag-list'})
        #         if tag_list:
        #             for tag in tag_list.find_all('li'):
        #                 self.tags.append(tag.text.replace('\xa0–', '').strip())
        #         else:
        #             print(f'{WARNING_COLOR}WARNING: No tags for {self.link}?page={i}{NORMAL_COLOR}')
        #
        #     div = soup.find('div', {'class': 'b-story-body-x'})
        #     if not div:
        #         break
        #
        #     self.content += str(div.find('div'))
