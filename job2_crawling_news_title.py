from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import re
import time
import datetime

category = ['Politics', 'Economic', 'Social', 'Culture', 'World', 'IT']
pages = [101, 101, 101, 71, 94, 73]
url = 'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=100#&date=%2000:00:00&page=2'

options = webdriver.ChromeOptions()
#options.add_argument('headless')
options.add_argument('lang=kr_KR')
driver = webdriver.Chrome('./chromedriver.exe', options=options)
driver.get(url)
df_title = pd.DataFrame()

for i in range(0, 6): #세션
    titles = []
    for j in range(1, pages[i]): #페이지
        url = 'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=10{}#&date=%2000:00:00&page={}'.format(i, j)
        driver.get(url)
        time.sleep(1)
        for k in range(1, 5): #x_path
            for l in range(1, 6): #x_path
                x_path = '//*[@id="section_body"]/ul[{}]/li[{}]/dl/dt[2]/a'.format(k, l)
                try:
                    title = driver.find_element('xpath', x_path).text
                    title = re.compile('[^가-힣 ]').sub(' ', title)
                    titles.append(title)
                except NoSuchElementException as e:
                    x_path = '//*[@id="section_body"]/ul[{}]/li[{}]/dl/dt/a'.format(k, l)
                    title = driver.find_element('xpath', x_path).text
                    title = re.compile('[^가-힣]').sub(' ', title)
                    titles.append(title)
                except:
                    print('error', i, j, k, l)
        if j % 10 == 0:
            df_section_title = pd.DataFrame(titles, columns=['titles'])
            df_section_title['category'] = category[i]
            df_title = pd.concat([df_title, df_section_title], ignore_index=True)
            df_title.to_csv('./crawling_data/crawling_data_{}_{}.csv'.format(category[i], j), index=False)
            titles = []

print(df_title.head())
print(df_title.category.value_counts())













