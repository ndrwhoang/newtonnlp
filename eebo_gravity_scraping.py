# Libraries
import pandas as pd
from bs4 import BeautifulSoup 
import requests
from selenium import webdriver


# Accessing website
browser = webdriver.Firefox()
browser.get('https://eebo.chadwyck.com/search')
# Search for keyword
keyword = 'gravity'
search_bar = browser.find_element_by_xpath("//input[@class = 'fields']")
search_bar.send_keys(keyword)
# Define time range
lower = '1625'
upper = '1685'
min_date = browser.find_element_by_xpath("//input[@name = 'DATE1']")
browser.find_element_by_xpath("//input[@name = 'DATE1']").clear()
min_date.send_keys(lower)
max_date = browser.find_element_by_xpath("//input[@name = 'DATE2']")
browser.find_element_by_xpath("//input[@name = 'DATE2']").clear()
max_date.send_keys(upper)
# Showing 40 results per page
display_size = browser.find_element_by_xpath("//select[@name = 'SIZE']/option[@value='40']")
display_size.click()
# Search
search_button = browser.find_element_by_xpath("//input[@type = 'submit']")
search_button.click()


# Number of pages of results
no_of_pages = 10


# Scraping text
full_text = []
for i in range(0,no_of_pages):
    # Get all full text links
    link = browser.find_elements_by_xpath("//a[contains(@href, '%s')]" % '&DISPLAY=AUTHOR&WARN=')
    # Accessing each link and scrape text element
    for j in link:
        print(j.get_attribute('href'))
        url = j.get_attribute('href')
        page_response = requests.get(url)
        soup = BeautifulSoup(page_response.text, 'lxml')
        text = soup.find_all('td')
        a = ' '.join(text[1].get_text(strip=True, separator = ' ' ).split())
        # Adding the article to full_text list
        full_text.append(a)
    # Moving to next page
    next_page = browser.find_elements_by_xpath("//a[contains(@href, '%s')]" % '&SIZE=40&RETRIEVEFROM=')[i]
    next_page.click()


# Scraping metadata
title = []
date = []
for i in range(0, no_of_pages):
    url = browser.current_url
    page_response = requests.get(url)
    mainsoup = BeautifulSoup(page_response.text, 'lxml')
    # Scraping titles
    title_ph = mainsoup.find_all('i')
    for ititle in title_ph:
        title.append(ititle.text)
    # Scraping date
    date_ph = mainsoup.find_all('span', {'class' : 'boldtext'}, string = 'Date:')
    for idate in date_ph:
        ph = idate.next_sibling
        date.append(ph.replace('\n', '').replace('   ', ''))
#    author_ph = mainsoup.find_all('span', {'class' : 'boldtext'}, string = 'Date:')
    # Moving to next page
    next_page = browser.find_elements_by_xpath("//a[contains(@href, '%s')]" % '&SIZE=40&RETRIEVEFROM=')[i]
    next_page.click()


# Create dataframe
df = pd.DataFrame({'title' : title,
                   'date' : date,
                   'full_text' : full_text})
# Filter out records that are too large to be displayed on the site
df[df['full_text'].str.contains('Warning')]
df = df[~df.full_text.str.contains('Warning')]


# Export to csv
df.to_csv('eebogravity.csv', encoding = 'utf-8')
    





