"""
Author: Ben Pollack

This script is meant to analyze web pages using TF-IDF to discover websites that
are similar in content. The use case for this script would be to determine if
there are mutiple web servers running the same application even if the content
is not 100% the same. the smaller the distance between two URLs the more likely
they contain similar content. Exact matches should result in a distance of 0.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from bs4 import BeautifulSoup
from bs4.element import Comment
from selenium import webdriver
import urllib.request
import sys
import matplotlib.pyplot as plt          # plotting
import numpy as np                       # dense matrices
from scipy.sparse import csr_matrix      # sparse matrices
import pandas as pd                      # data frames
import json
"""
Code to extract content borrowed from:
https://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text
"""
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)


def snapshot(webpage):
    DRIVER = 'chromedriver'
    driver = webdriver.Chrome(DRIVER)
    DRIVER = 'chromedriver'
    driver = webdriver.Chrome(DRIVER)
    filename = webpage.replace('/', '').strip()
    filename = filename.replace(':', '')
    filename = './media/' + filename + '.png'
    driver.get(url)
    screenshot = driver.save_screenshot(filename)
    driver.quit()

"""
Code to take screenshots of webpages taken from:
https://medium.com/@ronnyml/website-screenshot-generator-with-python-593d6ddb56cb
"""
url_file = sys.argv[1] # path to file with URLs

responses = []
response_listing = []

# get content of the pages
with open(url_file) as file_in:
    for url in file_in:
        try:
            html = urllib.request.urlopen(url).read()
            response_listing.append(url.strip())
            responses.append(html)
            snapshot(url)
        except:
            print('This URL is causing an issue: ' + url)

"""
TF-IDF was taken from a machine learning class I took at CMU.
I will need to look into whether the algorithm I am using is right
for what I am trying to do. Results from testing seem to be return good results.
"""

vectorizer = TfidfVectorizer()
tf_idf_vectors = vectorizer.fit_transform(responses)


model_tf_idf = NearestNeighbors(metric='euclidean', algorithm='brute')
model_tf_idf.fit(tf_idf_vectors)
count=0
masterlist = []
while count < len(response_listing):
    distances, indices = model_tf_idf.kneighbors(tf_idf_vectors[count], n_neighbors=len(response_listing))
    neworder = []
    for x in indices.flatten():
        neworder.append(response_listing[x])

    df = pd.DataFrame(indices.flatten(), columns=['index']);
    df["distance"] = distances.flatten()
    df["URL"] = neworder
    temp = [response_listing[count]] * (len(response_listing))
    df["primary_URL"] = temp
    masterlist.append(df)
    count += 1

print(masterlist)
