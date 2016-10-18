#!/home/esteva/anaconda/bin/python
"""Script for scraping images from google image search.

Usage:
    ./scrape_google_images.py [keywords] [to] [search] --dir="/tmp/scrape/test_images"

Example:
    ./scrape_google_images.py malignant melanoma --dir="/tmp/scrape"

"""

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import json
import os
import urllib2
import time
import sys

import argparse

parser = argparse.ArgumentParser(description='Search term to scrape')
parser.add_argument('search_terms', type=str, nargs='+', help='The search terms.')
parser.add_argument('--dir', default='/tmp/scrape')
args = parser.parse_args()


searchterm = '+'.join(args.search_terms)
url = "https://www.google.co.in/search?q="+searchterm+"&source=lnms&tbm=isch"
browser = webdriver.Chrome()
browser.get(url)
header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
counter = 0
succounter = 0

save_dir = os.path.join(args.dir, searchterm)
if not os.path.exists(save_dir):
    print "Creating directory %s" % save_dir
    os.makedirs(save_dir)

for i in range(200):
    browser.execute_script("window.scrollBy(0,10000)")
    if i % 10 == 0:
        time.sleep(0.5)
        try:
            browser.find_element("id", "smb").click()
        except:
            pass


for x in browser.find_elements_by_xpath("//div[@class='rg_meta']"):
    counter = counter + 1
    print "Total Count:", counter
    print "Succsessful Count:", succounter
    print "URL:",json.loads(x.get_attribute('innerHTML'))["ou"]

    img = json.loads(x.get_attribute('innerHTML'))["ou"]
    imgtype = json.loads(x.get_attribute('innerHTML'))["ity"]
    try:
        req = urllib2.Request(img, headers={'User-Agent': header})
        raw_img = urllib2.urlopen(req).read()
        filename = os.path.join(save_dir, searchterm + "_" + str(counter) + "." + imgtype)
#       File = open(os.path.join(searchterm , searchterm + "_" + str(counter) + "." + imgtype), "wb")
        print filename
        File = open(filename, "wb")
        File.write(raw_img)
        File.close()
        succounter = succounter + 1
    except:
            print "can't get img"

print succounter, "pictures succesfully downloaded"
browser.close()
