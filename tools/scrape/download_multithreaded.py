"""Example script to download images in a multithreaded way images from dog_urls.txt into images/"""

import os
import urllib
import requests
import time
import threading

def progress(i, num_total, success, t):
    """Function for printing progress and time elapsed."""
    dt = time.time() - t
    print "%d/%d, successful: %d, Elapsed Time: %0.3f, Time Remaining: %0.3f" % (
            i, num_total, success, dt,  1.0 * dt / (i+1) * (num_total-i-1))

def progress_string(i, num_total, success, t):
    """Function for printing progress and time elapsed."""
    dt = time.time() - t
    return "%d/%d, successful: %d, Elapsed Time: %0.3f, Time Remaining: %0.3f" % (
            i, num_total, success, dt,  1.0 * dt / (i+1) * (num_total-i-1))

def fetch(url, classname):
    """Downloads the url images into images/dogbreed/[url without special characters]"""

    # Save into
    classname = classname.replace(" ", "_")
    classname = classname.replace("\"","")
    classpath = os.path.join(os.getcwd(), 'images', classname)
    if not os.path.exists(classpath):
        os.makedirs(classpath)
    ext = os.path.splitext(url)[1]
    savename = "".join([char for char in url if char.isalnum()]) + ext
    savename = os.path.join(classpath, savename)

    if os.path.exists(savename):
        return
    else:
        r = requests.head(url, timeout=0.2)
        if r.status_code == requests.codes.ok:
            urllib.urlretrieve(url, savename)


def readfile(filename):
    return [line.strip() for line in open(filename).readlines()]


class ThreadDownload(threading.Thread):

    def __init__(self, datasubset, threadName):
        threading.Thread.__init__(self)
        self.dataset = datasubset
        self.threadName = threadName


    def run(self):
        """Function passed into the threaded."""
        print 'Launching Thread %s' % self.threadName
        t = time.time()
        success_counter = 0
        for i, entry in enumerate(self.dataset):
            if i % 5 == 0:
                with open("Thread-%s.log" % self.threadName, 'a') as f:
                    f.write(progress_string(i, len(self.dataset), success_counter, t))
                    f.write('\n')
            freebase_id = entry.split(',')[0]
            classname = classes[freebase_id]
            urls = entry.split(',')[1:]

            # Multiple urls in a single entry mean the image is hosted on multiple sites.
            # Download the first image that is still live.
            for url in urls:
                try:
                    fetch(url, classname)
                    success_counter += 1
                    break
                except:
                    pass

# Format: [freebase_id1],[url0],[url1],[url2],...
#         [freebase_id2],[url0],[url1],[url2],...
#         [freebase_id3],[url0],[url1],[url2],...
#         [freebase_id4],[url0],[url1],[url2],...
dataset = readfile('dog_urls.txt')

# Read in the classes, assert they each have the /m/ prefix in their freebase id,
# and conver them to a dictionary for later looking up.
classes = readfile('dog_classes.txt')
for entry in classes:
    freebase_id = entry.split(',')[1]
    assert freebase_id[:3] == "/m/"
classes = {entry.split(',')[1] : entry.split(',')[0] for entry in classes}


N = 24
def batch_iterator(dataset, N):
    chunk_size = len(dataset) / N
    for i in range(N):
        yield(dataset[i*chunk_size:min(len(dataset), (i+1)*chunk_size)])

t = time.time()
threads = []
for i, sub in enumerate(batch_iterator(dataset, N)):
    thread = ThreadDownload(sub, str(i))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()

print 'Total Elapsed Time: %0.3f' % (time.time() - t)
