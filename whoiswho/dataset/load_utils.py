import requests
from tqdm import tqdm
import os
import json
import sys
from time import time
# name: the version of whoiswho data, ['v1', 'v2', 'v3']
# type: data partition of the data, ['train', 'valid', 'test']
# path: saved dir
# partition: none
DATA_PATH = "https://lfs.aminer.cn/misc/ND-data"
NAME_SET = set(['v1', 'v2', 'v3'])
TYPE_SET = set(['train', 'valid', 'test'])

# function to display download progress bar
def progress_bar(progress, total, speed):
    filled = int(progress * 40 // total)
    bar = '█' * filled + '-' * (40 - filled)
    sys.stdout.write('\r|%s| %d%% (%.2f KB/s)' % (bar, progress / total * 100, speed/1024))
    sys.stdout.flush()


def LoadData(name: float, type: str, path = './', partition=None):
    if name not in NAME_SET:
        raise ValueError(f"NAME must in {NAME_SET}")
    if type not in TYPE_SET:
        raise ValueError(f"TYPE must in {TYPE_SET}")
    url_list = []
    # Check if the download path exists, and create it if it doesn't
    if not os.path.exists(path):
        os.makedirs(path)

    
    # Define the URL of the data to download
    if type == 'train':
        url = os.path.join(DATA_PATH, 
                        f"na-{name}",
                        f"train_author.json")
        url_list.append((url, "train_author.json"))
        url = os.path.join(DATA_PATH, 
                        f"na-{name}",
                        f"train_pub.json")
        url_list.append((url, "train_pub.json"))


    # Define the request headers
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    ret = []
    
    for url, filename in url_list:
        if os.path.exists(filename):
            print(f"{filename} already downloaded...")
            with open(os.path.join(path, filename), 'r') as f:
                content = json.load(f)
            ret.append(content)
            continue
        # print(f"Downloading {filename}")
        # Send the request and get the response
        response = requests.get(url, headers=headers, stream=True)
        
        # check if the request was successful
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.reason}")
            response.raise_for_status()
        
        # Get the total size of the file in bytes
        total_size = int(response.headers.get('content-length', 0))

        # Define a chunk size of 1 MB
        chunk_size = 1024 * 1024
        
        # open a file for writing in JSON format
        progress = 0
        start_time = None
        speed = 0



        # open a file for writing in JSON format
        with open(os.path.join(path, filename), 'w') as f:

            # iterate over the response content in chunks
            for chunk in response.iter_content(chunk_size=1024):

                # calculate the progress and download speed
                progress += len(chunk)
                if not start_time:
                    start_time = time()
                else:
                    download_time = time() - start_time
                    speed = progress / download_time

                # write the chunk to the file
                f.write(chunk.decode('utf-8', errors='ignore'))
                
                # update the progress bar and download speed
                progress_bar(progress, total_size, speed)
        
        # check if the file was downloaded successfully
        if total_size != 0 and progress != total_size:
            print(f"\nError: failed to download the {filename} file")
        else:
            print(f"\nDownload {filename} successful!")
        
        with open(os.path.join(path, filename), 'r') as f:
            content = json.load(f)
        ret.append(content)
    
    return ret