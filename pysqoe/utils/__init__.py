import os
import wget
import zipfile
from config import cfg


def download(url, filename):
    r"""
    Download file from the internet.

    Args:
        url (string): URL of the webpage
        filename (string): Path to store the downloaded file.
    """
    return wget.download(url, out=filename)

def extract(filename, extract_dir):
    r"""Extract zip file.
    
    Args:
        filename (string): Path of the zip file.
        extract_dir (string): Directory to store the extracted results.
    """
    if os.path.splitext(filename)[1] == '.zip':
        if not os.path.isdir(extract_dir):
            os.makedirs(extract_dir)
        with zipfile.ZipFile(filename) as z:
            z.extractall(extract_dir)
    else:
        raise Exception('Unsupport extension {} of the compressed file {}.' % 
            (os.path.splitext(filename)[1]), filename)
