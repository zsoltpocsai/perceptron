import numpy as np
import gzip
from urllib.request import urlopen
from pathlib import Path

train_images_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
train_labels_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
test_images_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
test_labels_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

default_files_dir = 'mnist-files'

paths = { 'train': { 'images': { 'url': train_images_url, 'file': 'mnist_train_images.gz' },
                     'labels': { 'url': train_labels_url, 'file': 'mnist_train_labels.gz' } },

          'test':  { 'images': { 'url': test_images_url,  'file': 'mnist_test_images.gz' },
                     'labels': { 'url': test_labels_url,  'file': 'mnist_test_labels.gz' } }
        }

def download_mnist(files_dir=''):
    _set_file_paths(files_dir)
    for dataset in ('train', 'test'):
        for type in ('images', 'labels'):
            _download(paths[dataset][type]['url'], paths[dataset][type]['file'])

def _set_file_paths(files_dir='', interactive=True):
    if interactive:
        files_dir = input(f"MNIST files directory ({default_files_dir}): ") or default_files_dir
    else:
        files_dir = files_dir or default_files_dir
    for dataset in ('train', 'test'):
        for type in ('images', 'labels'):
            file_name = paths[dataset][type]['file']
            paths[dataset][type]['file'] = Path(Path.cwd(), files_dir, file_name).__str__()

def _download(url, file_path):
    assert isinstance(file_path, str)

    path = Path(Path.cwd(), file_path)

    if not path.parent.exists():
        print(f"Creating folder '{path.parent}'.")
        path.parent.mkdir()

    if path.exists() and path.is_file():
        print(f"File '{path}' has been found.")
    else:
        print(f"File '{path}' is not found. Downloading... ", end='')
        response = urlopen(url)
        with open(path, 'wb') as file:
            file.write(response.read())
        response.close()
        print("Done.")

def load_mnist(files_dir=''):
    _set_file_paths(files_dir, interactive=False)
    mnist = {
        'train': { 'images': _load_images(paths['train']['images']['file']), 
                   'labels': _load_labels(paths['train']['labels']['file']) },
        'test': { 'images': _load_images(paths['test']['images']['file']), 
                  'labels': _load_labels(paths['test']['labels']['file']) }
    }
    return mnist

def _load_images(file):
    assert Path(file).exists(), f"Can't find '{file}'!"
    print(f"Loading images from '{file}'...")
    images = []
    with gzip.open(file, 'rb') as file:
        file.seek(4) # skip magic number
        num_of_images = int.from_bytes(file.read(4))
        rows = int.from_bytes(file.read(4))
        cols = int.from_bytes(file.read(4))
        for i in range(num_of_images):
            images.append(_read_image(file, rows, cols))
    return np.array(images, dtype=np.uint8)

def _read_image(file, rows, cols):
    image = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(int.from_bytes(file.read(1)))
        image.append(row)
    return np.array(image, dtype=np.uint8)

def _load_labels(file):
    assert Path(file).exists(), f"Can't find '{file}'!"
    print(f"Loading labels from '{file}'...")
    labels = []
    with gzip.open(file, 'rb') as file:
        file.seek(4) # skip magic number
        num_of_labels = int.from_bytes(file.read(4))
        for i in range(num_of_labels):
            labels.append(_read_label(file))
    return np.array(labels, dtype=np.uint8)

def _read_label(file):
    return np.array([int.from_bytes(file.read(1))])
