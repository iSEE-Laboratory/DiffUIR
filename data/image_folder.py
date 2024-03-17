"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    # assert os.path.isdir(dir), '%s is not a valid directory' % dir
    if 'rain1400' in dir:
        for root, _, fnames in sorted(os.walk(dir)):
            if 'ground_truth' in root:
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        for i in range(14):
                            images.append(path)
            else:
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        images.append(path)
    elif 'RESIDE' in dir:
        for root, _, fnames in sorted(os.walk(dir)):
            if 'clear' in root:
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        for i in range(35):
                            images.append(path)
            else:
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        images.append(path)
    elif len(dir)==4:
        for di in dir:
            for root, _, fnames in sorted(os.walk(di)):
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        images.append(path)
    else:
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images[:min(max_dataset_size, len(images))]

def make_dataset_all(dirA, dirB, dirC, dirD, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dirA), '%s is not a valid directory' % dirA
    assert os.path.isdir(dirB), '%s is not a valid directory' % dirB
    assert os.path.isdir(dirC), '%s is not a valid directory' % dirC
    assert os.path.isdir(dirD), '%s is not a valid directory' % dirD

    for root, _, fnames in sorted(os.walk(dirA)):
        if 'ground_truth' in root:
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    for i in range(14):
                        images.append(path)
        else:
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    for root, _, fnames in sorted(os.walk(dirB)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    
    for root, _, fnames in sorted(os.walk(dirC)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                
    for root, _, fnames in sorted(os.walk(dirD)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images[:min(max_dataset_size, len(images))]


def make_dataset_3(dirB, dirC, dirD, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dirB), '%s is not a valid directory' % dirB
    assert os.path.isdir(dirC), '%s is not a valid directory' % dirC
    assert os.path.isdir(dirD), '%s is not a valid directory' % dirD

    for root, _, fnames in sorted(os.walk(dirB)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    
    for root, _, fnames in sorted(os.walk(dirC)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                
    for root, _, fnames in sorted(os.walk(dirD)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images[:min(max_dataset_size, len(images))]

def make_dataset_2(dirB, dirC, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dirB), '%s is not a valid directory' % dirB
    assert os.path.isdir(dirC), '%s is not a valid directory' % dirC

    for root, _, fnames in sorted(os.walk(dirB)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    
    for root, _, fnames in sorted(os.walk(dirC)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images[:min(max_dataset_size, len(images))]

def make_dataset_5(dirB, dirC, dirD, dirE, dirF, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dirB), '%s is not a valid directory' % dirB
    assert os.path.isdir(dirC), '%s is not a valid directory' % dirC
    assert os.path.isdir(dirD), '%s is not a valid directory' % dirD
    assert os.path.isdir(dirE), '%s is not a valid directory' % dirE
    assert os.path.isdir(dirF), '%s is not a valid directory' % dirF

    for root, _, fnames in sorted(os.walk(dirB)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    
    for root, _, fnames in sorted(os.walk(dirC)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                
    for root, _, fnames in sorted(os.walk(dirD)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    
    for root, _, fnames in sorted(os.walk(dirE)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    for root, _, fnames in sorted(os.walk(dirF)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)


    return images[:min(max_dataset_size, len(images))]

def make_dataset_6(dirB, dirD, dirE, dirF, dirA, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dirB), '%s is not a valid directory' % dirB
    assert os.path.isdir(dirD), '%s is not a valid directory' % dirD
    assert os.path.isdir(dirE), '%s is not a valid directory' % dirE
    assert os.path.isdir(dirF), '%s is not a valid directory' % dirF
    assert os.path.isdir(dirA), '%s is not a valid directory' % dirA

    for root, _, fnames in sorted(os.walk(dirB)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                
    for root, _, fnames in sorted(os.walk(dirD)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    
    for root, _, fnames in sorted(os.walk(dirE)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    for root, _, fnames in sorted(os.walk(dirF)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                
    for root, _, fnames in sorted(os.walk(dirA)):
        if 'clear' in root:
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    for i in range(35):
                        images.append(path)
        else:
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)


    return images[:min(max_dataset_size, len(images))]

def make_dataset_4(dirB, dirC, dirD, dirE, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dirB), '%s is not a valid directory' % dirB
    assert os.path.isdir(dirC), '%s is not a valid directory' % dirC
    assert os.path.isdir(dirD), '%s is not a valid directory' % dirD
    assert os.path.isdir(dirE), '%s is not a valid directory' % dirE

    for root, _, fnames in sorted(os.walk(dirB)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    
    for root, _, fnames in sorted(os.walk(dirC)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                
    for root, _, fnames in sorted(os.walk(dirD)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    
    for root, _, fnames in sorted(os.walk(dirE)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)


    return images[:min(max_dataset_size, len(images))]

def make_dataset_all_text(dirA, dirB, max_dataset_size=float("inf")):
    text = []
    negative_text = []
    assert os.path.isdir(dirA), '%s is not a valid directory' % dirA
    assert os.path.isdir(dirB), '%s is not a valid directory' % dirB

    for root, _, fnames in sorted(os.walk(dirA)):
        for fname in fnames:
            if is_image_file(fname):
                for i in range(14):
                    # text.append('rAn image without rain')
                    text.append('rRemove the rain in the image.')
                    negative_text.append('rrain')
       
    for root, _, fnames in sorted(os.walk(dirB)):
        for fname in fnames:
            if is_image_file(fname):
                # text.append('lAn image with more light')
                text.append('lStrengthen the light in the image.')
                negative_text.append('l')

    return text[:min(max_dataset_size, len(text))], negative_text[:min(max_dataset_size, len(negative_text))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
