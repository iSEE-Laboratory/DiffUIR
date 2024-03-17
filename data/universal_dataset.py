import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset, make_dataset_all, make_dataset_all_text, make_dataset_3, make_dataset_5, make_dataset_6, make_dataset_4, make_dataset_2
from PIL import Image
from pathlib import Path
import numpy as np
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import Augmentor
import cv2

class AlignedDataset_all(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.equalizeHist = equalizeHist
        self.augment_flip = augment_flip
        self.crop_patch = crop_patch
        self.generation = generation
        self.image_size = image_size
        self.opt = opt
        #origin----------------------------------------------------------------------------------------------------------
        self.dir_Arain = os.path.join(opt.dataroot, 'rain1400/' + opt.phase + '/rainy_image')
        self.dir_Brain = os.path.join(opt.dataroot, 'rain1400/' + opt.phase + '/ground_truth')
        self.dir_Alsrw = os.path.join(opt.dataroot, 'LSRW/' + opt.phase + '/low')
        self.dir_Blsrw = os.path.join(opt.dataroot, 'LSRW/' + opt.phase + '/high')
        self.dir_Alol = os.path.join(opt.dataroot, 'LOL/' + opt.phase + '/low')
        self.dir_Blol = os.path.join(opt.dataroot, 'LOL/' + opt.phase + '/high')
        
        if opt.phase == 'train':
            self.dir_Asnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/synthetic')
            self.dir_Bsnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/gt')
            self.dir_Arain_syn = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/input')
            self.dir_Brain_syn = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/target')
            self.dir_Ablur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/input')
            self.dir_Bblur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/target')

            flog_prefix = os.path.join(opt.dataroot, 'RESIDE/OTS_ALPHA/')
            self.dir_Afog = flog_prefix + 'haze/OTS'
            self.dir_Bfog = flog_prefix + 'clear/clear_images'
        else:
            self.dir_Asnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-S/synthetic') #Snow100K-S Snow100K-L
            self.dir_Bsnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-S/gt')
            # self.dir_Asnow = os.path.join(opt.dataroot, 'Snow100K/' + 'realistic') #Snow100K-S Snow100K-L
            # self.dir_Bsnow = os.path.join(opt.dataroot, 'Snow100K/' + 'realistic')
            self.dir_Arain_syn = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test2800/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test2800/target')   #Test2800
            self.dir_Ablur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/GoPro/input')  #GoPro, HIDE,  Reblur_J, Reblur_R
            self.dir_Bblur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/GoPro/target')
            self.dir_Afog = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/hazy')
            self.dir_Bfog = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/gt')
            self.dir_Aasd = os.path.join(opt.dataroot, 'temp')
            self.dir_Basd = os.path.join(opt.dataroot, 'temp')
        
        #test
        if task == 'light':
            if opt.phase == 'train':
                self.A_paths = sorted(make_dataset_2(self.dir_Alol, self.dir_Alsrw, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset_2(self.dir_Blol, self.dir_Blsrw, opt.max_dataset_size))
            else:
                self.A_paths = sorted(make_dataset(self.dir_Alol, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Blol, opt.max_dataset_size))
        elif task == 'light_only':
            self.A_paths = sorted(make_dataset(self.dir_Alol, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Blol, opt.max_dataset_size))
        elif task == 'rain':
            self.A_paths = sorted(make_dataset(self.dir_Arain_syn, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Brain_syn, opt.max_dataset_size))
        elif task == 'snow':
            self.A_paths = sorted(make_dataset(self.dir_Asnow, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bsnow, opt.max_dataset_size))
        elif task == 'blur':
            self.A_paths = sorted(make_dataset(self.dir_Ablur, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bblur, opt.max_dataset_size))
        elif task == 'fog':
            self.A_paths = sorted(make_dataset(self.dir_Afog, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bfog, opt.max_dataset_size))
        elif task == '4':
            self.A_paths = sorted(make_dataset_4(self.dir_Arain_syn, self.dir_Alsrw, self.dir_Alol, self.dir_Asnow, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset_4(self.dir_Brain_syn, self.dir_Blsrw, self.dir_Blol, self.dir_Bsnow, opt.max_dataset_size))
        elif task == '5':
            self.A_paths = sorted(make_dataset_5(self.dir_Arain_syn, self.dir_Alsrw, self.dir_Alol, self.dir_Asnow, self.dir_Ablur, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset_5(self.dir_Brain_syn, self.dir_Blsrw, self.dir_Blol, self.dir_Bsnow, self.dir_Bblur, opt.max_dataset_size))
        elif task == '6':
            self.A_paths = sorted(make_dataset_6(self.dir_Arain_syn, self.dir_Alol, self.dir_Asnow, self.dir_Ablur, self.dir_Afog, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset_6(self.dir_Brain_syn, self.dir_Blol, self.dir_Bsnow, self.dir_Bblur, self.dir_Bfog, opt.max_dataset_size))
        else:
            self.A_paths = sorted(make_dataset(self.dir_Aasd, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Basd, opt.max_dataset_size))
    

        self.A_size = len(self.A_paths)  # get the size of dataset A
        print(self.A_size)
        self.B_size = len(self.B_paths)  # get the size of dataset B
        print(self.B_size)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        B_path = self.B_paths[index % self.B_size]

        condition = Image.open(A_path).convert('RGB') #condition
        gt = Image.open(B_path).convert('RGB') #gt
        
        if 'LOL' in A_path or 'LSRW' in A_path:
            condition = cv2.cvtColor(np.asarray(condition), cv2.COLOR_RGB2BGR)
            gt = cv2.cvtColor(np.asarray(gt), cv2.COLOR_RGB2BGR)
        
            if self.crop_patch:
                gt, condition = self.get_patch([gt, condition], self.image_size)
            if 'LOL' in A_path:
                condition = self.cv2equalizeHist(condition) if self.equalizeHist else condition
            else:
                condition = condition

            images = [[gt, condition]]
            p = Augmentor.DataPipeline(images)
            if self.augment_flip:
                p.flip_left_right(1)
            g = p.generator(batch_size=1)
            augmented_images = next(g)
            gt = cv2.cvtColor(augmented_images[0][0], cv2.COLOR_BGR2RGB)
            condition = cv2.cvtColor(augmented_images[0][1], cv2.COLOR_BGR2RGB)
        
            gt = self.to_tensor(gt)
            condition = self.to_tensor(condition)
        else:
            w, h = condition.size
            transform_params = get_params(self.opt, condition.size)
            A_transform = get_transform(self.opt, transform_params, grayscale=False)
            B_transform = get_transform(self.opt, transform_params, grayscale=False)
            condition = A_transform(condition)
            gt = B_transform(gt)
            if self.opt.phase == 'train':
                if h < 256 or w < 256:
                    osize = [256, 256]
                    resi = transforms.Resize(osize, transforms.InterpolationMode.BICUBIC)
                    condition = resi(condition)
                    gt = resi(gt)
                
        return {'adap': condition, 'gt': gt, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)
    
    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                return [p for ext in self.exts for p in Path(f'{flist}').glob(f'**/*.{ext}')]

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
        return []

    def cv2equalizeHist(self, img):
        (b, g, r) = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img = cv2.merge((b, g, r))
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)  # returns an image object.
        img_t = TF.to_tensor(img).float()
        return img_t

    def load_name(self, index, sub_dir=False):
        if self.condition:
            # condition
            name = self.input[index]
            if sub_dir == 0:
                return os.path.basename(name)
            elif sub_dir == 1:
                path = os.path.dirname(name)
                sub_dir = (path.split("/"))[-1]
                return sub_dir+"_"+os.path.basename(name)

    def get_patch(self, image_list, patch_size):
        i = 0
        h, w = image_list[0].shape[:2]
        rr = random.randint(0, h-patch_size)
        cc = random.randint(0, w-patch_size)
        for img in image_list:
            image_list[i] = img[rr:rr+patch_size, cc:cc+patch_size, :]
            i += 1
        return image_list

    def pad_img(self, img_list, patch_size, block_size=8):
        i = 0
        for img in img_list:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]
            bottom = 0
            right = 0
            if h < patch_size:
                bottom = patch_size-h
                h = patch_size
            if w < patch_size:
                right = patch_size-w
                w = patch_size
            bottom = bottom + (h // block_size) * block_size + \
                (block_size if h % block_size != 0 else 0) - h
            right = right + (w // block_size) * block_size + \
                (block_size if w % block_size != 0 else 0) - w
            img_list[i] = cv2.copyMakeBorder(
                img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            i += 1
        return img_list

    def get_pad_size(self, index, block_size=8):
        img = Image.open(self.input[index])
        patch_size = self.image_size
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        bottom = 0
        right = 0
        if h < patch_size:
            bottom = patch_size-h
            h = patch_size
        if w < patch_size:
            right = patch_size-w
            w = patch_size
        bottom = bottom + (h // block_size) * block_size + \
            (block_size if h % block_size != 0 else 0) - h
        right = right + (w // block_size) * block_size + \
            (block_size if w % block_size != 0 else 0) - w
        return [bottom, right]
