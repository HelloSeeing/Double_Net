import os
import sys
import numpy as np
import data_augment
import cv2
import time
from PIL import Image
import random

class Transform:
    def __init__(self):
        self._color_balance = data_augment.Color_Balance()

    def rotate_90(self, image1, image2):
        rows, cols = image1.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        image1_rotated = cv2.warpAffine(image1, M, (cols, rows))
        image2_rotated = cv2.warpAffine(image2, M, (cols, rows))
        return image1_rotated, image2_rotated

    def color_balance(self, image1, image2):
        image1_balanced = self._color_balance.apply(image1)
        image2_balanced = self._color_balance.apply(image2)
        return image1_balanced, image2_balanced

    def random_rotate(self, image1, image2):
        dice = random.randint(0, 3)
        for angle in range(dice):
            img1, img2 = self.rotate_90(image1, image2)
        return img1, img2

    def random_color_balance(self, image1, image2):
        dice = random.randint(0, 1)
        if dice == 1:
            return self.color_balance(image1, image2)
        return image1, image2

    def random_flip_on_x(self, image1, image2):
        dice = random.randint(0, 1)
        if dice == 1:
            image1 = cv2.flip(image1, flipCode=0)
            image2 = cv2.flip(image2, flipCode=0)
        return image1, image2

    def random_flip_on_y(self, image1, image2):
        dice = random.randint(0, 1)
        if dice == 1:
            image1 = cv2.flip(image1, flipCode=1)
            image2 = cv2.flip(image2, flipCode=1)
        return image1, image2

    def random_flip_on_xy(self, image1, image2):
        dice = random.randint(0, 1)
        if dice == 1:
            image1 = cv2.flip(image1, flipCode=-1)
            image2 = cv2.flip(image2, flipCode=-1)
        return image1, image2

class PY_Reader:
    def __init__(self,
                 default_image_height = 256,
                 default_image_width = 256,
                 image_depth = 3,
                 do_data_augment = None):

        self.default_image_width = default_image_width
        self.default_image_height = default_image_height
        self.default_size = (default_image_height, default_image_width)
        self.image_depth = image_depth
        self.do_data_augment = do_data_augment

        self.transform = Transform()

    def _read_py_function(self, image1_fname, image2_fname):
        image1_decoded = np.array(Image.open(image1_fname)).astype(np.uint8)
        image2_decoded = np.array(Image.open(image2_fname)).astype(np.uint8)
        return image1_decoded, image2_decoded

    def _resize_function(self, image1_decoded, image2_decoded):
        image1_resized = cv2.resize(image1_decoded, (self.default_image_height, self.default_image_width))
        image2_resized = cv2.resize(image2_decoded, (self.default_image_height, self.default_image_width))
        return image1_resized, image2_resized

    def read(self, image1_fnames, image2_fnames):
        """
        read images1_fnames to images and images2_fnames in python
        """
        images1 = []
        images2 = []
        target_size = (self.default_image_height, self.default_image_width)
        for image1_fname, image2_fname in zip(image1_fnames, image2_fnames):
            img1, img2 = self._read_py_function(image1_fname, image2_fname)
            if self.do_data_augment is not None:
                if 'center_expand' in self.do_data_augment:
                  img1, img2 = self.transform.center_expand(img1, img2, target_size)
                else:
                  img1, img2 = self._resize_function(img1, img2)
                if 'rotate' in self.do_data_augment:
                  img1, img2 = self.transform.rotate_90(img1, img2)
                if 'color_balance' in self.do_data_augment:
                  img1, img2 = self.transform.color_balance(img1, img2)
                if 'random_rotate' in self.do_data_augment:
                  img1, img2 = self.transform.random_rotate(img1, img2)
                if 'random color_balance' in self.do_data_augment:
                  img1, img2 = self.transform.random_color_balance(img1, img2)
                if 'flip' in self.do_data_augment:
                  img1, img2 = self.transform.random_flip_on_x(img1, img2)
                  img1, img2 = self.transform.random_flip_on_y(img1, img2)
                  img1, img2 = self.transform.random_flip_on_xy(img1, img2)
            else:
              img1, img2 = self._resize_function(img1, img2)
            images1.append(img1)
            images2.append(img2)

        images1 = np.stack(images1, axis = 0).astype(np.uint8)
        images2 = np.stack(images2, axis = 0).astype(np.uint8)

        return images1, images2

class PY_Parser:
    def __init__(self,
                 records_list_fname,
                 default_image_height = 256,
                 default_image_width = 256,
                 image_depth = 3,
                 do_data_augment = None,
                 batch_size = 1,
                 num_epochs = 1,
                 do_shuffle = False):
        self.records_list_fname = records_list_fname
        self.image1_names, self.image2_names, self.labels = self._parse_listfile(self.records_list_fname)
        self.reader = PY_Reader(default_image_height, default_image_width, image_depth, do_data_augment)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        ## For eternal loop parse. Edit in 2017.11.14
        self._loop = False
        if num_epochs > 0:
            self.image1_names = self.image1_names * num_epochs
            self.image2_names = self.image2_names * num_epochs
        elif num_epochs < 0:
            raise ValueError, 'num_epochs > 0 required, however it is %d' % num_epochs
        else:
            self._loop = True
        self._terminate = len(self.image1_names)
        self.do_shuffle = do_shuffle
        self._shuffle_examples()
        self._count = 0

    def _shuffle_examples(self):
        if self.do_shuffle:
            indices = range(self.total_num)
            random.shuffle(indices)
            shuffle_images1 = [self.image1_names[i] for i in indices]
            shuffle_images2 = [self.image2_names[i] for i in indices]
            shuffle_labels = [self.labels[i] for i in indices]
            self.image1_names = shuffle_images1[:]
            self.image2_names = shuffle_images2[:]
            self.labels = shuffle_labels[:]

    def _parse_listfile(self, records_list_fname):
        with open(records_list_fname, 'r') as fid:
            lines = fid.readlines()
        image1_names = [line.strip().split()[0] for line in lines]
        image2_names = [line.strip().split()[1] for line in lines]
        labels = [int(line.strip().split()[2]) for line in lines]
        return image1_names, image2_names, labels

    def __getitem__(self, n):
        begin = n % self._terminate
        end = begin + self.batch_size
        if end > self._terminate:
            image1_names = self.image1_names[begin:]
            image2_names = self.image2_names[begin:]
            labels = self.labels[begin:]
        else:
            image1_names = self.image1_names[begin:end]
            image2_names = self.image2_names[begin:end]
            labels = self.labels[begin:end]
        images1, images2 = self.reader.read(image1_names, image2_names)
        return images1, images2, labels

    def __iter__(self):
        return self

    def next(self):
        """
        Iteration in getting images and labels from records_list_fname
        """
        if not self._loop:
            if self._count < self._terminate:
                begin = self._count
                end = begin + self.batch_size
                if end > self._terminate:
                    end = self._terminate
                image1_names = self.image1_names[begin:end]
                image2_names = self.image2_names[begin:end]
                labels = self.labels[begin:end]
                images1, images2 = self.reader.read(image1_names, image2_names)
                self._count = end
                return images1, images2, labels
            else:
                self._count = 0
                self._shuffle_examples()
                raise StopIteration();
        else:
            begin = self._count % self._terminate
            end = begin + self.batch_size
            if end <= self._terminate:
                image1_names = self.image1_names[begin:end]
                image2_names = self.image2_names[begin:end]
                labels = self.labels[begin:end]
            else:
                image1_names = self.image1_names[begin:end]
                image2_names = self.image2_names[begin:end]
                labels = self.labels[begin:end]
                self._shuffle_examples()
                end -= self._terminate
                image1_names += self.image1_names[:end]
                image2_names += self.image2_names[:end]
                labels += self.labels[:end]
            images1, images2 = self.reader.read(image1_names, image2_names)
            self._count = end
            return images1, images2, labels

    @property
    def total_num(self):
        return len(self.image1_names)

def main():
    records_list_fname = '/home/dxh/skin_doctor/Trainer/U-net_Trainer/data/isic_2017/segments.trainval'
    batch_size = 20
    data_augments = ['flip']
    data_parser = PY_Parser(records_list_fname=records_list_fname, batch_size = batch_size, do_data_augment=data_augments)

    cnt = 0
    t0 = time.time()
    t1 = t0
    for images1, images2 in data_parser:
        curr_time = time.time()
        print '{}: {} {} cost time {}'.format(cnt, images1.shape, images2.shape, curr_time - t1)
        t1 = curr_time
        cnt += 1
    print 'batch cost time %.4f' % ((time.time() - t0) / cnt)

if __name__ == '__main__':
    main()

