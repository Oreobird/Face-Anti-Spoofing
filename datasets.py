import os
import numpy as np
import cv2
import random
import tensorflow as tf


class DataSet:
    def __init__(self, data_dir, batch_size=64, input_size=64):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.__train_num, self.__val_num, self.__test_num = self.__get_samples_num(os.path.join(self.data_dir, 'train.txt'),
                                                               os.path.join(self.data_dir, 'val.txt'),
                                                               os.path.join(self.data_dir, 'test.txt'))

    def __get_samples_num(self, train_label_file, val_label_file, test_label_file):
        train_num = 0
        val_num = 0
        test_num = 0
        with open(train_label_file) as f:
            train_num = len(f.readlines())
        with open(val_label_file) as f:
            val_num = len(f.readlines())
        with open(test_label_file) as f:
            test_num = len(f.readlines())
        return train_num, val_num, test_num


    def load_input_img(self, data_dir, file_name, color_space=None):
        img = cv2.imread(os.path.join(data_dir, file_name))
        
        if color_space:
            img = cv2.cvtColor(img, color_space)
        
        img = cv2.resize(img, (self.input_size, self.input_size)) / 255.0
        return img

    def load_input_imgpath_label(self, file_name, labels_num=1, shuffle=True):
        imgpath = []
        labels = []

        with open(os.path.join(self.data_dir, file_name)) as f:
            lines_list = f.readlines()
            if shuffle:
                random.shuffle(lines_list)
           
            for lines in lines_list:
                line = lines.rstrip().split(',')
                label = []
                if labels_num == 1:
                    label = int(line[1])
                else:
                    lab = line[1].split(' ')
                    for i in range(labels_num):
                        label.append(int(lab[i]))
                imgpath.append(line[0])
                labels.append(label)
        return np.array(imgpath), np.array(labels)

    def train_num(self):
        return self.__train_num

    def val_num(self):
        return self.__val_num

    def test_num(self):
        return self.__test_num

    def load_batch_data_label(self, filename_list, label_list, label_num=1, color_space=None, shuffle=True):
        file_num = len(filename_list)
        # print("file_num: %d" % file_num)
        if shuffle:
            idx = np.random.permutation(range(file_num))
            filename_list = filename_list[idx]
            label_list = label_list[idx]
        max_num = file_num - (file_num % self.batch_size)
        # print("max_num: %d" % max_num)
        for i in range(0, max_num, self.batch_size):
            # print(i)
            batch_x = []
            batch_y = []
            for j in range(self.batch_size):
                img = self.load_input_img(self.data_dir, filename_list[i + j], color_space)
                label = label_list[i + j]
                batch_x.append(img)
                batch_y.append(label)
            batch_x = np.array(batch_x, dtype=np.float32)
            if label_num == 1:
                batch_y = tf.keras.utils.to_categorical(batch_y, 2)
            else:
                batch_y = np.array(batch_y)
            if shuffle:
                idx = np.random.permutation(range(self.batch_size))
                batch_x = batch_x[idx]
                batch_y = batch_y[idx]
            yield batch_x, batch_y


class NUAA():
    def __init__(self, nuaa_data_dir, batch_size=64, input_size=64, class_num=1):
        self.dataset = DataSet(nuaa_data_dir, batch_size, input_size)
        self.data_dir = nuaa_data_dir
        self.batch_size = batch_size
        self.class_num = class_num

    def train_num(self):
        return self.dataset.train_num()

    def val_num(self):
        return self.dataset.val_num()

    def test_num(self):
        return self.dataset.test_num()

    def train_data_generator(self, input_name_list, output_name_list, label_file_name='train.txt', labels_num=1, shuffle=False):
        filename_list, label_list = self.dataset.load_input_imgpath_label(label_file_name, labels_num=labels_num, shuffle=shuffle)
        
        while True:
            file_num = len(filename_list)
            # print("file_num: %d" % file_num)
            if shuffle:
                idx = np.random.permutation(range(file_num))
                filename_list = filename_list[idx]
                label_list = label_list[idx]
            max_num = file_num - (file_num % self.batch_size)
            # print("max_num: %d" % max_num)
            for i in range(0, max_num, self.batch_size):
                batch_x_rgb = []
                batch_x_hsv = []
                batch_x_ycrcb = []
                batch_y = []
                for j in range(self.batch_size):
                    img_hsv = self.dataset.load_input_img(self.data_dir, filename_list[i + j], cv2.COLOR_BGR2HSV)
                    img_ycrcb = self.dataset.load_input_img(self.data_dir, filename_list[i + j], cv2.COLOR_BGR2YCrCb)

                    label = label_list[i + j]

                    batch_x_hsv.append(img_hsv)
                    batch_x_ycrcb.append(img_ycrcb)
                    batch_y.append(label)

                batch_x_hsv = np.array(batch_x_hsv, dtype=np.float32)
                batch_x_ycrcb = np.array(batch_x_ycrcb, dtype=np.float32)
                
                if labels_num == 1:
                    batch_y = tf.keras.utils.to_categorical(batch_y, 2)
                else:
                    batch_y = np.array(batch_y)
                    
                if shuffle:
                    idx = np.random.permutation(range(self.batch_size))

                    batch_x_hsv = batch_x_hsv[idx]
                    batch_x_ycrcb = batch_x_ycrcb[idx]
                    batch_y = batch_y[idx]
                
                yield ({input_name_list[0]: batch_x_hsv, input_name_list[1]: batch_x_ycrcb},
                       {output_name_list[0]: batch_y})
            
    def test_data_generator(self, input_name_list, output_name_list, label_file_name='train.txt', shuffle=False):
        filenames, labels = self.dataset.load_input_imgpath_label(label_file_name, labels_num=1, shuffle=shuffle)

        hsv_generator = self.dataset.load_batch_data_label(filenames, labels,
                                                           color_space=cv2.COLOR_BGR2HSV, shuffle=shuffle)
        ycrcb_generator = self.dataset.load_batch_data_label(filenames, labels,
                                                             color_space=cv2.COLOR_BGR2YCrCb,
                                                             shuffle=shuffle)
        while True:

            hsv_batch_x, hsv_batch_y = next(hsv_generator)
            # print(hsv_batch_y)
            ycrcb_batch_x, ycrcb_batch_y = next(ycrcb_generator)
            # print(hsv_batch_y)
            # print(ycrcb_batch_y)
            yield ({input_name_list[0]: hsv_batch_x, input_name_list[1]: ycrcb_batch_x},
                   {output_name_list[0]: hsv_batch_y})
