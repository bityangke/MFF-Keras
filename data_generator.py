import pdb
import numpy as np
import random

from PIL import Image

from datasets_video import *
from keras.preprocessing.image import ImageDataGenerator

import sys
sys.path.append('..')

from utils import *

richard = True # fixes everything

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return str(self._data[0])

    @property
    def num_frames(self):
        return self._data[1]

    @property
    def label(self):
        return self._data[2]

class MFFGenerator():
    def __init__(self, num_classes, index_file, data_root_path, img_dim,
        prefix='{:05d}.jpg', modality='RGBFlow', new_length=3, num_segments=4, 
        training=True, transform_fn=None, batch_size=100):

        self.training = training
        self.modality = modality
        self.num_segments = num_segments
        self.num_classes = num_classes
        self.data_root_path = data_root_path
        self.prefix = prefix
        self.new_length = new_length
        self.batch_size = batch_size
        self.img_dim = img_dim
        self.transform_fn = transform_fn


        if self.modality == 'RGB':
            self.num_channels = 3
        if self.modality == 'RGBFlow':
            self.num_channels = 3 + self.new_length*2
        if self.modality == 'Flow':
            self.num_channels = self.new_length*2
        if self.modality == 'RGBDiff':
            raise('Generator for RGBDiff not yet implemented')

        record_data = np.genfromtxt(index_file, delimiter=" ", dtype=int)
        self.video_list = [VideoRecord(item) for item in record_data]
        random.shuffle(self.video_list)
        self.video_list = self.video_list[:1000]

        self.segment_frames = np.zeros((len(self.video_list), self.num_segments))

        for i, record in enumerate(self.video_list):
            if training:
                self.segment_frames[i] = self._get_train_indices(record)
            else:
                self.segment_frames[i] = self._get_valtest_indices(record)
        
        if richard:
            self.standardize_data()

    # formerly sample_indices
    def _get_train_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments

        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + np.random.randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(np.random.randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return (offsets + 1).astype(int)

    # returns indices of middle frame from each segment
    def _get_valtest_indices(self, record):
        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return (offsets + 1).astype(int)

    def _load_image(self, directory, idx, isLast=False):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return Image.open(os.path.join(self.data_root_path, "rgb", directory, self.prefix.format(idx)))
            except Exception:
                print('error loading image:', os.path.join(self.data_root_path, "rgb", directory, self.prefix.format(idx)))
                return Image.open(os.path.join(self.data_root_path, "rgb", directory, self.prefix.format(1))).convert('RGB')
            
        elif self.modality == 'Flow':
            try:
                idx_skip = 1 + (idx-1)*self.new_length
                flow = Image.open(os.path.join(self.data_root_path, directory, self.prefix.format(idx_skip)))
            except Exception:
                print('error loading flow file:', os.path.join(self.data_root_path, directory, self.prefix.format(idx_skip)))
                flow = Image.open(os.path.join(self.data_root_path, directory, self.prefix.format(1))).convert('RGB')
            # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
            flow_x, flow_y, _ = flow.split()
            x_img = flow_x.convert('L')
            y_img = flow_y.convert('L')
            return [x_img, y_img]

        elif self.modality == 'RGBFlow':
            if isLast:
                return Image.open(os.path.join(self.data_root_path, "rgb", directory, self.prefix.format(idx)))
            else:
                x_img = Image.open(os.path.join(self.data_root_path, "flow/u", directory, self.prefix.format(idx))).convert('L')
                y_img = Image.open(os.path.join(self.data_root_path, "flow/v", directory, self.prefix.format(idx))).convert('L')
                return [x_img, y_img]


    def get(self, record, indices):
        siamese_input = list()
        for segment_index in indices:
            base_model_input = np.zeros((self.img_dim, self.img_dim, self.num_channels))
            for flow_index in range(self.new_length):
                
                flow_frame = segment_index + flow_index
                
                if self.modality == 'RGBFlow' or self.modality == 'Flow':
                    flow_frame = min(max(flow_frame, 1),record.num_frames-1)
                    x_flow, y_flow = self._load_image(record.path, flow_frame)
                    base_model_input[:, :, 2*flow_index] = resize_img(crop_1_to_1(x_flow), (self.img_dim, self.img_dim))
                    base_model_input[:, :, 2*flow_index + 1] = resize_img(crop_1_to_1(y_flow), (self.img_dim, self.img_dim))

            if self.modality == 'RGBFlow' or self.modality == 'RGB':
                rgb_frame = segment_index+self.new_length+1
                rgb_frame = min(max(rgb_frame, 1),record.num_frames)
                rgb = self._load_image(record.path, rgb_frame, isLast=True)
                base_model_input[:, :, -3:] = resize_img(crop_1_to_1(rgb), (self.img_dim, self.img_dim))

            siamese_input.append(base_model_input.astype(np.uint8))
        
        return siamese_input
    
    def standardize_data(self):
        data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)

        X = np.zeros((self.num_segments, len(self.video_list), self.img_dim, self.img_dim, self.num_channels))
        for j, segment_indices in enumerate(self.segment_frames):
            record = self.video_list[j]
            images = self.get(record, segment_indices.astype(int))
            pdb.set_trace()
            X[:, j, :, :, :] = images
        
        X_all_segments = X.reshape((-1, *X.shape[2:])) # (num_seg * batch_size, img dim, img dim, total channels)
        X_rgb = X_all_segments[:,:,:,-3:] # (num_seg * batch_size, img dim, img dim, 3)
        self.rgb_transformer = ImageDataGenerator(**data_gen_args)
        self.rgb_transformer.fit(X_rgb)
        X_rgb = None

        X_flow = np.transpose(X_all_segments[:,:,:,:-3], (0, 3, 1, 2)) # (num_seg * batch_size, total channels-3, img dim, img dim)
        X_flow = X_flow.reshape((-1, *X_flow.shape[2:])) # (num_seg * batch_size * num flow channels, img dim, img dim)
        self.flow_transformer = ImageDataGenerator(**data_gen_args)
        self.flow_transformer.fit(X_flow)
        X_all_segments = None
        X_flow = None

        # X_all_segments = X.reshape((-1, *X.shape[2:])) # (num_seg * batch_size, img dim, img dim, total channels)
        # self.transform_generator = ImageDataGenerator(**data_gen_args)
        # self.transform_generator.fit(X_all_segments)

    def generator(self):
        # TODO: SHUFFLE segment_frames
        num_batches = len(self.segment_frames)//self.batch_size

        while True:
            for i in range(num_batches):
                if i == num_batches - 1:
                    start_i = i * self.batch_size
                    batch_indices = self.segment_frames[start_i:]
                else:
                    start_i = self.batch_size*i
                    end_i = start_i + self.batch_size
                    # idx of Every example in the current batch
                    batch_indices = self.segment_frames[start_i:end_i]

                # A list of batches, one batch per segment
                X = np.zeros((self.num_segments, len(batch_indices), self.img_dim, self.img_dim, self.num_channels))
                Y = np.zeros((len(batch_indices), self.num_classes))
                
                for j, segment_indices in enumerate(batch_indices):
                    record = self.video_list[start_i + j]
                    images = self.get(record, segment_indices.astype(int))
                    X[:, j, :, :, :] = images
                    Y[j, record.label] = 1

                # if self.transform_fn:
                #     X_list = [self.transform_fn(segment) for segment in X]
                # else:
                if richard:
                    X = X.reshape((-1, *X.shape[2:])) # (num_seg * batch_size, img dim, img dim, total channels)
                    X_rgb = X[:,:,:,-3:] # (num_seg * batch_size, img dim, img dim, 3)
                    X_flow = np.transpose(X[:,:,:,:-3], (0, 3, 1, 2)) # (num_seg * batch_size, total channels-3, img dim, img dim)
                    X_flow = X_flow.reshape((-1, *X_flow.shape[2:])) # (num_seg * batch_size * num flow channels, img dim, img dim)

                    X_rgb = next(self.rgb_transformer.flow(X_rgb, batch_size=X_rgb.shape[0]))
                    X_flow = next(self.flow_transformer.flow(X_flow, batch_size=X_flow.shape[0]))

                    X_flow = X_flow.reshape((self.num_segments * len(batch_indices), *X_flow.shape[1:])) # (num_seg * batch_size, total channels-3, img dim, img dim)
                    X_flow = np.transpose(X_flow, (0, 2, 3, 1))

                    X = np.concatenate((X_flow, X_rgb), axis=0)
                    X = X.reshape((self.num_segments, len(batch_indices), self.img_dim, self.img_dim, self.num_channels))
                    X = [segment for segment in X]
                    X_flow = None
                    X_rgb = None
                    # X = X.reshape((-1, *X.shape[2:])) # Flatten segments and batches
                    # X = next(self.transform_generator.flow(X, batch_size=X.shape[0]))
                    # X = X.reshape((self.num_segments, len(batch_indices), self.img_dim, self.img_dim, self.num_channels))
                    # X = [segment for segment in X]
                else: 
                    X = [self.transform_fn(segment) for segment in X]
                pdb.set_trace()
                yield (X, Y)


