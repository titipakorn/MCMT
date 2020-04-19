"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import face_align
from retinaface import RetinaFace
import sklearn
import mxnet as mx
import json
import logging as log
from collections import namedtuple
from abc import ABC, abstractmethod
import sys
import os

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

sys.path.append(os.path.join(os.path.dirname(__file__),  '..', '..', 'common'))


class DetectorInterface(ABC):
    @abstractmethod
    def run_async(self, frames, index):
        pass

    @abstractmethod
    def wait_and_grab(self):
        pass


class Detector(DetectorInterface):
    """Wrapper class for detector"""

    def __init__(self, model_path='./retinaface-R50/R50', conf=.6, max_num_frames=1):
        self.net = detector = RetinaFace(
            model_path, 0, 0, 'net3')
        self.confidence = conf
        self.expand_ratio = (1., 1.)
        self.max_num_frames = max_num_frames

    def run_async(self, frames):
        assert len(frames) <= self.max_num_frames
        self.shapes = []
        all_detections = []
        scales = [720, 1280]
        target_size = scales[0]
        max_size = scales[1]
        for i in range(len(frames)):
            im_shape = frames[i].shape
            self.shapes.append(im_shape)
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            im_scale = float(target_size) / float(im_size_min)
            # prevent bigger axis from being more than max_size:
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)
            all_detections.append(self.net.detect(frames[i], self.confidence,
                                                  scales=[im_scale], do_flip=False))
        return all_detections

    def get_detections(self, frames):
        """Returns all detections on frames"""
        return self.run_async(frames)


class VectorCNN:
    """Wrapper class for a network returning a vector"""

    def __init__(self, image_size=(112, 112), model_path='./model-r100-ii/model,0'):
        ctx = mx.gpu(0)
        self.model = self.get_model(ctx, image_size, model_path, 'fc1')

    def get_model(ctx, image_size, model_str, layer):
        _vec = model_str.split(',')
        assert len(_vec) == 2
        prefix = _vec[0]
        epoch = int(_vec[1])
        print('loading', prefix, epoch)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers[layer+'_output']
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
        model.bind(
            data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        return model

    def get_align_input(self, img, points):
        nimg = face_align.norm_crop(
            img, landmark=points, image_size=112, mode='arcface')
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2, 0, 1))
        return aligned

    def get_feature(self, aligned):
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = sklearn.preprocessing.normalize(embedding).flatten()
        return embedding

    def forward(self, batch):
        """Performs forward of the underlying network on a given batch"""
        embedding = []
        for frame in batch:
            embedding.append(self.get_feature(self.get_align_input(frame)))
        return embedding


class DetectionsFromFileReader(DetectorInterface):
    """Read detection from *.json file.
    Format of the file should be:
    [
        {'frame_id': N,
         'scores': [score0, score1, ...],
         'boxes': [[x0, y0, x1, y1], [x0, y0, x1, y1], ...]},
        ...
    ]
    """

    def __init__(self, input_file, score_thresh):
        self.input_file = input_file
        self.score_thresh = score_thresh
        self.detections = []
        log.info('Loading {}'.format(input_file))
        with open(input_file) as f:
            all_detections = json.load(f)
        for source_detections in all_detections:
            detections_dict = {}
            for det in source_detections:
                detections_dict[det['frame_id']] = {
                    'boxes': det['boxes'], 'scores': det['scores']}
            self.detections.append(detections_dict)

    def run_async(self, frames, index):
        self.last_index = index

    def wait_and_grab(self):
        output = []
        for source in self.detections:
            valid_detections = []
            if self.last_index in source:
                for bbox, score in zip(source[self.last_index]['boxes'], source[self.last_index]['scores']):
                    if score > self.score_thresh:
                        bbox = [int(value) for value in bbox]
                        valid_detections.append((bbox, score))
            output.append(valid_detections)
        return output
