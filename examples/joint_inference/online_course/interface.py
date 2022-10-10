# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import cifar100_resnets as models
from cifar100_partition_net import resnet110_p1, resnet110_p2, resnet110_p1_head

from sedna.common.log import LOGGER as LOG
# LOG = logging.getLogger(__name__)

os.environ['BACKEND_TYPE'] = 'TORCH'


class CIFAR100Net(nn.Module):

    def __init__(self, model_type: str = "resnet18", temperature: int = 1):
        super().__init__()
        model_class = getattr(models, model_type)
        self.feature_extractor = model_class(num_classes=100)
        self.temperature = temperature

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        activations = self.feature_extractor(images)
        return activations / self.temperature


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(data: torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)


def accuracy(predictions: torch.Tensor, labels: torch.Tensor, reduce_mean: bool = True) -> torch.Tensor:
    predicted_props = F.softmax(predictions, dim=1)
    predicted_classes = torch.argmax(predicted_props, dim=1)
    correct_predictions = torch.sum(predicted_classes == labels)
    if reduce_mean:
        return correct_predictions / len(labels)
    return correct_predictions


class Estimator:

    def __init__(self, **kwargs):
        LOG.info(f"Initializing estimator with kwargs: {kwargs}")
        self.model = None
        self.model2 = None
        self.is_partitioned = False
        self.is_cloud_node = False
        self.device = get_device()
        self.model_path = ''

        if "is_partitioned" in kwargs:
            self.is_partitioned = kwargs["is_partitioned"]
        if "is_cloud_node" in kwargs:
            self.is_cloud_node = kwargs["is_cloud_node"]

        if self.is_cloud_node:
            self.model = CIFAR100Net("resnet110")
            self.model2 = resnet110_p2()
        else:
            if self.is_partitioned:
                self.model = resnet110_p1()
                self.model2 = resnet110_p1_head()
            else:
                self.model = CIFAR100Net("resnet20")

        if "model_path" in kwargs:
            self.model_path = kwargs["model_path"]

    def load(self, model_url=""):
        if self.model_path:
            model_url = self.model_path
        url_list = model_url.split(";", 1)
        checkpoint = torch.load(url_list[0], map_location=get_device())
        LOG.info(f"Load pytorch checkpoint {url_list[0]} finsihed!")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        LOG.info("Load pytorch state dict finished!")

        if self.is_cloud_node or self.is_partitioned:
            checkpoint = torch.load(url_list[1], map_location=get_device())
            LOG.info(f"Load pytorch checkpoint {url_list[1]} finsihed!")
            self.model2.load_state_dict(checkpoint['model_state_dict'])
            LOG.info("Load pytorch state dict finished!")

        LOG.info("Server model on device: %s", get_device())
        self.model = self.model.to(get_device())
        self.model.eval()
        if self.model2 is not None:
            self.model2 = self.model2.to(get_device())
            self.model2.eval()

    def predict(self, data, **kwargs):
        data = torch.from_numpy(np.asarray(data, dtype=np.float32))
        data = to_device(data, self.device)

        is_partitioned = False
        if "is_partitioned" in kwargs:
            is_partitioned = kwargs["is_partitioned"]
        if self.is_cloud_node:
            if is_partitioned:
                predictions = self.model2(data)
            else:
                predictions = self.model(data)
        else:
            predictions = self.model(data)
            if is_partitioned:
                trans_features = predictions[1]
                predictions = self.model2(predictions[0])

        props = F.softmax(predictions, dim=1)
        props_arr = props.detach().cpu().numpy().flatten().tolist()
        if not self.is_cloud_node and is_partitioned:
            props_arr = (props_arr, trans_features)
        return props_arr
