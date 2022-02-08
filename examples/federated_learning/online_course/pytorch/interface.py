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

from sedna.algorithms.aggregation import FedAvgV2
from sedna.algorithms.client_choose import SimpleClientChoose
from sedna.common.config import Context
from sedna.core.federated_learning import FederatedLearningV2

import cifar100_resnets as models

simple_chooser = SimpleClientChoose(per_round=int(Context.get_parameters('NUM_OF_SELECTED_CLIENTS', 2)))

fedavg = FedAvgV2()

# The function `get_transmitter_from_config()` returns an object instance.
transmitter = FederatedLearningV2.get_transmitter_from_config()


class Estimator:
    def __init__(self) -> None:
        self.model = self.build()
        self.pretrained = None
        self.saved = None
        self.hyperparameters = {
            "type": "basic",
            "rounds": 10,
            "target_accuracy": 0.5,
            "epochs": 10,
            "batch_size": 128,
            "optimizer": "SGD",
            "learning_rate": 0.1,
            "lr_schedule": "StepLR",
            "model_name": "cifar100_resnet",
            "momentum": 0.9,
            "weight_decay": 1e-4
        }

    @staticmethod
    def build():
        return models.resnet20(num_classes=100)
