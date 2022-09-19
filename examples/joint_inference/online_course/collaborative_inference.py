# on Kaggle: add CIFAR as inputs, this code is not needed
# offline: download cifar manually
from torchvision.datasets import CIFAR100 as download_helper
download_helper(".", train=True, download=True)
from tqdm import tqdm
#######################################################################################################################

import torchinfo
from torchinfo import summary
#######################################################################################################################
import pickle
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt

from typing import Type, List, Union
from imgaug import augmenters as iaa
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import cifar100_resnets as models


class CIFAR100(Dataset):

    def __init__(self, dataset_path: Path, image_transforms: tt.Compose,
                 image_augmentations: Union[None, Type[iaa.Augmenter]] = None,
                 length : int = 10000):
        super().__init__()
        data = pickle.load(dataset_path.open("rb"), encoding="bytes")
        self.images = data[b"data"][:length]
        self.labels = data[b"fine_labels"][:length]

        self.image_transforms = image_transforms
        self.image_augmentations = image_augmentations

        assert len(self.images) == len(self.labels), "Number of images and labels is not equal!"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple:
        image = self.images[index]
        label = self.labels[index]

        image = np.reshape(image, (3, 32, 32))
        image = np.transpose(image, (1, 2, 0))

        if self.image_augmentations is not None:
            image = self.image_augmentations.augment_image(image)
        image = self.image_transforms(Image.fromarray(image))
        return image, label


image_transformations = tt.Compose([
    tt.ToTensor(),
    tt.Normalize(
        mean=(0.5074, 0.4867, 0.4411),
        std=(0.2011, 0.1987, 0.2025)
    )
])

class CIFAR100Net(nn.Module):

    def __init__(self, model_type: str = "resnet18", temperature: int = 1):
        super().__init__()
        model_class = getattr(models, model_type)
        self.feature_extractor = model_class(num_classes=100)
        self.temperature = temperature

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        activations = self.feature_extractor(images)
        return activations / self.temperature


def accuracy(predictions: torch.Tensor, labels: torch.Tensor, reduce_mean: bool = True) -> torch.Tensor:
    predicted_props = F.softmax(predictions, dim=1)
    predicted_classes = torch.argmax(predicted_props, dim=1)
    correct_predictions = torch.sum(predicted_classes == labels)
    if reduce_mean:
        return correct_predictions / len(labels)
    return correct_predictions


def test_model(network: Type[nn.Module], data_loader: DataLoader) -> (float, float):
    num_correct_predictions = 0
    device = get_device()
    duration = .0

    for images, labels in data_loader:
        images = to_device(images, device)
        labels = to_device(labels, device)

        t_start = time.time()
        predictions = network(images)

        if get_device() == torch.device("cuda"):
            torch.cuda.current_stream().synchronize()
        duration += (time.time()-t_start)*1000.0
        num_correct_predictions += float(accuracy(predictions, labels, reduce_mean=False).item())
    return num_correct_predictions / len(data_loader.dataset), duration / len(data_loader.dataset)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(data: torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)
#######################################################################################################################
teacher_model = CIFAR100Net("resnet110")
student_model = CIFAR100Net("resnet20")

# Check the model summary
print(summary(teacher_model, input_size=(1, 3, 32, 32)))
#######################################################################################################################
print(summary(student_model, input_size=(1, 3, 32, 32)))
#######################################################################################################################
# TODO: on Kaggle: "/kaggle/input/pretrained-models/teacher_resnet110.pt"
#                  "/kaggle/input/pretrained-models/student_resnet20.pt"
teacher_checkpoint = torch.load("./models/teacher_resnet110.pt", map_location=get_device())
student_checkpoint = torch.load("./models/student_resnet20.pt", map_location=get_device())
teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
student_model.load_state_dict(student_checkpoint['model_state_dict'])
teacher_model = teacher_model.to(get_device())
student_model = student_model.to(get_device())

# switching to testing modus
teacher_model.eval()
student_model.eval()

# preparing testing dataset
SAMPLE_NUM = 500 # number of samples for testing
BATCH_SIZE = 1 # we fix the batch size equals 1 in this exercise.
# TODO: on Kaggle: Path("/kaggle/input/cifar100/test")
test_dataset = CIFAR100(Path("./cifar-100-python/test"), image_transformations, length=SAMPLE_NUM)
test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
#######################################################################################################################
def test_model(network: Type[nn.Module], data_loader: DataLoader):
    num_correct_predictions = 0
    device = get_device()
    accumulated_inference_duration = .0

    for images, labels in data_loader:
        images = to_device(images, device)
        labels = to_device(labels, device)
        t_start = time.time()
        predictions = network(images)
        accumulated_inference_duration += (time.time() - t_start) * 1000.0

        num_correct_predictions += float(accuracy(predictions, labels, reduce_mean=False).item())

    return num_correct_predictions / len(data_loader.dataset), accumulated_inference_duration / len(data_loader.dataset)
#######################################################################################################################
print('Predicting ... ')
start = time.time()
acc, batch_time = test_model(student_model, test_data_loader)
print("Student acc: {:.2f}, sample number: {}, avg_batch_time: {:.3f} ms, total time: {:.3f} seconds.".format(acc, SAMPLE_NUM, batch_time, time.time()-start))
#######################################################################################################################
print('Predicting ... ')
start = time.time()
acc, batch_time = test_model(teacher_model, test_data_loader)
print("Teacher acc: {:.2f}, sample number: {}, avg_batch_time: {:.3f} ms, total time: {:.3f} seconds.".format(acc, SAMPLE_NUM, batch_time, time.time()-start))
#######################################################################################################################
# ## First, we configure the local environment:
# !pip install -r /kaggle/input/kubeedge-sedna/lib/requirements.txt
import sys, os
# sys.path.append('/kaggle/input/kubeedge-sedna/lib')
# import nest_asyncio
# nest_asyncio.apply()
#######################################################################################################################
## environment variables
os.environ['MODEL_URL']="./models/student_resnet20.pt"
os.environ['BIG_MODEL_IP']="159.138.44.120"
os.environ['BIG_MODEL_PORT']="30500"

os.environ['HEM_NAME']="CrossEntropy"
os.environ['HEM_PARAMETERS']='[{"key":"threshold_cross_entropy","value":"0.85"}]'

os.environ['SERVICE_NAME']="edgeAI-course-collaborative-inference"
os.environ['NAMESPACE']="default"
os.environ['WORKER_NAME']="edgeworker-bvmtb"
os.environ['LC_SERVER']=""

os.environ['BACKEND_TYPE']="TORCH"
#######################################################################################################################
class Estimator:
    def __init__(self, **kwargs):
        """
        initialize logging configuration
        """
        self.model = None
        self.device = get_device()
        self.is_cloud_node = False
        self.local_loading = True
        if "is_cloud_node" in kwargs:
            self.is_cloud_node = kwargs["is_cloud_node"]
        if self.is_cloud_node:
            self.model = CIFAR100Net("resnet110")
        else:
            self.model = CIFAR100Net("resnet20")

    def load(self, model_url="", model_name=None, **kwargs):
        print(f"Loading {model_url}...")
        checkpoint = torch.load(model_url, map_location=get_device())
        print(f"Load pytorch checkpoint {model_url} finished!")

        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Load pytorch state dict finished!")

        self.model = self.model.to(get_device())
        self.model.eval()

    def predict(self, data, **kwargs):
        data = torch.from_numpy(np.asarray(data, dtype=np.float32))
        image = to_device(data, self.device)
        predictions = self.model(image)
        props = F.softmax(predictions, dim=1)
        props_arr = props.detach().cpu().numpy().flatten().tolist()
        return props_arr
#######################################################################################################################
from sedna.core.joint_inference import JointInference

inference_instance = JointInference(
    estimator=Estimator,
    hard_example_mining={
        "method": "CrossEntropy",
        "param": {
            "threshold_cross_entropy": 0.85
        }
    }
)

# collaborative inference
num_remote_samples = 0
duration = .0
num_correct_predictions = 0

print('Predicting ... ')
for image, label in tqdm(test_data_loader):
    start = time.time()
    is_hard_example, final_result, edge_result, cloud_result = (
        inference_instance.inference(image)
    )
    duration += (time.time() - start) * 1000.0
    if is_hard_example: num_remote_samples += 1
    torch_result = torch.from_numpy(np.asarray(final_result, dtype=np.float32).reshape(1, -1))
    num_correct_predictions += float(accuracy(torch_result, label, reduce_mean=False).item())
    break

print("Cross-model-collaborative acc: {:.2f}, processed sample number (remote teacher/local student): {}/{},  avg_inference_time: {:.3f} ms, total time: {:.3f} seconds.".format(
    num_correct_predictions/SAMPLE_NUM, num_remote_samples, SAMPLE_NUM-num_remote_samples, duration/SAMPLE_NUM, duration/1000.0))

#######################################################################################################################
from cifar100_partition_net import resnet110_p1, resnet110_p2, resnet110_p1_head

net_p1 = resnet110_p1()
net_p1_head = resnet110_p1_head(num_classes=100)
net_p2 = resnet110_p2(num_classes=100)

print(summary(net_p1, input_size=(1, 3, 32, 32)))
#######################################################################################################################
print(summary(net_p1_head, input_size=(1, 64, 8, 8)))
#######################################################################################################################
print(summary(net_p2, input_size=(1, 16, 8, 8)))
#######################################################################################################################
# extended estimator class
class Estimator:

    def __init__(self, **kwargs):
        self.model = None
        self.model2 = None
        self.local_loading = True
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
        if self.model_path: model_url = self.model_path
        url_list = model_url.split(";", 1)
        checkpoint = torch.load(url_list[0], map_location=get_device())
        print(f"Load pytorch checkpoint {url_list[0]} finsihed!")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Load pytorch state dict finished!")

        if self.is_cloud_node or self.is_partitioned:
            checkpoint = torch.load(url_list[1], map_location=get_device())
            print(f"Load pytorch checkpoint {url_list[1]} finsihed!")
            self.model2.load_state_dict(checkpoint['model_state_dict'])
            print("Load pytorch state dict finished!")

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
#######################################################################################################################
from sedna.core.joint_inference import JointInference

# update the model url. It contains the url of two partitioned models separated by ';'.
model_path_n = "./models/resnet110_p1.pt;./models/resnet110_p1_head.pt"
# model_path_n = "/kaggle/input/pretrained-models/resnet110_p1.pt;/kaggle/input/pretrained-models/resnet110_p1_head.pt"

inference_instance = JointInference(
    estimator=Estimator(is_partitioned=True, model_path=model_path_n), # add a new argument 'is_partitioned'
    hard_example_mining={
        "method": "CrossEntropy",
        "param": {
            "threshold_cross_entropy": 0.725
        }
    }
)

# collaborative inference
num_remote_samples = 0
duration = .0
num_correct_predictions = 0

print('predicting ... ')
for image, label in tqdm(test_data_loader):
    start = time.time()
    is_hard_example, final_result, edge_result, cloud_result = (
        inference_instance.inference(image, is_partitioned=True) # Inference using partitioned models
    )
    duration += (time.time() - start) * 1000.0
    if is_hard_example: num_remote_samples += 1
    torch_result = torch.from_numpy(np.asarray(final_result, dtype=np.float32).reshape(1, -1))
    num_correct_predictions += float(accuracy(torch_result, label, reduce_mean=False).item())

print("Partitioned-collaborative acc: {:.2f}, processed sample number (remote partition/local partition): {}/{},  avg_inference_time: {:.3f} ms, total time: {:.3f} seconds.".format(
    num_correct_predictions/SAMPLE_NUM, num_remote_samples, SAMPLE_NUM-num_remote_samples, duration/SAMPLE_NUM, duration/1000.0))

#######################################################################################################################
