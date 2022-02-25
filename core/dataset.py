from __future__ import annotations
from matplotlib.pyplot import annotate
import numpy as np
import scipy.misc
import imageio
import os
from scipy import io
from PIL import Image
from torchvision import transforms
from config import Config as cfg 
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode


class StanfordCar(Dataset):
    def __init__(self, data_path="./datasets/stankford_car", is_train=True):
        super(StanfordCar, self).__init__()
        self.is_train = is_train
        self.annos_path = os.path.join(data_path, "stanford_cars_annos.mat")
        self.imgs_path = os.path.join(data_path, "car_ims")
        self.class_names, self.annotations = self.get_annos_info(is_train)
        print("stanford car dataset category number is:", len(self.class_names))#196
        #load info
        self.images = [imageio.imread(os.path.join(data_path, path)) \
                       for path in self.annotations["images_name"]]
        self.labels = self.annotations["labels"]
        print("stanford car dataset category number is:", len(self.class_names))


    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        img = transforms.Resize((600, 600), InterpolationMode.BILINEAR)(img)
        if self.is_train:
            img = transforms.RandomCrop(cfg.INPUT_SIZE)(img)
            img = transforms.RandomHorizontalFlip()(img)
        else:
            img = transforms.CenterCrop(cfg.INPUT_SIZE)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def get_annos_info(self, is_train=True):
        infos = io.loadmat(self.annos_path)
        class_names = infos['class_names']
        class_names = [str(class_names[0, j][0]).replace(" ", "_").replace("/", "") \
                       for j in range(class_names.shape[1])]
        annotations_infos = infos["annotations"]
        train_annotations = {"images_name":[], "labels":[]}
        test_annotations = {"images_name":[], "labels":[]}
        for i in range(annotations_infos.shape[1]):
            name = str(annotations_infos[0, i][0])[2:-2]
            label = int(annotations_infos[0, i][5])
            is_test = int(annotations_infos[0, i][6])
            if(is_test):
                test_annotations["images_name"].append(name)
                test_annotations["labels"].append(label-1) #label从0开始
            else:
                train_annotations["images_name"].append(name)
                train_annotations["labels"].append(label-1) #label从0开始
        if(is_train):
            return class_names, train_annotations
        else:
            return class_names, test_annotations

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    ### StanfordCar dataset
    stanfordcar = StanfordCar()
    for data in stanfordcar:
        print(data[0].size(), data[1])
