import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
from sklearn.model_selection import train_test_split

from Dataset import Dataset


def prepare_data(img_folder):
    img_data_array = []
    class_name = []
    emotion_mapping = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral':6}

    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path = os.path.join(img_folder, dir1, file)
            image = Image.open(image_path)
            image = np.asarray(image)
            image = np.reshape(image, (48, 48))
            #print(image_path)
            #print(image)
            img_data_array.append(image)
            class_name.append(emotion_mapping.get(dir1))

    #print(img_data_array[0])

    return img_data_array, class_name

def my_get_dataloaders(path='FER2013dataset', bs = 64, augment = True):
    xtrain, ytrain = prepare_data(os.path.join(path, 'train'))
    #######################
    indexes = list(range(len(xtrain)))
    #np.random.shuffle(indexes)
    indexes = indexes[:1]
    xtrain = [xtrain[i] for i in indexes]
    ytrain = [ytrain[i] for i in indexes]
    #print(ytrain[:100])
    ######################
    xtest, ytest = prepare_data(os.path.join(path, 'test'))
    ######################
    indexes = list(range(len(xtest)))
    #np.random.shuffle(indexes)
    indexes = indexes[:64*4]
    xtest = [xtest[i] for i in indexes]
    ytest = [ytest[i] for i in indexes]
    ######################

    #impart datele de test in date de validare si date de test
    xtest, xval, ytest, yval = train_test_split(xtest, ytest, test_size=0.5, random_state = 30)

    if augment:
        #definesc setul de transformari care se aplica fiecarei imagini atunci cand este apelata __getitem__ din Dataset
        train_transform = transforms.Compose([

            #crop si resize in intervalul 0.8-1.2 la o portiune din imagine
            transforms.RandomApply([transforms.RandomResizedCrop(48, scale=(0.8, 1.2))], p=0.7),
            transforms.RandomHorizontalFlip(),

            #translatarea si rotirea random a imaginii
            transforms.RandomApply([transforms.RandomAffine(10, translate=(0.2, 0.2))], p=0.5),

            #transforms.ToTensor()

            transforms.TenCrop(40),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(0,), std=(255,))(t) for t in tensors])),
            transforms.Lambda(lambda tensors: torch.stack([transforms.RandomErasing(p=0.5)(t) for t in tensors])),
            transforms.Lambda(lambda tensors: torch.stack([transforms.Pad(4)(t) for t in tensors]))
        ])
    else:
        train_transform = None

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train = Dataset(xtrain, ytrain, train_transform)
    #imaginile din seturile de validare si test nu vor fi augmentate
    val = Dataset(xval, yval, test_transform)
    test = Dataset(xtest, ytest, test_transform)

    trainloader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=0)
    valloader = DataLoader(val, batch_size=bs, shuffle=True, num_workers=0)
    testloader = DataLoader(test, batch_size=bs, shuffle=True, num_workers=0)

    return trainloader, valloader, testloader

