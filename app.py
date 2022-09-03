import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import sys

import model
from logger import Logger
from hyperparam import setup_hparams
from checkpoint import restore

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_single_image(net, imgpath):
    image = Image.open(imgpath)

    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=(0,), std=(255,))(image)

    image = image.unsqueeze(0)

    net = net.eval()
    output = net(image)

    return output

if __name__ == "__main__":

    hps = setup_hparams(sys.argv[1:])

    logger = Logger()
    net = model.Vgg()
    net = net.to(device)

    restore(net, logger, hps)

    result = run_single_image(net, 'imaginetest.jpg')
    result = F.softmax(result, dim=1)
    _, result = torch.max(result, 1)
    emotion_mapping = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
    result = emotion_mapping[result[0].item()]

    print(result)