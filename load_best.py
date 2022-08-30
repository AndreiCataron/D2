import sys
import warnings

import torch
import torch.nn as nn
from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix

import model
from fer2013 import my_get_dataloaders
from logger import Logger
from hyperparam import setup_hparams
from checkpoint import restore

warnings.filterwarnings("ignore")
device = torch.device("cpu")


def correct_count(output, target, topk=(1,)):
    """Computes the top k corrrect count for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k)
    return res


def evaluate(net, dataloader, criterion, crops):
    net = net.eval()
    loss_tr, n_samples = 0.0, 0.0

    y_pred = []
    y_gt = []

    correct_count1 = 0
    correct_count2 = 0

    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # fuse crops and batchsize
        if crops:
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)

            # forward
            outputs = net(inputs)

            # combine results across the crops
            labels = torch.repeat_interleave(labels, repeats=ncrops, dim=0)
        else:
            inputs = inputs.view(-1, 1, 48, 48)
            outputs = net(inputs.float())


        loss = criterion(outputs, labels)

        # calculate performance metrics
        loss_tr += loss.item()

        # accuracy
        counts = correct_count(outputs, labels, topk=(1, 2))
        correct_count1 += counts[0].item()
        correct_count2 += counts[1].item()

        _, preds = torch.max(outputs.data, 1)
        preds = preds.to("cpu")
        labels = labels.to("cpu")
        n_samples += labels.size(0)

        y_pred.extend(pred.item() for pred in preds)
        y_gt.extend(y.item() for y in labels)

    acc1 = 100 * correct_count1 / n_samples
    acc2 = 100 * correct_count2 / n_samples
    loss = loss_tr / n_samples
    print("--------------------------------------------------------")
    print("Top 1 Accuracy: %2.6f %%" % acc1)
    print("Top 2 Accuracy: %2.6f %%" % acc2)
    print("Loss: %2.6f" % loss)
    print("Precision: %2.6f" % precision_score(y_gt, y_pred, average='micro'))
    print("Recall: %2.6f" % recall_score(y_gt, y_pred, average='micro'))
    print("F1 Score: %2.6f" % f1_score(y_gt, y_pred, average='micro'))
    print("Confusion Matrix:\n", confusion_matrix(y_gt, y_pred), '\n')


if __name__ == "__main__":
    # Important parameters
    hps = setup_hparams(sys.argv[1:])

    # build network
    logger = Logger()
    net = model.Vgg()
    net = net.to(device)

    restore(net, logger, hps)

    criterion = nn.CrossEntropyLoss()

    # Get data with no augmentation
    trainloader, valloader, testloader = my_get_dataloaders()

    print("Train")
    evaluate(net, trainloader, criterion, True)

    print("Val")
    evaluate(net, valloader, criterion, False)

    print("Test")
    evaluate(net, testloader, criterion, False)