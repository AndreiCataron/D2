import warnings

import torch

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(net, dataloader, criterion, Ncrop=False):
    net = net.eval()
    loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0

    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        inputs = inputs.view(64, 1, 48, 48)
        #print(inputs.shape)
        outputs = net(inputs.float())
        #print(outputs.shape)

        loss = criterion(outputs, labels)

        # calculate performance metrics
        loss_tr += loss.item()

        _, preds = torch.max(outputs.data, 1)
        correct_count += (preds == labels).sum().item()
        n_samples += labels.size(0)

    acc = 100 * correct_count / n_samples
    loss = loss_tr / n_samples

    return acc, loss