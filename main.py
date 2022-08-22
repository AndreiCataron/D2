import sys

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fer2013 import my_get_dataloaders
from checkpoint import save
from hyperparam import setup_hparams
from train import train
from evaluate import evaluate
from logger import Logger
import model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run(net, logger, hps):
    #incarc datele
    trainloader, valloader, testloader = my_get_dataloaders(bs=hps['bs'])

    #setez functiile retelei
    net = net.to(device)

    learning_rate = float(hps['lr'])
    scaler = GradScaler()

    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.0001)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5, verbose=True)

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    print("Training", hps['name'], "on", device)
    for epoch in range(hps['start_epoch'], hps['n_epochs']):

        #train
        acc_tr, loss_tr = train(net, trainloader, criterion, optimizer, scaler)
        logger.loss_train.append(loss_tr)
        logger.acc_train.append(acc_tr)

        #evaluate
        acc_v, loss_v = evaluate(net, valloader, criterion)
        logger.loss_val.append(loss_v)
        logger.acc_val.append(acc_v)

        # Update learning rate
        scheduler.step(acc_v)

        if acc_v > best_acc:
            best_acc = acc_v

            #salvez cel mai bun model pana in acest moment
            save(net, logger, hps, epoch + 1)
            logger.save_plt(hps)

        #salvez oricum la o anumita frecventa
        if (epoch + 1) % hps['save_freq'] == 0:
            save(net, logger, hps, epoch + 1)
            logger.save_plt(hps)

        print('Epoch %2d' % (epoch + 1),
              'Train Accuracy: %2.4f %%' % acc_tr,
              'Val Accuracy: %2.4f %%' % acc_v,
              sep='\t\t')

    #calculez acuratetea pe setul de test
    acc_test, loss_test = evaluate(net, testloader, criterion)
    print('Test Accuracy: %2.4f %%' % acc_test,
          'Test Loss: %2.6f' % loss_test,
          sep='\t\t')


if __name__ == '__main__':
    hps = setup_hparams(sys.argv[1:])
    logger = Logger()
    net = model.Vgg()

    run(net, logger, hps)
