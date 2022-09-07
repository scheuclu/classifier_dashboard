import torch.optim as optim
import torch
import torch.nn as nn
from torchmetrics.functional import precision_recall
import evals

def compute_loss(net, inp, labels):
    criterion = nn.CrossEntropyLoss()
    outputs = net(inp)
    loss = criterion(outputs, labels)
    return loss



def train_model(model=None, data=None, optimizer='Adam', lr=0.001, num_epochs=1000):

    # net = Net(layers)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']


    if optimizer=='Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)


    plot_epoch=[]
    plot_loss_train=[]
    plot_loss_test=[]

    precs = []
    recalls = []
    evaluator = evals.Evaluator()

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        # zero the parameter gradients
        plot_epoch.append(epoch)
        model.train()
        optimizer.zero_grad()

        loss_train = compute_loss(model, X_train, y_train.type(torch.LongTensor))
        loss_train.backward()
        optimizer.step()
        plot_loss_train.append(loss_train.item())

        model.eval()
        loss_test = compute_loss(model, X_test, y_test.type(torch.LongTensor))
        plot_loss_test.append(loss_test.item())

        prec, recall = evaluator.prec_recall(model)
        #prec, recall = precision_recall(model(X_test), y_test.type(torch.LongTensor), average='micro', num_classes=2)
        precs.append(prec)
        recalls.append(recall)


    return model, dict(epochs=plot_epoch, train=plot_loss_train, test=plot_loss_test, precision=precs, recall=recalls)

