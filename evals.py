import dataprep
import torch.nn.functional as F
from torch import argmax
from sklearn.metrics import confusion_matrix
from torchmetrics.functional import precision_recall
import torch


class Evaluator():
    def __init__(self):
        self.data = dataprep.read_data()

    def prec_recall(self, net):
        X_test = self.data['X_test']
        y_test = self.data['y_test']
        prec, recall = precision_recall(
            net(X_test),
            y_test.type(torch.LongTensor),
            average='micro',
            num_classes=2)
        return prec, recall

    def confusion_matrix(self, net):
        X_test = self.data['X_test']
        y_test = self.data['y_test']

        pred = net.forward(X_test)
        pred = F.softmax(pred)
        pred = argmax(pred, dim=1)

        m = confusion_matrix(y_test.detach().numpy(), pred.detach().numpy())
        return m