import torch
import torch.nn as nn
import torch.nn.functional as F

import dataprep
from optim import train_model


class Net(nn.Module):
    def __init__(self, layer_sizes=[2]):
        super().__init__()
        last = 16
        self.layers = nn.ModuleList([])
        for outsize in layer_sizes:
            self.layers.append(nn.Linear(last,outsize))
            last=outsize
            print("AAA")

    def forward(self, x):

        for layer in self.layers:
           x = layer(x)
           x = torch.sigmoid(x)
        return x



if __name__ == "__main__":
    model = Net(layer_sizes=[2])
    data = dataprep.read_data()
    net, train_logs = train_model(model, data, lr=0.1, num_epochs=1000)

    pred = net.forward(data['X_test'])
    pred = F.softmax(pred)
    pred = torch.argmax(pred, dim=1)
    from sklearn.metrics import confusion_matrix
    m = confusion_matrix(data["y_test"].detach().numpy(), pred.detach().numpy())