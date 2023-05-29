from torchsummary import summary
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):

    def __init__(self,cfg, input_size, num_classes, dimension=128):
        super(LSTM, self).__init__()
        self.dimension = cfg["weight"]
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=dimension,
                            num_layers=cfg["num_layers"],
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2*dimension, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y,_ = self.lstm(x)
        y = self.fc(y)
        y = self.softmax(y)
        return y


if __name__ == "__main__":
    cfg = {
    "num_layers":2,
    "weight":128
    }

    model = LSTM(cfg=cfg, input_size=(100,32), num_classes=3)
    summary(model, (100,32))