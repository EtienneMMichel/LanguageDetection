from torchsummary import summary
import torch.nn as nn

class FeedForwardModel(nn.Module):

    def __init__(self, cfg, input_size, num_classes):
        super(FeedForwardModel, self).__init__()
        self.input_size = input_size
        self.cfg = cfg
        layers = [
            # nn.Flatten(start_dim=1),
            # nn.Linear(reduce(operator.mul, input_size, 1), cfg["weight"]),
            nn.Linear(input_size, cfg["weight"]),

            nn.ReLU(inplace=True),
            ]
        for _ in range(cfg["num_layers"]):
            layers.append(nn.Linear(cfg["weight"],cfg["weight"]))
            layers.append(nn.Dropout(p=.4))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(cfg["weight"], num_classes))          
        self.classifier = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.classifier(x)
        y = self.softmax(y)
        return y


if __name__ == "__main__":
    cfg = {
    "num_layers":2,
    "weight":128
    }

    model = FeedForwardModel(cfg=cfg, input_size=(100,32), num_classes=3)
    summary(model, (100,32))