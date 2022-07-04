import torch


class Encoder(torch.nn.Module):

    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(torch.nn.ReLU())
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(torch.nn.Module):

    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(torch.nn.ReLU())
        layers = layers[:-1]  # Remove last ReLu
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ExtendedAutoEncoder(torch.nn.Module):

    def __init__(self, layer_sizes):
        super().__init__()
        self.encoder = Encoder(layer_sizes)
        self.decoder = Decoder(list(reversed(layer_sizes)))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
