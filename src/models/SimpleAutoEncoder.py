import torch


class Encoder(torch.nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class Decoder(torch.nn.Module):

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.linear(x)


class SimpleAutoEncoder(torch.nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, input_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
