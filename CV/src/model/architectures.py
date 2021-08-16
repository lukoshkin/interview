import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self, in_channels=1, base_width=16):
        super().__init__()
        bw = base_width  # alias
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels, bw, kernel_size=3,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bw),
            nn.LeakyReLU(.2),

            nn.Conv2d(
                bw, bw*2, kernel_size=3,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bw*2),
            nn.LeakyReLU(.2),

            nn.Conv2d(
                bw*2, bw*4, kernel_size=3,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bw*4),
            nn.LeakyReLU(.2),

            nn.Conv2d(
                bw*4, bw*8, kernel_size=3,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bw*8),
            nn.LeakyReLU(.2),

            nn.Conv2d(bw*8, 1, kernel_size=2)
        )

    def forward(self, X):
        return self.main(X).flatten()

    def predict(self, X):
        return self.forward(X).sigmoid()


class Encoder(nn.Module):
    def __init__(self, in_channels=1, base_width=16, code_size=100):
        super().__init__()
        bw = base_width  # alias
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels, bw, kernel_size=3,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bw),
            nn.LeakyReLU(.2),

            nn.Conv2d(
                bw, bw*2, kernel_size=3,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bw*2),
            nn.LeakyReLU(.2),

            nn.Conv2d(
                bw*2, bw*4, kernel_size=3,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bw*4),
            nn.LeakyReLU(.2),

            nn.Conv2d(
                bw*4, bw*8, kernel_size=3,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bw*8),
            nn.LeakyReLU(.2),

            nn.Conv2d(bw*8, code_size, kernel_size=2)
        )

    def forward(self, X):
        return self.main(X).flatten(1)


class Decoder(nn.Module):
    def __init__(self, code_size=100, base_width=16, out_channels=1):
        super().__init__()
        bw = base_width  # alias
        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                code_size, bw*8, kernel_size=2,
                stride=2, bias=False),
            nn.BatchNorm2d(bw*8),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(
                bw*8, bw*4, kernel_size=3,
                stride=2, bias=False),
            nn.BatchNorm2d(bw*4),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(
                bw*4, bw*2, kernel_size=3,
                stride=2, bias=False),
            nn.BatchNorm2d(bw*2),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(
                bw*2, bw, kernel_size=3,
                stride=2, bias=False),
            nn.BatchNorm2d(bw),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(bw, out_channels, kernel_size=2)
        )

    def forward(self, X):
        return self.main(X[..., None, None])


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=1, base_width=16, code_size=100):
        super().__init__()
        self.enc = Encoder(in_channels, base_width, code_size)
        self.dec = Decoder(code_size, base_width, in_channels)

    def forward(self, X):
        return self.dec(self.enc(X))
