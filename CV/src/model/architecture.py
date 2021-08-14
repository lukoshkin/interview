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

