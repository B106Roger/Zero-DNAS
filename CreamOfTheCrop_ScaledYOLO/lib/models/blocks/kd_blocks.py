

class FeatureAdaptation(nn.Module):
    def __init__(self):
        super(FeatureAdaptation, self).__init__()
        self.adaptation_layer1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU())
        self.adaptation_layer2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU())
        self.adaptation_layer3 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.ReLU())

    def forward(self, features, layer):
        if layer == 1:
            return self.adaptation_layer1(features)
        elif layer == 2:
            return self.adaptation_layer2(features)
        elif layer == 3:
            return self.adaptation_layer3(features)