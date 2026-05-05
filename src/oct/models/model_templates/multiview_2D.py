import torch
import torch.nn as nn
import torchvision


class SeparateChannelsResNet18V2(nn.Module):
    def __init__(self, num_channels=1, num_classes=2, num_views=4, dropout_rate=0.5):
        super(SeparateChannelsResNet18V2, self).__init__()

        resnet = torchvision.models.resnet18(pretrained=False)
        self.shared_bb = nn.Sequential(*list(resnet.children())[4:-1])
        self.num_views = num_views

        if self.num_views == 8:
            raise NotImplementedError("This model only supports 4 views")
        elif self.num_views == 4:
            self.view_specific_layers = []
            for i in range(4):
                self.view_specific_layers.append(
                    nn.Sequential(
                        nn.Conv2d(num_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
                        nn.BatchNorm2d(16),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                    )
                )
            self.view_specific_layers = nn.ModuleList(self.view_specific_layers)
        else:
            raise ValueError("Invalid number of views - must be 4 or 8")

        self.fc = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(resnet.fc.in_features, num_classes))  # Add dropout to reduce overfitting

    def forward(self, x):
        if self.num_views == 4:
            view1, view2, view3, view4 = torch.chunk(x, 4, dim=1)

            features1 = self.view_specific_layers[0](view1)
            features2 = self.view_specific_layers[1](view2)
            features3 = self.view_specific_layers[2](view3)
            features4 = self.view_specific_layers[3](view4)

            combined_features = torch.cat([features1, features2, features3, features4], dim=1)
        else:
            raise NotImplementedError("This model only supports 4 views")

        x = self.shared_bb(combined_features)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SeparateChannelsResNet18(nn.Module):
    def __init__(self, num_channels=1, num_classes=2, num_views=4):
        super(SeparateChannelsResNet18, self).__init__()

        # Load the ResNet model without its final fully connected layer
        resnet = torchvision.models.resnet18(pretrained=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.num_views = num_views

        # Modify the first layer to have three separate layers for each channel
        if self.num_views == 8:
            self.features[0] = nn.ModuleList(
                [
                    nn.Conv2d(num_channels, 8, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.Conv2d(num_channels, 8, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.Conv2d(num_channels, 8, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.Conv2d(num_channels, 8, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.Conv2d(num_channels, 8, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.Conv2d(num_channels, 8, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.Conv2d(num_channels, 8, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.Conv2d(num_channels, 8, kernel_size=7, stride=2, padding=3, bias=False),
                ]
            )
        else:
            self.features[0] = nn.ModuleList(
                [
                    nn.Conv2d(num_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.Conv2d(num_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.Conv2d(num_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.Conv2d(num_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
                ]
            )
        # Add a new fully connected layer for the modified input size
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        # Separate input channels
        if self.num_views == 8:
            channel1, channel2, channel3, channel4, channel5, channel6, channel7, channel8 = torch.chunk(x, 8, dim=1)

            # Process each channel through the modified first layers
            features_channel1 = self.features[0][0](channel1)
            features_channel2 = self.features[0][1](channel2)
            features_channel3 = self.features[0][2](channel3)
            features_channel4 = self.features[0][3](channel4)
            features_channel5 = self.features[0][4](channel5)
            features_channel6 = self.features[0][5](channel6)
            features_channel7 = self.features[0][6](channel7)
            features_channel8 = self.features[0][7](channel8)

            # Concatenate the features
            combined_features = torch.cat(
                [
                    features_channel1,
                    features_channel2,
                    features_channel3,
                    features_channel4,
                    features_channel5,
                    features_channel6,
                    features_channel7,
                    features_channel8,
                ],
                dim=1,
            )
        else:
            channel1, channel2, channel3, channel4 = torch.chunk(x, 4, dim=1)

            # Process each channel through the modified first layers
            features_channel1 = self.features[0][0](channel1)
            features_channel2 = self.features[0][1](channel2)
            features_channel3 = self.features[0][2](channel3)
            features_channel4 = self.features[0][3](channel4)

            # Concatenate the features
            combined_features = torch.cat([features_channel1, features_channel2, features_channel3, features_channel4], dim=1)

        # Continue with the rest of the ResNet model
        x = self.features[1:](combined_features)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
