"""
Custom cnn feature extractor, where we put observation types as channels.
"""

import torch.nn as nn
import torch.nn.init as init
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ConvBlock2D(nn.Module):
    """
    A basic 2D convolution block consisting of a convolutional layer, batch normalization, ReLU activation, and max pooling.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock2D, self).__init__()
        kwargs = {
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            **kwargs,
        }  # 2D kernel and padding

        # Conv2d takes (batch_size, channels, height, width)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),  # 2D convolution
            nn.BatchNorm2d(out_channels),  # Batch normalization for 2D
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2, stride=2
            ),  # Pooling over spatial dimensions (H, W)
        )

    def forward(self, x):
        return self.conv(x)


class TrafficlightFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the traffic light problem.

    The input is a 3D tensor of shape (channels, height, width).
    Channels correspond to the number of frames in the phase sequence.
    Height and width are the spatial dimensions of the intersection.
    """

    def __init__(self, observation_space, features_dim):
        super(TrafficlightFeatureExtractor, self).__init__(
            observation_space, features_dim
        )

        # Input shape is (4, 29, 29), so channels=4
        input_channels = observation_space.shape[0]  # Should be 4

        # Build the 2D convolutional layers
        self.conv_model = nn.Sequential(
            ConvBlock2D(input_channels, 16),  # (4, 29, 29) -> (16, 14, 14)
            ConvBlock2D(16, 32),  # (16, 14, 14) -> (32, 7, 7)
            ConvBlock2D(32, 64),  # (32, 7, 7) -> (64, 3, 3)
            nn.Dropout(p=0.1),
            nn.Flatten(),  # Flatten the output
        )

        # Calculate the output dimension after the convolutions and pooling
        conv_output_dim = 64 * 3 * 3  # Adjust this if you change pooling or input size

        # Define the classification head
        self.classification_head = nn.Sequential(
            nn.Linear(
                conv_output_dim, features_dim
            ),  # Connect to the feature dimension
            nn.LeakyReLU(),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, observations):
        features = self.conv_model(observations)
        logits = self.classification_head(features)
        return logits
