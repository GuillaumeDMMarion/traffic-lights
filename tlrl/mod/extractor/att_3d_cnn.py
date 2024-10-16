"""
Custom cnn feature extractor, where we put observation types as channels.
We include depth attention for the stacked frames.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DepthAttention(nn.Module):
    """
    Module to compute attention weights over the depth dimension of the stacked tensor.

    Workflow:
    1. Global average pooling across spatial dimensions (height, width).
    2. Further aggregation across channels.
    3. Compute attention weights by passing the aggregated vector through a fully
       connected layer, followed by a sigmoid activation.

    The output of this module is a set of attention weights of shape [batch_size, depth_size].
    """

    def __init__(self, depth_size):
        super(DepthAttention, self).__init__()
        self.attention_fc = nn.Linear(depth_size, depth_size)

    def forward(self, x):
        # Global average pooling across spatial dimensions (height, width)
        attention = torch.mean(x, dim=[3, 4])  # [batch_size, channels, depth]

        # Aggregate across channels too.
        attention = torch.mean(attention, dim=1)  # [batch_size, depth]

        # Linear layer and apply activation.
        attention = self.attention_fc(attention)
        attention = torch.sigmoid(attention)  # Attention weights in range [0, 1]

        # Reshape attention weights and apply to input
        attention = (
            attention.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        )  # Shape: [batch_size, channels, depth, 1, 1]
        weighted_x = x * attention

        return weighted_x


class ConvBlock3D(nn.Module):
    """
    A basic 3D convolution block consisting of a convolutional layer, batch normalization, ReLU activation, and max pooling.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock3D, self).__init__()
        kwargs = {
            "kernel_size": (3, 3, 3),
            "stride": 1,
            "padding": 1,
            **kwargs,
        }  # 3D kernel and padding

        # Conv3d takes (batch_size, channels, depth, height, width)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, **kwargs),  # 3D convolution
            nn.BatchNorm3d(out_channels),  # Batch normalization for 3D
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(
                kernel_size=(1, 2, 2), stride=(1, 2, 2)
            ),  # Pooling over spatial dimensions only (H, W)
        )

    def forward(self, x):
        return self.conv(x)


class TrafficlightFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the traffic light problem, using a self attended convolution.

    The input is a 4D tensor of shape (channels, depth, height, width).
    Channels is the number of frames in the phase sequence.
    Depth is the number of frames in the vehicle sequence
    Height and width are the spatial dimensions of the intersection.
    """

    def __init__(self, observation_space, features_dim):
        super(TrafficlightFeatureExtractor, self).__init__(
            observation_space, features_dim
        )

        # Input shape is (4, 3, 29, 29), so channels=4 and depth=3
        input_channels = observation_space.shape[0]  # Should be 4

        # Initialize the attention
        self.depth_attention = DepthAttention(3)

        # Build the 3D convolutional layers
        self.conv_model = nn.Sequential(
            ConvBlock3D(input_channels, 16),  # (4, 3, 29, 29) -> (16, 3, 14, 14)
            ConvBlock3D(16, 32),  # (16, 3, 14, 14) -> (32, 3, 7, 7)
            ConvBlock3D(32, 64),  # (32, 3, 7, 7) -> (64, 3, 3, 3)
            nn.Dropout(p=0.1),
            nn.Flatten(),  # Flatten the output
        )

        # Calculate the output dimension after the convolutions and pooling
        conv_output_dim = (
            64 * 3 * 3 * 3
        )  # Adjust this if you change pooling or input size

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
            elif isinstance(m, nn.Conv3d):
                init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, observations):
        attended = self.depth_attention(observations)
        features = self.conv_model(attended)
        logits = self.classification_head(features)
        return logits
