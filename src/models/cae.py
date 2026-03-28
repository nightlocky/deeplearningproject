import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for anomaly detection.

    Architecture:
    - Encoder: progressively downsamples input image
    - Decoder: reconstructs image back to original size

    Input:  (B, 3, 224, 224)
    Output: (B, 3, 224, 224)
    """

    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),   # (B,16,112,112)
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (B,32,56,56)
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (B,64,28,28)
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (B,128,14,14)
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B,64,28,28)
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # (B,32,56,56)
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),   # (B,16,112,112)
            nn.ReLU(),

            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # (B,3,224,224)
            nn.Sigmoid()  # ensures output is in [0,1]
        )

    def forward(self, x):
        """
        Forward pass through autoencoder.

        Args:
            x (Tensor): input images

        Returns:
            Tensor: reconstructed images
        """
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed