import torch
import torch.nn as nn
import math


class TFNPLayer(nn.Module):
    """
    Topological-Fractal Neural Processor (TFNP) Layer.
    Inspired by cosmic geometry: twisted toroidal modulation,
    spiral (phi-scaling), and vibrational activation.
    """

    def __init__(self, in_channels, out_channels, alpha=3.5, phi=1.618, f=1.0):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            alpha (float): Twist factor (phase shearing)
            phi (float): Spiral scaling factor (default: golden ratio)
            f (float): Frequency for vibrational activation
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.alpha = alpha
        self.phi = phi
        self.f = f

    def forward(self, x, t=0.0):
        """
        Forward pass through the TFNP layer.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
            t (float): Time parameter for vibration phase (default: 0.0)

        Returns:
            Tensor: Output tensor of shape (B, out_channels, H, W)
        """
        batch, channels, height, width = x.shape

        # Generate polar grid
        phi_grid = torch.linspace(0, 2 * math.pi, height).unsqueeze(1).repeat(1, width)
        theta_grid = torch.linspace(0, 2 * math.pi, width).unsqueeze(0).repeat(height, 1)

        # Apply twist modulation
        twist = (theta_grid + self.alpha * phi_grid) % (2 * math.pi)
        twist = twist.unsqueeze(0).unsqueeze(0).repeat(batch, channels, 1, 1)
        x_twisted = x * torch.cos(twist)

        # Spiral scale (φ)
        x_scaled = x_twisted * self.phi

        # Vibrational Tesla/Merkaba activation
        sin_term = torch.sin(torch.tensor(2 * math.pi * self.f * t))
        x_activated = sin_term * self.conv(x_scaled)

        return x_activated


# Example usage for debugging
if __name__ == "__main__":
    model = TFNPLayer(in_channels=3, out_channels=64)
    input_tensor = torch.rand(1, 3, 32, 32)
    output_tensor = model(input_tensor, t=1.0)
    print("Output shape:", output_tensor.shape)  # Expected: [1, 64, 32, 32]

