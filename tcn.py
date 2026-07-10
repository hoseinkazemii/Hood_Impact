import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    """1D dilated causal convolution block for local temporal patterns.
    Captures local temporal structure with increasing receptive field through dilation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Causal padding: pad only on the left
        self.padding = ((kernel_size - 1) * dilation) // 2
        
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        self.norm = nn.GroupNorm(1, out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time) 
        Returns:
            out: (batch, channels, time)
        """
        # Remove future padding (causal)
        out = self.conv(x)
        
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        return out + self.residual(x)


class TemporalConvStack(nn.Module):
    """Stack of dilated convolutions with exponentially increasing dilation."""
    
    def __init__(
        self,
        channels: int,
        num_conv_layers: int = 10,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        for i in range(num_conv_layers):
            dilation = 2 ** i
            layers.append(TemporalConvBlock(
                channels, channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout
            ))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(
        self,
        x: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Args:
            x: (total_time_points, channels)
            batch_indices: (total_time_points,)
            batch_size: int
            
        Returns:
            out: (total_time_points, channels)
        """
        # Separate by batch, pad to same length, apply convs, then unpad
        max_len = 0
        sequences = []
        lengths = []
        
        for b in range(batch_size):
            mask = batch_indices == b
            seq = x[mask]  # (T_b, channels)
            sequences.append(seq)
            lengths.append(seq.shape[0])
            max_len = max(max_len, seq.shape[0])
        
        # Pad and stack: (batch, channels, max_len)
        padded = torch.zeros(batch_size, x.shape[1], max_len, device=x.device, dtype=x.dtype)
        for b, seq in enumerate(sequences):
            padded[b, :, :lengths[b]] = seq.T
        
        # Apply conv layers
        out = padded
        for layer in self.layers:
            out = layer(out)
        
        # Unpad and flatten back
        result = torch.zeros_like(x)
        for b in range(batch_size):
            mask = batch_indices == b
            result[mask] = out[b, :, :lengths[b]].T
        
        return result