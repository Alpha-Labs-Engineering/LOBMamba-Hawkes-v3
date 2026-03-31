# LOBMamba-Hawkes v3: Continuous-Time Microstructure Perception

**A continuous-time, asynchronous perception engine for high-frequency Limit Order Book (LOB) data, combining Mamba-3 with a Hawkes exponential decay filter.**

## The Problem: The Order Book is Not a Metronome
For the last five years, the quantitative finance community has been trying to forcefully shove high-frequency Level 3 Market-By-Order (MBO) data into Transformer architectures. 

But Transformers are structurally blind to the realities of microstructure. They treat sequential events like a cancelled order, a microsecond pause, an aggressive sweep — as equally spaced tokens. When you use positional encoding to analyze a market that operates in continuous time, you are actively destroying your own alpha. Furthermore, standard Transformers scale quadratically ($O(L^2)$), making microsecond-level sequence lengths computationally unviable.

## The Solution: Mamba + Hawkes
We needed a perception engine that understood the physics of order flow contagion and could scale linearly ($O(L)$). 

**LOBMamba-Hawkes v3** treats the market as it actually exists: a continuous-time point process. 

### Architecture Highlights
1. **The Hawkes Filter:** Before embedding, input features are dampened by a learnable exponential time-decay filter ($\lambda = \exp(-\kappa \cdot dt)$). This allows the network to physically understand the difference between a 1-microsecond pause and a 1-second pause.
2. **Phase A Aligned Mamba-3 Backbone:** Utilizes the 3-term exponential-trapezoidal recurrence of Mamba-3. We implement the critical Phase A alignments (BC biases initialized to ones, $A_{floor}$ clamping) to ensure mathematical stability over massive sequence lengths without the hidden states drifting to zero or blowing up.
3. **Hawkes Landmark Attention:** A sparse attention mechanism that uses the Hawkes decay weights to identify and attend only to historical ticks that retained high structural energy, providing exact historical recall without the $O(L^2)$ penalty.

## Usage

The architecture is built natively in PyTorch and falls back to a pure PyTorch Mamba implementation if the official CUDA kernels are not present in your environment. 

### Basic Implementation

```python
import torch
from lob_mamba import LOBMambaV3

# Define dimensions
batch_size = 32
seq_len = 1000      # Ticks per sequence
d_input = 10        # e.g., Price, Size, and various MBO flags

# x represents your physical LOB features
x = torch.randn(batch_size, seq_len, d_input)

# dt represents the time since the previous tick (e.g., in microseconds)
dt = torch.rand(batch_size, seq_len, 1) * 50000.0 

# Initialize the engine
perception_engine = LOBMambaV3(
    d_input=d_input,
    d_model=64,
    num_layers=4,
    d_state=16
)

# Extract the continuous-time state vector
state = perception_engine(x, dt)
# Output shape: [32, 64] -> Ready for your RL Agent or Classifier



Requirements
torch >= 2.0.0

mamba-ssm (Optional, highly recommended for CUDA acceleration)