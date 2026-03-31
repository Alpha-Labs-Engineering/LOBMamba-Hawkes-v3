import torch
from lob_mamba_v3 import LOBMambaV3

def main():
    print("Initializing LOBMamba-Hawkes v3 Environment...")
    
    # Define our abstract dimensions
    batch_size = 2
    seq_len = 500       # 500 ticks per sequence
    d_input = 10        # e.g., Price, Size, Side, and 7 LOB event flags
    d_model = 64        # The internal hidden dimension of the Mamba state
    
    # 1. Generate Dummy Feature Data
    # This represents the physical state of the order book
    x = torch.randn(batch_size, seq_len, d_input)
    
    # 2. Generate Dummy Time Deltas
    # This represents the time elapsed since the *previous* tick in microseconds.
    # Notice how we aren't using a fixed interval.
    dt_micros = torch.rand(batch_size, seq_len, 1) * 25000.0 
    
    print(f"Generated Input Tensor: {x.shape} (Batch, Seq_Len, Features)")
    print(f"Generated Time Tensor: {dt_micros.shape} (Batch, Seq_Len, dt_micros)")
    
    # Initialize the perception engine
    model = LOBMambaV3(
        d_input=d_input,
        d_model=d_model,
        num_layers=4,
        d_state=16,
        expand=2,
        pool_size=10,       # Averages the last 10 states to reduce terminal noise
        attn_heads=4,
        max_landmarks=50,   # Attention mechanism looks back at the 50 most energetic ticks
    )
    
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")
    print("Running forward pass...")
    
    # Run the continuous-time forward pass
    # No gradients needed for a simple inference test
    with torch.no_grad():
        state_vector = model(x, dt_micros)
        
    print(f"\nSuccess! Extracted Microstructure State Vector: {state_vector.shape}")
    print("This [Batch, d_model] vector is ready to be fed to an RL Actor-Critic or classifier.")

if __name__ == "__main__":
    main()