import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Compatibility: nn.RMSNorm was added in PyTorch 2.4
if not hasattr(nn, 'RMSNorm'):
    class RMSNorm(nn.Module):
        def __init__(self, normalized_shape, eps=1e-8):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.eps = eps
        def forward(self, x):
            rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x / rms * self.weight
    nn.RMSNorm = RMSNorm

# 3-Tier CUDA Import Strategy:
#   Tier 1: Official Mamba-3 CUDA (Triton SISO kernel, requires source install)
#   Tier 2: Mamba-2 CUDA kernel (pip install mamba-ssm)
#   Tier 3: Pure PyTorch Mamba3Block fallback (no CUDA required)

# Tier 1: Official Mamba-3
try:
    from mamba_ssm.modules.mamba3 import Mamba3 as CUDAMamba3
    HAS_CUDA_MAMBA3 = True
except (ImportError, Exception):
    HAS_CUDA_MAMBA3 = False

# Tier 2: Mamba-2
try:
    from mamba_ssm import Mamba as CUDAMamba
    HAS_CUDA_MAMBA = True
except (ImportError, Exception):
    HAS_CUDA_MAMBA = False

if HAS_CUDA_MAMBA3:
    print("🚀 Mamba-3 CUDA (Triton SISO) Successfully Imported!")
elif HAS_CUDA_MAMBA:
    print("🚀 Mamba-2 CUDA Imported (Mamba-3 unavailable, using Mamba-2 fallback)")
else:
    print("⚠️ No CUDA Mamba available — using pure PyTorch Mamba3Block")


class Mamba3Block(nn.Module):
    """
    Pure PyTorch Mamba-3 Selective State Space Model block.

    Implements the 3-term exponential-trapezoidal recurrence with complex-valued
    state dynamics via data-dependent RoPE. No short causal convolution — replaced
    by BC biases + the implicit width-2 convolution from the 3-term update.

    Core recurrence:
        h_t = exp(Δ_t * A_t) * h_{t-1}
            + (1 - λ_t) * Δ_t * exp(Δ_t * A_t) * B_{t-1} * x_{t-1}
            + λ_t * Δ_t * B_t * x_t
        y_t = C_t * h_t

    When mamba-ssm is available, LOBMambaV3 auto-switches to CUDA-optimized kernels.
    """

    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Input projection: d_model -> 2 * d_inner (x-path and gate)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Data-dependent SSM parameter projections
        # B, C, dt, lambda all projected from input (no short conv)
        # Layout: [B(N), C(N), dt(1), lambda(1)]
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1 + 1, bias=False)

        # Discretization step-size projection
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # A parameter: log-space negative values for stable dynamics
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_floor = 1e-4  # Prevents A from drifting to zero (infinite state retention)

        # BC biases (Mamba-3: replaces short causal convolution)
        # Initialized to ONES — biggest single ablation component (+0.77 perplexity)
        self.B_bias = nn.Parameter(torch.ones(d_state))
        self.C_bias = nn.Parameter(torch.ones(d_state))

        # RMSNorm for B, C (Mamba-3: mirrors QKNorm in Transformers)
        self.B_norm = nn.RMSNorm(d_state)
        self.C_norm = nn.RMSNorm(d_state)

        # RoPE frequencies for complex-valued state dynamics
        # Enables parity/counting without explicit complex arithmetic
        self.rope_freqs = nn.Parameter(torch.randn(d_state // 2) * 0.01)

        # D parameter: skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        Pure PyTorch Mamba-3 forward pass.
        Uses exp-trapezoidal scan + complex RoPE. No CUDA kernel dependency.
        This block is only instantiated when CUDAMamba is unavailable.
        """
        batch, seq_len, _ = x.shape

        # 1. Input projection and split
        xz = self.in_proj(x)                        # [B, L, 2*d_inner]
        x_path, z = xz.chunk(2, dim=-1)             # Each [B, L, d_inner]
        x_path = F.silu(x_path)

        # 2. Project SSM parameters (selective: data-dependent)
        ssm_params = self.x_proj(x_path)
        B_raw = ssm_params[:, :, :self.d_state]
        C_raw = ssm_params[:, :, self.d_state:2*self.d_state]
        dt_input = ssm_params[:, :, -2:-1]           # [B, L, 1]
        lam_input = ssm_params[:, :, -1:]            # [B, L, 1]

        # 3. Apply BC biases + normalization (Mamba-3: replaces short conv)
        B = self.B_norm(B_raw + self.B_bias)
        C = self.C_norm(C_raw + self.C_bias)

        # 4. Discretization step size
        dt = F.softplus(self.dt_proj(dt_input))      # [B, L, d_inner]

        # 5. Trapezoidal interpolation weight
        lam = torch.sigmoid(lam_input)               # [B, L, 1] in (0, 1)

        # 6. Apply data-dependent RoPE for complex dynamics
        B, C = self._apply_rope(B, C, dt)

        # 7. Continuous A parameter (negative for stability, clamped by A_floor)
        A = -torch.exp(self.A_log)                   # [d_inner, d_state]
        A = torch.clamp(A, max=-self.A_floor)        # Enforce minimum decay magnitude

        # 8. Exponential-trapezoidal selective scan
        y = self._trapezoidal_scan(x_path, dt, A, B, C, lam)

        # 9. Skip connection
        y = y + x_path * self.D.unsqueeze(0).unsqueeze(0)

        # 10. Gate and output projection
        y = y * F.silu(z)
        return self.out_proj(y)

    def _apply_rope(self, B, C, dt):
        """Apply data-dependent RoPE to B and C projections.

        Implements complex-valued SSM dynamics without explicit complex numbers.
        Rotation angles accumulate from dt (step sizes) and learned frequencies,
        enabling stable state-tracking (parity, counting) that real-valued SSMs cannot.
        """
        freqs = F.softplus(self.rope_freqs)  # [N//2], positive
        cumsum_dt = torch.cumsum(dt.mean(-1, keepdim=True), dim=1)  
        angles = cumsum_dt * freqs.unsqueeze(0).unsqueeze(0)       

        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        B_even, B_odd = B[..., ::2], B[..., 1::2]
        min_dim = min(B_even.shape[-1], cos_a.shape[-1])
        B_rot_even = B_even[..., :min_dim] * cos_a[..., :min_dim] - B_odd[..., :min_dim] * sin_a[..., :min_dim]
        B_rot_odd = B_even[..., :min_dim] * sin_a[..., :min_dim] + B_odd[..., :min_dim] * cos_a[..., :min_dim]
        B_rot = torch.stack([B_rot_even, B_rot_odd], dim=-1).flatten(-2)

        C_even, C_odd = C[..., ::2], C[..., 1::2]
        C_rot_even = C_even[..., :min_dim] * cos_a[..., :min_dim] - C_odd[..., :min_dim] * sin_a[..., :min_dim]
        C_rot_odd = C_even[..., :min_dim] * sin_a[..., :min_dim] + C_odd[..., :min_dim] * cos_a[..., :min_dim]
        C_rot = torch.stack([C_rot_even, C_rot_odd], dim=-1).flatten(-2)

        return B_rot, C_rot

    def _trapezoidal_scan(self, x, dt, A, B, C, lam):
        """
        3-term exponential-trapezoidal selective scan.

        h_t = exp(Δ_t * A) * h_{t-1}
            + (1 - λ_t) * Δ_t * exp(Δ_t * A) * B_{t-1} * x_{t-1}
            + λ_t * Δ_t * B_t * x_t

        Sequential implementation for inference correctness. Training should use
        the chunked parallel scan from the official mamba-ssm package when available.
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        device = x.device

        # Discretize: dA = exp(A * dt)
        dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1))  # [B, L, d_inner, N]

        # Pre-compute state-input products: B_t * x_t -> [B, L, d_inner, N]
        Bx = B.unsqueeze(2) * x.unsqueeze(-1)

        # Expand lambda for broadcasting
        lam_exp = lam.unsqueeze(-1)  # [B, L, 1, 1]

        h = torch.zeros(batch, d_inner, d_state, device=device)
        ys = []

        for t in range(seq_len):
            dA_t = dA[:, t]       # [B, d_inner, N]
            dt_t = dt[:, t]       # [B, d_inner]
            Bx_t = Bx[:, t]      # [B, d_inner, N]
            lam_t = lam_exp[:, t] # [B, 1, 1]

            if t == 0:
                # First step: 2-term fallback (no x_{t-1})
                h = dA_t * h + dt_t.unsqueeze(-1) * Bx_t
            else:
                Bx_prev = Bx[:, t-1]  # [B, d_inner, N]
                # 3-term exponential-trapezoidal update
                term_prev = (1 - lam_t) * dt_t.unsqueeze(-1) * dA_t * Bx_prev
                term_curr = lam_t * dt_t.unsqueeze(-1) * Bx_t
                h = dA_t * h + term_prev + term_curr

            # Output: y_t = C_t * h_t
            C_t = C[:, t]  # [B, N]
            y_t = (h * C_t.unsqueeze(1)).sum(-1)  # [B, d_inner]
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # [B, L, d_inner]  


class HawkesLandmarkAttention(nn.Module):
    """
    Sparse attention from current Mamba-3 state to Hawkes-selected landmark ticks.

    The Hawkes filter serves double duty:
      1. Dampens feature dimensions for the Mamba path
      2. Selects which historical ticks deserve attention slots

    High-weight ticks = high contagion events = structurally important = worth remembering.
    """
    def __init__(self, d_model, n_heads=4, max_landmarks=100):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_landmarks = max_landmarks
        self.d_head = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.scale = self.d_head ** -0.5

    def forward(self, mamba_output, hawkes_skip, hawkes_weights):
        """
        Args:
            mamba_output:   (batch, seq_len, d_model) - Mamba-3 backbone output
            hawkes_skip:    (batch, seq_len, d_model) - skip connection from Hawkes filter
            hawkes_weights: (batch, seq_len) - scalar weights from Hawkes filter

        Returns:
            (batch, d_model) - attention-enhanced representation
        """
        B, L, D = hawkes_skip.shape

        k = min(self.max_landmarks, L)
        _, landmark_idx = torch.topk(hawkes_weights, k, dim=1)
        landmark_idx_sorted, _ = torch.sort(landmark_idx, dim=1)

        idx_expanded = landmark_idx_sorted.unsqueeze(-1).expand(-1, -1, D)
        landmark_kv = torch.gather(hawkes_skip, 1, idx_expanded)  

        query = mamba_output[:, -1:, :]  

        Q = self.W_Q(query).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(landmark_kv).view(B, k, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(landmark_kv).view(B, k, self.n_heads, self.d_head).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale  
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = (attn_probs @ V).transpose(1, 2).reshape(B, 1, D)

        return self.W_O(attn_out.squeeze(1))  


class LOBMambaV3(nn.Module):
    """
    Continuous-time asynchronous perception engine for high-frequency data.
    Combines Mamba-3 with a Hawkes exponential time-decay filter.

    Key design choices (Phase A alignment with official Mamba-3):
      - BC biases initialized to ONES (paper's biggest ablation: +0.77 perplexity)
      - 3-tier CUDA: Official Mamba3 -> CUDAMamba (Mamba-2) -> Mamba3Block (PyTorch)
      - Hawkes-gated sparse attention for exact historical recall
      - Gated fusion between Mamba state and attention output
      - Skip connection from Hawkes-filtered features to attention KV

    Args:
        d_input:       Number of input features per tick (excluding time delta)
        d_model:       Hidden dimension for Mamba backbone
        num_layers:    Number of stacked Mamba-3 layers
        d_state:       SSM state dimension
        expand:        Expansion factor for inner dimension (d_inner = d_model * expand)
        pool_size:     Number of terminal states to average for output
        attn_heads:    Number of attention heads for landmark attention
        max_landmarks: Maximum number of landmark ticks for sparse attention
    """

    def __init__(self, d_input, d_model=64, num_layers=4,
                 d_state=16, expand=2, pool_size=10,
                 attn_heads=4, max_landmarks=100):
        super().__init__()
        self.d_model = d_model
        self.pool_size = pool_size
        self.d_inner = d_model * expand

        # --- Learnable Hawkes Contagion Parameter ---
        # CRITICAL INITIALIZATION: Set to -12.0 to prevent gradient underflow on microsecond data.
        # The decay rate kappa is calculated via F.softplus(hawkes_decay) to ensure positivity.
        # If initialized to 0, kappa ≈ 0.693. When multiplied by microsecond time deltas 
        # (e.g., dt=1,000,000 for 1 second), the exponential exp(-kappa * dt) underflows 
        # to exactly 0.0. This kills all gradients and turns the filter into a dead node.
        # Initializing to -12.0 yields kappa ≈ 6.14e-6. This ensures the exponential term 
        # decays gracefully (e.g., retaining ~54% energy over a 100ms gap) rather than collapsing.
        # We figured this out the hard way, we're letting you know so that you don't need to.
        self.hawkes_decay = nn.Parameter(torch.tensor([-12.0]))

        # Tick embeddings
        self.tick_embedding = nn.Linear(d_input, d_model)
        self.skip_proj = nn.Linear(d_input, d_model)

        # Mamba-3 Backbone (3-tier CUDA strategy)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            if HAS_CUDA_MAMBA3:
                layer = CUDAMamba3(
                    d_model=d_model, d_state=d_state, expand=expand, 
                    headdim=min(64, d_model * expand)
                )
            elif HAS_CUDA_MAMBA:
                layer = CUDAMamba(
                    d_model=d_model, d_state=d_state, d_conv=4, expand=expand
                )
            else:
                layer = Mamba3Block(
                    d_model=d_model, d_state=d_state, expand=expand
                )
            self.layers.append(layer)
            self.norms.append(nn.LayerNorm(d_model))

        self.final_norm = nn.LayerNorm(d_model)

        self.landmark_attention = HawkesLandmarkAttention(
            d_model=d_model, n_heads=attn_heads, max_landmarks=max_landmarks
        )

        # Gated Fusion
        self.fusion_gate = nn.Linear(d_model * 2, d_model)
        nn.init.constant_(self.fusion_gate.bias, 2.0)

        # State Compression
        self.state_compression = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

        self._current_epoch = 0
        self._skip_reconnect_epoch = 500  

    def set_epoch(self, epoch):
        """Called by training loop to enable scheduled skip-connection reconnection."""
        self._current_epoch = epoch

    def get_rope_penalty(self, max_freq=5000.0):
        """Soft quadratic penalty on RoPE frequencies exceeding Nyquist limit.
        Add this to the total loss: total_loss += rope_lambda * model.get_rope_penalty()
        """
        penalty = torch.tensor(0.0, device=self.hawkes_decay.device)
        for layer in self.layers:
            if hasattr(layer, 'rope_freqs'):
                freqs = F.softplus(layer.rope_freqs)
                excess = torch.clamp(freqs - max_freq, min=0.0)
                penalty = penalty + excess.pow(2).sum()
        return penalty

    def forward(self, x, dt):
        """
        Encode tick sequence into a compressed state vector.

        Args:
            x:  [batch, seq_len, d_input] - feature tensor
            dt: [batch, seq_len, 1] - time since last event (e.g., in microseconds)
        Returns:
            state: [batch, d_model] - compressed microstructure state
        """
        # Safety clamp: prevent numerical overflow from extreme input values
        x = torch.clamp(x, min=-1e8, max=1e8)

        # --- The Hawkes Exponential Filter ---
        decay_rate = F.softplus(self.hawkes_decay)
        hawkes_weight = torch.exp(-decay_rate * dt)  # [B, L, 1]

        # Apply structural dampening to inputs
        filtered_ticks = x * hawkes_weight

        # Hawkes weights as 1D for landmark selection
        hawkes_weights_1d = hawkes_weight.squeeze(-1)  # [B, L]

        # --- Skip Connection (with scheduled gradient reconnection) ---
        if self._current_epoch < self._skip_reconnect_epoch:
            x_skip = self.skip_proj(filtered_ticks.detach())
        else:
            x_skip = self.skip_proj(filtered_ticks)

        # --- Mamba-3 Backbone ---
        emb = self.tick_embedding(filtered_ticks) * math.sqrt(self.d_model)

        for mamba_layer, norm in zip(self.layers, self.norms):
            emb = emb + mamba_layer(norm(emb))

        emb = self.final_norm(emb)  # [B, L, d_model]

        # --- Sparse Landmark Attention ---
        attn_out = self.landmark_attention(
            mamba_output=emb,
            hawkes_skip=x_skip,
            hawkes_weights=hawkes_weights_1d,
        )  # [B, d_model]

        # --- Gated Fusion ---
        # State Pooling: average last k Mamba states (anti-noise)
        seq_len = emb.size(1)
        actual_pool = min(self.pool_size, seq_len)
        if actual_pool > 1:
            mamba_terminal = emb[:, -actual_pool:, :].mean(dim=1)  # [B, d_model]
        else:
            mamba_terminal = emb[:, -1, :]

        gate_input = torch.cat([mamba_terminal, attn_out], dim=-1)  # [B, 2*d_model]
        gate = torch.sigmoid(self.fusion_gate(gate_input))          # [B, d_model]
        fused = gate * mamba_terminal + (1 - gate) * attn_out       # [B, d_model]

        return self.state_compression(fused)


# --- Local Testing Block ---
if __name__ == "__main__":
    print(f"Mamba-3 CUDA available: {HAS_CUDA_MAMBA3}")
    print(f"Mamba-2 CUDA available: {HAS_CUDA_MAMBA}")

    batch_size = 2
    seq_len = 1000
    d_input = 9

    # Generate dummy features and random asynchronous time deltas
    dummy_x = torch.randn(batch_size, seq_len, d_input)
    dummy_dt = torch.rand(batch_size, seq_len, 1) * 50000.0  

    print(f"\\nInjecting Features: {dummy_x.shape} | Time Deltas: {dummy_dt.shape}")

    model = LOBMambaV3(
        d_input=d_input,
        d_model=64,
        num_layers=4,
        d_state=16,
        expand=2,
        pool_size=10,
        attn_heads=4,
        max_landmarks=100,
    )

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Verify BC bias initialization (should be 1.0, not 0.0)
    for layer in model.layers:
        if hasattr(layer, 'B_bias'):
            print(f"B_bias mean: {layer.B_bias.data.mean().item():.1f} (expected: 1.0)")
            print(f"C_bias mean: {layer.C_bias.data.mean().item():.1f} (expected: 1.0)")
            break

    with torch.no_grad():
        output = model(dummy_x, dummy_dt)

    print(f"Extracted Microstructure State Vector: {output.shape}")
    print(f"State values range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"Initial Hawkes Decay Rate: {F.softplus(model.hawkes_decay).item():.6f}")

    # Verify fusion gate bias
    gate_bias_mean = model.fusion_gate.bias.data.mean().item()
    print(f"Fusion gate bias (mean): {gate_bias_mean:.2f} -> sigmoid = {torch.sigmoid(torch.tensor(gate_bias_mean)).item():.4f}")

    # Verify RoPE penalty
    rope_pen = model.get_rope_penalty()
    print(f"RoPE frequency penalty: {rope_pen.item():.6f}")

    print("\\n[v3] LOBMamba-Hawkes v3 forward pass completed successfully.")