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
    """

    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_floor = 1e-4  

        # BC biases (Mamba-3: initialized to ONES for stability)
        self.B_bias = nn.Parameter(torch.ones(d_state))
        self.C_bias = nn.Parameter(torch.ones(d_state))

        self.B_norm = nn.RMSNorm(d_state)
        self.C_norm = nn.RMSNorm(d_state)

        self.rope_freqs = nn.Parameter(torch.randn(d_state // 2) * 0.01)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)                        
        x_path, z = xz.chunk(2, dim=-1)             
        x_path = F.silu(x_path)

        ssm_params = self.x_proj(x_path)
        B_raw = ssm_params[:, :, :self.d_state]
        C_raw = ssm_params[:, :, self.d_state:2*self.d_state]
        dt_input = ssm_params[:, :, -2:-1]           
        lam_input = ssm_params[:, :, -1:]            

        B = self.B_norm(B_raw + self.B_bias)
        C = self.C_norm(C_raw + self.C_bias)
        dt = F.softplus(self.dt_proj(dt_input))      
        lam = torch.sigmoid(lam_input)               

        B, C = self._apply_rope(B, C, dt)

        A = -torch.exp(self.A_log)                   
        A = torch.clamp(A, max=-self.A_floor)        

        y = self._trapezoidal_scan(x_path, dt, A, B, C, lam)
        y = y + x_path * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        
        return self.out_proj(y)

    def _apply_rope(self, B, C, dt):
        freqs = F.softplus(self.rope_freqs)  
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
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        device = x.device

        dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1))  
        Bx = B.unsqueeze(2) * x.unsqueeze(-1)  
        lam_exp = lam.unsqueeze(-1)  

        h = torch.zeros(batch, d_inner, d_state, device=device)
        ys = []

        for t in range(seq_len):
            dA_t = dA[:, t]       
            dt_t = dt[:, t]       
            Bx_t = Bx[:, t]      
            lam_t = lam_exp[:, t] 

            if t == 0:
                h = dA_t * h + dt_t.unsqueeze(-1) * Bx_t
            else:
                Bx_prev = Bx[:, t-1]  
                term_prev = (1 - lam_t) * dt_t.unsqueeze(-1) * dA_t * Bx_prev
                term_curr = lam_t * dt_t.unsqueeze(-1) * Bx_t
                h = dA_t * h + term_prev + term_curr

            C_t = C[:, t]  
            y_t = (h * C_t.unsqueeze(1)).sum(-1)  
            ys.append(y_t)

        return torch.stack(ys, dim=1)  


class HawkesLandmarkAttention(nn.Module):
    """
    Sparse attention from current Mamba-3 state to Hawkes-selected landmark ticks.
    Selects historical ticks that retained high structural energy.
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
    Continuous-time asynchronous perception engine for High-Frequency data.
    Combines Mamba-3 with a Hawkes exponential time-decay filter.
    """

    def __init__(self, d_input, d_model=64, num_layers=4,
                 d_state=16, expand=2, pool_size=10,
                 attn_heads=4, max_landmarks=100):
        super().__init__()
        self.d_model = d_model
        self.pool_size = pool_size
        self.d_inner = d_model * expand

        # Learnable Hawkes Contagion Parameter
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
        self._current_epoch = epoch

    def get_rope_penalty(self, max_freq=5000.0):
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
            x:  [batch, seq_len, d_input] - LOB feature tensor
            dt: [batch, seq_len, 1] - time since last event (e.g., in microseconds)
        Returns:
            state: [batch, d_model] - compressed microstructure state
        """
        # The Hawkes Exponential Filter
        decay_rate = F.softplus(self.hawkes_decay)
        hawkes_weight = torch.exp(-decay_rate * dt)  # [B, L, 1]

        # Apply structural dampening to inputs
        filtered_ticks = x * hawkes_weight

        # Hawkes weights as 1D for landmark selection
        hawkes_weights_1d = hawkes_weight.squeeze(-1)  

        # Skip Connection
        if self._current_epoch < self._skip_reconnect_epoch:
            x_skip = self.skip_proj(filtered_ticks.detach())
        else:
            x_skip = self.skip_proj(filtered_ticks)

        # Mamba-3 Backbone
        emb = self.tick_embedding(filtered_ticks) * math.sqrt(self.d_model)

        for mamba_layer, norm in zip(self.layers, self.norms):
            emb = emb + mamba_layer(norm(emb))

        emb = self.final_norm(emb)  

        # Sparse Landmark Attention
        attn_out = self.landmark_attention(
            mamba_output=emb,
            hawkes_skip=x_skip,
            hawkes_weights=hawkes_weights_1d,
        )  

        # Gated Fusion & State Pooling
        seq_len = emb.size(1)
        actual_pool = min(self.pool_size, seq_len)
        if actual_pool > 1:
            mamba_terminal = emb[:, -actual_pool:, :].mean(dim=1)  
        else:
            mamba_terminal = emb[:, -1, :]

        gate_input = torch.cat([mamba_terminal, attn_out], dim=-1)  
        gate = torch.sigmoid(self.fusion_gate(gate_input))          
        fused = gate * mamba_terminal + (1 - gate) * attn_out       

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

    with torch.no_grad():
        output = model(dummy_x, dummy_dt)

    print(f"Extracted Microstructure State Vector: {output.shape}")
    print(f"Initial Hawkes Decay Rate: {F.softplus(model.hawkes_decay).item():.6f}")
    print("\\n[v3] LOBMamba-Hawkes v3 generic forward pass completed successfully.")