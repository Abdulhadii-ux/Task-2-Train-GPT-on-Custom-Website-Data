import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import time
import html

st.set_page_config(
    page_title="STEMP21 AI Assistant",
    page_icon="🔬",
    layout="centered"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
  }

  .stApp {
      background: #f0f6ff;
      color: #1a2a4a;
  }

  .top-bar {
      background: #0a3d7c;
      padding: 0.75rem 2rem;
      display: flex;
      align-items: center;
      gap: 1rem;
      border-radius: 12px;
      margin-bottom: 1.5rem;
  }

  .top-bar-logo {
      font-size: 1.5rem;
      font-weight: 700;
      color: #ffffff;
      letter-spacing: -0.5px;
  }

  .top-bar-logo span {
      color: #5bb8ff;
  }

  .top-bar-tagline {
      color: #90bde0;
      font-size: 0.82rem;
      font-weight: 300;
  }

  .hero-card {
      background: linear-gradient(135deg, #0a3d7c 0%, #1565c0 60%, #1e88e5 100%);
      border-radius: 16px;
      padding: 2rem 2.5rem;
      margin-bottom: 1.5rem;
      color: white;
  }

  .hero-card h2 {
      font-size: 1.6rem;
      font-weight: 700;
      margin: 0 0 0.4rem 0;
      color: #ffffff;
  }

  .hero-card p {
      font-size: 0.9rem;
      color: #b3d4f5;
      margin: 0;
      font-weight: 300;
  }

  .stat-row {
      display: flex;
      justify-content: flex-start;
      gap: 1rem;
      margin-top: 1.25rem;
      flex-wrap: wrap;
  }

  .stat-pill {
      background: rgba(255,255,255,0.15);
      border: 1px solid rgba(255,255,255,0.25);
      border-radius: 20px;
      padding: 0.35rem 1rem;
      font-size: 0.78rem;
      color: #e8f4ff;
      font-weight: 500;
  }

  .stat-pill span {
      color: #5bb8ff;
      font-weight: 700;
  }

  .card {
      background: #ffffff;
      border: 1px solid #d0e4f7;
      border-radius: 14px;
      padding: 1.5rem;
      margin-bottom: 1.25rem;
      box-shadow: 0 2px 8px rgba(10,61,124,0.06);
  }

  .card-title {
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 2px;
      color: #1565c0;
      font-weight: 600;
      margin-bottom: 1rem;
  }

  .output-box {
      background: #f5faff;
      border: 1px solid #bbdaf7;
      border-left: 4px solid #1565c0;
      border-radius: 10px;
      padding: 1.25rem 1.5rem;
      font-family: 'Inter', sans-serif;
      font-size: 0.9rem;
      line-height: 1.8;
      color: #1a2a4a;
      min-height: 120px;
      white-space: pre-wrap;
      word-break: break-word;
  }

  .prompt-highlight {
      color: #0a3d7c;
      font-weight: 600;
  }

  div[data-testid="stTextArea"] textarea {
      background: #f5faff !important;
      border: 1.5px solid #90bde0 !important;
      color: #1a2a4a !important;
      font-family: 'Inter', sans-serif !important;
      font-size: 0.9rem !important;
      border-radius: 10px !important;
  }

  div[data-testid="stTextArea"] textarea:focus {
      border-color: #1565c0 !important;
      box-shadow: 0 0 0 3px rgba(21,101,192,0.12) !important;
  }

  .stButton > button {
      background: #1565c0 !important;
      color: white !important;
      border: none !important;
      border-radius: 10px !important;
      font-weight: 600 !important;
      font-family: 'Inter', sans-serif !important;
      font-size: 0.92rem !important;
      padding: 0.65rem 2rem !important;
      width: 100% !important;
      transition: background 0.2s ease !important;
  }

  .stButton > button:hover {
      background: #0a3d7c !important;
  }

  .stButton > button:disabled {
      background: #90bde0 !important;
      cursor: not-allowed !important;
  }

  .stSlider > div > div > div {
      color: #1565c0 !important;
  }

  div[data-testid="stFileUploader"] {
      background: #f5faff !important;
      border: 2px dashed #90bde0 !important;
      border-radius: 12px !important;
      padding: 1rem !important;
  }

  .error-box {
      background: #fff0f0;
      border: 1px solid #ffb3b3;
      border-left: 4px solid #e53935;
      border-radius: 10px;
      padding: 1rem 1.25rem;
      color: #c62828;
      font-size: 0.88rem;
  }

  .success-badge {
      display: inline-block;
      background: #e3f2fd;
      color: #0a3d7c;
      border: 1px solid #90caf9;
      border-radius: 20px;
      padding: 0.3rem 1rem;
      font-size: 0.8rem;
      font-weight: 600;
  }

  .footer {
      text-align: center;
      color: #90bde0;
      font-size: 0.75rem;
      padding: 1.5rem 0 0.5rem 0;
      border-top: 1px solid #d0e4f7;
      margin-top: 1rem;
  }
</style>
""", unsafe_allow_html=True)


# ─── Hyperparameters ──────────────────────────────────────────────────────────
VOCAB_SIZE = 50257
BLOCK_SIZE = 128
N_EMBED    = 128
N_HEAD     = 4
N_LAYER    = 4
DROPOUT    = 0.1


# ─── Model Architecture ───────────────────────────────────────────────────────
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.dropout = nn.Dropout(DROPOUT)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        head_size = k.shape[-1]
        wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size    = N_EMBED // N_HEAD
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(N_HEAD)])
        self.proj    = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBED, 4 * N_EMBED),
            nn.GELU(),
            nn.Linear(4 * N_EMBED, N_EMBED),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa  = MultiHeadAttention()
        self.ff  = FeedForward()
        self.ln1 = nn.LayerNorm(N_EMBED)
        self.ln2 = nn.LayerNorm(N_EMBED)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.pos_emb   = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks    = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f      = nn.LayerNorm(N_EMBED)
        self.lm_head   = nn.Linear(N_EMBED, VOCAB_SIZE, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss


@st.cache_resource
def load_model(path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = MiniGPT().to(device)
    state  = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device

@st.cache_resource
def load_tokenizer():
    return tiktoken.get_encoding("gpt2")


def generate(model, device, enc, prompt, max_tokens, temperature, top_k):
    input_ids = torch.tensor(
        enc.encode(prompt), dtype=torch.long
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond  = input_ids[:, -BLOCK_SIZE:]
            logits, _ = model(idx_cond)
            logits    = logits[:, -1, :] / temperature
            v, ix     = torch.topk(logits, min(top_k, logits.size(-1)))
            probs     = F.softmax(v, dim=-1)
            next_idx  = torch.multinomial(probs, num_samples=1)
            next_tok  = torch.gather(ix, -1, next_idx)
            input_ids = torch.cat([input_ids, next_tok], dim=1)

    return enc.decode(input_ids[0].tolist())


# ─── UI ───────────────────────────────────────────────────────────────────────

# Top bar
st.markdown("""
<div class="top-bar">
  <div>
    <div class="top-bar-logo">STEMP<span>21</span></div>
    <div class="top-bar-tagline">Science · Technology · Engineering · Mathematics · Pakistan</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Hero card with stats
param_count = sum(p.numel() for p in MiniGPT().parameters()) / 1e6
st.markdown(f"""
<div class="hero-card">
  <h2>🔬 STEMP21 AI Assistant</h2>
  <p>An AI language model trained on STEMP21 content — ask about STEM education, curriculum, and programs in Pakistan.</p>
  <div class="stat-row">
    <div class="stat-pill"><span>{param_count:.1f}M</span> Parameters</div>
    <div class="stat-pill"><span>{N_LAYER}</span> Layers</div>
    <div class="stat-pill"><span>{N_HEAD}</span> Attention Heads</div>
    <div class="stat-pill"><span>{N_EMBED}</span> Embed Dim</div>
  </div>
</div>
""", unsafe_allow_html=True)


# Model upload card
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="card-title">📂 Load Model</p>', unsafe_allow_html=True)

model_ready = False
uploaded = st.file_uploader(
    "Upload your minigpt.pt file",
    type=["pt"],
    label_visibility="collapsed"
)

if uploaded is not None:
    with open("minigpt_uploaded.pt", "wb") as f:
        f.write(uploaded.read())
    try:
        model, device = load_model("minigpt_uploaded.pt")
        enc = load_tokenizer()
        model_ready = True
        st.success(f"✓ Model loaded on {device.upper()}")
    except Exception as e:
        st.markdown(f'<div class="error-box">❌ Failed to load model: {e}</div>', unsafe_allow_html=True)
else:
    st.info("Upload your trained `minigpt.pt` to get started.")

st.markdown('</div>', unsafe_allow_html=True)


# Settings card
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="card-title">⚙️ Generation Settings</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    max_tokens  = st.slider("Tokens to generate", 20, 500, 150, 10)
with col2:
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.05,
                            help="Lower = focused, Higher = creative")
with col3:
    top_k = st.slider("Top-K", 1, 100, 40,
                      help="Sample from the top-K most likely tokens")

st.markdown('</div>', unsafe_allow_html=True)


# Prompt card
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="card-title">💬 Your Prompt</p>', unsafe_allow_html=True)

prompt = st.text_area(
    "Enter your prompt",
    value="STEMP21 is an organization that",
    height=90,
    label_visibility="collapsed"
)

generate_btn = st.button("Generate Response ▶", disabled=not model_ready)
st.markdown('</div>', unsafe_allow_html=True)


# Output
if generate_btn:
    if not model_ready:
        st.warning("⚠️ Please upload your minigpt.pt file first.")
    elif not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Generating..."):
            start = time.time()
            try:
                output         = generate(model, device, enc, prompt, max_tokens, temperature, top_k)
                elapsed        = time.time() - start
                generated_part = output[len(prompt):]

                display_html = (
                    f'<span class="prompt-highlight">{html.escape(prompt)}</span>'
                    f'{html.escape(generated_part)}'
                )

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<p class="card-title">📄 Output</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="output-box">{display_html}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<p style='color:#90bde0;font-size:0.78rem;margin-top:0.5rem;'>"
                    f"Generated {max_tokens} tokens in {elapsed:.2f}s</p>",
                    unsafe_allow_html=True
                )
                st.code(output, language=None)
                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.markdown(
                    f'<div class="error-box">❌ Generation failed: {e}</div>',
                    unsafe_allow_html=True
                )


# Footer
st.markdown("""
<div class="footer">
  STEMP21 · Science Technology Engineering Mathematics Pakistan · AI Assistant
</div>
""", unsafe_allow_html=True)
