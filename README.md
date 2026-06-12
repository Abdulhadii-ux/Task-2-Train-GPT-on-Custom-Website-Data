# MiniGPT (STEMP21) — GPT-style Language Model from Scratch (PyTorch)

This repository builds a **decoder-only GPT-style language model** from scratch using **PyTorch**, trained on scraped **STEMP21** website text, and runs inference through a clean **Streamlit** web UI.

---

## ✨ What you get

- **Custom GPT-2-like Transformer** (decoder-only)
- **Wikipedia/STEMP21-style pipeline** (here: STEMP21 website scraping → text → tokens → train)
- **Interactive Streamlit app** to generate text from a prompt

> Files included in this repo: a trained model (`minigpt.pt`), scraped dataset (`stemp21_data.txt`), and the training notebook/script.

---

## 📁 Repository structure

```text
week_2task/
  app_stemp21.py          # Streamlit UI for loading model + generating text
  scrape.py               # Scrapes STEMP21 pages into stemp21_data.txt
  stemp21_data.txt       # Raw scraped text dataset
  stemp21_train.ipynb    # Colab training notebook (train and save minigpt.pt)
  minigpt.pt              # Trained model weights
```

---

## 🧠 Model architecture (what’s implemented)

Decoder-only Transformer with:

- **Causal self-attention** (each token attends only to previous tokens)
- **Multi-head attention**
- **Feed-forward MLP**
- **Residual connections + LayerNorm**

**Key hyperparameters (from `app_stemp21.py` / training code):**

- `VOCAB_SIZE`: 50257 (GPT-2 BPE vocab via `tiktoken`)
- `BLOCK_SIZE`: 128 (context window)
- `N_EMBED`: 128
- `N_HEAD`: 4
- `N_LAYER`: 4
- `DROPOUT`: 0.1

---

## 🔄 Data pipeline

1. **Scrape STEMP21 pages**
   - `scrape.py` downloads a list of STEMP21 URLs and saves cleaned text into `stemp21_data.txt`.
2. **Tokenize**
   - Training uses GPT-2 BPE tokenizer (`tiktoken.get_encoding('gpt2')`).
3. **Train**
   - `stemp21_train.ipynb` trains the model and saves weights to `minigpt.pt`.

---

## 🚀 Run the Streamlit app

### 1) Install dependencies

```bash
pip install streamlit torch tiktoken
```

### 2) Start the app

```bash
streamlit run app_stemp21.py
```

### 3) Use it

- Upload `minigpt.pt` (the app also supports uploading a custom `.pt` model)
- Enter a prompt
- Click **Generate Response**

---

## 🎛️ Text generation controls

In the UI you can tune:

- **Tokens to generate**: how many tokens to produce
- **Temperature**: lower = more predictable, higher = more random
- **Top-K**: samples from the top-K most likely next tokens

---

## 📝 Notes

- The model is trained on **scraped website text**, not the full GPT-2/Wikipedia corpus.
- Generation quality depends heavily on dataset size/cleanliness and training steps.

---

## 👤 Author

- **Abdul Hadi**


---

## 📌 Suggested workflow (if you want to retrain)

1. Run `scrape.py` to update `stemp21_data.txt`
2. Open `stemp21_train.ipynb` in Google Colab
3. Train and download/save `minigpt.pt`
4. Run `streamlit run app_stemp21.py`

