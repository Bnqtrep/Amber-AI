import gradio as gr
import subprocess
import os
import torch
import json
from train_shakespeare import TinyGPT, CharVocab

# 全局变量存储训练进程
train_process = None

def check_model_status():
    has_model = os.path.exists('model.pt')
    has_vocab = os.path.exists('vocab.json')
    if has_model and has_vocab:
        return "✅ Model ready (model.pt + vocab.json found)"
    elif has_model:
        return "⚠️ model.pt found but vocab.json missing. Extract it first."
    else:
        return "❌ No model found. Please train first."

def run_training(epochs, batch_size, seq_len, lr, n_embd, n_layer, n_head, dropout, input_file):
    global train_process

    if not os.path.exists(input_file):
        yield f"Error: Input file '{input_file}' not found.", "Error"
        return

    cmd = [
        "python", "train_shakespeare.py",
        "--input", input_file,
        "--epochs", str(int(epochs)),
        "--batch_size", str(int(batch_size)),
        "--seq_len", str(int(seq_len)),
        "--lr", str(float(lr)),
        "--n_embd", str(int(n_embd)),
        "--n_layer", str(int(n_layer)),
        "--n_head", str(int(n_head)),
        "--dropout", str(float(dropout)),
    ]

    train_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    log_output = ""
    for line in train_process.stdout:
        log_output += line
        yield log_output, "🟡 Training in progress..."

    train_process.stdout.close()
    train_process.wait()

    if train_process.returncode == 0:
        yield log_output, "✅ Training Complete!"
    else:
        yield log_output, "⚠️ Training Stopped"

def stop_training():
    global train_process
    if train_process is not None and train_process.poll() is None:
        train_process.terminate()
        return "🛑 Training stopped by user."
    return "No training process running."

def run_generate(prompt, temperature, length):
    if not os.path.exists('model.pt') or not os.path.exists('vocab.json'):
        return "Error: model.pt or vocab.json not found. Please train first."

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load vocab
    with open('vocab.json', 'r', encoding='utf-8') as f:
        stoi = json.load(f)
    vocab = CharVocab(vocab=stoi)

    # Load checkpoint
    ckpt = torch.load('model.pt', map_location=device)
    model_args = ckpt.get('args', {})
    seq_len = model_args.get('seq_len', 128)

    model = TinyGPT(
        vocab_size=vocab.vocab_size,
        seq_len=seq_len,
        n_embd=model_args.get('n_embd', 256),
        n_layer=model_args.get('n_layer', 4),
        n_head=model_args.get('n_head', 8),
        dropout=model_args.get('dropout', 0.1)
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # Generate
    idx = torch.tensor([vocab.encode(prompt[-seq_len:])], dtype=torch.long, device=device)
    generated = prompt

    with torch.no_grad():
        for _ in range(int(length)):
            if idx.size(1) < seq_len:
                pad_len = seq_len - idx.size(1)
                inp = torch.cat([torch.zeros((1, pad_len), dtype=torch.long, device=device), idx], dim=1)
            else:
                inp = idx[:, -seq_len:]
            logits = model(inp)
            logits = logits[:, -1, :] / max(temperature, 0.1)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ch = vocab.decode([int(next_id)])
            generated += ch
            idx = torch.cat([idx, next_id], dim=1)

    return generated

# ==================== Gradio UI ====================
with gr.Blocks(title="Shakespeare AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎭 Shakespeare Transformer GUI
    Train a character-level Transformer on Shakespeare text, then generate new passages.
    """)

    with gr.Tab("🚀 Train"):
        gr.Markdown("### Configure parameters and start training")

        with gr.Row():
            with gr.Column():
                epochs = gr.Number(value=3, label="Epochs")
                batch_size = gr.Number(value=64, label="Batch Size")
                seq_len = gr.Number(value=128, label="Sequence Length")
                lr = gr.Number(value=0.0003, label="Learning Rate")
            with gr.Column():
                n_embd = gr.Number(value=256, label="Embedding Dim")
                n_layer = gr.Number(value=4, label="Transformer Layers")
                n_head = gr.Number(value=8, label="Attention Heads")
                dropout = gr.Number(value=0.1, label="Dropout")

        input_file = gr.Text(value="input.txt", label="Input Text File")

        with gr.Row():
            train_btn = gr.Button("▶ Start Training", variant="primary", size="lg")
            stop_btn = gr.Button("⏹ Stop Training", variant="stop", size="lg")

        status_text = gr.Text(value=check_model_status(), label="Model Status", interactive=False)
        log_output = gr.Textbox(label="Training Logs", lines=25, interactive=False)

        train_btn.click(
            fn=run_training,
            inputs=[epochs, batch_size, seq_len, lr, n_embd, n_layer, n_head, dropout, input_file],
            outputs=[log_output, status_text]
        )
        stop_btn.click(fn=stop_training, outputs=status_text)

    with gr.Tab("✨ Generate"):
        gr.Markdown("### Generate Shakespeare-style text")

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Text(value="To be", label="Prompt / Seed Text")
                temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
                length = gr.Slider(50, 2000, value=300, step=50, label="Generate Length (chars)")
                gen_btn = gr.Button("✨ Generate", variant="primary", size="lg")

            with gr.Column(scale=2):
                output_text = gr.Textbox(label="Generated Text", lines=20, interactive=False)

        gen_btn.click(
            fn=run_generate,
            inputs=[prompt, temperature, length],
            outputs=output_text
        )

    gr.Markdown("---\n💡 **Tip:** Lower temperature (0.2–0.5) = more predictable. Higher (1.0–1.5) = more creative.")

if __name__ == '__main__':
    demo.launch()
