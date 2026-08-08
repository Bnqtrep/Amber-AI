import gradio as gr
import subprocess
import os
import torch
import json
import re
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from train_shakespeare import TinyGPT, CharVocab

# 全局状态
train_process = None
loss_history = []  # [(step, loss), ...]

def parse_loss(line):
    """从日志行提取 step 和 avg_loss"""
    match = re.search(r"step\s+(\d+)\s+\|\s+avg_loss\s+([\d.]+)", line)
    if match:
        return int(match.group(1)), float(match.group(2))
    return None

def draw_loss_chart():
    """用 matplotlib 画折线图，返回 PIL Image"""
    if not loss_history:
        return None

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    steps = [p[0] for p in loss_history]
    losses = [p[1] for p in loss_history]

    ax.plot(steps, losses, marker='o', markersize=3, linewidth=1.5, color="#000000")
    ax.fill_between(steps, losses, alpha=0.15, color="#000000")

    ax.set_title('Training Loss Curve (Real-time)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Avg Loss')
    ax.grid(True, linestyle='--', alpha=0.4)

    # 显示最新数值
    if len(losses) > 0:
        ax.annotate(f'{losses[-1]:.4f}', 
                    xy=(steps[-1], losses[-1]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9, color="#000000", fontweight='bold')

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def check_model_status():
    has_model = os.path.exists('model.pt')
    has_vocab = os.path.exists('vocab.json')
    if has_model and has_vocab:
        return "Model ready"
    elif has_model:
        return "model.pt found but vocab.json missing"
    else:
        return "No model found. Please train first."

def run_training(epochs, batch_size, seq_len, lr, n_embd, n_layer, n_head, dropout, input_file):
    global train_process, loss_history
    loss_history = []  # 重置

    if not os.path.exists(input_file):
        yield f"Error: Input file '{input_file}' not found.", None, "Error"
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
    chart = None

    for line in train_process.stdout:
        log_output += line
        parsed = parse_loss(line)
        if parsed:
            loss_history.append(parsed)
            chart = draw_loss_chart()
        yield log_output, chart, "Training in progress..."

    train_process.stdout.close()
    train_process.wait()

    status = "Training Complete!" if train_process.returncode == 0 else "Training Stopped"
    yield log_output, chart, status

def stop_training():
    global train_process
    if train_process is not None and train_process.poll() is None:
        train_process.terminate()
        return "Training stopped by user.", None
    return "No training process running.", None

def run_generate(prompt, temperature, length):
    if not os.path.exists('model.pt') or not os.path.exists('vocab.json'):
        return "Error: model.pt or vocab.json not found. Please train first."

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('vocab.json', 'r', encoding='utf-8') as f:
        stoi = json.load(f)
    vocab = CharVocab(vocab=stoi)

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
with gr.Blocks(
    title="Shakespeare AI",
    theme=gr.themes.Soft(),
    css="""
    :root {
        --color-accent: #000000 !important;
        --color-accent-soft: #000000 !important;
        --primary-50: #000000 !important;
        --primary-100: #000000 !important;
        --primary-200: #000000 !important;
        --primary-300: #000000 !important;
        --primary-400: #000000 !important;
        --primary-500: #000000 !important;
        --primary-600: #000000 !important;
        --primary-700: #000000 !important;
        --primary-800: #000000 !important;
        --primary-900: #000000 !important;
    }

    .gradio-container, .gradio-container * {
        color: black !important;
    }

    button, .gr-button, .gradio-container button {
        background-color: #7f7f7f !important;
        border-color: #666666 !important;
        color: white !important;
    }
    """
) as demo:
    gr.Markdown("""
    # Amber - AI
    Will upgrade the dataset later
    """)

    with gr.Tab("Train"):
        gr.Markdown("### Configure parameters and start training")

        with gr.Row():
            # 左侧：参数
            with gr.Column(scale=1):
                epochs = gr.Number(value=3, label="Epochs", precision=0)
                batch_size = gr.Number(value=64, label="Batch Size", precision=0)
                seq_len = gr.Number(value=128, label="Sequence Length", precision=0)
                lr = gr.Number(value=0.0003, label="Learning Rate")
                n_embd = gr.Number(value=256, label="Embedding Dim", precision=0)
                n_layer = gr.Number(value=4, label="Transformer Layers", precision=0)
                n_head = gr.Number(value=8, label="Attention Heads", precision=0)
                dropout = gr.Number(value=0.1, label="Dropout")
                input_file = gr.Text(value="input.txt", label="Input Text File")

                with gr.Row():
                    train_btn = gr.Button("▶ Start", variant="primary")
                    stop_btn = gr.Button("⏹ Stop", variant="stop")

                status_text = gr.Text(value=check_model_status(), label="Model Status", interactive=False)

            # 中间：日志
            with gr.Column(scale=1):
                log_output = gr.Textbox(label="Training Logs", lines=28, interactive=False)

            # 右侧：实时图表
            with gr.Column(scale=1):
                gr.Markdown("#### Loss Function graph")
                chart_image = gr.Image(label=None, interactive=False, height=400)

        train_btn.click(
            fn=run_training,
            inputs=[epochs, batch_size, seq_len, lr, n_embd, n_layer, n_head, dropout, input_file],
            outputs=[log_output, chart_image, status_text]
        )
        stop_btn.click(fn=stop_training, outputs=[status_text, chart_image])

    with gr.Tab("Generate"):
        gr.Markdown("### Generate Shakespeare-style text")

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Text(value="To be", label="Prompt / Seed Text")
                temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
                length = gr.Slider(50, 2000, value=300, step=50, label="Generate Length (chars)")
                gen_btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column(scale=2):
                output_text = gr.Textbox(label="Generated Text", lines=20, interactive=False)

        gen_btn.click(
            fn=run_generate,
            inputs=[prompt, temperature, length],
            outputs=output_text
        )

    

if __name__ == '__main__':
    demo.launch()
