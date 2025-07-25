import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# Load model and tokenizer
def model_load():
    base_model_name = "TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0"
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model_path = os.path.join("tinylama-finetuned01-sext", "checkpoint-100")
    model = PeftModel.from_pretrained(base_model, model_path)
    return model, tokenizer

model, tokenizer = model_load()

# Chat memory
chat_history = []

def chat(input_text, mode="Story-wise"):
    global chat_history

    if input_text.strip().lower() == "stop":
        return "Goodbye."

    if mode == "Story-wise" and chat_history:
        input_text = " ".join(chat_history) + " " + input_text

    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=1.2,
        top_p=0.9,
        do_sample=True,
    )
    if mode == "Story-wise" and chat_history:
        history = " ".join(chat_history)
        response = tokenizer.decode(outputs[0][len(history):], skip_special_tokens=True)
        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
        chat_history = [outputs]
        return response
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response
    

with gr.Blocks() as Lena:
    gr.HTML("""
<div id="ageModal" style="position: fixed; z-index: 9999; background: rgba(0, 0, 0, 0.92); top: 0; left: 0; width: 100%; height: 100%; color: #f5f5f5; display: flex; flex-direction: column; align-items: center; justify-content: center; font-family: Arial, sans-serif; padding: 30px; text-align: center;">
    
    <div style="background-color: #1e1e1e; padding: 40px 30px; border-radius: 12px; box-shadow: 0 0 20px rgba(255, 0, 0, 0.4); max-width: 500px;">

        <h1 style="color: #ff4d4d; font-size: 28px; margin-bottom: 15px;">NSFW Content Warning 🔞</h1>

        <p style="font-size: 16px; margin-bottom: 25px;">
            This experience contains adult, erotic, and NSFW content. You must be <strong>18 years or older</strong> to enter.
        </p>

        <p style="font-size: 14px; color: #ccc; margin-bottom: 30px;">
            By clicking "Yes", you confirm you are of legal age and agree to our 
            <a href="#" style="color: #ff9999; text-decoration: underline;">Terms of Service</a> and 
            <a href="#" style="color: #ff9999; text-decoration: underline;">Privacy Policy</a>. Content may include fictional erotic material not suitable for minors.
        </p>

        <div style="display: flex; justify-content: center; gap: 20px;">
            <button onclick="document.getElementById('ageModal').style.display='none'" style="padding: 12px 28px; background-color: #28a745; color: white; border: none; border-radius: 6px; font-size: 16px; cursor: pointer;">
                Yes, I'm 18+
            </button>
            <button onclick="window.location.href='https://google.com'" style="padding: 12px 28px; background-color: #dc3545; color: white; border: none; border-radius: 6px; font-size: 16px; cursor: pointer;">
                No, Take Me Back
            </button>
        </div>

    </div>
</div>
""")

    gr.Markdown("# Lena Sexter AI")
    mode = gr.Radio(["Random", "Story-wise"], label="Chat Mode", value="Random")
    input_box = gr.Textbox(label="Enter your message")
    output_box = gr.Textbox(label="Lena is Responding")
    submit_btn = gr.Button("Send")

    def respond_fn(input_text, mode):
        return chat(input_text, mode)

    submit_btn.click(respond_fn, [input_box, mode], output_box)

Lena.launch()
