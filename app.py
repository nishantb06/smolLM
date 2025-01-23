import torch
import gradio as gr
from smollm_training import SmolLMLightning, SmolLMConfig, tokenizer


# Load the model
def load_model():
    config = SmolLMConfig()
    model = SmolLMLightning.load_from_checkpoint(
        "best-checkpoint.ckpt",
        config=config,
        lr=1e-3,  # These parameters don't matter for inference
        warmup_steps=10,
        max_steps=25000,
    )
    model.eval()
    return model


def generate_text(prompt, max_tokens, temperature=0.8, top_k=40):
    """Generate text based on the prompt"""
    try:
        # Encode the prompt
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Move to the same device as model
        prompt_ids = prompt_ids.to(model.device)

        # Generate text
        with torch.no_grad():
            generated_ids = model.model.generate(
                prompt_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        # Decode the generated text
        generated_text = tokenizer.decode(generated_ids[0].tolist())

        return generated_text

    except Exception as e:
        return f"An error occurred: {str(e)}"


# Load the model globally
model = load_model()

# Create the Gradio interface
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(
            label="Enter your prompt", placeholder="Once upon a time...", lines=3
        ),
        gr.Slider(
            minimum=50,
            maximum=500,
            value=100,
            step=10,
            label="Maximum number of tokens",
        ),
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10),
    title="SmolLM Text Generator",
    description="Enter a prompt and the model will generate a continuation.",
    examples=[
        ["Once upon a time", 100],
        ["The future of AI is", 200],
        ["In a galaxy far far away", 150],
    ],
)

if __name__ == "__main__":
    demo.launch()
