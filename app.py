import torch
import gradio as gr
from smollm_training import SmolLMConfig, tokenizer, SmolLM


# Load the model
def load_model():
    config = SmolLMConfig()
    model = SmolLM(config)  # Create base model instead of Lightning model

    # Load just the model weights
    state_dict = torch.load("model_weights.pt", map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    return model


def generate_text(prompt, max_tokens, temperature=0.8, top_k=40):
    """Generate text based on the prompt"""
    try:
        # Encode the prompt
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Move to device if needed
        device = next(model.parameters()).device
        prompt_ids = prompt_ids.to(device)

        # Generate text
        with torch.no_grad():
            generated_ids = model.generate(  # Call generate directly on base model
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
