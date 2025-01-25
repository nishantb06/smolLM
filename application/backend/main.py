from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from smollm_training import SmolLMConfig, tokenizer, SmolLM

app = FastAPI()


# Model loading
def load_model():
    config = SmolLMConfig()
    model = SmolLM(config)

    try:
        state_dict = torch.load("/app/weights/model_weights.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


try:
    model = load_model()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 40


@app.post("/text/generate")
async def generate_text(request: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Encode the prompt
        prompt_ids = tokenizer.encode(request.prompt, return_tensors="pt")

        # Move to device if needed
        device = next(model.parameters()).device
        prompt_ids = prompt_ids.to(device)

        # Generate text
        with torch.no_grad():
            generated_ids = model.generate(
                prompt_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
            )

        # Decode the generated text
        generated_text = tokenizer.decode(generated_ids[0].tolist())

        return {"output_text": generated_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
