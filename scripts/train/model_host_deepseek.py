#!/usr/bin/env python3
"""
FastAPI model host for serving the Qwen model and providing model outputs.
This service is designed to work with the redundant token eviction algorithm.
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from typing import Optional, Dict, Any
import uvicorn
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
from verl.workers.rollout.vllm_rollout.redundant_token_eviction import redundant_token_eviction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration

# Initialize FastAPI app
app = FastAPI(
    title="Model Host API",
    description="API for serving the Qwen model with attention outputs",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None

class ModelInput(BaseModel):
    """Input model for text generation requests."""
    text: str
    reduction_score: float

class ModelOutput(BaseModel):
    """Output model for text generation responses."""
    generated_text: str
    input_ids: list
    attention_weights: Optional[list] = None
    model_output: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    model_name: str

@app.on_event("startup")
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((Exception,))
)
async def startup_event():
    """Initialize the model and tokenizer on startup."""
    global model, tokenizer
    
    try:
        logger.info("Loading model and tokenizer...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModel.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
            attn_implementation="eager",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        logger.info(f"Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health information."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device="unknown",
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )

@app.get("/health", response_model=HealthResponse)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((Exception,))
)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device="unknown",
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )



@app.post("/get_reduced_reasoning_chain")
@retry(
    stop=stop_after_attempt(3),                          # Reduced from 5 to 3 attempts
    wait=wait_exponential(multiplier=2, min=2, max=30),  # Wait: 2s, 4s, 8s, 16s, 30s
    retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
)
async def get_reduced_reasoning_chain(request: ModelInput):
    """Get reduced reasoning chain with attention weights for the eviction algorithm."""
    global model, tokenizer

    print("Model output requested")
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize input
        inputs = tokenizer(
            request.text, 
            return_tensors="pt",
        )
        
        # Move to device - get the device from the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        target_layers = [i for i in range(model.config.num_hidden_layers)]

        attn_store = {}
        def make_hook(layer_id):
            def hook(module, input, output):
                # output is (hidden_states, attn_weights) in Qwen3's self_attn
                attn_weights = output[1]  # shape: [batch, heads, seq_len, seq_len]
                # breakpoint()
                attn_store[layer_id] = attn_weights.mean(dim=1).detach().cpu()
            return hook
        
        # Register hook only for the target layer
        handles = [model.layers[i].self_attn.register_forward_hook(make_hook(i)) for i in target_layers]

        with torch.no_grad():
            _ = model(**inputs)

        for h in handles:
            h.remove()
        

        
        ordered_layer_indices = sorted(attn_store.keys())
        attention_weights = torch.stack([attn_store[i] for i in ordered_layer_indices], dim=0)

        if attention_weights.dim() == 4 and attention_weights.size(1) == 1:
            attention_weights = attention_weights.squeeze(1)
        elif attention_weights.dim() == 3:
            # Already [layers, seq_len, seq_len]
            pass
        else:
            raise ValueError(
                f"Unexpected attention tensor shape from attn_store: {attention_weights.shape}. "
                "Expected [layers, 1, seq_len, seq_len] or [layers, seq_len, seq_len]."
            )

        # Create a synthetic head dimension -> [layers, heads=1, seq_len, seq_len]
        attention_weights = attention_weights.unsqueeze(1)

        input_tokenized = tokenizer(request.text, return_tensors="pt")

        # Run the eviction algorithm
        steps_to_evict, new_reasoning_chain = redundant_token_eviction(
            reasoning_chain=request.text,
            attention_weights=attention_weights,
            tokenizer=tokenizer,
            input_ids=input_tokenized['input_ids'],
            target_reduction=request.reduction_score
        )
        
        
        return {'new_reasoning_chain': new_reasoning_chain}
        
    except Exception as e:
        logger.error(f"Model output error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model output: {str(e)}")

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "model_host_deepseek:app",
        host="0.0.0.0",
        port=8008,
        reload=False,
        log_level="info"
    ) 