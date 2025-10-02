import torch
import numpy as np
from typing import List, Tuple, Set
import re

def calculate_uniformity_score(step_importance: dict) -> float:
    """
    Calculate how uniform the attention scores are across reasoning steps.
    Returns a value between 0 and 1, where 1 means very uniform (don't evict).
    
    Args:
        step_importance: Dictionary mapping step index to importance score
    
    Returns:
        Uniformity score (0-1)
    """
    # Use entropy-based uniformity for stability.
    # Maps perfectly uniform distributions to 1.0 and highly peaked ones to ~0.0.
    if len(step_importance) <= 1:
        return 1.0

    values = np.array(list(step_importance.values()), dtype=float)
    # Clamp negatives to zero (should not happen, but for safety)
    values = np.clip(values, a_min=0.0, a_max=None)

    total = float(values.sum())
    if total <= 0.0:
        return 1.0

    p = values / total
    # Numerical stability for log(0)
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    max_entropy = float(np.log(len(values)))
    if max_entropy == 0.0:
        return 1.0

    return float(entropy / max_entropy)

def determine_eviction_percentage(uniformity_score: float, target_reduction: float = 0.25) -> float:
    """
    Determine what percentage of reasoning steps to evict based on uniformity.
    
    Args:
        uniformity_score: How uniform the attention scores are (0-1)
        target_reduction: Target reduction percentage of reasoning steps (e.g., 0.25 for 25% reduction)
    
    Returns:
        Percentage of steps to evict (0-1)
    """
    # If very uniform, don't evict anything
    if uniformity_score > 0.8:
        return 0.0
    
    # If not uniform, evict based on target reduction
    # Scale the reduction based on uniformity
    # Less uniform = more eviction
    eviction_percentage = target_reduction * (1 - uniformity_score)
    
    return min(eviction_percentage, 0.8)  # Cap at 80% to avoid evicting too much

def redundant_token_eviction(
    reasoning_chain: str,
    attention_weights: torch.Tensor,
    tokenizer,
    input_ids: torch.Tensor,
    trigger_token: str = "</think>",
    target_reduction: float = 0.25
) -> Tuple[List[int], List[str]]:
    """
    Implements Algorithm 1: Redundant Token Eviction via Self-summarization with dynamic budget
    
    Args:
        reasoning_tokens: List of reasoning tokens T = {t1, ..., tL}
        attention_weights: Attention weights from model output [layers, heads, seq_len, seq_len]
        tokenizer: Tokenizer for decoding tokens
        input_ids: Input token IDs
        trigger_token: Trigger token (default: "</think>")
        target_reduction: Target reduction percentage (e.g., 0.25 for 25% reduction)
    
    Returns:
        Tuple of (steps_to_evict, new_reasoning_chain)
    """
    
    # Step 1: Find trigger token position
    trigger_token_id = tokenizer.encode(trigger_token, add_special_tokens=False)[0]
    trigger_positions = (input_ids[0] == trigger_token_id).nonzero(as_tuple=True)[0]
    
    if len(trigger_positions) == 0:
        raise ValueError(f"Trigger token '{trigger_token}' not found in input")
    
    # Use the last occurrence of the trigger token
    trigger_pos = trigger_positions[-1].item()
    
    # Step 2: Get attention weights from trigger token to all tokens
    num_layers, num_heads, seq_len, _ = attention_weights.shape
    
    # Initialize attention scores for each token at each layer and head
    attention_scores = torch.zeros(num_layers, num_heads, seq_len)
    
    # Step 3-6: Calculate attention scores from trigger token to each token
    for layer in range(num_layers):
        for head in range(num_heads):
            for token_pos in range(seq_len):
                # α_</think>→t^(l,h) - attention weight from trigger token to token t
                attention_scores[layer, head, token_pos] = attention_weights[layer, head, trigger_pos, token_pos]
    
    # Step 7: Segment reasoning chain into steps
    reasoning_steps, reasoning_token_positions = segment_reasoning_chain(reasoning_chain, tokenizer)

    
    # Step 8-10: Calculate step importance scores
    step_importance = calculate_step_importance(reasoning_steps, reasoning_token_positions, attention_scores, input_ids)
    
    # Calculate uniformity score
    uniformity_score = calculate_uniformity_score(step_importance)

    if uniformity_score != uniformity_score:  # check for NaN
        # If uniformity_score is NaN, return original reasoning (no eviction)
        return [], reasoning_chain
    
    # Determine eviction percentage based on uniformity
    eviction_percentage = determine_eviction_percentage(uniformity_score, target_reduction)
    
    print(f"Eviction percentage: {eviction_percentage}")
    print(f"Uniformity score: {uniformity_score}")
    print(f"len of reasoning steps: {len(reasoning_steps)}")
    print(f"First reasoning step: {reasoning_steps[0]}")
    
    # Calculate the number of reasoning steps to evict based on percentage
    num_steps_to_evict = int(len(reasoning_steps) * eviction_percentage)
    
    # Step 11: Sort steps by importance (ascending order - least important first)
    sorted_steps = sorted(step_importance.items(), key=lambda x: x[1])
    
    # Step 12-22: Evict reasoning steps based on importance until we reach the target step reduction
    steps_to_evict = []
    
    for step_idx, step_importance_score in sorted_steps:
        if len(steps_to_evict) >= num_steps_to_evict:
            break
        
        # Add this step to eviction list
        steps_to_evict.append(step_idx)
    
    # Create new reasoning chain by removing evicted steps
    new_reasoning_steps = []
    for i, step in enumerate(reasoning_steps):
        if i not in steps_to_evict:
            new_reasoning_steps.append(step)

    print(f"Length of new reasoning steps: {len(new_reasoning_steps)}")
    
    # Reconstruct the reasoning chain by concatenating all remaining tokens
    # This maintains the original tokenization structure
    all_tokens = []
    for step in new_reasoning_steps:
        all_tokens.extend(step)
    
    # Convert the complete token list back to text using the tokenizer
    new_reasoning_chain = tokenizer.convert_tokens_to_string(all_tokens)
    
    return steps_to_evict, new_reasoning_chain

def segment_reasoning_chain(reasoning_chain: str, tokenizer) -> Tuple[List[List[str]], List[List[int]]]:
    """
    Segment reasoning chain into distinct steps using token-based approach.
    
    Returns:
        Tuple of (reasoning_steps, reasoning_token_positions)
        - reasoning_steps: List of reasoning steps, each step is a list of tokens
        - reasoning_token_positions: List of token positions for each step
    """
    # Define split tokens that indicate new reasoning steps
    split_tokens = [
        "Wait", "Alternatively", "Another angle", "Another approach", "But wait", "Hold on", "Hmm", "Maybe",
        "Looking back", "Okay", "Let me", "First", "Then", "Alright", "Compute", "Correct", "Good", "Got it",
        "I don't see any errors", "I think", "Let me double-check", "Let's see", "Now", "Remember", "Seems solid",
        "Similarly", "So", "Starting", "That's correct", "That seems right", "Therefore", "Thus"
    ]
    
    # Get the original tokens
    original_tokenized_chain = tokenizer.tokenize(reasoning_chain)

    reasoning_chain_number = []
    for i, token in enumerate(original_tokenized_chain):
        reasoning_chain_number.append([token, i])

    
    # Get text splits using split_by_tokens
    reasoning_steps_segments = split_by_tokens(reasoning_chain_number, split_tokens)
    
    # Convert the segments back to token lists, preserving the original token order
    reasoning_steps = []
    reasoning_token_positions = []
    for segment in reasoning_steps_segments:
        # Extract just the tokens from the [token, position] pairs
        step_tokens = [token for token, _ in segment]
        reasoning_steps.append(step_tokens)
        
        # Extract the positions from the [token, position] pairs
        positions = [pos for _, pos in segment]
        reasoning_token_positions.append(positions)
    
    return reasoning_steps, reasoning_token_positions

def split_by_tokens(reasoning_chain: List[List], split_tokens: List[str]) -> List[List[List]]:
    """
    Split reasoning chain by tokens.
    
    Args:
        reasoning_chain: List of [token, position] pairs
        split_tokens: List of tokens to split on
    
    Returns:
        List of reasoning chain segments, each segment is a list of [token, position] pairs
    """
    # Sort tokens by length (longest first) to avoid substring confusion
    split_tokens = sorted(split_tokens, key=len, reverse=True)
    
    # Convert the 2D array to a string for regex matching
    chain_text = ''.join([token for token, _ in reasoning_chain])
    
    # Create a regex pattern to match any token, with word boundaries if applicable
    token_pattern = r'(' + '|'.join([re.escape(token) for token in split_tokens]) + r')'
    
    # Find all the split positions
    matches = list(re.finditer(token_pattern, chain_text))
    if not matches:
        return [reasoning_chain]  # No tokens present

    steps = []
    prev_end = 0
    
    for match in matches:
        start = match.start()
        # Gather the section before this token (if any)
        if start > prev_end:
            # Find the corresponding token positions in the 2D array
            section_tokens = []
            current_pos = 0
            for token, pos in reasoning_chain:
                if current_pos >= prev_end and current_pos < start:
                    section_tokens.append([token, pos])
                current_pos += len(token)
            if section_tokens:
                steps.append(section_tokens)
        
        # Find the token and everything after it until the next token
        # We'll handle this in the next iteration
        prev_end = start

    # After the last match, add the last section
    if matches:
        last_start = matches[-1].start()
        last_section_tokens = []
        current_pos = 0
        for token, pos in reasoning_chain:
            if current_pos >= last_start:
                last_section_tokens.append([token, pos])
            current_pos += len(token)
        if last_section_tokens:
            steps.append(last_section_tokens)

    return steps if steps else [reasoning_chain]

def calculate_step_importance(
    reasoning_steps: List[List[str]], 
    reasoning_token_positions: List[List[int]],
    attention_scores: torch.Tensor,
    input_ids: torch.Tensor
) -> dict:
    """
    Calculate importance score for each reasoning step.
    
    Simple implementation that:
    1. For each step, use the provided token positions
    2. Sum the attention scores for those positions
    3. Average across layers and heads
    
    Args:
        reasoning_steps: List of reasoning steps
        reasoning_token_positions: List of token positions for each step
        attention_scores: Attention scores [layers, heads, seq_len]
        input_ids: Input token IDs
    
    Returns:
        Dictionary mapping step index to importance score
    """
    num_layers, num_heads, seq_len = attention_scores.shape
    step_importance = {}
    
    # Calculate step importance using simple approach
    for step_idx, step_tokens in enumerate(reasoning_steps):
        total_attention = 0.0
        
        token_positions = reasoning_token_positions[step_idx]
        
        # Calculate attention for unique positions only
        for pos in token_positions:
            # Sum attention across all layers and heads for this token
            token_attention = attention_scores[:, :, pos].sum().item()
            total_attention += token_attention
        
        # Average the attention scores
        if token_positions:
            step_importance[step_idx] = total_attention / (num_layers * num_heads * len(token_positions))
        else:
            step_importance[step_idx] = 0.0
    
    return step_importance

def run_eviction_algorithm(attn_store, tokenizer, input_tokenized, target_reduction=0.25):
    """
    Run the redundant token eviction algorithm on the current example.
    
    Args:
        attn_store: Dict mapping layer index -> attention tensor of shape [batch, seq_len, seq_len]
        tokenizer: Tokenizer for decoding
        input_tokenized: Tokenized input
        target_reduction: Target reduction percentage (e.g., 0.25 for 25% reduction)
    
    Returns:
        Tuple of (steps_to_evict, new_reasoning_chain)
    """
    # Build attention weights from the provided attn_store
    # Expected by downstream: [layers, heads, seq_len, seq_len]
    # We only have averaged-over-heads attention per layer, so we create a single synthetic head.
    if not attn_store:
        raise ValueError("attn_store is empty; cannot run eviction algorithm without attention tensors")

    # Ensure layers are ordered by layer index
    ordered_layer_indices = sorted(attn_store.keys())
    per_layer_attn = [attn_store[layer_idx] for layer_idx in ordered_layer_indices]

    # Stack to [layers, batch, seq_len, seq_len]
    attention_weights = torch.stack(per_layer_attn, dim=0)

    # Remove batch dimension if present and size 1 -> [layers, seq_len, seq_len]
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
    
    # Extract reasoning tokens (everything before </think>)
    input_text = tokenizer.decode(input_tokenized['input_ids'][0], skip_special_tokens=True)
    reasoning_chain = input_text.split('</think>')[0] if '</think>' in input_text else input_text
        
    # Run the eviction algorithm
    steps_to_evict, new_reasoning_chain = redundant_token_eviction(
        reasoning_chain=reasoning_chain,
        attention_weights=attention_weights,
        tokenizer=tokenizer,
        input_ids=input_tokenized['input_ids'],
        target_reduction=target_reduction
    )
    
    print(f"Reasoning steps to evict: {sorted(steps_to_evict)}")
    print(f"Number of reasoning steps to evict: {len(steps_to_evict)}")
    
    # Calculate tokens evicted and show which reasoning steps are being evicted
    reasoning_steps, _ = segment_reasoning_chain(reasoning_chain, tokenizer)
    tokens_evicted = sum(len(reasoning_steps[i]) for i in steps_to_evict)
    total_tokens = len(tokenizer.tokenize(reasoning_chain))
    
    # Show which reasoning steps are being evicted
    for step_idx in sorted(steps_to_evict):
        if step_idx < len(reasoning_steps):
            # Convert tokens back to text properly
            step_text = tokenizer.convert_tokens_to_string(reasoning_steps[step_idx])
            # # Clean up any remaining special characters
            # step_text = step_text.replace("Ġ", " ").replace("Ċ", "\n").replace("ÄĬ", "").strip()
            step_tokens = len(reasoning_steps[step_idx])
            print(f"Step {step_idx} ({step_tokens} tokens): {step_text[:100]}...")
    
    print(f"\nNew reasoning chain length: {len(tokenizer.tokenize(new_reasoning_chain))} tokens")
    print(f"Original reasoning chain length: {total_tokens} tokens")
    print(f"Tokens evicted: {tokens_evicted} tokens")
    print(f"Step reduction: {(len(steps_to_evict) / len(reasoning_steps) * 100):.1f}% of reasoning steps")
    print(f"Token reduction: {(tokens_evicted / total_tokens * 100):.1f}% of original length")
    
    return steps_to_evict, new_reasoning_chain

# Test function to demonstrate the algorithm
def test_algorithm():
    """
    Test the redundant token eviction algorithm with a simple example.
    """
    print("Testing Redundant Token Eviction Algorithm...")
    
    # This would be called from the notebook with actual model output
    # For now, we'll just show the structure
    print("Algorithm implementation complete!")
    print("To use this algorithm:")
    print("1. Import the functions from this file")
    print("2. Call run_eviction_algorithm(attn_store, tokenizer, input_tokenized, target_reduction)")
    print("3. The function will return (steps_to_evict, new_reasoning_chain)")
    
    return True 