import torch
import numpy as np
from typing import List, Tuple, Set
import re

# Random Chunk Eviction Algorithm
# This module implements random chunk eviction instead of attention-based redundant token eviction.
# It randomly selects reasoning chunks to remove based on a target reduction percentage.

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

def random_token_eviction(
    reasoning_chain: str,
    attention_weights: torch.Tensor = None,
    tokenizer=None,
    input_ids: torch.Tensor = None,
    trigger_token: str = "</think>",
    target_reduction: float = 0.25,
    random_seed: int = None
) -> Tuple[List[int], List[str]]:
    """
    Implements random chunk eviction: randomly choose chunks to remove from reasoning chain
    (Modified from attention-based eviction to random selection)
    
    Args:
        reasoning_chain: The reasoning chain text
        attention_weights: Attention weights (kept for backward compatibility, not used)
        tokenizer: Tokenizer for decoding tokens
        input_ids: Input token IDs
        trigger_token: Trigger token (default: "</think>")
        target_reduction: Target reduction percentage (e.g., 0.25 for 25% reduction)
        random_seed: Random seed for reproducibility (optional)
    
    Returns:
        Tuple of (steps_to_evict, new_reasoning_chain)
    """
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Step 1: Find trigger token position
    trigger_token_id = tokenizer.encode(trigger_token, add_special_tokens=False)[0]
    trigger_positions = (input_ids[0] == trigger_token_id).nonzero(as_tuple=True)[0]
    
    if len(trigger_positions) == 0:
        raise ValueError(f"Trigger token '{trigger_token}' not found in input")
    
    # Step 2: Segment reasoning chain into steps
    reasoning_steps, reasoning_token_positions = segment_reasoning_chain(reasoning_chain, tokenizer)
    
    if len(reasoning_steps) <= 1:
        # If only one step or no steps, return original reasoning
        return [], reasoning_chain
    
    # Calculate the number of reasoning steps to evict based on percentage

    print(f"len of reasoning steps: {len(reasoning_steps)}")
    print(f"target reduction: {target_reduction}")
    num_steps_to_evict = int(len(reasoning_steps) * target_reduction)
    
    # Ensure we don't evict all steps (keep at least one)
    num_steps_to_evict = min(num_steps_to_evict, len(reasoning_steps) - 1)
    
    print(f"Target reduction: {target_reduction}")
    print(f"Number of reasoning steps: {len(reasoning_steps)}")
    print(f"Steps to evict: {num_steps_to_evict}")
    
    # Step 3: Randomly select steps to evict
    all_step_indices = list(range(len(reasoning_steps)))
    steps_to_evict = np.random.choice(all_step_indices, size=num_steps_to_evict, replace=False).tolist()
    
    # Create new reasoning chain by removing evicted steps
    new_reasoning_steps = []
    for i, step in enumerate(reasoning_steps):
        if i not in steps_to_evict:
            new_reasoning_steps.append(step)
    
    # Create new reasoning chain by removing evicted steps
    new_reasoning_steps = []
    for i, step in enumerate(reasoning_steps):
        if i not in steps_to_evict:
            new_reasoning_steps.append(step)
    
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

def run_eviction_algorithm(attn_store=None, tokenizer=None, input_tokenized=None, target_reduction=0.25, random_seed=None):
    """
    Run the random chunk eviction algorithm on the current example.
    
    Args:
        attn_store: Dict mapping layer index -> attention tensor (kept for backward compatibility, not used)
        tokenizer: Tokenizer for decoding
        input_tokenized: Tokenized input
        target_reduction: Target reduction percentage (e.g., 0.25 for 25% reduction)
        random_seed: Random seed for reproducibility (optional)
    
    Returns:
        Tuple of (steps_to_evict, new_reasoning_chain)
    """
    # Extract reasoning tokens (everything before </think>)
    input_text = tokenizer.decode(input_tokenized['input_ids'][0], skip_special_tokens=True)
    reasoning_chain = input_text.split('</think>')[0] if '</think>' in input_text else input_text
        
    # Run the random chunk eviction algorithm
    steps_to_evict, new_reasoning_chain = random_token_eviction(
        reasoning_chain=reasoning_chain,
        attention_weights=None,  # Not used in random eviction
        tokenizer=tokenizer,
        input_ids=input_tokenized['input_ids'],
        target_reduction=target_reduction,
        random_seed=random_seed
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
    Test the random chunk eviction algorithm with a simple example.
    """
    print("Testing Random Chunk Eviction Algorithm...")
    
    # This would be called from the notebook with actual model output
    # For now, we'll just show the structure
    print("Algorithm implementation complete!")
    print("To use this algorithm:")
    print("1. Import the functions from this file")
    print("2. Call run_eviction_algorithm(attn_store, tokenizer, input_tokenized, target_reduction, random_seed)")
    print("3. The function will return (steps_to_evict, new_reasoning_chain)")
    
    return True 