"""
Bare-Bones Chat Agent for Llama 3.2-1B-Instruct

This is a minimal chat interface that demonstrates:
1. How to load a model without quantization
2. How chat history is maintained and fed back to the model
3. The difference between plain text history and tokenized input

No classes, no fancy features - just the essentials.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# CONFIGURATION - Change these settings as needed
# ============================================================================

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# System prompt - This sets the chatbot's behavior and personality
# Change this to customize how the chatbot responds
SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise and friendly."

# Context management settings
ENABLE_HISTORY = False              # Toggle conversation history on/off
USE_FIXED_WINDOW = True            # Toggle Fixed Window context management
MAX_MESSAGES = 10                  # Maximum messages to keep in Fixed Window mode

# ============================================================================
# LOAD MODEL (NO QUANTIZATION)
# ============================================================================

# Get Hugging Face token from environment
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    print("✓ Hugging Face token loaded from .env file", flush=True)
else:
    print("⚠ Warning: No HF_TOKEN found in .env file", flush=True)
    print("  Some models may require authentication", flush=True)

print("Loading model (this takes 1-2 minutes)...", flush=True)

# Load tokenizer (converts text to numbers and vice versa)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)

# Load model in half precision (float16) for efficiency
# Use float16 on GPU, or float32 on CPU if needed
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=hf_token,                             # Pass HF token for authentication
    dtype=torch.float16,                        # Use FP16 for efficiency
    device_map="auto",                          # Automatically choose GPU/CPU
    low_cpu_mem_usage=True
)

model.eval()  # Set to evaluation mode (no training)
print(f"✓ Model loaded! Using device: {model.device}", flush=True)
print("✓ Memory usage: ~2.5 GB (FP16)\n", flush=True)

# ============================================================================
# CHAT HISTORY - This is stored as PLAIN TEXT (list of dictionaries)
# ============================================================================
# The chat history is a list of messages in this format:
# [
#   {"role": "system", "content": "You are a helpful assistant"},
#   {"role": "user", "content": "Hello!"},
#   {"role": "assistant", "content": "Hi! How can I help?"},
#   {"role": "user", "content": "What's 2+2?"},
#   {"role": "assistant", "content": "2+2 equals 4."}
# ]
#
# This is PLAIN TEXT - humans can read it
# The model CANNOT use this directly - it needs to be tokenized first

chat_history = []

# Add system prompt to history (this persists across the entire conversation)
chat_history.append({
    "role": "system",
    "content": SYSTEM_PROMPT
})

# ============================================================================
# CONTEXT MANAGEMENT - Fixed Window Approach
# ============================================================================

def apply_fixed_window(history, max_messages):
    """
    Keep only the last N messages, preserving the system prompt.

    Args:
        history: List of chat messages
        max_messages: Maximum number of messages to retain (including system)

    Returns:
        Trimmed history with system prompt + last (max_messages-1) messages
    """
    if len(history) > max_messages:
        # Keep system prompt (index 0) + last (max_messages-1) messages
        return [
            history[0],                      # system prompt
            *history[-(max_messages-1):]     # recent messages
        ]
    return history

# ============================================================================
# CHAT LOOP
# ============================================================================

print("="*70)
print("Chat started! Type 'quit' or 'exit' to end the conversation.")
print(f"History: {'ON' if ENABLE_HISTORY else 'OFF'}", end="")
if ENABLE_HISTORY and USE_FIXED_WINDOW:
    print(f" | Fixed Window: ON ({MAX_MESSAGES} msgs)")
else:
    print()  # New line
print("="*70 + "\n")

while True:
    # ========================================================================
    # STEP 1: Get user input (PLAIN TEXT)
    # ========================================================================
    user_input = input("You: ").strip()
    
    # Check for exit commands
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\nGoodbye!")
        break
    
    # Skip empty inputs
    if not user_input:
        continue
    
    # ========================================================================
    # STEP 2: Add user message to chat history (PLAIN TEXT)
    # ========================================================================
    # The chat history grows with each exchange (if history is enabled)
    # We append the new user message to the existing history
    if ENABLE_HISTORY:
        # Normal mode: append to existing history
        chat_history.append({
            "role": "user",
            "content": user_input
        })
    else:
        # No history mode: reset to only system + current message
        chat_history = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
    
    # At this point, chat_history looks like:
    # [
    #   {"role": "system", "content": "You are helpful..."},
    #   {"role": "user", "content": "Hello!"},
    #   {"role": "assistant", "content": "Hi!"},
    #   {"role": "user", "content": "What's 2+2?"},      ← Just added
    # ]
    # This is still PLAIN TEXT

    # Apply Fixed Window context management if enabled
    if ENABLE_HISTORY and USE_FIXED_WINDOW:
        chat_history = apply_fixed_window(chat_history, MAX_MESSAGES)

    # ========================================================================
    # STEP 3: Convert chat history to model input (TOKENIZATION)
    # ========================================================================
    # The model needs numbers (tokens), not text
    # apply_chat_template() does two things:
    #   1. Formats the chat history with special tokens (like <|start|>, <|end|>)
    #   2. Converts the formatted text into token IDs (numbers)
    
    # First, apply_chat_template formats the history and converts to tokens
    input_ids = tokenizer.apply_chat_template(
        chat_history,                    # Our PLAIN TEXT history
        add_generation_prompt=True,      # Add prompt for assistant's response
        return_tensors="pt"              # Return as PyTorch tensor (numbers)
    ).to(model.device)

    # Create attention mask (1 for all tokens since we have no padding)
    attention_mask = torch.ones_like(input_ids)

    # Now input_ids is TOKENIZED - it's a tensor of numbers like:
    # tensor([[128000, 128006, 9125, 128007, 271, 2675, 527, 264, ...]])
    # These numbers represent our entire conversation history

    # ========================================================================
    # STEP 4: Generate assistant response (MODEL INFERENCE)
    # ========================================================================
    # The model looks at the ENTIRE chat history (in tokenized form)
    # and generates a response

    print("Assistant: ", end="", flush=True)

    with torch.no_grad():  # Don't calculate gradients (we're not training)
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,   # Explicitly pass attention mask
            max_new_tokens=512,              # Maximum length of response
            do_sample=True,                  # Use sampling for variety
            temperature=0.7,                 # Lower = more focused, higher = more random
            top_p=0.9,                       # Nucleus sampling
            pad_token_id=tokenizer.eos_token_id
        )
    
    # outputs contains: [original input tokens + new generated tokens]
    # We only want the NEW tokens (the assistant's response)
    
    # ========================================================================
    # STEP 5: Decode the response (DETOKENIZATION)
    # ========================================================================
    # Extract only the newly generated tokens
    new_tokens = outputs[0][input_ids.shape[1]:]
    
    # Convert tokens (numbers) back to text (PLAIN TEXT)
    assistant_response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True  # Remove special tokens like <|end|>
    )
    
    print(assistant_response)  # Display the response
    
    # ========================================================================
    # STEP 6: Add assistant response to chat history (PLAIN TEXT)
    # ========================================================================
    # This is crucial! We add the assistant's response to the history
    # so the model remembers what it said in future turns
    # (only if history is enabled)

    if ENABLE_HISTORY:
        chat_history.append({
            "role": "assistant",
            "content": assistant_response
        })
    
    # Now chat_history has grown again:
    # [
    #   {"role": "system", "content": "You are helpful..."},
    #   {"role": "user", "content": "Hello!"},
    #   {"role": "assistant", "content": "Hi!"},
    #   {"role": "user", "content": "What's 2+2?"},
    #   {"role": "assistant", "content": "4"}              ← Just added
    # ]
    
    # When the loop repeats:
    # - User enters new message
    # - We add it to chat_history
    # - We tokenize the ENTIRE history (including all previous exchanges)
    # - Model sees everything and generates response
    # - We add response to history
    # - Repeat...
    
    # This is how the chatbot "remembers" the conversation!
    # Each turn, we feed it the ENTIRE conversation history
    
    print()  # Blank line for readability

# ============================================================================
# SUMMARY OF HOW CHAT HISTORY WORKS
# ============================================================================
"""
PLAIN TEXT vs TOKENIZED:

1. PLAIN TEXT (chat_history):
   - Human-readable format
   - List of dictionaries: [{"role": "user", "content": "Hi"}, ...]
   - Stored in memory between turns
   - Gets longer with each message

2. TOKENIZED (input_ids):
   - Numbers (token IDs)
   - Created fresh each turn from chat_history
   - This is what the model actually "reads"
   - Example: [128000, 128006, 9125, 128007, ...]

PROCESS EACH TURN:
   User input (text)
   ↓
   Add to chat_history (text)
   ↓
   Tokenize entire chat_history (text → numbers)
   ↓
   Model generates response (numbers)
   ↓
   Decode response (numbers → text)
   ↓
   Add response to chat_history (text)
   ↓
   Loop back to start

WHY FEED ENTIRE HISTORY?
- The model has no memory between calls
- Each generation is independent
- To "remember" previous turns, we must include them in the input
- This is why context length matters - longer conversations = more tokens

WHAT HAPPENS AS CONVERSATION GROWS?
- chat_history gets longer (more messages)
- Tokenized input gets longer (more tokens)
- Eventually hits model's max context length (for Llama 3.2: 128K tokens)
- Then you need context management (truncation, summarization, etc.)
- But for this simple demo, we let it grow without limit
"""
