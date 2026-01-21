```
def generate_character_response(model, tokenizer, prompt, max_length=100):
    """
    Generate text in the character's voice.
    """
    # Convert the prompt text to token IDs
    model.eval()
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    # Moving the input for the GPU used during training
    device = next(model.parameters()).device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Generate text
    # We use several parameters to control the generation:
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,          # Maximum total length including prompt
            num_return_sequences=1,          # Generate one response
            no_repeat_ngram_size=3,         # Don't repeat the same 2-word phrases
            do_sample=True,                  # Use sampling instead of always picking most likely
            top_k=50,                        # Only consider the top 50 most likely next tokens
            top_p=0.92,                      # Use nucleus sampling (cumulative probability)
            temperature=0.8,                 # Control randomness (higher = more random)
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
    
    # Convert the tokens back to text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the prompt from the output to show only generated text
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text
```
