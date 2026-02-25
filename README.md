```
def create_dataset(file_path, tokenizer, block_size=128):
    """
    Convert our text file into a format the model can train on.
    
    The TextDataset class handles reading the file, tokenizing it, and breaking
    it into chunks of the right size. The block_size parameter controls how many
    tokens each training example contains - we use 128 because it's long enough
    to capture context but short enough to train quickly.
    """
    dataset = transformers.TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size  # How many tokens per training example
    )
    
    # This data collator handles batching and creating the input/target pairs
    # For language modeling, the target is just the input shifted by one token
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # mlm=False means we're doing causal language modeling (predicting next word)
    )
    
    return dataset, data_collator

def fine_tune_model(model, tokenizer, train_dataset, data_collator, output_dir='character_model', dataset_size=None):
    """
    Fine-tune the model on our character's dialogue.
    """
    if dataset_size and dataset_size < 200:
        num_epochs = 20  # More epochs for very small datasets
        learning_rate = 2e-5  # Lower learning rate
    elif dataset_size and dataset_size < 500:
        num_epochs = 15
        learning_rate = 3e-5
    else:
        num_epochs = 10
        learning_rate = 5e-5
        
    # Training arguments control how the training process works
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,           # Where to save the fine-tuned model
        overwrite_output_dir=True,       # Overwrite if directory exists
        num_train_epochs=num_epochs,     # How many times to go through the data
        per_device_train_batch_size=2,   # How many examples to process at once
        gradient_accumulation_steps=2,   # Accumulate gradients
        save_steps=500,                  # Save a checkpoint every 500 steps
        save_total_limit=2,              # Only keep the 2 most recent checkpoints
        logging_steps=50,                # Print progress every 100 steps
        learning_rate=learning_rate,     # How fast the model learns (smaller = more careful)
        warmup_steps=100,                # Gradually increase learning rate at start
        weight_decay=0.01,               # Add weight decay for regularization
    )
    
    # The Trainer handles all the complex training loop for us
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # This is where the actual training happens
    print(f"\nStarting fine-tuning with {num_epochs} epochs...")
    trainer.train()
        
    # Save the final model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

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

def interactive_generation(model, tokenizer):
    """
    Let the user have a conversation with the character model.
    """
    print("Type a prompt and see what your character says!")
    print("Type 'quit' to exit\n\n")
    
    while True:
        user_input = input("Your prompt: ")
        
        if user_input.lower() == 'quit':
            print("Quitting...\n\n")
            break
        
        print("\nGenerating response...\n")
        response = generate_character_response(model, tokenizer, user_input, max_length=60)
        print(f"Character says: {response}\n")
```
