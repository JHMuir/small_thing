```
# Display results
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Show the image
    axes[0].imshow(img)
    axes[0].set_title("Input Image", fontsize=12)
    axes[0].axis('off')
    
    # Show predictions as bar chart
    species = [class_names[i][:20] for i in top_indices]  # Truncate long names
    probs = [predictions[0][i] * 100 for i in top_indices]
    colors = ['green' if i == 0 else 'steelblue' for i in range(len(species))]
    
    axes[1].barh(range(len(species)), probs, color=colors)
    axes[1].set_yticks(range(len(species)))
    axes[1].set_yticklabels(species)
    axes[1].set_xlabel('Confidence (%)')
    axes[1].set_title('Top Predictions', fontsize=12)
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()
```


```
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
