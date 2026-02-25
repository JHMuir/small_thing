```
def prepare_training_data(dialogue_list, output_file='character_dialogue.txt', min_dataset_size=500):
    """
    Prepare the dialogue for training with aggressive cleaning.
    """
    cleaned_lines = []
    
    for line in dialogue_list:
        if not line or not isinstance(line, str):
            continue
            
        # Remove only the most problematic characters, keep readable text
        line = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', line)  # Remove control characters
        line = re.sub(r'[^\x20-\x7e]', '', line)  # Keep printable ASCII
        
        # Remove script artifacts
        line = re.sub(r'\[.*?\]', '', line)  # Remove stage directions
        line = re.sub(r'\(.*?\)', '', line)  # Remove parenthetical notes
        
        # Clean up whitespace and punctuation
        line = ' '.join(line.split())  # Normalize whitespace
        line = re.sub(r'([.!?]){3,}', r'\1\1', line)  # Limit repeated punctuation
        
        line = line.strip()
        
        # Keep lines that have reasonable content
        if len(line) < 3:
            continue
            
        # Check if the line has enough actual words
        words = line.split()
        if len(words) < 2:
            continue
            
        cleaned_lines.append(line)
    
    # Data augmentation for small datasets
    if len(cleaned_lines) < min_dataset_size:
        print(f"Augmenting dataset from {len(cleaned_lines)} to ~{min_dataset_size} lines...")
        augmented_lines = cleaned_lines.copy()
        
        # Create variations by combining consecutive lines
        for i in range(len(cleaned_lines) - 1):
            if random.random() < 0.5:  # 50% chance to combine
                combined = f"{cleaned_lines[i]} {cleaned_lines[i+1]}"
                if len(combined) < 200:  # Don't create overly long lines
                    augmented_lines.append(combined)
        
        # Add lines multiple times with slight variations
        while len(augmented_lines) < min_dataset_size and len(cleaned_lines) > 0:
            line = random.choice(cleaned_lines)
            augmented_lines.append(line)
            
        cleaned_lines = augmented_lines[:min_dataset_size]
    
    # Write to file with proper formatting
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            # Add the end-of-text token for GPT-2
            f.write(f"{line}\n")
    
    print(f"Training data saved to {output_file}")
    print(f"Total lines: {len(cleaned_lines)}")
    
    
    return output_file
```
