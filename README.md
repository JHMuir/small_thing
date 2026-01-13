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
