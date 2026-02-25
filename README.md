```
train_dataset, data_collator = create_dataset(training_file, tokenizer, block_size=128)
model, tokenizer = fine_tune_model(model, tokenizer, train_dataset, data_collator, dataset_size=len(dialogue))
interactive_generation(model, tokenizer)
```
