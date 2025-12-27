
# Define the train and test datasets
training_dataset = tokenized_datasets['train']
testing_dataset = tokenized_datasets['test']

# Initialize the trainer class
trainer = Trainer(
# Add arguments to the class
    model=model,
    args=training_args,
    train_dataset=training_dataset,
    eval_dataset=testing_dataset
)