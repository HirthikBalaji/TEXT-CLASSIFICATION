import dataset_preprocessing
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

# initialize the model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
train_dataset , eval_dataset = dataset_preprocessing.CREATE_DS()
# define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    logging_dir='./logs',
    per_device_train_batch_size=16,
    learning_rate=2e-5,
)

# initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# fine-tune the model
trainer.train()
