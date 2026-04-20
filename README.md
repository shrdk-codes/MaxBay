# SOMEONE PLZZ TRY THE CODE IF IT WORKS TELL ME

# How to run
pip install requirements.text

# Run
python app.py

# PLZZ MAKE ISSUE IF ANYTHING GOES WRONG


# The Process Breakdown
1. Gathering the Textbooks
Python
dataset = load_dataset("json", data_files="dataset.json")
dataset = dataset["train"].train_test_split(test_size=0.1)
What is this? We are opening our data file (the "textbook") and splitting it into two piles.

The Purpose: We keep 90% of the data for the AI to study (Training set) and hide 10% of it (Test set). Later, we use that hidden 10% to "quiz" the AI and see if it actually learned or if it just memorized the answers.

2. Choosing the Student and the Translator
Python
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
What is this? * The Model: This is our "student," a pre-trained AI named BLOOM.

The Tokenizer: This is the "translator."

The Purpose: Computers don’t read words; they read numbers. The Tokenizer turns your human sentences into a list of numbers that the Model can process.

3. Formatting the Lessons
Python
def tokenize_function(examples):
    texts = [f"Instruction: {inst}\nResponse: {resp}" for inst, resp in zip(examples["instruction"], examples["response"])]
    # ... (tokenization logic)
What is this? We are organizing our notes into a consistent "Flashcard" format: Question (Instruction) on one side and Answer (Response) on the other.

The Purpose: Consistency is key. By formatting everything the same way, the AI learns that every time it sees "Instruction," it needs to prepare to give a "Response."

4. Setting the Training Rules
Python
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
)
What is this? These are the "Study Rules."

Batch Size: How many flashcards the AI looks at at once.

Epochs: How many times the AI will read through the entire textbook (in this case, 3 times).

The Purpose: This balances how fast the AI learns versus how much memory your computer uses.

5. Starting the Study Session
Python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)
trainer.train()
What is this? This is the "Tutor" (the Trainer) overseeing the study session.

The Purpose: The Trainer hands the data to the model, checks the errors the model makes, and adjusts the model's "brain" (weights) so it makes fewer mistakes next time.

6. Graduation (Saving the Model)
Python
trainer.save_model("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
What is this? We are saving the AI's new "brain" into a folder.





