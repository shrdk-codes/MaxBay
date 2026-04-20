# SOMEONE PLZZ TRY THE CODE IF IT WORKS TELL ME

# How to run
pip install requirements.text

# Run
python app.py

# PLZZ MAKE ISSUE IF ANYTHING GOES WRONG




The Training Process: Step-by-Step

1. Organizing the Study Materials

The Code: load_dataset & train_test_split

What is this? We take our big pile of data and split it into two folders.

The Goal: We give the AI 90% of the material to study. We keep 10% hidden. After the AI finishes studying, we use that secret 10% to give it a "final exam" to make sure it actually learned and didn't just memorize the answers.

2. Choosing the Student & the Translator

The Code: AutoTokenizer & AutoModel

What is this? We pick our "student" (the AI brain) and a "translator."

The Goal: AI brains don't read English; they read numbers. The Translator (Tokenizer) turns your words into code. It’s like translating a book into Morse code so a machine can process the signals.

3. Setting the Reading Rules

The Code: pad_token & eos_token

What is this? We define where a sentence starts and where it ends.

The Goal: Imagine reading a book with no punctuation. You wouldn't know where one thought ends and the next begins.

The "Stop" Sign: We add "End of Sentence" (EOS) markers so the AI knows when to stop talking.

The "Filler": We use "Padding" (PAD) to make all practice sentences the same length so the AI doesn't get confused by different page sizes.

4. Creating Study Flashcards

The Code: tokenize_function

What is this? We turn your data into a clear "Instruction & Response" format.

The Goal: This teaches the AI that its job is to be an assistant that follows orders. Every time it sees an instruction, it prepares a helpful answer.

🔍 Line-by-Line Breakdown (For Humans)

Here is exactly what those technical lines are doing in plain English:

The Code Line

What it's actually doing...

dataset = load_dataset(...)

Opening the book: Grabbing your data file so the AI can start reading.

train_test_split(test_size=0.1)

Saving some for the test: Keeping 10% of the pages aside for the final exam.

AutoTokenizer.from_pretrained(...)

Hiring a Translator: Loading the tool that turns words into numbers.

AutoModelForCausalLM.from_pretrained(...)

Choosing the Student: Picking the base AI brain (BLOOMZ) we want to train.

tokenizer.pad_token = tokenizer.eos_token

Making a "Stop" sign: Telling the AI that the sign for "End of Sentence" is also the sign for "Empty Space."

f"Instruction: {inst}\nResponse: {resp}"

Writing the Flashcard: Putting your raw data into a clear Question/Answer format.

padding="max_length", truncation=True

Trimming the pages: Cutting off sentences that are too long and adding "filler" to sentences that are too short so they are all the same size.

result["labels"] = result["input_ids"].copy()

Checking the Answer Key: Telling the AI that the words it's reading are the exact words it should learn to predict.

per_device_train_batch_size=1

One page at a time: Telling the AI not to try and read 100 pages at once so it doesn't get overwhelmed.

gradient_accumulation_steps=4

Taking Notes: The AI looks at 4 pages individually but only updates its "brain" after finishing all 4. This saves computer memory.

num_train_epochs=3

Triple-checking: Telling the AI to read the entire textbook 3 times from start to finish.

trainer.train()

Opening the classroom doors: This is the "Go" button that starts the actual learning process.

trainer.save_model("fine_tuned_model")

Graduation: Taking the finished "Smarter Brain" and putting it in a safe folder for you to use.

📈 How do we know it worked?

During the training, the computer will print out a number called Loss.

High Loss: The AI is still confused and guessing randomly (like a student failing a quiz).

Low Loss: The AI is starting to "get it" and its answers are becoming almost perfect.

🚀 What do I do with the finished folder?

Once the script finishes, you will see a new folder called fine_tuned_model.

The Brain: Inside is the AI's new "knowledge."

The Usage: You can load this folder into another script to build a chatbot that answers exactly how you taught it to in your data.

The Result: You now have an AI that is no longer just "general," but is a specialist in your data!





