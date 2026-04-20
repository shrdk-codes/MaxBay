# MaxBay

> If you try this project and it works (or breaks), please open an Issue with details.

## Quick Start

### Requirements
- Python 3.9+ recommended

### Install
```bash
pip install -r requirements.txt
```

### Run
```bash
python app.py
```

## Troubleshooting / Support
If anything goes wrong:
1. Copy the full error message / stack trace
2. Mention your OS + Python version
3. Describe what you ran and what you expected to happen
4. Open a GitHub Issue

---

# Training Process (Step-by-Step)

This section explains what the training script is doing in simple terms.

## 1) Organizing the Study Materials

**Code:** `load_dataset`, `train_test_split`

**What is this?**  
We load the dataset and split it into two parts:
- **90%** training data (what the model learns from)
- **10%** test data (a “final exam”)

**Goal:**  
Make sure the model learns general patterns instead of memorizing the training set.

---

## 2) Choosing the Student & the Translator

**Code:** `AutoTokenizer`, `AutoModelForCausalLM`

**What is this?**
- **Tokenizer = Translator**: turns text into tokens (numbers)
- **Model = Student**: the base language model you fine-tune (e.g., BLOOMZ)

**Goal:**  
Convert human text into a numeric format the model can learn from.

---

## 3) Setting the Reading Rules

**Code:** `pad_token`, `eos_token`

**What is this?**
- **EOS (End of Sentence)** tells the model when to stop.
- **PAD (Padding)** makes all sequences the same length for batching.

Common setting:
- `tokenizer.pad_token = tokenizer.eos_token`

**Goal:**  
Keep training stable and prevent the model from getting confused by variable-length inputs.

---

## 4) Creating Study Flashcards (Instruction → Response)

**Code:** `tokenize_function`

**What is this?**  
We format each example like:

```text
Instruction: <your instruction>
Response: <your response>
```

**Goal:**  
Teach the model to behave like an assistant that follows instructions.

---

# Line-by-Line Breakdown (Plain English)

| Code | What it’s doing |
|------|------------------|
| `dataset = load_dataset(...)` | Loads your dataset so the model can read it |
| `train_test_split(test_size=0.1)` | Holds out 10% for evaluation |
| `AutoTokenizer.from_pretrained(...)` | Loads the tokenizer (“translator”) |
| `AutoModelForCausalLM.from_pretrained(...)` | Loads the base model (“student”) |
| `tokenizer.pad_token = tokenizer.eos_token` | Uses EOS as the padding token |
| `f"Instruction: {inst}\nResponse: {resp}"` | Formats training examples consistently |
| `padding="max_length", truncation=True` | Pads short inputs and truncates long inputs |
| `result["labels"] = result["input_ids"].copy()` | Trains the model to predict the next tokens of the same text |
| `per_device_train_batch_size=1` | Trains on 1 example per step (low memory) |
| `gradient_accumulation_steps=4` | Accumulates 4 steps before updating weights |
| `num_train_epochs=3` | Runs through the dataset 3 times |
| `trainer.train()` | Starts training |
| `trainer.save_model("fine_tuned_model")` | Saves the final fine-tuned model |

---

## How do we know training worked?

During training you’ll see a metric called **loss**:

- **High loss**: model is still confused / guessing
- **Lower loss**: model is learning patterns and improving

---

##  What do I do with the saved model?

After training finishes, you should see a folder called:

- `fine_tuned_model/`

This contains your fine-tuned model files. You can load that folder in another script to run inference and build a chatbot that responds the way you trained it.

---

## Contribution
Open for contribution 
