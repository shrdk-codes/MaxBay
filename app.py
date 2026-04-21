import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
import torch

# 1. Load dataset (splits into train/test if you have multiple files, or use train_test_split)
dataset = load_dataset("json", data_files="dataset.json")

# 2. Split into train and eval if you only have one file
dataset = dataset["train"].train_test_split(test_size=0.1)  # 90% train, 10% eval

# 3. Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

# 4. Set pad token
tokenizer.pad_token = tokenizer.eos_token

# 5. Proper tokenization for causal LM
def tokenize_function(examples):
    # Combine instruction and response with a clear format
    texts = [
        f"Instruction: {inst}\nResponse: {resp}" 
        for inst, resp in zip(examples["instruction"], examples["response"])
    ]
    
    # Tokenize with labels = input_ids (for next-token prediction)
    result = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,  # Set a reasonable max length
        return_tensors=None  # Return lists, not tensors
    )
    
    # For causal LM, labels are the same as input_ids
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# 6. Training arguments (fixed syntax)
training_args = TrainingArguments(
    output_dir="test_trainer",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,  # Add this
    logging_steps=10,    # Add this
    save_strategy="epoch",
    # evaluation_strategy="epoch",  # Uncomment if you want eval
    dataloader_pin_memory=torch.cuda.is_available(),  # Only pin memory when a GPU accelerator is available
)

# 7. Trainer without custom metrics (loss is automatically computed)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    # Remove compute_metrics for causal LM, or use perplexity instead
)

# 8. Train
trainer.train()

# 9. Save
trainer.save_model("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")

# ============================================
# CHAT INTERFACE
# ============================================

import gradio as gr

def chat_with_model(message, history):
    # Format the input like training data
    prompt = f"Instruction: {message}\nResponse:"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,      # Maximum length of response
            do_sample=True,          # Enable sampling
            temperature=0.7,         # Control randomness (lower = more focused)
            top_p=0.9,               # Nucleus sampling
            top_k=50,                # Top-k sampling
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens (response part)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part (remove the prompt)
    response = generated_text[len(prompt):].strip()
    
    # Stop at next "Instruction:" if model generates one
    if "Instruction:" in response:
        response = response.split("Instruction:")[0].strip()
    
    return response

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_with_model,
    title="MaxBay",
    description="Chat with maxbay",
    examples=[
        "Who are you",
        "I m so sad",
        "Remidies for cold",
    ],
    theme="soft",
)

# Launch the chat app
if __name__ == "__main__":
    demo.launch(share=False)  # Set share=True to get a public link
