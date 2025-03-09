import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

def load_preprocessed_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def prepare_dataset(data, log_data):
    examples = []
    for entry in log_data:

        prompt = (
            f"Log Information:\n"
            f"- Log ID: {entry.get('_id', 'N/A')}\n"
            f"- Agent Name: {entry.get('agent', {}).get('name', 'N/A')}\n"
            f"- Manager Name: {entry.get('manager', {}).get('name', 'N/A')}\n"
            f"- File: {entry.get('data', {}).get('file', 'N/A')}\n"
            f"- Rule ID: {entry.get('rule', {}).get('id', 'N/A')}\n"
            f"- Rule Level: {entry.get('rule', {}).get('level', 'N/A')}\n"
            f"- Rule Description: {entry.get('rule', {}).get('description', 'N/A')}\n"
            f"- Groups: {', '.join(entry.get('rule', {}).get('groups', [])) or 'None'}\n"
            f"- Decoder Name: {entry.get('decoder', {}).get('name', 'N/A')}\n"
            f"- Full Log: {entry.get('full_log', 'N/A')}\n"
            f"- Timestamp: {entry.get('timestamp', 'N/A')}\n"
            "\nBased on the information provided, categorize the log and provide potential threats, a detailed description, and mitigation strategies:\n"
        )

        matched_rule = next((rule for rule in data if str(rule.get("rule_id", "")) == str(entry.get('rule', {}).get('id', 'N/A'))), None)

        if matched_rule:
            description = matched_rule.get('description', ["No description available."])
            mitigation = matched_rule.get('mitigation_strategies', ["No mitigation strategies available."])
            target = (
                f"Category: {', '.join(matched_rule.get('possible_attacks', ['Unknown']))}\n"
                f"Description: {description[0] if description else 'No description available.'}\n"
                f"Mitigation: {mitigation[0] if mitigation else 'No mitigation strategies available.'}"
            )
        else:
            target = "Category: None\nDescription: No threats detected.\nMitigation: No actions required."

        examples.append({"input": prompt, "output": target})

    return examples


def tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["input"], padding="max_length", truncation=True, max_length=512
    )
    labels = tokenizer(
        examples["output"], padding="max_length", truncation=True, max_length=512
    )
    inputs["labels"] = labels["input_ids"]
    return inputs

# Main
if __name__ == "__main__":

    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    preprocessed_path = "preprocessed_data.json"
    log_path = "logs7000.jsonl"
    data = load_preprocessed_data(preprocessed_path)
    log_data = load_jsonl(log_path)

    dataset = prepare_dataset(data, log_data)
    hf_dataset = Dataset.from_pandas(pd.DataFrame(dataset))

    tokenized_dataset = hf_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["input", "output"]
    )

    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=500,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer),
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    trainer.save_model("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("Fine-tuned model saved!")