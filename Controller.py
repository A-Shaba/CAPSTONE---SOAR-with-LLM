import os
import json
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


model_name = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def generate_response(entry):
    prompt = (
        f"Log ID: {entry.get('_id', 'N/A')}\n"
        f"Agent Name: {entry.get('agent', {}).get('name', 'N/A')}\n"
        f"Agent ID: {entry.get('agent', {}).get('id', 'N/A')}\n"
        f"Manager Name: {entry.get('manager', {}).get('name', 'N/A')}\n"
        f"File: {entry.get('data', {}).get('file', 'N/A')}\n"
        f"Title: {entry.get('data', {}).get('title', 'N/A')}\n"
        f"Rule ID: {entry.get('rule', {}).get('id', 'N/A')}\n"
        f"Rule Level: {entry.get('rule', {}).get('level', 'N/A')}\n"
        f"Rule Description: {entry.get('rule', {}).get('description', 'N/A')}\n"
        f"Fired Times: {entry.get('rule', {}).get('firedtimes', 'N/A')}\n"
        f"Mail Alert: {entry.get('rule', {}).get('mail', 'N/A')}\n"
        f"PCI DSS: {', '.join(entry.get('rule', {}).get('pci_dss', [])) or 'None'}\n"
        f"GDPR: {', '.join(entry.get('rule', {}).get('gdpr', [])) or 'None'}\n"
        f"Groups: {', '.join(entry.get('rule', {}).get('groups', [])) or 'None'}\n"
        f"Decoder Name: {entry.get('decoder', {}).get('name', 'N/A')}\n"
        f"Full Log: {entry.get('full_log', 'N/A')}\n"
        f"Input Type: {entry.get('input', {}).get('type', 'N/A')}\n"
        f"Location: {entry.get('location', 'N/A')}\n"
        f"Timestamp: {entry.get('timestamp', 'N/A')}\n"
        "Conclusion: "
    )


    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)


    attention_mask = inputs['attention_mask']


    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=attention_mask,
        max_new_tokens=2000,  #limite per i nuovi token da generare
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )


    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response



def monitor_file(file_path):
    print(f"Monitoring changes in {file_path}...")
    last_position = 0

    while True:
        try:
            
            file_size = os.path.getsize(file_path)


            if file_size > last_position:
                with open(file_path, "r") as f:
                    
                    f.seek(last_position)


                    new_data = f.read()
                    last_position = f.tell()


                    for line in new_data.splitlines():
                        try:
                            entry = json.loads(line)


                            log_id = entry.get("_id", "ID not identified")
                            log_title = entry.get("data", {}).get("title", "Title not identified")
                            print("-"*80)
                            print(f"\nNew log detected\n ID: {log_id} \n Title: {log_title}")

                            # Genera la risposta
                            bot_response = generate_response(entry)
                            print("\nLLM response:")
                            print(f"{bot_response} \n")

                        except json.JSONDecodeError as e:
                            print(f"Error on decoding the JSON: {e}")
            time.sleep(1)
        except FileNotFoundError:
            print(f"File not found: {file_path}. wait...")
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nMonitoraggio interrotto.")
            break


file_to_monitor = "/var/ossec/logs/alerts/alerts.json"
monitor_file(file_to_monitor)
