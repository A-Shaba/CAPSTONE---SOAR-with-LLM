from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

model_name = "./fine_tuned_model"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def generate_response(entry):

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

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)

    attention_mask = inputs['attention_mask']

    
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=attention_mask, 
        max_new_tokens=2000,  
        num_beams=5, 
        no_repeat_ngram_size=2, 
        early_stopping=True
    )


    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

print("Inizia la conversazione con il bot. Scrivi 'fine chat' per terminare.\n")

while True:
    
    print("\nInserisci i dati strutturati in formato JSON (o digita 'fine chat' per uscire):")
    user_input_raw = input("Tu: ")

   
    if user_input_raw.lower() == "fine chat":
        print("Bot: Fine della chat. Alla prossima!")
        break

    try:
        
        user_input = json.loads(user_input_raw)
      
        bot_response = generate_response(user_input)
        print("Bot:", bot_response)
    except json.JSONDecodeError:
        print("Errore: L'input fornito non è un JSON valido. Riprova.")