import json
from datetime import datetime
from collections import defaultdict


def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def preprocess_logs(logs, rules, mitre):
    processed_data = []

    rules_dict = {rule["Rule ID"]: rule for rule in rules}
    
    mitre_techniques = {entry["target ID"]: entry for entry in mitre}

    
    for rule_id, rule in rules_dict.items():
        rule_entry = {
            "rule_id": rule_id,
            "rule_description": rule["Description"],
            "mitre_id": rule["MITRE IDs"],
            "possible_attacks": [],
            "description": [],
            "mitigation_strategies": [],
        }

        for mitre_id in rule["MITRE IDs"]:
            if mitre_id in mitre_techniques:
                mitre_entry = mitre_techniques[mitre_id]
                rule_entry["possible_attacks"].append(mitre_entry["target name"])
                rule_entry["description"].append(mitre_entry["target description"])
                rule_entry["mitigation_strategies"].append(mitre_entry["mitigation description"])

        processed_data.append(rule_entry)

    return processed_data

def save_preprocessed_data(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":

    logs_path = "logs7000.jsonl"
    rules_path = "rules.jsonl"
    mitre_path = "mitreTM.jsonl"
    output_path = "preprocessed_data.json"

    logs = load_jsonl(logs_path)
    rules = load_jsonl(rules_path)
    mitre = load_jsonl(mitre_path)

    preprocessed_data = preprocess_logs(logs, rules, mitre)

    save_preprocessed_data(preprocessed_data, output_path)
    print(f"Preprocessing completato. Dati salvati in {output_path}")
