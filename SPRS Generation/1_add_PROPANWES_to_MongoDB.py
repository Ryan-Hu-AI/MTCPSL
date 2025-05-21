import json
from pymongo import MongoClient
"""
Use the PROPANEWS dataset from the paper 'Faking Fake News for Real Fake News Detection: Propaganda-loaded Training Data Generation.'
Independently generate files for the training set, validation set, and test set, which will serve as seeds for the LLM Summarised-Propaganda-Reason-Score generation.
"""
# Path to your JSONL file
label_file_path = "../PROPANEWS Dataset/train.jsonl"
output_label_file_path = "PROPANEWS_train.jsonl"

output_data = []
with open(label_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        if data["label"] == True:
            label = 1
        else:
            label = 0

        output_data.append({
            "id": data["path"],
            "txt": data["txt"],
            "label": label
        })

# Write the results into a JSONL file
with open(output_label_file_path, 'w', encoding='utf-8') as output_file:
    for entry in output_data:
        output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

client = MongoClient("mongodb://localhost:27017")
db = client['Propaganda_Seed'] 
collection = db['PROPANEWS_train']

# Open the JSONL file and insert data into MongoDB
with open(output_label_file_path, "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        collection.insert_one(data)

client.close()