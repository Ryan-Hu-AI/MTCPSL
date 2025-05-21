# Use this file to generate score for summarized propaganda
import os
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
import json
import textwrap
import re
import pymongo
import jsonlines
from tqdm import tqdm
import itertools

if torch.cuda.is_available():
    print(f"Total {torch.cuda.device_count()} GPU(s) Available.")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No Available GPU.")

torch.cuda.empty_cache()

# Select the collection to use
collection_choice = 0
input_db_collection_name = ['PROPANEWS_train_Summarize_Propaganda_Loaded_Language',
                            'PROPANEWS_train_Summarize_Propaganda_Appeal_to_Fear',
                            'PROPANEWS_train_Summarize_Propaganda_Exaggeration',
                            'PROPANEWS_dev_Summarize_Propaganda_Loaded_Language',
                            'PROPANEWS_dev_Summarize_Propaganda_Appeal_to_Fear',
                            'PROPANEWS_dev_Summarize_Propaganda_Exaggeration',
                            'PROPANEWS_test_Summarize_Propaganda_Loaded_Language',
                            'PROPANEWS_test_Summarize_Propaganda_Appeal_to_Fear',
                            'PROPANEWS_test_Summarize_Propaganda_Exaggeration']

def building_prop_dataset(txt_seed):
    prompt = [
        "Article: ###" + txt_seed + "###",
    ]
    model.eval()
    with torch.no_grad():
        score_counter = 0
        score = None
        while True:
            if score_counter < 5:
                prop_scoring = generate_prop_score(prompt[0])
                score = get_prop_score(prop_scoring)

                if score is not None:
                    break
                else:
                    score_counter += 1
            else:
                score = 0
                break
    return score, prop_scoring


# Generate propaganda scores based on the summarized propaganda.
def generate_prop_score(text):
    prompt = get_style_text_prompt(text)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.000000000000001, top_p=0.000000000000001)
        final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        final_outputs =cut_off_text(final_outputs, '</s>')
        final_outputs = remove_substring(final_outputs, prompt)

    return final_outputs

def get_style_text_prompt(instruction):
    prompt_template = B_INST + SYSTEM_PROMPT1 + instruction + E_INST
    return prompt_template

# Extract the user's questions and the system's responses.
def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text

# Extract the system's responses.
def remove_substring(string, substring):
    return string.replace(substring, "")
# Extract propaganda score.
def get_prop_score(LLM_scoring):
    scores = re.findall(r'\b(0(\.\d)?|1(\.0)?)\b', LLM_scoring)
    if scores:
        return float(scores[0][0])
    else:
        return None


model_id="./Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=False, device_map='cuda:0', torch_dtype=torch.bfloat16)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

DEFAULT_SYSTEM_PROMPT1 = "Assess the propaganda level of the following article using a scale from 0 to 1, with increments of 0.1 (e.g., 0, 0.1, 0.2, ... 0.9, 1). As an editor known for being helpful, respectful, honest, and professional, your task is to provide a score that reflects the article's level of propaganda. Please provide only the score without explaining your reasoning."

SYSTEM_PROMPT1 = B_SYS + DEFAULT_SYSTEM_PROMPT1 + E_SYS
pip = pipeline("text-generation",
               model=model,
               tokenizer=tokenizer,
               do_sample=True,
               top_k=30,
               num_return_sequences=1,
               eos_token_id=tokenizer.eos_token_id)


output_db_collection_name = input_db_collection_name[collection_choice] + '_score'
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['Propaganda_News']
collection = db[output_db_collection_name]

seed_start = 0  #Set the starting point after the interruption in the document.
jsonl_file = "./data/" + input_db_collection_name[collection_choice] + ".json" # Export propaganda news json file from MongoDB. This path for scoring read.
print("Processing: "+jsonl_file)
# Open the JSONL file for reading
# Calculate the number of documents
record_items = 0
with open(jsonl_file, 'r', encoding='utf-8') as f:
    for _ in jsonlines.Reader(f):
        record_items += 1

with open(jsonl_file, 'r', encoding='utf-8') as f:
    records_to_skip = seed_start  # Set the starting point after the interruption in the document.
    reader = itertools.islice(jsonlines.Reader(f), records_to_skip, None)

    generate_counter = 0
    total_items = record_items - records_to_skip

    # Create a tqdm progress bar
    with tqdm(total=total_items, desc="Processing JSONL") as pbar:
        for item in reader:
            # Access the "id" and "text" fields from each JSON object
            doc_id = item["doc_id"]
            news_label = item["news_label"]
            prop_label = item["prop_label"]
            prop_text = item["prop_text"]
            generate_explanation = item["generate_explanation"]
            original_txt = item["original_txt"]

            score, prop_scoring = building_prop_dataset(prop_text)

            data = {
                "doc_id":doc_id,
                "news_label": news_label,
                "prop_label": prop_label,
                "prop_text": prop_text,
                "LLM_prop_scoring": prop_scoring,
                "prop_score": score,
                "generate_explanation": generate_explanation,
                "original_txt": original_txt,
            }
            collection.insert_one(data)
            generate_counter += 1

            if generate_counter % 5 == 0:
                    torch.cuda.empty_cache()
            # Update the progress bar
            pbar.update(1)

client.close()



