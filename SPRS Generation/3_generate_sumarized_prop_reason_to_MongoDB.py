# Use this file to generate summarized propaganda, reason
import os
import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
def building_prop_dataset(prop_label, txt_seed):
    prompt = [
        "Rewriting style: " + prop_label + ". Rewrite the sentence and Explain the rewriting sentence: \n" + txt_seed + "\n",
    ]

    model.eval()
    with torch.no_grad():
        while True:
            prop_and_explanation = generate_prop_text(prompt[0])
            # Extract the content of the Rewrite and Explanation from the generated_text.
            prop_txt, generate_explanation = get_prop_text_explanation(prop_and_explanation)
            if prop_txt == "...":
                continue
            if prop_txt == "":
                continue
            if prop_txt == "...;":
                continue
            if prop_txt is False and generate_explanation is False:
                continue
            break
    return prop_txt, generate_explanation

# Generate propaganda text and create explanations based on original news.
def generate_prop_text(text):
    prompt = get_style_text_prompt(text)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.8)
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

# Extract the reasons for the propaganda.
def get_prop_text_explanation(text):
    try:
        # Find the index of "Rewrite:"
        start_index = text.index("Rewrite:")
    except ValueError:
        # "Rewrite:" not found, return False or None or any other desired value
        return False, False  # You can use None or any other suitable value as needed

    try:
        # Find the index of "Explanation:"
        end_index = text.index("Explanation:")
    except ValueError:
        # "Rewrite:" not found, return False or None or any other desired value
        return False, False  # You can use None or any other suitable value as needed

    # Extract the text between "Rewrite:" and "Explanation:"
    prop_text = text[start_index + len("Rewrite:"):end_index].strip()
    generate_explanation = text[end_index + len("Explanation:"):].strip()
    return prop_text, generate_explanation

model_id="./Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=False, device_map='cuda:0', torch_dtype=torch.bfloat16)
pip = pipeline("text-generation",
               model=model,
               tokenizer=tokenizer,
               do_sample=True,
               top_k=30,
               num_return_sequences=1,
               eos_token_id=tokenizer.eos_token_id)

# In Section 3 of our paper, Dataset Generation, we generated three types of propaganda.
prop_label_list =["Loaded_Language", "Exaggeration", "Appeal_to_Fear"]
prop_label = prop_label_list[0]
db_collection_name = 'PROPANEWS_train_Summarize_Propaganda_Loaded_Language'
seed_start = 0  # Set the starting point after the interruption in the document.
jsonl_file = "./data/PROPANEWS_train_less_than_1500.jsonl"  # Input truncated news path
prop_txt = ""
generate_explanation = ""
txt_seed = ""

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# Prompt to generate summarizing news and reasons.Generate for summarizing news and reasons.
DEFAULT_SYSTEM_PROMPT1 = """\
You are a helpful, respectful, honest and professional editor. You are also a publicity expert.
Your task is to help me summarize and rewrite sentences in a new style that I assign.
Avoid directly copying the original text. 
The rewrite sentence needs to be constrained to a minimum of 30 words and a maximum of 150 words.
You also need to explain the reasons you modified it.
The explanation sentence needs to be constrained to a minimum of 30 words and a maximum of 150 words.
The reasons for explaining should be a complete sentence.
Then, when I need you to rewrite the sentence, I will say: [Style:...; Rewrite the sentence and Explain the rewritten sentence:]. 
I need you to answer in the format of the tags [Rewrite:] and [Explanation:]. 
Your answer should only answer the question once and not have any incomplete sentence after the answer is done.
Please provide a direct answer to the question below without posing any further questions.
For example:
I will give you: Style: Appeal to Authority, Rewrite the sentence and Explain the rewriting sentence: ...
I want you to response in list format:
\nRewrite:...
\nExplanation:...

If you don't know the answer to a question, please don't share false information.
"""

SYSTEM_PROMPT1 = B_SYS + DEFAULT_SYSTEM_PROMPT1 + E_SYS

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['Propaganda_News']
collection = db[db_collection_name]

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
            txt_seed = item["txt"]
            news_label = item["news_label"]

            # Generate summarized propaganda and reason
            prop_txt, generate_explanation = building_prop_dataset(prop_label, txt_seed)

            data = {
                "doc_id":doc_id,
                "news_label": news_label,
                "prop_label": prop_label,
                "prop_text": prop_txt,
                "generate_explanation": generate_explanation,
                "original_txt": txt_seed
            }
            collection.insert_one(data)
            generate_counter += 1

            if generate_counter % 5 == 0:
                    torch.cuda.empty_cache()
            # Update the progress bar
            pbar.update(1)

client.close()