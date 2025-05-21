from pymongo import MongoClient
import jsonlines
import re
# Truncate news articles with more than 1500 words down to 1500 words, to be used as seeds for generating summarised propaganda and reasons.
# In test data, we truncate words with more than 1000 words down to 1000 words
client = MongoClient("mongodb://localhost:27017")
db = client['Propaganda_Seed']
collection = db['PROPANEWS_train']

# Define the maximum text length (e.g., 1000)
# Define the word count range
min_word_count = 0
max_word_count = 1500
keep_word_numbers = 1500
# Query documents where the text length is less than 1500 characters
documents = collection.find({})

# Define the output JSONL file path
output_file = "PROPANEWS_train_less_than_1500.jsonl"

def count_words(text):
    words = text.split()
    return len(words)

# Open the JSONL file for writing
with jsonlines.open(output_file, mode='w') as writer:
    for document in documents:
        field_value = document["txt"]
        id_value = document["id"]
        label_value = document["label"]
        word_count = count_words(field_value)

        # Check if the word count exceeds the max limit and trim the words if necessary
        if word_count > max_word_count:
            words = field_value.split()
            field_value = ' '.join(words[:keep_word_numbers])  # Keep only the first 1500 words
            word_count = max_word_count  # Update the word count to 1500

        # Only include documents within the specified word count range
        if word_count <= max_word_count:
            data = {
                "doc_id": id_value,
                "txt": field_value,
                "news_label": label_value
            }

            writer.write(data)