import random
from pymongo import MongoClient
import jsonlines
import os
"""
Data Configuration :
(original_txt, news_label, prop_txt_1, prop_label_1,prop_score_1, reason1)
(original_txt, news_label, prop_txt_2, prop_label_2,prop_score_2, reason2)
(original_txt, news_label, prop_txt_3, prop_label_3,prop_score_3, reason3)

"""
random_seed = 42 # for random chose data pair
choice_prop = 0


random.seed(random_seed)
prop_dbs = ["PROPANEWS_train_Summarize_Propaganda_LL_E_AF_score"]
file_names = ["PROPANEWS_train_Summarize_Propaganda_LL_E_AF_highest_score"]

#Output folder names, "train" uses repeat data, while the other datasets all use nonrepeat.
output_folder = file_names[choice_prop] + "_r" + str(random_seed) +"_prop_pair_news_nonrepeat_dataset_balanced"
os.makedirs(output_folder, exist_ok=True)

# Write the data from the DB into prop_seed.jsonl and remove the _id attribute
def sample_prop_seed( file_name, input ):
    with jsonlines.open(file_name, 'w') as writer:
        for data in input:
            if '_id' in data:
                del data['_id']
            writer.write(data)

# Generate a training data pair for each of the three types of propaganda corresponding to a single news item
def sample_prop_pair_news_repeat_dataset(input_file_name, output_file_name):
    processed_data = []

    # Read data from the input file
    with jsonlines.open(input_file_name, 'r') as reader:
        for item in reader:
            # Generate three pairs for each original_txt
            for i in range(1, 4):  # Assuming there are three sets of prop data available in each item
                pair = {
                    'original_txt': item['original_txt'],
                    'news_label': item['news_label'],
                    'prop_txt': item[f'prop_txt_{i}'],
                    'prop_score': item[f'prop_score_{i}'],
                    'prop_label': item[f'prop_label_{i}'],
                    'reason': item[f'generate_explanation_{i}']
                }
                processed_data.append(pair)

    random.shuffle(processed_data)
    with jsonlines.open(output_file_name, 'w') as writer:
        for data in processed_data:
            writer.write(data)

# Generate a test data pair where one news item is randomly associated with only one type of propaganda.
def sample_prop_pair_news_non_repeat_dataset_unique(input_file_name, output_file_name):
    processed_data = []
    seen_doc_id = set()

    # Read the data and randomly select prop_data
    with jsonlines.open(input_file_name, 'r') as reader:
        counter = 0
        for item in reader:
            counter += 1
            # Ensure that each original_txt is selected only once
            if item['doc_id'] not in seen_doc_id:
                prop_options = []
                for i in range(1, 4):  # Collect all prop_data
                    prop_options.append({
                        'original_txt': item['original_txt'],
                        'news_label': item['news_label'],
                        'prop_txt': item[f'prop_txt_{i}'],
                        'prop_score': item[f'prop_score_{i}'],
                        'prop_label': item[f'prop_label_{i}'],
                        'reason': item[f'generate_explanation_{i}']
                    })
                selected_pair = random.choice(prop_options)
                processed_data.append(selected_pair)
                seen_doc_id.add(item['doc_id'])
            else:
                print(str(counter))

    random.shuffle(processed_data)
    with jsonlines.open(output_file_name, 'w') as writer:
        for data in processed_data:
            writer.write(data)

# Generate a test data pair where a news item is only associated with the propaganda that has the highest score
def sample_prop_pair_news_non_repeat_high_score_dataset_unique(input_file_name, output_file_name):
    processed_data = []
    seen_doc_id = set()

    with jsonlines.open(input_file_name, 'r') as reader:
        counter = 0
        for item in reader:
            counter += 1
            if item['doc_id'] not in seen_doc_id:
                prop_options = []
                for i in range(1, 4):
                    prop_options.append({
                        'original_txt': item['original_txt'],
                        'news_label': item['news_label'],
                        'prop_txt': item[f'prop_txt_{i}'],
                        'prop_score': item[f'prop_score_{i}'],
                        'prop_label': item[f'prop_label_{i}'],
                        'reason': item[f'generate_explanation_{i}']
                    })

                # Compare and select the prop_score with the highest value. If the scores are the same, randomly choose one of the two.
                highest_score_option = max(prop_options, key=lambda x: x['prop_score'])
                highest_score = highest_score_option['prop_score']
                top_options = [option for option in prop_options if option['prop_score'] == highest_score]
                selected_option = random.choice(top_options)
                processed_data.append(selected_option)

            else:
                print(str(counter))

    random.shuffle(processed_data)
    with jsonlines.open(output_file_name, 'w') as writer:
        for data in processed_data:
            writer.write(data)


client = MongoClient("mongodb://localhost:27017")
db = client["Propaganda_News"]
collection = db[prop_dbs[choice_prop]]

pipeline = [
    {
        "$match": {
            "prop_text_1": {"$ne": "", "$ne": "..."},  # Exclude cases where prop_txt is an empty string or contains only "..."
            "prop_text_2": {"$ne": "", "$ne": "..."},
            "prop_text_3": {"$ne": "", "$ne": "..."},
            "$expr": {
                "$and": [ # Retain only the parts with fewer than 1500 words.
                    {"$gt": [{"$size": {"$split": ["$prop_txt_1", " "]}}, 0]},
                    {"$lt": [{"$size": {"$split": ["$prop_txt_1", " "]}}, 1500]},
                    {"$gt": [{"$size": {"$split": ["$prop_txt_2", " "]}}, 0]},
                    {"$lt": [{"$size": {"$split": ["$prop_txt_2", " "]}}, 1500]},
                    {"$gt": [{"$size": {"$split": ["$prop_txt_3", " "]}}, 0]},
                    {"$lt": [{"$size": {"$split": ["$prop_txt_3", " "]}}, 1500]}
                ]
            }
        }
    },
]

all_data = list(collection.aggregate(pipeline))
client.close()
print(len(all_data))

#train_seed, dev_seed, test_seed, snopes_seed, politifact_seed
sample_prop_seed(output_folder + '/train_seed.jsonl', all_data)
sample_prop_pair_news_repeat_dataset(output_folder + '/train_seed.jsonl', output_folder + '/train_SPRS_repeat.jsonl')

# nonrepaeat dataset for dev_seed, test_seed, snopes_seed, politifact_seed
#sample_prop_pair_news_non_repeat_dataset_unique(output_folder + '/dev_seed.jsonl', output_folder + '/dev_SPRS_nonrepeat.jsonl')


# Generate test_seed.jsonl, politifact_seed.jsonl, and snopes_seed.jsonl, with the option to generate data with the highest score
#sample_prop_pair_news_non_repeat_high_score_dataset_unique(output_folder + '/test_seed.jsonl', output_folder + '/test_SPRS_nonrepeat_highest_score.jsonl')
