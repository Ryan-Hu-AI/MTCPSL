import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.utils.data import DataLoader
import argparse
import numpy as np
import json
from my_models import Multi_Task_CounterPropaganda_Semantic_Learning
from my_dataset_process import News_Propa_Reason_Score_Inputs_PropaFakeDataset
import random
import time
from pymongo import MongoClient
import re
from sklearn.metrics import classification_report, mean_squared_error
import torch
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import RocCurveDisplay
matplotlib.use('Agg')
from sklearn.metrics import roc_auc_score

if torch.cuda.is_available():
    print(f"Total {torch.cuda.device_count()} GPU(s) Available.")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No Available GPU.")

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('--max_sequence_length', default=512)
parser.add_argument('--data_dir', default='../PROPANEWS_SPRS Dataset/test_data/', required=False)
parser.add_argument('--eval_batch_size', default=2, type=int)
parser.add_argument('--model_name', default='roberta-large', required=False)
parser.add_argument('--use_checkpoint', default=True, type=bool, required=False)
parser.add_argument('--checkpoint_path', default='../output/30n_100p_100s_r8_PROPANEWS_SPRS_MTCPSL_P_Mask_R_fcr_task_nprs_inputs_news_repeat/Epoch_21_news.pt', type=str, required=False)
parser.add_argument('--best_type', default='News', type=str, required=False)
parser.add_argument('--test_dataset', default='politifact_SPRS_nonrepeat.jsonl', type=str, required=False)
#parser.add_argument('--test_dataset', default='politifact_SPRS_nonrepeat_highest_score.jsonl', type=str, required=False)
parser.add_argument('--test_output_name', default='politifact_SPRS_nr', type=str, required=False)
parser.add_argument('--test_news_repeat', default='no', type=str, required=False)
parser.add_argument('--prop_model', default="Multi_Task_CounterPropaganda_Semantic_Learning", required=False)
parser.add_argument('--training_news_repeat', default='yes', type=str, required=False)
parser.add_argument('--database', default='PROPANEWS_SPRS', type=str, required=False) #Save data in MongoDB
parser.add_argument('--collection', default='Epoch30_MTCPSL_P_Mask_R_test', type=str, required=False)
parser.add_argument('--seed', default=8, type=float, required=False)
args = parser.parse_args()

# Save data in MongoDB
client = MongoClient('mongodb://localhost:27017')
db = client[args.database]
collection = db[args.collection]

# fix random seed
random.seed(int(args.seed))
np.random.seed(int(args.seed))
torch.manual_seed(int(args.seed))
torch.backends.cudnn.enabled = True

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
output_dir = '/'.join(args.checkpoint_path.split('/')[:-1])

# Get our MTCPSL model class
model_class = globals().get(args.prop_model)
if model_class:
    model = model_class(args.model_name).cuda()
else:
    raise ValueError(f"Model {args.prop_model} not found!")

if args.use_checkpoint == True:
    model_path = args.checkpoint_path
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=True)

#Initiate data loader
test_set = News_Propa_Reason_Score_Inputs_PropaFakeDataset(os.path.join(args.data_dir, args.test_dataset), args.model_name, args.max_sequence_length)
test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=test_set.collate_fn)
test_output_file = os.path.join(output_dir, args.best_type + '_'+args.test_output_name+'_pred.json')

with torch.no_grad():
    model.eval()
    test_news_outputs = []
    test_prop_outputs = []
    test_prop_score_outputs = []
    test_news_labels = []
    test_prop_labels = []
    test_prop_score_labels = []

    for batch_idx, (input_ids_original, attention_mask_original, input_ids_prop, attention_mask_prop, input_ids_reason,
                    attention_mask_reason, news_label, prop_label, prop_score_label) in enumerate(test_loader):
        # Fake News Detection output, Propaganda Classification output, Propaganda Score Regression output
        news_outputs, prop_outputs, prop_score_outputs, _ = model(input_ids_news=input_ids_original,
                                           attention_mask_news=attention_mask_original,
                                           input_ids_prop=input_ids_prop, attention_mask_prop=attention_mask_prop,
                                           reason=input_ids_reason, attention_mask_reason=attention_mask_reason, propaganda_score=prop_score_label)
        test_news_outputs.append(news_outputs)
        test_prop_outputs.append(prop_outputs)
        test_prop_score_outputs.append(prop_score_outputs)
        test_news_labels.append(news_label)
        test_prop_labels.append(prop_label)
        test_prop_score_labels.append(prop_score_label)

    test_news_outputs = torch.cat(test_news_outputs, dim=0)
    test_news_labels = torch.cat(test_news_labels, dim=0)
    test_prop_outputs = torch.cat(test_prop_outputs, dim=0)
    test_prop_labels = torch.cat(test_prop_labels, dim=0)
    test_prop_score_outputs = torch.cat(test_prop_score_outputs, dim=0)
    test_prop_score_labels = torch.cat(test_prop_score_labels, dim=0)


    # Calculate the accuracy and f1-score of propaganda classification task
    _, prop_predicted_labels = torch.max(test_prop_outputs, 1)
    prop_classification_rep = classification_report(test_prop_labels.cpu(), prop_predicted_labels.cpu(), digits=4)


    # Calculate propaganda level score error
    test_prop_score_outputs = test_prop_score_outputs.squeeze(dim=1)
    test_prop_score_mse = mean_squared_error(test_prop_score_labels.cpu().numpy(), test_prop_score_outputs.cpu().numpy())

    # Calculate the accuracy, f1-score and AUC of fake news detection task
    test_news_outputs_bool = test_news_outputs > 0.5 
    test_news_labels_bool = test_news_labels == 1
    news_classification_rep = classification_report(test_news_labels_bool.cpu(), test_news_outputs_bool.cpu(), digits=4)
    RocCurveDisplay.from_predictions(test_news_labels_bool.cpu(), test_news_outputs.cpu(), name=args.prop_model,
                                     color="darkorange", pos_label=1, plot_chance_level=True)
    print(f"Estimator: {args.prop_model}.")
    true_auc = roc_auc_score(test_news_labels.cpu().numpy(), test_news_outputs.detach().cpu().numpy())
    print(f"Test News True_AUC: {true_auc}.")
    
    plt.axis("square")
    plt.legend()
    plt.savefig(os.path.join(output_dir, args.test_output_name + "_" + args.best_type + "_AUC.png"))
    matplotlib.pyplot.close()

    news_pattern = re.compile(r'\b(\w+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)\b')
    news_matches = re.findall(news_pattern, news_classification_rep)
    accuracy_pattern = re.compile(r'accuracy\s+([0-9.]+)')
    news_accuracy_matches = re.findall(accuracy_pattern, news_classification_rep)
    prop_pattern = re.compile(r'^(?:\s*(\d+|macro avg|weighted avg))\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)$',
                              re.MULTILINE)
    prop_matches = re.findall(prop_pattern, prop_classification_rep)
    prop_accuracy_matches = re.findall(accuracy_pattern, prop_classification_rep)

    classification_rep_dict = {
        "Pretrain_model": args.model_name,
        "Model": args.prop_model,
        "Training_News_repeat": args.training_news_repeat,
        "Test_news_repeat": args.test_news_repeat,
        "Best_Type": args.best_type,
        "Checkpoint_Path": args.checkpoint_path,
        "Test_Dataset": args.test_dataset,
        "Random_Seed": args.seed,
        "News_AUC": true_auc,
        "News_Accuracy": float(news_accuracy_matches[0]),
        "News_True_Precision": float(news_matches[1][1]),
        "News_True_Recall": float(news_matches[1][2]),
        "News_True_F1-score": float(news_matches[1][3]),
        "News_True_Support": int(news_matches[1][4]),
        "News_Fake_Precision": float(news_matches[0][1]),
        "News_Fake_Recall": float(news_matches[0][2]),
        "News_Fake_F1-score": float(news_matches[0][3]),
        "News_Fake_Support": int(news_matches[0][4]),
        "News_Macro_Avg_Precision": float(news_matches[2][1]),
        "News_Macro_Avg_Recall": float(news_matches[2][2]),
        "News_Macro_Avg_F1-score": float(news_matches[2][3]),
        "News_Weighted_Avg_Precision": float(news_matches[3][1]),
        "News_Weighted_Avg_Recall": float(news_matches[3][2]),
        "News_Weighted_Avg_F1-score": float(news_matches[3][3]),
        "Prop_Score_MSE": float(test_prop_score_mse),
        "Prop_Accuracy": float(prop_accuracy_matches[0]),
        "Loaded_Language_Precision": float(prop_matches[0][1]),
        "Loaded_Language_Recall": float(prop_matches[0][2]),
        "Loaded_Language_F1-score": float(prop_matches[0][3]),
        "Loaded_Language_Support": int(prop_matches[0][4]),
        "Exaggeration_Precision": float(prop_matches[1][1]),
        "Exaggeration_Recall": float(prop_matches[1][2]),
        "Exaggeration_F1-score": float(prop_matches[1][3]),
        "Exaggeration_Support": int(prop_matches[1][4]),
        "Appeal_to_Fear_Precision": float(prop_matches[2][1]),
        "Appeal_to_Fear_Recall": float(prop_matches[2][2]),
        "Appeal_to_Fear_F1-score": float(prop_matches[2][3]),
        "Appeal_to_Fear_Support": int(prop_matches[2][4]),
        "Propaganda_Macro_Avg_Precision": float(prop_matches[3][1]),
        "Propaganda_Macro_Avg_Recall": float(prop_matches[3][2]),
        "Propaganda_Macro_Avg_F1-score": float(prop_matches[3][3]),
        "Propaganda_Weighted_Avg_Precision": float(prop_matches[4][1]),
        "Propaganda_Weighted_Avg_Recall": float(prop_matches[4][2]),
        "Propaganda_Weighted_Avg_F1-score": float(prop_matches[4][3]),
        "All_Propaganda_Data_Support": int(prop_matches[4][4])
    }
    result = collection.insert_one(classification_rep_dict)
    print(f"Insert success，ID: {result.inserted_id} \n")

    test_news_outputs = [float(o) for o in test_news_outputs]
    test_prop_outputs = [float(o) for o in prop_predicted_labels]
    test_prop_score_outputs = [float(o) for o in test_prop_score_outputs]
    with open(test_output_file, 'w') as f:
        json.dump({'news_output': test_news_outputs, 'prop_output': test_prop_outputs, 'prop_score_output': test_prop_score_outputs}, f)




