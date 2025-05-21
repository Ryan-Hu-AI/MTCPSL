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
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

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
parser.add_argument('--politifact_test_dataset', default='politifact_SPRS_nonrepeat.jsonl', type=str, required=False)
parser.add_argument('--snopes_test_dataset', default='snopes_SPRS_nonrepeat.jsonl', type=str, required=False)
parser.add_argument('--training_test_dataset', default='test_SPRS_nonrepeat.jsonl', type=str, required=False)

parser.add_argument('--politifact_test_output_name', default='politifact_SPRS_nr', type=str, required=False)
parser.add_argument('--snopes_test_output_name', default='snopes_SPRS_nr', type=str, required=False)
parser.add_argument('--training_test_output_name', default='test_SPRS_nr', type=str, required=False)
parser.add_argument('--test_news_repeat', default='no', type=str, required=False)
parser.add_argument('--prop_model', default="Multi_Task_CounterPropaganda_Semantic_Learning", required=False)
parser.add_argument('--seed', default=8, type=float, required=False) #設定Ture，表示執行時要給路徑參數
args = parser.parse_args()

# fix random seed
random.seed(int(args.seed))
np.random.seed(int(args.seed))
torch.manual_seed(int(args.seed))
torch.backends.cudnn.enabled = True

# Get our MTCPSL model class
model_class = globals().get(args.prop_model)
if model_class:
    model = model_class(args.model_name).cuda()
else:
    raise ValueError(f"Model {args.prop_model} not found!")

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
output_dir = '/'.join(args.checkpoint_path.split('/')[:-1])

if args.use_checkpoint == True:
    model_path = args.checkpoint_path 
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=True)

#Initiate data loader
politifact_test_set = News_Propa_Reason_Score_InputsV3PropaFakeDataset(os.path.join(args.data_dir, args.politifact_test_dataset), args.model_name, args.max_sequence_length)
politifact_test_loader = DataLoader(politifact_test_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=politifact_test_set.collate_fn)
politifact_vector_output_file = os.path.join(output_dir, args.best_type + '_'+args.politifact_test_output_name+'_vect.json')

snopes_test_set = News_Propa_Reason_Score_InputsV3PropaFakeDataset(os.path.join(args.data_dir, args.snopes_test_dataset), args.model_name, args.max_sequence_length)
snopes_test_loader = DataLoader(snopes_test_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=snopes_test_set.collate_fn)
snopes_vector_output_file = os.path.join(output_dir, args.best_type + '_'+args.snopes_test_output_name+'_vect.json')

training_test_set = News_Propa_Reason_Score_InputsV3PropaFakeDataset(os.path.join(args.data_dir, args.training_test_dataset), args.model_name, args.max_sequence_length)
training_test_loader = DataLoader(training_test_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=training_test_set.collate_fn)
training_vector_output_file = os.path.join(output_dir, args.best_type + '_'+args.training_test_output_name+'_vect.json')


with torch.no_grad():
    model.eval()

    for batch_idx, (input_ids_original, attention_mask_original, input_ids_prop, attention_mask_prop, input_ids_reason,
                    attention_mask_reason, news_label, prop_label, prop_confidence_score_label) in enumerate(politifact_test_loader):
        # In Section 5: Experiment - T-SNE Visualization, we utilized the trained model to obtain embeddings from the Politifact dataset.
        _,_,_, embeddings = model(return_embeddings=True, input_ids_news=input_ids_original,
                                           attention_mask_news=attention_mask_original,
                                           input_ids_prop=input_ids_prop, attention_mask_prop=attention_mask_prop,
                                           reason=input_ids_reason, attention_mask_reason=attention_mask_reason, propaganda_confidence_score=prop_confidence_score_label)

        # Open file in append mode to avoid overwriting
        with open(politifact_vector_output_file, 'a') as f:
            for i, emb in enumerate(embeddings):
                emb['news_label'] = news_label[i].item() if news_label is not None else None
                emb['prop_label'] = prop_label[i].tolist() if prop_label is not None else None
                emb['prop_confidence_score_label'] = prop_confidence_score_label[
                    i].item() if prop_confidence_score_label is not None else None
                f.write(json.dumps(emb) + "\n")


    for batch_idx, (input_ids_original, attention_mask_original, input_ids_prop, attention_mask_prop, input_ids_reason,
                    attention_mask_reason, news_label, prop_label, prop_confidence_score_label) in enumerate(snopes_test_loader):
        # In Section 5: Experiment - T-SNE Visualization, we utilized the trained model to obtain embeddings from the Snopes dataset.
        _,_,_, embeddings = model(return_embeddings=True, input_ids_news=input_ids_original,
                                           attention_mask_news=attention_mask_original,
                                           input_ids_prop=input_ids_prop, attention_mask_prop=attention_mask_prop,
                                           reason=input_ids_reason, attention_mask_reason=attention_mask_reason, propaganda_confidence_score=prop_confidence_score_label)

        # Open file in append mode to avoid overwriting
        with open(snopes_vector_output_file, 'a') as f:
            for i, emb in enumerate(embeddings):
                emb['news_label'] = news_label[i].item() if news_label is not None else None
                emb['prop_label'] = prop_label[i].tolist() if prop_label is not None else None
                emb['prop_confidence_score_label'] = prop_confidence_score_label[
                    i].item() if prop_confidence_score_label is not None else None
                f.write(json.dumps(emb) + "\n")


    for batch_idx, (input_ids_original, attention_mask_original, input_ids_prop, attention_mask_prop, input_ids_reason,
                    attention_mask_reason, news_label, prop_label, prop_confidence_score_label) in enumerate(training_test_loader):
        # In Section 5: Experiment - T-SNE Visualization, we utilized the trained model to obtain embeddings from the in domain test dataset.
        _,_,_, embeddings = model(return_embeddings=True, input_ids_news=input_ids_original,
                                           attention_mask_news=attention_mask_original,
                                           input_ids_prop=input_ids_prop, attention_mask_prop=attention_mask_prop,
                                           reason=input_ids_reason, attention_mask_reason=attention_mask_reason, propaganda_confidence_score=prop_confidence_score_label)

        # Open file in append mode to avoid overwriting
        with open(training_vector_output_file, 'a') as f:
            for i, emb in enumerate(embeddings):
                emb['news_label'] = news_label[i].item() if news_label is not None else None
                emb['prop_label'] = prop_label[i].tolist() if prop_label is not None else None
                emb['prop_confidence_score_label'] = prop_confidence_score_label[
                    i].item() if prop_confidence_score_label is not None else None
                f.write(json.dumps(emb) + "\n")

print("model path:", model_path)







