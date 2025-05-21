import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
import pandas as pd
from my_models import Multi_Task_CounterPropaganda_Semantic_Learning
from my_dataset_process import News_Propa_Reason_Score_Inputs_PropaFakeDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
import json
from tqdm import tqdm
import random
import time
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


if torch.cuda.is_available():
    print(f"Total {torch.cuda.device_count()} GPU(s) Available.")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No Available GPU.")

# Configuration
parser = argparse.ArgumentParser()
parser.add_argument('--max_sequence_length', default=512, type=int)
parser.add_argument('--model_name', default='roberta-large', required=False )
parser.add_argument('--data_dir', default='../PROPANEWS_SPRS Dataset/', required=False)
parser.add_argument('--prop_model', default="Multi_Task_CounterPropaganda_Semantic_Learning", required=False)
parser.add_argument('--data_name', default="30n_100p_100s_r8_PROPANEWS_SPRS_MTCPSL_P_Mask_R_fcr_task_nprs_inputs_news_repeat", required=False) #Name for output dir
parser.add_argument('--warmup_epoch', default=5, type=int)
parser.add_argument('--max_epoch', default=30, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--eval_batch_size', default=2, type=int)
parser.add_argument('--accumulate_step', default=8, type=int)
parser.add_argument('--output_dir', default='../output/', required=False)
parser.add_argument('--news_weight', default=0.33, type=float, required=False)  # Loss Weight for Fake News Detection
parser.add_argument('--prop_weight', default=1, type=float, required=False) #Loss Weight for Propaganda Classification
parser.add_argument('--prop_score_weight', default=1, type=float, required=False) #Loss Weight for Propaganda Score Regression
parser.add_argument('--seed', default=8, type=float, required=False)
args = parser.parse_args()

random.seed(int(args.seed))
np.random.seed(int(args.seed))
torch.manual_seed(int(args.seed))
torch.backends.cudnn.enabled = True

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

output_dir = os.path.join(args.output_dir, args.model_name + "_" + args.data_name + "_" + timestamp)
os.makedirs(output_dir)

# Get our MTCPSL model class
model_class = globals().get(args.prop_model)
if model_class:
    model = model_class(args.model_name).cuda()
else:
    raise ValueError(f"Model {args.prop_model} not found!")

# Initiate data loader
train_set = News_Propa_Reason_Score_Inputs_PropaFakeDataset(os.path.join(args.data_dir, 'train_SPRS_repeat.jsonl'), args.model_name, args.max_sequence_length)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=train_set.collate_fn)
dev_set = News_Propa_Reason_Score_Inputs_PropaFakeDataset(os.path.join(args.data_dir, 'dev_SPRS_nonrepeat.jsonl'), args.model_name, args.max_sequence_length)
dev_loader = DataLoader(dev_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=dev_set.collate_fn)

# Define loss
news_critera = nn.BCELoss()
prop_critera = nn.CrossEntropyLoss()
prop_score_critera = nn.MSELoss()

state = dict(model=model.state_dict())

# Optimizer setting
param_groups = [
    {
        'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
        'lr': 5e-5, 'weight_decay': 1e-05
    },
]

batch_num = len(train_set) // (args.batch_size * args.accumulate_step)
+ (len(train_set) % (args.batch_size * args.accumulate_step) != 0)

optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=batch_num*args.warmup_epoch,
                                           num_training_steps=batch_num*args.max_epoch)

best_dev_news_auc = 0
best_dev_prop_f1 = 0
model_path = os.path.join(output_dir,'best.pt')

training_losses = []
training_news_losses = []
training_prop_losses = []
training_prop_score_losses = []
dev_news_aucs = []
dev_prop_accuracies = []
dev_prop_f1_scores = []
dev_prop_score_mse = []

for epoch in range(args.max_epoch):
    training_loss = 0
    training_news_loss = 0
    training_prop_loss = 0
    training_prop_score_loss = 0
    model.train()

    for batch_idx, (input_ids_original, attention_mask_original, input_ids_prop, attention_mask_prop, input_ids_reason, attention_mask_reason, news_label, prop_label, prop_score_label) in enumerate(tqdm(train_loader)):
        # Fake News Detection output, Propaganda Classification output, Propaganda Score Regression output
        news_outputs, prop_outputs, prop_score_outputs, _ = model(input_ids_news=input_ids_original,
                                           attention_mask_news=attention_mask_original,
                                           input_ids_prop=input_ids_prop, attention_mask_prop=attention_mask_prop,
                                           reason=input_ids_reason, attention_mask_reason=attention_mask_reason, propaganda_score=prop_score_label)

        # In paper 4.Method, Multi-Task Learning
        # Cross Entropy loss for Fake News Detection
        current_loss_news = news_critera(news_outputs.view(-1), news_label)
        training_news_loss += current_loss_news.item()
        # Cross Entropy loss for Propaganda Classification
        current_loss_prop = prop_critera(prop_outputs, prop_label)
        training_prop_loss += current_loss_prop.item()
        # Mean Square Error loss for Propaganda Score Regression
        current_loss_prop_score = prop_score_critera(prop_score_outputs.view(-1), prop_score_label)
        training_prop_score_loss += current_loss_prop_score.item()

        # In paper 4.Method, we set α1 = 0.33,α2 = 1,α3 = 1
        total_loss = (args.news_weight * current_loss_news) + (args.prop_weight * current_loss_prop) + (args.prop_score_weight * current_loss_prop_score)
        total_loss.backward()
        training_loss += total_loss.item()

        if (batch_idx + 1) % args.accumulate_step == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()

    training_news_losses.append(training_news_loss)
    training_prop_losses.append(training_prop_loss)
    training_prop_score_losses.append(training_prop_score_loss)
    training_losses.append(training_loss)
    print(f"Epoch {epoch}: Weights of News = {args.news_weight:.2f}, Weights of Propaganda = {args.prop_weight:.2f}, Weights of Propaganda Score = {args.prop_score_weight:.2f}")
    print(f"Training Loss: {training_loss:4f}, Training News Loss: {training_news_loss:4f}, Training Prop Loss: {training_prop_loss:4f}, Training Prop Score Loss: {training_prop_score_loss:4f}")
    # Train the last batch
    if batch_num % args.accumulate_step != 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 5.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad()

    # Validation
    with torch.no_grad():
        model.eval()
        dev_news_outputs = []
        dev_prop_outputs = []
        dev_prop_score_outputs = []
        dev_news_labels = []
        dev_prop_labels = []
        dev_prop_score_labels = []

        for batch_idx, (input_ids_original, attention_mask_original, input_ids_prop, attention_mask_prop, input_ids_reason, attention_mask_reason, news_label, prop_label, prop_score_label) in enumerate(dev_loader):
            news_outputs, prop_outputs, prop_score_outputs, _ = model(input_ids_news=input_ids_original, attention_mask_news=attention_mask_original,
                                               input_ids_prop=input_ids_prop, attention_mask_prop=attention_mask_prop,
                                               reason=input_ids_reason, attention_mask_reason=attention_mask_reason, propaganda_score=prop_score_label)
            dev_news_outputs.append(news_outputs)
            dev_prop_outputs.append(prop_outputs)
            dev_prop_score_outputs.append(prop_score_outputs)
            dev_news_labels.append(news_label)
            dev_prop_labels.append(prop_label)
            dev_prop_score_labels.append(prop_score_label)

        dev_news_outputs = torch.cat(dev_news_outputs, dim=0)
        dev_prop_outputs = torch.cat(dev_prop_outputs, dim=0)
        dev_prop_score_outputs = torch.cat(dev_prop_score_outputs, dim=0)
        dev_news_labels = torch.cat(dev_news_labels, dim=0)
        dev_prop_labels = torch.cat(dev_prop_labels, dim=0)
        dev_prop_score_labels = torch.cat(dev_prop_score_labels, dim=0)

        # Calculate the accuracy and f1-score of fake news detection task
        dev_news_outputs_bool = dev_news_outputs > 0.5
        dev_news_accuracy = (dev_news_outputs_bool == dev_news_labels.bool()).sum().item() / len(dev_news_labels)
        dev_news_auc = roc_auc_score(dev_news_labels.cpu().numpy(), dev_news_outputs.detach().cpu().numpy())
        dev_news_f1 = f1_score(dev_news_labels.cpu().numpy(), [1 if o > 0.5 else 0 for o in dev_news_outputs])

        # Calculate the accuracy and f1-score of propaganda classification task
        _, predicted_labels = torch.max(dev_prop_outputs, 1)
        dev_prop_accuracy = (predicted_labels == dev_prop_labels).sum().item() / dev_prop_labels.size(0)
        dev_prop_f1 = f1_score(dev_prop_labels.cpu().numpy(), predicted_labels.cpu().numpy(), average='weighted')

        # Calculate propaganda level score error
        dev_prop_score_outputs = dev_prop_score_outputs.squeeze(dim=1)
        dev_prop_score_mse = mean_squared_error(dev_prop_score_labels.cpu().numpy(),
                                                     dev_prop_score_outputs.cpu().numpy())

        print(f"Dev  News Task - AUC: {dev_news_auc}, F1: {dev_news_f1}")
        print(f"Dev Propaganda News Task - Accuracy {dev_prop_accuracy}, F1-score: {dev_prop_f1}")
        print(f"Dev Propaganda Score Task - MSE {dev_prop_score_mse}")


        dev_news_aucs.append(dev_news_auc)
        dev_prop_accuracies.append(dev_prop_accuracy)
        dev_prop_f1_scores.append(dev_prop_f1)
        dev_prop_score_mse.append(dev_prop_score_mse)

        if dev_news_auc > best_dev_news_auc:
            model_path = os.path.join(output_dir, 'Epoch_' + str(epoch) + '_news_best.pt')
            print(f"Saving to {model_path}")
            best_dev_news_auc = dev_news_auc
            torch.save(state, model_path)
        else:
            model_path = os.path.join(output_dir, 'Epoch_' + str(epoch) + '_news.pt')
            print(f"Saving to {model_path}")
            torch.save(state, model_path)
        print(f"Epoch {epoch} DEV AUC: {dev_news_auc:.2f}. Best DEV AUC: {best_dev_news_auc:.2f}.")


# Draw training loss
epochs = list(range(1, len(training_losses) + 1))
plt.figure(figsize=(12, 8))
plt.plot(epochs, training_losses, label='Total Training Loss', color='black')
plt.plot(epochs, training_news_losses, label='News Task Loss', color='blue', linestyle='--')
plt.plot(epochs, training_prop_losses, label='Propaganda Task Loss', color='red', linestyle='--')
plt.plot(epochs, training_prop_score_losses, label='Propaganda Score Task Loss', color='green', linestyle='--')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
loss_chart_path = os.path.join(output_dir, 'loss_over_epochs.png')
plt.savefig(loss_chart_path)
plt.close()

# Draw evaluation metrics
plt.figure(figsize=(10, 7))
plt.plot(dev_news_aucs, label='News Task AUC', color='blue')
plt.plot(dev_prop_accuracies, label='Propaganda Task Accuracy', color='red')
plt.plot(dev_prop_f1_scores, label='Propaganda Task F1 Score', color='green')
plt.title('Validation Metrics over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.legend()
validation_chart_path = os.path.join(output_dir, 'validation_metrics.png')
plt.savefig(validation_chart_path)
plt.close()


# Data preparation
data = {
    "training_losses": training_losses,
    "training_news_losses": training_news_losses,
    "training_prop_losses": training_prop_losses,
    "training_prop_score_losses": training_prop_score_losses,
    "dev_news_aucs": dev_news_aucs,
    "dev_prop_accuracies": dev_prop_accuracies,
    "dev_prop_f1_scores": dev_prop_f1_scores,
    "dev_prop_score_mse": dev_prop_score_mse
}

df = pd.DataFrame(data)
loss_csv_path = os.path.join(output_dir, 'loss.csv')
df.to_csv(loss_csv_path, index=False)