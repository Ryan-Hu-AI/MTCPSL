import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class Multi_Task_CounterPropaganda_Semantic_Learning(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert_news = AutoModel.from_pretrained(model_name)# In paper 4.Method, we use pre-trian RoBERTa Large
        self.linear_news = nn.Linear(self.bert_news.config.hidden_size, 1) # As fake news classifier
        self.linear_propaganda = nn.Linear(self.bert_news.config.hidden_size, 3)# As propaganda classifier (Loaded Language, Exaggeration, Appeal to Fear)
        self.linear_prop_score = nn.Linear(self.bert_news.config.hidden_size, 1)# As propaganda score regression

    def forward(self, input_ids_news=None, attention_mask_news=None, input_ids_prop=None, attention_mask_prop=None,
                reason=None, attention_mask_reason=None, propaganda_score=None, return_embeddings=False):
        embeddings = []

        if input_ids_news is not None:
            hidden_states_news = self.bert_news(input_ids=input_ids_news, attention_mask=attention_mask_news)[0]
            cls_embeddings_news = hidden_states_news[:, 0, :]# Get propaganda sentence embedding

            # When need output news embeddings
            if return_embeddings:
                for i in range(cls_embeddings_news.size(0)):
                    embeddings.append({'cls_embeddings_news': cls_embeddings_news[i].detach().cpu().numpy().tolist()})

            if input_ids_prop is not None:
                hidden_states_prop = self.bert_news(input_ids=input_ids_prop, attention_mask=attention_mask_prop)[0]
                cls_embeddings_prop = hidden_states_prop[:, 0, :]# Get reason sentence embedding

                if return_embeddings: # When need output propaganda embeddings
                    for i in range(cls_embeddings_prop.size(0)):
                        embeddings[i]['cls_embeddings_prop'] = cls_embeddings_prop[i].detach().cpu().numpy().tolist()

                if reason is not None:
                    hidden_states_reason = self.bert_news(input_ids=reason, attention_mask=attention_mask_reason)[0]
                    cls_embeddings_reason = hidden_states_reason[:, 0, :]# Get reason sentence embedding

                    if return_embeddings: # When need output reason embeddings
                        for i in range(cls_embeddings_reason.size(0)):
                            embeddings[i]['cls_embeddings_reason'] = cls_embeddings_reason[i].detach().cpu().numpy().tolist()

                    # Synthesize counterpropaganda semantic
                    counterpropaganda_semantic = cls_embeddings_prop - cls_embeddings_reason

                    if return_embeddings: # When need output counterpropaganda embeddings
                        for i in range( counterpropaganda_semantic.size(0)):
                            embeddings[i]['counterpropaganda_semantic'] = counterpropaganda_semantic[i].detach().cpu().numpy().tolist()
                    propaganda_pred = self.linear_propaganda(counterpropaganda_semantic)# propaganda classifier output


                    # Synthesize amplified_propaganda_news
                    amplified_propaganda_news = cls_embeddings_news - counterpropaganda_semantic

                    if return_embeddings: # When need output amplified_propaganda_news embeddings
                        for i in range(amplified_propaganda_news.size(0)):
                            embeddings[i]['amplified_propaganda_news'] = amplified_propaganda_news[i].detach().cpu().numpy().tolist()

                    logits_news = self.linear_news(amplified_propaganda_news)
                    news_pred = torch.sigmoid(logits_news)# fake news classifier output

                    if propaganda_score is not None:
                        logits_prop_score = self.linear_prop_score(counterpropaganda_semantic)
                        prop_score_pred = torch.sigmoid(logits_prop_score)# propaganda score regression output
                        #input news, propaganda, reason, score
                        return news_pred, propaganda_pred, prop_score_pred, embeddings
                    else:  # input news, propaganda, reason (for ablation study)
                        return news_pred, propaganda_pred, embeddings

                else:
                    enhanced_vector_news_prop = cls_embeddings_news - cls_embeddings_prop
                    if return_embeddings:
                        for i in range(enhanced_vector_news_prop.size(0)):
                            embeddings[i]['enhanced_vector_news_prop'] = enhanced_vector_news_prop[i].detach().cpu().numpy().tolist()

                    logits_news = self.linear_news(enhanced_vector_news_prop)
                    news_pred = torch.sigmoid(logits_news)

                    propaganda_pred = self.linear_propaganda(cls_embeddings_prop)

                    if propaganda_score is not None:
                        logits_prop_score = self.linear_prop_score(cls_embeddings_prop)
                        prop_score_pred = torch.sigmoid(logits_prop_score)
                        # input news, propaganda, score (for ablation study)
                        return news_pred, propaganda_pred, prop_score_pred, embeddings
                    else:  # input news, propaganda (for ablation study)
                        return news_pred, propaganda_pred, embeddings
            else:
                if reason is not None:
                    hidden_states_reason = \
                        self.bert_news(input_ids=reason, attention_mask=attention_mask_reason)[0]
                    cls_embeddings_reason = hidden_states_reason[:, 0, :]

                    if return_embeddings:
                        for i in range(cls_embeddings_reason.size(0)):
                            embeddings[i]['cls_embeddings_reason'] = cls_embeddings_reason[i].detach().cpu().numpy().tolist()

                    enhanced_vector_news_reason = cls_embeddings_news + cls_embeddings_reason
                    for i in range(enhanced_vector_news_reason.size(0)):
                        embeddings[i]['enhanced_vector_news_reason'] = enhanced_vector_news_reason[i].detach().cpu().numpy().tolist()

                    logits_news = self.linear_news(enhanced_vector_news_reason)
                    news_pred = torch.sigmoid(logits_news)
                    # input news, reason (for ablation study)
                    return news_pred, embeddings
                else:
                    logits_news = self.linear_news(cls_embeddings_news)
                    news_pred = torch.sigmoid(logits_news)
                    # input news (for ablation study)
                    return news_pred, embeddings
