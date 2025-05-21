import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import argparse
import os
matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive environments

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', default='../output', type=str, required=True)
parser.add_argument('--output_path', default='../output', type=str, required=True)
parser.add_argument('--random_state', default=42, type=str, required=False)
args = parser.parse_args()

# In Section 5: Experiment - T-SNE Visualization
# Load the embeddings and labels from the JSONL file
embeddings_news = []
embeddings_prop = []
embeddings_reason = []
enhanced_prop_reason = []
enhanced_news_prop_reason = []
news_labels = []
prop_labels = []

file_path = os.path.join(args.output_path, args.file_name)
output_file_name = '.'.join(file_path.split('.')[:-1])

with open(file_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        if 'cls_embeddings_news' in data:
            embeddings_news.append(data['cls_embeddings_news'])
            news_labels.append(data['news_label'])
        if 'cls_embeddings_prop' in data:
            embeddings_prop.append(data['cls_embeddings_prop'])
        if 'cls_embeddings_reason' in data:
            embeddings_reason.append(data['cls_embeddings_reason'])
        if 'counterpropaganda_semantic' in data:
            counterpropaganda_semantic.append(data['counterpropaganda_semantic'])
        if 'amplified_propaganda_news' in data:
            amplified_propaganda_news.append(data['amplified_propaganda_news'])
        if 'prop_label' in data:
            prop_labels.append(data['prop_label'])

# Convert lists to numpy arrays
embeddings_news = np.array(embeddings_news)
embeddings_prop = np.array(embeddings_prop)
embeddings_reason = np.array(embeddings_reason)
enhanced_prop_reason = np.array(enhanced_prop_reason)
enhanced_news_prop_reason = np.array(enhanced_news_prop_reason)
news_labels = np.array(news_labels)
prop_labels = np.array(prop_labels)

# Combine all embeddings into a single array
all_embeddings = np.concatenate((embeddings_news, embeddings_prop, embeddings_reason,
                                 enhanced_prop_reason, enhanced_news_prop_reason), axis=0)

# Create labels for the embeddings
embedding_labels = np.array([0] * len(embeddings_news) +
                            [1] * len(embeddings_prop) +
                            [2] * len(embeddings_reason) +
                            [3] * len(enhanced_prop_reason) +
                            [4] * len(enhanced_news_prop_reason))

# Apply t-SNE to reduce the dimensionality of the embeddings
tsne = TSNE(n_components=2, random_state=int(args.random_state))
embeddings_2d = tsne.fit_transform(all_embeddings)

# Define a custom colormap with distinct colors
cmap = ListedColormap(['red', 'blue', 'green', 'purple', 'orange'])

# Visualize the reduced embeddings
plt.figure(figsize=(12, 10))

# Plot the t-SNE results
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=embedding_labels, cmap=cmap, alpha=0.7)

# Add a legend with custom labels
legend_labels = ['News', 'Propaganda', 'Reason', 'Enhanced Prop-Reason', 'Enhanced News-Prop-Reason']
handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                      markerfacecolor=cmap(i), markersize=10) for i, label in enumerate(legend_labels)]
plt.legend(handles=handles, title='Embedding Types')

plt.title('t-SNE Visualization of News, Propaganda, Reason, and Enhanced Vectors')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

# Save the plot to a file
output_file = output_file_name + '_r' + args.random_state + '_TSNE_5_inputs_2D.png'
plt.savefig(output_file)
print("t-SNE visualization saved to" + output_file)

