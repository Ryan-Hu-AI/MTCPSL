# 📰 MTCPSL: Multi-Task Counterpropaganda Semantic Learning for Detecting Fake News

This repository contains a PyTorch-based evaluation script for testing a **multi-task model** that jointly performs:

* Fake news detection
* Propaganda type classification
* Propaganda level score regression

## 📦 Features

* Support for multi-task evaluation using shared RoBERTa-based encoder.
* Prediction of true/false news, propaganda type (e.g., Loaded Language, Exaggeration, Appeal to Fear), and a numeric propaganda level score.
* Results saved to both JSON and MongoDB.
* Automatic generation of evaluation metrics: Accuracy, F1-score, ROC-AUC, and MSE.

---

## 🧩 Project Structure

* `my_models.py`: Defines the MTCPSL model class.
* `my_dataset_process.py`: Defines the custom dataset class for loading evaluation data.
* `main_eval.py`: Main script to load model, evaluate dataset, and save results (your current script).
* `output/`: Directory for model checkpoints and generated results.
* `PROPANEWS_SPRS Dataset/`: Contains JSONL-formatted test datasets.

---

## 🚀 Quick Start

### 1. Environment Setup

Install required packages:

| Package      | Version       |
| ------------ | ------------- |
| Python       | 3.10          |
| PyTorch      | 2.1.2 + cu118 |
| Transformers | 4.26.1        |
| NumPy        | ≥ 1.21        |
| scikit-learn | ≥ 1.0         |
| matplotlib   | ≥ 3.4         |
| pymongo      | ≥ 4.0         |


Ensure MongoDB is installed and running locally at `mongodb://localhost:27017`.

### 2. Run Evaluation

```bash
python RoBERTa Large Based/test_SPRS_MTCPSL_P_Mask_R.py \
    --checkpoint_path ../output/.../Epoch_21_news.pt \
    --test_dataset politifact_SPRS_nonrepeat.jsonl \
    --test_output_name politifact_SPRS_nr
```

You can configure all options via command-line arguments. Below are the key ones:

| Argument             | Description                                                                  |
| -------------------- | ---------------------------------------------------------------------------- |
| `--model_name`       | Pretrained transformer model name (default: `roberta-large`)                 |
| `--checkpoint_path`  | Path to the fine-tuned checkpoint                                            |
| `--test_dataset`     | Filename of the JSONL test data                                              |
| `--test_output_name` | Prefix used for output files and plots                                       |
| `--prop_model`       | Model class name (default: `Multi_Task_CounterPropaganda_Semantic_Learning`) |
| `--database`         | MongoDB database name                                                        |
| `--collection`       | MongoDB collection name for storing evaluation metrics                       |

---

## 📂 Input Format

Each input sample in the JSONL file should contain:

```json
{
  "news": "...",
  "propaganda_summary": "...",
  "reason": "...",
  "propaganda_score": 0.85,
  "label_news": 0 or 1,
  "label_propaganda": "Loaded Language" | "Exaggeration" | ...
}
```

---

## 📊 Output

* AUC plot saved as:

  ```
  output_dir/politifact_SPRS_nr_News_AUC.png
  ```
* JSON output predictions:

  ```json
  {
    "news_output": [...],
    "prop_output": [...],
    "prop_score_output": [...]
  }
  ```
* MongoDB stores evaluation metrics (e.g., Accuracy, F1, MSE) for tracking performance across runs.

---

## 📈 Evaluation Metrics

* **Fake News Detection:**

  * Accuracy, Precision, Recall, F1-score (per class and macro avg)
  * ROC-AUC curve

* **Propaganda Type Classification:**

  * Class-wise precision, recall, F1
  * Macro and weighted average metrics

* **Propaganda Severity Score:**

  * Mean Squared Error (MSE)

---

## 📌 Notes

* Ensure your GPU is available and properly set via `CUDA_VISIBLE_DEVICES`.
* If using a different model class, modify the `--prop_model` argument accordingly.
* Default settings use seed `8` for reproducibility.

---

## 🧪 Example MongoDB Result Schema

```json
{
  "Pretrain_model": "roberta-large",
  "Model": "Multi_Task_CounterPropaganda_Semantic_Learning",
  "News_AUC": 0.84,
  "News_Accuracy": 0.81,
  "Loaded_Language_F1-score": 0.72,
  ...
}
```

---

## 📬 Citation

If you find this project useful, please cite:

> MING-JHE HU, and HUNG-YU KAO, "Multi-Task Counterpropaganda Semantic Learning for Detecting Fake News", 2025.


