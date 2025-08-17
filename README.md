# VAE-DNN Based Phishing Website Detection

This project implements a Variational Autoencoder (VAE) combined with a
Deep Neural Network (DNN) classifier for phishing website detection
using TensorFlow/Keras.  
It supports Weights & Biases (W&B) for training visualization and
command-line arguments (argparse) for flexible configuration.  
**New features include: automatic best-threshold search, strong-confidence analysis, and extended metric reporting.**

---

## Features

- TensorFlow/Keras implementation of VAE-DNN  
- Supports multiple datasets: ISCX-URL-2016, URLSet, Dephides  
- Two encoder/decoder architectures selectable via command-line  
- β-VAE with configurable KL-loss warm-up  
- Synthetic feature generation for data augmentation   
- **Automatic threshold scanning (optimize for Accuracy / F1 / Precision / Recall)**  
- **Strong confidence analysis (auto-detect highly confident positive/negative predictions)**  
- Full training visualization with W&B  
- Saves training metrics, JSON reports, and final figures  

---

## Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/2020147485/phish_vae_dnn.git
cd phish_vae_dnn
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Login to W&B (Optional)
```bash
wandb login
```
> If you don't have an account, choose "Create a W&B account" when prompted.

---

## How to Run

```bash
python demo_dnn_vae.py         --dataset iscx         --beta 2.0         --warmup_epochs 30         --latent_dim 32         --vae_epochs 100         --clf_epochs 50         --clf_batch_size 64        --clf_input concat  --optimize_for f1
```

### Command-line Arguments
| Parameter         | Default | Description |
|-------------------|---------|-------------|
| `--dataset`        | iscx  | Dataset type: `urlset`, `iscx`, `dephides` |
| `--beta`           | 2.0      | β-VAE KL divergence weight |
| `--warmup_epochs`  | 30      | Warm-up epochs for β scheduling |
| `--latent_dim`     | 32   | Latent space dimension for VAE |
| `--vae_epochs`     | 100       | Number of VAE training epochs |
| `--clf_epochs`     | 50    | Number of classifier training epochs |
| `--clf_batch_size` | 64      | Classifier training batch size |
| `--clf_val_split`  | 0.1     | Validation split for classifier |
| `--optimize_for`   | f1     | Metric for threshold search: `accuracy`, `f1`, `precision`, `recall` |
| `--clf_input`      | contact     | DNN input: x= Original feature; z=VAEs z_mean; concat= concatenate X and z_mean |


**Recommended parameter sets:**  
- **ISCX**: β=2.0 , warmup=30, VAE=100, CLF=30  
- **URLSet**: β=1.0 , warmup=50, VAE=150, CLF=50  
- **Dephides**: β=1.5 , warmup=90, VAE=150, CLF=50  

---

## Training Results

### Reported Metrics
For each run, the following metrics are saved into JSON:
- **Overall metrics:** Accuracy, Precision, Recall, F1, False Positive Rate  
- **Threshold scanning:** Best threshold found (optimize for selected metric)  
- **Efficiency metrics:** Inference time per instance, memory usage, model size  
- **Strong-confidence metrics:**  
  - Strong positive threshold & count  
  - Strong negative threshold & count  
  - Coverage of strong-confidence samples  
  - Accuracy on strong positive and strong negative subsets  

Example:
```json
{
  "accuracy": 0.8570,
  "precision": 0.8151,
  "recall": 0.8788,
  "f1": 0.8458,
  "false_positive_rate": 0.1605,
  "inference_time_ms_per_instance": 168.3,
  "avg_mem_mb_single_url": 0.0052,
  "max_mem_mb_single_url": 0.1953,
  "model_size_mb": 0.1801,
  "best_threshold": 0.38,
  "optimize_for": "f1",
  "strong_pos_threshold": 0.9,
  "strong_neg_threshold": 0.1,
  "strong_pos_count": 4591,
  "strong_neg_count": 7331,
  "strong_conf_total": 11922,
  "strong_conf_coverage": 0.5699,
  "strong_pos_accuracy": 0.9695,
  "strong_neg_accuracy": 0.9690
}
```

---

## Output Files
- `encoder_dataset.h5` → Trained VAE weights  
- `classifier_dataset.h5` → Trained DNN classifier weights  
- `vae_dnn_metrics_*.json` → Full evaluation report with thresholds and strong-confidence metrics  

---

## W&B Dashboard
You can view detailed logs and training visualizations on W&B.

---

## Known Limitations / Future Improvements

- Latent space visualization currently PCA only → Future: add t-SNE or UMAP  
- Synthetic data quality depends on β and warm-up → Future: Auto-tune these parameters  
- Classifier and VAE trained sequentially → Future: joint end-to-end training  
- Hyperparameters tuned manually → Future: integrate Optuna for optimization  
- Tested only on three datasets; ISCX works best, while URLSet (PhishStorm) and Dephides suffer from noisy preprocessing. Future: more robust feature cleaning and dataset expansion  
