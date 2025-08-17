import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import argparse
import os
import gc
import time
import json
import psutil
import wandb
from wandb.integration.keras import WandbCallback
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

parser = argparse.ArgumentParser(description="VAE-DNN phishing detection (coupled, fair threshold)")
parser.add_argument('--dataset', type=str, choices=['iscx', 'urlset', 'dephides'], default='iscx')
parser.add_argument('--beta', type=float, default=2.0)
parser.add_argument('--warmup_epochs', type=int, default=30)
parser.add_argument('--latent_dim', type=int, default=32)
parser.add_argument('--vae_epochs', type=int, default=100)
parser.add_argument('--clf_epochs', type=int, default=50)
parser.add_argument('--clf_batch_size', type=int, default=64)
parser.add_argument('--clf_val_split', type=float, default=0.1)
parser.add_argument('--clf_input', type=str, choices=['x', 'z', 'concat'], default='x',
                    help='DNN input: x= Original feature; z=VAEs z_mean; concat= concatenate X and z_mean')

parser.add_argument('--optimize_for', type=str, choices=['f1','precision','recall','accuracy'], default='f1',
                    help='Which metric should be used to select the threshold on the validation set')
args = parser.parse_args()


dataset = args.dataset
beta = args.beta
warmup_epochs = args.warmup_epochs
latent_dim = args.latent_dim
vae_epochs = args.vae_epochs

if dataset == 'dephides':
    df = pd.read_csv('val_features.csv')
    X = df.drop(columns=['class'])
    y = df['class'].astype(int)
    model_type = 'alt'
elif dataset == 'urlset':
    df = pd.read_csv('urlset_cleaned.csv', encoding='ISO-8859-1')
    X = df.drop(columns=['label'])
    y = df['label'].astype(int)
    model_type = 'alt'
else:  # iscx
    df = pd.read_csv('Phishing_Infogain.csv')
    X = df.drop(columns=['class'])
    y = df['class'].map({'benign': 0, 'phishing': 1}).astype(int)
    model_type = 'iscx'


x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = MinMaxScaler().fit(x_tr)
x_train = scaler.transform(x_tr)
x_test  = scaler.transform(x_te)
y_train, y_test = y_tr, y_te


wandb.init(project='vae-dnn-phishing', config={
    'dataset': dataset,
    'beta': beta,
    'warmup_epochs': warmup_epochs,
    'latent_dim': latent_dim,
    'vae_epochs': vae_epochs,
    'clf_epochs': args.clf_epochs,
    'clf_batch_size': args.clf_batch_size,
    'clf_input': args.clf_input,
    'optimize_for': args.optimize_for,
})


from vae import create_vae

class KLLossWarmUp(tf.keras.callbacks.Callback):
    def __init__(self, vae_model, max_beta, warmup_epochs):
        super().__init__()
        self.vae = vae_model
        self.max_beta = max_beta
        self.warmup_epochs = warmup_epochs
    def on_epoch_begin(self, epoch, logs=None):
        new_beta = min(self.max_beta, self.max_beta * (epoch + 1) / max(1, self.warmup_epochs))
        self.vae.beta = new_beta
        wandb.log({'KL_beta': new_beta})
        print(f"[Warm-up] Epoch {epoch + 1}: KL β = {new_beta:.4f}")

print('Training VAE on all samples:')
vae = create_vae(input_dim=X.shape[1], latent_dim=latent_dim, beta=beta, model_type=model_type)

vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=lambda y_true, y_pred: 0.0)

x_tr2, x_val2 = train_test_split(x_train, test_size=0.1, random_state=42)
warmup_callback = KLLossWarmUp(vae, max_beta=beta, warmup_epochs=warmup_epochs)
_ = vae.fit(
    x_tr2,
    epochs=vae_epochs,
    batch_size=64,
    validation_data=(x_val2, x_val2),
    callbacks=[warmup_callback]  
)


z_mean_tr, _, _ = vae.encoder.predict(x_train, verbose=0)
z_mean_te, _, _ = vae.encoder.predict(x_test,  verbose=0)


if args.clf_input == 'x':
    Xtr_clf, Xte_clf = x_train, x_test
elif args.clf_input == 'z':
    Xtr_clf, Xte_clf = z_mean_tr, z_mean_te
else:  # concat
    Xtr_clf = np.concatenate([x_train, z_mean_tr], axis=1)
    Xte_clf = np.concatenate([x_test,  z_mean_te], axis=1)


def create_dnn_classifier(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

print(f"Training DNN classifier on {dataset} dataset (clf_input={args.clf_input}):")
Xtr_tr, X_val, ytr_tr, y_val = train_test_split(
    Xtr_clf, y_train, test_size=args.clf_val_split, random_state=42, stratify=y_train
)

classifier = create_dnn_classifier(input_dim=Xtr_clf.shape[1])
_ = classifier.fit(
    Xtr_tr, ytr_tr,
    epochs=args.clf_epochs,
    batch_size=args.clf_batch_size,
    validation_data=(X_val, y_val),
    callbacks=[WandbCallback(log_weights=False, log_gradients=False)] 
)



def _ensure_binary(y):
    y = np.asarray(y).reshape(-1)
    uniq = set(np.unique(y))
    if uniq <= {0, 1}:
        return y.astype(int)
    mapping = {'benign': 0, 'normal': 0, 'legit': 0, 'phish': 1, 'phishing': 1, 'malicious': 1}
    out = []
    for v in y:
        s = str(v).lower()
        if s in mapping:
            out.append(mapping[s])
        else:
            try:
                iv = int(v)
                out.append(1 if iv == max(uniq) else 0)
            except Exception:
                out.append(0)
    return np.array(out, dtype=int)

def metrics_at_threshold(y_true, y_prob, th):
    y_true = _ensure_binary(y_true)
    y_pred = (np.asarray(y_prob).reshape(-1) >= th).astype(int)
    acc  = float((y_pred == y_true).mean())
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    fpr = float(fp) / float(fp + tn + 1e-12)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'false_positive_rate': fpr}

def scan_thresholds_on_val(y_val, yprob_val, optimize_for='f1', num_points=91):
    ths = np.linspace(0.05, 0.95, num_points)
    best_th, best_val, best_metrics = 0.5, -1.0, None
    for th in ths:
        m = metrics_at_threshold(y_val, yprob_val, th)
        score = m[optimize_for]
        if score > best_val:
            best_val, best_th, best_metrics = score, float(th), m
    return best_th, best_metrics


# First, select the threshold on the validation set
yprob_val = classifier.predict(X_val, verbose=0).reshape(-1)
best_th, _ = scan_thresholds_on_val(y_val, yprob_val, optimize_for=args.optimize_for)

# Evaluate on the test set using the locking threshold
y_true   = y_test.values if hasattr(y_test, 'values') else y_test
yprob_te = classifier.predict(Xte_clf, verbose=0).reshape(-1)
metrics_cls = metrics_at_threshold(y_true, yprob_te, best_th)
chosen_threshold = best_th



def predict_one_fn(x_row):
    x1 = x_row.reshape(1, -1)
    if args.clf_input == 'x':
        X_in = x1
    elif args.clf_input == 'z':
        z_mean, _, _ = vae.encoder.predict(x1, verbose=0)
        X_in = z_mean
    else:  # conca
        z_mean, _, _ = vae.encoder.predict(x1, verbose=0)
        X_in = np.concatenate([x1, z_mean], axis=1)
    return classifier.predict(X_in, verbose=0)


def measure_single_infer_with_pipeline(predict_one_fn, X, n_samples=128, warmup=3, samples_per_pred=5, sample_interval=0.002):
    proc = psutil.Process(os.getpid())
    k = min(n_samples, len(X))

    for i in range(min(warmup, len(X))):
        _ = predict_one_fn(X[i])
    times_ms, deltas_mb = [], []
    for i in range(k):
        gc.collect()
        baseline = proc.memory_info().rss
        t0 = time.perf_counter()
        _ = predict_one_fn(X[i])
        peak = baseline
        for _ in range(samples_per_pred):
            m = proc.memory_info().rss
            if m > peak:
                peak = m
            time.sleep(sample_interval)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt_ms)
        deltas_mb.append(max(0.0, (peak - baseline) / (1024 ** 2)))
    return float(np.mean(times_ms)), float(np.mean(deltas_mb)), float(np.max(deltas_mb))

infer_ms, mem_avg_mb, mem_max_mb = measure_single_infer_with_pipeline(
    predict_one_fn, x_test, n_samples=128, warmup=3
)



def get_model_size_mb(path_or_dir):
    if os.path.isfile(path_or_dir):
        return os.path.getsize(path_or_dir) / (1024 ** 2)
    total = 0
    for root, _, files in os.walk(path_or_dir):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 ** 2)

os.makedirs('results', exist_ok=True)
clf_path = f'results/classifier_{dataset}_{args.clf_input}.h5'
classifier.save(clf_path)

model_size_mb = get_model_size_mb(clf_path)
if args.clf_input in ('z', 'concat'):
    enc_path = f'results/encoder_{dataset}.h5'
    vae.encoder.save(enc_path)
    model_size_mb += get_model_size_mb(enc_path)


wandb.run.summary['best_threshold'] = float(chosen_threshold)
wandb.run.summary['optimize_for']  = args.optimize_for
wandb.run.summary['dataset'] = dataset

wandb_metrics = {
    'accuracy(%)': metrics_cls['accuracy'],
    'precision(%)': metrics_cls['precision'],
    'recall(%)': metrics_cls['recall'],
    'f1(%)': metrics_cls['f1'],
    'false_positive_rate(%)': metrics_cls['false_positive_rate'],
    'inference_time_ms_per_instance(ms/sample)': infer_ms,
    'avg_mem_mb_single_url(MB/URL)': mem_avg_mb,
    'max_mem_mb_single_url(MB/URL)': mem_max_mb,
    'model_size_mb(MB)': model_size_mb,
}
wandb.log(wandb_metrics)


strong_pos_th = 0.9
strong_neg_th = 0.1

mask_strong_pos = yprob_te >= strong_pos_th
mask_strong_neg = yprob_te <= strong_neg_th

acc_strong_pos = (y_true[mask_strong_pos] == (yprob_te[mask_strong_pos] >= chosen_threshold)).mean() if mask_strong_pos.any() else None
acc_strong_neg = (y_true[mask_strong_neg] == (yprob_te[mask_strong_neg] >= chosen_threshold)).mean() if mask_strong_neg.any() else None

strong_pos_count = int(mask_strong_pos.sum())
strong_neg_count = int(mask_strong_neg.sum())

strong_conf_total = strong_pos_count + strong_neg_count
strong_conf_coverage = strong_conf_total / len(y_true)

strong_conf_metrics = {
    "strong_pos_threshold": strong_pos_th,
    "strong_neg_threshold": strong_neg_th,
    "strong_pos_count": int(mask_strong_pos.sum()),
    "strong_neg_count": int(mask_strong_neg.sum()),
    "strong_conf_total": strong_conf_total,         
    "strong_conf_coverage": float(strong_conf_coverage), 
    "strong_pos_accuracy(%)": float(acc_strong_pos) if acc_strong_pos is not None else None,
    "strong_neg_accuracy(%)": float(acc_strong_neg) if acc_strong_neg is not None else None
}

full_metrics = {**wandb_metrics,
    'best_threshold': float(chosen_threshold),
    'optimize_for': args.optimize_for,
    **strong_conf_metrics
}

with open(f'results/vae_dnn_metrics_{dataset}_{args.clf_input}.json', 'w', encoding='utf-8') as f:
    json.dump(full_metrics, f, indent=2, ensure_ascii=False)
print(f"[Saved] results/vae_dnn_metrics_{dataset}_{args.clf_input}.json")

wandb.finish()
