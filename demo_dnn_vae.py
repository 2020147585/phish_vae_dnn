import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from vae import create_vae
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import argparse
import os
import wandb
from wandb.integration.keras import WandbCallback


parser = argparse.ArgumentParser(description="VAE-DNN phishing detection")
parser.add_argument('--dataset', type=str, choices=['iscx', 'urlset', 'dephides'], default='iscx')
parser.add_argument('--beta', type=float, default=2.0)
parser.add_argument('--warmup_epochs', type=int, default=30)
parser.add_argument('--latent_dim', type=int, default=32)
parser.add_argument('--vae_epochs', type=int, default=100)
parser.add_argument('--clf_epochs', type=int, default=50)
parser.add_argument('--clf_batch_size', type=int, default=64)
parser.add_argument('--clf_val_split', type=float, default=0.1)
args = parser.parse_args()

dataset = args.dataset
beta = args.beta
warmup_epochs = args.warmup_epochs
latent_dim = args.latent_dim
vae_epochs = args.vae_epochs

if dataset == 'dephides':
    df = pd.read_csv("val_features.csv")
    X = df.drop(columns=["class"])
    y = df["class"].astype(int)
    model_type = "alt"
    beta = args.beta
    vae_epochs = args.vae_epochs

elif dataset == 'urlset':
    df = pd.read_csv("urlset_cleaned.csv", encoding="ISO-8859-1")
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)
    model_type = "alt"
    beta = args.beta
    vae_epochs = args.vae_epochs

else:  # iscx
    df = pd.read_csv("Phishing_Infogain.csv")
    X = df.drop(columns=["class"])
    y = df["class"].map({"benign": 0, "phishing": 1}).astype(int)
    model_type = "iscx"
    beta = args.beta
    vae_epochs = args.vae_epochs

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

wandb.init(project="vae-dnn-phishing", config={
    "dataset": dataset,
    "beta": beta,
    "warmup_epochs": warmup_epochs,
    "latent_dim": latent_dim,
    "vae_epochs": vae_epochs,
    "clf_epochs": args.clf_epochs,
    "clf_batch_size": args.clf_batch_size
})

def create_dnn_classifier(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

print(f"Training DNN classifier on {dataset} dataset:")
classifier = create_dnn_classifier(input_dim=X.shape[1])
classifier.fit(
    x_train, 
    y_train, 
    epochs=args.clf_epochs, 
    batch_size=args.clf_batch_size, 
    validation_split=args.clf_val_split,
    callbacks=[WandbCallback(log_weights=True)]
)


class KLLossWarmUp(tf.keras.callbacks.Callback):
    def __init__(self, vae_model, max_beta, warmup_epochs):
        super().__init__()
        self.vae = vae_model
        self.max_beta = max_beta
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        new_beta = min(self.max_beta, self.max_beta * (epoch + 1) / self.warmup_epochs)
        self.vae.beta = new_beta
        wandb.log({"KL_beta": new_beta})
        print(f"[Warm-up] Epoch {epoch + 1}: KL β = {new_beta:.4f}")


print("Training VAE on all samples:")
vae = create_vae(input_dim=X.shape[1], latent_dim=latent_dim, beta=beta, model_type=model_type)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=lambda y_true, y_pred: 0.0)

x_phish_tr, x_phish_val = train_test_split(x_train, test_size=0.1)
warmup_callback = KLLossWarmUp(vae, max_beta=beta, warmup_epochs=warmup_epochs)
history = vae.fit(
    x_phish_tr, 
    epochs=vae_epochs, 
    batch_size=64, 
    validation_data=(x_phish_val, x_phish_val), 
    callbacks=[warmup_callback, WandbCallback(log_weights=True)]
)


def generate_features(vae, num_samples=200):
    latent_dim = vae.encoder.output[0].shape[1]
    random_latent = np.random.normal(size=(num_samples, latent_dim))
    generated_features = vae.decoder.predict(random_latent)
    return generated_features

def generate_features_by_interpolation(vae, x_train, y_train, alpha_vals, samples_per_alpha):
    z_phish, _, _ = vae.encoder.predict(x_train[y_train == 1])
    z_benign, _, _ = vae.encoder.predict(x_train[y_train == 0])
    mu_phish = np.mean(z_phish, axis=0)
    mu_benign = np.mean(z_benign, axis=0)
    latent_list = []
    for alpha in alpha_vals:
        z_interp = alpha * mu_phish + (1 - alpha) * mu_benign
        z_interp = np.tile(z_interp, (samples_per_alpha, 1)) + np.random.normal(0, 0.2, size=(samples_per_alpha, len(mu_phish)))
        latent_list.append(z_interp)
    z_total = np.vstack(latent_list)
    generated_features = vae.decoder.predict(z_total)
    return generated_features

if dataset == 'iscx':
    generated_features = generate_features_by_interpolation(vae, x_train, y_train, alpha_vals=[0.2, 0.4, 0.6, 0.8], samples_per_alpha=75)
else:
    generated_features = generate_features(vae, num_samples=300)


def evaluate_generated_features(classifier, features, threshold_high=0.9, threshold_low=0.7):
    predictions = classifier.predict(features).flatten()
    strong_mask = predictions >= threshold_high
    weak_mask = predictions <= threshold_low
    strong = features[strong_mask]
    weak = features[weak_mask]
    strong_conf = predictions[strong_mask]
    weak_conf = predictions[weak_mask]
    return strong, strong_conf, weak, weak_conf, predictions

strong, strong_conf, weak, weak_conf, confidences = evaluate_generated_features(classifier, generated_features)
print(f"Strong confidence (>0.9): {len(strong)} samples")
print(f"Weak confidence (<0.7): {len(weak)} samples")


os.makedirs("results", exist_ok=True)
suffix = f"_{dataset}"

def plot_confidence_histogram(confidences, save_path):
    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=20, color='skyblue', edgecolor='black')
    plt.title('Confidence Distribution of Synthetic Samples')
    plt.xlabel('Prediction Confidence (Phishing Probability)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

plot_confidence_histogram(confidences, save_path=f"results/confidence_histogram{suffix}.png")

def plot_latent_space(vae, X, y, save_path):
    z_mean, _, _ = vae.encoder.predict(X)
    if z_mean.shape[1] > 2:
        z_mean = PCA(n_components=2).fit_transform(z_mean)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter, ticks=[0, 1], label="Class (0: Benign, 1: Phishing)")
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.title("Latent Space Visualization (VAE)")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

plot_latent_space(vae, x_test, y_test, save_path=f"results/latent_space{suffix}.png")

def plot_training_loss(history, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='train_loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val_loss')
    plt.title("VAE Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

plot_training_loss(history, save_path=f"results/training_loss{suffix}.png")

high_confidence = confidences >= 0.95
trusted_count = np.sum(high_confidence)
print(f"Trusted generated features (confidence >= 0.95): {trusted_count} / {len(confidences)}")


wandb.log({
    "strong_conf_samples": len(strong),
    "weak_conf_samples": len(weak),
    "trusted_generated_features": trusted_count
})
wandb.finish()
