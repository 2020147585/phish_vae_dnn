import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from vae import create_vae


def load_and_prepare_data():
    df = pd.read_csv("Phishing_Infogain.csv")
    X = df.drop(columns=["class"])
    y = df["class"].map({"benign": 0, "phishing": 1}).astype(int)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, X.columns


def train_vae(x_train, latent_dim=4, epochs=30, batch_size=64):
    vae = create_vae(input_dim=x_train.shape[1], latent_dim=latent_dim, beta=10.0)
    vae.compile(
    optimizer=keras.optimizers.Adam(),
    loss=lambda y_true, y_pred: 0.0  
)
    x_tr, x_val = train_test_split(x_train, test_size=0.1)
    history = vae.fit(x_tr, epochs=20, batch_size=64, validation_data=(x_val, x_val))
    
    return vae, history




def plot_latent_space(vae, X, y, method="pca", save_path="latent_space.png"):
    print("Encoding full dataset into latent space...")
    z_mean, _, _ = vae.encoder.predict(X)

    if method == "pca" and z_mean.shape[1] > 2:
        print(f"Latent dim = {z_mean.shape[1]}, reducing to 2D via PCA...")
        z_mean = PCA(n_components=2).fit_transform(z_mean)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter, ticks=[0, 1], label="Class (0: Benign, 1: Phishing)")
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.title("Latent Space Visualization (VAE)")
    plt.grid(True)

    
    plt.savefig(save_path)
    print(f"[Saved] Latent space visualization saved to: {save_path}")
    
    print("\nLatent z_mean statistics:")
    phishing_latent = z_mean[y == 1]
    benign_latent = z_mean[y == 0]
    print(f"Phishing latent mean: {np.mean(phishing_latent, axis=0)}")
    print(f"Phishing latent std:  {np.std(phishing_latent, axis=0)}")
    print(f"Benign latent mean:   {np.mean(benign_latent, axis=0)}")
    print(f"Benign latent std:    {np.std(benign_latent, axis=0)}")




def run_tabular_demo():
    print("Loading and preprocessing CSV data...")
    X_scaled, y, feature_names = load_and_prepare_data()

    
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    x_train_phishing = x_train[y_train == 1]

    print(f"Training VAE on phishing-only samples: {x_train_phishing.shape[0]} samples")
    vae, history = train_vae(x_train_phishing, latent_dim=4, epochs=30)
    plot_training_loss(history, save_path="plot_training_loss.png")


    print("Visualizing latent space...")
    plot_latent_space(vae, x_test, y_test,save_path="vae_latent_space.png")

    print("All done.")


def plot_training_loss(history, save_path="plot_training_loss.png"):
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
    print(f"[Saved] Latent space visualization saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    run_tabular_demo()

