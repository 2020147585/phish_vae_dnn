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

#dephides
'''
df = pd.read_csv("val_features.csv")
X = df.drop(columns=["class"])
y = df["class"].astype(int)
print(df.shape)
print(df.head())
'''
#ISCX-URL-2016

df = pd.read_csv("Phishing_Infogain.csv")
X = df.drop(columns=["class"])
y = df["class"].map({"benign": 0, "phishing": 1}).astype(int)

#urlset
'''
df = pd.read_csv("urlset_cleaned.csv", encoding="ISO-8859-1")
X = df.drop(columns=["label"])
y = df["label"].astype(int)
print(df.head())
print(X.shape, y.shape)
'''


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


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

print("Training DNN classifier:")
classifier = create_dnn_classifier(input_dim=X.shape[1])

#dephides
'''
classifier.fit(x_train, y_train, epochs=50, batch_size=64, validation_split=0.1)
'''

#ISCX-URL-2016 use

classifier.fit(x_train, y_train, epochs=30, batch_size=64, validation_split=0.1)

#urlset use
'''
classifier.fit(x_train, y_train, epochs=50, batch_size=64, validation_split=0.1)
'''


print("\n[Evaluating DNN on real test set:]")
y_pred_probs = classifier.predict(x_test).flatten()
y_pred_labels = (y_pred_probs > 0.5).astype(int)
print("Classification Report:")
print(classification_report(y_test, y_pred_labels, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_labels))
print("Accuracy:", accuracy_score(y_test, y_pred_labels))



class KLLossWarmUp(tf.keras.callbacks.Callback):
    def __init__(self, vae_model, max_beta, warmup_epochs):
        super().__init__()
        self.vae = vae_model
        self.max_beta = max_beta
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        new_beta = min(self.max_beta, self.max_beta * (epoch + 1) / self.warmup_epochs)
        self.vae.beta = new_beta
        print(f"[Warm-up] Epoch {epoch + 1}: KL β = {new_beta:.4f}")


print("Training VAE on all samples:")

x_phish = x_train
#dephides
'''
vae = create_vae(input_dim=X.shape[1], latent_dim=32, beta=1.5)
'''

#ISCX-URL-2016 use

vae = create_vae(input_dim=X.shape[1], latent_dim=32, beta=2.0)

#urlset use
'''
vae = create_vae(input_dim=X.shape[1], latent_dim=32, beta=1.0)
'''

vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=lambda y_true, y_pred: 0.0)

x_phish_tr, x_phish_val = train_test_split(x_phish, test_size=0.1)

#dephides
'''
warmup_callback = KLLossWarmUp(vae, max_beta=1.5, warmup_epochs=90)
history = vae.fit(x_phish_tr, epochs=150, batch_size=64, validation_data=(x_phish_val, x_phish_val),callbacks=[warmup_callback])
'''

# ISCX-URL-2016 use

warmup_callback = KLLossWarmUp(vae, max_beta=2.0, warmup_epochs=30)
history = vae.fit(x_phish_tr, epochs=100, batch_size=64, validation_data=(x_phish_val, x_phish_val),callbacks=[warmup_callback])

# urlset use
'''
warmup_callback = KLLossWarmUp(vae, max_beta=2.0, warmup_epochs=50)
history = vae.fit(x_phish_tr, epochs=150, batch_size=64, validation_data=(x_phish_val, x_phish_val),callbacks=[warmup_callback])
'''

#for urlset, dephides use
def generate_features(vae, num_samples=200):
    latent_dim = vae.encoder.output[0].shape[1]
    random_latent = np.random.normal(size=(num_samples, latent_dim))
    generated_features = vae.decoder.predict(random_latent)
    return generated_features

#for ISCX-URL-2016 use
def generate_features_by_interpolation(vae, x_train, y_train, alpha_vals, samples_per_alpha):

    print("Encoding phishing and benign samples...")
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

    print(f"Decoding {z_total.shape[0]} interpolated latent samples...")
    generated_features = vae.decoder.predict(z_total)
    return generated_features



print("Generating synthetic features with VAE:")

#for urlset, dephides use
#generated_features = generate_features(vae, num_samples=300)

#for ISCX-URL-2016 use
generated_features = generate_features_by_interpolation(vae, x_train, y_train, alpha_vals=[0.2, 0.4, 0.6, 0.8], samples_per_alpha=75)



def evaluate_generated_features(classifier, features, threshold_high=0.9, threshold_low=0.7):
    predictions = classifier.predict(features).flatten()
    strong_mask = predictions >= threshold_high
    weak_mask = predictions <= threshold_low
    strong = features[strong_mask]
    weak = features[weak_mask]
    strong_conf = predictions[strong_mask]
    weak_conf = predictions[weak_mask]
    return strong, strong_conf, weak, weak_conf, predictions

print("Evaluating synthetic features:")
strong, strong_conf, weak, weak_conf, confidences = evaluate_generated_features(
    classifier, generated_features
)

print(f"Strong confidence (>0.9): {len(strong)} samples")
print(f"Weak confidence (<0.7): {len(weak)} samples")


def plot_confidence_histogram(confidences, save_path="plot_confidence_histogram.png"):
    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=20, color='skyblue', edgecolor='black')
    plt.title('Confidence Distribution of Synthetic Samples')
    plt.xlabel('Prediction Confidence (Phishing Probability)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(save_path)
plot_confidence_histogram(confidences,save_path="plot_confidence_histogram.png")


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
plot_latent_space(vae, x_test, y_test,save_path="vae_latent_space.png")


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
plot_training_loss(history, save_path="plot_training_loss.png")


high_confidence = confidences >= 0.95
trusted_count = np.sum(high_confidence)
print(f"Trusted generated features (confidence >= 0.95): {trusted_count} / {len(confidences)}")
