VAE-DNN Based Phishing Website Detection

This project implements a Variational Autoencoder (VAE) combined with a
Deep Neural Network (DNN) classifier for phishing website detection
using TensorFlow/Keras.
It supports Weights & Biases (W&B) for training visualization and
command-line arguments (argparse) for flexible configuration.

------------------------------------------------------------------------

Features

-    TensorFlow/Keras implementation of VAE-DNN
-    Supports multiple datasets: ISCX-URL-2016, URLSet(phishstorm), Dephides
-    Two encoder/decoder architectures selectable via command-line
-    β-VAE with configurable KL-loss warm-up
-    Synthetic feature generation for data augmentation
-    Latent space visualization with PCA
-    Confidence histogram and trusted sample metrics
-    Full training visualization with W&B
-    Saves training metrics, plots, and final report

------------------------------------------------------------------------

 Environment Setup

1. Clone Repository

    git clone https://github.com/2020147585/phish_vae_dnn.git
    cd phish_vae_dnn

2. Install Dependencies

    pip install -r requirements.txt

3. Login to W&B (Optional)

    wandb login

------------------------------------------------------------------------

🚀 How to Run

    python demo_dnn_vae.py \
        --dataset iscx \
        --beta 2.0 \
        --warmup_epochs 30 \
        --latent_dim 32 \
        --vae_epochs 100 \
        --clf_epochs 50 \
        --clf_batch_size 64

Command-line Arguments

  -----------------------------------------------------------------------
  Parameter                        Default         Description
  -------------------------------- --------------- ----------------------
  --dataset                        iscx            Dataset type: iscx,
                                                   urlset, dephides

  --beta                           2.0             β-VAE KL divergence
                                                   weight

  --warmup_epochs                  30              Warm-up epochs for β
                                                   scheduling

  --latent_dim                     32              Latent space dimension
                                                   for VAE

  --vae_epochs                     100             Number of VAE training
                                                   epochs

  --clf_epochs                     50              Number of classifier
                                                   training epochs

  --clf_batch_size                 64              Classifier training
                                                   batch size

  --clf_val_split                  0.1             Validation split for
                                                   classifier
  -----------------------------------------------------------------------

------------------------------------------------------------------------

 Training Results

 
 Metrics (Accuracy, loss, reconstruction_loss, etc..)


 Latent Space Visualization



 Confidence Histogram



------------------------------------------------------------------------

 Final Report

    Trusted generated features (confidence >= 0.95): XX / Total
    Strong confidence (>0.9): XX
    Weak confidence (<0.7): XX

------------------------------------------------------------------------

 Output Files

-   training_loss_*.png → VAE training loss curve
-   latent_space_*.png → Latent space visualization via PCA
-   confidence_histogram_*.png → Confidence histogram of synthetic
    samples
-   vae_model_weights.h5 → Trained VAE weights
-   classifier_model_weights.h5 → Trained DNN classifier weights

------------------------------------------------------------------------

 W&B Dashboard

You can view detailed logs and training visualizations on W&B

------------------------------------------------------------------------


 Known Limitations / Future Improvements

-    Latent space visualization currently PCA only → Future: add t-SNE
    or UMAP
-    Synthetic data quality depends on β and warm-up → Future:
    Auto-tune these parameters
-    Classifier and VAE trained sequentially → Future: joint
    end-to-end training
-    Hyperparameters tuned manually → Future: integrate Optuna for
    optimization
-    Tested only on three datasets, moreover, URLSet(phishstorm) and Dephises did not perform well in the initial data cleaning stage. → Expand to more phishing datasets and well clean for robustness


