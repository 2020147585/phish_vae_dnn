This project implements a phishing website detection model using a Variational Autoencoder (VAE) combined with a Deep Neural Network (DNN) classifier.  
It extracts URL-based features, compresses them into a latent representation using VAE, generates synthetic samples through feature interpolation, and finally classifies phishing vs. benign URLs using a DNN.  

The model supports three different datasets:  
Phishing_Infogain.csv – Original feature-based dataset on ISCX-URL-2016  
urlset_cleaned.csv – URL-based dataset(cleaned)  
val_features.csv – Raw URL text data converted into features of DEPHIDES dataset(val.text)  

Variational Autoencoder (VAE) with:  
KL warm-up strategy  
Beta-VAE adjustable KL loss weight  
Latent space visualization using PCA  
Training loss curve plotting  

Feature interpolation  

Deep Neural Network (DNN) classifier for phishing detection  
Automatic dataset compatibility and input dimension detection  
Output visualizations:  
Latent space (2D projection)  
Prediction confidence histogram  
Training loss curves  

Install Dependencies:  
pip install -r requirements.txt  

Run the main program (automatically detects feature dimensions):  
python demo_dnn_vae.py  

output:  
plot_training_loss.png – Training loss curve of VAE  
vae_latent_space.png – 2D latent space visualization  
plot_confidence_histogram.png – Confidence distribution of predictions  
Accuracy  
confidence(strong or weak)  


Hyperparameters  
You can adjust these parameters in the code:  
latent_dim – Latent space size of VAE  
beta – KL divergence weight (Beta-VAE)  
epochs, batch_size – Training iterations and batch size  
warmup_epochs – Number of epochs for KL warm-up  


!!!  
Currently, due to possible issues with hyperparameter tuning or data quality,   
the output results for **DEPHIDES (val\_features.csv)** have not yet reached   
the desired success level. Work is ongoing to make improvements and conduct further testing.  
