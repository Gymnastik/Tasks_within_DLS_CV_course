## These tasks were completed as part of the Deep Learning(a bit of ML and mostly CV) course at [Deep Learning School (MIPT)](https://www.dlschool.org/)
### Main framework: PyTorch
#### Each topic has a link to the notebook itself as well as a link to the notebook in the google colab (much easier to navigate by the table of contents).
1. [Customer churn prediction. (Telco customer churn dataset) Classical ML](Churn_prediction.ipynb) ([Colab](https://colab.research.google.com/drive/1FH-85LxBdQdW8LxnRp32H20pWKSQoYIt?usp=sharing))
   * Data preprocessing, filling missing values
   * Exploratory analysis (statistics, correlation matrices, cross information, plots)
   * Feature engineering
   * sklearn pipelines
   * Writing custom transformers for preprocessing in sklearn
   * Oversampling (SMOTE, ADASYN, SMOTEENN, using imbalanced learn library)
   * ImbLearn pipelines
   * Writing a function to easily compare metrics on different models
   * Use GridSearch to find hyperparameters
   * Gradient boosting (XGB, Catboost)
   * Stacking 7 different models with logistic regression as meta-algorithm.
   
3. [Classifying Simpsons characters (kaggle competition)](Classification_of_Simpsons_series_characters.ipynb)
   ([Colab](https://colab.research.google.com/drive/1NFBKi9QqfwxVN2pJE8rC4UUT_BXVXuBT?usp=sharing))
   * Transfer learning
   * Augmentations
   * Weighted Sampling
   * Ensemble of 3 models
   * Custom dataset class with extended capabilities
   * Wrapper for the basic Subset, with the possibility to apply a different augmentations to each subset.
   
3. [Medical images segmentation](Segmentation_of_dermatoscopic_images.ipynb) ([Colab](https://colab.research.google.com/drive/1NR6dmXTBELhtjWTmLMtOL9GoQFCKZ6Vc?usp=sharing))
   * Custom Dataset and Subset classes with additional functionality.
   * Augmentations based on Albumentations library.
   * Two variants of SegNet (single class and modular, where each block is implemented as a separate class)
   * Two variants of U-Net (with MaxPooling and Upsampling, and with Conv and transposed convolutions stacks with stride 2)
   * Train and validation function with additional functionality (visualization, sheduler, etc.)
   * Writing IoU from scratch as a metric
   * Writing simple loss functions from scratch: BCE loss, DICE loss, Focal loss
   * Write custom segmentation losses based on arxiv.org articles:<br>
     TotalVariation loss, Tversky loss, Lovasz-Hinge loss, Structural Similarity Loss (SSL)
   * Comparison of all models and losses and general conclusions for the whole experiment.
   
4. [Autoencoders](Autoencoders.ipynb) 
   ([Colab](https://colab.research.google.com/drive/1A1zU22z4iNPuzuNLBgOyOXO_WwNxC6dZ?usp=sharing))
   * Data assembly and preprocessing function. (downloading, transforms, sorting, additional attributes)
   * Implementation of the classic Autoencoder in two variants (linear layers, convolutions)
   * Train and validation function with additional functionality (visualization, sheduler, etc.)
   * Random image generation
   * Vector arithmetic in latent space. "Making people in photos smile."
   * Implementing Variational Autoencoder in two variants (linear layers, convolutions)
   * Writing a composite loss for VAE (KL-divergence + log-likelihood)
   * Generate random digits (VAE trained on MNIST dataset)
   * Conditional VAE implementation
   * Generate digits digits with class selection (CVAE trained on MNIST dataset)
   * Comparison of distributions in latent space (reduce dimensionality to 2 by TSNE).
   * Denoiser implementation based on VAE.
   * Implement VAE-based face recognition (look for the closest ones in latent space using cosine similarity)
   
5. [Generative adversarial networks](GANs.ipynb) ([Colab](https://colab.research.google.com/drive/1JCd4wBrm6I2JA8SE5EB8j-_FUrKAEA9A?usp=sharing))
   * Custom Dataset class with additional functionality (insert augmentations into class)
   * Wapper for standard Dataloader with return of batches to the selected device
   * DCGAN architecture implementation
   * GAN-hacks (add label smoothing and Gaussian noise to improve learning stability)
   * Train function with additional features (schedulers, smoothing factor, etc.)
   * Face generation
   * Evaluation of generated images quality by Leave One Out 1NN Classifier accuracy
   * Comparison of distributions of fake and real images (reduce dimensionality to 2 by TSNE)
