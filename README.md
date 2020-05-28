# latent-space-visualization
Builds from Dr. Krenn's selfies repository here: https://github.com/aspuru-guzik-group/selfies/blob/master/VariationalAutoEncoder_with_SELFIES/chemistryVAE.py

Added functionalities:
- Save and load VAE models
- Visualize lower-dimensional structure of the latent space with common tools such as PCA and t-SNE
- K-means clustering to identify molecules in each cluster of t-SNE plot
- Switch from training to evaluating the model and plotting PCA and t-SNE quickly
- Plot epochs vs loss and epochs vs reconstruction quality
