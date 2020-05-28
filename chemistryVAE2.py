#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time
import numpy as np
import torch
import pandas as pd
import selfies
import yaml
import matplotlib.pyplot as plt
from torch import nn
from random import shuffle

sys.path.append('VAE_dependencies')
from data_loader import multiple_smile_to_hot, multiple_selfies_to_hot, len_selfie, split_selfie, hot_to_selfies
from rdkit.Chem import MolFromSmiles
from rdkit import rdBase
from selfies import decoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances_argmin
from mpl_toolkits.mplot3d import Axes3D
rdBase.DisableLog('rdApp.error')


def _make_dir(directory):
    os.makedirs(directory)

"""
def save_models(encoder, decoder, epoch):
    out_dir = './saved_models/{}'.format(epoch)
    _make_dir(out_dir)
    torch.save(encoder, '{}/E'.format(out_dir))
    torch.save(decoder, '{}/D'.format(out_dir))
"""


class VAE_encode(nn.Module):
    
    def __init__(self, layer_1d, layer_2d, layer_3d, latent_dimension):
        """
        Fully Connected layers to encode molecule to latent space
        """
        super(VAE_encode, self).__init__()
        
        # Reduce dimension upto second last layer of Encoder
        self.encode_nn = nn.Sequential(
            nn.Linear(len_max_molec1Hot, layer_1d),
            nn.ReLU(),
            nn.Linear(layer_1d, layer_2d),
            nn.ReLU(),
            nn.Linear(layer_2d, layer_3d),
			nn.ReLU()
        )
        
        # Latent space mean
        self.encode_mu = nn.Linear(layer_3d, latent_dimension) 
        
        # Latent space variance 
        self.encode_log_var = nn.Linear(layer_3d, latent_dimension)
        
        
    def reparameterize(self, mu, log_var):
        """
        This trick is explained well here:
            https://stats.stackexchange.com/a/16338
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) 
        return eps.mul(std).add_(mu)
    
    
    def forward(self, x):
        """
        Pass throught the Encoder
        """
        # Get results of encoder network
        h1 = self.encode_nn(x)
         
        # latent space
        mu = self.encode_mu(h1)
        log_var = self.encode_log_var(h1)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
        


class VAE_decode(nn.Module):
    
    def __init__(self, latent_dimension, gru_stack_size, gru_neurons_num):
        """
        Through Decoder
        """
        super(VAE_decode, self).__init__()
        self.gru_stack_size = gru_stack_size
        self.gru_neurons_num = gru_neurons_num

        # Simple Decoder
        self.decode_RNN  = nn.GRU(
                input_size  = latent_dimension, 
                hidden_size = gru_neurons_num,
                num_layers  = gru_stack_size,
                batch_first = False)                
        
        self.decode_FC = nn.Sequential(
            nn.Linear(gru_neurons_num, len_alphabet),
        )
    

    def init_hidden(self, batch_size = 1):
        weight = next(self.parameters())
        return weight.new_zeros(self.gru_stack_size, batch_size, self.gru_neurons_num)
                 
                       
    def forward(self, z, hidden):
        """
        A forward pass throught the entire model.
        """
        # Decode
        l1, hidden = self.decode_RNN(z, hidden)    
        decoded = self.decode_FC(l1)        # fully connected layer

        return decoded, hidden



def is_correct_smiles(smiles):    
    """
    Using RDKit to calculate whether molecule is syntactically and semantically valid.
    """
    if smiles == "":
        return 0
    try:
        MolFromSmiles(smiles, sanitize=True)
        return 1
    except Exception:
        return 0


def sample_latent_space(latent_dimension, total_samples): 
    model_encode.eval()
    model_decode.eval()
    
    fancy_latent_point=torch.normal(torch.zeros(latent_dimension),torch.ones(latent_dimension))

    hidden = model_decode.init_hidden() 
    gathered_atoms = []
    for ii in range(len_max_molec):                 # runs over letters from molecules (len=size of largest molecule)
        fancy_latent_point = fancy_latent_point.reshape(1, 1, latent_dimension) 
        fancy_latent_point=fancy_latent_point.to(device)
        decoded_one_hot, hidden = model_decode(fancy_latent_point, hidden)
        decoded_one_hot = decoded_one_hot.flatten()
        decoded_one_hot = decoded_one_hot.detach()
        soft = nn.Softmax(0)
        decoded_one_hot = soft(decoded_one_hot)
        _,max_index=decoded_one_hot.max(0)

        gathered_atoms.append(max_index.data.cpu().numpy().tolist())
            
    model_encode.train()
    model_decode.train()
    
    #test molecules visually
    if total_samples <= 5:
        print('Sample #', total_samples, decoder(hot_to_selfies(gathered_atoms, encoding_alphabet)))
    
    return gathered_atoms



def latent_space_quality(latent_dimension, encoding_alphabet, sample_num):
    total_correct = 0
    all_correct_molecules = set()
    print(f"latent_space_quality:"
          f" Take {sample_num} samples from the latent space")
    
    for sample_i in range(1, sample_num + 1):
        molecule_pre = ''
        for ii in sample_latent_space(latent_dimension, sample_i):
            molecule_pre += encoding_alphabet[ii]
        molecule = molecule_pre.replace(' ', '')

        if type_of_encoding == 1:  # if SELFIES, decode to SMILES
            molecule = selfies.decoder(molecule)

        if is_correct_smiles(molecule):
            total_correct += 1
            all_correct_molecules.add(molecule)

    return total_correct, len(all_correct_molecules)


def quality_in_validation_set(data_valid):    
    x = [i for i in range(len(data_valid))]  # random shuffle input
    shuffle(x)
    data_valid = data_valid[x]
    
    for batch_iteration in range(min(25,num_batches_valid)):  # batch iterator
        
        current_smiles_start, current_smiles_stop = batch_iteration * batch_size, (batch_iteration + 1) * batch_size
        inp_smile_hot = data_valid[current_smiles_start : current_smiles_stop]
    
        inp_smile_encode = inp_smile_hot.reshape(inp_smile_hot.shape[0], inp_smile_hot.shape[1] * inp_smile_hot.shape[2])
        latent_points, mus, log_vars = model_encode(inp_smile_encode)
        latent_points = latent_points.reshape(1, batch_size, latent_points.shape[1])
    
        hidden = model_decode.init_hidden(batch_size = batch_size)
        decoded_one_hot = torch.zeros(batch_size, inp_smile_hot.shape[1], inp_smile_hot.shape[2]).to(device)
        for seq_index in range(inp_smile_hot.shape[1]):
            decoded_one_hot_line, hidden  = model_decode(latent_points, hidden)            
            decoded_one_hot[:, seq_index, :] = decoded_one_hot_line[0]
    
        decoded_one_hot = decoded_one_hot.reshape(batch_size, inp_smile_hot.shape[1], inp_smile_hot.shape[2])
        _, label_atoms  = inp_smile_hot.max(2)   
          
        # assess reconstruction quality
        _, label_atoms_decoded = decoded_one_hot.max(2)
        
        """
        # print a few decoded molecules to visually test reconstruction
        print('Validation set decoded molecules:')
        print(label_atoms_decoded[:2])
        """
        
        flat_decoded = label_atoms_decoded.flatten()
        flat_input = label_atoms.flatten()
        equal_position_count = 0
        num_position = 0
        for mol in range(len(flat_decoded)):
            if flat_decoded[mol] == flat_input[mol]:
                equal_position_count += 1
            num_position += 1
           
        quality = equal_position_count / num_position *100
    return(quality)

# plugged in from this resource: https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels

 
def train_model(data_train, data_valid, num_epochs, latent_dimension, lr_enc, lr_dec, KLD_alpha, sample_num, encoding_alphabet):
    """
    Train the Variational Auto-Encoder
    """
    print('num_epochs: ',num_epochs)
    # initialize an instance of the model
    optimizer_encoder = torch.optim.Adam(model_encode.parameters(), lr=lr_enc)
    optimizer_decoder = torch.optim.Adam(model_decode.parameters(), lr=lr_dec)

    data_train = data_train.clone().detach()
    data_train=data_train.to(device)

    #print(data)
    quality_valid_list=[0,0,0,0];
    quality_valid_diff=[]
    num_deviates = 0
    if settings['evaluate']['evaluate_model']:
        num_epochs = 1
    for epoch in range(num_epochs):
        x = [i for i in range(len(data_train))]  # random shuffle input
        shuffle(x)

        data_train  = data_train[x]
        latent_points_combined = []
        output_mol_combined = [] # decoded molecules in string format
        input_mol_combined = [] # input molecules in string format
        start = time.time()
        for batch_iteration in range(num_batches_train):  # batch iterator
            
            loss, recon_loss, kld = 0., 0., 0.

            # manual batch iterations
            current_smiles_start, current_smiles_stop = batch_iteration * batch_size, (batch_iteration + 1) * batch_size 
            inp_smile_hot = data_train[current_smiles_start : current_smiles_stop]

            # reshaping for efficient parallelization
            inp_smile_encode = inp_smile_hot.reshape(inp_smile_hot.shape[0], inp_smile_hot.shape[1] * inp_smile_hot.shape[2]) 
            latent_points, mus, log_vars = model_encode(inp_smile_encode)
            z = latent_points.detach().numpy()
            latent_points_combined.extend(z)
            
            latent_points = latent_points.reshape(1, batch_size, latent_points.shape[1])
            
            # standard Kullback–Leibler divergence
            kld += -0.5 * torch.mean(1. + log_vars - mus.pow(2) - log_vars.exp()) 

            # initialization hidden internal state of RNN (RNN has two inputs and two outputs:)
            #    input: latent space & hidden state
            #    output: onehot encoding of one character of molecule & hidden state
            #    the hidden state acts as the internal memory
            hidden = model_decode.init_hidden(batch_size = batch_size)
                                                                       
            # decoding from RNN N times, where N is the length of the largest molecule (all molecules are padded)
            decoded_one_hot = torch.zeros(batch_size, inp_smile_hot.shape[1], inp_smile_hot.shape[2]).to(device) 
            
            for seq_index in range(inp_smile_hot.shape[1]):
                decoded_one_hot_line, hidden  = model_decode(latent_points, hidden)
                decoded_one_hot[:, seq_index, :] = decoded_one_hot_line[0]
            
            test_decoded_one_hot = decoded_one_hot.reshape(batch_size, inp_smile_hot.shape[1], inp_smile_hot.shape[2])
            decoded_one_hot = decoded_one_hot.reshape(batch_size * inp_smile_hot.shape[1], inp_smile_hot.shape[2])
            _, label_atoms  = inp_smile_hot.max(2)
            test_label_atoms = label_atoms
            
            if settings['evaluate']['evaluate_model']:
                # add the new batch of decoded molecules, in string format, to memory
                _, decoded_mol = test_decoded_one_hot.max(2)
                output_mol = []
                for mol in decoded_mol:
                    output_mol.append(decoder(hot_to_selfies(mol, encoding_alphabet)))
                output_mol_combined.extend(output_mol)
                
                # add the input molecules in string format to memory
                input_mol = []
                for mol in test_label_atoms:
                    input_mol.append(decoder(hot_to_selfies(mol, encoding_alphabet)))
                input_mol_combined.extend(input_mol)
                
            else:
            
                label_atoms     = label_atoms.reshape(batch_size * inp_smile_hot.shape[1])
                
                # we use cross entropy of expected symbols and decoded one-hot
                criterion   = torch.nn.CrossEntropyLoss()
                recon_loss += criterion(decoded_one_hot, label_atoms)
    
                loss += recon_loss + KLD_alpha * kld 
    
                # perform back propogation
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(model_decode.parameters(), 0.5)
                optimizer_encoder.step()
                optimizer_decoder.step()
    
                if batch_iteration % 30 == 0:     
                    end = time.time()     
                    
                    _, label_atoms_decoded = test_decoded_one_hot.max(2)
                    """
                    # print a few decoded molecules to visually test reconstruction
                    print('Training set decoded molecules:')
                    print(label_atoms_decoded[:2])
                    """
                    
                    flat_decoded = label_atoms_decoded.flatten()
                    flat_input = test_label_atoms.flatten()
                    equal_position_count = 0
                    num_position = 0
                    for mol in range(len(flat_decoded)):
                        if flat_decoded[mol] == flat_input[mol]:
                            equal_position_count += 1
                        num_position += 1
                    quality = equal_position_count / num_position *100
    
                    qualityValid=quality_in_validation_set(data_valid)
                    new_line = 'Epoch: %d,  Batch: %d / %d,\t(loss: %.4f\t| quality: %.4f | quality_valid: %.4f)\tELAPSED TIME: %.5f' % (epoch, batch_iteration, num_batches_train, loss.item(), quality, qualityValid, end - start)
                    print(new_line)
                    start = time.time()
        
        #print(len(input_mol_combined), input_mol_combined[0])
        #print(len(output_mol_combined), output_mol_combined[0])
        #print(len(latent_points_combined), latent_points_combined[0])
                
        if settings['plot']['plot_PCA']:
            if epoch  % 10 == 0:
                print('PCA Projection:')
                Z_pca = PCA(n_components=2).fit_transform(latent_points_combined)
                
                #print(len(Z_pca), Z_pca[0])
                
                # preprocessing step to standardize the data by scaling features
                # to lie between 0 and 1 (allows robustness to small standard
                # deviations of features)
                #Z_pca = MinMaxScaler().fit_transform(Z_pca)
                
                df = pd.DataFrame(np.transpose((Z_pca[:,0],Z_pca[:,1])))
                df.columns = ['x','y']
                
                plt.scatter(x=df['x'], y=df['y'],
                            cmap= 'viridis', marker='.',
                            s=10,alpha=0.5, edgecolors='none')
                plt.show()
                
        if settings['plot']['plot_tSNE']:
            if epoch % 20 == 0:
                print('tSNE projection:')
                n_comp = settings['tSNE']['n_components']
                perplexity = settings['tSNE']['perplexity']
                # see https://distill.pub/2016/misread-tsne/ for hyperparameter tuning tips
                Z_tsne = TSNE(n_components=n_comp, perplexity=perplexity).fit_transform(latent_points_combined)
                print('Projection finished.')
                
                #print(len(Z_tsne), Z_tsne[0])
                
                # preprocessing step to standardize the data by scaling features
                # to lie between 0 and 1 (allows robustness to small standard
                # deviations of features)
                #Z_tsne = MinMaxScaler().fit_transform(Z_tsne)
                if not settings['evaluate']['evaluate_model']:
                    if n_comp == 2:
                        df = pd.DataFrame(np.transpose((Z_tsne[:,0],Z_tsne[:,1])))
                        df.columns = ['x','y']
                        
                        print('Plotting projection...')
                        plt.scatter(x=df['x'], y=df['y'],
                                    cmap= 'viridis', marker='.',
                                    s=10,alpha=0.5, edgecolors='none')
                        plt.show()
                        #plt.savefig('VAE_dependencies/Saved_models/VAE_{}-comp_tsne_fig_epoch_{}'.format(settings['plot']['n_components'],settings['training_VAE']['num_epochs']))
                    
                    elif n_comp == 3:
                        df = pd.DataFrame(np.transpose((Z_tsne[:,0],Z_tsne[:,1],Z_tsne[:,2])))
                        df.columns = ['x', 'y', 'z']
                        
                        print('Plotting projection...')
                        fig=plt.figure()
                        plot = fig.add_subplot(111, projection='3d')
                        plot.scatter(df['x'], df['y'], df['z'], marker='.')
                else:
                    num_clusters = settings['kmeans']['num_clusters']
                    if n_comp == 2:
                        df = pd.DataFrame(np.transpose((Z_tsne[:,0],Z_tsne[:,1])))
                        df.columns = ['x','y']
                        
                        print('Finding clusters...')
                        # perform k-means clustering
                        centers, labels = find_clusters(Z_tsne, 20)
                        
                        #print(len(labels), labels[:20])
                        #print(len(centers), centers[0])
                        print('Clusters found.')
                        
                        print('Plotting projection...')
                        plt.scatter(x=df['x'], y=df['y'], c=labels,
                                    cmap= 'viridis', marker='.',
                                    s=10,alpha=0.5, edgecolors='none')
                        plt.show()
                        plt.savefig('VAE_dependencies/Saved_models/VAE_{}-comp_{}-perplexity_{}-clusters_tsne_fig_epoch_{}'.format(n_comp, perplexity, num_clusters, settings['training_VAE']['num_epochs']))                    
                    
                    elif n_comp == 3:
                        df = pd.DataFrame(np.transpose((Z_tsne[:,0],Z_tsne[:,1],Z_tsne[:,2])))
                        df.columns = ['x', 'y', 'z']
                        
                        print('Finding clusters...')
                        # perform k-means clustering
                        centers, labels = find_clusters(Z_tsne, 20)
                        
                        #print(len(labels), labels[:20])
                        #print(len(centers), centers[0])
                        print('Clusters found.')
                        
                        print('Plotting projection...')
                        fig=plt.figure()
                        plot = fig.add_subplot(111, projection='3d')
                        plot.scatter(df['x'], df['y'], df['z'], c=labels, marker='.')
                        plot.figure.savefig('VAE_dependencies/Saved_models/VAE_{}-comp_{}-perplexity_{}-clusters_tsne_fig_epoch_{}'.format(n_comp, perplexity, num_clusters, settings['training_VAE']['num_epochs']))
                    
                    cluster_to_mol = {}
                    for i in range(len(labels)):
                        label = labels[i]
                        input_mol = input_mol_combined[i]
                        output_mol = output_mol_combined[i]
                        
                        if label not in cluster_to_mol:
                            cluster_to_mol[label] = [centers[label], [], []]
                            
                        cluster_to_mol[label][1].append(input_mol)
                        cluster_to_mol[label][2].append(output_mol)
                        
                    save = input("Enter 'y' to save tSNE proj and results to file.")
                    if save == 'y':
                        f = open('VAE_dependencies/Saved_models/VAE_{}-comp_{}-perplexity_{}-clusters_tsne_clusters_epoch_{}'.format(n_comp, perplexity, num_clusters, settings['training_VAE']['num_epochs']), "w+")
                    for label in cluster_to_mol:
                        print('LABEL: '+str(label))
                        print('CENTER: '+str(cluster_to_mol[label][0]))
                        print('INPUT MOLECULES:')
                        for mol in cluster_to_mol[label][1]:
                            print('\t'+ mol)
                        print('OUTPUT MOLECULES:')
                        for mol in cluster_to_mol[label][2]:
                            print('\t'+ mol)
                        print('\n\n')
                        if save == 'y':
                            f.write('LABEL: '+str(label)+'\n')
                            f.write('CENTER: '+str(cluster_to_mol[label][0])+'\n')
                            f.write('INPUT MOLECULES:\n')
                            for mol in cluster_to_mol[label][1]:
                                f.write('\t'+ mol+'\n')
                            f.write('OUTPUT MOLECULES:\n')
                            for mol in cluster_to_mol[label][2]:
                                f.write('\t'+ mol+'\n')
                            f.write('\n\n')
                    if save == 'y':
                        f.close()
                
        if not settings['evaluate']['evaluate_model'] and settings['plot']['plot_quality']:
            recons_quality_valid.append(qualityValid)
            recons_quality_train.append(quality)
            
        if not settings['evaluate']['evaluate_model'] and settings['plot']['plot_loss']:
            recons_loss.append(loss.item())
        
        if not settings['evaluate']['evaluate_model'] and settings['evaluate']['evaluate_metrics']:
            qualityValid = quality_in_validation_set(data_valid)
            
            if len(quality_valid_diff) > 30:
                quality_valid_diff.pop(0)
            quality_valid_diff.append(qualityValid - quality_valid_list[len(quality_valid_list)-1])
            quality_valid_list.append(qualityValid)
    
            # only measure validity of reconstruction improved
            quality_increase = len(quality_valid_list) - np.argmax(quality_valid_list)
            if quality_increase == 1 and quality_valid_list[-1] > 50.:
                corr, unique = latent_space_quality(latent_dimension,sample_num = sample_num, encoding_alphabet=encoding_alphabet)
            else:
                corr, unique = -1., -1.
    
            new_line = 'Validity: %.5f %% | Diversity: %.5f %% | Reconstruction: %.5f %%' % (corr * 100. / sample_num, unique * 100. / sample_num, qualityValid)
    
            print(new_line)
            with open('results.dat', 'a') as content:
                content.write(new_line + '\n')
    
            if quality_valid_list[-1] < 70. and epoch > 200:
                break
            
            if qualityValid < quality - 5:
                num_deviates += 1
                
            if num_deviates == 10:
                print('Early stopping criteria: validation set quality deviates from training set quality')
                break
        
        
        
            

def get_selfie_and_smiles_encodings_for_dataset(filename_data_set_file_smiles):
    """
    Returns encoding, alphabet and length of largest molecule in SMILES and SELFIES, given a file containing SMILES molecules.
    input:
        csv file with molecules. Column's name must be 'smiles'.
    output:
        - selfies encoding
        - selfies alphabet
        - longest selfies string
        - smiles encoding (equivalent to file content)
        - smiles alphabet (character based)
        - longest smiles string
    """

    df = pd.read_csv(filename_data_set_file_smiles)
    smiles_list = np.asanyarray(df.smiles)
    smiles_alphabet = list(set(''.join(smiles_list)))
    largest_smiles_len = len(max(smiles_list, key=len))
    print('--> Translating SMILES to SELFIES...')
    selfies_list = list(map(selfies.encoder, smiles_list))
    largest_selfies_len = max(len_selfie(s) for s in selfies_list)

    all_selfies_chars = split_selfie(''.join(selfies_list))
    all_selfies_chars.append('[epsilon]')
    selfies_alphabet = list(set(all_selfies_chars))
    print('Finished translating SMILES to SELFIES.')
    return(selfies_list, selfies_alphabet, largest_selfies_len, smiles_list, smiles_alphabet, largest_smiles_len)
    
    
if __name__ == '__main__':   
    try:
        content = open('logfile.dat', 'w')
        content.close()
        content = open('results.dat', 'w') 
        content.close()

        if os.path.exists("settings.yml"):        
            user_settings=yaml.safe_load(open("settings.yml","r"))
            settings = user_settings
        else:
            print("Expected a file settings.yml but didn't find it.")
            print()
            exit()
       
        
        print('--> Acquiring data...')        
        type_of_encoding = settings['data']['type_of_encoding']
        file_name_smiles = settings['data']['smiles_file']
        
        selfies_list, selfies_alphabet, largest_selfies_len, smiles_list, smiles_alphabet, largest_smiles_len=get_selfie_and_smiles_encodings_for_dataset(file_name_smiles)
        print('Finished acquiring data.')

        if type_of_encoding == 0:
            print('Representation: SMILES')            
            encoding_alphabet=smiles_alphabet
            encoding_alphabet.append(' ') # for padding
            encoding_list=smiles_list
            largest_molecule_len = largest_smiles_len
            print('--> Creating one-hot encoding...')
            data = multiple_smile_to_hot(smiles_list, largest_molecule_len, encoding_alphabet)
            print('Finished creating one-hot encoding.')
        elif type_of_encoding == 1:
            print('Representation: SELFIES')            
            
            encoding_alphabet=selfies_alphabet
            encoding_list=selfies_list
            largest_molecule_len=largest_selfies_len
            
            print('--> Creating one-hot encoding...')
            data = multiple_selfies_to_hot(encoding_list, largest_molecule_len, encoding_alphabet)
            print('Finished creating one-hot encoding.')

        len_max_molec = data.shape[1]
        len_alphabet = data.shape[2]
        len_max_molec1Hot = len_max_molec * len_alphabet
        print(' ')
        print('Alphabet has ', len_alphabet, ' letters, largest molecule is ', len_max_molec, ' letters.')
         
        data_parameters = settings['data']
        batch_size = data_parameters['batch_size']
 
        encoder_parameter = settings['encoder'] 
        decoder_parameter = settings['decoder']
        training_parameters = settings['training_VAE']
  
        model_encode = VAE_encode(**encoder_parameter)
        model_decode = VAE_decode(**decoder_parameter)
       
        model_encode.train()
        model_decode.train()
           
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('*'*15, ': -->', device)        
  
        data = torch.tensor(data, dtype=torch.float).to(device)
  
        train_valid_test_size=[0.5, 0.5, 0.0]    
        x = [i for i in range(len(data))]  # random shuffle input
        shuffle(x)
        data = data[x]
        idx_traintest=int(len(data)*train_valid_test_size[0])
        idx_trainvalid=idx_traintest+int(len(data)*train_valid_test_size[1])    
        data_train=data[0:idx_traintest]
        data_valid=data[idx_traintest:idx_trainvalid]
        data_test=data[idx_trainvalid:]
         
        num_batches_train = int(len(data_train) / batch_size)
        num_batches_valid = int(len(data_valid) / batch_size)
      
        model_encode = VAE_encode(**encoder_parameter).to(device)
        model_decode = VAE_decode(**decoder_parameter).to(device)
        
        if settings['plot']['plot_quality'] and not settings['evaluate']['evaluate_model']:
            recons_quality_valid = []
            recons_quality_train = []
        if settings['plot']['plot_loss'] and not settings['evaluate']['evaluate_model']:
            recons_loss = []
        
        if settings['evaluate']['evaluate_model']:
            if os.path.exists('VAE_dependencies/Saved_models/VAE_encode_epoch_{}'.format(settings['training_VAE']['num_epochs'])):
                model_encode.load_state_dict(torch.load('VAE_dependencies/Saved_models/VAE_encode_epoch_{}'.format(settings['training_VAE']['num_epochs'])))
                model_decode.load_state_dict(torch.load('VAE_dependencies/Saved_models/VAE_decode_epoch_{}'.format(settings['training_VAE']['num_epochs'])))
                model_encode.eval()
                model_decode.eval()
            else: 
                print('No models saved in file with ' + str(settings['training_VAE']['num_epochs']) + ' epochs')
                
        print("start training")
        train_model(data_train=data_train, data_valid=data_valid, **training_parameters, encoding_alphabet=encoding_alphabet)
        
        if not settings['evaluate']['evaluate_model']:
            torch.save(model_encode.state_dict(), 'VAE_dependencies/Saved_models/VAE_encode_epoch_{}'.format(settings['training_VAE']['num_epochs']))
            torch.save(model_decode.state_dict(), 'VAE_dependencies/Saved_models/VAE_decode_epoch_{}'.format(settings['training_VAE']['num_epochs']))
            #plot epoch vs reconstruction loss / quality
            print(recons_quality_valid, recons_quality_train, recons_loss)
            if settings['plot']['plot_quality']:
                line1, = plt.plot(recons_quality_valid, label='Validation set')
                line2, = plt.plot(recons_quality_train, label='Training set')
                plt.xlabel('Epochs')
                plt.ylabel('Reconstruction Quality (%)')
                plt.legend(handles=[line1, line2])
                plt.show()
            if settings['plot']['plot_loss']:
                plt.plot(recons_loss)
                plt.xlabel('Epochs')
                plt.ylabel('Reconstruction Loss')
                plt.show()

        
        with open('COMPLETED', 'w') as content:
            content.write('exit code: 0')


    except AttributeError:
        _, error_message,_ = sys.exc_info()
        print(error_message)