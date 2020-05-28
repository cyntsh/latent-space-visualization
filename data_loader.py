#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:47:01 2020

@author: CS
"""

"""
This file is to encode SMILES and SELFIES into one-hot encodings
"""
import re
import numpy as np

def smile_to_hot(smile, largest_smile_len, alphabet):
    """
    Go from a single smile string to a one-hot encoding.
    """
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # integer encode input smile
    for _ in range(largest_smile_len-len(smile)):
        smile+=' '
        
    atoms = []
    for i in range(len(smile)):
        atoms.append(smile[i])        
        
    integer_encoded = [char_to_int[char] for char in atoms]

    # one hot-encode input smile
    onehot_encoded = list()
    for value in integer_encoded:
    	letter = [0 for _ in range(len(alphabet))]
    	letter[value] = 1
    	onehot_encoded.append(letter)
    return integer_encoded, np.array(onehot_encoded)
    

def multiple_smile_to_hot(smiles_list, largest_molecule_len, alphabet):
    """
    Convert a list of smile strings to a one-hot encoding
    
    Returned shape (num_smiles x len_of_largest_smile x len_smile_encoding)
    """
    hot_list = []
    for smile in smiles_list:
        _, onehot_encoded = smile_to_hot(smile, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)
        
def len_selfie(molecule):
    """Returns the length of selfies <molecule>, in other words, the
     number of characters in the sequence."""
    return molecule.count('[') + molecule.count('.')


def split_selfie(molecule):
    """Splits the selfies <molecule> into a list of character strings.
    """
    return re.findall(r'\[.*?\]|\.', molecule)

def selfies_to_hot(molecule, largest_selfie_len, alphabet):
    """
    Go from a single selfies string to a one-hot encoding.
    """
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # pad with [epsilon]
    molecule += '[epsilon]' * (largest_selfie_len - len_selfie(molecule))
    # integer encode
    char_list = split_selfie(molecule)
    integer_encoded = [char_to_int[char] for char in char_list]
    # one hot-encode the integer encoded selfie
    onehot_encoded = list()
    for index in integer_encoded:
        letter = [0] * len(alphabet)
        letter[index] = 1
        onehot_encoded.append(letter)

    return integer_encoded, np.array(onehot_encoded)

def hot_to_selfies(molecule, alphabet):
    """
    Go from the one-hot encoding to the corresponding selfies string
    """
    one_hot = ''
    for _ in molecule:
        'pass'
        one_hot += alphabet[_]
    return one_hot
    

def multiple_selfies_to_hot(selfies_list, largest_molecule_len, alphabet):
    """
    Convert a list of selfies strings to a one-hot encoding
    """
    hot_list = []
    for selfiesI in selfies_list:
        _, onehot_encoded = selfies_to_hot(selfiesI, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)