# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:46:42 2019

@author: Tiago
"""
from tensorflow.keras.models import model_from_json
import numpy as np
from utils import *
import rdkit as rd
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import PandasTools
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdMolDescriptors import GetAtomPairFingerprint
from rdkit.Chem.AtomPairs import Torsions


class Predictor(object):
    def __init__(self, config,property_identifier):
        """
        Constructor for the Predictor object.
        Parameters
        ----------
        config: Configuration parameters 
        tokens: Table with all possible tokens in SMILES
        labels: True values of Predictor training dataset to perform the 
                denormalization step
        property_identifier: String identifying the property to optimize
        
        Returns
        -------
        This function loads the Predictor models already trained
        """
        super(Predictor, self).__init__()
        
        if property_identifier != "bbb":
            self.labels = reading_csv(config,property_identifier)
        
        self.tokens = [
                 'H','Cl', 'Br','B', 'C', 'N', 'O', 'P', 'S', 'F', 'I',
                '(', ')', '[', ']', '=', '#', '@', '*', '%', '0', '1', '2',
                '3', '4', '5', '6', '7', '8', '9', '.', '/', '\\', '+', '-',
                 'c', 'n', 'o', 's','p', ' ']
        
        self.config = config

        loaded_models = []
        if property_identifier == "a2d":
            model_path = "predictor_a2d\\model"
        elif property_identifier == "bbb":
            model_path = "predictor_bbb\\model"
        elif property_identifier == "kor":
            model_path = "predictor_kor\\model"
            
        for i in range(5):
            
            json_file= open(model_path+str(i)+".json", 'r')
            
            
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)            
            # load weights into new model
            loaded_model.load_weights(model_path+str(i)+".h5")
            print("Models " + str(i) + " loaded from disk!")
            loaded_models.append(loaded_model)
        
        self.loaded_models = loaded_models

        
    def predict(self, smiles,property_identifier):
        """
        This function performs the prediction of the property in study
        Parameters
        ----------
        smiles: List of SMILES strings to perform the prediction      
        Returns
        -------
        Before do the prediction, this function performs the SMILES' padding 
        and tokenization. It also performs the denormalization step and compute 
        the mean value of the prediction of the 5 models.
        """
        
        if property_identifier == "a2d":
            smiles_padded,kl = pad_seq(smiles,self.tokens,self.config.paddSize)
            
            d = smilesDict(self.tokens)
      
            tokens = tokenize(smiles_padded,self.tokens)
                              
            smiles_int = smiles2idx(tokens,d)
            
            prediction = []
                
            for m in range(len(self.loaded_models)):
                
                prediction.append(self.loaded_models[m].predict(smiles_int))
    
            prediction = np.array(prediction).reshape(len(self.loaded_models), -1)
            
            prediction = denormalization(prediction,self.labels)
                    
            prediction = np.mean(prediction, axis = 0)
        
        elif property_identifier == "bbb":
        
            mols = [Chem.MolFromSmiles(x) for x in smiles]
            morgan_fp = np.array([np.array(Chem.GetMorganFingerprintAsBitVect(x, 2, nBits = 2048)) for x in mols])
            
            prediction = []
                
            for m in range(len(self.loaded_models)):
                
                prediction.append(self.loaded_models[m].predict(morgan_fp))
    
            prediction = np.array(prediction).reshape(len(self.loaded_models), -1)
            
                    
            prediction = np.mean(prediction, axis = 0)
     
        return prediction
            
