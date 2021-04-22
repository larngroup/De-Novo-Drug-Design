 # -*- coding: utf-8 -*-
import numpy as np
import os
import tensorflow as tf
from keras import backend as K
from utils import *
from dnnQSAR import Model
from alternativeQSAR import build_models
from tokens import tokens_table
from prediction import Predictor
from grid_search_models import grid_search

import rdkit as rd
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import RDConfig
from rdkit.Chem import PandasTools
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdMolDescriptors import GetAtomPairFingerprint
from rdkit.Chem.AtomPairs import Torsions

# modeling
import sklearn as sk
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

os.environ["CUDA_VISIBLE_DEVICES"]="0"
session = tf.compat.v1.Session()
K.set_session(session)

config_file = 'configPredictor.json' # Name of the configuration file 
property_identifier = 'bbb'
model_type = 'dnn' # 'dnn', 'SVR', 'RF', or 'KNN'
descriptor = 'SMILES' # The type of model's descriptor can be 'SMILES' or 'ECFP'. If we want to use 
#rnn architecture we use SMILES. Conversely, if we want to use a fully connected architecture, we use ECFP descriptors. 
searchParameters = False # True (gridSearch) or False (train with the optimal parameters)
    
def main():
    """
    Main routine: Script that evokes all the necessary routines 
    """
    
    # load model configurations
    config = load_config(config_file,property_identifier)
    directories([config.checkpoint_dir])
    
    # Load the table of possible tokens
    token_table = tokens_table().table

    # Read and extract smiles and labels from the csv file
    smiles_raw,labels_raw = reading_csv(config,property_identifier)
    
    print("BBB+: ", np.sum(labels_raw))
    
#    mols = [Chem.MolFromSmiles(x) for x in smiles_raw]
#
#    morgan_fp = [Chem.GetMorganFingerprintAsBitVect(x, 2, nBits = 2048) for x in mols]
#
#
#    # convert the RDKit explicit vectors into numpy arrays
#    morg_fp_np = []
#    for fp in morgan_fp:
#      arr = np.zeros((1,))
#      DataStructs.ConvertToNumpyArray(fp, arr)
#      morg_fp_np.append(arr)
#  
#  
#    x_morg = morg_fp_np
#    
#    x_morg_rsmp, y_morg_rsmp = SMOTE().fit_resample(x_morg, labels_raw)



    
    # Padd each SMILES string with spaces until reaching the size of the largest molecule
    smiles_padded,padd = pad_seq(smiles_raw,token_table,0)
    config.paddSize = padd
            
    # Compute the dictionary that makes the correspondence between each token and unique integers
    tokenDict = smilesDict(token_table)

	# Tokenize - transform the SMILES strings into lists of tokens 
    tokens = tokenize(smiles_padded,token_table)   

    # Transforms each token to the respective integer, according to the previously computed dictionary
    smiles_int = smiles2idx(tokens,tokenDict)


    if searchParameters:
    	# Split data into training, validation and testing sets.
        data = data_division(config,smiles_int,labels_raw,False,model_type,descriptor)
        
        # Normalize the label 
        data,data_aux = normalize(data)
        
        # Drop Rate
        drop_rate = [0.1,0.3,0.5]
        # Batch size
        batch_size = [16,32]
        # Learning Rate
        learning_rate=[0.001,0.0001,0.01]
        # Number of cells
        number_units = [64,128,256]
        # Activation function
        activation = ['linear','softmax', 'relu']
        # Memory cell
        rnn = ['LSTM','GRU']
        epochs = [100]
        counter = 0
        for dr in drop_rate:
            for bs in batch_size:
                for lr in learning_rate:
                    for nu in number_units:
                        for act in activation:
                            for nn in rnn:
                                for ep in epochs:
                                    
                                    param_identifier = [str(dr)+"_"+str(bs)+"_"+str(lr)+"_"+
                                                    str(nu)+"_"+nn+"_"+act+"_"+str(ep)]
                                    counter += 1
                                    if counter > 304:
                                        print("\nTesting this parameters: ") 
                                        print(param_identifier)
                                        config.dropout = dr
                                        config.batch_size = bs
                                        config.lr = lr 
                                        config.n_units = nu
                                        config.activation_rnn = act
                                        config.rnn = nn
                                        Model(config,data,searchParameters,descriptor)
 
    
    if model_type == 'dnn' and descriptor == 'SMILES':
        # Data splitting and Cross-Validation for the SMILES-based neural network
        data_rnn_smiles = data_division(config,smiles_int,labels_raw,True,model_type,descriptor)
        x_test = data_rnn_smiles[2]
        y_test = data_rnn_smiles[3]
        data_cv = cv_split(data_rnn_smiles,config)
    
    elif model_type == 'dnn' and descriptor == 'ECFP':
    	# Data splitting and Cross-Validation for the ECFP-based neural network
        data_rnn_ecfp = data_division(config,x_morg_rsmp,y_morg_rsmp,True,model_type,descriptor)
        x_test = data_rnn_ecfp[2]
        y_test = data_rnn_ecfp[3]
        data_cv = cv_split(data_rnn_ecfp,config)
    else:
    	# Data splitting, cross-validation and grid-search for the other standard QSAR models        
        data_otherQsar = data_division(config,data_ecfp,labels_raw,True,model_type,descriptor)    
        x_test = data_otherQsar[2]
        y_test = data_otherQsar[3]
        data_cv = cv_split(data_otherQsar,config)
        best_params = grid_search(data_otherQsar,model_type)

    i = 0
#    utils = []
    metrics = []
    for split in data_cv:
        print('\nCross validation, fold number ' + str(i) + ' in progress...')
        data_i = []
        train, val = split
        
        if model_type != 'dnn' or descriptor == 'ECFP':
            X_train = data_rnn_ecfp[0][train]
            y_train = np.array(data_rnn_ecfp[1])[train]
            X_val = data_rnn_ecfp[0][val]
            y_val = np.array(data_rnn_ecfp[1])[val]
            y_train = y_train.reshape(-1,1)
            y_val= y_val.reshape(-1,1)
            
        else:
            X_train = data_rnn_smiles[0][train]
            y_train = np.array(data_rnn_smiles[1])[train]
            X_val = data_rnn_smiles[0][val]
            y_val = np.array(data_rnn_smiles[1])[val]
            
#            X_train = smiles_int[train]
#            y_train = np.array(labels_raw)[train]
#            X_val = smiles_int[val]
#            y_val = np.array(labels_raw)[val]
            y_train = y_train.reshape(-1,1)
            y_val= y_val.reshape(-1,1)
            

        data_i.append(X_train)
        data_i.append(y_train)
        data_i.append(x_test)
        data_i.append(y_test)
        data_i.append(X_val)
        data_i.append(y_val)
   
#        data_i,data_aux = normalize(data_i)
         
#        utils.append(data_aux)
                
        config.model_name = "model" + str(i)
        
        if model_type == 'dnn':
            Model(config,data_i,False,descriptor)
#        else:
#            build_models(data_i,model_type,config,best_params)

        i+=1
    
    # Model's evaluation with two example SMILES strings 
    predictor= Predictor(config,token_table,model_type,descriptor)
#    list_ss = ["NC(=O)c1cccc(OC2CC3CCC(C2)N3C2(c3ccccc3)CC2)c1","CN(C)C(CNC(CN)Cc1ccc(O)cc1)Cc1ccc(O)cc1"] #3.85 e 1.73
#    prediction = predictor.predict(list_ss,utils)
#    print(prediction)
    
    # Model's evaluation with the test set
    metrics = predictor.evaluator(data_i)
    
    print("\n\nAccuracy: ",metrics[0], "\nAUC: ", metrics[1], "\nSpecificity: ", metrics[2], "\nSensitivity: ",metrics[3], "\nMCC: ", metrics[4])

if __name__ == '__main__': 
    
    start = time.time()
    print("start time:", start)
    main()
    end = time.time()
    print("\n\n Finish! Time is (s):", end - start)