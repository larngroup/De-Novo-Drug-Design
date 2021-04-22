# -*- coding: utf-8 -*-
import csv
import numpy as np
import math
import random
import json
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from bunch import Bunch
import time
import os
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
from random import seed 
from random import randint
from imblearn.over_sampling import SMOTE

def reading_csv(config,property_identifier):
    """
    This function loads the SMILES strings and the respective labels of the 
    specified property by the identifier.
    ----------
    config: configuration file
    property_identifier: Identifier of the property we will use. It could be 
    (jak2,logP or kor)
    
    Returns
    -------
    smiles, labels: Lists with the loaded data. We only select the SMILES with
    length under a certain threshold defined in the configuration file. Also, 
    remove the duplicates in the original dataset.
    """
    if property_identifier == 'bbb':
        filepath = config.datapath_jak2
        idx_smiles = 0
        idx_labels = 1
    elif property_identifier == 'a2d':
        filepath = config.datapath_a2d
        idx_smiles = 0
        idx_labels = 1
    elif property_identifier == 'kor':
        filepath = config.datapath_kor
        idx_smiles = 0
        idx_labels = 1        
    
    raw_smiles = []
    raw_labels = []
    
    with open(filepath, 'r') as csvFile:
        reader = csv.reader(csvFile)
        
        it = iter(reader)
        next(it, None)  # skip first item.   
        permeable = 0
        for row in it:
            try:
                if "[S@@H0]" in row[idx_smiles] or "[n+]" in row[idx_smiles] or "[o+]" in row[idx_smiles] or "[c@@]" in row[idx_smiles]:
                    print("MERDA",row[idx_smiles])
                elif permeable < 1249 or float(row[idx_labels]) == 0:
                    raw_smiles.append(row[idx_smiles])
                    raw_labels.append(float(row[idx_labels]))
                    if float(row[idx_labels]) == 1:
                        permeable = permeable + 1

            except:
                pass
    
    smiles = []
    labels = []
#    and raw_smiles[i] not in smiles
    #and 'L' not in raw_smiles[i] and 'Cl' not in raw_smiles[i] and 'Br' not in raw_smiles[i]
    for i in range(len(raw_smiles)):
        if len(raw_smiles[i]) <= config.smile_len_threshold  and 'a' not in raw_smiles[i] and 'Z' not in raw_smiles[i] and 'K' not in raw_smiles[i]:
            smiles.append(raw_smiles[i])
            labels.append(raw_labels[i])
            
    return smiles, labels
        

def get_tokens(smiles):
    """
    This function extracts the dictionary that makes the correspondence between
    each charachter and an integer.
    ----------
    tokens: Set of characters
    
    Returns
    -------
    tokenDict: Returns the dictionary that maps characters to integers
    """
    tokens = []
    
    for smile in smiles:
        for token in smile:
            if token not in tokens:
                tokens.append(token)
    return tokens
           

def smilesDict(tokens):
    """
    This function computes the dictionary that makes the correspondence between 
    each token and an given integer.
    ----------
    tokens: Set of characters
    
    Returns
    -------
    tokenDict: Returns the dictionary that maps characters to integers
    """
    tokenDict = dict((token, i) for i, token in enumerate(tokens))
    return tokenDict

def pad_seq(smiles,tokens,paddSize):
    """
    This function performs the padding of each SMILE.
    ----------
    smiles: Set of SMILES strings with different sizes;
    tokens: Set of characters;
    paddSize: Integer that specifies the maximum size of the padding    
    
    Returns
    -------
    newSmiles: Returns the padded smiles, all with the same size.
    maxLength: Integer that indicates the paddsize. It will be used to perform 
                padding of new sequences.
    """
    maxSmile= max(smiles, key=len)
    maxLength = 0
    
    if paddSize != 0:
       maxLength = paddSize
    else:
        maxLength = len(maxSmile) 

    for i in range(0,len(smiles)):
        if len(smiles[i]) < maxLength:
            smiles[i] = smiles[i] + tokens[-1]*(maxLength - len(smiles[i]))
    
    return smiles,maxLength
             
def smiles2idx(smiles,tokenDict):
    """
    This function transforms each SMILES token to the correspondent integer,
    according the token-integer dictionary previously computed.
    ----------
    smiles: Set of SMILES strings with different sizes;
    tokenDict: Dictionary that maps the characters to integers;    
    
    Returns
    -------
    newSmiles: Returns the transformed smiles, with the characters replaced by 
    the numbers. 
    """           
    newSmiles =  np.zeros((len(smiles), len(smiles[0])))
    for i in range(0,len(smiles)):
        for j in range(0,len(smiles[i])):
            newSmiles[i,j] = tokenDict[smiles[i][j]]
            
    return newSmiles

def data_division(config,smiles_int,labels,cross_validation,model_type,descriptor):
    """
    This function divides data in two or three sets. If we are performing 
    grid_search we divide between training, validation and testing sets. On 
    the other hand, if we are doing cross-validation, we just divide between 
    train/validation and test sets because the train/validation set will be then
    divided during CV.
    ----------
    config: configuration file;
    smiles_int: List with SMILES strings set;
    labels: List with label property set;
    cross_validation: Boolean indicating if we are dividing data to perform 
                      cross_validation or not;
    model_type: String indicating the type of model (dnn, SVR, KNN or RF)
    descriptor: String indicating the descriptor (ECFP or SMILES)
    Returns
    -------
    data: List with the sets of the splitted data.
    """ 
    data = []
   
    idx_test = np.array(random.sample(range(0, len(smiles_int)), math.floor(config.percentage_test*len(smiles_int))))
    train_val_set = np.delete(smiles_int,idx_test,0)
    train_val_labels = np.delete(labels,idx_test)
    
    test_set = np.array(smiles_int)[idx_test.astype(int)]
    labels = np.array(labels)
    test_labels = labels[idx_test]

    if not cross_validation:
        idx_val = np.array(random.sample(range(0, len(train_val_set)), math.floor(config.percentage_test*len(train_val_set))))
        train_set = np.delete(train_val_set,idx_val,0)
        train_labels = np.delete(train_val_labels,idx_val)
        val_set = train_val_set[idx_val]
        train_val_labels = np.array(train_val_labels)
        val_labels = train_val_labels[idx_val]
        
        data.append(train_set)
        data.append(train_labels)
        data.append(test_set)
        data.append(test_labels)
        data.append(val_set)
        data.append(val_labels)
    else:
        data.append(train_val_set)
        data.append(train_val_labels)
        data.append(test_set)
        data.append(test_labels)
  
    return data

def cv_split(data,config):
    """
    This function performs the data spliting into 5 consecutive folds. Each 
    fold is then used once as a test set while the 4 remaining folds 
    form the training set.
    ----------
    config: configuration file;
    data: List with the list of SMILES strings set and a list with the label;
    Returns
    -------
    data: object that contains the indexes for training and testing for the 5 
          folds
    """
    train_val_smiles = data[0]
    train_val_labels = data[1]
    cross_validation_split = KFold(n_splits=config.n_splits, shuffle=True)
    data_cv = list(cross_validation_split.split(train_val_smiles, train_val_labels))
    return data_cv

def sensitivity(y_true, y_pred):
    from keras import backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    from keras import backend as K
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def matthews_correlation(y_true, y_pred):
    from keras import backend as K
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

def r_square(y_true, y_pred):
    """
    This function implements the coefficient of determination (R^2) measure
    ----------
    y_true: True label   
    y_pred: Model predictions 
    Returns
    -------
    Returns the R^2 metric to evaluate regressions
    """
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))


#concordance correlation coeﬃcient (CCC)
def ccc(y_true,y_pred):
    """
    This function implements the concordance correlation coefficient (ccc)
    ----------
    y_true: True label   
    y_pred: Model predictions 
    Returns
    -------
    Returns the ccc measure that is more suitable to evaluate regressions.
    """
    from keras import backend as K
    num = 2*K.sum((y_true-K.mean(y_true))*(y_pred-K.mean(y_pred)))
    den = K.sum(K.square(y_true-K.mean(y_true))) + K.sum(K.square(y_pred-K.mean(y_pred))) + K.int_shape(y_pred)[-1]*K.square(K.mean(y_true)-K.mean(y_pred))
    return num/den

def normalize(data):
    """
    This function implements the percentile normalization step (to avoid the 
    interference of outliers).
    ----------
    data: List of label lists. It contains the y_train, y_test, and y_val (validation)
    Returns
    -------
    Returns z_train, z_test, z_val (normalized targets) and data (values to 
    perform the denormalization step). 
    """
    data_aux = np.zeros(2)
    y_train = data[1]
    y_test = data[3]
    y_val = data[5]
#    m_train = np.mean(y_train)
#    sd_train = np.std(y_train)
#    m_test = np.mean(y_test)
#    sd_test = np.std(y_test)
#    
#    z_train = (y_train - m_train) / sd_train
#    z_test = (y_test - m_test) / sd_test
#    
#    max_train = np.max(y_train)
#    min_train = np.min(y_train)
#    max_val = np.max(y_val)
#    min_val = np.min(y_val)
#    max_test = np.max(y_test)
#    min_test = np.min(y_test)
#    
    q1_train = np.percentile(y_train, 5)
    q3_train = np.percentile(y_train, 90)
#    
    q1_test = np.percentile(y_test, 5)
    q3_test = np.percentile(y_test, 90)
    
    q1_val = np.percentile(y_val, 5)
    q3_val = np.percentile(y_val, 90)

#    z_train = (y_train - min_train) / (max_train - min_train)
#    z_test = (y_test - min_test) / (max_test - min_test)
    
#    data[1] = (y_train - q1_train) / (q3_train - q1_train)
#    data[3]  = (y_test - q1_test) / (q3_test - q1_test)
#    data[5]  = (y_val - q1_val) / (q3_val - q1_val)
    data[1] = (y_train - q1_train) / (q3_train - q1_train)
    data[3]  = (y_test - q1_test) / (q3_test- q1_test)
    data[5]  = (y_val - q1_val) / (q3_val - q1_val)
    
    
    data_aux[1] = q1_train
    data_aux[0] = q3_train
#    data[2] = m_train
#    data[3] = sd_test
   
    return data,data_aux

def denormalization(predictions,data):
    """
    This function implements the denormalization step.
    ----------
    predictions: Output from the model
    data: q3 and q1 values to perform the denormalization
    Returns
    -------
    Returns the denormalized predictions.
    """
    for l in range(len(predictions)):
        
        max_train = data[l][0]
        min_train = data[l][1]
#        m_train = data[l][2]
#        sd_train = data[l][3]
       
        for c in range(len(predictions[0])):
            predictions[l,c] = (max_train - min_train) * predictions[l,c] + min_train
#            predictions[l,c] = predictions[l,c] * sd_train + m_train
  
    return predictions


def load_config(config_file,property_identifier):
    """
    This function loads the configuration file in .json format. Besides, it 
    creates the directory of this experiment to save the created models
    ----------
    config_file: name of the configuration file;
    property_identifier: string that indicates the property we will use;
    Returns
    -------
    This function returns the configuration file.
    """
    print("Loading configuration file...")
    
    with open(config_file, 'r') as config_file:
        config_dict = json.load(config_file)
        config = Bunch(config_dict)
        exp_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        config.checkpoint_dir = os.path.join('experiments',property_identifier + '-' + exp_time+'\\', config.exp_name, 'checkpoints\\')
        #config.output = config.output + exp_time
    print("Configuration file loaded successfully!")
    return config;


def directories(dirs):
	try:
		for dir_ in dirs:
			if not os.path.exists(dir_):
				os.makedirs(dir_)
		return 0
	except Exception as err:
		print('Creating directories error: {}'.format(err))
		exit(-1)
        

def SMILES_2_ECFP(smiles, radius=3, bit_len=4096, index=None):
    """
    This function transforms a list of SMILES strings into a list of ECFP with 
    radius 3.
    ----------
    smiles: List of SMILES strings to transform
    Returns
    -------
    This function return the SMILES strings transformed into a vector of 4096 elements
    """
    fps = np.zeros((len(smiles), bit_len))
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        arr = np.zeros((1,))
        try:
    
            mol = MurckoScaffold.GetScaffoldForMol(mol)
     
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bit_len)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps[i, :] = arr
        except:
            print(smile)
            fps[i, :] = [0] * bit_len
    return pd.DataFrame(fps, index=(smiles if index is None else index))

def regression_plot(y_true,y_pred):
    """
    Function that graphs a scatter plot and the respective regression line to 
    evaluate the QSAR models.
    Parameters
    ----------
    y_true: True values from the label
    y_pred: Predictions obtained from the model
    Returns
    -------
    This function returns a scatter plot.
    """
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred)
    ax.plot([np.min(y_true), np.max(y_true)], [np.min(y_true), np.max(y_true)], 'k--', lw=4)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    plt.show()
    
def tokenize(smiles,token_table):
    """
    Function that transforms the SMILES strings into a list of tokens
    Parameters
    ----------
    smiles: Set of SMILES strings
    token_table: List with all possible tokens
    Returns
    -------
    This function returns the list with tokenized SMILES
    """
    tokenized = []
    
    for xx,smile in enumerate(smiles):
        N = len(smile)
        i = 0
        j= 0
        token = []
        print(xx,smile)
        while (i < N):
            for j in range(len(token_table)):
                symbol = token_table[j]
                if symbol == smile[i:i + len(symbol)]:
                    token.append(symbol)
                    i += len(symbol)
                    break
        while len(token)< 65:
            token.append(' ')
        tokenized.append(token)

    return tokenized
