# -*- coding: utf-8 -*-
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np

def readFile(filename):
    """
    Function that reads and extracts the SMILES strings from the .smi file
    Parameters
    ----------
    filename: directory of the .smi file
    Returns
    -------
    List of SMILES strings
    """

    file = open(filename,"r")
    lines = file.readlines()
    
    smiles = []
    for line in lines:
        x =line.split("  ")
        smiles.append(x[0].strip())
#    print(len(smiles))
    return smiles

def count_duplicates(seq_a,seq_b): 
    '''takes as argument a sequence and
    returns the number of duplicate elements'''
    seq_a = list(np.unique(seq_a))[1:]
    seq_both = seq_a + seq_b
#    print(seq_a)
#    print(seq_b)
#    print(seq_both)
    number_duplicates =  len(seq_both) - len(set(seq_both))
    perc_duplicates = (number_duplicates / len(seq_both)) * 100
    return number_duplicates,perc_duplicates

def diversity(file_A,file_B):

    td = 0
    
    fps_A = []
    for i, row in enumerate(file_A):
        try:
            mol = Chem.MolFromSmiles(row)
            fps_A.append(AllChem.GetMorganFingerprint(mol, 3))
        except:
            print('ERROR: Invalid SMILES!')
            
        
    
    if file_b == None:
        for ii in range(len(fps_A)):
            for xx in range(len(fps_A)):
                ts = 1 - DataStructs.TanimotoSimilarity(fps_A[ii], fps_A[xx])
                td += ts          
      
        td = td/len(fps_A)**2
    else:
        fps_B = []
        for j, row in enumerate(file_B):
            try:
                mol = Chem.MolFromSmiles(row)
                fps_B.append(AllChem.GetMorganFingerprint(mol, 3))
            except:
                print('ERROR: Invalid SMILES!') 
        
        
        for jj in range(len(fps_A)):
            for xx in range(len(fps_B)):
                ts = 1 - DataStructs.TanimotoSimilarity(fps_A[jj], fps_B[xx]) 
                td += ts
        
        td = td / (len(fps_A)*len(fps_B))
    print("Tanimoto distance: " + str(td))  
    return td

if __name__ == '__main__':
    """
    This is the main routine. The two arguments you have to change are the name
    of the files. If you want to compute internal similarity just put the 
    filename_a and the filename_b as 'None'. If you want to compare two sets, 
    write its names properly and it will be computed the Tanimoto distance.
    Note that it is the Tanimoto distance, not Tanimoto similarity. 
    """ 
    filename_a = 'generated_7719_iter85.smi'
    filename_b = None
    
    file_a = readFile(filename_a)
    file_b = None
    if filename_b != None:
        file_b = readFile(filename_b)
        
        number_duplicates,perc_duplicates = count_duplicates(file_a,file_b)
    
    


    diversity(file_a,file_b)
    