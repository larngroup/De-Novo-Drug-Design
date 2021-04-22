# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:33:27 2020

@author: Tiago
"""
from rdkit import Chem
import csv

def read_files():
    unique_mols = []
    unique_labels = []
    mols_a = []
    label_a = []
    mols_b = []
    label_b = []
    
    for file in ["BBBP", "ci600312d_si_001"]:
        
        file_name = file + '.csv'
        with open(file_name) as f:
            lines = f.readlines()
            for idx,l in enumerate(lines):
                
                if file == "BBBP":
                    l = l.strip().split(',')
                    
                    if idx > 0:
                        smi = l[3]
                        try:
                            mol = Chem.MolFromSmiles(smi, sanitize=True)
                            s = Chem.MolToSmiles(mol)
                            mols_a.append(s)
                            label_a.append(l[2])
                            
                            if smi not in unique_mols and len(s)<65:
                                unique_mols.append(smi)
                                unique_labels.append(l[2])
                        except:
                            print(smi)
                            print('ERROR: Invalid SMILES!')
                            
                        
                    
                elif file == "ci600312d_si_001":
                    l = l.strip().split(';')
                    
                    if idx > 2 and len(l[5]) ==1:
                        smi = l[2]
                        try:
                            mol = Chem.MolFromSmiles(smi, sanitize=True)
                            s = Chem.MolToSmiles(mol)
                            mols_b.append(s)
                            label_b.append(l[5])
                            
                            if smi not in unique_mols and len(s)<65:
                                unique_mols.append(smi)
                                unique_labels.append(l[5])
                                
                        except:
                            print(smi)
                            print('ERROR: Invalid SMILES!')    
                elif file == "ci034205dsi20031027_054116":
                    l = l.strip().split(';')
                    try:
                        mol = Chem.MolFromSmiles(smi, sanitize=True)
                        s = Chem.MolToSmiles(mol)
                        if smi not in unique_mols and len(s)<65:
                            unique_mols.append(smi)
                            unique_labels.append('0')
                    except:
                            print(smi)
                            print('ERROR: Invalid SMILES!')                        
                    
    unique_labels_int = [int(elem) for elem in unique_labels]      
          
#    for i,num in enumerate(unique_labels):
#        unique_labels_int.append(int(num))
        
    return mols_a,mols_b,unique_mols,unique_labels_int

def save_mols(smiles,labels):
    
    filename = 'data_bbb.csv'
    with open(filename, mode='w') as w_file:
        file_writer = csv.writer(w_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow(['SMILES','bbb'])
        for i,smile in enumerate(smiles):
            file_writer.writerow([smile,labels[i]])
    print("File " + filename + " successfully saved!")

if __name__ == "__main__":
    file_a,file_b,smiles_unique,labels_unique = read_files()
    
    save_mols(smiles_unique,labels_unique)


    
    
    
    
    