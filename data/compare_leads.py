# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:48:10 2020

@author: Tiago
"""
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np

def file_reader(file_name, drop_first=True):
    
    molObjects = []
    
    with open(file_name) as f:
        for l in f:
            if drop_first:
                drop_first = False
                continue
                
            
            
            if 'Antagonist' in l:
                l = l.strip().split(",")
                try:
                    mol = Chem.MolFromSmiles(l[9], sanitize=True)
                    molObjects.append(Chem.MolToSmiles(mol))
                except:
                    mol = Chem.MolFromSmiles(l[8], sanitize=True)
                    molObjects.append(Chem.MolToSmiles(mol))
                    
    return list(set(molObjects))

def smi_reader(file_name, drop_first=False):
    
    mol_SMILES = []
    
    with open(file_name) as f:
        for l in f:
            if drop_first:
                drop_first = False
                continue
                
            l = l.strip().split(",")[0]

            try:
                mol = Chem.MolFromSmiles(l, sanitize=True)
                mol_SMILES.append(Chem.MolToSmiles(mol))
            except:
                print("Invalid\n")            
            
    return list(set(mol_SMILES))


def main(true_inhibitor_path,generated_leads):
    smiles_true = file_reader(true_inhibitor_path)
    
    smiles_generated = smi_reader(generated_leads)
    
    d = dict()
    similarity = []
    for m in smiles_generated:    	
        mol = Chem.MolFromSmiles(m, sanitize=True)
    	
        fp_m = AllChem.GetMorganFingerprint(mol, 3)   	
        dist = [DataStructs.TanimotoSimilarity(fp_m, AllChem.GetMorganFingerprint(Chem.MolFromSmiles(n, sanitize=True), 3)) for n in smiles_true]    
        d[m] = smiles_true[dist.index(max(dist))]
        similarity.append(max(dist))
    
    print("Done")
    

if __name__ == "__main__":

    true_inhibitor = "A2A_receptor_interactions.csv"

    generated = "generated_7719_iter85.smi"
    main(true_inhibitor,generated)