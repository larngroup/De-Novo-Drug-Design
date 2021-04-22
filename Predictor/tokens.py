# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:08:18 2019

@author: Tiago
"""

class tokens_table(object):
    """
    This class has a list with all the necessary tokens to build the SMILES strings
    Note that a token is not necessarily a character. It can be a two characters like Br.
    ----------
    tokens: List with symbols and characters used in SMILES notation.
    """
    def __init__(self):
        tokens = ['[C@H]','[C@@H]','[C@@]','[C@]','Cl', 'Br', 'I','[CH2-]','[CH-]','[O+]',
                    '[n+]','[O-]','[Cl]','[Cl-]','[nH]','[N+]','[N+]','[NH+]','[NH-]','[OH-]',
                  '[B-]','[C+]','[C-]','[CH]','[H+]','[C]','[S]','[H]', 'H','B', 'C', 'N', 'I',
                  'O', 'P', 'S', 'F','[N+]','[Br-]','[N@]','[P]','[N@@]','[NH2+]',
                 '[N-]','[N]','[O]','[P+]','[S+]','[s+]','[NH]','(', ')','=', '#', 
                 '@', '*', '%', '0', '1', '2','3', '4', '5', '6', '7', '8', 
                 '9', '.', '/', '\\', '+', '-', 'c', 'n', 'o', 's','p', ' ']
#        tokens = [
#                 'H','Se','As','se','Si','Cl', 'Br','B', 'C', 'N', 'O', 'P', 'S', 'F', 'I',
#                '(', ')', '[', ']', '=', '#', '@', '*', '%', '0', '1', '2',
#                '3', '4', '5', '6', '7', '8', '9', '.', '/', '\\', '+', '-',
#                 'c', 'n', 'o', 's','p', ' ']
#        
  
        self.table = tokens
        self.table_len = len(self.table)