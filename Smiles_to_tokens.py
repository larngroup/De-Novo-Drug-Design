import numpy as np

class SmilesToTokens(object):
    """
    Function that performs SMILES tokenization and transformation to one-hot 
    encoding using all possible tokens including starting, ending and padding 
    characters.
    Parameters
    ----------
    smiles: List with SMILES strings
    Returns
    -------
    This function returns an one-hot encoding array.
    """
    def __init__(self):
        atoms = [
                 'H','B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I'
                ]
        special = [
                '(', ')', '[', ']', '=', '#', '@', '*', '%', '0', '1', '2',
                '3', '4', '5', '6', '7', '8', '9', '.', '/', '\\', '+', '-',
                 'c', 'n', 'o', 's','p'
                ]
        padding = ['G', 'A', 'E'] #Go, Padding ,End
        self.table = sorted(atoms, key=len, reverse=True) + special + padding
        self.table_len = len(self.table)

    def tokenize(self, smiles):
        N = len(smiles)
        i = 0
        j= 0
        token = []
        while (i < N):
            for j in range(self.table_len):
                symbol = self.table[j]
                if symbol == smiles[i:i + len(symbol)]:
                    token.append(symbol)
                    i += len(symbol)
                    break
        return token

    def one_hot_encode(self, trans_char): #create one hot encode table
        transl_one_hot = {} #create dictionary
        for i, char in enumerate(self.table): #create one hot encode vector for each character
            lista = np.zeros(self.table_len) #create zero list for each char
            lista[i] = 1 #set 1 on the correct position
            transl_one_hot[char] = lista #save list in dictionary

        result = np.array([transl_one_hot[s] for s in trans_char]) #find the vector corresponding to the character
        result = result.reshape(1, result.shape[0], result.shape[1])
        #print("\nTransl_one_hot:\n",transl_one_hot,"\n")
        return result, transl_one_hot
