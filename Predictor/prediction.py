# -*- coding: utf-8 -*-
from tensorflow.keras.models import model_from_json
import numpy as np
#from sklearn.externals import joblib
from utils import *
from sklearn.metrics import accuracy_score
from utils import specificity,sensitivity,matthews_correlation
from tensorflow.keras.optimizers import Adam

class Predictor(object):
    def __init__(self, config, tokens,model_type,descriptor_type):
        """
        This class loads the previously trained models, whether they are the 
        DNN-based or the standard QSAR models, to evaluate them with the test set
        ----------
        config: Configuration file
        tokens: List with all possible tokens forming a SMILES string
        model_type: String that indicates the type of model (RNN, SVR, KNN or RF)
        descriptor_type: String that indicates the used descriptor (SMILES or ECFP)
        Returns
        -------
        This function loads the trained models from the cross-validation step and evaluates 
        them to predict the actual output of some molecules or to use with the test set.
        """
        super(Predictor, self).__init__()
        self.tokens = tokens
        self.config = config
        self.model_type = model_type
        self.descriptor_type = descriptor_type
        loaded_models = []
        for i in range(5):
            
            if model_type == 'dnn':
                json_file = open(self.config.checkpoint_dir + "model"+str(i)+".json", 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                loaded_model = model_from_json(loaded_model_json)
                # load weights into new model
                loaded_model.load_weights(self.config.checkpoint_dir + "model"+str(i)+".h5")
            
            else:
                # Load the model from the file 
                loaded_model = joblib.load(self.config.checkpoint_dir + "model"+str(i)+".pkl") 
            

            print("Model " + str(i) + " loaded from disk!")
            loaded_models.append(loaded_model)
        
        self.loaded_models = loaded_models
        
    def predict(self, smiles,data):
        """
        This method predicts the desired property of newly generated SMILES
        ----------
        smiles: Newly generated molecules
        data: values to denormalize the predictions
        Returns
        -------
        This function returns the prediction for KOR affinity 
        """        
         
        if self.model_type == 'dnn' and self.descriptor_type == 'SMILES':
            
            smiles_padded,kl = pad_seq(smiles,self.tokens,self.config.paddSize)
            d = smilesDict(self.tokens)  
            tokenized_smiles = tokenize(smiles_padded,self.tokens)   
            data_2_predict = smiles2idx(tokenized_smiles,d)
            
        else:
            data_2_predict = SMILES_2_ECFP(smiles)
            
        prediction = []
            
        for m in range(len(self.loaded_models)):
                prediction.append(self.loaded_models[m].predict(data_2_predict))
                
        prediction = np.array(prediction).reshape(len(self.loaded_models), -1)
        
        prediction = denormalization(prediction,data)
                
        prediction = np.mean(prediction, axis = 0)
     
        return prediction
            
    def evaluator(self,data):
        """
        This function evaluates the QSAR models previously trained
        ----------
        data: List with testing SMILES and testing labels
        Returns
        -------
        This function evaluates the model with the training data
        """
        
        print("\n------- Evaluation with test set -------")
        smiles = data[2]
        label = data[3]
        metrics = []
        prediction = []
        opt = Adam(lr=self.config.lr, beta_1=self.config.beta_1, beta_2=self.config.beta_2, amsgrad=False)
        for m in range(len(self.loaded_models)):
            self.loaded_models[m].compile(loss=self.config.loss_criterium, optimizer = opt,  metrics=['accuracy','AUC',specificity,sensitivity,matthews_correlation])
            metrics.append(self.loaded_models[m].evaluate(smiles,label))
            prediction.append(self.loaded_models[m].predict(smiles))
                
        prediction = np.array(prediction).reshape(len(self.loaded_models), -1)
        prediction = np.mean(prediction, axis = 0)
        
        if self.model_type == 'dnn':
            metrics = np.array(metrics).reshape(len(self.loaded_models), -1)
            metrics = metrics[:,1:]
            
        metrics = np.mean(metrics, axis = 0)
      
        return metrics
