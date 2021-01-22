# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 00:39:28 2019

@author: Tiago
"""
from rdkit import Chem
import tensorflow as tf
from prediction import *
import numpy as np
from Smiles_to_tokens import SmilesToTokens
from predictSMILES import *
from model import Model 
from tqdm import tqdm
from utils import * 
from tqdm import trange
from keras.models import Sequential
from keras import losses
import keras.backend as K
from keras import optimizers
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import QED
from sascorer_calculator import SAscore

sess = tf.compat.v1.InteractiveSession()            
       
class Reinforcement(object):
    def __init__(self, generator, predictor_a2d, predictor_bbb,configReinforce):  
        """
        Constructor for the Reinforcement object.
        Parameters
        ----------
        generator: generative model object that produces string of characters 
            (trajectories)
        predictor: object of any predictive model type
            predictor accepts a trajectory and returns a numerical
            prediction of desired property for the given trajectory
        configReinforce: bunch
            Configuration file containing all the necessary specification and
            parameters. 
        Returns
        -------
        object Reinforcement used for implementation of Reinforcement Learning 
        model to bias the Generator
        """

        super(Reinforcement, self).__init__()
        self.generator_unbiased = generator
        self.generator_biased = generator
        self.generator = generator
        self.configReinforce = configReinforce
        self.generator_unbiased.model.load_weights(self.configReinforce.model_name_unbiased)
        self.generator_biased.model.load_weights(self.configReinforce.model_name_unbiased)
        self.token_table = SmilesToTokens()
        self.table = self.token_table.table
        self.predictor_a2d = predictor_a2d
        self.predictor_bbb = predictor_bbb
        self.get_reward_MO = get_reward_MO
        self.threshold_greedy = 0.1
        self.n_table = len(self.token_table.table)
        self.preds_range =  [3.,1.28,1.284,1.015]#[2.,1.4,1.284,1.015] #3.2,1.29
        self.best_model = '0.5'
#        self.adam = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        self.adam = optimizers.Adam(clipvalue=4)
        self.scalarization_mode = 'chebyshev' # it can be 'linear' or 'chebyshev'

    def custom_loss(self,magic_matrix):
        def lossfunction(y_true,y_pred):
#            return (1/10)* K.sum(((tf.compat.v1.math.log_softmax(y_pred))*y_true)*magic_matrix)
            return (1/self.configReinforce.batch_size)*K.sum(losses.categorical_crossentropy(y_true,y_pred)*magic_matrix)
       
        return lossfunction

    def get_policy_model(self,aux_array):

        self.generator_biased = Sequential()
        self.generator_biased = Model(self.configReinforce) 
        self.generator_biased.model.compile(
                optimizer=self.adam,
                loss = self.custom_loss(aux_array))
        
        return self.generator_biased.model
        

    def policy_gradient(self, gamma=1):    
        """
        Implementation of the policy gradient algorithm.

        Parameters:
        -----------
        self.n_batch: int
            number of trajectories to sample per batch.    
        gamma: float (default 0.97)
            factor by which rewards will be discounted within one trajectory.
            Usually this number will be somewhat close to 1.0.

        Returns
        -------
        This function returns, at each iteration, the graphs with average 
        reward and averaged loss from the batch of generated trajectories 
        (SMILES). Moreover it returns the average reward for QED and KOR properties.
        Also, it returns the used weights and the averaged scaled reward for
         """
         
        pol = 0.5
        cumulative_rewards = []
        cumulative_rewards_a2d = []
        cumulative_rewards_bbb = [] 
        previous_weights = []

        w_a2d = 0.5 
    
        weights = [w_a2d,1-w_a2d]

        # Initialize the variable that will contain the output of each prediction
        dimen = len(self.table)
        states = np.empty(0).reshape(0,dimen)
        pol_rewards_a2d = []
        pol_rewards_bbb = []

        all_rewards = []
        all_losses = []
        # Re-compile the model to adapt the loss function and optimizer to the RL problem
        self.generator_biased.model = self.get_policy_model(np.arange(43))
        self.generator_biased.model.load_weights(self.configReinforce.model_name_unbiased)
        memory_smiles = []
        for i in range(self.configReinforce.n_iterations):
            
            for j in trange(self.configReinforce.n_policy, desc='Policy gradient progress'):
                
                cur_reward = 0
                cur_reward_a2d = 0
                cur_reward_bbb = 0
               
                # Necessary object to transform new generated smiles to one-hot encoding
                token_table = SmilesToTokens()
                aux_matrix = np.zeros((65,1))
                
                ii = 0
                
                for m in range(self.configReinforce.batch_size):
                    # Sampling new trajectory
                    reward = 0
                    uniq = True
                    
                    while reward == 0:
                        predictSMILES =  PredictSMILES(self.generator_unbiased,self.generator_biased,True,self.threshold_greedy,self.configReinforce,False) # generate new trajectory
                        trajectory = predictSMILES.sample() 
       
                        try:                     
                            s = trajectory[0] # because predictSMILES returns a list of smiles strings
                            if 'A' in s: # A is the padding character
                                s = remove_padding(trajectory[0])
                                
                            print("Validation of: ", s) 
        
                            mol = Chem.MolFromSmiles(s)
         
                            trajectory = 'G' + Chem.MolToSmiles(mol) + 'E'
#                                trajectory = 'GCCE'
                        
                            if len(memory_smiles) > 30:
                                    memory_smiles.remove(memory_smiles[0])                                    
                            memory_smiles.append(s)
                            
                            if len(trajectory) > 65:
                                reward = 0
                            else:
                            	rewards = self.get_reward_MO(self.predictor_a2d,self.predictor_bbb,trajectory[1:-1],memory_smiles)
                            	print(rewards)
                            	reward = scalarization(rewards,self.scalarization_mode,weights,self.preds_range,m)

                            print(reward)

                           
                        except:
                            reward = 0
                            print("\nInvalid SMILES!")
        
                    # Converting string of characters to one-hot enconding
                    trajectory_input,_ = token_table.one_hot_encode(token_table.tokenize(trajectory))
                    ti,_ = token_table.one_hot_encode(token_table.tokenize(trajectory))
                    discounted_reward = reward
                    cur_reward += reward
                    cur_reward_a2d += rewards[0]
                    cur_reward_bbb += rewards[1]
                                
                    
                    # "Following" the trajectory and accumulating the loss
                    idxs = 0
                    for p in range(1,len(trajectory_input[0,:,])):
                        
                        state = []
                        state = np.reshape(trajectory_input[0,p,:], [1, dimen])
                        idx = np.nonzero(state)
                        state[idx] = state[:,idx[1]]*discounted_reward
#                            output = self.generator_biased.model.predict(trajectory_input[:,0:p,:])
#
                        inp = ti[:,0:p,:]
              
                        inp_p = padding_one_hot(inp,self.table) # input to padd
                        mat = np.zeros((1,65))
                        mat[:,idxs] = 1

                        if ii == 0:
                            inputs = inp_p
                            aux_matrix = mat
                        else:
                            inputs = np.dstack([inputs,inp_p])
                            aux_matrix = np.dstack([aux_matrix,mat])
        
                        discounted_reward = discounted_reward * gamma
                        
                        states = np.vstack([states, state])
                        ii += 1
                        idxs += 1
                        
                # Doing backward pass and parameters update
             
                states = states[:,np.newaxis,:]
                inputs = np.moveaxis(inputs,-1,0)

                aux_matrix = np.squeeze(aux_matrix)
                aux_matrix = np.moveaxis(aux_matrix,-1,0)
                
                self.generator_biased.model.compile(optimizer = self.adam, loss = self.custom_loss(aux_matrix))
                #update weights based on the provided collection of samples, without regard to any fixed batch size.
                loss = self.generator_biased.model.train_on_batch(inputs,states) # update the weights with a batch
                
                # Clear out variables
                states = np.empty(0).reshape(0,dimen)
                inputs = np.empty(0).reshape(0,0,dimen)

                cur_reward = cur_reward / self.configReinforce.batch_size
                cur_reward_a2d = cur_reward_a2d / self.configReinforce.batch_size
                cur_reward_bbb = cur_reward_bbb / self.configReinforce.batch_size
 
                # serialize model to JSON
                model_json = self.generator_biased.model.to_json()
                with open(self.configReinforce.model_name_biased + "_" + self.scalarization_mode + '_' +str(pol)+".json", "w") as json_file:
                    json_file.write(model_json)
                # serialize weights to HDF5
                self.generator_biased.model.save_weights(self.configReinforce.model_name_biased + "_"+self.scalarization_mode + '_' +str(pol)+".h5")
                print("Updated model saved to disk")
                
                if len(all_rewards) > 2: # decide the threshold of the next generated batch 
                    self.threshold_greedy = compute_thresh(all_rewards[-3:],self.configReinforce.threshold_set)
 
                all_rewards.append(moving_average(all_rewards, cur_reward)) 
                pol_rewards_a2d.append(moving_average(pol_rewards_a2d, cur_reward_a2d)) 
                pol_rewards_bbb.append(moving_average(pol_rewards_bbb, cur_reward_bbb))                   

                all_losses.append(moving_average(all_losses, loss))
    
            plot_training_progress(all_rewards,all_losses)
            plot_individual_rewds(pol_rewards_a2d,pol_rewards_bbb)
        cumulative_rewards.append(np.mean(all_rewards[-15:]))
        cumulative_rewards_a2d.append(np.mean(pol_rewards_a2d[-15:]))
        cumulative_rewards_bbb.append(np.mean(pol_rewards_bbb[-15:]))
        pol+=1

        plot_MO(cumulative_rewards_a2d,cumulative_rewards_bbb,cumulative_rewards,previous_weights)
        return cumulative_rewards_a2d,cumulative_rewards_bbb,cumulative_rewards,previous_weights
    
    def test_generator(self, n_to_generate,iteration, original_model):
        """
        Function to generate molecules with the specified generator model. 

        Parameters:
        -----------

        n_to_generate: Integer that indicates the number of molecules to 
                    generate
        iteration: Integer that indicates the current iteration. It will be 
                   used to build the filename of the generated molecules                       
        original_model: Boolean that specifies generator model. If it is 
                        'True' we load the original model, otherwise, we 
                        load the fine-tuned model 

        Returns
        -------
        The plot containing the distribuiton of the property we want to 
        optimize. It saves one file containing the generated SMILES strings. Also,
        this function returns the SMILES strings, the predictions for KOR affinity
        and QED, and, also, the percentages of valid and unique molecules.
        """
        
        
        if original_model:
             self.generator.model.load_weights(self.configReinforce.model_name_unbiased)
             print("....................................")
             print("original model load_weights is DONE!")
        else:
             self.generator.model.load_weights(self.configReinforce.model_name_biased + "_" + self.scalarization_mode + "_" + self.best_model+ ".h5")
             print("....................................")
             print("updated model load_weights is DONE!")
        
        generated = []
        pbar = tqdm(range(n_to_generate))
        for i in pbar:
            pbar.set_description("Generating molecules...")
            predictSMILES = PredictSMILES(self.generator,None,False,self.threshold_greedy,self.configReinforce,True)
            generated.append(predictSMILES.sample())
    
        sanitized,valid = canonical_smiles(generated,sanitize=True, throw_warning=False)# validar 
        
        san_with_repeated = []
        for smi in sanitized:
            if len(smi) > 1:
                san_with_repeated.append(smi)
        
        unique_smiles = list(np.unique(san_with_repeated))
        percentage_unq = (len(unique_smiles)/len(san_with_repeated))*100
        
        # prediction pIC50 for a2d
        prediction_a2d = self.predictor_a2d.predict(san_with_repeated,"a2d")
        
        # prediction pIC50 KOR
        prediction_bbb = self.predictor_bbb.predict(san_with_repeated,"bbb")
        
        vld = plot_hist(prediction_a2d,n_to_generate,valid,"a2d")
        evaluate_bbb(prediction_bbb)
        
        div = diversity(unique_smiles)
            
        with open(self.configReinforce.file_path_generated + '_' + str(len(san_with_repeated)) + '_iter'+str(iteration)+".smi", 'w') as f:
            for i,cl in enumerate(san_with_repeated):
                data = str(san_with_repeated[i]) + " ," +  str(prediction_a2d[i])+ ", " +str(prediction_bbb[i]) 
                f.write("%s\n" % data)  
                
        
        return san_with_repeated,prediction_a2d,prediction_bbb,vld,percentage_unq
                    

    def compare_models(self, n_to_generate,individual_plot):
        """
        Function to generate molecules with the both models

        Parameters:
        -----------
        n_to_generate: Integer that indicates the number of molecules to 
                    generate
                    
        individual_plot: Boolean that indicates if we want to represent the 
                         property distribution of the pre-trained model.

        Returns
        -------
        The plot that contains the distribuitons of the properties we want to 
        optimize originated by the original and fine-tuned models. Besides 
        this, it saves a "generated.smi" file containing the valid generated 
        SMILES and the respective property value in "generated\" folder. Also,
        it returns the differences between the means of the original and biased
        predictions for both properties, the percentage of valid and the 
        internal diversity.
        """

        self.generator.model.load_weights(self.configReinforce.model_name_unbiased)
        print("\n --------- Original model LOADED! ---------")
        
        generated_unb = []
        pbar = tqdm(range(n_to_generate))
        for i in pbar:
            pbar.set_description("Generating molecules...")
            predictSMILES = PredictSMILES(self.generator,None,False,self.threshold_greedy,self.configReinforce,False)
            generated_unb.append(predictSMILES.sample())
    
        sanitized_unb,valid_unb = canonical_smiles(generated_unb, sanitize=True, throw_warning=False) # validar 
        unique_smiles_unb = list(np.unique(sanitized_unb))[1:]
        
        san_with_repeated_unb = []
        
        for smi in sanitized_unb:
            if len(smi)>1:
                san_with_repeated_unb.append(smi)
                
        
        #prediction a2d
        prediction_a2d_unb = self.predictor_a2d.predict(san_with_repeated_unb,"a2d")

        #prediction kor
        prediction_bbb_unb = self.predictor_bbb.predict(san_with_repeated_unb,"bbb")
        
        desirable_unb = 0
        for pred in prediction_a2d_unb:
            if pred >= 6.5:
                desirable_unb +=1
        perc_desirable_unb = desirable_unb/len(san_with_repeated_unb)
        
        if individual_plot:
            plot_hist(prediction_a2d_unb,n_to_generate,valid_unb,"a2d")
            evaluate_bbb(prediction_bbb_unb)            
        
        # Load Biased Generator Model 
        self.generator.model.load_weights(self.configReinforce.model_name_biased + "_" + self.scalarization_mode +  "_" + self.best_model + ".h5")
        print("\n --------- Updated model LOADED! ---------")
        
        generated_b = []
        pbar = tqdm(range(n_to_generate))
        for i in pbar:
            pbar.set_description("Generating molecules...")
            predictSMILES = PredictSMILES(self.generator,None,False,self.threshold_greedy,self.configReinforce,True)
            generated_b.append(predictSMILES.sample())
    
        sanitized_b,valid_b = canonical_smiles(generated_b, sanitize=True, throw_warning=False) # validar 
                
        san_with_repeated_b = []
        for smi in sanitized_b:
            if len(smi) > 1:
                san_with_repeated_b.append(smi)
        
        unique_smiles_b = list(np.unique(san_with_repeated_b))
        percentage_unq_b = (len(unique_smiles_b)/len(san_with_repeated_b))*100
        
        #prediction kor
        prediction_a2d_b = self.predictor_a2d.predict(san_with_repeated_b, "a2d")
        
        #prediction kor
        prediction_bbb_b = self.predictor_bbb.predict(san_with_repeated_b,"bbb")
      
        # plot both distributions together and compute the % of valid generated by the biased model 
        dif_a2d, valid_a2d = plot_hist_both(prediction_a2d_unb,prediction_a2d_b,n_to_generate,valid_unb,valid_b,"a2d")
        dif_bbb = evaluate_bbb(prediction_bbb_unb,prediction_bbb_b)
#        dif_kor, valid_kor = plot_hist_both(prediction_bbb_unb,prediction_bbb_b,n_to_generate,valid_unb,valid_b,"bbb")
        
        # Compute the internal diversity
        div = diversity(unique_smiles_b)
        
        desirable = 0
        for pred in prediction_a2d_b:
            if pred >= 6.5:
                desirable +=1
        perc_desirable_b = desirable/len(san_with_repeated_b)        
        
        return dif_a2d,dif_bbb,valid_a2d,div,percentage_unq_b,perc_desirable_b,perc_desirable_unb

    def drawMols(self):
        """
        Function that draws chemical graphs of compounds generated by the opmtized
        model.

        Parameters:
        -----------
        self: it contains the Generator and the configuration parameters

        Returns
        -------
        This function returns a figure with the specified number of molecular 
        graphs indicating the pIC50 for KOR and the QED.
        """
        DrawingOptions.atomLabelFontSize = 50
        DrawingOptions.dotsPerAngstrom = 100
        DrawingOptions.bondLineWidth = 3
                  
        self.generator.model.load_weights(self.configReinforce.model_name_biased + "_" + self.scalarization_mode + "_" + self.best_model+ ".h5")

        generated = []
        pbar = tqdm(range(self.configReinforce.n_to_generate))
        for i in pbar:
            pbar.set_description("Generating molecules...")
            predictSMILES = PredictSMILES(self.generator,None,False,self.threshold_greedy,self.configReinforce)
            generated.append(predictSMILES.sample())
    
        sanitized,valid = canonical_smiles(generated,sanitize=True, throw_warning=False) 
        
        unique_smiles = list(np.unique(sanitized))[1:]
        
        # prediction pIC50 KOR
        prediction_kor = self.predictor_kor.predict(unique_smiles)
        
        # prediction qew
        mol_list = smiles2mol(unique_smiles)
        prediction_qed = qed_calculator(mol_list)
                
        ind = np.random.randint(0, len(mol_list), self.configReinforce.n_to_draw)
        mols_to_draw = [mol_list[i] for i in ind]
        
        legends = []
        for i in ind:
            legends.append('pIC50 for KOR: ' + str(round(prediction_kor[i],2)) + '|| QED: ' + str(round(prediction_qed[i],2)))
        
        img = Draw.MolsToGridImage(mols_to_draw, molsPerRow=1, subImgSize=(300,300), legends=legends)
            
        img.show()
        
    def property_checker(self, n_to_generate):
        
        """
        Function to generate molecules with the specified generator model. 

        Parameters:
        -----------

        n_to_generate: Integer that indicates the number of molecules to 
                    generate
        iteration: Integer that indicates the current iteration. It will be 
                   used to build the filename of the generated molecules                       
        original_model: Boolean that specifies generator model. If it is 
                        'True' we load the original model, otherwise, we 
                        load the fine-tuned model 

        Returns
        -------
        The plot containing the distribuiton of the property we want to 
        optimize. It saves one file containing the generated SMILES strings.
        """
#        
        sample = True
        self.generator.model.load_weights(self.configReinforce.model_name_biased + "_" + self.scalarization_mode + "_" + self.best_model+ ".h5")
        print("....................................")
        print("updated model load_weights is DONE!")
        
        generated = []
        pbar = tqdm(range(n_to_generate))
        for i in pbar:
            pbar.set_description("Generating molecules...")
            predictSMILES = PredictSMILES(self.generator,None,False,self.threshold_greedy,self.configReinforce,sample)
            generated.append(predictSMILES.sample())

        sanitized,valid = canonical_smiles(generated,sanitize=True, throw_warning=False)
        
        san_with_repeated = []
        for smi in sanitized:
            if len(smi) > 1:
                san_with_repeated.append(smi)
        
        unique_smiles = list(set(san_with_repeated))
        percentage_unq = (len(unique_smiles)/len(san_with_repeated))*100
        
        vld = (valid/n_to_generate)*100
        
        prediction_a2d = self.predictor_a2d.predict(unique_smiles,"a2d")
        prediction_bbb = self.predictor_bbb.predict(unique_smiles,"bbb")
        
        
#        desirable_mols = []
#        for idx in range(0,len(prediction)): 
#            if prediction[idx] > 6.5:
#                desirable_mols.append(san_with_repeated[idx])
#                
#        perc_desirable = len(desirable_mols)/len(san_with_repeated)
#        perc_unique_desirable = len(list(set(desirable_mols)))/len(desirable_mols)
        
            
        # Compute the internal diversity
        div = diversity(unique_smiles)
        
        with open(self.configReinforce.file_path_generated + ".smi", 'w') as f:
            f.write("Number of molecules: %s\n" % str(len(unique_smiles)))
            f.write("Percentage of valid molecules: %s\n" % str(vld))
            f.write("Internal Tanimoto similarity: %s\n\n" % str(div))
            f.write("SMILES, pIC50, Active, BBB, MW, logP, SAS, QED\n")
            for i,smi in enumerate(unique_smiles):
                mol = Chem.MolFromSmiles(smi)
                list_mol = smiles2mol(smi)
                prediction_sas = SAscore(list_mol)
                
                active = "0"
                permeable = "0"
                if prediction_a2d[i] > 6.5:
                    active = "1"
                
                if prediction_bbb[i]> 0.98:
                    permeable = "1"


                q = QED.qed(mol)
                mw, logP = Descriptors.MolWt(mol), Crippen.MolLogP(mol)
                data = str(unique_smiles[i]) + " ," +  str(np.round(prediction_a2d[i],2)) + " ," + active + " ," + permeable + " ," + str(np.round(mw,2)) + " ," + str(np.round(logP,2))+ " ," + str(np.round(prediction_sas[0],2)) + " ," + str(np.round(q,2))
                f.write("%s\n" % data)  