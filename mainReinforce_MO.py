# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 19:39:49 2019

@author: Tiago
"""
from reinforce_MO import Reinforcement
from keras.models import Sequential
from model import Model  
from prediction import Predictor
import numpy as np
from utils import *

config_file = 'configReinforce.json' # Configuration file 
        
def main():
    # load configuration file
    configReinforce,exp_time=load_config(config_file)
    
    # Load Generator object
    generator_model = Sequential()
    generator_model=Model(configReinforce)
    generator_model.model.load_weights(configReinforce.model_name_unbiased)

     # Load the Predictor object of A2D affinity 
    predictor_a2d = Predictor(configReinforce,'a2d')
    
    # Load the Predictor object of BBB property
    predictor_bbb = Predictor(configReinforce,'bbb')

    # Initialize lists to evaluate the model
    difs_a2d = [] # List with the differences between the averages of affinity distributions for a2d (G_0 and G_optimized)
    difs_bbb = []
    divs = [] # List with the internal diversities of the G_optimized generated molecules 
    perc_valid = [] # List with the % of valid SMILES generated by G_optimized
    uniqs = [] # List with the % of unique SMILES strings 

    # Create Reinforcement Learning (RL) object
    RL_obj = Reinforcement(generator_model, predictor_a2d, predictor_bbb, configReinforce)
    
#    RL_obj.drawMols()
#    RL_obj.property_checker(configReinforce.n_to_generate)
    
     # SMILES generation test with the unbiased Generator
#    smiles_original, prediction_original_a2d,prediction_original_bbb,valid,unique = RL_obj.test_generator(configReinforce.n_to_generate,0,True)

    # RL training  
    cumulative_rewards_bbb,cumulative_rewards_a2d,cumulative_rewards,previous_weights = RL_obj.policy_gradient()
    
    # SMILES generation test after 60 RL training iterations 
#    smiles_iteration85,prediction_iteration85_a2d,prediction_iteration85_bbb,valid,unique = RL_obj.test_generator(configReinforce.n_to_generate,85, False)
   
    # Plot the changes in the distribution after applying RL
    plot_evolution(prediction_original_a2d,prediction_iteration85_a2d,'a2d')
#    plot_evolution(prediction_original_bbb,prediction_iteration85_bbb,'bbb')
#    
#    # Other way of evaluating the differences before and after applying RL. It 
#    # evaluates the internal diversity, validity and uniqueness
    for k in range(20):
        print("\nGeneration test:" + str(k))
        dif_a2d,dif_bbb,valid,div,unique,active_unbiased,active_biased = RL_obj.compare_models(configReinforce.n_to_generate,True)
        difs_a2d.append(dif_a2d)
        difs_bbb.append(dif_bbb)
        divs.append(div)
        perc_valid.append(valid)
        uniqs.append(unique)
        
    print("\nMean value difference for A2DR: " + str(np.mean(difs_a2d)))
    print("Mean value difference for BBB: " + str(np.mean(difs_bbb)))
    print("Mean value diversity: " + str(np.mean(divs)))
    print("Mean value validity: " + str(np.mean(perc_valid)))
    print("Mean value uniqueness: " + str(np.mean(uniqs)))
if __name__ == '__main__':
    main()