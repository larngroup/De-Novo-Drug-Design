# -*- coding: utf-8 -*-
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input,GRU, Dropout
from tensorflow.keras.callbacks import  ModelCheckpoint
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tokens import tokens_table
from tensorflow.keras.callbacks import EarlyStopping
from utils import specificity,sensitivity,matthews_correlation
from matplotlib import pyplot as plt

class BaseModel(object):
    def __init__(self, config,data,searchParams,descriptor_type):
        """
        This function implements the DNN-based QSAR models.
        ----------
        config: Configuration file (It contains all the necessary parameters)
        data: List containing Training and Validation data
        searchParams: Boolean indicating if we are searching for the best parameters
                      or implementing the optimal model
        descriptor_type: String that specifies the model descriptor (SMILES or ECFP)
        Returns
        -------
        This function outputs the trained QSAR model that was specified in the input
        """
        self.config = config
        self.model = None
        self.data = data
        # RNN parameters
        self.dropout = self.config.dropout
        self.learning_rate = self.config.lr
        self.n_units = self.config.n_units
        self.epochs = self.config.num_epochs
        self.activation_rnn = self.config.activation_rnn
        self.batch_size = self.config.batch_size
        self.rnn = self.config.rnn
        # Fully connected nn parameters
        self.units_dense1 = self.config.units_dense1
        self.units_dense2 = self.config.units_dense2
        self.units_dense3 = self.config.units_dense3
        self.dropout_dense = self.config.dropout_dense
        
        self.searchParams = searchParams
        self.descriptor_type = descriptor_type
        
class Model(BaseModel):
    
    def __init__(self, config,data,searchParams,descriptor_type):
        super(Model, self).__init__(config,data,searchParams,descriptor_type)
        token_table = tokens_table()
        self.build_model(token_table.table_len)
        
    
    def build_model(self, n_table):
        """
        Depending on the descriptor type, it implements two different architectures
        For SMILES strings, we use a RNN. For the ECFP vectors, we use FCNN with 3 FC layers
        """
        self.n_table = n_table
        self.model = Sequential()
        
        X_train = self.data[0]
        y_train = self.data[1]
        X_val = self.data[4]
        y_val = self.data[5]
        
        if self.descriptor_type == 'SMILES':
            self.model = Sequential()
            self.model.add(Input(shape=(self.config.input_length,)))
            self.model.add(Embedding(n_table, 256, input_length=self.config.input_length))
    
            if self.rnn == 'LSTM':
                self.model.add(LSTM(self.n_units, return_sequences=True, input_shape=(None,512,self.config.input_length),dropout = self.dropout))
                self.model.add(LSTM(512,dropout = self.dropout))
    
        
            elif self.rnn == 'GRU':
                self.model.add(GRU(256, return_sequences=True, input_shape=(None,256,self.config.input_length),dropout = self.dropout))
#                self.model.add(GRU(self.n_units, return_sequences=True, dropout = self.dropout))
                self.model.add(GRU(256,dropout = self.dropout))
    
            self.model.add(Dense(256, activation='relu'))
#            self.model.add(Dropout(self.dropout_dense))
            self.model.add(Dense(128, activation='relu'))
#            self.model.add(Dropout(self.dropout_dense))
            self.model.add(Dense(1, activation='sigmoid'))
            
            
        else:
            # ECFP-based model
            self.model = Sequential()
            self.model.add(Input(shape=(2048,)))
            self.model.add(Dense(self.units_dense1, activation='relu'))
            self.model.add(Dropout(self.dropout_dense))
            self.model.add(Dense(self.units_dense2, activation='relu'))
            self.model.add(Dropout(self.dropout_dense))
            self.model.add(Dense(self.units_dense3, activation='relu'))
            self.model.add(Dropout(self.dropout_dense))
            self.model.add(Dense(1, activation='sigmoid'))
            
        
        opt = Adam(lr=self.learning_rate, beta_1=self.config.beta_1, beta_2=self.config.beta_2, amsgrad=False)
            	
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15, restore_best_weights=True)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        self.model.compile(loss=self.config.loss_criterium, optimizer = opt,  metrics=['accuracy','AUC',specificity,sensitivity,matthews_correlation])

#        lrateh_size=self.config.batch_size,validation_data=(X_test, y_test),callbacks=[lrate])
        result = self.model.fit(X_train, y_train,
          epochs=self.epochs,
          batch_size=self.batch_size,validation_data=(X_val, y_val),callbacks=[es,mc])
        self.model.summary()

        #-----------------------------------------------------------------------------
        # Plot learning curves including R^2 and RMSE
        #-----------------------------------------------------------------------------
        
#        # plot training curve for R^2 (beware of scale, starts very low negative)
#        plt.plot(result.history['accuracy'])
#        plt.plot(result.history['val_accuracy'])
#        plt.title('model Accuracy')
#        plt.ylabel('Acc')
#        plt.xlabel('epoch')
#        plt.legend(['train', 'test'], loc='upper left')
#        plt.show()
                   
        # plot training curve for rmse
        plt.plot(result.history['specificity'])
        plt.plot(result.history['val_specificity'])
        plt.title('specificity')
        plt.ylabel('spec')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        # plot training curve for rmse
        plt.plot(result.history['sensitivity'])
        plt.plot(result.history['val_sensitivity'])
        plt.title('sensitivity')
        plt.ylabel('sensi')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        if self.searchParams:
            metrics = self.model.evaluate(self.data[2],self.data[3])        
            print("\n\nMean_squared_error: ",metrics[0],"\nR_square: ", metrics[1], "\nRoot mean square: ",metrics[2], "\nCCC: ",metrics[3])
            values= [self.dropout,self.batch_size,self.learning_rate,self.n_units,
                 self.rnn,self.activation_rnn,self.epochs,metrics[0],metrics[1],metrics[2],metrics[3]] 
                          
            file=[i.rstrip().split(',') for i in open('grid_results.csv').readlines()]
            file.append(values)
            file=pd.DataFrame(file)
            file.to_csv('grid_results.csv',header=None,index=None)
        else:
            
            filepath=""+self.config.checkpoint_dir + ""+ self.config.model_name
#             serialize model to JSON
            model_json = self.model.to_json()
            with open(str(filepath + ".json"), "w") as json_file:
                json_file.write(model_json)
                # serialize weights to HDF5
            self.model.save_weights(str(filepath + ".h5"))
            print("Saved model to disk")
        
