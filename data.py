import numpy as np
import os
import glob
import pandas as pd
import re            # To match regular expression for extracting labels
from configuration import GAMMA
import tensorflow as tf
from tensorflow import keras
from configuration import FULL_PATH, FEATURE_NUMBER, MAX_TRACE_LENGTH




def data_generator(file_list, batch_size):
    i = 0
    
   
    dir_game=[]
   
    while True:
        if i*batch_size >= len(file_list):  # This loop is used to run the generatorindefinitely.
            i = 0
            
            #np.random.shuffle(file_list)
        else:
            file_chunk = file_list[i*batch_size:(i+1)*batch_size] 
           
            labels = []
            
            data = []
            trace_b=[]
            y_b=[]
            for file in file_chunk:
               
                dir_game=file 
                y_batch=[]
                
                nn=0
                temp = pd.read_csv(open(os.path.join(FULL_PATH,file),'r'),header=None) # Change this line to read any other type of file 
               
                rows=temp.shape[0]
                trace_b.append(rows)
                
               
                tp=temp.iloc[:,1:87]

                
                data.extend(tp.values.reshape(1,-1))
                
                label=temp.iloc[-1,87]

               
                
                for p in range(0, rows):
		    if p+1==rows and label==1:
                        y_home = float(1)
		        y_away = float(0)
		        y_end = float(1)
		        y_batch.append((y_home, y_away, y_end))
                                             
		        break
		    if p+1==rows and label==-1:
		        y_home = float(0)
		        y_away = float(1)
		        y_end = float(1)
		        y_batch.append((y_home, y_away, y_end))
                        
		        break      
		        
		    if p+1<rows :
		        y_home = 0
		        y_away = 0
		        y_end =  0
		        y_batch.append((y_home, y_away, y_end))


                    

                y_b.extend(np.asarray(y_batch).reshape(1,-1))
                              
            
            trace_b = np.asarray(trace_b)
            
            
            data = np.asarray(data)
            
            data=tf.keras.preprocessing.sequence.pad_sequences(data,maxlen=MAX_TRACE_LENGTH*FEATURE_NUMBER, padding='post', dtype='float32')
            data=np.array(data)
            
            y_b=tf.keras.preprocessing.sequence.pad_sequences(y_b,maxlen=MAX_TRACE_LENGTH*3, padding='post', dtype='float32')
            y_b = np.asarray(y_b)
            
            yield data, y_b,trace_b,dir_game
            i = i + 1








