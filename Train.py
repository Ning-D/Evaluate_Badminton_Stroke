import csv
import tensorflow as tf
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from natsort import natsorted
import glob
import pandas as pd
import re            
from data import data_generator
from td_prediction_lstm_V3 import td_prediction_lstm_V3

from utils import handle_trace_length, get_together_training_batch, compromise_state_trace_length
from configuration import FULL_PATH, MODEL_TYPE, MAX_TRACE_LENGTH, FEATURE_NUMBER, BATCH_SIZE, GAMMA, H_SIZE, \
    model_train_continue, FEATURE_TYPE, ITERATE_NUM, learning_rate, SPORT, save_mother_dir
import sys

np.set_printoptions(threshold=sys.maxsize)
random.seed(0)



LOG_DIR = save_mother_dir + "/models/hybrid_sl_log_NN/Scale-three-cut_together_log_train_feature" + str(
    FEATURE_TYPE) + "_batch" + str(
    BATCH_SIZE) + "_iterate" + str(
    ITERATE_NUM) + "_lr" + str(
    learning_rate) + "_" + str(MODEL_TYPE) + "_MaxTL" + str(MAX_TRACE_LENGTH)
SAVED_NETWORK = save_mother_dir + "/models/hybrid_sl_saved_NN/Scale-three-cut_together_saved_networks_feature" + str(
    FEATURE_TYPE) + "_batch" + str(
    BATCH_SIZE) + "_iterate" + str(
    ITERATE_NUM) + "_lr" + str(
    learning_rate) + "_" + str(MODEL_TYPE) + "_MaxTL" + str(MAX_TRACE_LENGTH)


file_list = [file for file in os.listdir(FULL_PATH) if file.endswith('.csv')]
file_list.sort(key=lambda x:(os.path.splitext(x)[0][0], int(os.path.splitext(x)[0][1:])))

random.shuffle(file_list)

number_of_total_game = len(file_list)
print(number_of_total_game)
split_ratio = 0.95
split_index = int(number_of_total_game * split_ratio)
files=file_list[0:split_index]
number_of_total_game = len(files)
print("Total number of files: ",number_of_total_game)


def normalize(v, axis=-1, order=2):
    l2 = np.linalg.norm(v, ord = order, axis=axis, keepdims=True)
    l2[l2==0] = 1
    return v/l2





def write_game_average_csv(data_record):
    """
    write the cost of training
    :param data_record: the recorded cost dict
    """
    try:
        if os.path.exists(LOG_DIR + '/avg_cost_record.csv'):
            with open(LOG_DIR + '/avg_cost_record.csv', 'a') as csvfile:
                fieldnames = (data_record[0]).keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for record in data_record:
                    writer.writerow(record)
        else:
            with open(LOG_DIR + '/avg_cost_record.csv', 'w') as csvfile:
                fieldnames = (data_record[0]).keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for record in data_record:
                    writer.writerow(record)
    except:
        if os.path.exists(LOG_DIR + '/avg_cost_record2.csv'):
            with open(LOG_DIR + '/avg_cost_record.csv', 'a') as csvfile:
                fieldnames = (data_record[0]).keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for record in data_record:
                    writer.writerow(record)
        else:
            with open(LOG_DIR + '/avg_cost_record2.csv', 'w') as csvfile:
                fieldnames = (data_record[0]).keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for record in data_record:
                    writer.writerow(record)


def train_network(sess, model):
    """
    training thr neural network game by game
    :param sess: session of tf
    :param model: nn model
    :return:
    """
    L_home=[]
    L_away=[]
    L_end=[]
    x_list=[]
    y_list=[]
    fig,axs=plt.subplots(2)
    game_number = 0
    global_counter = 0
    converge_flag = False

    # loading network
    saver = tf.train.Saver(max_to_keep=1000)
    merge = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    sess.run(tf.global_variables_initializer())
    if model_train_continue:
        checkpoint = tf.train.get_checkpoint_state(SAVED_NETWORK)
        if checkpoint and checkpoint.model_checkpoint_path:
            check_point_game_number = int((checkpoint.model_checkpoint_path.split("-"))[-1])
            game_number_checkpoint = check_point_game_number % number_of_total_game
            game_number = check_point_game_number
            game_starting_point = 0
            saver.restore(sess, checkpoint.model_checkpoint_path)
          
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
   
    game_diff_record_all = []
    TT=0
    #while True:

    for epcho in range(1,ITERATE_NUM+1):        
       
        Loss_total=0
        l_all=0
        acc_total=0
        game_diff_record_dict = {}
       
        iteration_now = game_number / number_of_total_game + 1
        game_diff_record_dict.update({"Iteration": iteration_now})
        generated_data = data_generator(files, batch_size = BATCH_SIZE)  
               

        num = 0
        for data, target,trace,_ in generated_data:
            batch_loss=0
           
            v_diff_record = []
            game_number += 5
            
            game_cost_record = []
            state_input=data
            s_t0 = state_input
            I=s_t0
            I=s_t0.reshape(BATCH_SIZE,-1,FEATURE_NUMBER)
            T=0
       
          
                
            L_home=[]
            L_away=[]
            L_end=[]

            [outputs_t1, readout_t1_batch] = sess.run([model.outputs, model.read_out],
                                                          feed_dict={model.trace_lengths: trace,
                                                                     model.rnn_input: I})

                

            target=target.reshape(BATCH_SIZE,-1,3)
            TT=target
           
            
            
                
            for th in range(BATCH_SIZE):
                for pp in range(trace[th]):
		    if pp+1<trace[th]:
		        target[th][pp][0] = GAMMA*readout_t1_batch[th][pp+1][0]
		        target[th][pp][1] = GAMMA*readout_t1_batch[th][pp+1][1]
		        target[th][pp][2] = GAMMA*readout_t1_batch[th][pp+1][2]
            
            
            [diff, read_out, cost_out, summary_train, _] = sess.run(
                    [model.diff, model.read_out, model.cost, merge, model.train_step],
                    feed_dict={model.y: target,
                               model.trace_lengths: trace,
                               model.rnn_input: I})

            v_diff_record.append(diff)

           
            game_cost_record.append(cost_out)
                
            T=T+1

    


           
           
            readout_t1_batch_R=read_out
                      
            target_R=target
            MSEcost = np.mean(np.square(target_R - readout_t1_batch_R))
            
            num = num + 1
            
            Loss_total=Loss_total+MSEcost
            if num >= number_of_total_game/BATCH_SIZE: break


       


         
                   
        train_writer.add_summary(summary_train, global_step=global_counter)
        train_writer.flush()
            
        cost_per_game_average = sum(game_cost_record) / len(game_cost_record)
        write_game_average_csv([{"iteration": epcho, "game": game_number,
                                     "cost_per_game_average": cost_per_game_average}])
      
        AVG_MSEcost=Loss_total/float(number_of_total_game)
        x_list.append(epcho) 
        y_list.append(AVG_MSEcost)
        
        print('iteration :', epcho , 'AVG_MSEcost',AVG_MSEcost)
        game_diff_record_all.append(game_diff_record_dict)
        if epcho%1==0:
            saver.save(sess, 'saved_models'+'_gamma'+str(GAMMA)+'_hd'+str(H_SIZE)+'_iter'+str(ITERATE_NUM)+'_lr'+str(learning_rate)+'/CNN_New'+str(epcho))
    axs[0].plot(x_list,y_list,'o-',color='b',linewidth=1)

    
    plt.savefig('fn'+'_gamma'+str(GAMMA)+'_hd'+str(H_SIZE)+'_iter'+str(ITERATE_NUM)+'_lr'+str(learning_rate)+'.png')
    
    

def train_start():
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)
    if not os.path.isdir(SAVED_NETWORK):
        os.mkdir(SAVED_NETWORK)

    sess = tf.InteractiveSession()
    if MODEL_TYPE == "v3":
        nn = td_prediction_lstm_V3(FEATURE_NUMBER, H_SIZE, MAX_TRACE_LENGTH, learning_rate)
    elif MODEL_TYPE == "v4":
        nn = td_prediction_lstm_V4(FEATURE_NUMBER, H_SIZE, MAX_TRACE_LENGTH, learning_rate)
        #print(nn)
    else:
        raise ValueError("MODEL_TYPE error")
    train_network(sess, nn)


if __name__ == '__main__':
    train_start()
