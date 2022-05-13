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
from configuration  import FULL_PATH, MODEL_TYPE, MAX_TRACE_LENGTH, FEATURE_NUMBER, BATCH_SIZE, GAMMA, H_SIZE, \
    model_train_continue, FEATURE_TYPE, ITERATE_NUM, learning_rate, SPORT, save_mother_dir
import sys
import argparse
from matplotlib.ticker import FormatStrFormatter
np.set_printoptions(threshold=sys.maxsize)
random.seed(0)
from scipy.stats import spearmanr
parser = argparse.ArgumentParser()

parser.add_argument('--iter_number', type=str, default='',
                    help='load csv have labeled')

args = parser.parse_args()
ii = args.iter_number
ii=int(ii)
print(BATCH_SIZE)

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
    FULL_PATH='./'+str(DATA_STORE)

    file_list = [file for file in os.listdir(FULL_PATH) if file.endswith('.csv')]
    file_list.sort(key=lambda x:(os.path.splitext(x)[0][0], int(os.path.splitext(x)[0][1:])))
    number_of_total_game = len(file_list)
  
    files=file_list
    
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
 
    for iii in range(ii,ii+1):   
        saver.restore(sess, 'saved_models'+'_gamma'+str(GAMMA)+'_hd'+str(H_SIZE)+'_iter'+str(ITERATE_NUM)+'_lr'+str(learning_rate)+'/CNN_New'+str(iii))
        print('saved_models'+'_gamma'+str(GAMMA)+'_hd'+str(H_SIZE)+'_iter'+str(ITERATE_NUM)+'_lr'+str(learning_rate)+'/CNN_New'+str(iii))
        
        Loss_total=0
        l_all=0
        acc_total=0
        game_diff_record_dict = {}
        
        iteration_now = game_number / number_of_total_game + 1
        game_diff_record_dict.update({"Iteration": iteration_now})
        generated_data = data_generator(files, batch_size = BATCH_SIZE)  
               
        b_point=0
        f_point=0
        
        num = 0
        Back_action_all=0
        Front_action_all=0
        stroke_list=[]
        stroke_list_all=[]
        action_max_all=0
        Type=0
        len0=0
        PPPP=1
        
        p1_point=0
        n_list = ['Serve','Drop','Smash','Clear','Lift','Drive','Block','Push','Net Kill','Net Shot']
        
        for stroke_type in range(len(n_list)):      
            globals()['p_'+str(playername)+'_'+str(n_list[stroke_type])]=[]
        for stroke_type in range(len(n_list)):      
            globals()['p_'+str(playername)+'_'+str(n_list[stroke_type])+'_all']=[]
       




        for data, target,trace,dir_game in generated_data:

            
            print(dir_game)
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



            
             
            stroke_csv=pd.read_csv(strokename+'/'+str(dir_game))
            player_name=stroke_csv['Player name']
            player_name=list(player_name)
            stroke_name=stroke_csv['Stroke Type']
            stroke_name=list(stroke_name)
            player_court=stroke_csv['Player']
            player_court=list(player_court)
            player_point=stroke_csv['Point']
            player_point=list(player_point)
            instance=stroke_csv['Instance number']
            instance=list(instance)
            player_n=stroke_csv['Player name']
            player_n=list(player_n)


            
            if player_point[-1]=='Front Point':
                    f_point=f_point+1
            if player_point[-1]=='Back Point':
                    b_point=b_point+1
               
          
            readout_t1_batch=readout_t1_batch[:,:trace[0],:]
            
            for j in range(trace[0]):

                L_home.append(readout_t1_batch[0][j][0])
                   
            
                L_away.append(readout_t1_batch[0][j][1])
                    
                L_end.append(readout_t1_batch[0][j][2])                        

               
                
            fig,ax = plt.subplots(tight_layout=True)
            ax.set_xticks(instance)
            for label in (ax.get_xticklabels()+ax.get_yticklabels()):
	        label.set_fontsize(15)             
            ax.plot(instance,L_home,label = "Back_player")
            ax.plot(instance,L_away,label = "Front_player")
            ax.plot(instance,L_end,label = "Rally_end",ls="--")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
             
                
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) 
                
            ax.set_ylabel('Q(s,a)',fontsize=20)

            labels = [item.get_text() for item in ax.get_xticklabels()]
            labels = stroke_name
                
            ax.set_xticklabels(stroke_name, minor=False, rotation=30)



               

            plt.legend()
      
            directory='./Players/'+str(game)+'/'+str(playername)
            if not os.path.exists(directory):
                os.makedirs(directory)                
            ddd_game=dir_game.replace('.csv','')     
            plt.savefig('./Players/'+str(game)+'/'+str(playername)+'/'+str(iii)+str(ddd_game)+'.png', dpi=900)
               
            plt.clf()
            

            
            Back_action=np.ediff1d(L_home, to_begin=L_home[0])
            Front_action=np.ediff1d(L_away, to_begin=L_away[0])
       
            y3=[-1,1]

            p_t=[]
            p_name=[]
            pp_name=[]
            for p in range(len(player_name)):
                if player_name[p]==playername and player_court[p]=='B':
                       
                   p_t.append(int(p+1))
                   p_name.append(Back_action[p])

                    
                   globals()['p_'+str(player_name[p])+'_'+str(stroke_name[p])].append(Back_action[p])
                       
                   Back_action_all+=Back_action[p]
                      
                if player_name[p]==playername and player_court[p]=='F':
                  
                   p_t.append(int(p+1))
                       
                   p_name.append(Front_action[p])
                      



                   globals()['p_'+str(player_name[p])+'_'+str(stroke_name[p])].append(Front_action[p])
                   Front_action_all+=Front_action[p]
                       


                globals()['p_'+str(playername)+'_'+str(stroke_name[p])+'_all'].append(Back_action_all)
                globals()['p_'+str(playername)+'_'+str(stroke_name[p])+'_all'].append(Front_action_all)
            if len(p_name)==0:
                len0+=1
                action_max=0
            if len(p_name)!=0:

                action_max=max(p_name)
            action_max_all+=action_max
            global pre_p_name
            global pre_p_t
            if player_point[-1]=='Front Point' and player_n[-1]==playername and player_court[-1]=='F':
                    p1_point=p1_point+1
            if player_point[-1]=='Back Point' and player_n[-1]==playername and player_court[-1]=='B':
                    p1_point=p1_point+1
            if player_point[-1]=='Back Point' and player_n[-1]!=playername and player_court[-1]=='F':
                    p1_point=p1_point+1
            if player_point[-1]=='Front Point' and player_n[-1]!=playername and player_court[-1]=='B':
                    p1_point=p1_point+1                
            
            num = num + 1
            if num >= number_of_total_game/BATCH_SIZE: break
 


        
       
        name_list = ['Serve','Drop','Smash','Clear','Lift','Drive','Block','Push','Net Kill','Net Shot']
      
        for t in range(len(name_list)):  
            
           stroke_list=stroke_list+globals()['p_'+str(playername)+'_'+str(name_list[t])]
       
        cc=['b', 'g', 'r','c','m', 'y', 'k', 'tab:grey', 'tab:brown', 'tab:pink']
        barlist=plt.bar((list(range(1, len(stroke_list)+1))), stroke_list)
        k=0
        for tp in range(len(name_list)):
            
            for i in range(len(globals()['p_'+str(playername)+'_'+str(name_list[tp])])): 
              
                barlist[k+i].set_color(cc[tp])
            k=len(globals()['p_'+str(playername)+'_'+str(name_list[tp])])+k
        plt.legend()
      




        plt.clf()
       
        fig, ax = plt.subplots(tight_layout=True)





     
        name_list = ['Serve','Drop','Smash','Clear','Lift','Drive','Block','Push','Net Kill','Net Shot']
    
        for t in range(len(name_list)):  
           
           if len(globals()['p_'+str(playername)+'_'+str(name_list[t])])!=0:
               stroke_list_all=stroke_list_all+[sum(globals()['p_'+str(playername)+'_'+str(name_list[t])])/len(globals()['p_'+str(playername)+'_'+str(name_list[t])])]
               Type+=1
               
           else:
               stroke_list_all=stroke_list_all+[0]
        
        cc=['b', 'g', 'r','c','m', 'y', 'k', 'tab:grey', 'tab:brown','tab:pink']
       
        barlist=plt.bar(name_list, stroke_list_all)
        k=0
        for tp in range(len(name_list)):
            barlist[tp].set_color(cc[tp])
            

        for label in (ax.get_xticklabels()+ax.get_yticklabels()):
	    label.set_fontsize(15)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) 
        
        
        ax.set_xticklabels(name_list, rotation = 30,fontsize=15)
        
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels = stroke_name
        myTitle=str(playername)
        ax.set_title(myTitle, loc='center', wrap=True, fontsize=30)
        
        ax.set_ylabel('Average action value', fontsize=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
      
        average_maximum_action_all.append(action_max_all/float(number_of_total_game))
       
        
        average_action_all.append(sum(stroke_list_all)/float(Type))
        

        p1_p.append(p1_point)
       
        


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
     
     GM=[1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21]

     PN=['LIN D.','LEE C.W.','NG K.L.','NISHIMOTO','LIN D.','SHI Y.Q.', 'CHOU T.C.','TSUNEYAMA', 'LEE Z.J.','CHRISTIE','CHOU T.C.','AXELSEN','HUANG Y.X.','LIN D.', 'LEE D.K.','LEE C.W.','VITTINGHUS','LIN D.','LEVERDEZ','KIDAMBI','LEE Z.J.','AXELSEN','CHOU T.C.','ANTONSEN','LEE Z.J.','CHEN L.','LU G.Z.','AXELSEN','SON W.H.','SHI Y.Q.','ANTONSEN ', 'SHI Y.Q.','SHI Y.Q.','AXELSEN','CHEN L.','GINTING','MOMOTA','GINTING','GINTING','ANTONSEN','MOMOTA','ANTONSEN']

     ST=['./stroke1','./stroke1','./stroke2','./stroke2','./stroke3','./stroke3','./stroke4','./stroke4','./stroke5','./stroke5','./stroke6','./stroke6','./stroke7','./stroke7','./stroke8','./stroke8','./stroke9','./stroke9','./stroke10','./stroke10','./stroke11','./stroke11','./stroke12','./stroke12','./stroke13','./stroke13','./stroke14','./stroke14','./stroke15','./stroke15','./stroke16','./stroke16','./stroke17','./stroke17','./stroke18','./stroke18','./stroke19','./stroke19','./stroke20','./stroke20','./stroke21','./stroke21']

     DS=['S1',
'S1','S2','S2','S3','S3','S4','S4','S5','S5','S6','S6','S7','S7','S8','S8','S9','S9','S10','S10','S11','S11','S12','S12','S13','S13','S14','S14','S15','S15','S16','S16','S17','S17',
'S18','S18','S19','S19','S20','S20','S21','S21']

     global pre_p_name
     global pre_p_t


     global p1
     global p1_p     
     p1_p=[]
    
     global player_name
     global game
     global playername
     global strokename
     global DATA_STORE
     global average_maximum_action_all
     global average_action_all
     average_maximum_action_all=[]
     average_action_all=[]
     global tt
     
    
     for tt in range(42):
         game = GM[tt]
         playername = PN[tt]
         strokename = ST[tt]
         DATA_STORE = DS[tt]
         train_network(sess, nn)















     

if __name__ == '__main__':
    train_start()
