import tensorflow as tf
from configuration import MODEL_TYPE, MAX_TRACE_LENGTH, FEATURE_NUMBER, BATCH_SIZE, GAMMA, H_SIZE, \
    model_train_continue, FEATURE_TYPE, ITERATE_NUM, learning_rate, SPORT, save_mother_dir


class td_prediction_lstm_V3:
    def __init__(self, FEATURE_NUMBER, H_SIZE, MAX_TRACE_LENGTH, learning_rate, rnn_type='bp_last_step'):
        """
        define a shallow dynamic LSTM
        """
    def __init__(self, FEATURE_NUMBER, H_SIZE, MAX_TRACE_LENGTH, learning_rate, rnn_type='bp_last_step'):
        """
        define a dynamic LSTM
        """
        with tf.name_scope("LSTM_layer"):
            self.rnn_input = tf.placeholder(tf.float32, [None,MAX_TRACE_LENGTH, FEATURE_NUMBER], name="x_1")
            self.trace_lengths = tf.placeholder(tf.int32, [None], name="tl")
          
            self.lstm_cell = tf.contrib.rnn.LSTMCell(num_units=H_SIZE * 2, state_is_tuple=True,
                                                     initializer=tf.random_uniform_initializer(-0.05, 0.05))

            self.rnn_output, self.rnn_state = tf.nn.dynamic_rnn(  # while loop dynamic learning rnn
                inputs=self.rnn_input, cell=self.lstm_cell, sequence_length=self.trace_lengths, dtype=tf.float32,
                scope=rnn_type + '_rnn')

            # [batch_size, max_time, cell.output_size]
            self.outputs = tf.stack(self.rnn_output)

         
        num_layer_1 = H_SIZE * 2
        num_layer_2 = 1000
        num_layer_3 = 3

        with tf.name_scope("Dense_Layer_first"):
            self.W1 = tf.get_variable('w1_xaiver', [num_layer_1, num_layer_2],
                                      initializer=tf.contrib.layers.xavier_initializer())
            self.b1 = tf.Variable(tf.zeros([num_layer_2]), name="b_1")
            self.y1 = tf.matmul(self.outputs, self.W1) + self.b1
            self.activation1 = tf.nn.relu(self.y1, name='activation')
            

        with tf.name_scope("Dense_Layer_second"):
            self.W2 = tf.get_variable('w2_xaiver', [num_layer_2, num_layer_3],
                                      initializer=tf.contrib.layers.xavier_initializer())
            self.b2 = tf.Variable(tf.zeros([num_layer_3]), name="b_2")
            self.y2 = tf.matmul(self.activation1, self.W2) + self.b2
            self.read_out= tf.nn.sigmoid(self.y2, name='activation')
           

       
        self.y = tf.placeholder("float", [BATCH_SIZE,MAX_TRACE_LENGTH, num_layer_3])

        with tf.name_scope("cost"):
            self.readout_action = self.read_out
            self.cost = tf.reduce_mean(tf.square(self.y - self.readout_action))
            self.diff = tf.reduce_mean(tf.abs(self.y - self.readout_action))
        tf.summary.histogram('cost', self.cost)

        with tf.name_scope("train"):
            
            self.train_step = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(self.cost)
           
