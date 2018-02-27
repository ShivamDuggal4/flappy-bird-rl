import tensorflow as tf
import numpy as np
import time
import utils
import random
import json
from layers import create_conv,fc,clipped_error
import os



""" 
    FLOW:
        * AtariGames class will create an instance of Model, loading its hyperparameters and will call init function
        * init will build the graph and will initialize the session
        * Atari Game will be played and after each episode , some batches from replay memory will be refreshed to replay buffer and model will be trained on their batches
"""


class Model:

    def __init__(self,hyperparameters_dir):
    	print(hyperparameters_dir)
        hyperparameters = json.loads(open(hyperparameters_dir).read())
        print(hyperparameters)

        self.image_rows = hyperparameters['image_rows']
        self.image_columns = hyperparameters['image_columns']
        self.image_channels = hyperparameters['image_channels']

        self.conv1_size = hyperparameters['conv1_size']
        self.conv1_filters = hyperparameters['conv1_filters']

        self.conv2_size = hyperparameters['conv2_size']
        self.conv2_filters = hyperparameters['conv2_filters']

        self.conv3_size = hyperparameters['conv3_size']
        self.conv3_filters = hyperparameters['conv3_filters']


        self.fc1_output_size = hyperparameters['fc1_output_size']

        self.fc2_output_size = hyperparameters['fc2_output_size']

        self.learning_rate = hyperparameters['learning_rate']
        self.batch_size = hyperparameters['batch_size']

        self.max_batches_to_train = hyperparameters['max_batches_to_train']
        self.GAMMA = hyperparameters['GAMMA']

        self.train_log_name = './logs/train'

    def init(self):
        print("Building the model input")
        self.build_model_input()

        self.model_output = self.build_main_network()

        self.loss = self.build_loss(self.model_output,self.tf_reward,self.tf_labels)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.tf_optimizer = optimizer.minimize(self.loss, global_step=tf.Variable(0, trainable=False))

        self.init_session()


    def build_model_input(self):
        # Images image_rows*image_columns*image_channels
        self.tf_images = tf.placeholder(tf.float32, [None, self.image_rows, self.image_columns, self.image_channels], name='images')
        self.tf_labels = tf.placeholder(tf.int64,[None], name='labels')
        self.tf_reward = tf.placeholder(tf.float32,[None], name='reward')
        tf.summary.image("input_img", self.tf_images, max_outputs=self.batch_size)



    def build_main_network(self):
    	print("Building the main network")

    	shape1 = (self.conv1_size, self.conv1_size, self.image_channels, self.conv1_filters)
        conv1, self.conv_w_1, self.conv_b_1 = create_conv('1', self.tf_images, shape1, relu=True, max_pooling=True, padding='SAME',stride=4)

        shape2 = (self.conv2_size, self.conv2_size, self.conv1_filters, self.conv2_filters)
        conv2, self.conv_w_2, self.conv_b_2 = create_conv('2', conv1, shape2, relu=True, max_pooling=True, padding='SAME',stride=2)

        shape3 = (self.conv3_size, self.conv3_size, self.conv2_filters, self.conv3_filters)
        conv3, self.conv_w_3, self.conv_b_3 = create_conv('3', conv2, shape3, relu=True, max_pooling=True, padding='SAME',stride=2)

        
        self.fc1_input_size = int(np.prod(conv3.get_shape()[1:]))
        print(self.fc1_input_size)
        flattened_conv3 = tf.reshape(conv3,(-1,self.fc1_input_size))
    	fc1 = fc('4',flattened_conv3,self.fc1_input_size,self.fc1_output_size,relu=True)

        self.fc2_input_size = self.fc1_output_size
        fc2 = fc('5',fc1,self.fc2_input_size,self.fc2_output_size)

        return fc2


    def build_loss(self,model_output,actual_action_reward,actual_action_label):
        #actual_action_label = tf.contrib.layers.flatten(actual_action_label)
        self.actual_action_label = actual_action_label
        actual_action_one_hot_labels = tf.one_hot(actual_action_label, depth=self.fc2_output_size)
        self.actual_action_one_hot_labels = actual_action_one_hot_labels

        model_reward = tf.reduce_sum(actual_action_one_hot_labels*model_output,1)
        self.model_reward = model_reward

        self.difference  = model_reward - actual_action_reward

        loss = tf.reduce_mean(tf.square(model_reward - actual_action_reward))

        return loss




    def init_session(self):
    	print("Creating Sesssion")
        self.sess = tf.Session()
        # Tensorboard
        self.tf_tensorboard = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.train_log_name, self.sess.graph)
        #self.test_writer = tf.summary.FileWriter(self.test_log_name)
        
        #  Create session
        print "creating saver"
        self.saver = tf.train.Saver()
        
        print "Initializing global variables"
        # Init variables
        self.sess.run(tf.global_variables_initializer())
        print "Initialized global variables"

        self.train_writer_it = 0
        #self.test_writer_it = 0



    def optimize(self, images,reward,labels, tb_save=False):
        tensors = [self.tf_optimizer, self.loss, self.model_output,self.model_reward,self.actual_action_label ,self.actual_action_one_hot_labels,self.difference]
        _, loss, model_output,model_reward,actual_action_label,actual_action_one_hot_labels,difference = self.sess.run(tensors,
            feed_dict={
            self.tf_images: images,
            self.tf_labels: labels,
            self.tf_reward: reward
        })

        print("Model Output: " + str(model_output))

        print("Model Reward: " + str(model_reward))
        print("Model LABEL: " + str(actual_action_label))
        print("Model ONE_HOT: " + str(actual_action_one_hot_labels))
        print("Model and Actual Difference: " + str(difference))

        if tb_save:
            # Write data to tensorboard
            self.train_writer.add_summary(summary, self.train_writer_it)
            self.train_writer_it += 1

        return loss
    

    def save_session(self, step):
        checkpoint_dir = "checkpoint"
        model_name = "flappyBird.model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)



    def load_session(self):
        print(" [*] Reading checkpoint...")
        checkpoint_dir = "checkpoint"
        model_name = "flappyBird.model-2"
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def print_train_loss(self, images,labels, epoch, iteration):

        tensors = [self.loss]
        train_loss, train_accuracy = self.sess.run(tensors,
            feed_dict={
            self.tf_images: images,
            self.tf_labels: labels
        })
        
        print("Epoch: [%4d/%4d] time: %.4f, loss: %.8f, accuracy: %.8f" \
            % (epoch, iteration,
                time.time() - self.start_time, train_loss, train_accuracy))


    def refresh_replay_buffer(self,agent_replay_memory):
        self.replay_buffer = random.sample(agent_replay_memory,self.batch_size*self.max_batches_to_train)


    def predict_action(self,state):
        model_output = self.sess.run(self.model_output,
            feed_dict={
            self.tf_images: state
        })

        return np.argmax(np.array(model_output))

    def predict_Q_value(self,state):
        model_output = self.sess.run(self.model_output,
            feed_dict={
            self.tf_images: state
        })

        return np.max(np.array(model_output))

    def update_model(self,agent_replay_memory,load_session):
        if load_session == True:
            self.load_session()
        print "Updating model"
        self.refresh_replay_buffer(agent_replay_memory)
        counter = 0
        train_batch = 0
        while(train_batch+self.batch_size<=len(self.replay_buffer)):
            state_batch = []
            actual_value_batch = []
            action_batch = []
            for element in self.replay_buffer[train_batch:train_batch+self.batch_size]:
                state = element.state
                next_state = element.next_state
                reward = element.reward
                action = element.action
                done = element.result

                print("Reward: " + str(reward))
                '''
                if done==True:
                    actual_value = reward
                else:
                    actual_value = reward + self.GAMMA * np.max(self.predict_Q_value(next_state))
                '''
                actual_value = reward

                state_batch.append(state)
                actual_value_batch.append(actual_value)
                action_batch.append(action)

            state_batch,actual_value_batch = utils.form_batches(state_batch,actual_value_batch,self.image_rows,self.image_columns,self.image_channels)
            print("Actions")
            print np.array(action_batch).reshape(-1)
            print("Action Values")
            print actual_value_batch
            loss = self.optimize(state_batch,actual_value_batch,np.array(action_batch).reshape(-1))

            if np.mod(counter,1)==0:
                print("LOSS: " + str(loss))

            counter+=1
            train_batch = train_batch + self.batch_size
        self.save_session(2)
                
        

