import tensorflow as tf
import numpy as np
import time
import matplotlib
# OSX fix
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns


from src.layers import conv_layer
from src.config import *
from src.utils import init_weights, gen_data


class CryptoNet(object):
    def __init__(self, sess, msg_len=MSG_LEN, batch_size=BATCH_SIZE,
                 epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE):
        """
        Args:
            sess: TensorFlow session
            msg_len: The length of the input message to encrypt.
            key_len: Length of Alice and Bob's private key.
            batch_size: Minibatch size for each adversarial training
            epochs: Number of epochs in the adversarial training
            learning_rate: Learning Rate for Adam Optimizer
        """

        self.sess = sess
        self.msg_len = msg_len
        self.key_len = self.msg_len
        self.N = self.msg_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.k=10000
        self.eve_only=0
        self.k_freeze=0
        self.stuck=1
        self.msg1, self.key1 = gen_data(n=self.batch_size, msg_len=self.msg_len, key_len=self.key_len)
        for i in range(len(self.key1)):
            self.key1[i]=self.key1[0]
        self.build_model()

    def build_model(self):
        # Weights for fully connected layers
        self.w_alice = init_weights("alice_w", [2 * self.N, 2 * self.N])
#        tf.get_variables("alice_b0",shape=[2 * self.N],initializer=tf.zeros_initializer())
        self.w_bob = init_weights("bob_w", [2 * self.N, 2 * self.N])
#        tf.get_variables("bob_b0",shape=[2 * self.N],initializer=tf.zeros_initializer())
        self.w_eve1 = init_weights("eve_w1", [self.N, 2 * self.N])
#        tf.get_variables("eve_b0",shape=[2 * self.N],initializer=tf.zeros_initializer())
        self.w_eve2 = init_weights("eve_w2", [2 * self.N, 2 * self.N])
#        tf.get_variables("eve_b1",shape=[2 * self.N],initializer=tf.zeros_initializer())

        # Placeholder variables for Message and Key
        self.msg = tf.placeholder("float", [None, self.msg_len])
        self.key = tf.placeholder("float", [None, self.key_len])

        # Alice's network
        # FC layer -> Conv Layer (4 1-D convolutions)
        self.alice_input = tf.concat([self.msg, self.key],1)
        self.alice_hidden = tf.nn.sigmoid(tf.matmul(self.alice_input, self.w_alice))
        self.alice_hidden = tf.expand_dims(self.alice_hidden, 2)
        self.alice_output = tf.squeeze(conv_layer(self.alice_hidden, "alice"))

        # Bob's network
        # FC layer -> Conv Layer (4 1-D convolutions)
        self.bob_input = tf.concat([self.alice_output, self.key],1)
        self.bob_hidden = tf.nn.sigmoid(tf.matmul(self.bob_input, self.w_bob))
        self.bob_hidden = tf.expand_dims(self.bob_hidden, 2)
        self.bob_output = tf.squeeze(conv_layer(self.bob_hidden, "bob"))

        # Eve's network
        # FC layer -> FC layer -> Conv Layer (4 1-D convolutions)
        self.eve_input = self.alice_output
        self.eve_hidden1 = tf.nn.sigmoid(tf.matmul(self.eve_input, self.w_eve1))
        self.eve_hidden2 = tf.nn.sigmoid(tf.matmul(self.eve_hidden1, self.w_eve2))
        self.eve_hidden2 = tf.expand_dims(self.eve_hidden2, 2)
        self.eve_output = tf.squeeze(conv_layer(self.eve_hidden2, "eve"))

    def train(self):
        # Loss Functions
        t=time.time()
        self.decrypt_err_eve = tf.reduce_mean(tf.abs(self.msg - self.eve_output))
        self.decrypt_err_bob = tf.reduce_mean(tf.abs(self.msg - self.bob_output))
#        self.loss_bob = self.decrypt_err_bob*0.5+(1-self.decrypt_err_eve)**2
        self.d=tf.reduce_mean(tf.ones(self.N)-tf.abs(self.alice_output))
        self.loss_bob=(self.decrypt_err_bob*0.5)+(1-self.decrypt_err_eve)**3#+self.d**2

        # Get training variables corresponding to each network
        self.t_vars = tf.trainable_variables()
        self.alice_or_bob_vars = [var for var in self.t_vars if 'alice_' in var.name or 'bob_' in var.name]
        self.eve_vars = [var for var in self.t_vars if 'eve_' in var.name]

        # Build the optimizers
        self.bob_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.loss_bob, var_list=self.alice_or_bob_vars)
        self.eve_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.decrypt_err_eve, var_list=self.eve_vars)
        self.bob_errors, self.eve_errors,self.d_from_bit = [], [], []

        # Begin Training
        tf.global_variables_initializer().run()
        plt.ion()
        sns.set_style("darkgrid")
        plt.show()
        e=0
        for i in range(self.epochs):
        # plot live
            if i<1000 and i%100==0:
                self.plot_live()
            if i %(min(500,self.epochs/10))==0:
                self.plot_live()
            
            f=0
            if self.k==5000:
                if self.bob_errors[-1]>0.05*0.0625:
                    if self.stuck:
                        #self.take_loss2()
                        e=0
                        self.k=0
                        print('CHANGE BACK LOSS2 (LONG TIME) : BOB ADVANTAGE',i)
#            iterations = 2000
            if (i+1)%1000==0:
                a=sum(self.bob_errors[-100:])/100
                b=sum(self.eve_errors[-100:])/100
                print ('Training is at ', i+1,' batches',a,b,self.d_from_bit[-1])
            self._train('bob')
#            bob_loss, _ = self._train('bob', iterations)
#            self.bob_errors.append(bob_loss)

#            print 'Training Eve, Epoch:', i + 1
            self._train('eve')
#            _, eve_loss = self._train('eve', iterations)
#            self.eve_errors.append(eve_loss)

            if (i<5000 or self.k>4900) and self.bob_errors[-1]/self.msg_len < 0.05*0.0625:
                a=sum(self.bob_errors[-100:])/100
                if a/self.msg_len <0.0625 :
                    if (self.msg_len/2-self.eve_errors[-1])/self.msg_len<0.05*0.0625:
                        b=sum(self.msg_len/2-self.eve_errors[-100+j] for j in range(100))/100
                        if b/self.msg_len<0.05*0.0625:
                            if self.bob_errors[-1]/self.msg_len < 0.0625*0.05 and a/self.msg_len < 0.05*0.0625 and self.stuck:
#                                self.eve_only=1
#                                self.k_freeze=1
#                                self.take_loss3()
#                                self.stuck=0
#                                print('NOW EVE ONLY')
                                break
                if not e:
                    if self.stuck:
                        self.take_loss1()
                        e=1
                        self.k=0
                        print('CHANGE LOSS HERE : EVE ADVANTAGE ',i)
                        f=1
#                        self.k_freeze=1
            elif (i<5000 or self.k>1000) and e and self.bob_errors[-1]/self.msg_len>2*0.0625:
                a=sum(self.bob_errors[-100:])/100
                if a/self.msg_len > 2*0.0625 and self.stuck:
                    self.take_loss0()
                    e=0
                    self.k=0
                    f=1
                    print('CHANGE BACK LOSS0 HERE : BOB ADVANTAGE ',i)
            if not f:
                self.k+=1

        m,k=gen_data(n=2, msg_len=self.msg_len, key_len=self.key_len)
        print(m,k)
        print(self.sess.run(self.alice_output,feed_dict={self.msg: m, self.key: k}))
        print(self.sess.run(self.bob_output,feed_dict={self.msg: m, self.key: k}))
        print(self.sess.run(self.eve_output,feed_dict={self.msg: m, self.key: k}))
        plt.ioff()
        plt.clf()
        print('le temps du training etait de : ', time.time()-t)
        self.plot_errors()

    def _train(self, network):
#        bob_decrypt_error, eve_decrypt_error = 1., 1.

#        bs = self.batch_size
        # Train Eve for two minibatches to give it a slight computational edge
#        if network == 'eve':
#            bs *= 2
#
#        msg_in_val, key_val = gen_data(n=bs, msg_len=self.msg_len, key_len=self.key_len)
        if network == 'bob':
            msg_in_val, key_val = gen_data(n=self.batch_size, msg_len=self.msg_len, key_len=self.key_len)
            if self.k_freeze :
                key_val=self.key1
            self.d_from_bit.append(self.sess.run(self.d,feed_dict={self.msg: msg_in_val, self.key: key_val}))
            if self.eve_only:
                decrypt_err= self.sess.run(self.decrypt_err_bob,
                              feed_dict={self.msg: msg_in_val, self.key: key_val})
            else:
                _, decrypt_err = self.sess.run([self.bob_optimizer, self.decrypt_err_bob],
                                               feed_dict={self.msg: msg_in_val, self.key: key_val})
            self.bob_errors.append(decrypt_err*self.msg_len/2)
#            if self.k and self.k<10:
#                print(self.sess.run(self.w_alice))
#                self.k+=1
#            print(decrypt_err)
#bob_decrypt_error = min(bob_decrypt_error, decrypt_err)

        elif network == 'eve':
            for j in range(2):
                msg_in_val, key_val = gen_data(n=self.batch_size, msg_len=self.msg_len, key_len=self.key_len)
#                key_val=self.key1
                _, decrypt_err = self.sess.run([self.eve_optimizer, self.decrypt_err_eve],
                                               feed_dict={self.msg: msg_in_val, self.key: key_val})
                if j:
                    self.eve_errors.append(decrypt_err*self.msg_len/2)
            #eve_decrypt_error = min(eve_decrypt_error, decrypt_err)

        #bob_decrypt_error, eve_decrypt_error


    def take_loss0(self):
        self.loss_bob=(self.decrypt_err_bob*0.5)+(1-self.decrypt_err_eve)**2
        opt=tf.train.AdamOptimizer(self.learning_rate)
        self.bob_optimizer = opt.minimize(self.loss_bob, var_list=self.alice_or_bob_vars)
        var=[]
        for v in self.alice_or_bob_vars:
            var.append(opt.get_slot(v, "m"))
            var.append(opt.get_slot(v, "v"))
        var.extend(list(opt._get_beta_accumulators()))
        tf.variables_initializer(var).run()


    def take_loss1(self):
        self.loss_bob=(self.decrypt_err_bob*0.5)**1.5+abs(1-self.decrypt_err_eve)+self.d**2
        opt=tf.train.AdamOptimizer(self.learning_rate)
        self.bob_optimizer = opt.minimize(self.loss_bob, var_list=self.alice_or_bob_vars)
        var=[]
        for v in self.alice_or_bob_vars:
            var.append(opt.get_slot(v, "m"))
            var.append(opt.get_slot(v, "v"))
        var.extend(list(opt._get_beta_accumulators()))
        tf.variables_initializer(var).run()

    def take_loss2(self):
        self.loss_bob=(self.decrypt_err_bob*0.5)+abs(1-self.decrypt_err_eve)+self.d**2
        opt=tf.train.AdamOptimizer(self.learning_rate)
        self.bob_optimizer = opt.minimize(self.loss_bob, var_list=self.alice_or_bob_vars)
        var=[]
        for v in self.alice_or_bob_vars:
            var.append(opt.get_slot(v, "m"))
            var.append(opt.get_slot(v, "v"))
        var.extend(list(opt._get_beta_accumulators()))
        tf.variables_initializer(var).run()

    def take_loss3(self):
        self.loss_bob=(self.decrypt_err_bob*0.5)
        opt=tf.train.AdamOptimizer(self.learning_rate)
        self.bob_optimizer = opt.minimize(self.loss_bob, var_list=self.alice_or_bob_vars)
        var=[]
        for v in self.alice_or_bob_vars:
            var.append(opt.get_slot(v, "m"))
            var.append(opt.get_slot(v, "v"))
        var.extend(list(opt._get_beta_accumulators()))
        tf.variables_initializer(var).run()

    def plot_live(self):
        plt.clf()
        X = range(len(self.bob_errors))
        plt.plot(lissage(X,self.bob_errors,100)[1])
        plt.plot(lissage(X,self.eve_errors,100)[1])
        plt.plot(self.d_from_bit)
        plt.legend(['bob', 'eve','distance from true bit encryption'])
        plt.xlabel('Batches')
        plt.ylabel('Average decryption over one batch')
        plt.pause(0.00001)

    def plot_errors(self):
        """
        Plot Lowest Decryption Errors achieved by Bob and Eve per epoch
        """
        sns.set_style("darkgrid")
        X = np.linspace(0,len(self.bob_errors)-1,len(self.bob_errors))
        print(len(X),len(self.bob_errors))
        plt.plot(lissage(X,self.bob_errors,100)[1])
        plt.plot(lissage(X,self.eve_errors,100)[1])
        plt.plot(self.d_from_bit)
        plt.legend(['bob', 'eve','distance from true bit encryption'])
        plt.xlabel('Batches')
        plt.ylabel('Average decryption error over one batch')
        plt.title('END OF TRAINING')
        plt.show()




def lissage(Lx,Ly,p):
        Lxo=[]
        Lyo=[]
        for i in range(len(Lx)):
            Lxo.append(Lx[i])
        for i in range(len(Ly)):
            val=0
            c=0
            for k in range(2*p):
                if i-p+k>=0 and i-p+k<len(Ly):
                    val+=Ly[i-p+k]
                    c+=1
            Lyo.append(val/c)
        return Lxo,Lyo
