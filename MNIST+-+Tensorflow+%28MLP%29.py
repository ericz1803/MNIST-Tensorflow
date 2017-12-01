
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import math
#from helper import batches
from tqdm import tqdm
print("All modules imported successfully.")


# All of the files can be found on Kaggle.
# 
# https://www.kaggle.com/c/digit-recognizer/data

# In[2]:


train_X=pd.read_csv("C:/Users/Eric Zhou/Downloads/train.csv")
validation_X=train_X.loc[40000:]
train_X=train_X.loc[:39999]
train_y=train_X['label']
validation_y=validation_X['label']
del train_X['label']
del validation_X['label']


# In[3]:


train_X=np.array(train_X)
train_y=np.array(train_y)
validation_X=np.array(validation_X)
validation_y=np.array(validation_y)
print(train_X.shape, train_y.shape, validation_X.shape, validation_y.shape)
#test data
test_X=pd.read_csv("C:/Users/Eric Zhou/Downloads/test.csv")
test_X.astype(dtype='float32')


# In[4]:


#convert to one-hot array
train_y=np.array(pd.get_dummies(train_y))
validation_y=np.array(pd.get_dummies(validation_y))
print(train_y.shape, validation_y.shape)


# In[5]:


tf.cast(train_X, tf.float32)
tf.cast(train_y, tf.float32)
tf.cast(validation_X, tf.float32)
tf.cast(validation_y, tf.float32)


# In[6]:


features=tf.placeholder(tf.float32, shape=[None, 784])
labels=tf.placeholder(tf.float32)


# In[7]:


#set some parameters
batch_size=128

nodes_hl1=1000
nodes_hl2=500
nodes_hl3=100

output_size=10

num_epochs=50


# In[8]:


#create variables(weights and biases)
weights_input_hl1=tf.get_variable('weights_input_hl1', dtype=tf.float32, 
  initializer=tf.truncated_normal([784, nodes_hl1], dtype=tf.float32, stddev=np.sqrt(2/784)))
biases_hl1=tf.get_variable('biases_hl1', [nodes_hl1], dtype=tf.float32, 
  initializer=tf.zeros_initializer)

weights_hl1_hl2=tf.get_variable('weights_hl1_hl2', dtype=tf.float32, 
  initializer=tf.truncated_normal([nodes_hl1, nodes_hl2], dtype=tf.float32, stddev=np.sqrt(2/nodes_hl1)))
biases_hl2=tf.get_variable('biases_hl2', [nodes_hl2], dtype=tf.float32, 
  initializer=tf.zeros_initializer)

weights_hl2_hl3=tf.get_variable('weights_hl2_hl3', dtype=tf.float32, 
  initializer=tf.truncated_normal([nodes_hl2, nodes_hl3], dtype=tf.float32, stddev=np.sqrt(2/nodes_hl2)))
biases_hl3=tf.get_variable('biases_hl3', [nodes_hl3], dtype=tf.float32, 
  initializer=tf.zeros_initializer)

weights_hl3_output=tf.get_variable('weights_hl3_output', dtype=tf.float32, 
  initializer=tf.truncated_normal([nodes_hl3, output_size], dtype=tf.float32, stddev=np.sqrt(2/nodes_hl3)))


# In[9]:


#create saver
saver=tf.train.Saver(max_to_keep=1)


# In[10]:


#dropout rate
train_keep=0.8
keep_amt=train_keep

#training pass
def forward_pass(x, keep_amt):
    dropout_rate=tf.constant(keep_amt)
    l1=tf.add(tf.matmul(x, weights_input_hl1), biases_hl1)
    l1=tf.nn.elu(l1)
    l1=tf.nn.dropout(l1, dropout_rate)
    l2=tf.add(tf.matmul(l1, weights_hl1_hl2), biases_hl2)
    l2=tf.nn.elu(l2)
    l2=tf.nn.dropout(l2, dropout_rate)
    l3=tf.add(tf.matmul(l2, weights_hl2_hl3), biases_hl3)
    l3=tf.nn.elu(l3)
    l3=tf.nn.dropout(l3, dropout_rate)
    output_layer=tf.matmul(l3, weights_hl3_output)
    return output_layer


# In[11]:


#cost and gradient descent
logits=forward_pass(features,keep_amt)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(cost)

#accuracy
output=tf.nn.softmax(logits)
accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1)),tf.float32))

#used later for predicting the test data
prediction=tf.argmax(tf.nn.softmax(logits=output),1)


# In[12]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #restore weights if file found
    try:
        saver.restore(sess, "/tmp/model.ckpt")
        print("Model restored.")
    except:
        print("No save file found.")

    
    batch_count = int(math.ceil(len(train_X)/batch_size))

    for epoch in range(num_epochs):
        
        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch+1, num_epochs), unit='batches')
        train_loss=0.0
        # The training cycle
        keep_amt=train_keep
        for batch_i in batches_pbar:
            # Get a batch of training features and labels
            batch_start = batch_i*batch_size
            batch_features = train_X[batch_start:batch_start + batch_size]
            batch_labels = train_y[batch_start:batch_start + batch_size]
            #train
            _, c = sess.run([optimizer, cost], feed_dict={features: batch_features, labels: batch_labels})
            train_loss+=c
        #set keep amount to 100% for testing
        keep_amt=1.0    
        print("Training Loss = {}, Validation Accuracy = {}"
              .format(train_loss, sess.run(accuracy, feed_dict={features: validation_X, labels: validation_y})))
        
        #save model after every 5 epochs
        #uncomment these lines if you want to save the model
        #if epoch%5==0:
        #    save_path = saver.save(sess, "/tmp/model.ckpt")
        #    print("Model saved in file: {}".format(save_path))
    print("Training Finished!")
    keep_amt=1.0
    predictions=sess.run(prediction, feed_dict={features: test_X})#.eval(session=sess)


# In[14]:


predictions


# In[15]:


#use replace the 0s in the sample submission file with the outputs from the neural  net
submission=pd.read_csv("C:/Users/Eric Zhou/Downloads/sample_submission.csv")
for x in range(len(predictions)):
    submission['Label'][x]+=predictions[x]


# In[16]:


print(submission)


# In[17]:


submission.to_csv("C:/Users/Eric Zhou/Downloads/submission.csv", index=False)

