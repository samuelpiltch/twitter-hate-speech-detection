import re
import random
import datetime

import pandas as pd
import numpy as np
import tensorflow as tf

# Load data
df = pd.read_csv('./data/labeled_data.csv')
tweets = df.values

max_tweet_length = 30 # Max word count of tweet

words_list = np.load('./data/words_list.npy')
words_list = words_list.tolist()
words_list = [word.decode('UTF-8') for word in words_list]  # Encode words as UTF-8

# Words to tokens
# ids = np.zeros((tweets.shape[0], max_tweet_length), dtype='int32')
#
# for i, tweet in enumerate(tweets):
#     text = tweet[6]
#
#     # Text cleaning
#     text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text)
#
#     split_text = text.split()
#
#     # Tokenize text
#     for j, word in enumerate(split_text):
#         try:
#             ids[i][j] = words_list.index(word)
#         except ValueError:
#             ids[i][j] = 399999 # Vector for unkown words
#         if j == max_tweet_length - 1:
#             break
#
# np.save('./data/ids_matrix', ids)

ids = np.load('./data/ids_matrix.npy')

# Convert tokens to word vectors using Glove's 50 dimension pre-trained word vectors
word_vec_dimension = 50
word_vectors = np.load('./data/word_vectors.npy')

output_labels = []
for tweet in tweets:
    label = tweet[5]
    if label == 0:
        label = [1, 0, 0]
    if label == 1:
        label = [0, 1, 0]
    if label == 2:
        label = [0, 0, 1]
    output_labels.append(label)

# Model hyperparameters
output_classes = 3

lstm_units = 64

data_size = 24783
batch_size = 33
iterations = 100000

training_ids, training_labels = ids[:16523], output_labels[:16523]
testing_ids, testing_labels = ids[16523:], output_labels[16523:]

def get_train_batch():
    global batch_index, tweets

    arr = np.zeros([batch_size, max_tweet_length])
    labels = []

    for tweet in range(batch_size):
        num = random.randint(0, training_ids.shape[0]-1)

        arr[tweet] = training_ids[num]

        label = tweets[num][5]
        if label == 0: # Hate speech
            label = [1, 0, 0]
        if label == 1: # Offensive language
            label = [0, 1, 0]
        if label == 2: # Neither
            label = [0, 0, 1]
        labels.append(label)

    return arr, labels

# Model
tf.reset_default_graph()

input_data = tf.placeholder(tf.int32, [batch_size, max_tweet_length], name="Tweets")
labels = tf.placeholder(tf.float32, [batch_size, output_classes], name="Labels")

data = tf.Variable(tf.zeros([batch_size, max_tweet_length, word_vec_dimension]), dtype=tf.float32, name="InputData")
data = tf.nn.embedding_lookup(word_vectors, input_data)

lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstm_units, output_classes]), name="Weights")
bias = tf.Variable(tf.constant(0.1, shape=[output_classes]), name="Bias")
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

# Tensorboard setup
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "./tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

# Model training
for i in range(iterations + 1):
    batch, batch_labels = get_train_batch()
    sess.run(optimizer, {input_data: batch, labels: batch_labels})

    # Write summary to Tensorboard every 50 epochs
    if (i % 50 == 0):
        summary = sess.run(merged, {input_data: batch, labels: batch_labels})
        writer.add_summary(summary, i)

    # Save the network every 10,000 epochs
    if (i % 10000 == 0 and i != 0):
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
        print("Saved to", save_path, "-", int(i/iterations*100), "%")
writer.close()

# Model prediction
saver.restore(sess, tf.train.latest_checkpoint('models'))

def get_sentence_matrix(sentence):
    arr = np.zeros([batch_size, max_tweet_length])
    sentence_matrix = np.zeros([batch_size, max_tweet_length], dtype='int32')
    split = sentence.split()
    for i, word in enumerate(split):
        if (i < max_tweet_length):
            try:
                sentence_matrix[0, i] = words_list.index(word)
            except ValueError:
                sentence_matrix[0, i] = 399999 #Vector for unkown words
    return sentence_matrix

correct = 0
for i in range(8260):
    input_matrix = get_sentence_matrix(tweets[16523 + i][6])
    predicted = sess.run(prediction, {input_data: input_matrix})[0]
    if (np.argmax(predicted) == tweets[16523 + i][5]):
        correct += 1

print("Accuracy on validation data:", correct/8259)
