# Created by xjiao004 at 02/05/2020
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .data_loader import sentence_proc, pad_proc, transform_data
from .losses import loss_function
from .config import BATCH_SIZE

def evaluate(encoder, decoder, sentence, vocab, reverse_vocab, units, input_length, output_length, start_index):
  attention_plot = np.zeros((output_length, input_length))

  x_max_len = input_length - 4
  sentence = sentence_proc(sentence)
  sentence = pad_proc(sentence, x_max_len, vocab)

  inputs = transform_data(sentence, vocab)
  inputs = tf.convert_to_tensor([inputs])

  result = ''

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([start_index], 0)

  for t in range(output_length):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()

    result += reverse_vocab[predicted_id] + ' '

    if reverse_vocab[predicted_id] == '<STOP>':
      return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot


# In[28]:


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
  fig = plt.figure(figsize=(300,300))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')

  fontdict = {'fontsize': 10}

  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()


