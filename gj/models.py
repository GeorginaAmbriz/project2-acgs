# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 08:59:57 2019

@author: Geo
"""

import torch 
import torch.nn as nn

#import numpy as np
#import torch.nn.functional as F
#import math, time
import copy 
#from torch.autograd import Variable
from torch.nn.parameter import Parameter
#import matplotlib.pyplot as plt

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is 
# what the main script expects. If you modify the contract, 
# you must justify that choice, note it in your report, and notify the TAs 
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which 
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention. 


def clones(module, N):
    """
    A helper function for producing N identical layers (each with their own parameters).
    
    inputs: 
        module: a pytorch nn.module
        N (int): the number of copies of that module to return

    returns:
        a ModuleList with the copies of the module (the ModuleList is itself also a module)
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



# Problem 1
    
class Layer(nn.Module):
  
  def __init__(self, hidden_size):
  
    super(Layer, self).__init__()
    self.hidden_size=hidden_size
    self.fc = nn.Linear (self.hidden_size, self.hidden_size, bias = True)
    self.rec = nn.Linear (self.hidden_size, self.hidden_size, bias = True)
    
    
class RNN(nn.Module): # Implement a stacked vanilla RNN with Tanh nonlinearities.
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
    """
    emb_size:     The number of units in the input embeddings
    hidden_size:  The number of hidden units per layer
    seq_len:      The length of the input sequences
    vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
    num_layers:   The depth of the stack (i.e. the number of hidden layers at 
                  each time-step)
    dp_keep_prob: The probability of *not* dropping out units in the 
                  non-recurrent connections.
                  Do not apply dropout on recurrent connections.
    """
    super(RNN, self).__init__()

    # TODO ========================
    # Initialization of the parameters of the recurrent and fc layers. 
    # Your implementation should support any number of stacked hidden layers 
    # (specified by num_layers), use an input embedding layer, and include fully
    # connected layers with dropout after each recurrent layer.
    # Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding 
    # modules, but not recurrent modules.
    #
    # To create a variable number of parameter tensors and/or nn.Modules 
    # (for the stacked hidden layer), you may need to use nn.ModuleList or the 
    # provided clones function (as opposed to a regular python list), in order 
    # for Pytorch to recognize these parameters as belonging to this nn.Module 
    # and compute their gradients automatically. You're not obligated to use the
    # provided clones function.
    
    self.emb_size = emb_size 
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.num_layers = num_layers
    self.dp_keep_prob = dp_keep_prob
    
    self.embedding = nn.Embedding (self.vocab_size, self.emb_size)
    self.linear_fc1 = nn.Linear (self.hidden_size, self.emb_size, bias = True)
    self.linear_rec1 = nn.Linear (self.hidden_size, self.hidden_size, bias = True)
    h_1=Layer(hidden_size)
    self.h_layers = clones(h_1, self.num_layers)
    self.linear_out= nn.Linear (self.hidden_size, self.vocab_size, bias = True)

    self.init_weights() 

  def init_weights(self):
    # TODO ========================
    # Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
    # and output biases to 0 (in place). The embeddings should not use a bias vector.
    # Initialize all other (i.e. recurrent and linear) weights AND biases uniformly 
    # in the range [-k, k] where k is the square root of 1/hidden_size
    
    self.embedding.weight = Parameter(torch.Tensor(self.vocab_size, self.emb_size, 
                                                   ).uniform_(-0.1, 0.1))
    self.linear_out.weight = Parameter(torch.Tensor(self.hidden_size, 
                                                    self.vocab_size).uniform_(-0.1, 0.1))
    self.linear_out.bias = Parameter(torch.zeros(self.vocab_size))

  def init_hidden(self):
    # TODO ========================
    # initialize the hidden states to zero
    """
    This is used for the first mini-batch in an epoch, only.
    """
    #return a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)

    return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).requires_grad_()

  def forward(self, inputs, hidden):
    # TODO ========================
    # Compute the forward pass, using nested python for loops.
    # The outer for loop should iterate over timesteps, and the 
    # inner for loop should iterate over hidden layers of the stack. 
    # 
    # Within these for loops, use the parameter tensors and/or nn.modules you 
    # created in __init__ to compute the recurrent updates according to the 
    # equations provided in the .tex of the assignment.
    #
    # Note that those equations are for a single hidden-layer RNN, not a stacked
    # RNN. For a stacked RNN, the hidden states of the l-th layer are used as 
    # inputs to to the {l+1}-st layer (taking the place of the input sequence).

    """
    Arguments:
        - inputs: A mini-batch of input sequences, composed of integers that 
                    represent the index of the current token(s) in the vocabulary.
                        shape: (seq_len, batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)
    
    Returns:
        - Logits for the softmax over output tokens at every time-step.
              **Do NOT apply softmax to the outputs!**
              Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does 
              this computation implicitly.
                    shape: (seq_len, batch_size, vocab_size)
        - The final hidden states for every layer of the stacked RNN.
              These will be used as the initial hidden states for all the 
              mini-batches in an epoch, except for the first, where the return 
              value of self.init_hidden will be used.
              See the repackage_hiddens function in ptb-lm.py for more details, 
              if you are curious.
                    shape: (num_layers, batch_size, hidden_size)
    """
    print (inputs.shape)
    logits = torch.zeros(self.seq_len, self.batch_size, self.vocab_size).requires_grad_()
    drop = nn.Dropout(self.dp_keep_prob) 
    tanh = nn.Tanh()
    old_hidden = hidden.clone()
    
    
                  
    # iterate over timesteps
    for i in range(self.seq_len):
        
      embeds = self.embedding(inputs[i])
      x=drop(embeds)      
      h_0= tanh(self.linear_fc1(x))
      
      # iterate over hidden layers
      
      for j in range(self.num_layers):
        
        if i == 0 and j == 0:
          
          hidden[i]  = tanh(self.h_layers[i].fc (drop(h_0))) 
         
        elif j > 0 and i == 0:
          
          hidden[i] = tanh(self.h_layers[i].fc (drop(hidden[i-1])))
                  
        elif j == 0 and i > 0:
          
          hidden[i] = tanh(self.h_layers[i].fc (drop(h_0)) + 
                                  self.h_layers[i].rec (old_hidden[i]))          
        else:
          
          hidden[i] = tanh(self.h_layers[i].fc (drop(hidden[i-1])) +
                                  self.h_layers[i].rec(old_hidden(i)))
                  
      out_layer = self.linear_out (hidden[self.num_layers - 1] )
                  
      logits[j] = out_layer
      
      old_hidden = hidden.clone()
    
    return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

  def generate(self, input, hidden, generated_seq_len):
    # TODO ========================
    # Compute the forward pass, as in the self.forward method (above).
    # You'll probably want to copy substantial portions of that code here.
    # 
    # We "seed" the generation by providing the first inputs.
    # Subsequent inputs are generated by sampling from the output distribution, 
    # as described in the tex (Problem 5.3)
    # Unlike for self.forward, you WILL need to apply the softmax activation 
    # function here in order to compute the parameters of the categorical 
    # distributions to be sampled from at each time-step.

    """
    Arguments:
        - input: A mini-batch of input tokens (NOT sequences!)
                        shape: (batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)
        - generated_seq_len: The length of the sequence to generate.
                       Note that this can be different than the length used 
                       for training (self.seq_len)
    Returns:
        - Sampled sequences of tokens
                    shape: (generated_seq_len, batch_size)
    """
    pass
    #return samples

