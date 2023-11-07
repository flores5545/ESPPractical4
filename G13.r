# Pablo Flores Fernandez: s1828496
# Chenning Jin: s2499193
# Jiapeng Ma: s2592596

# Contributions:
library(datasets)
data(iris)

d <- c(4, 8, 7, 3)
netup <- function(d){
  # This function will return a list presenting the network
  # Input: 
  #   d: a vector giving the number of nodes in each layer of a network
  # Output:
  #   a list representing the network, including:
  #     h: a list of nodes for each layer. h[[l]] should be a vector of length d[l],
  #       which will contain the node values for layer l.
  #     W: a list of weight matrices. W[[l]] is the weight matrix linking layer l to layer l+1. 
  #       Initialize the elements with U (0, 0.2) random deviates.
  #     b: a list of offset vectors. b[[l]] is the offset vector linking layer l to layer l+1. 
  #       Initialize the elements with U (0, 0.2) random deviates.
  
}

forward <- function(nn, inp){
  # forward should compute the remaining node values implied by inp, and return the updated network list 
  # Input:
  #   nn is a network list as returned by netup
  #   inp a vector of input values for the first layer. 
  # Output:
  #   return the updated network list (as the only return object).


  
}

backward <- function(nn, k){
  # This function computes the derivatives of the loss corresponding to output class k for network nn (returned from forward)
  # Input:
  #   nn: network, returned from forward
  #   k: output class
  # Output:
  #   A list of updated list including:
  #     dh, dW and db, which are the derivatives w.r.t the nodes, weights and offsets, respectively

  
}

train <- function(nn, inp, k, eta=0.1, mb=10, nstep=10000){
  # This function is used to train the network
  # Input:
  #   nn: the network
  #   inp: input data in the rows of matrix
  #   k: a vector with corresponding labels (1, 2, 3 . . . )
  #   eta: the step size η 
  #   mb: the number of data to randomly sample to compute the gradient. 
  #   nstep: the number of optimization steps to take.

  
}

# Train a 4-8-7-3 network to classify irises to species based on the 4 characteristics given in the iris dataset in R.
# Divide the iris data into training data and test data, 
# where the test data consists of every 5th row of the iris dataset, starting from row 5
# set the seed to provide an example in which training has worked and the loss has been substantially reduced from pre- to post-training
#


# compute the misclassification rate (i.e. the proportion misclassified) for the test set.





