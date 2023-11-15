# Pablo Flores Fernandez: s1828496
# Chenning Jin: s2499193
# Jiapeng Ma: s2592596

# Contributions:


# This file contains code to train a neural network for classification using
# Stochastic Gradient  Descent as the optimisation method to update the 
# hyperparameters of the network. First, we write a function to represent a  
# neural network given a vector with the length of each layer. Then, we code a 
# function to perform a forward pass of the neural network, updating the value 
# of each node according to some given input, using the ReLU activation function
# as the nonlinearity. Moreover, we implement a function to perform a backward 
# pass of the neural network, where we compute the derivatives of the loss
# function. For this project, we use the Softmax function to compute the 
# probability that the output variable is in a given class, and the 
# cross-entropy loss function, as often used in classification problems


netup <- function(d){
  # This function will return a list representing the neural network
  # Input: 
  #   d: a vector giving the number of nodes in each layer of a network
  # Output:
  #   a list representing the network, including:
  #     h: a list of nodes for each layer. h[[l]] should be a vector of length 
  #       d[l], which will contain the node values for layer l
  #     W: a list of weight matrices. W[[l]] is the weight matrix linking layer   
  #       l to layer l+1. Initialize the elements with U(0, 0.2) random deviates
  #     b: a list of offset vectors. b[[l]] is the offset vector linking layer l  
  #      to layer l+1. Initialize the elements with U(0, 0.2) random deviates
  
  # Initialize the nodes, weights, and offset vector
  h <- vector("list", length = length(d))
  W <- vector("list", length = length(d) - 1)
  b <- vector("list", length = length(d) - 1)
  
  for (l in 1:length(d)) { # Initialize the nodes corresponding to the given d
    h[[l]] <- rep(0, d[l])
  }
  
  # Initialize the elements of the weight matrices and the offset vectors with 
  # U(0, 0.2) random deviates
  for (l in 1:(length(d) - 1)) {
    W[[l]] <- matrix(runif(d[l] * d[l + 1], 0, 0.2), nrow = d[l], ncol = d[l + 1])
    b[[l]] <- runif(d[l + 1], 0, 0.2)
  }
  
  nn <- list(h=h, W=W, b=b) # List representing the neural network
  return (nn)
}

# ReLU activation function
relu <- function(x) {
  return(pmax(0, x))
}

forward <- function(nn, inp){
  # This function computes the remaining node values implied by the given input
  # and returns the updated network list 
  # Input:
  #   nn is a network list as returned by netup
  #   inp a vector of input values for the first layer
  # Output:
  #   return the updated network list (as the only return object)
  
  nn$h[[1]] <- inp # Set the value of the nodes in the first layer
  
  # Perform the forward pass through the network
  for (l in 1:(length(nn$h) - 1)) {
    # Compute the weighted sum of inputs and apply the ReLU activation function
    nn$h[[l + 1]] <- relu(nn$W[[l]] %*%  nn$h[[l]] + nn$b[[l]])
  }
  return (nn)
}


backward <- function(nn, k) {
  # This function computes the derivatives of the loss function corresponding to
  # output class k for network nn (returned from forward)
  # Input:
  #   nn: network, returned from forward
  #   k: output class
  # Output:
  #   A list of updated list including:
  #     dh, dW and db, which are the derivatives w.r.t the nodes, weights and 
  #     offsets, respectively
  
  # Number of layers
  n_layers <- length(nn$h)
  
  # Compute the probabilities for the last layer
  logits <- nn$h[[n_layers]]
  exp_logits <- exp(logits)
  sum_exp_logits <- sum(exp_logits)
  probabilities <- exp_logits / sum_exp_logits
  
  # Initialize dh for the last layer L
  dh <- vector("list", length = n_layers)
  dh[[n_layers]] <- probabilities
  dh[[n_layers]][k] <- dh[[n_layers]][k] - 1  # Subtract 1 only from the true class k
  
  # Initialize the derivative of the loss w.r.t. weights and biases
  dW <- vector("list", length = n_layers - 1)
  db <- vector("list", length = n_layers - 1)
  
  # Perform the backward pass through the network
  for (l in seq(n_layers - 1, 1, by = -1)) {
    
    # Recall that ReLU was used as the activation function
    # Compute derivatives with respect to biases directly from dh of the next layer
    db[[l]] <- dh[[l + 1]]
    db[[l]][nn$h[[l]] <= 0] <- 0  
    
    # Compute gradients for weights as the outer product of dh of the next layer
    # and the node values from the current layer
    dW[[l]] <- dh[[l + 1]] %*% t(nn$h[[l]]) 
    dW[[l]][nn$h[[l]] <= 0] <- 0

    # Update dh for the previous layer using the chain rule
    dh[[l]] <- t(nn$W[[l]]) %*% dh[[l + 1]]
    dh[[l]][nn$h[[l]] <= 0] <- 0
  }
  
  # Store the derivatives in the network list
  nn$dW <- dW
  nn$db <- db
  nn$dh <- dh 
  
  return (nn)
}


train <- function(nn, inp, k, eta = 0.01, mb = 10, nstep = 10000) {
  # This function is used to train the network
  # Input:
  #   nn: the network
  #   inp: input data
  #   k: a vector with corresponding labels 
  #   eta: the step size
  #   mb: the number of data to randomly sample to compute the gradient
  #   nstep: the number of optimization steps to take
  

  for (step in 1:nstep) {
    # Sample a minibatch
    idx <- sample(1:nrow(inp), mb, replace = TRUE)
    X_mb <- inp[idx, ]
    K_mb <- k[idx]
    
    # Initialize gradients
    gradients_W <- vector("list", length = length(nn$W))
    gradients_b <- vector("list", length = length(nn$b))
    
    # Loop over each layer to initialize gradients
    for (l in 1:length(gradients_W)) {
      gradients_W[[l]] <- matrix(0, nrow = nrow(nn$W[[l]]), ncol = ncol(nn$W[[l]]))
    }
    
    for (l in 1:length(gradients_b)) {
      gradients_b[[l]] <- matrix(0, nrow = length(nn$b[[l]]), ncol = 1)
    }

    
    # Accumulate gradients over the minibatch
    for (i in 1:mb) {
      # Forward pass
      nn <- forward(nn, X_mb[i, ])
      
      # Backward pass
      nn <- backward(nn, K_mb[i])
      
      # Accumulate gradients
      for (l in 1:length(nn$W)) {
        gradients_W[[l]] <- gradients_W[[l]] + nn$dW[[l]]
        gradients_b[[l]] <- gradients_b[[l]] + nn$db[[l]]
      }
    }
    
    # Update weights and biases
    for (l in 1:length(nn$W)) {
      nn$W[[l]] <- nn$W[[l]] - eta * (1/mb) * gradients_W[[l]]
      nn$b[[l]] <- nn$b[[l]] - eta * (1/mb) * gradients_b[[l]]
    }
    
    # print out the loss to monitor training every few steps
    if (step %% 1000 == 0) {
      
    }
  }
  
  return(nn)
}
  

# Train a 4-8-7-3 network to classify irises to species based on the 4 characteristics given in the iris dataset in R.
# Divide the iris data into training data and test data, 
# where the test data consists of every 5th row of the iris dataset, starting from row 5
# set the seed to provide an example in which training has worked and the loss has been substantially reduced from pre- to post-training
#

d <- c(4, 8, 7, 3)
library(datasets)
data(iris) 

iris_matrix <- as.matrix(iris)

# Divide the iris data into training and testing data
list <- seq(5, nrow(iris), by = 5)
test_data <- iris[list, ]
train_data <- iris[-list,]

inp <- as.matrix(train_data[, 1:4])
k <- as.integer(train_data[, 5])



# After training write code to classify the test data to species according to the class predicted as most probable for each iris in the test set
# compute the misclassification rate (i.e. the proportion misclassified) for the test set.
nn <- netup(d)
train <- train(nn, inp, k)
predicted_class <- predict(train, test_data[, 1:4])

misclassification_rate <- sum(predicted_class != iris_test[, 5])/length(test_data[, 5])
cat("Misclassification Rate:", misclassification_rate)



f <- forward(nn, inp)




