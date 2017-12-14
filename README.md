# Background-Subtraction-GMM

Implementation of Stauffer Grimson algorithm for background subtraction based on adaptive modelling of background/foreground using Gaussian Mixture Model. 

Each pixel's history is modelled as a mixture of gaussians and the parameters are updated using Maximum Likelihood estimate. 

k = Number of Gaussians
alpha = learning rate/step
weightThresh = maximum weight for a gaussian (if greater then normalized) 
inSigma = initial value of standard deviation
