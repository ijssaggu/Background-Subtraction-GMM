# Background-Subtraction-GMM

Implementation of Stauffer Grimson algorithm for background subtraction based on adaptive modelling of background/foreground using Gaussian Mixture Model. 

Each pixel's history is modelled as a mixture of gaussians and the parameters are updated using Maximum Likelihood estimate. 

k = Number of Gaussians <br />
alpha = learning rate/step <br />
weightThresh = maximum weight for a gaussian (if greater then normalized)<br />
inSigma = initial value of standard deviation

![](umcp.gif)
