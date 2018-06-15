# bikeshare

Bikeshare is a simple CDSW project designed to demonstrate two features of CDSW 1.4: Experiments and Models.

## Demo outline

We create a linear regression model to predict the number of rides for a bike-share scheme given some information about the day, using Experiments to track metrics and outputs for 3 different versions as we add complexity.

Version 1 is a very basic linear regression model. We run this file in an Experiment and get various metrics out, including the chosen hyperparameters, model coefficients, and train/test scores.

Version 2, which adds a simple feature engineering step (one hot encoding of season), shows improved performance over version 1.

Version 3 however, which also adds a large number of polynomial (quadratic) features, has a severe over-fitting problem and performs poorly on the test data.

As a result we decide to deploy version 2, so we go back to that Experiment and save the model file from the output into the project itself. Now we can deploy a Model, pointing to the function predict\_rides in bikeshare\_deploy.py, which will result in the model file we just saved being loaded into our Model API.