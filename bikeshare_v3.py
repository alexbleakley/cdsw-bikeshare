import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from datetime import datetime
import cdsw

# # Train elasticnet linear regression model with k-folds cross validation

# Load data, split into train and test

data = pd.read_csv('bikeshare.csv')
feature_cols = ['season','yr','mnth','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed']
data_features = data[feature_cols]
data_labels = data[['cnt']]

train_features = data_features[:486]
train_labels = data_labels[:486]
test_features = data_features[486:]
test_labels = data_labels[486:]

# Train model

encode_season = OneHotEncoder(categorical_features=[0], sparse = False)
poly = PolynomialFeatures(degree=2)
elasticnet = ElasticNetCV(normalize = True, l1_ratio = [.1, .5, .7, .9, .95, .99, 1], random_state = 288793)

model = Pipeline([('encode_season', encode_season),
                  ('poly', poly), 
                  ('elasticnet', elasticnet)])

model.fit(train_features, train_labels)

# # Results

# ## Hyperparameters selected by CV
model.named_steps['elasticnet'].l1_ratio_
cdsw.track_metric("l1_ratio", model.named_steps['elasticnet'].l1_ratio_)

model.named_steps['elasticnet'].alpha_
cdsw.track_metric("alpha", model.named_steps['elasticnet'].alpha_)

# ## Model coefficients
model.named_steps['elasticnet'].intercept_
cdsw.track_metric("intercept", model.named_steps['elasticnet'].intercept_)

model.named_steps['elasticnet'].coef_

# ## r squared scores
r_train = model.score(train_features, train_labels)
r_train
cdsw.track_metric("r_train", r_train)

r_test = model.score(test_features, test_labels)
r_test
cdsw.track_metric("r_test", r_test)

# ## Persist model during experiment 
filename = 'bikeshare_model.pkl'
joblib.dump(model, filename)
cdsw.track_file(filename)

#timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
#joblib.dump(model, 'bikeshare_model_' + timestamp + '.pkl')