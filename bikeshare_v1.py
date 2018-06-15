import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.externals import joblib
from datetime import datetime
import cdsw

# # Train elasticnet linear regression model with k-folds cross validation

# Load data, split into train and test

data = pd.read_csv('bikeshare.csv')
feature_cols = ['season','yr','mnth','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed']
data_features = data[feature_cols].values
data_labels = data[['cnt']].values

train_features = data_features[:486]
train_labels = data_labels[:486]
test_features = data_features[486:]
test_labels = data_labels[486:]

# Train model

model = ElasticNetCV(normalize = True, l1_ratio = [.1, .5, .7, .9, .95, .99, 1], random_state = 288793)
model.fit(train_features, train_labels)

# # Results

# ## Hyperparameters selected by CV

model.l1_ratio_
cdsw.track_metric("l1_ratio", model.l1_ratio_)

model.alpha_
cdsw.track_metric("alpha", model.alpha_)

# ## Model coefficients
model.intercept_
cdsw.track_metric("intercept", model.intercept_)

zip(feature_cols, model.coef_)
for i in range(0,len(feature_cols)):
  cdsw.track_metric(feature_cols[i], model.coef_[i])
  
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
