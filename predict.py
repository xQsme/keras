import pandas as pd
import numpy as np
import keras
import random
from keras.models import Sequential
from keras.layers import Dense

concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()

# concrete_data.shape
# concrete_data.describe()
# concrete_data.isnull().sum()

concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column
predictors_norm = (predictors - predictors.mean()) / predictors.std()
n_cols = predictors_norm.shape[1] # number of predictors

# define regression model
def regression_model(hidden = 2, nodes = 10):
    # create model
    model = Sequential()
    model.add(Dense(nodes, activation='relu', input_shape=(n_cols,)))
    for i in range(hidden):
        model.add(Dense(nodes, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# build the model
# for i in range(50):

#     nodes = random.randint(1, 50)
#     hidden = random.randint(1, 25)
#     model = regression_model(hidden, nodes)

#     # fit the model
#     model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=0)

#     result = model.predict(predictors_norm)

#     diff = target - result[0]
#     total = sum(diff.abs())

#     # print(result)
#     # print hidden, nodes and avg with good formatting:
#     print("hidden: %d, nodes: %d, total: %f" % (hidden, nodes, total))

model = regression_model(17, 6)

# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=10000, verbose=0)

result = model.predict(predictors_norm)

diff = target - result[0]

print(diff.to_string(index=False))