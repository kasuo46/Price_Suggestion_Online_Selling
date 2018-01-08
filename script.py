import numpy as np
import pandas as pd
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation
from keras.layers import concatenate, GRU, Embedding, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K
from keras import optimizers
from keras import initializers
from sklearn.metrics import mean_squared_log_error

train = pd.read_csv('../input/train.tsv', sep='\t', encoding='utf-8')
test = pd.read_csv('../input/test.tsv', sep='\t', encoding='utf-8')

train['category_name'].fillna(value="missing/missing/missing", inplace=True)
train['brand_name'].fillna(value="missing", inplace=True)
train['item_description'].fillna(value="No description yet", inplace=True)
test['category_name'].fillna(value="missing/missing/missing", inplace=True)
test['brand_name'].fillna(value="missing", inplace=True)

le_category_name = LabelEncoder()
le_category_name.fit(np.hstack((train['category_name'], test['category_name'])))
train['category_name_id'] = le_category_name.transform(train['category_name'])
test['category_name_id'] = le_category_name.transform(test['category_name'])

le_brand_name = LabelEncoder()
le_brand_name.fit(np.hstack((train['brand_name'], test['brand_name'])))
train['brand_name_id'] = le_brand_name.transform(train['brand_name'])
test['brand_name_id'] = le_brand_name.transform(test['brand_name'])

tok = Tokenizer()
text_train_orig = np.hstack((train['category_name'].str.lower(),
                             train['name'].str.lower(),
                             train['item_description'].str.lower()))
tok.fit_on_texts(text_train_orig)
train["seq_category_name"] = tok.texts_to_sequences(train['category_name'].str.lower())
test["seq_category_name"] = tok.texts_to_sequences(test['category_name'].str.lower())
train["seq_item_description"] = tok.texts_to_sequences(train['item_description'].str.lower())
test["seq_item_description"] = tok.texts_to_sequences(test['item_description'].str.lower())
train["seq_name"] = tok.texts_to_sequences(train['name'].str.lower())
test["seq_name"] = tok.texts_to_sequences(test['name'].str.lower())

# MAX_SEQ_CATEGORY_NAME = np.max([train['seq_category_name'].apply(len).max(),
#         test['seq_category_name'].apply(len).max()])
MAX_SEQ_CATEGORY_NAME = 20
# MAX_SEQ_NAME = np.max([train['seq_name'].apply(len).max(),
#         test['seq_name'].apply(len).max()])
MAX_SEQ_NAME = 20
MAX_SEQ_ITEM_DESCRIPTION = 60
MAX_CATEGORY_NAME_ID = np.max([train['category_name_id'].max(),
        test['category_name_id'].max()]) + 1
MAX_BRAND_NAME_ID = np.max([train['brand_name_id'].max(),
        test['brand_name_id'].max()]) + 1
MAX_ITEM_CONDITION_ID = np.max([train['item_condition_id'].max(),
        test['item_condition_id'].max()]) + 1
# MAX_TEXT = np.max([np.max(train['seq_category_name'].max()),
#                   np.max(test['seq_category_name'].max()),
#                   np.max(train['seq_item_description'].max()),
#                   np.max(test['seq_item_description'].max()),
#                   np.max(train['seq_name'].max()),
#                   np.max(test['seq_name'].max()),]) + 2
MAX_TEXT = len(tok.word_index) + 1

train['target'] = np.log1p(train['price'])
target_scaler = MinMaxScaler(feature_range=(-1, 1))
train['target'] = target_scaler.fit_transform(train['target'].reshape(-1,1))

dtrain, dval = train_test_split(train, random_state=1987, test_size=0.01)


def get_keras_data(d):
    X = {'seq_name': pad_sequences(d['seq_name'], maxlen=MAX_SEQ_NAME),
         'seq_item_description': pad_sequences(d['seq_item_description'],
                                               maxlen=MAX_SEQ_ITEM_DESCRIPTION),
         'brand_name_id': np.array(d['brand_name_id']),
         'category_name_id': np.array(d['category_name_id']),
         'seq_category_name': pad_sequences(d['seq_category_name'],
                                            maxlen=MAX_SEQ_CATEGORY_NAME),
         'item_condition_id': np.array(d['item_condition_id']),
         'shipping': np.array(d['shipping'])}
    return X


Xtrain = get_keras_data(dtrain)
Xval = get_keras_data(dval)
Xtest = get_keras_data(test)


def rmsle(y, pred):
    return mean_squared_log_error(y, pred)**0.5
    
    
def rmsle_cust(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred), axis=-1))
    
    
def get_model():
    seq_name = Input(shape=[Xtrain['seq_name'].shape[1]], name='seq_name')
    seq_item_description = Input(shape=[Xtrain['seq_item_description'].shape[1]],
                                 name='seq_item_description')
    brand_name_id = Input(shape=[1], name='brand_name_id')
    category_name_id = Input(shape=[1], name='category_name_id')
    seq_category_name = Input(shape=[Xtrain['seq_category_name'].shape[1]], 
                          name='seq_category_name')
    item_condition_id = Input(shape=[1], name='item_condition_id')
    shipping = Input(shape=[1], name='shipping')
    
    emb_seq_name = Embedding(MAX_TEXT, 20) (seq_name)
    emb_seq_item_description = Embedding(MAX_TEXT, 60) (seq_item_description)
    emb_brand_name_id = Embedding(MAX_BRAND_NAME_ID, 10) (brand_name_id)
    emb_category_name_id = Embedding(MAX_CATEGORY_NAME_ID, 10) (category_name_id)
    emb_seq_category_name = Embedding(MAX_TEXT, 20) (seq_category_name)
    emb_item_condition_id = Embedding(MAX_ITEM_CONDITION_ID, 5) (item_condition_id)
    
    rnn_layer1 = GRU(16) (emb_seq_item_description)
    rnn_layer2 = GRU(8) (emb_seq_category_name)
    rnn_layer3 = GRU(8) (emb_seq_name)
    
    main_layer = concatenate([Flatten() (emb_brand_name_id),
                              Flatten() (emb_category_name_id),
                              Flatten() (emb_item_condition_id),
                              rnn_layer1,
                              rnn_layer2,
                              rnn_layer3,
                              shipping])
    main_layer = Dense(128) (main_layer)
    # main_layer = BatchNormalization() (main_layer)
    main_layer = Activation('relu') (main_layer)
    main_layer = Dropout(0.25) (main_layer)
    main_layer = Dense(128) (main_layer)
    # main_layer = BatchNormalization() (main_layer)
    main_layer = Activation('relu') (main_layer)
    main_layer = Dropout(0.1) (main_layer)
    main_layer = Dense(1) (main_layer)
    # main_layer = BatchNormalization() (main_layer)
    output = Activation('linear') (main_layer)
    model = Model([seq_name, seq_item_description, brand_name_id, category_name_id,
                   seq_category_name, item_condition_id, shipping], output)
    model.compile(loss='mse', optimizer='adam', metrics=[rmsle_cust])
    return model
    
    
model = get_model()
model.summary()

model.fit(Xtrain, dtrain['target'], epochs=6, batch_size=256*256, validation_data=(Xval, dval['target']), verbose=1)

val_preds = model.predict(Xval)
val_preds = target_scaler.inverse_transform(val_preds)
val_preds = np.exp(val_preds) - 1

y_true = np.array(dval['price'].values)
y_pred = val_preds[:,0]
v_rmsle = rmsle(y_true, y_pred)
print(" RMSLE error on dev test: "+str(v_rmsle))