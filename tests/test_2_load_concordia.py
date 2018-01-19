import os
import sys
import time
import warnings

import dill
from nose.tools import raises
import numpy as np
from pymongo import MongoClient

sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
sys.path = [os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')] + sys.path
os.environ['is_test_suite'] = 'True'

import redis

from concordia import Concordia, load_concordia

def do_setup():
    import aml_utils

    ####################################################################
    # Setup- train model, create direct db connections, set global constants, etc.
    #####################################################################
    # TODO: create another model that uses a different algo (logisticRegression, perhaps), so we can have tests for our logic when using multiple models but each predicting off the same features
    ml_predictor_titanic, df_titanic_test = aml_utils.train_basic_binary_classifier()
    row_ids = [i for i in range(df_titanic_test.shape[0])]
    df_titanic_test['row_id'] = row_ids


    persistent_db_config = {
        'db': '__concordia_test_env'
        , 'host': 'localhost'
        , 'port': 27017
    }

    in_memory_db_config = {
        'db': 8
        , 'host': 'localhost'
        , 'port': 6379
    }


    host = in_memory_db_config['host']
    port = in_memory_db_config['port']
    db = in_memory_db_config['db']
    rdb = redis.StrictRedis(host=host, port=port, db=db)

    host = persistent_db_config['host']
    port = persistent_db_config['port']
    db = persistent_db_config['db']
    client = MongoClient(host=host, port=port)
    mdb = client[db]

    concord = load_concordia(persistent_db_config=persistent_db_config)

    existing_training_rows, _, _ = concord._get_training_data_and_predictions(model_id)
    len_existing_training_rows = existing_training_rows.shape[0]

    existing_live_rows = concord.retrieve_from_persistent_db(val_type='live_features', row_id=None, model_id=model_id)
    len_existing_live_rows = len(existing_live_rows)

    return ml_predictor_titanic, df_titanic_test, concord, rdb, mdb, len_existing_training_rows, len_existing_live_rows

model_id = 'ml_predictor_titanic_1'

ml_predictor_titanic, df_titanic_test, concord, rdb, mdb, len_existing_training_rows, len_existing_live_rows = do_setup()


len_existing_live_preds = len(concord.retrieve_from_persistent_db(val_type='live_predictions'))


def test_load_concordia_model_already_exists():

    redis_key_model = concord.make_redis_model_key(model_id)
    starting_val = rdb.get(redis_key_model)

    assert starting_val is not None

    model = concord._get_model(model_id)
    assert type(model) == type(ml_predictor_titanic)


def test_load_concordia_get_model_after_deleting_from_redis():
    rdb.delete(concord.make_redis_model_key(model_id))
    model = concord._get_model(model_id)
    assert type(model) == type(ml_predictor_titanic)


def test_load_concordia_values_already_exist_in_db():
    training_features, training_predictions, training_labels = concord._get_training_data_and_predictions(model_id)

    assert training_features.shape[0] > 400
    assert training_predictions.shape[0] > 400
    assert training_labels.shape[0] > 400
    assert training_features.shape[0] == training_predictions.shape[0] == training_labels.shape[0]


def test_load_concordia_existing_training_features_and_preds_match():
    global df_titanic_test
    df_titanic_test = df_titanic_test.copy()
    df_titanic_test = df_titanic_test.reset_index(drop=True)
    df_titanic_test['row_id'] = df_titanic_test.name
    test_preds = ml_predictor_titanic.predict_proba(df_titanic_test)
    test_labels = df_titanic_test['survived']


    assert True

    training_features, training_predictions, training_labels = concord._get_training_data_and_predictions(model_id)

    training_features = training_features.set_index('row_id', drop=False)
    training_predictions = training_predictions.set_index('row_id', drop=False)
    training_labels = training_labels.set_index('row_id', drop=False)

    feature_ids = set(training_features['row_id'])
    prediction_ids = set(training_predictions['row_id'])
    label_ids = set(training_labels['row_id'])

    print('df_titanic_test.columns')
    print(df_titanic_test.columns)
    print(df_titanic_test.row_id)

    for idx, row in df_titanic_test.iterrows():
        row = row.to_dict()
        print('row')
        print(row)

        print('id')
        print(row['row_id'])
        assert row['row_id'] in feature_ids
        concord_row = training_features.loc[row['row_id']].to_dict()

        for key in df_titanic_test.columns:
            concord_val = concord_row[key]
            direct_val = row[key]
            if direct_val != concord_val:
                assert (np.isnan(concord_val) and np.isnan(direct_val))

        assert row['row_id'] in prediction_ids
        pred_row = training_predictions.loc[row['row_id']]
        concord_pred = pred_row['prediction']
        direct_pred = test_preds[idx]
        assert round(direct_pred[0], 5) == round(concord_pred[0], 5)
        assert round(direct_pred[1], 5) == round(concord_pred[1], 5)

        assert row['row_id'] in label_ids
        label_row = training_labels.loc[row['row_id']]
        concord_label = label_row['label']
        direct_label = test_labels[idx]
        assert round(direct_label, 5) == round(concord_label, 5)
        assert round(direct_label, 5) == round(concord_label, 5)


def test_load_concordia_single_predict_matches_model_prediction():

    features = df_titanic_test.iloc[0].to_dict()
    concord_pred = concord.predict(features=features, model_id=model_id)

    raw_model_pred = ml_predictor_titanic.predict(features)

    assert raw_model_pred == concord_pred


@raises(ValueError)
def test_load_concordia_predict_passing_in_missing_model_id_raises_error():

    features = df_titanic_test.iloc[0].to_dict()
    concord_pred = concord.predict(model_id=None, features=features)

    assert False

@raises(ValueError)
def test_load_concordia_predict_passing_in_bad_model_id_raises_error():

    features = df_titanic_test.iloc[0].to_dict()
    concord_pred = concord.predict(model_id='totally_made_up_and_bad_model_id', features=features)

    assert False


def test_load_concordia_predict_adds_features_to_db():

    features = df_titanic_test.iloc[1].to_dict()
    len_existing_live_rows = len(concord.retrieve_from_persistent_db(val_type='live_features', row_id=features['name'], model_id=model_id))
    assert len_existing_live_rows > 0
    concord_pred = concord.predict(features=features, model_id=model_id)

    raw_model_pred = ml_predictor_titanic.predict(features)

    assert raw_model_pred == concord_pred

    saved_feature = concord.retrieve_from_persistent_db(val_type='live_features', row_id=features['name'], model_id=model_id)
    print('Did we remember to change the .iloc location to 2?')
    len_saved_feature = len(saved_feature)


    assert len_saved_feature - len_existing_live_rows == 1


def test_load_concordia_predict_multiple_times_with_the_same_features_adds_features_to_db_multiple_times():
    features = df_titanic_test.iloc[1].to_dict()
    len_existing_live_rows = len(concord.retrieve_from_persistent_db(val_type='live_features', row_id=features['name'], model_id=model_id))
    assert len_existing_live_rows > 0


    concord_pred = concord.predict(features=features, model_id=model_id)

    raw_model_pred = ml_predictor_titanic.predict(features)

    assert raw_model_pred == concord_pred

    saved_features = concord.retrieve_from_persistent_db(val_type='live_features', row_id=features['name'], model_id=model_id)
    len_saved_features = len(saved_features)


    assert len_saved_features - len_existing_live_rows == 1


def test_load_concordia_predict_adds_prediction_to_db():

    assert len_existing_live_preds > 0

    saved_predictions = concord.retrieve_from_persistent_db(val_type='live_predictions')
    len_saved_predictions = len(saved_predictions)


    assert len_saved_predictions - len_existing_live_preds == 3



def test_load_concordia_single_predict_proba_matches_model_prediction():

    features = df_titanic_test.iloc[0].to_dict()
    concord_pred = concord.predict_proba(features=features, model_id=model_id)

    raw_model_pred = ml_predictor_titanic.predict_proba(features)

    assert raw_model_pred[0] == concord_pred[0]
    assert raw_model_pred[1] == concord_pred[1]



@raises(ValueError)
def test_load_concordia_predict_proba_passing_in_missing_model_id_raises_error():

    features = df_titanic_test.iloc[0].to_dict()
    concord_pred = concord.predict_proba(model_id=None, features=features)

    assert False

@raises(ValueError)
def test_load_concordia_predict_proba_passing_in_bad_model_id_raises_error():

    features = df_titanic_test.iloc[0].to_dict()
    concord_pred = concord.predict_proba(model_id='totally_made_up_and_bad_model_id', features=features)

    assert False


def test_load_concordia_predict_proba_adds_features_to_db():

    features = df_titanic_test.iloc[1].to_dict()
    len_existing_live_rows = len(concord.retrieve_from_persistent_db(val_type='live_features', row_id=features['name'], model_id=model_id))
    assert len_existing_live_rows > 3

    concord_pred = concord.predict_proba(features=features, model_id=model_id)

    raw_model_pred = ml_predictor_titanic.predict_proba(features)

    assert raw_model_pred[0] == concord_pred[0]
    assert raw_model_pred[1] == concord_pred[1]

    saved_feature = concord.retrieve_from_persistent_db(val_type='live_features', row_id=features['name'], model_id=model_id)
    len_saved_feature = len(saved_feature)
    assert len_saved_feature - len_existing_live_rows == 1


def test_load_concordia_predict_proba_multiple_times_with_the_same_features_adds_features_to_db_multiple_times():
    features = df_titanic_test.iloc[1].to_dict()
    len_existing_live_rows = len(concord.retrieve_from_persistent_db(val_type='live_features', row_id=features['name'], model_id=model_id))
    assert len_existing_live_rows > 3
    concord_pred = concord.predict_proba(features=features, model_id=model_id)

    raw_model_pred = ml_predictor_titanic.predict_proba(features)

    assert raw_model_pred[0] == concord_pred[0]
    assert raw_model_pred[1] == concord_pred[1]

    saved_features = concord.retrieve_from_persistent_db(val_type='live_features', row_id=features['name'], model_id=model_id)
    len_saved_features = len(saved_features)
    assert len_saved_features - len_existing_live_rows == 1


def test_load_concordia_predict_proba_adds_prediction_to_db():
    saved_predictions = concord.retrieve_from_persistent_db(val_type='live_predictions')
    len_saved_predictions = len(saved_predictions)
    assert len_saved_predictions - len_existing_live_preds == 6



def test_load_concordia_df_predict_matches_model_predictions():

    concord_pred = concord.predict(model_id=model_id, features=df_titanic_test)

    raw_model_pred = ml_predictor_titanic.predict(df_titanic_test)

    for idx, pred in enumerate(concord_pred):
        assert pred == concord_pred[idx]


def test_load_concordia_df_predict_proba_matches_model_predictions():

    concord_pred = concord.predict_proba(model_id=model_id, features=df_titanic_test)

    raw_model_pred = ml_predictor_titanic.predict_proba(df_titanic_test)

    for idx, pred in enumerate(concord_pred):
        concord_pred_row = concord_pred[idx]
        assert pred[0] == concord_pred_row[0]
        assert pred[1] == concord_pred_row[1]



def test_load_concordia_insert_training_features_and_preds_again():
    global df_titanic_test
    df_titanic_test = df_titanic_test.copy()
    df_titanic_test = df_titanic_test.reset_index(drop=True)
    test_preds = ml_predictor_titanic.predict_proba(df_titanic_test)
    test_labels = df_titanic_test['survived']
    concord.add_data_and_predictions(model_id=model_id, data=df_titanic_test, predictions=test_preds, row_ids=df_titanic_test['name'], actuals=df_titanic_test['survived'])

    assert True

    training_features, training_predictions, training_labels = concord._get_training_data_and_predictions(model_id)

    training_features = training_features.set_index('row_id', drop=False)
    training_predictions = training_predictions.set_index('row_id', drop=False)
    training_labels = training_labels.set_index('row_id', drop=False)

    feature_ids = set(training_features['row_id'])
    prediction_ids = set(training_predictions['row_id'])
    label_ids = set(training_labels['row_id'])

    for idx, row in df_titanic_test.iterrows():
        row = row.to_dict()
        assert row['row_id'] in feature_ids
        concord_row = training_features.loc[row['row_id']]
        assert concord_row.shape[0] == 2
        concord_row = concord_row.iloc[1].to_dict()

        for key in df_titanic_test.columns:
            concord_val = concord_row[key]
            direct_val = row[key]
            if direct_val != concord_val:
                assert (np.isnan(concord_val) and np.isnan(direct_val))

        assert row['row_id'] in prediction_ids
        pred_row = training_predictions.loc[row['row_id']]
        assert pred_row.shape[0] == 2
        concord_pred = pred_row.iloc[1]['prediction']
        direct_pred = test_preds[idx]
        assert round(direct_pred[0], 5) == round(concord_pred[0], 5)
        assert round(direct_pred[1], 5) == round(concord_pred[1], 5)

        assert row['row_id'] in label_ids
        label_row = training_labels.loc[row['row_id']]
        assert label_row.shape[0] == 2
        concord_label = label_row.iloc[1]['label']
        direct_label = test_labels[idx]
        assert round(direct_label, 5) == round(concord_label, 5)
        assert round(direct_label, 5) == round(concord_label, 5)

