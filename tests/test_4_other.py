import os
import random
import sys
import time
import warnings

from auto_ml import load_ml_model
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
    file_name = '_test_suite_saved_pipeline.dill'
    ml_predictor_titanic.save(file_name)
    ml_predictor_titanic = load_ml_model(file_name)
    os.remove(file_name)



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

model_id = 'ml_predictor_titanic_3'

ml_predictor_titanic, df_titanic_test, concord, rdb, mdb, len_existing_training_rows, len_existing_live_rows = do_setup()


len_existing_live_preds = len(concord.retrieve_from_persistent_db(val_type='live_predictions'))


def test_df_no_row_id_is_ok_with_default_row_id_feild():

    features = df_titanic_test[:15].copy()

    if 'row_id' in features.columns:
        features = features.drop('row_id')

    assert 'row_id' not in features.columns
    assert 'name' in features.columns

    concord._insert_df_into_db(df=features, val_type='training_features', row_id=None, model_id=model_id)

    assert True

@raises(ValueError)
def test_df_no_row_id_raises_error_with_no_row_id_field():

    features = df_titanic_test[:15].copy()
    del features['name']

    assert 'row_id' not in features.columns
    assert 'name' not in features.columns

    concord._insert_df_into_db(df=features, val_type='training_features', row_id=None, model_id=model_id)

    assert False


@raises(ValueError)
def test_df_no_model_id_raises_error_always():

    features = df_titanic_test[:15].copy()
    assert 'row_id' not in features.columns
    assert 'name' in features.columns

    concord._insert_df_into_db(df=features, val_type='training_features', row_id=None, model_id=None)

    assert False


@raises(ValueError)
def test_df_no_model_id_raises_error_always():

    features = df_titanic_test[:15].copy()
    assert 'row_id' not in features.columns
    assert 'name' in features.columns

    concord._insert_df_into_db(df=features, val_type='training_features', row_id=None, model_id=None)

    assert False


# def test_add_data_and_predictions_takes_in_dicts():
#     # TODO: design a test for this. insert it, then try to retieve it after
#     pass


# def test_works_on_larger_datasets
