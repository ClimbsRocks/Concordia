import os
import random
import sys
import time
import warnings

from auto_ml import load_ml_model
import dill
from nose.tools import raises
import numpy as np
import pandas as pd
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



def test_compare_proba_predictions_finds_a_fixed_delta_of_1():
    model_id = 'ml_predictor_titanic_{}'.format(random.random())

    concord.add_model(model=ml_predictor_titanic, model_id=model_id, feature_importances=ml_predictor_titanic.feature_importances_)

    concord.predict_proba(model_id, df_titanic_test)

    train_preds = ml_predictor_titanic.predict_proba(df_titanic_test)

    modified_preds = []
    for pred in train_preds:
        pred[0] = pred[0] - 0.1
        pred[1] = pred[1] + 0.1
        modified_preds.append(pred)

    concord.add_data_and_predictions(model_id=model_id, features=df_titanic_test, predictions=modified_preds, row_ids=df_titanic_test.name, actuals=df_titanic_test.survived)

    results = concord.analyze_prediction_discrepancies(model_id=model_id, return_summary=True, return_deltas=True, return_matched_rows=False, sort_column=None, min_date=None, date_field=None, verbose=True)

    deltas = results['deltas']

    class_0_delta_val = round(deltas['class_0_delta'].mean(), 5)
    class_1_delta_val = round(deltas['class_1_delta'].mean(), 5)
    assert class_0_delta_val == -0.10000
    assert class_1_delta_val == 0.10000


def test_compare_proba_predictions_finds_no_deltas_when_deltas_do_not_exist():
    model_id = 'ml_predictor_titanic_{}'.format(random.random())

    concord.add_model(model=ml_predictor_titanic, model_id=model_id, feature_importances=ml_predictor_titanic.feature_importances_)

    concord.predict_proba(model_id, df_titanic_test)

    train_preds = ml_predictor_titanic.predict_proba(df_titanic_test)

    concord.add_data_and_predictions(model_id=model_id, features=df_titanic_test, predictions=train_preds, row_ids=df_titanic_test.name, actuals=df_titanic_test.survived)

    results = concord.analyze_prediction_discrepancies(model_id=model_id, return_summary=True, return_deltas=True, return_matched_rows=False, sort_column=None, min_date=None, date_field=None, verbose=True)

    deltas = results['deltas']

    class_0_delta_val = round(deltas['class_0_delta'].mean(), 5)
    class_1_delta_val = round(deltas['class_1_delta'].mean(), 5)
    assert class_0_delta_val == 0
    assert class_1_delta_val == 0


def test_compare_predict_predictions_finds_a_fixed_delta_of_1():
    model_id = 'ml_predictor_titanic_{}'.format(random.random())

    concord.add_model(model=ml_predictor_titanic, model_id=model_id, feature_importances=ml_predictor_titanic.feature_importances_)

    concord.predict(model_id, df_titanic_test)

    train_preds = ml_predictor_titanic.predict(df_titanic_test)

    modified_preds = []
    for pred in train_preds:
        pred = pred - 0.1
        modified_preds.append(pred)

    concord.add_data_and_predictions(model_id=model_id, features=df_titanic_test, predictions=modified_preds, row_ids=df_titanic_test.name, actuals=df_titanic_test.survived)

    results = concord.analyze_prediction_discrepancies(model_id=model_id, return_summary=True, return_deltas=True, return_matched_rows=False, sort_column=None, min_date=None, date_field=None, verbose=True)

    deltas = results['deltas']

    prediction_delta_val = round(deltas['delta'].mean(), 5)
    assert prediction_delta_val == -0.10000


def test_compare_predict_predictions_finds_no_deltas_when_deltas_do_not_exist():
    model_id = 'ml_predictor_titanic_{}'.format(random.random())

    concord.add_model(model=ml_predictor_titanic, model_id=model_id, feature_importances=ml_predictor_titanic.feature_importances_)

    concord.predict(model_id, df_titanic_test)

    train_preds = ml_predictor_titanic.predict(df_titanic_test)

    concord.add_data_and_predictions(model_id=model_id, features=df_titanic_test, predictions=train_preds, row_ids=df_titanic_test.name, actuals=df_titanic_test.survived)

    results = concord.analyze_prediction_discrepancies(model_id=model_id, return_summary=True, return_deltas=True, return_matched_rows=False, sort_column=None, min_date=None, date_field=None, verbose=True)

    deltas = results['deltas']
    prediction_delta_val = round(deltas['delta'].mean(), 5)
    assert prediction_delta_val == 0


def test_find_missing_cols():
    columns = [
        'feature_1'
        , 'feature_2_train'
        , 'feature_2_live'
        , 'feature_3_train'
        , 'feature_3_live'

        , 'feature_4_live'

        , 'feature_5_train'
    ]
    df = pd.DataFrame(0, index=np.arange(10), columns=columns)
    results = concord.find_missing_columns(df)

    assert len(results['matched_cols']) == 2
    assert 'feature_2' in results['matched_cols']
    assert 'feature_3' in results['matched_cols']
    assert len(results['live_columns_not_in_train']) == 1
    assert 'feature_4' in results['live_columns_not_in_train']
    assert len(results['train_columns_not_in_live']) == 1
    assert 'feature_5' in results['train_columns_not_in_live']


def test_compare_features_finds_no_deltas_when_deltas_do_not_exist():
    model_id = 'ml_predictor_titanic_{}'.format(random.random())

    concord.add_model(model=ml_predictor_titanic, model_id=model_id, feature_importances=ml_predictor_titanic.feature_importances_)

    concord.predict(model_id, df_titanic_test)

    train_preds = ml_predictor_titanic.predict(df_titanic_test)

    concord.add_data_and_predictions(model_id=model_id, features=df_titanic_test, predictions=train_preds, row_ids=df_titanic_test.name, actuals=df_titanic_test.survived)

    results = concord.analyze_feature_discrepancies(model_id=model_id, return_summary=True, return_deltas=True, return_matched_rows=False, sort_column=None, min_date=None, date_field=None, verbose=True, ignore_duplicates=True)

    print('results')
    print(results)

    deltas = results['deltas']
    for col in deltas.columns:
        prediction_delta_val = round(deltas[col].mean(), 5)
        assert prediction_delta_val == 0


def test_compare_features_works_even_with_no_feature_importances():
    model_id = 'ml_predictor_titanic_{}'.format(random.random())

    concord.add_model(model=ml_predictor_titanic, model_id=model_id, feature_importances=None)

    concord.predict(model_id, df_titanic_test)

    train_preds = ml_predictor_titanic.predict(df_titanic_test)

    concord.add_data_and_predictions(model_id=model_id, features=df_titanic_test, predictions=train_preds, row_ids=df_titanic_test.name, actuals=df_titanic_test.survived)

    results = concord.analyze_feature_discrepancies(model_id=model_id, return_summary=True, return_deltas=True, return_matched_rows=False, sort_column=None, min_date=None, date_field=None, verbose=True, ignore_duplicates=True)

    print('results')
    print(results)

    deltas = results['deltas']
    for col in deltas.columns:
        prediction_delta_val = round(deltas[col].mean(), 5)
        assert prediction_delta_val == 0


@raises(TypeError)
def test_bad_feature_importances_type_raises_type_error():
    model_id = 'ml_predictor_titanic_{}'.format(random.random())

    concord.add_model(model=ml_predictor_titanic, model_id=model_id, feature_importances=['this will not work'])



def test_compare_features_finds_deltas_when_deltas_do_exist():
    model_id = 'ml_predictor_titanic_{}'.format(random.random())


    concord.add_model(model=ml_predictor_titanic, model_id=model_id, feature_importances=ml_predictor_titanic.feature_importances_)

    concord.predict(model_id, df_titanic_test)

    col_deltas = {}
    for col in df_titanic_test.columns:
        if df_titanic_test[col].dtype != 'object':
            col_delta = random.random()
            df_titanic_test[col] = df_titanic_test[col] - col_delta
            col_deltas[col] = col_delta
        else:
            col_deltas[col] = 0

    train_preds = ml_predictor_titanic.predict(df_titanic_test)

    concord.add_data_and_predictions(model_id=model_id, features=df_titanic_test, predictions=train_preds, row_ids=df_titanic_test.name, actuals=df_titanic_test.survived)

    results = concord.analyze_feature_discrepancies(model_id=model_id, return_summary=True, return_deltas=True, return_matched_rows=False, sort_column=None, min_date=None, date_field=None, verbose=True, ignore_duplicates=True)

    deltas = results['deltas']
    for col in deltas.columns:
        if col == 'model_id':
            continue
        prediction_delta_val = round(np.nanmean(deltas[col].values), 5)
        delta_val = round(col_deltas[col], 5)
        if col == 'age':
            assert prediction_delta_val >= -delta_val
        else:
            assert prediction_delta_val == -delta_val


def test_raises_warning_when_it_appears_row_id_types_mismatch():
    model_id = 'ml_predictor_titanic_{}'.format(random.random())

    concord.add_model(model=ml_predictor_titanic, model_id=model_id, feature_importances=ml_predictor_titanic.feature_importances_)

    concord.predict(model_id, df_titanic_test)

    df_titanic_test['name'] = df_titanic_test.name.apply(lambda val: '{}_{}'.format(val, random.random()))
    train_preds = ml_predictor_titanic.predict(df_titanic_test)

    concord.add_data_and_predictions(model_id=model_id, features=df_titanic_test, predictions=train_preds, row_ids=df_titanic_test.name, actuals=df_titanic_test.survived)


    with warnings.catch_warnings(record=True) as w:
        results = concord.analyze_prediction_discrepancies(model_id=model_id, return_summary=True, return_deltas=True, return_matched_rows=False, sort_column=None, min_date=None, date_field=None, verbose=True, ignore_duplicates=True)
        assert len(w) == 1

    assert results['deltas'].shape[0] == 0


@raises(TypeError)
def test_raises_error_when_date_is_specified_without_datefield_but_min_date_is_not_datetime():
    model_id = 'ml_predictor_titanic_{}'.format(random.random())

    concord.add_model(model=ml_predictor_titanic, model_id=model_id, feature_importances=ml_predictor_titanic.feature_importances_)

    concord.predict(model_id, df_titanic_test)

    train_preds = ml_predictor_titanic.predict(df_titanic_test)

    concord.add_data_and_predictions(model_id=model_id, features=df_titanic_test, predictions=train_preds, row_ids=df_titanic_test.name, actuals=df_titanic_test.survived)


    results = concord.analyze_prediction_discrepancies(model_id=model_id, min_date='12345', date_field=None)

    assert False

