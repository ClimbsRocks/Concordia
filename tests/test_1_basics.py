import os
import sys
import warnings

from auto_ml.utils_models import load_ml_model
import dill
from nose.tools import raises
import numpy as np
from pymongo import MongoClient

sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
sys.path = [os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')] + sys.path
os.environ['is_test_suite'] = 'True'

import redis

from concordia import Concordia

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
    # row_ids = [i for i in range(df_titanic_test.shape[0])]
    # df_titanic_test['row_id'] = df_titanic_test.name

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
    client.drop_database(db)
    mdb = client[db]

    rdb.flushdb()

    concord = Concordia(in_memory_db_config=in_memory_db_config, persistent_db_config=persistent_db_config, default_row_id_field='name')

    return ml_predictor_titanic, df_titanic_test, concord, rdb, mdb

model_id = 'ml_predictor_titanic_1'

ml_predictor_titanic, df_titanic_test, concord, rdb, mdb = do_setup()


def test_add_new_model():

    redis_key_model = concord.make_redis_model_key(model_id)
    starting_val = rdb.get(redis_key_model)

    assert starting_val is None

    importances_dict = ml_predictor_titanic.feature_importances_

    concord.add_model(model=ml_predictor_titanic, model_id=model_id, feature_importances=importances_dict)

    post_insert_val = rdb.get(redis_key_model)
    assert post_insert_val is not None



def test_get_model():
    model = concord._get_model(model_id)
    assert type(model) == type(ml_predictor_titanic)


def test_get_model_after_deleting_from_redis():
    rdb.delete(concord.make_redis_model_key(model_id))
    model = concord._get_model(model_id)
    assert type(model) == type(ml_predictor_titanic)


def test_insert_training_features_and_preds():
    df_titanic_local = df_titanic_test.copy()
    test_preds = ml_predictor_titanic.predict_proba(df_titanic_local)
    test_labels = list(df_titanic_local['survived'])

    concord.add_data_and_predictions(model_id=model_id, features=df_titanic_local, predictions=test_preds, row_ids=df_titanic_local['name'], actuals=df_titanic_local['survived'])

    assert True

    training_features, training_predictions, training_labels = concord._get_training_data_and_predictions(model_id)

    training_features = training_features.set_index('name', drop=False)

    # training_predictions['name'] = training_predictions.row_id
    training_predictions = training_predictions.set_index('row_id', drop=False)

    # training_labels['name'] = training_predictions.row_id
    training_labels = training_labels.set_index('row_id', drop=False)

    feature_ids = set(training_features['row_id'])
    prediction_ids = set(training_predictions['row_id'])
    label_ids = set(training_labels['row_id'])



    idx = 0
    for _, row in df_titanic_test.iterrows():
        row = row.to_dict()
        assert row['name'] in feature_ids
        concord_row = training_features.loc[row['name']].to_dict()

        for key in df_titanic_test.columns:
            concord_val = concord_row[key]
            direct_val = row[key]
            if direct_val != concord_val:
                assert (np.isnan(concord_val) and np.isnan(direct_val))

        assert row['name'] in prediction_ids
        pred_row = training_predictions.loc[row['name']]
        concord_pred = pred_row['prediction']
        direct_pred = test_preds[idx]
        assert round(direct_pred[0], 5) == round(concord_pred[0], 5)
        assert round(direct_pred[1], 5) == round(concord_pred[1], 5)

        assert row['name'] in label_ids
        label_row = training_labels.loc[row['name']]
        concord_label = label_row['label']
        direct_label = test_labels[idx]
        assert round(direct_label, 5) == round(concord_label, 5)
        assert round(direct_label, 5) == round(concord_label, 5)
        idx += 1



def test_single_predict_matches_model_prediction():

    features = df_titanic_test.iloc[0].to_dict()
    concord_pred = concord.predict(features=features, model_id=model_id)

    raw_model_pred = ml_predictor_titanic.predict(features)

    assert raw_model_pred == concord_pred


@raises(ValueError)
def test_predict_passing_in_missing_model_id_raises_error():

    features = df_titanic_test.iloc[0].to_dict()
    concord_pred = concord.predict(model_id=None, features=features)

    assert False

@raises(ValueError)
def test_predict_passing_in_bad_model_id_raises_error():

    features = df_titanic_test.iloc[0].to_dict()
    concord_pred = concord.predict(model_id='totally_made_up_and_bad_model_id', features=features)

    assert False


def test_predict_adds_features_to_db():
    features = df_titanic_test.iloc[1].to_dict()
    concord_pred = concord.predict(features=features, model_id=model_id)

    raw_model_pred = ml_predictor_titanic.predict(features)

    assert raw_model_pred == concord_pred

    saved_feature = concord.retrieve_from_persistent_db(val_type='live_features', row_id=features['name'], model_id=model_id)
    print('Did we remember to change the .iloc location to 2?')
    len_saved_feature = len(saved_feature)
    assert len_saved_feature == 1


def test_predict_multiple_times_with_the_same_features_adds_features_to_db_multiple_times():
    features = df_titanic_test.iloc[1].to_dict()
    concord_pred = concord.predict(features=features, model_id=model_id)

    raw_model_pred = ml_predictor_titanic.predict(features)

    assert raw_model_pred == concord_pred

    saved_features = concord.retrieve_from_persistent_db(val_type='live_features', row_id=features['name'], model_id=model_id)
    len_saved_features = len(saved_features)
    assert len_saved_features == 2


def test_predict_adds_prediction_to_db():
    saved_predictions = concord.retrieve_from_persistent_db(val_type='live_predictions')
    len_saved_predictions = len(saved_predictions)
    assert len_saved_predictions == 3



def test_single_predict_proba_matches_model_prediction():

    features = df_titanic_test.iloc[0].to_dict()
    concord_pred = concord.predict_proba(features=features, model_id=model_id)

    raw_model_pred = ml_predictor_titanic.predict_proba(features)

    assert raw_model_pred[0] == concord_pred[0]
    assert raw_model_pred[1] == concord_pred[1]



@raises(ValueError)
def test_predict_proba_passing_in_missing_model_id_raises_error():

    features = df_titanic_test.iloc[0].to_dict()
    concord_pred = concord.predict_proba(model_id=None, features=features)

    assert False

@raises(ValueError)
def test_predict_proba_passing_in_bad_model_id_raises_error():

    features = df_titanic_test.iloc[0].to_dict()
    concord_pred = concord.predict_proba(model_id='totally_made_up_and_bad_model_id', features=features)

    assert False


def test_predict_proba_adds_features_to_db():
    features = df_titanic_test.iloc[1].to_dict()
    concord_pred = concord.predict_proba(features=features, model_id=model_id)

    raw_model_pred = ml_predictor_titanic.predict_proba(features)

    assert raw_model_pred[0] == concord_pred[0]
    assert raw_model_pred[1] == concord_pred[1]

    saved_feature = concord.retrieve_from_persistent_db(val_type='live_features', row_id=features['name'], model_id=model_id)
    len_saved_feature = len(saved_feature)
    assert len_saved_feature == 3


def test_predict_proba_multiple_times_with_the_same_features_adds_features_to_db_multiple_times():
    features = df_titanic_test.iloc[1].to_dict()
    concord_pred = concord.predict_proba(features=features, model_id=model_id)

    raw_model_pred = ml_predictor_titanic.predict_proba(features)

    assert raw_model_pred[0] == concord_pred[0]
    assert raw_model_pred[1] == concord_pred[1]

    saved_features = concord.retrieve_from_persistent_db(val_type='live_features', row_id=features['name'], model_id=model_id)
    len_saved_features = len(saved_features)
    assert len_saved_features == 4


def test_predict_proba_adds_prediction_to_db():
    saved_predictions = concord.retrieve_from_persistent_db(val_type='live_predictions')
    len_saved_predictions = len(saved_predictions)
    assert len_saved_predictions == 6




def test_df_predict_matches_model_predictions():

    concord_pred = concord.predict(model_id=model_id, features=df_titanic_test)

    raw_model_pred = ml_predictor_titanic.predict(df_titanic_test)

    for idx, pred in enumerate(concord_pred):
        assert pred == concord_pred[idx]


def test_df_predict_proba_matches_model_predictions():

    concord_pred = concord.predict_proba(model_id=model_id, features=df_titanic_test)

    raw_model_pred = ml_predictor_titanic.predict_proba(df_titanic_test)

    for idx, pred in enumerate(concord_pred):
        concord_pred_row = concord_pred[idx]
        assert pred[0] == concord_pred_row[0]
        assert pred[1] == concord_pred_row[1]

def test_add_labels_takes_in_single_items_and_lists():

    row = df_titanic_test.iloc[0].to_dict()
    small_df = df_titanic_test.iloc[:10]

    concord.add_label(row_id=row['name'], model_id=model_id, label=row['survived'])
    result = concord.retrieve_from_persistent_db(val_type='live_labels', row_id=row['name'], model_id=model_id)
    assert len(result) == 1


    concord.add_label(row_id=small_df['name'], model_id=model_id, label=small_df['survived'])
    result = concord.retrieve_from_persistent_db(val_type='live_labels', row_id=None, model_id=model_id)
    assert len(result) == 11

    model_id_list = [model_id for x in range(small_df.shape[0])]
    concord.add_label(row_id=small_df['name'], model_id=model_id_list, label=small_df['survived'])
    result = concord.retrieve_from_persistent_db(val_type='live_labels', row_id=None, model_id=model_id)
    assert len(result) == 21


def test_list_all_models_returns_lots_of_info():
    results = concord.list_all_models()
    assert len(results) == 1

    assert 'model' not in results[0]

    expected_properties = ['model_id', 'feature_names', 'feature_importances', 'description', 'date_added']
    for prop in expected_properties:
        assert prop in results[0]

# def test_list_all_models_returns_useful_info():
#     model_descriptions = concord.list_all_models()

#     assert len(model_descriptions) == 1

#     assert isinstance(model_descriptions[0], dict)
#     expected_fields = ['namespace', 'val_type', 'train_or_serve', 'row_id', 'model', 'model_id', 'feature_names', 'feature_importances', 'description', 'date_added', 'num_predictions', 'last_prediction_time']
#     for field in expected_fields:
#         assert field in model_descriptions[0]


# def test_list_all_models_raises_warning_before_models_have_been_added_and_returns_empty_list():
#     with warnings.catch_warnings(record=True) as w:

#         model_descriptions = concord.list_all_models()
#         print('we should be throwing a warning for the user to give them useful feedback')
#         assert len(w) == 1
#     assert isinstance(model_descriptions, list)
#     assert len(model_descriptions) == 0
