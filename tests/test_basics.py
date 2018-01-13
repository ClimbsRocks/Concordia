import os
import sys

import dill
from nose.tools import raises
from pymongo import MongoClient

print('Starting now')

sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
sys.path = [os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')] + sys.path
os.environ['is_test_suite'] = 'True'
print('sys.path')
print(sys.path)

import redis

from concordia import Concordia

def do_setup():
    import aml_utils


    ####################################################################
    # Setup- train model, create direct db connections, set global constants, etc.
    #####################################################################
    # TODO: create another model that uses a different algo (logisticRegression, perhaps), so we can have tests for our logic when using multiple models but each predicting off the same features
    ml_predictor_titanic, df_titanic_test = aml_utils.train_basic_binary_classifier()
    row_ids = [i for i in range(df_titanic_test.shape[0])]
    df_titanic_test['row_id'] = row_ids

    namespace = '__test_env'

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

    concord = Concordia(in_memory_db_config=in_memory_db_config, persistent_db_config=persistent_db_config, namespace=namespace, default_row_id_field='name')

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


    return ml_predictor_titanic, df_titanic_test, namespace, concord, rdb, mdb

model_id = 'ml_predictor_titanic_1'

ml_predictor_titanic, df_titanic_test, namespace, concord, rdb, mdb = do_setup()


def test_list_all_models_raises_warning_before_models_have_been_added_and_returns_empty_list():
    with warnings.catch_warnings(record=True) as w:

        model_descriptions = concord.list_all_models()
        print('we should be throwing a warning for the user to give them useful feedback')
        assert len(w) == 1
    assert isinstance(model_descriptions, list)
    assert len(model_descriptions) == 0



@raises(ValueError)
def test_add_new_model_requires_row_id_field():

    redis_key_model = concord.make_redis_model_key(model_id)
    starting_val = rdb.get(redis_key_model)

    assert starting_val is None


    concord.add_model(model=ml_predictor_titanic, model_id=model_id)
    assert False


def test_add_new_model():
    print('running test now')

    redis_key_model = concord.make_redis_model_key(model_id)
    starting_val = rdb.get(redis_key_model)

    assert starting_val is None


    concord.add_model(model=ml_predictor_titanic, model_id=model_id, row_id_field='name')

    post_insert_val = rdb.get(redis_key_model)
    assert post_insert_val is not None



def test_get_model():
    model = concord.get_model(model_id)
    assert type(model) == type(ml_predictor_titanic)


def test_get_model_after_deleting_from_redis():
    rdb.delete(concord.make_redis_model_key(model_id))
    model = concord.get_model(model_id)
    assert type(model) == type(ml_predictor_titanic)


def test_insert_training_features_and_preds():
    test_preds = ml_predictor_titanic.predict_proba(df_titanic_test)
    concord.add_data_and_predictions(model_id=model_id, data=df_titanic_test, predictions=test_preds, row_ids=df_titanic_test['name'], actuals=df_titanic_test['survived'])

    assert True

    concord_data = concord._get_training_data_and_predictions(model_id)

    concord_ids = set(concord_data.row_id)
    for row_id in df_titanic_test['row_id']:
        assert row_id in concord_ids



def test_insert_training_features_and_preds():
    test_preds = ml_predictor_titanic.predict_proba(df_titanic_test)
    concord.add_data_and_predictions(model_id=model_id, data=df_titanic_test, predictions=test_preds, row_ids=df_titanic_test['name'], actuals=df_titanic_test['survived'])

    assert True

    # TODO: we might need to iterate through the generator and append to a list
    saved_feature_data = list(mdb.features.find({'train_or_serve': 'train', 'model_id': model_id}))
    df_saved = pd.DataFrame(saved_feature_data)

    concord_ids = set(df_saved.name)
    for name in df_titanic_test['name']:
        assert name in concord_ids

    # We are assuming our preds are all saved with the row_id
    saved_preds = list(mdb.predictions.find({'train_or_serve': 'train', 'model_id': model_id}))
    df_preds = pd.DataFrame(saved_preds)
    for idx, name in enumerate(df_titanic_test.name):
        concord_row = df_preds[df_preds.name == name].to_dict()
        assert concord_row['prediction'] == test_preds[idx]








# Starting here, these have not been duplicated for proba yet
def test_single_predict_matches_model_prediction():

    features = df_titanic_test.iloc[0].to_dict()
    print('features')
    print(features)
    concord_pred = concord.predict(features=features, model_id=model_id)

    raw_model_pred = ml_predictor_titanic.predict(features)

    assert raw_model_pred == concord_pred


def test_df_predict_matches_model_predictions():

    features = df_titanic_test.iloc[0].to_dict()
    concord_pred = concord.predict(model_id=model_id, features=features)

    raw_model_pred = ml_predictor_titanic.predict(df_titanic_test)

    assert raw_model_pred == concord_pred


def test_predict_takes_in_model_id_or_model_ids_interchangeably():

    features = df_titanic_test.iloc[0].to_dict()
    concord_pred1 = concord.predict(model_id=model_id, features=features)
    concord_pred2 = concord.predict(model_id=[model_id], features=features)

    raw_model_pred = ml_predictor_titanic.predict(df_titanic_test)

    assert raw_model_pred == concord_pred1
    assert raw_model_pred == concord_pred2


def test_predict_model_id_can_be_list_or_single_string():

    features = df_titanic_test.iloc[0].to_dict()
    concord_pred1 = concord.predict(model_id=model_id, features=features)
    concord_pred2 = concord.predict(model_id=[model_id], features=features)
    concord_pred3 = concord.predict(model_ids=[model_id], features=features)
    concord_pred4 = concord.predict(model_ids=model_id, features=features)

    raw_model_pred = ml_predictor_titanic.predict(df_titanic_test)

    assert raw_model_pred == concord_pred1
    assert raw_model_pred == concord_pred2
    assert raw_model_pred == concord_pred3
    assert raw_model_pred == concord_pred4


@raises(ValueError)
def test_predict_passing_in_model_id_and_model_ids_raises_error():

    features = df_titanic_test.iloc[0].to_dict()
    concord_pred = concord.predict(model_id=model_id, model_ids=model_id, features=features)

    raw_model_pred = ml_predictor_titanic.predict(df_titanic_test)

    assert False


@raises(ValueError)
def test_predict_passing_in_neither_model_id_nor_model_ids_raises_error():

    features = df_titanic_test.iloc[0].to_dict()
    concord_pred = concord.predict(model_id=None, model_ids=None, features=features)

    raw_model_pred = ml_predictor_titanic.predict(df_titanic_test)

    assert False


def test_predict_adds_features_to_db():
    # TODO: make this a different idx location when we duplicate for proba
    features = df_titanic_test.iloc[1].to_dict()
    print('features')
    print(features)
    concord_pred = concord.predict(features=features, model_id=model_id)

    raw_model_pred = ml_predictor_titanic.predict(features)

    assert raw_model_pred == concord_pred

    saved_feature = list(mdb.features.find({'name': features['name'], 'train_or_serve': 'train', 'model_id': model_id}))
    print('Did we remember to change the .iloc location to 2?')
    assert len(saved_feature) == 1


def test_predict_multiple_times_with_the_same_features_adds_features_to_db_multiple_times():
    # TODO: make this a different idx location when we duplicate for proba
    features = df_titanic_test.iloc[1].to_dict()
    print('features')
    print(features)
    concord_pred = concord.predict(features=features, model_id=model_id)

    raw_model_pred = ml_predictor_titanic.predict(features)

    assert raw_model_pred == concord_pred

    saved_feature = list(mdb.features.find({'name': features['name'], 'train_or_serve': 'train', 'model_id': model_id}))
    print('Did we remember to change the .iloc location to 2?')
    assert len(saved_feature) == 2


def test_predict_multiple_times_with_the_same_features_adds_features_to_db_multiple_times():
    # TODO: make this a different idx location when we duplicate for proba
    features = df_titanic_test.iloc[1].to_dict()
    print('features')
    print(features)
    concord_pred = concord.predict(features=features, model_id=model_id)

    raw_model_pred = ml_predictor_titanic.predict(features)

    assert raw_model_pred == concord_pred

    saved_feature = list(mdb.features.find({'name': features['name'], 'train_or_serve': 'train', 'model_id': model_id}))
    print('Did we remember to change the .iloc location to 2?')
    assert len(saved_feature) == 2


def test_predict_adds_prediction_to_db():
    saved_predictions = list(mdb.predictions.find({'model_id': model_id}))
    assert len(saved_predictions) > 4
## End section to duplicate for proba
# TODO: Duplicate all the predic tests
# Duplicating the above  tests for predict_proba






















def test_list_all_models_returns_useful_info():
    model_descriptions = concord.list_all_models()

    assert len(model_descriptions) == 1

    assert isinstance(model_descriptions[0], dict)
    expected_fields = ['namespace', 'val_type', 'train_or_serve', 'row_id', 'model', 'model_id', 'feature_names', 'feature_importances', 'description', 'date_added', 'num_predictions', 'last_prediction_time']
    for field in expected_fields:
        assert field in model_descriptions[0]



















def test_preston_does_not_get_overly_ambitious_in_mvp_scoping():
    model_descriptions = concord.list_all_models()

    assert model_descriptions[0]['last_prediction_time'] is None
    assert model_descriptions[0]['num_predictions'] == 0








# if __name__ == '__main__':
#     do_setup()
#     test_add_new_model()
#     test_get_model()
#     test_get_model_after_deleting_from_redis()
#     test_insert_training_features_and_preds()

