import codecs
import datetime
import json
import warnings

import dill
import numpy as np
import pandas as pd
import pickle
from pymongo import MongoClient
import redis



# Implementation Thoughts
# We will have only a handful of collections
    # features
    # predictions
    # labels
# Then, each collection will use a compound index of the following fields
    # model_id
    # train_or_serve
    # row_id

class Concordia():

    def __init__(self, persistent_db_config=None, in_memory_db_config=None, namespace='_concordia', default_row_id_field=None):

        print('Welcome to Concordia! We\'ll do our best to take a couple stressors off your plate and give you more confidence in your machine learning systems in production.')
        # TODO: save everyting to a persistent DB
        # Then, when people want to load a concordia instance, all they have to do is provide the persistent_db_config, and namespace (and we will, of course, provide defaults for those, so if they're using default configs, it's all automatic)
        self.persistent_db_config = {
            'host': 'localhost'
            , 'port': 27017
            , 'db': '_concordia'
        }

        if persistent_db_config is not None:
            self.persistent_db_config.update(persistent_db_config)

        self.in_memory_db_config = {
            'host': 'localhost'
            , 'port': 6379
            , 'db': 0
        }

        if in_memory_db_config is not None:
            self.in_memory_db_config.update(in_memory_db_config)

        self._create_db_connections()

        self.namespace = namespace
        # Not sure what the idea was behind valid_prediction_types.
        self.valid_prediction_types = set([str, int, float, list, 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
        self.default_row_id_field = default_row_id_field

        params_to_save = {
            'persistent_db_config': self.persistent_db_config
            , 'in_memory_db_config': self.in_memory_db_config
            , 'namespace': self.namespace
            , 'default_row_id_field': self.default_row_id_field
            # , 'valid_prediction_types': self.valid_prediction_types
        }
        # TODO: save these params to a DB.

        self.insert_into_persistent_db(val=params_to_save, val_type='concordia_config', row_id='_intentionally_blank', model_id='_intentionally_blank')


    def set_params(self, params_dict):
        for k, v in params_dict.items():
            self[k] = v




    def _create_db_connections(self):
        host = self.in_memory_db_config['host']
        port = self.in_memory_db_config['port']
        db = self.in_memory_db_config['db']
        self.rdb = redis.StrictRedis(host=host, port=port, db=db)

        host = self.persistent_db_config['host']
        port = self.persistent_db_config['port']
        db = self.persistent_db_config['db']
        client = MongoClient(host=host, port=port)
        self.mdb = client[db]

        return self


    def add_model(self, model, model_id, row_id_field=None, feature_names=None, feature_importances=None, description=None):
        # FUTURE: allow the user to not use a row_id_field and just track live predictions. we will likely add in our own prediction_id field
        if row_id_field is None:
            raise(ValueError('row_id_field is required. It specifies which feature is going to be unique for each row. row_id_field enables us to compare features between training and serving environments.'))
        print('One thing to keep in mind is that each model_id must be unique in each db configuration. So if two Concordia instances are using the same database configurations, you should make sure their model_ids do not overlap.')
        # TODO: warn the user if that key exists already
        # maybe even take in errors='raise', but let the user pass in 'ignore' and 'warn' instead

        redis_key_model = self.make_redis_model_key(model_id)
        stringified_model = codecs.encode(dill.dumps(model), 'base64').decode()
        self.rdb.set(redis_key_model, stringified_model)

        # redis_key_model_info = '{}_{}_{}'.format(self.namespace, model_id, 'model_info')
        # TODO: get feature names automatically if possible

        mdb_doc = {
            'namespace': self.namespace
            , 'val_type': 'model_info'
            , 'model': stringified_model
            , 'model_id': model_id
            , 'feature_names': feature_names
            , 'feature_importances': feature_importances
            , 'description': description
            , 'date_added': datetime.datetime.now()
            # TODO: query these from our predictions table
            # , 'num_predictions': 0
            # , 'last_prediction_time': None
        }
        # self.mdb['_example_collection'].insert_one(mdb_doc)
        self.insert_into_persistent_db(mdb_doc, val_type=mdb_doc['val_type'], row_id=mdb_doc['model_id'], model_id=mdb_doc['model_id'])

        return self


    def list_all_models(self):
        pass


    # def insert_into_in_memory_db(self, val, val_type, row_id, model_id):
    #     pass


    def retrieve_from_persistent_db(self, val_type, row_id=None, model_id=None):
        query_params = {
            'row_id': row_id
            , 'model_id': model_id
        }
        if row_id is None:
            del query_params['row_id']
        if model_id is None:
            del query_params['model_id']

        result = self.mdb[val_type].find(query_params)

        # Handle the case where we have multiple predictions from the same row, or any other instances where we have multiple results for the same set of ids
        if isinstance(result, dict):
            result = [result]
        elif not isinstance(result, list):
            result = list(result)

        return result


    def check_row_id(self, val, row_id, idx=None):
        if isinstance(row_id, list):
            row_id = row_id[idx]
        if row_id is None:
            calculated_row_id = val.get(self.default_row_id_field, None)
            if calculated_row_id is None:
                print('You must pass in a row_id for anything that gets saved to the db.')
                print('This input is missing a value for "row_id"')
                if self.default_row_id_field is not None:
                    print('This input is also missing a value for "{}", the default_row_id_field'.format(self.default_row_id_field))
                raise(ValueError('Missing "row_id" field'))
            else:
                row_id = calculated_row_id

        assert row_id is not None
        val['row_id'] = row_id

        return val

    def check_model_id(self, val, model_id, idx=None):
        if isinstance(model_id, list):
            model_id = model_id[idx]
        if model_id is None:
            calculated_model_id = val.get('model_id', None)
            if calculated_model_id is None:
                print('You must pass in a model_id for anything that gets saved to the db.')
                print('This input is missing a value for "model_id"')
                raise(ValueError('Missing "model_id" field'))
            else:
                model_id = calculated_model_id

        assert model_id is not None
        val['model_id'] = model_id

        return val


    def _insert_df_into_db(self, df, val_type, row_id, model_id):

        df_cols = set(df.columns)
        if 'row_id' not in df_cols:
            if row_id is not None:
                df['row_id'] = row_id
            else:
                if self.default_row_id_field not in df_cols:
                    print('You must pass in a row_id for anything that gets saved to the db.')
                    print('This input is missing a value for "row_id"')
                    if self.default_row_id_field is not None:
                        print('This input is also missing a value for "{}", the default_row_id_field'.format(self.default_row_id_field))
                    raise(ValueError('Missing "row_id" field'))

        if 'model_id' not in df_cols:
            if model_id is not None:
                df['model_id'] = model_id
            else:
                print('You must pass in a model_id for anything that gets saved to the db.')
                print('This input is missing a value for "model_id"')
                raise(ValueError('Missing "model_id" field'))

        chunk_min_idx = 0
        chunk_size = 1000

        while chunk_min_idx < df.shape[0]:

            max_idx = min(df.shape[0], chunk_min_idx + chunk_size)
            df_chunk = df.iloc[chunk_min_idx: max_idx]

            df_chunk = df_chunk.to_dict('records')

            self.mdb[val_type].insert_many(df_chunk)

            del df_chunk
            chunk_min_idx += chunk_size


    def insert_into_persistent_db(self, val, val_type, row_id=None, model_id=None):
        val = val.copy()

        if isinstance(val, dict):
            val = self.check_row_id(val=val, row_id=row_id)
            val = self.check_model_id(val=val, model_id=model_id)

            for k, v in val.items():
                if isinstance(v, np.generic):
                    val[k] = np.asscalar(v)

            self.mdb[val_type].insert_one(val)


        else:
            self._insert_df_into_db(df=val, val_type=val_type, row_id=row_id, model_id=model_id)

        return self


    def make_redis_model_key(self, model_id):
        return '{}_{}_{}'.format(self.namespace, model_id, 'model')


    def _get_model(self, model_id):
        redis_key_model = self.make_redis_model_key(model_id)
        redis_result = self.rdb.get(redis_key_model)
        if redis_result is 'None' or redis_result is None:
            # Try to get it from MongoDB
            mdb_result = self.retrieve_from_persistent_db(val_type='model_info', row_id=None, model_id=model_id)
            if mdb_result is None or len(mdb_result) == 0:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('We could not find a corresponding model for model_id {}'.format(model_id))
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                error_string = 'We could not find a corresponding model for model_id {}'.format(model_id)
                raise(ValueError(error_string))
            else:
                model = mdb_result[0]['model']

                self.rdb.set(redis_key_model, model)
                redis_result = self.rdb.get(redis_key_model)


        redis_result = dill.loads(codecs.decode(redis_result, 'base64'))
        # try:
        # except AttributeError:
        #     print('redis_result')
        #     print(redis_result)
        #     print('type(redis_result)')
        #     print(type(redis_result))
        #     redis_result = dill.load(redis_result)

        return redis_result


    # This can handle both individual dictionaries and Pandas DataFrames as inputs
    def add_data_and_predictions(self, model_id, data, predictions, row_ids, actuals=None, model_type=None):

        data['row_id'] = row_ids
        data['model_id'] = model_id
        data['model_type'] = model_type

        if isinstance(data, pd.DataFrame):
            prediction_docs = []
            for idx, pred in enumerate(predictions):
                if type(pred) not in self.valid_prediction_types:
                    pred = list(pred)
                pred_doc = {
                    'prediction': pred
                    , 'row_id': row_ids.iloc[idx]
                    , 'model_id': model_id
                }
                prediction_docs.append(pred_doc)
            predictions_df = pd.DataFrame(prediction_docs)

            if actuals is not None:
                actuals_docs = []
                for idx, actual in enumerate(actuals):
                    actual_doc = {
                        'label': actual
                        , 'row_id': row_ids.iloc[idx]
                        , 'model_id': model_id
                    }
                    actuals_docs.append(actual_doc)
                actuals_df = pd.DataFrame(actuals_docs)


            self.insert_into_persistent_db(val=data, val_type='training_features')

            self.insert_into_persistent_db(val=predictions_df, val_type='training_predictions')

            if actuals is not None:
                self.insert_into_persistent_db(val=actuals_df, val_type='training_labels')

        elif isinstance(data, dict):
            self.insert_into_persistent_db(val=data, val_type='training_features', row_id=row_id, model_id=model_id)
            self.insert_into_persistent_db(val=predictions, val_type='training_predictions', row_id=row_id, model_id=model_id)
            if actuals is not None:
                self.insert_into_persistent_db(val=actuals, val_type='training_labels', row_id=row_id, model_id=model_id)

        return self





    # FUTURE: add in model_type, which will just get the most recent model_id for that model_type
    # NOTE: we will return whatever the base model returns. We will not modify the output of that model at all (so if the model is an auto_ml model that returns a single float for a single item prediction, that's what we return. if it's a sklearn model that returns a list with a single float in it, that's what we return)
    # NOTE: it is explicitly OK to call predict multiple times with the same data. If you want to filter out duplicate rows, you may do that with "drop_duplicates=True" at analytics time
    def predict(self, features, model_id, row_id=None, shadow_models=None):
        return self._predict(features=features, model_id=model_id, row_id=row_id, shadow_models=shadow_models, proba=False)


    def predict_proba(self, features, model_id, row_id=None, shadow_models=None):
        return self._predict(features=features, model_id=model_id, row_id=row_id, shadow_models=shadow_models, proba=True)


    def predict_all(self, data):
        pass


    # TODO: right now I think this all assumes single item predictions, not batch predictions on many rows
    def _predict(self, features=None, model_id=None, row_id=None, model_ids=None, shadow_models=None, proba=False):
        if row_id is None and self.default_row_id_field is None:
            raise(ValueError('Missing row_id. Please pass in a value for "model_id", or set a "default_row_id_field" on this Concordia instance'))

        model = self._get_model(model_id=model_id)

        if row_id is None:
            row_id = features[self.default_row_id_field]

        # FUTURE: input verification here before we get predictions.
        self.insert_into_persistent_db(val=features, val_type='live_features', row_id=row_id, model_id=model_id)

        if proba == True:
            prediction = model.predict_proba(features)
        else:
            prediction = model.predict(features)

        # Mongo doesn't handle np.ndarrays. it prefers lists.
        pred_for_saving = prediction
        if isinstance(pred_for_saving, np.ndarray):
            pred_for_saving = list(pred_for_saving)
            clean_pred_for_saving = []
            for item in pred_for_saving:
                if isinstance(item, np.ndarray):
                    item = list(item)
                clean_pred_for_saving.append(item)
            pred_for_saving = clean_pred_for_saving

        pred_doc = {
            'prediction': pred_for_saving
            , 'row_id': row_id
            , 'model_id': model_id
        }
        if isinstance(features, pd.DataFrame):
            pred_doc = pd.DataFrame(pred_doc)
        self.insert_into_persistent_db(val=pred_doc, val_type='live_predictions', row_id=row_id, model_id=model_id)

        return prediction

    def remove_model(model_ids):
        pass


    def load_from_db(self, query_params, start_time=None, end_time=None, num_results=None):
        if start_time is not None:
            query_params['']

        # results = mdb.
        pass



    def _get_training_data_and_predictions(self, model_id, row_id=None):
        training_features = self.retrieve_from_persistent_db(val_type='training_features', row_id=row_id, model_id=model_id)
        training_features = pd.DataFrame(training_features)

        training_predictions = self.retrieve_from_persistent_db(val_type='training_predictions', row_id=row_id, model_id=model_id)
        training_predictions = pd.DataFrame(training_predictions)

        training_labels = self.retrieve_from_persistent_db(val_type='training_labels', row_id=row_id, model_id=model_id)
        training_labels = pd.DataFrame(training_labels)

        return training_features, training_predictions, training_labels






    # def delete_data(self, model_id, row_ids):
    #     pass







    # def add_outcome_values(self, model_ids, row_ids, y_labels):
    #     pass


    # def reconcile_predictions(self):
    #     pass


    # def reconcile_features(self):
    #     pass


    # def reconcile_labels(self):
    #     pass


    # def reconcile_all(self):
    #     pass


    # def track_features_over_time(self):
    #     pass


    # def track_missing_features(self):
    #     pass


    # def get_values(self, model_id, val_type):
    #     # val_type is in ['training_features', 'serving_features', 'training_predictions', 'serving_predictions', 'training_labels', 'serving_labels']
    #     pass






    # # These are explicitly out of scopre for our initial implementation
    # def custom_predict(self):
    #     pass


    # def custom_db_insert(self):
    #     pass


    # # Lay out what the API must be
    #     #
    # def custom_db_retrieve(self):
    #     pass


    # def save_from_redis_to_mongo(self):
    #     pass



def load_concordia(persistent_db_config=None, namespace='_concordia'):
    default_db_config = {
        'host': 'localhost'
        , 'port': 27017
        , 'db': '_concordia'
    }

    default_db_config.update(persistent_db_config)

    print('default_db_config')
    print(default_db_config)

    # FUTURE: allow the user to pass in a custom query/db connection, replicating what we do when they do a custom replace of retrieve_from_persistent_db
    client = MongoClient(host=default_db_config['host'], port=default_db_config['port'])
    mdb = client[default_db_config['db']]
    concordia_info = mdb['concordia_config'].find_one({})

    if 'model_id' in concordia_info:
        del concordia_info['model_id']
    if '_id' in concordia_info:
        del concordia_info['_id']
    if 'row_id' in concordia_info:
        del concordia_info['row_id']

    concord = Concordia(**concordia_info)

    # make db_connection optional
    # Load up the data from the db using the db_connection info
    # then, self._create_db_connections()
    # then, i think we're set.
    # we just need to make sure we consistently save all the info we need to the db.
    return concord
