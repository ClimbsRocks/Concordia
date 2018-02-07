import codecs
import datetime
import json
import numbers
import warnings

import dill
import numpy as np
import pandas as pd
import pickle
from pymongo import MongoClient
import redis
from tabulate import tabulate


class Concordia():

    def __init__(self, persistent_db_config=None, in_memory_db_config=None, default_row_id_field=None):

        print('Welcome to Concordia! We\'ll do our best to take a couple stressors off your plate and give you more confidence in your machine learning systems in production.')
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

        self.valid_prediction_types = set([str, int, float, list, 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
        self.default_row_id_field = default_row_id_field

        params_to_save = {
            'persistent_db_config': self.persistent_db_config
            , 'in_memory_db_config': self.in_memory_db_config
            , 'default_row_id_field': self.default_row_id_field
        }

        self.insert_into_persistent_db(val=params_to_save, val_type='concordia_config', row_id='_intentionally_blank', model_id='_intentionally_blank')


    def set_params(self, params_dict):
        for k, v in params_dict.items():
            setattr(self, k, v)




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


    # feature_importances is a dict, with keys as feature names, and values being the importance of each feature. it doesn't matter how the imoprtances are calculated, we'll just sort by those values
    def add_model(self, model, model_id, feature_names=None, feature_importances=None, description=None, features_to_save='all'):
        print('One thing to keep in mind is that each model_id must be unique in each db configuration. So if two Concordia instances are using the same database configurations, you should make sure their model_ids do not overlap.')

        redis_key_model = self.make_redis_model_key(model_id)
        stringified_model = codecs.encode(dill.dumps(model), 'base64').decode()
        self.rdb.set(redis_key_model, stringified_model)

        redis_key_features = self.make_redis_key_features(model_id)
        stringified_features = json.dumps(features_to_save)
        self.rdb.set(redis_key_features, stringified_features)

        if feature_importances is not None:
            if not isinstance(feature_importances, dict):
                raise(TypeError('feature_importances must be a dict, where each key is a feature name, and each value is the importance of that feature'))
            for k, v in feature_importances.items():
                if isinstance(v, np.generic):
                    feature_importances[k] = np.asscalar(v)


        mdb_doc = {
            'val_type': 'model_info'
            , 'model': stringified_model
            , 'model_id': model_id
            , 'feature_names': feature_names
            , 'feature_importances': json.dumps(feature_importances)
            , 'description': description
            , 'date_added': datetime.datetime.now()
            , 'features_to_save': stringified_features
        }

        self.insert_into_persistent_db(mdb_doc, val_type=mdb_doc['val_type'], row_id=mdb_doc['model_id'], model_id=mdb_doc['model_id'])

        return self

    def add_label(self, row_id, model_id, label):
        label_doc = {
            'row_id': row_id
            , 'model_id': model_id
            , 'label': label
        }

        if not isinstance(row_id, numbers.Number) and not isinstance(row_id, np.generic) and not isinstance(row_id, str):
            if isinstance(model_id, str):
                label_doc['model_id'] = [model_id for x in range(len(row_id))]
            label_doc = pd.DataFrame(label_doc)

        self.insert_into_persistent_db(val=label_doc, val_type='live_labels', row_id=label_doc['row_id'], model_id=label_doc['model_id'])


    def list_all_models(self, verbose=True):
        live_models = self.retrieve_from_persistent_db(val_type='model_info')
        if verbose:
            print('Here are all the models that have been added to concordia for live predictions:')
            model_names = [x['model_id'] for x in live_models]
            print(model_names)
        for model_info in live_models:
            del model_info['model']
        return live_models


    def retrieve_from_persistent_db(self, val_type, row_id=None, model_id=None, min_date=None, date_field=None):
        if min_date is not None and date_field is None and not (isinstance(min_date, datetime.datetime) or isinstance(min_date, datetime.date)):
            print('You have specified a min_date, but not a date_field')
            print('Without the date_field specified, Concordia will query against the "_concordia_created_at" field, which is of type datetime.datetime.')
            print('Therefore, your min_date must be of type datetime.datetime, but it is not right now. It is of type: '.format(type(min_date)))
            raise(TypeError('min_date must be of type datetime if date_field is unspecified'))

        query_params = {
            'row_id': row_id
            , 'model_id': model_id
        }
        if row_id is None:
            del query_params['row_id']
        if model_id is None:
            del query_params['model_id']

        if min_date is not None:
            if date_field is None:
                query_params['_concordia_created_at'] = {'$gte': min_date}
            else:
                query_params[date_field] = {'$gte': min_date}

        result = self.mdb[val_type].find(query_params)

        # Handle the case where we have multiple predictions from the same row, or any other instances where we have multiple results for the same set of ids
        if isinstance(result, dict):
            result = [result]
        elif not isinstance(result, list):
            result = list(result)

        return result


    def check_row_id(self, val, row_id, idx=None):

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
        if '_id' in val:
            del val['_id']
        if '_id_' in val:
            del val['_id_']
        val['_concordia_created_at'] = datetime.datetime.utcnow()

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
        return '_concordia_{}_{}'.format(model_id, 'model')


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

        return redis_result


    def _get_features_to_save(self, model_id):
        redis_key = self.make_redis_key_features(model_id)
        redis_result = self.rdb.get(redis_key)

        if redis_result is None or redis_result is 'None':
            mdb_result = self.retrieve_from_persistent_db(val_type='model_info', row_id=None, model_id=model_id)
            if mdb_result is None or len(mdb_result) == 0:
                return 'all'
            else:
                try:
                    features = mdb_result[0]['features_to_save']
                except KeyError:
                    features = json.dumps('all')
                self.rdb.set(redis_key, features)
                redis_result = self.rdb.get(redis_key)

        if isinstance(redis_result, bytes):
            redis_result = redis_result.decode('utf-8')
        redis_result = json.loads(redis_result)
        return redis_result


    def make_redis_key_features(self, model_id):
        return '_concordia_{}_{}'.format(model_id, 'features_to_save')


    # This can handle both individual dictionaries and Pandas DataFrames as inputs
    def add_data_and_predictions(self, model_id, features, predictions, row_ids, actuals=None):
        if not isinstance(features, pd.DataFrame):
            print('Training features must be a pandas DataFrame, not a {}'.format(type(features)))
            raise(TypeError('Training features must be a pandas DataFrame'))

        features = features.copy()

        features['row_id'] = row_ids
        features['model_id'] = model_id
        features_to_save = self._get_features_to_save(model_id=model_id)
        concordia_features_to_save = ['row_id', 'model_id']

        if features_to_save == 'all':
            features_to_save = list(features.columns)
        else:
            features_to_save = features_to_save + concordia_features_to_save
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

        saving_features = features[features_to_save]
        self.insert_into_persistent_db(val=saving_features, val_type='training_features')

        self.insert_into_persistent_db(val=predictions_df, val_type='training_predictions')

        if actuals is not None:
            self.insert_into_persistent_db(val=actuals_df, val_type='training_labels')

        #     if features_to_save == 'all':
        #         features_to_save = features.keys()
        #     else:
        #         features_to_save = features_to_save + concordia_features_to_save
        #     saving_features = {}
        #     for k, v in features.items():
        #         if k in features_to_save:
        #             saving_features[k] = v
        #     self.insert_into_persistent_db(val=saving_features, val_type='training_features', row_id=row_id, model_id=model_id)
        #     self.insert_into_persistent_db(val=predictions, val_type='training_predictions', row_id=row_id, model_id=model_id)
        #     if actuals is not None:
        #         self.insert_into_persistent_db(val=actuals, val_type='training_labels', row_id=row_id, model_id=model_id)

        return self





    # FUTURE: add in model_type, which will just get the most recent model_id for that model_type
    # NOTE: we will return whatever the base model returns. We will not modify the output of that model at all (so if the model is an auto_ml model that returns a single float for a single item prediction, that's what we return. if it's a sklearn model that returns a list with a single float in it, that's what we return)
    # NOTE: it is explicitly OK to call predict multiple times with the same data. If you want to filter out duplicate rows, you may do that with "drop_duplicates=True" at analytics time
    def predict(self, model_id, features, row_id=None, shadow_models=None):
        return self._predict(features=features, model_id=model_id, row_id=row_id, shadow_models=shadow_models, proba=False)


    def predict_proba(self, model_id, features, row_id=None, shadow_models=None):
        return self._predict(features=features, model_id=model_id, row_id=row_id, shadow_models=shadow_models, proba=True)


    def predict_all(self, data):
        pass


    def _predict(self, features=None, model_id=None, row_id=None, model_ids=None, shadow_models=None, proba=False):
        features = features.copy()

        model = self._get_model(model_id=model_id)

        if row_id is None:
            row_id = features[self.default_row_id_field]

        features_to_save = self._get_features_to_save(model_id=model_id)
        if features_to_save == 'all':
            saving_features = features
        else:
            saving_features = features[features_to_save]
        # FUTURE: input verification here before we get predictions.
        self.insert_into_persistent_db(val=saving_features, val_type='live_features', row_id=row_id, model_id=model_id)


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

    # def remove_model(self, model_id, verbose=True):
    #     if verbose == True:
    #         print('Removing model {}'.format(model_id))
    #         print('Note that this will remove the model from being able to make predictions.')
    #         print('We will keep historical data associated with this model (features, predictions, labels, etc.) so you can continue to perform analysis on it.')
    #     # TODO: remove the model from our model_info table.
    #     # This is the only place we are deleting something from the db, so we might need to create a new helper function (delete_from_db) for it.
    #     pass


    def match_training_and_live(self, df_train, df_live, row_id_field=None):
        # The important part here is our live predictions
        # So we'll left join the two, keeping all of our live rows

        # TODO: leverage the per-model row_id_field we will build out soon
        # TODO: some accounting for rows that don't match
        cols_to_drop = ['_id', '_concordia_created_at']
        for col in cols_to_drop:
            try:
                del df_train[col]
            except:
                pass
            try:
                del df_live[col]
            except:
                pass

        df = pd.merge(df_live, df_train, on='row_id', how='inner', suffixes=('_live', '_train'))

        if df.shape[0] == 0 and df_train.shape[0] > 0 and df_live.shape[0] > 0:
            print('\nWe have saved data for both training and live environments, but were not able to match them together on shared row_id values. Here is some information about the row_id column to help you debug.')
            print('\nTraining row_id.head')
            print(df_train.row_id.head())
            print('\nLive row_id.head')
            print(df_live.row_id.head())
            print('\nTraining row_id described:')
            print(df_train.row_id.describe())
            print('\nLive row_id described:')
            print(df_live.row_id.describe())

            warnings.warn('While we have saved data for this model_id for both live and training environments, we were not able to match them on the same row_id.')
        return df


    def compare_one_row_predictions(self, row):
        train_pred = row.prediction_train
        live_pred = row.prediction_live

        count_lists = 0
        if isinstance(train_pred, list) or isinstance(train_pred, pd.Series):
            count_lists += 1
        if isinstance(live_pred, list) or isinstance(live_pred, pd.Series):
            count_lists += 1
        if count_lists == 1:
            print('It appears you are comparing predictions of different types (only one of them is a lsit). This might be from comparing predictions where one was a probability prediction, and one was not. We have not yet built out that functionality. Please make sure all predictions are consistent types.')
            raise(TypeError('Predictions are of different types. Only one of the predictions is a list'))

        if count_lists == 2:
            return_val = {}
            for idx, train_proba_pred in enumerate(train_pred):
                live_proba_pred = live_pred[idx]
                return_val['class_{}_delta'.format(idx)] = train_proba_pred - live_proba_pred

        else:
            delta = train_pred - live_pred
            return_val = {'delta': delta}

        return pd.Series(return_val)






    def analyze_prediction_discrepancies(self, model_id, return_summary=True, return_deltas=True, return_matched_rows=False, sort_column=None, min_date=None, date_field=None, verbose=True, ignore_nans=True, ignore_duplicates=True):
        # TODO 1: add input checking for min_date must be a datetime if date_field is none
        # TODO 2: add logging if we have values for both training and live, but no matches when merging

        # 1. Get live data (only after min_date)
        live_predictions = self.retrieve_from_persistent_db(val_type='live_predictions', row_id=None, model_id=model_id, min_date=min_date, date_field=date_field)
        # 2. Get training_data (only after min_date- we are only supporting the use case of training data being added after live data)
        training_predictions = self.retrieve_from_persistent_db(val_type='training_predictions', row_id=None, model_id=model_id, min_date=min_date, date_field=date_field)

        live_predictions = pd.DataFrame(live_predictions)
        training_predictions = pd.DataFrame(training_predictions)


        if ignore_nans == True:
            if verbose:
                print('Ignoring nans')
            live_predictions = live_predictions[pd.notnull(live_predictions.prediction)]
            training_predictions = training_predictions[pd.notnull(training_predictions.prediction)]

        if ignore_duplicates == True:
            if verbose:
                print('Ignoring duplicates')
            live_predictions.drop_duplicates(subset='row_id', inplace=True)
            training_predictions.drop_duplicates(subset='row_id', inplace=True)

        if verbose == True:
            print('Found {} relevant live predictions'.format(live_predictions.shape[0]))
            print('Found a max of {} possibly relevant train predictions'.format(training_predictions.shape[0]))
        # 3. match them up (and provide a reconciliation of what rows do not match)


        df_live_and_train = self.match_training_and_live(df_live=live_predictions, df_train=training_predictions)

        print('Found {} rows that appeared in both our training and live datasets'.format(df_live_and_train.shape[0]))
        # All of the above should be done using helper functions
        # 4. Go through and analyze all feature discrepancies!
            # Ideally, we'll have an "impact_on_predictions" column, though maybe only for our top 10 or top 100 features
        deltas = df_live_and_train.apply(self.compare_one_row_predictions, axis=1)


        summary = self.summarize_prediction_deltas(df_deltas=deltas)

        return_val = self.create_analytics_return_val(summary=summary, deltas=deltas, matched_rows=df_live_and_train, return_summary=return_summary, return_deltas=return_deltas, return_matched_rows=return_matched_rows, verbose=verbose)
        return return_val


    def create_analytics_return_val(self, summary, deltas, matched_rows, return_summary=True, return_deltas=True, return_matched_rows=False, verbose=True):

        return_val = {}
        if return_summary == True:
            return_val['summary'] = summary
        if return_deltas == True:
            return_val['deltas'] = deltas
        if return_matched_rows == True:
            return_val['matched_rows'] = matched_rows

        if verbose:
            print('\n\n******************')
            print('Deltas:')
            print('******************\n')
            # What we want to do here is have each row be a metric, with two columns
            # The metric name, and the metric value
            sorted_keys = sorted(summary.keys())
            printing_val = []
            for key in sorted_keys:
                printing_val.append((key, summary[key]))
            print(tabulate(printing_val, headers=['Metric', 'Value'], floatfmt='.3f', tablefmt='psql'))

        return return_val

    def find_missing_columns(self, df):
        columns = set(df.columns)
        results = {
            'train_columns_not_in_live': []
            , 'live_columns_not_in_train': []
            , 'matched_cols': []
        }

        for col in df.columns:
            if col[-6:] == '_train':

                live_col = col[:-6] + '_live'
                if live_col not in columns:
                    results['train_columns_not_in_live'].append(col[:-6])
                else:
                    results['matched_cols'].append(col[:-6])
            elif col[-5:] == '_live':
                train_col = col[:-5] + '_train'
                if train_col not in columns:
                    results['live_columns_not_in_train'].append(col[:-5])

        return results


    def summarize_one_delta_col(self, deltas, prefix):

        results = {}

        percentiles = [5, 25, 50, 75, 95, 99]

        results['{}_num_rows_with_deltas'.format(prefix)] = len([x for x in deltas if x != 0])
        results['{}_num_rows_with_no_deltas'.format(prefix)] = len([x for x in deltas if x == 0])
        results['{}_avg_delta'.format(prefix)] = np.mean(deltas)
        results['{}_median_delta'.format(prefix)] = np.median(deltas)
        for percentile in percentiles:
            results['{}_{}th_percentile_delta'.format(prefix, percentile)] = np.percentile(deltas, percentile)

        abs_deltas = np.abs(deltas)
        results['{}_avg_abs_delta'.format(prefix)] = np.mean(abs_deltas)
        results['{}_median_abs_delta'.format(prefix)] = np.median(abs_deltas)
        for percentile in percentiles:
            results['{}_{}th_percentile_abs_delta'.format(prefix, percentile)] = np.percentile(abs_deltas, percentile)

        return results


    def summarize_prediction_deltas(self, df_deltas):
        if 'delta' in df_deltas.columns:
            result = self.summarize_one_delta_col(df_deltas.delta, prefix='prediction')

        else:
            result = {}
            for col in df_deltas.columns:
                if col[-6:] == '_delta':
                    result.update(self.summarize_one_delta_col(df_deltas[col], col[:-6]))

        return result



    def summarize_feature_deltas(self, df_deltas, feature_importances):
        col_results = []

        for col in df_deltas.columns:
            # TODO: figure out if a column is categorical. if it is, handle deltas differently (probably just count of vals that are different)
            col_result = self.summarize_one_delta_col(deltas=df_deltas[col], prefix=col)
            col_result['feature'] = col
            if feature_importances is not None:
                importance = feature_importances.get(col, 0)
                col_result['feature_importance'] = importance
            col_results.append(col_result)

        return col_results


    def analyze_feature_discrepancies(self, model_id, return_summary=True, return_deltas=True, return_matched_rows=False, sort_column=None, min_date=None, date_field=None, verbose=True, ignore_duplicates=True):

        # 1. Get live data (only after min_date)
        live_features = self.retrieve_from_persistent_db(val_type='live_features', row_id=None, model_id=model_id, min_date=min_date, date_field=date_field)
        # 2. Get training_data (only after min_date- we are only supporting the use case of training data being added after live data)
        training_features = self.retrieve_from_persistent_db(val_type='training_features', row_id=None, model_id=model_id, min_date=min_date, date_field=date_field)

        live_features = pd.DataFrame(live_features)
        training_features = pd.DataFrame(training_features)

        if ignore_duplicates == True:
            if len(set(live_features['row_id'])) < live_features.shape[0]:
                live_features.drop_duplicates(subset='row_id', inplace=True)
            if len(set(training_features['row_id'])) < training_features.shape[0]:
                # Keep the most recently added features
                training_features.sort_values(by='_concordia_created_at', ascending=True, inplace=True)
                training_features.drop_duplicates(subset='row_id', inplace=True, keep='last')

        # 3. match them up (and provide a reconciliation of what rows do not match)
        df_live_and_train = self.match_training_and_live(df_live=live_features, df_train=training_features)
        # All of the above should be done using helper functions
        # 4. Go through and analyze all feature discrepancies!
            # Ideally, we'll have an "impact_on_predictions" column, though maybe only for our top 10 or top 100 features
        # TODO: find columns missing from only one (train or live)
        # TODO: find rows missing frm only one (train or live)
        column_comparison = self.find_missing_columns(df_live_and_train)
        matched_cols = column_comparison['matched_cols']

        deltas = df_live_and_train.apply(lambda row: self.compare_one_row_features(row=row, features_to_compare=matched_cols), axis=1)

        model_info = self.retrieve_from_persistent_db(val_type='model_info', model_id=model_id)
        feature_importances = json.loads(model_info[0]['feature_importances'])

        summary_list = self.summarize_feature_deltas(df_deltas=deltas, feature_importances=feature_importances)

        if feature_importances is not None:
            printing_list = sorted(summary_list, key=lambda val: val['feature_importance'], reverse=True)
            printing_list = [val for val in printing_list if val['feature_importance'] > 0]
        else:
            printing_list = summary_list

        if verbose == True:
            printing_tuples = []
            for feature in printing_list:
                tup = []
                name = feature['feature']

                tup.append(name)
                try:
                    tup.append(feature['feature_importance'])
                except KeyError:
                    tup.append('n/a')

                tup.append(feature['{}_num_rows_with_deltas'.format(name)])
                tup.append(feature['{}_avg_delta'.format(name)])
                tup.append(feature['{}_avg_abs_delta'.format(name)])
                tup.append(feature['{}_95th_percentile_abs_delta'.format(name)])
                try:
                    feature_range = training_features[name].max() - training_features[name].min()
                    tup.append(feature['{}_avg_abs_delta'.format(name)] / feature_range)
                except TypeError:
                    tup.append('n/a')
                printing_tuples.append(tup)

            print(tabulate(printing_tuples, headers=['Feature', 'Importance', 'Num rows with deltas', 'Avg delta', 'Avg abs delta', '95th pct avg abs delta', 'Avg abs delta / feature range'], floatfmt='.3f', tablefmt='psql'))

        return_val = self.create_analytics_return_val(summary=summary_list, deltas=deltas, matched_rows=df_live_and_train, return_summary=return_summary, return_deltas=return_deltas, return_matched_rows=return_matched_rows, verbose=False)
        return return_val



    def compare_one_row_features(self, row, features_to_compare):
        result = {}
        for feature in features_to_compare:
            train_val = row[feature + '_train']
            live_val = row[feature + '_live']
            try:
                delta = train_val - live_val
                if np.isnan(delta):
                    if np.isnan(train_val) and np.isnan(live_val):
                        delta = 0
                    else:
                        # TODO: figure out how we want to handle missing values.
                        # This is the case where we have the column in both places
                        # But, for this particular row, we have the value for that feature in only one of our train and live
                        pass
            except TypeError:
                if str(train_val) != str(live_val):
                    delta = 1
                else:
                    delta = 0
            result[feature] = delta

        return pd.Series(result)

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



def load_concordia(persistent_db_config=None):
    default_db_config = {
        'host': 'localhost'
        , 'port': 27017
        , 'db': '_concordia'
    }

    if persistent_db_config is not None:
        default_db_config.update(persistent_db_config)

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
    if '_concordia_created_at' in concordia_info:
        del concordia_info['_concordia_created_at']

    concord = Concordia(**concordia_info)

    return concord
