
import dill
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

    def __init__(self, persistent_db_config=None, in_memory_db_config=None, namespace='_concordia'):

        print('Welcome to Concordia! We\'ll do our best to take a couple stressors off your plate and give you more confidence in your machine learning systems in production.')
        # TODO: Default values
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


    def add_model(self, model, model_id, feature_names=None, feature_importances=None):
        print('One thing to keep in mind is that each model_id must be unique in each db configuration. So if two Concordia instances are using the same database configurations, you should make sure their model_ids do not overlap.')

        redis_key_model = '{}_{}_{}'.format(self.namespace, model_id, 'model')
        self.rdb.set(redis_key_model, dill.dumps(model))

        # redis_key_model_info = '{}_{}_{}'.format(self.namespace, model_id, 'model_info')
        # TODO: get feature names automatically if possible

        mdb_doc = {
            'namespace': self.namespace
            , 'value_type': 'model_info'
            , 'train_or_serve': 'model_info'
            , 'row_id': 'model_info'
            , 'model': dill.dumps(model)
            , 'model_id': model_id
            , 'feature_names': feature_names
            , 'feature_importances': feature_importances
        }
        self.mdb.insert_one(mdb_doc)

        # TODO: warn the user if that key exists already
        # maybe even take in errors='raise', but let the user pass in 'ignore' and 'warn' instead


        pass


    def predict(self, features, model_ids, shadow_models=None):
        pass


    def predict_proba(self, features, model_ids, shadow_models=None):
        pass


    def add_training_data_and_predictions(self, model_ids, data, predictions):
        pass


    def add_outcome_values(self, model_ids, row_ids, y_labels):
        pass


    def reconcile_predictions(self):
        pass


    def reconcile_features(self):
        pass


    def reconcile_labels(self):
        pass


    def reconcile_all(self):
        pass


    def track_features_over_time(self):
        pass


    def track_missing_features(self):
        pass


    def get_values(self, model_id, value_type):
        # value_type is in ['training_features', 'serving_features', 'training_predictions', 'serving_predictions', 'training_labels', 'serving_labels']
        pass






    # These are explicitly out of scopre for our initial implementation
    def custom_predict(self):
        pass


    def custom_db_insert(self):
        pass


    # Lay out what the API must be
        #
    def custom_db_retrieve(self):
        pass


    def save_from_redis_to_mongo(self):
        pass



def load_concordia(name, db_connection):
    pass
