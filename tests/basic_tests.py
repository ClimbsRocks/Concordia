import os
import sys

import dill

print('Starting now')

sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
sys.path = [os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')] + sys.path
os.environ['is_test_suite'] = 'True'
print('sys.path')
print(sys.path)

import redis

import aml_utils
from concordia import Concordia

# TODO: flush the redis db

print('Training an ML model now')
ml_predictor_titanic = aml_utils.train_basic_binary_classifier()
print('ML model is now trained')

namespace = '__test_env'

print('Creating concord now')
concord = Concordia(in_memory_db_config={'db': 8}, namespace=namespace)
print('concord created')

rdb = redis.StrictRedis(host='localhost', port=6379, db=8)
rdb.flushdb()


def test_add_new_model():
    print('running test now')
    model_id = 'ml_predictor_titanic_1'

    # Make sure that nothing is in either mdb or rdb beforehand
    # TODO: figure out the key
    redis_key_model = '{}_{}_{}'.format(namespace, model_id, 'model')
    starting_val = rdb.get(redis_key_model)

    assert starting_val is None

    # assert that mongo is also none

    concord.add_model(model=ml_predictor_titanic, model_id=model_id)


    # Assert that there is something in the db now
    post_insert_val = rdb.get(redis_key_model)
    assert post_insert_val is not None
    print(post_insert_val)

if __name__ == '__main__':
    test_add_new_model()