# Concordia

[![Build Status](https://travis-ci.org/ClimbsRocks/Concordia.svg?branch=master)](https://travis-ci.org/ClimbsRocks/Concordia)
[![Coverage Status](https://coveralls.io/repos/github/ClimbsRocks/Concordia/badge.svg?branch=master&cacheBuster=1)](https://coveralls.io/github/ClimbsRocks/Concordia?branch=master)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)]((https://img.shields.io/github/license/mashape/apistatus.svg))

Concordia is a part of a suite of open-source machine learning packages that allow organizations to more rapidly develop and deploy machine learning models.

## Installation

`pip install concordia`

## Description

Concordia is a tracking and analytics tool for machine learning models running in production.

Using Concordia, you should be able to rapidly have confidence in your shipped ML models.

If everything's working as expected, you should have quick proof that you're in a position to scale up this model.

If things are not going according to plan, you should be able to see that rapidly, and have a suite of information to hone in on the root cause of those discrepancies.


## Basic Setup

```

from concordia import Concordia
concord = Concordia()

ml_predictor = load_ml_model()
concord.add_model(model=model, model_id='model123')

```


## Basic Usage

#### In your training environment

The goal here is to save the features and predictions as you calcate them in your training environment. Then, we can compare these to the features and predictions coming from your live environment. We use the row_ids to match rows across the two environments.

```

df = load_my_data()
ml_predictor = train_ml_model()

predictions = ml_predictor.predict(df)

concord.add_data_and_predictions(model_id='model123', features=df, predictions=predictions, row_ids=df['my_row_identifier'])

```


#### In your live environment

```

from concordia import load_concordia
concord = load_concordia()

data = get_live_data()

prediction = concord.predict(model_id='model123', features=data, row_id=data['my_row_identifier'])

```


#### In your analytics environment

```

from Concordia import load_concordia
concord = load_concordia()

concord.analyze_prediction_discrepancies(model_id='model123')

concord.analyze_feature_discrepancies(model_id='model123')

```


## Infrastructure Assumptions

Concordia relies on MongoDB and Redis. These can be either local, or in the cloud. You can specify DB credentials and connection options when creating and loading Concordia.


## Database Configuration

You can easily specify your own DB connection. You'll need to do this both when creating the Concordia instance in the first place, as well as when you `load_concordia()` to get access to that same Concordia instance later.

```

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

concord = Concordia(in_memory_db_config=in_memory_db_config, persistent_db_config=persistent_db_config)

# To load this instance of Concordia later, use that same persistent_db_config
concord = load_concordia(persistent_db_config=persistent_db_config)

```


## What does Concordia do, under the hood?

- It ensures a consistent db schema
- It ensures a consistent way of saving data (predictions and features), so that data can be matched up and compared later
- It builds in a suite of analytics tools for analyzing discrepancies
- All of this happens automatically for each model that you ship

