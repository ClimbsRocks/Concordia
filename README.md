# Concordia

[![Build Status](https://travis-ci.org/ClimbsRocks/Concordia.svg?branch=master)](https://travis-ci.org/ClimbsRocks/Concordia)
[![Coverage Status](https://coveralls.io/repos/github/ClimbsRocks/Concordia/badge.svg?branch=master&cacheBuster=1)](https://coveralls.io/github/ClimbsRocks/Concordia?branch=master)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)]((https://img.shields.io/github/license/mashape/apistatus.svg))

Concordia is a part of a suite of open-source machine learning packages that allow organizations to more rapidly develop and deploy machine learning models.

## Installation

`pip install concordia`

## Description

Using Concordia, you should be able to rapidly have confidence in your shipped ML models.

If everything's working as expected, you should be able to see that, and heave a sigh of relief.

If things are not going according to plan, again, you should be able to see that rapidly, and nearly as quickly see the root cause of those discrepancies.

It's designed to work across environments, with many critical parameters configurable (such as database settings).


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


## Advanced Options










## Fully contained example























