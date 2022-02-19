# Datasets

This directory contains implementation of dataset utility classes for time series
prediction modelling. There are two types of modelling this packages supports, 
described in [this paper](https://arxiv.org/abs/1902.00683). Modelling types are:
* Predictive Modelling
* Simulation Modelling

### Predictive Modelling

Model learns to predict future states of a dynamical system based on the sequence of
previous time states. Dataset is prepared in form:
```math
Y_{T+N } = f(X_{T-M})
```

### Simulation Modelling

Model learns to predict system outputs based on its forcing. Dataset is prepared in form:
```math
Y_{T+N} = f(U_{T+N})
```

Simulation datasets support target masking, which causes model to be evaluated using
only M last samples of the predicted trajectory. This allows more accurate evaluation
since models have time to warmup before actual inference.

### Classes

Module contains following classes:
* `NumpyPredictiveSequenceGenerator` - array based generator for predictive modelling
* `NumpySimulationSequenceGenerator` - array based generator for simulation modelling
* `PandasPredictiveSequenceGenerator` - pandas based generator for predictive modelling
* `PandasSimulationSequenceGenerator` - pandas based generator for predictive modelling
* `TrochDataLoaderMixin` - class extending functionality of generators by adding torch efficient data loaders
* `ChunkMixin` - class extending functionality of generators by allowing them to split sequences with any ration (by default only train and test sets are generated)