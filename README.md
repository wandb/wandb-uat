# `wandb-uat`
User acceptance testing for the [Weights & Biases](https://wandb.com) python SDK library.

## Introduction
This repository contains user acceptance tests for the
[Weights & Biases](https://github.com/wandb/wandb) library
and utilities to run the tests in cloud environments.

## Installation
The recommended installation includes optional dependencies, which will provide the user
with the most complete experience logging rich data types.

```shell
$ pip install wandb[media]==0.13.9
```

## Testing
A test suite has been created to validate basic functionality of the library.
By default, it will run all the tests and report a non-zero exit status if the script
did not successfully execute all the tests.

The tests assume the python environment has been initialized and
a [W&B API key](https://wandb.ai/authorize) has been configured.

```shell
$ git clone https://github.com/wandb/wandb-uat.git
$ cd wandb-uat/
$ ./bin/test.sh
```

The test script supports skipping categories of tests if certain packages are not available.
For example:

```shell
$ ./bin/test.sh --skip-torch
```

Usage help is available with:
```shell
$ ./bin/test.sh --help
```
