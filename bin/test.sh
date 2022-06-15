#!/bin/bash

# install wandb
pip install wandb

# Run all tests
for test in tests/**/*.py; do
  echo "Running $test"
  $test
done
