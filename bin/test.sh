#!/bin/bash

# Run all tests
for test in tests/**/*.py; do
  echo "Running $test"
  $test
done
