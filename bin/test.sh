#!/bin/bash

ARGS=()
PASSED=()
FAILED=()
SKIPS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --all)
      ALL=yes
      shift
      ;;
    --skip-media)
      SKIPS+=("^tests/02-media/")
      shift
      ;;
    --skip-torch)
      SKIPS+=("^tests/03-torch/")
      shift
      ;;
    --skip-tensorflow)
      SKIPS+=("^tests/04-tensorflow/")
      shift
      ;;
    --skip-keras)
      SKIPS+=("^tests/05-keras/")
      shift
      ;;
    --*)
      echo "ERROR: unknown option"
      exit 1
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done


if  [ ${#ARGS[@]} -ne 0 ]; then
  TESTS=${ARGS[@]}
else
  TESTS="tests/*/*.py"
fi

for t in $TESTS; do
  skip=0
  for s in ${SKIPS[@]}; do
    if [[ "$t" =~ $s ]]; then
      skip=1
      break
    fi
  done
  if [ $skip -ne 0 ]; then
    continue
  fi
  python $t
  R=$?
  if [ $R -ne 0 ]; then
    FAILED+=("$t")
  else
    PASSED+=("$t")
  fi
done

echo ""
echo "-------"
echo "Results"
echo "-------"

if [ ${#PASSED[@]} -ne 0 ]; then
  echo "Passed tests:"
  for t in "${PASSED[@]}"; do
    echo "  $t"
  done
fi

if [ ${#FAILED[@]} -ne 0 ]; then
  echo "Failed tests:"
  for t in "${FAILED[@]}"; do
    echo "  $t"
  done
  echo ""
  echo "Failed!"
  exit 1
else
  echo ""
  echo "Success!"
fi
