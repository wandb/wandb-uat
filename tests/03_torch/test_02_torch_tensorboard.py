#!/usr/bin/env python

from test_01_torch_basic import main


if __name__ == "__main__":
    from types import SimpleNamespace

    args = SimpleNamespace(data_size=60_000, batch=128, tensorboard=True)
    main(args)
