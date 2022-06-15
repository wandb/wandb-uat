#!/usr/bin/env python

from test_01_basic import main


if __name__ == "__main__":
    from types import SimpleNamespace

    args = SimpleNamespace(
        data_size=60_000, batch=128, epochs=3, tensorboard=True
    )
    main(args)
