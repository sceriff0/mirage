"""``python -m drape`` -- the same entry point as the ``drape`` console script."""

from drape.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
