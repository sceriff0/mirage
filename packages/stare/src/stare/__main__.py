"""``python -m stare`` -- the same entry point as the ``stare`` console script."""

from stare.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
