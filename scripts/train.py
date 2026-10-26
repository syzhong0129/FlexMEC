"""Train, select by validation mean MCC and evaluate each fold."""
import argparse
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Set environment variables before running; see configs/e06.env.example and README.md.",
    )
    parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from flexmec.engine.runner import run
    run()


if __name__ == "__main__":
    main()
