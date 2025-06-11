# from src.test_vton import test_vton
from src.SAM_inference import test_image
import argparse


def terminal_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--inference", action="store_true")
    return parser.parse_known_args()[0]


def main():
    args = terminal_args()
    # if args.test:
    #     test_vton()
    # args.inference:
    test_image()


if __name__ == "__main__":
    main()
