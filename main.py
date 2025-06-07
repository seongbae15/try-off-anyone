# # from src.test_vton import test_vton
# from src.inference import test_image
from src.train import train

# import argparse
from src.train_g import train_g


# def terminal_args():
#     parser = argparse.ArgumentParser(description="Simple example of a training script.")
#     parser.add_argument("--test", action="store_true")
#     parser.add_argument("--inference", action="store_true")
#     parser.add_argument("--train", action="store_true")
#     return parser.parse_known_args()[0]


def main():
    train()

    train_g()
    # args = terminal_args()
    # if args.train:
    #     train()
    # elif args.inference:
    #     test_image()


if __name__ == "__main__":
    main()
