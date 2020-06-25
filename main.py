from pdb_gan.imports import *


def main():
    # fetch = Fetcher()
    # fetch.run()

    preprocess = Preprocessor(mode="binary")
    preprocess.run()


if __name__ == "__main__":
    main()
