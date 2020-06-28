from pdb_gan.imports import *


def main():
    fetch = Fetcher()
    fetch.run()
    
    for mode in ["binary", "gray", "rgb"]:
        log.info("Running on '{}' mode...".format(mode))
        preprocess = Preprocessor(mode=mode, angle="front")
        preprocess.run()


if __name__ == "__main__":
    main()
