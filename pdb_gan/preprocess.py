from pdb_gan.imports import *


def process_img(path: str):
    """
    Extracts 64x64 arrays for each image given the path.

    :param path: path to the image in png format
    :return: 64x64 array
    """
    try:
        img = plt.imread(path)[8:, :-8][::3, ::3]
        img[53:, :12] = np.zeros(img[53:, :12].shape)
        # img = img[:, :, :3]
        img = np.dot(img[:, :, :3], [0.299, 0.587, 0.144])
        assert img.shape == (64, 64), \
            log.error("File at {} has shape {} instead of (64, 64)".format(path, img.shape))
    except:
        log.error("Unable to process: {}".format(path))
        img = np.zeros((64, 64))

    return img, path[5:-4]


if __name__ == "__main__":
    path = os.listdir("temp")
    iterable = ["temp/" + file for file in path]
    workers = mp.cpu_count()
    log.info("Processing images using {} workers".format(workers))

    imgs, lbls = [], []
    with Progress() as progress:
        task = progress.add_task("[green]Generating 64x64 arrays...", total=len(iterable))
        with mp.Pool(processes=workers) as pool:
            for array, label in pool.imap(process_img, iterable):
                progress.update(task_id=task, advance=1)
                if array.any():
                    imgs.append(array)
                    lbls.append(label)

    assert len(imgs) == len(lbls), \
        log.error("imgs and lbls lists do not share the same length")

    log.info("Saving to npy file...")
    np.save("images_gray64.npy", np.array(imgs))
    np.save("labels_gray64.npy", np.array(lbls))
