from pdb_gan.imports import *


class Preprocessor:
    def __init__(self, mode: str):
        path = "temp"
        assert os.path.exists(path), log.error("No 'temp' folder found")
        assert len(os.listdir(path)) > 0, log.error("'temp' folder is empty")

        self.dim = 64
        self.mode = mode
        self.iterable = ["temp/" + file for file in os.listdir(path)]
        self.workers = mp.cpu_count()

    def process_img(self, path: str):
        """
        Extracts arrays for each image given the path and dimension (e.g. 28x28).
        """
        try:
            img = plt.imread(path)[8:, :-8][::3, ::3]
            img[53:, :12] = np.zeros(img[53:, :12].shape)

            if self.mode == "rgb":
                img = img[:, :, :3]
                assert img.shape == (self.dim, self.dim, 3), \
                    log.error("File at {} has shape {} instead of ({}, {}, 3)".format(
                        path, img.shape, self.dim, self.dim))
            if self.mode == "gray":
                img = np.dot(img[:, :, :3], [0.299, 0.587, 0.144])
                assert img.shape == (self.dim, self.dim), \
                    log.error("File at {} has shape {} instead of ({}, {})".format(
                        path, img.shape, self.dim, self.dim))
            if self.mode == "binary":
                img = np.dot(img[:, :, :3], [0.299, 0.587, 0.144]).astype(bool)
                assert img.shape == (self.dim, self.dim), \
                    log.error("File at {} has shape {} instead of ({}, {})".format(
                        path, img.shape, self.dim, self.dim))

        except OSError:
            log.error("Unable to process: {}".format(path))
            if self.mode == "rgb":
                img = np.zeros((self.dim, self.dim, 3))
            if self.mode == "gray":
                img = np.zeros((self.dim, self.dim))
            if self.mode == "binary":
                img = np.zeros((self.dim, self.dim))

        return img, path[5:-4]

    def save(self, data: list):
        imgs, lbls = data

        log.info("Saving to npy file...")
        np.save("data\images_{}{}.npy".format(self.mode, self.dim), np.array(imgs))
        np.save("data\labels_{}{}.npy".format(self.mode, self.dim), np.array(lbls))

    def run(self):
        assert self.mode is "rgb" or self.mode is "gray" or self.mode is "binary", \
            log.error("'mode' argument must be either 'rgb', 'gray', or 'binary'")

        log.info("Processing images using {} workers".format(self.workers))

        imgs, lbls = [], []
        with Progress() as progress:
            task = progress.add_task("[green]Generating {}x{} arrays...".format(self.dim, self.dim),
                                     total=len(self.iterable))
            with mp.Pool(processes=self.workers) as pool:
                for array, label in pool.imap(self.process_img, self.iterable):
                    progress.update(task_id=task, advance=1)
                    # If array only contains zeros, then do not append
                    if array.any():
                        imgs.append(array)
                        lbls.append(label)

        assert len(imgs) == len(lbls), \
            log.error("imgs and lbls length mismatch")

        self.save(data=[imgs, lbls])
