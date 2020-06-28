from pdb_gan.imports import *


class Preprocessor:
    def __init__(self, mode: str, angle: str = ".png"):
        path = "temp"
        assert os.path.exists(path), log.error("No 'temp' folder found")
        assert len(os.listdir(path)) > 0, log.error("'temp' folder is empty")
        
        if angle != ".png":
            assert any([view in angle for view in ["front", "side", "top"]]), log.error("Wrong 'angle' argument")

        self.dim = 64
        self.mode = mode
        self.iterable = ["temp/" + file for file in os.listdir(path) if angle in file]
        self.workers = mp.cpu_count()

    def process_img(self, path: str):
        """
        Extracts arrays for each image given the path and dimension (e.g. 28x28).
        """
        try:
            # 64x64
            img = plt.imread(path)[4:-4, 4:-4][::3, ::3]
            img[52:, :13] = np.zeros(img[52:, :13].shape)
            # 28x28
            # img = plt.imread(path)[2:-2, 2:-2][::7, ::7]
            # img[23:, :6] = np.zeros(img[23:, :6].shape)
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
        
        path = "data"
        try:
            assert os.path.exists(path)
        except AssertionError:
            os.mkdir(path)

        log.info("Saving to npy file...")
        np.save("data/images_{}{}.npy".format(self.mode, self.dim), np.array(imgs))
        np.save("data/labels.npy")

    def run(self):
        assert self.mode is "rgb" or self.mode is "gray" or self.mode is "binary", \
            log.error("'mode' argument must be either 'rgb', 'gray', or 'binary'")

        log.info("Processing images using {} workers".format(self.workers))

        imgs, lbls = [], []
        with Progress() as progress:
            task = progress.add_task("[green]Generating {}x{} arrays...".format(self.dim, self.dim),
                                     total=len(self.iterable))
            with mp.Pool(processes=self.workers) as pool:
                for array, label in pool.imap(self.process_img, self.iterable, chunksize=500):
                    progress.update(task_id=task, advance=1)
                    # If array only contains zeros, then do not append
                    if array.any():
                        imgs.append(array)
                        lbls.append(label)

        assert len(imgs) == len(lbls), \
            log.error("imgs and lbls length mismatch")

        self.save(data=[imgs, lbls])
