from pdb_gan.imports import *


def plot_examples(n: int, images: np.ndarray, labels: np.ndarray, show_title: bool = True):
    examples = [choice(range(len(images))) for _ in range(n ** 2)]
    fig, axes = plt.subplots(n, n, figsize=(8, 8))

    counter = 0
    for row in range(n):
        for col in range(n):
            # If RGB, then change background from black to white
            if images.shape[-1] == 3:
                img = np.where(images[examples[counter]] == 0, 1, images[examples[counter]])
            else:
                img = images[examples[counter]]

            axes[row, col].imshow(img, cmap="binary", interpolation="nearest")
            axes[row, col].axis("off")
            if show_title:
                axes[row, col].set_title(labels[examples[counter]])

            counter += 1

    fig.tight_layout()
    fig.show()
    fig.savefig("gray64.png")


if __name__ == "__main__":
    imgs, lbls = np.load("images_gray64.npy"), np.load("labels_gray64.npy")
    plot_examples(n=10, images=imgs, labels=lbls, show_title=False)
