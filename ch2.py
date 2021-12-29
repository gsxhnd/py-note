from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import matplotlib


def show_openml_image():
    mnist = fetch_openml('mnist_784')
    X, y = mnist["data"], mnist["target"]

    some_digit = X.loc[[36000]].values
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    show_openml_image()
