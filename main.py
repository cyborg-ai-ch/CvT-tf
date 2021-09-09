from tensorflow.keras.models import Model
from numpy import expand_dims, squeeze, argmax, zeros, isnan
import matplotlib.pyplot as plt
from models.cvt import ConvolutionalVisionTransformer
from dataloader.DataLoader import DataLoader, DataLoaderCifar
from utils.KeyEvent import init_key_event
from utils.Weights import load_weights, save_weights
from typing import List
from config.config import SPEC


def wait_on_plot(figures: List[plt.Figure]):
    figure_closed = False
    while not figure_closed:
        try:
            for figure in figures:
                figure.canvas.flush_events()
        except Exception:
            figure_closed = True
            plt.close()
            for figure in figures:
                try:
                    figure.canvas.flush_events()
                except Exception:
                    continue
            plt.ioff()
            plt.ion()


def run(loader: DataLoader, batch_size=256, epochs=10, spec=SPEC):
    model = ConvolutionalVisionTransformer(spec=spec, num_classes=loader.num_classes)
    model(zeros([1] + loader.image_size))
    model.summary()
    # dataset = loader.load_images(batch_size=batch_size)

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot(111)
    y_loss = [0]
    x = [0]
    plot_loss: plt.Line2D = ax.plot(x, y_loss, "ro", label="loss")[0]
    fig.show()

    index = 0
    fig.legend()
    x = []
    y_loss = []

    stop = [False]

    def stop_training(e):
        stop[0] = True

    init_key_event("q", False, fig, stop_training)

    for epoch in range(epochs):
        for data in loader.batch_generator(batch_size, "train"):
            index += 1
            x.append(index)
            losses = model.train_step(data)
            loss = losses["loss"]
            y_loss.append(loss)
            plot_loss.set_xdata(x)
            plot_loss.set_ydata(y_loss)
            ax.set_xlim(0, index + 1)
            maximum = max(y_loss[-(len(y_loss) // 3):])
            ax.set_ylim(0, maximum + maximum / 5.0)
            fig.canvas.draw()
            fig.canvas.flush_events()
            print("loss {:.3E}"
                  .format(loss))
            if stop[0] or isnan(loss):
                if isnan(loss):
                    stop[0] = True
                break
        if stop[0]:
            break
        ax.plot([index, index], [0, 100], "k--")
    return model, fig


def test(model: Model, loader: DataLoader, number_of_images=100):
    labels_true = []
    labels = []
    count = 0
    for x, y_true in loader.get_random_test_images(number_of_images):
        # fig1 = plt.figure()
        # plt.imshow(x)
        x = expand_dims(x, axis=0)
        y = model(x).numpy()
        cat_predict = int(squeeze(argmax(y)))
        cat_true = int(y_true)
        match = cat_predict == cat_true
        if match:
            count += 1
        labels.append(cat_predict)
        labels_true.append(cat_true)
        # fig2 = plt.figure()
        # plt.plot(range(len(y_true), y_true), "bo")
        # wait_on_plot([fig1, fig2])
    plt.figure()
    plt.plot(range(len(labels)), labels_true, "bo")
    plt.plot(range(len(labels)), labels, "go")
    plt.title('match: ' + str(count) + ' cat: ' + str(number_of_images) + ' yield: ' + str(count/number_of_images))
    plt.show()


if __name__ == '__main__':
    plt.switch_backend("TkAgg")
    from matplotlib import rcsetup
    from os.path import isfile

    plt.ion()

    loader = DataLoaderCifar()
    rcsetup.validate_backend("TkAgg")
    if isfile("weights/weights.npy"):
        model = ConvolutionalVisionTransformer(spec=SPEC, num_classes=loader.num_classes)
        load_weights(model, "weights", input_shape=[1] + loader.image_size)
    else:
        model, figure = run(loader, epochs=100, batch_size=128)
        save_weights(model, "weights")
        wait_on_plot([figure])
    test(model, loader)
