from tensorflow.keras.models import Model
from numpy import expand_dims, squeeze
import matplotlib.pyplot as plt
from models.AutoEncoder import AutoEncoder
from dataloader.DataLoader import DataLoader
from utils.KeyEvent import init_key_event
from utils.Weights import load_weights, save_weights
from typing import List


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


def run(loader: DataLoader, batch_size=256, epochs=10):

    model = AutoEncoder()
    dataset = loader.load_images(batch_size=batch_size)

    plt.ion()
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
        for data in dataset:
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
            if stop[0]:
                break
        if stop[0]:
            break
        ax.plot([index, index], [0, 100], "k--")
        dataset = dataset.shuffle(batch_size)
    return model, fig


def test_reconstruction(model: Model, loader: DataLoader):
    for x in loader.get_random_test_images(10):
        fig1 = plt.figure()
        plt.imshow(x)
        x = expand_dims(x, axis=0)
        y = model(x)
        fig2 = plt.figure()
        plt.imshow(squeeze(y.numpy()))
        wait_on_plot([fig1, fig2])


if __name__ == '__main__':
    plt.switch_backend("TkAgg")
    from matplotlib import rcsetup
    from os.path import isfile

    loader = DataLoader("data")
    rcsetup.validate_backend("TkAgg")
    if isfile("weights/weights.npy"):
        model = AutoEncoder()
        load_weights(model, "weights")
    else:
        model, figure = run(loader, epochs=20, batch_size=125)
        save_weights(model, "weights")
        wait_on_plot([figure])
    test_reconstruction(model, loader)
