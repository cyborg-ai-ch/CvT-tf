from tensorflow.keras.models import Model
from os import urandom
from numpy import squeeze, argmax, zeros, isnan, array, stack as np_stack, count_nonzero
import matplotlib.pyplot as plt
from models.cvt import ConvolutionalVisionTransformer
from dataloader.DataLoader import DataLoader, DataLoaderCifar
from utils.LossPlot import LossPlot
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


def train(loader: DataLoader, batch_size=512, epochs=10, spec=SPEC, start_weights=None, learning_rate=5e-4):
    model = ConvolutionalVisionTransformer(spec=spec, learning_rate=learning_rate, num_classes=loader.num_classes)

    if start_weights is None or start_weights == "":
        model(zeros([1] + loader.image_size))
    else:
        load_weights(model, start_weights, [1] + loader.image_size)

    model.summary()
    stop = [False]

    def stop_training(e):
        stop[0] = True

    plot = LossPlot()
    plot.events.on_key("q", stop_training)

    for epoch in range(epochs):
        for data in loader.batch_generator(batch_size, "train"):
            losses = model.train_step(data, validation_data=loader.validation_set(size=128))
            print(f"{[f'{key} {value}' for key, value in losses.items()]}")
            plot.update(losses["loss"], losses["val_loss"])
            if stop[0] or isnan(losses["loss"]):
                if isnan(losses["loss"]):
                    stop[0] = True
                break
        if stop[0]:
            break
    return model, plot.figure


def test(model: Model, loader: DataLoader, number_of_images=1000, split="test", seed=None):
    test_images = number_of_images
    x, y_true = loader.get_random_test_images(test_images, split=split, seed=seed)
    y = model(x).numpy()
    y = squeeze(argmax(y, axis=-1))
    y_true = squeeze(y_true)
    hits = count_nonzero(y == y_true)
    fig = plt.figure()
    plt.plot(range(len(y)), y_true, "bo")
    plt.plot(range(len(y)), y, "go")
    plt.title('match: ' + str(hits) + ' cat: ' + str(test_images) + ' yield: ' + str(hits/test_images))
    plt.show()
    return fig


if __name__ == '__main__':
    plt.switch_backend("TkAgg")
    from matplotlib import rcsetup
    from os.path import isfile

    plt.ion()

    loader = DataLoaderCifar(image_size=[72, 72, 3])
    rcsetup.validate_backend("TkAgg")
    if isfile("weights/weights.npy"):
        model = ConvolutionalVisionTransformer(spec=SPEC)
        load_weights(model, "weights", input_shape=[1] + loader.image_size)
    else:
        model, figure = train(loader,
                              epochs=300,
                              batch_size=512,
                              start_weights="",
                              learning_rate=1e-3)
        save_weights(model, "weights")
        wait_on_plot([figure])
    figure = test(model, loader, number_of_images=1000, split="test", seed=None)
    wait_on_plot([figure])
