import matplotlib.pyplot as plt
from .KeyEvent import KeyEvents


class LossPlot:

    def __init__(self):
        self._fig: plt.Figure = plt.figure()
        self._ax: plt.Axes = self._fig.add_subplot(111)
        self._plot_loss: plt.Line2D = self._ax.plot([0, 100], [0, 0], "ro", label="loss")[0]
        self._plot_val_loss: plt.Line2D = self._ax.plot([0, 100], [0, 0], "go", label="val loss")[0]
        self._y_loss = []
        self._y_val_loss = []
        self._total_max = 0.0
        self._fig.legend()
        self._ax.grid(True)
        self._fig.show()
        self.events = KeyEvents(self._fig)
        self.events.on_key("r", self.resize)
        self.events.on_key("h", self.full_size)

    def update(self, loss: float, val_loss: float):
        total_max = max([loss, val_loss])
        x = range(len(self._y_loss) + 1)
        self._total_max = max([self._total_max, total_max])
        self._y_loss.append(loss)
        self._y_val_loss.append(val_loss)
        self._plot_loss.set_xdata(x)
        self._plot_loss.set_ydata(self._y_loss)
        self._plot_val_loss.set_xdata(x)
        self._plot_val_loss.set_ydata(self._y_val_loss)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def resize(self, *args):
        y = self._y_loss
        length = len(y)
        self._ax.set_xlim(0, length + 1)
        max_index = length // 3 if length // 3 < 1000 else 1000
        maximum = max(y[-max_index:])
        self._ax.set_ylim(0, maximum + maximum / 5.0)

    def full_size(self, *args):
        self._ax.set_xlim(0.0, len(self._y_loss) + 1)
        self._ax.set_ylim(0.0, self._total_max * 5.0/4.0)

    @property
    def figure(self):
        return self._fig
