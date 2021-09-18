from matplotlib.backends._backend_tk import FigureCanvasTk
import matplotlib.pyplot as plt
import tkinter as tk
from typing import Callable


class KeyEvents:

    def __init__(self, fig: plt.Figure):
        self.events = {}
        self._canvas: FigureCanvasTk = fig.canvas
        self._widget: tk.Widget = self._canvas.get_tk_widget()
        self._event_name = "<KeyPress>"
        self._tk_event = self._widget.bind(self._event_name, self._wrapper)

    def _wrapper(self, e: tk.Event):
        key = e.char
        if key in self.events:
            self.events[key](e)

    def on_key(self, key: str, callback: Callable[[tk.Event], None]):
        self.events[key] = callback
