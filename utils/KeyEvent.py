from matplotlib.backends._backend_tk import FigureCanvasTk
import matplotlib.pyplot as plt
import tkinter as tk
from typing import Callable


def init_key_event(key: str, on_key_release: bool, fig: plt.Figure, callback: Callable[[tk.Event], None]):
    canvas: FigureCanvasTk = fig.canvas
    widget: tk.Widget = canvas.get_tk_widget()
    event_name = "<KeyRelease>" if on_key_release else "<KeyPress>"

    def _wrapper(e: tk.Event):
        if e.char == key:
            callback(e)

    widget.bind(event_name, _wrapper)
