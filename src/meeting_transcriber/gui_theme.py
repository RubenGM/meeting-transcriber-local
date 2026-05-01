from __future__ import annotations

from tkinter import ttk


PAD_SMALL = 6
PAD_MEDIUM = 10
PAD_LARGE = 16


def configure_theme(style: ttk.Style) -> None:
    style.configure("Primary.TButton", padding=(14, 7))
    style.configure("TLabelFrame", padding=PAD_MEDIUM)

