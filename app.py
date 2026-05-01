"""Sonelgaz AI Grid Intelligence Console — futuristic glassmorphism desktop app.

Architecture (how pieces connect)
-------------------------------
``train_models.py`` (offline)
    Loads ``sonelgaz_consumption_data.csv`` → fits scalers + 4 models → writes
    artifacts under ``models/`` (see ``models/README.md``).

``app.py`` (this file, runtime)
    Loads the same CSV + model artifacts, then:
    - **Grid Dashboard**: KPI cards + Matplotlib chart of the last 100 hours.
    - **Forecast Center**: User picks a *horizon label* (Daily/Weekly/Monthly/Quarterly);
      each label routes to a different model, but the **scalar output is still
      next-hour kWh** (same target as training). The long curve on the chart is a
      **visualisation helper**, not a full multi-step model rollout.
    - **Telemetry Data**: Read-only snapshot of recent rows.

Branding: colours are aligned with ``sonalgaz.webp`` (orange #F29125, blue #288BCB).
"""

import tkinter as tk
import math
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from tkinter import messagebox
from tkinter import ttk

import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
DATA_FILE  = Path("sonelgaz_consumption_data.csv")
LOGO_FILE  = Path("sonalgaz.webp")
FEATURES   = ["Hour", "DayOfWeek", "Month", "Season", "Temperature", "Current", "IsHoliday"]
HORIZONS   = ["Daily", "Weekly", "Monthly", "Quarterly"]
MODEL_NAMES = ["Random Forest", "XGBoost", "LSTM", "Transformer"]

# ── Brand palette (extracted from Sonalgaz logo) ───────────────────────────────
C_BG       = "#050E1A"   # deep background
C_SIDEBAR  = "#07111E"   # sidebar glass layer
C_SURFACE  = "#0C1E30"   # card surface
C_SURFACE2 = "#0F2744"   # elevated surface / hover state
C_BORDER   = "#1A4A6A"   # default border
C_ORANGE   = "#F29125"   # primary accent — logo orange
C_ORANGE_H = "#D97B13"   # hovered orange
C_BLUE     = "#288BCB"   # secondary accent — logo blue
C_BLUE_H   = "#1A5E8A"   # dimmed / hover blue
C_TEAL     = "#00D4AA"   # positive / online indicator
C_PURPLE   = "#A259FF"   # transformer model accent
C_TEXT     = "#E8F4FD"   # primary text
C_MUTED    = "#8FB2CC"   # secondary / label text
C_FAINT    = "#2A4A62"   # very subtle text / separators
C_DANGER   = "#FF6D6D"   # error / warning


# ── Global matplotlib dark theme ───────────────────────────────────────────────
def _apply_mpl_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor":   C_SURFACE,
        "axes.facecolor":     C_SURFACE,
        "axes.edgecolor":     C_BORDER,
        "axes.labelcolor":    C_TEXT,
        "axes.titlecolor":    C_TEXT,
        "axes.titlesize":     12,
        "axes.titleweight":   "bold",
        "xtick.color":        C_MUTED,
        "ytick.color":        C_MUTED,
        "text.color":         C_TEXT,
        "grid.color":         "#132638",
        "grid.alpha":         0.55,
        "legend.facecolor":   C_SURFACE,
        "legend.edgecolor":   C_BORDER,
        "legend.labelcolor":  C_TEXT,
        "legend.fontsize":    9,
    })


_apply_mpl_theme()


# ── Reusable glow-hover button ─────────────────────────────────────────────────
class GlowButton(ctk.CTkButton):
    """CTkButton that lights up its border on mouse-enter (glow effect)."""

    def __init__(self, master, glow_color: str = C_ORANGE, base_border: str = C_FAINT, **kwargs):
        kwargs.setdefault("border_width", 1)
        kwargs.setdefault("border_color", base_border)
        self._glow_color  = glow_color
        self._base_border = base_border
        self._base_fg     = kwargs.get("fg_color", "transparent")
        super().__init__(master, **kwargs)
        self.bind("<Enter>", self._on_enter, add="+")
        self.bind("<Leave>", self._on_leave, add="+")

    def _on_enter(self, _=None):
        self.configure(border_color=self._glow_color, border_width=2, fg_color=C_SURFACE2)

    def _on_leave(self, _=None):
        self.configure(border_color=self._base_border, border_width=1, fg_color=self._base_fg)


# ── Main application ───────────────────────────────────────────────────────────
class SonelgazApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Sonelgaz AI  ·  Grid Intelligence Console")
        self.geometry("1440x900")
        self.minsize(1200, 780)
        self.configure(fg_color=C_BG)

        self.logo_img        = None
        self.data            = None
        self._pulse_state    = False
        self.horizon_model_vars = {
            "Daily": tk.StringVar(value="Random Forest"),
            "Weekly": tk.StringVar(value="XGBoost"),
            "Monthly": tk.StringVar(value="LSTM"),
            "Quarterly": tk.StringVar(value="Transformer"),
        }

        self.load_resources()
        self.setup_ui()

    # ── Resource loading ───────────────────────────────────────────────────────

    def load_resources(self) -> None:
        try:
            with open(MODELS_DIR / "rf_model.pkl", "rb") as f:
                self.rf_model = pickle.load(f)
            with open(MODELS_DIR / "xgb_model.pkl", "rb") as f:
                self.xgb_model = pickle.load(f)
            with open(MODELS_DIR / "scaler_X.pkl", "rb") as f:
                self.scaler_X = pickle.load(f)
            with open(MODELS_DIR / "scaler_y.pkl", "rb") as f:
                self.scaler_y = pickle.load(f)

            self.lstm_model = self._load_keras_model(
                MODELS_DIR / "lstm_architecture.json",
                MODELS_DIR / "lstm_weights.weights.h5",
                MODELS_DIR / "lstm_model.keras",
                MODELS_DIR / "lstm_model.h5",
            )
            self.transformer_model = self._load_keras_model(
                MODELS_DIR / "transformer_architecture.json",
                MODELS_DIR / "transformer_weights.weights.h5",
                MODELS_DIR / "transformer_model.keras",
                MODELS_DIR / "transformer_model.h5",
            )

            if DATA_FILE.exists():
                self.data = pd.read_csv(DATA_FILE)
                self.data["Timestamp"] = pd.to_datetime(self.data["Timestamp"])
            else:
                raise FileNotFoundError(f"Missing data file: {DATA_FILE}")

        except Exception as exc:
            print(f"[Startup] {exc}")
            messagebox.showerror("Startup Error", f"Could not load resources:\n{exc}")

    @staticmethod
    def _load_keras_model(arch: Path, weights: Path, fallback_keras: Path, fallback_h5: Path):
        if arch.exists() and weights.exists():
            with open(arch, "r", encoding="utf-8") as f:
                model = tf.keras.models.model_from_json(f.read())
            model.load_weights(weights)
            return model
        if fallback_keras.exists():
            return tf.keras.models.load_model(fallback_keras, compile=False)
        if fallback_h5.exists():
            return tf.keras.models.load_model(fallback_h5, compile=False)
        raise FileNotFoundError(f"No model files found for '{arch.stem}'.")

    # ── UI scaffolding ─────────────────────────────────────────────────────────

    def setup_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(
            self, width=272, corner_radius=0,
            fg_color=C_SIDEBAR,
            border_color=C_BORDER, border_width=1,
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # Content column
        self.shell = ctk.CTkFrame(self, fg_color="transparent")
        self.shell.grid(row=0, column=1, sticky="nsew", padx=16, pady=16)
        self.shell.grid_columnconfigure(0, weight=1)
        self.shell.grid_rowconfigure(1, weight=1)

        self._build_sidebar()
        self._build_topbar()

        # Main display panel (whole page scrollable).
        self.main_frame = ctk.CTkScrollableFrame(
            self.shell,
            fg_color=C_SURFACE,
            corner_radius=18,
            border_color=C_BORDER, border_width=1,
        )
        self.main_frame.grid(row=1, column=0, sticky="nsew", pady=(12, 0))

        self.show_dashboard()

    # ── Sidebar ────────────────────────────────────────────────────────────────

    def _build_sidebar(self) -> None:
        # Logo
        if LOGO_FILE.exists():
            try:
                img = Image.open(LOGO_FILE).convert("RGBA")
                img.thumbnail((200, 120))
                self.logo_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                ctk.CTkLabel(self.sidebar, image=self.logo_img, text="").pack(pady=(28, 2))
            except Exception:
                pass

        ctk.CTkLabel(
            self.sidebar,
            text="SONELGAZ  AI",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=C_TEXT,
        ).pack(pady=(6, 0))

        ctk.CTkLabel(
            self.sidebar,
            text="⚡ Electricity  ·  💧 Water  ·  🔥 Gas",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=C_MUTED,
        ).pack(pady=(3, 14))

        self._rule(self.sidebar)

        # Navigation
        nav_defs = [
            ("  ◈   Project Overview", self.show_overview),
            ("  ◈   Grid Dashboard",  self.show_dashboard),
            ("  ◈   Forecast Center", self.show_prediction),
            ("  ◈   Telemetry Data",  self.show_data),
        ]
        for label, cmd in nav_defs:
            GlowButton(
                self.sidebar,
                text=label,
                command=cmd,
                fg_color="transparent",
                text_color=C_MUTED,
                hover_color=C_SURFACE2,
                font=ctk.CTkFont(size=13, weight="bold"),
                anchor="w",
                height=46,
                corner_radius=10,
                glow_color=C_ORANGE,
                base_border=C_FAINT,
            ).pack(fill="x", padx=14, pady=4)

        self._rule(self.sidebar)

        # System status glass card
        status = ctk.CTkFrame(
            self.sidebar,
            fg_color=C_SURFACE,
            corner_radius=14,
            border_color=C_TEAL, border_width=1,
        )
        status.pack(fill="x", padx=14, pady=(18, 8))

        # Animated pulse dot
        pulse_row = ctk.CTkFrame(status, fg_color="transparent")
        pulse_row.pack(fill="x", padx=12, pady=(10, 4))
        self._pulse_canvas = tk.Canvas(
            pulse_row, width=12, height=12,
            bg=C_SURFACE, highlightthickness=0,
        )
        self._pulse_canvas.pack(side="left")
        self._dot = self._pulse_canvas.create_oval(1, 1, 11, 11, fill=C_TEAL, outline=C_TEAL)
        ctk.CTkLabel(
            pulse_row, text="  SYSTEM  ONLINE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=C_TEAL,
        ).pack(side="left")
        self._animate_pulse()

        # Mini KPI pair
        kpi_row = ctk.CTkFrame(status, fg_color="transparent")
        kpi_row.pack(fill="x", padx=8, pady=(4, 12))
        kpi_row.grid_columnconfigure((0, 1), weight=1)
        self._mini_kpi(kpi_row, "MODELS", "4",  C_BLUE,   0, 0)
        self._mini_kpi(kpi_row, "INPUTS", "7",  C_ORANGE, 0, 1)

        # Version footer
        ctk.CTkLabel(
            self.sidebar,
            text="v2.0  ·  Sonelgaz Grid Console",
            font=ctk.CTkFont(size=9),
            text_color=C_FAINT,
        ).pack(side="bottom", pady=10)

    def _mini_kpi(self, parent, label: str, value: str, color: str, row: int, col: int) -> None:
        box = ctk.CTkFrame(parent, fg_color=C_SURFACE2, corner_radius=8)
        box.grid(row=row, column=col, padx=4, pady=4, sticky="ew")
        ctk.CTkLabel(box, text=label, font=ctk.CTkFont(size=8, weight="bold"), text_color=C_MUTED).pack(pady=(6, 0))
        ctk.CTkLabel(box, text=value, font=ctk.CTkFont(size=18, weight="bold"), text_color=color).pack(pady=(0, 6))

    # ── Top bar ────────────────────────────────────────────────────────────────

    def _build_topbar(self) -> None:
        bar = ctk.CTkFrame(
            self.shell,
            fg_color=C_SIDEBAR,
            corner_radius=14,
            border_color=C_ORANGE, border_width=1,
        )
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_columnconfigure(1, weight=1)

        # Orange accent stripe at top edge
        tk.Canvas(bar, height=2, bg=C_ORANGE, highlightthickness=0, bd=0).grid(
            row=0, column=0, columnspan=3, sticky="ew"
        )

        ctk.CTkLabel(bar, text="⚡", font=ctk.CTkFont(size=22), text_color=C_ORANGE).grid(
            row=1, column=0, padx=(16, 6), pady=12
        )
        ctk.CTkLabel(
            bar,
            text="SONELGAZ  FUTURE GRID INTELLIGENCE  MONITOR",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=C_TEXT,
        ).grid(row=1, column=1, sticky="w", pady=12)

        self._clock_lbl = ctk.CTkLabel(
            bar,
            text="",
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color=C_MUTED,
        )
        self._clock_lbl.grid(row=1, column=2, padx=18, pady=12)
        self._tick_clock()

    # ── Animations ─────────────────────────────────────────────────────────────

    def _animate_pulse(self) -> None:
        self._pulse_state = not self._pulse_state
        color = C_TEAL if self._pulse_state else C_BLUE_H
        self._pulse_canvas.itemconfig(self._dot, fill=color, outline=color)
        self.after(900, self._animate_pulse)

    def _tick_clock(self) -> None:
        self._clock_lbl.configure(text=datetime.now().strftime("%d %b %Y   %H:%M:%S"))
        self.after(1000, self._tick_clock)

    # ── Layout primitives ──────────────────────────────────────────────────────

    def _rule(self, parent, color: str = C_FAINT, padx: int = 14, pady: int = 8) -> None:
        tk.Canvas(parent, height=1, bg=color, highlightthickness=0, bd=0).pack(
            fill="x", padx=padx, pady=pady
        )

    def clear_main_frame(self) -> None:
        for w in self.main_frame.winfo_children():
            w.destroy()

    def _page_header(self, title: str, subtitle: str) -> None:
        wrap = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        wrap.pack(fill="x", padx=24, pady=(22, 8))

        # Left accent bar
        tk.Canvas(wrap, width=4, bg=C_ORANGE, highlightthickness=0, bd=0).pack(
            side="left", fill="y", padx=(0, 14)
        )
        col = ctk.CTkFrame(wrap, fg_color="transparent")
        col.pack(side="left")
        ctk.CTkLabel(col, text=title, font=ctk.CTkFont(size=28, weight="bold"), text_color=C_TEXT).pack(anchor="w")
        ctk.CTkLabel(col, text=subtitle, font=ctk.CTkFont(size=12), text_color=C_MUTED).pack(anchor="w", pady=(2, 0))

    def _glass_card(self, parent, accent: str = C_BLUE, **grid_kw) -> ctk.CTkFrame:
        card = ctk.CTkFrame(
            parent,
            fg_color=C_SURFACE2,
            corner_radius=12,
            border_color=accent, border_width=1,
        )
        if grid_kw:
            card.grid(**grid_kw)
        return card

    def _kpi_card(
        self, parent,
        icon: str, label: str, value: str, unit: str,
        accent: str, **grid_kw
    ) -> ctk.CTkFrame:
        card = self._glass_card(parent, accent=accent, **grid_kw)
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=14, pady=14)

        ctk.CTkLabel(
            inner,
            text=f"{icon}  {label}",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=C_MUTED,
        ).pack(anchor="w")

        val_row = ctk.CTkFrame(inner, fg_color="transparent")
        val_row.pack(anchor="w", pady=(6, 0))
        ctk.CTkLabel(
            val_row, text=value,
            font=ctk.CTkFont(size=30, weight="bold"), text_color=accent,
        ).pack(side="left")
        ctk.CTkLabel(
            val_row, text=f"  {unit}",
            font=ctk.CTkFont(size=12), text_color=C_MUTED,
        ).pack(side="left", anchor="s")

        # Glow underline
        tk.Canvas(inner, height=2, bg=accent, highlightthickness=0).pack(fill="x", pady=(10, 0))
        return card

    def _model_badge(self, parent, name: str, color: str) -> ctk.CTkFrame:
        badge = ctk.CTkFrame(parent, fg_color=color, corner_radius=6)
        ctk.CTkLabel(
            badge, text=name,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=C_BG,
        ).pack(padx=9, pady=3)
        return badge

    def _chart_panel(self, parent, title: str, subtitle: str = "", **pack_kw) -> ctk.CTkFrame:
        wrap = ctk.CTkFrame(
            parent,
            fg_color=C_SURFACE2,
            corner_radius=14,
            border_color=C_BORDER, border_width=1,
        )
        wrap.pack(**pack_kw)
        hdr = ctk.CTkFrame(wrap, fg_color="transparent")
        hdr.pack(fill="x", padx=14, pady=(10, 0))
        ctk.CTkLabel(hdr, text=title, font=ctk.CTkFont(size=13, weight="bold"), text_color=C_TEXT).pack(side="left")
        if subtitle:
            ctk.CTkLabel(hdr, text=subtitle, font=ctk.CTkFont(size=10), text_color=C_MUTED).pack(side="right")
        tk.Canvas(wrap, height=1, bg=C_BORDER, highlightthickness=0).pack(fill="x", padx=14, pady=(6, 0))
        return wrap

    def _get_model_bundle(self, model_name: str):
        """Resolve UI model name to (model_object, accent_color, requires_sequence_input)."""
        model_registry = {
            "Random Forest": (self.rf_model, C_BLUE, False),
            "XGBoost": (self.xgb_model, C_ORANGE, False),
            "LSTM": (self.lstm_model, C_TEAL, True),
            "Transformer": (self.transformer_model, C_PURPLE, True),
        }
        return model_registry[model_name]

    # ── Dashboard ──────────────────────────────────────────────────────────────

    def show_overview(self) -> None:
        self.clear_main_frame()
        self._page_header(
            "Project Overview",
            "Problem statement, solution, contributors, and how the prototype works end-to-end.",
        )

        # Contributors (requested to appear first).
        contributors_card = ctk.CTkFrame(
            self.main_frame,
            fg_color=C_SURFACE2,
            corner_radius=14,
            border_color=C_ORANGE,
            border_width=1,
        )
        contributors_card.pack(fill="x", padx=24, pady=(4, 10))

        ctk.CTkLabel(
            contributors_card,
            text="Project Contributors",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=C_TEXT,
        ).pack(anchor="w", padx=14, pady=(10, 4))
        tk.Canvas(contributors_card, height=1, bg=C_ORANGE, highlightthickness=0).pack(
            fill="x", padx=14, pady=(0, 8)
        )

        names = ctk.CTkFrame(contributors_card, fg_color="transparent")
        names.pack(fill="x", padx=10, pady=(0, 10))
        names.grid_columnconfigure((0, 1), weight=1)
        self._kpi_card(
            names, "👤", "CREATOR", "Aya Bounafa", "", C_ORANGE,
            row=0, column=0, sticky="ew", padx=4, pady=4,
        )
        self._kpi_card(
            names, "👤", "CREATOR", "Belsem Chaghi", "", C_BLUE,
            row=0, column=1, sticky="ew", padx=4, pady=4,
        )

        # Problem statement + solution + how it works (large type for readability in demos).
        about_wrap = 1020
        about_card = ctk.CTkFrame(
            self.main_frame,
            fg_color=C_SURFACE2,
            corner_radius=14,
            border_color=C_BLUE,
            border_width=1,
        )
        about_card.pack(fill="x", padx=24, pady=(0, 10))

        ctk.CTkLabel(
            about_card,
            text="Problem statement",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=C_ORANGE,
        ).pack(anchor="w", padx=18, pady=(14, 6))
        tk.Canvas(about_card, height=2, bg=C_ORANGE, highlightthickness=0).pack(fill="x", padx=18, pady=(0, 10))

        ctk.CTkLabel(
            about_card,
            text=(
                "Face à une demande d’électricité et de gaz qui peut dépasser ce que le réseau peut produire "
                "ou acheminer à certains moments, les opérateurs observent des tensions sur le système — "
                "jusqu’à des coupures ou des interruptions pour les usagers lorsque l’offre ne suit pas la "
                "montée rapide de la consommation."
            ),
            font=ctk.CTkFont(size=18),
            text_color=C_TEXT,
            justify="left",
            wraplength=about_wrap,
        ).pack(anchor="w", padx=18, pady=(0, 14))

        ctk.CTkLabel(
            about_card,
            text="Solution with this AI system",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=C_BLUE,
        ).pack(anchor="w", padx=18, pady=(6, 6))
        tk.Canvas(about_card, height=2, bg=C_BLUE, highlightthickness=0).pack(fill="x", padx=18, pady=(0, 10))

        ctk.CTkLabel(
            about_card,
            text=(
                "Ce prototype utilise l’intelligence artificielle pour estimer la consommation à venir "
                "(prochaine heure, à partir des signaux disponibles). En anticipant la demande, on peut "
                "mieux planifier la production, le dispatching et les investissements — afin de rapprocher "
                "l’offre de la demande et réduire le risque de coupures liées à un déséquilibre prévisible."
            ),
            font=ctk.CTkFont(size=18),
            text_color=C_TEXT,
            justify="left",
            wraplength=about_wrap,
        ).pack(anchor="w", padx=18, pady=(0, 8))

        ctk.CTkLabel(
            about_card,
            text=(
                "Remarque : les données utilisées ici sont synthétiques (démonstration). "
                "Le même pipeline peut être branché sur des données réelles une fois disponibles."
            ),
            font=ctk.CTkFont(size=15),
            text_color=C_MUTED,
            justify="left",
            wraplength=about_wrap,
        ).pack(anchor="w", padx=18, pady=(0, 14))

        ctk.CTkLabel(
            about_card,
            text="How this application works (technical)",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=C_TEXT,
        ).pack(anchor="w", padx=18, pady=(4, 6))

        how_lines = [
            "1. Data — generate_data.py builds sonelgaz_consumption_data.csv (hourly features + kWh target).",
            "2. Training — train_models.py fits scalers and four models, saves everything under models/.",
            "3. App — app.py loads CSV + artifacts: Dashboard for trends, Forecast Center to pick horizon "
            "and model mapping then infer next-hour kWh, Telemetry to validate inputs.",
        ]
        for line in how_lines:
            ctk.CTkLabel(
                about_card,
                text=line,
                font=ctk.CTkFont(size=15),
                text_color=C_MUTED,
                justify="left",
                wraplength=about_wrap,
            ).pack(anchor="w", padx=18, pady=4)
        ctk.CTkLabel(
            about_card,
            text="Tip: full technical detail is in README.md and SYSTEM_DESIGN.md.",
            font=ctk.CTkFont(size=14),
            text_color=C_FAINT,
            justify="left",
            wraplength=about_wrap,
        ).pack(anchor="w", padx=18, pady=(10, 16))

        # Project essentials in a bento layout.
        overview = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        overview.pack(fill="both", expand=True, padx=24, pady=(0, 22))
        overview.grid_columnconfigure((0, 1), weight=1)
        overview.grid_rowconfigure((0, 1), weight=1)

        module_card = self._glass_card(overview, accent=C_BLUE, row=0, column=0, sticky="nsew", padx=6, pady=6)
        ctk.CTkLabel(
            module_card,
            text="Core Modules",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=C_TEXT,
        ).pack(anchor="w", padx=12, pady=(10, 4))
        for line in [
            "• generate_data.py  → synthetic hourly dataset",
            "• train_models.py   → train + evaluate + save artifacts",
            "• app.py            → dashboard, forecast, telemetry UI",
            "• models/           → scalers + ML/DL model files",
        ]:
            ctk.CTkLabel(
                module_card,
                text=line,
                font=ctk.CTkFont(size=11),
                text_color=C_MUTED,
            ).pack(anchor="w", padx=12, pady=2)

        flow_card = self._glass_card(overview, accent=C_ORANGE, row=0, column=1, sticky="nsew", padx=6, pady=6)
        ctk.CTkLabel(
            flow_card,
            text="Execution Flow",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=C_TEXT,
        ).pack(anchor="w", padx=12, pady=(10, 4))
        for line in [
            "1) python generate_data.py",
            "2) python train_models.py",
            "3) python app.py",
            "4) Use Forecast Center for interactive prediction",
        ]:
            ctk.CTkLabel(
                flow_card,
                text=line,
                font=ctk.CTkFont(size=11),
                text_color=C_MUTED,
            ).pack(anchor="w", padx=12, pady=2)

        models_card = self._glass_card(overview, accent=C_TEAL, row=1, column=0, sticky="nsew", padx=6, pady=6)
        ctk.CTkLabel(
            models_card,
            text="Model Stack",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=C_TEXT,
        ).pack(anchor="w", padx=12, pady=(10, 4))
        for line in [
            "• Random Forest (tabular regressor)",
            "• XGBoost (boosted trees)",
            "• LSTM (24-hour sequence model)",
            "• Transformer (attention-based sequence model)",
        ]:
            ctk.CTkLabel(
                models_card,
                text=line,
                font=ctk.CTkFont(size=11),
                text_color=C_MUTED,
            ).pack(anchor="w", padx=12, pady=2)

        notes_card = self._glass_card(overview, accent=C_PURPLE, row=1, column=1, sticky="nsew", padx=6, pady=6)
        ctk.CTkLabel(
            notes_card,
            text="Key Notes",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=C_TEXT,
        ).pack(anchor="w", padx=12, pady=(10, 4))
        for line in [
            "• Input data is synthetic (confidential real data not used).",
            "• UI horizons can be mapped manually to any model.",
            "• Main scalar output is next-hour consumption (kWh).",
            "• Forecast curve is an illustrative projection layer.",
        ]:
            ctk.CTkLabel(
                notes_card,
                text=line,
                font=ctk.CTkFont(size=11),
                text_color=C_MUTED,
            ).pack(anchor="w", padx=12, pady=2)

    def show_dashboard(self) -> None:
        self.clear_main_frame()
        self._page_header("Grid Dashboard", "Real-time consumption overview and infrastructure load analytics.")

        if self.data is None:
            ctk.CTkLabel(
                self.main_frame,
                text="⚠  Data source unavailable — regenerate CSV and retrain models.",
                text_color=C_DANGER,
                font=ctk.CTkFont(size=14, weight="bold"),
            ).pack(pady=32)
            return

        avg_cons = self.data["Consumption"].mean()
        max_cons = self.data["Consumption"].max()
        min_cons = self.data["Consumption"].min()
        avg_temp = self.data["Temperature"].tail(168).mean()

        # ── Row 1: Bento KPI cards ──
        bento = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        bento.pack(fill="x", padx=24, pady=(4, 8))
        bento.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self._kpi_card(bento, "⚡", "AVG  CONSUMPTION", f"{avg_cons:,.1f}", "kWh", C_BLUE,   row=0, column=0, sticky="ew", padx=6, pady=4)
        self._kpi_card(bento, "🔺", "PEAK  LOAD",        f"{max_cons:,.1f}", "kWh", C_ORANGE, row=0, column=1, sticky="ew", padx=6, pady=4)
        self._kpi_card(bento, "🔻", "MIN  LOAD",         f"{min_cons:,.1f}", "kWh", C_MUTED,  row=0, column=2, sticky="ew", padx=6, pady=4)
        self._kpi_card(bento, "🌡", "AVG  TEMP  (7d)",   f"{avg_temp:,.1f}", "°C",  C_TEAL,   row=0, column=3, sticky="ew", padx=6, pady=4)

        # ── Row 2: Chart + side insights ──
        row2 = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        row2.pack(fill="both", expand=True, padx=24, pady=(0, 22))
        row2.grid_columnconfigure(0, weight=3)
        row2.grid_columnconfigure(1, weight=1)
        row2.grid_rowconfigure(0, weight=1)

        # Main chart card
        chart_card = self._glass_card(row2, accent=C_BLUE, row=0, column=0, sticky="nsew", padx=(0, 8), pady=4)
        ctk.CTkLabel(
            chart_card,
            text="Power Utilization Stream  –  Last 100 Hours",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=C_TEXT,
        ).pack(anchor="w", padx=14, pady=(10, 2))
        tk.Canvas(chart_card, height=1, bg=C_BLUE, highlightthickness=0).pack(fill="x", padx=14, pady=(0, 4))

        fig, ax = plt.subplots(figsize=(9, 3.8))
        last_100 = self.data.tail(100)
        y = last_100["Consumption"].values
        x = np.arange(len(y))
        trend = last_100["Consumption"].rolling(8, min_periods=1).mean().values

        ax.fill_between(x, y, alpha=0.15, color=C_BLUE)
        ax.plot(x, y, color=C_BLUE, linewidth=2, label="Consumption")
        ax.plot(x, trend, color=C_ORANGE, linewidth=1.8, linestyle="--", label="8h Trend")
        ax.set_xlabel("Hours  →  Now")
        ax.set_ylabel("kWh")
        ax.set_title("Power Utilization Stream")
        ax.legend()
        ax.grid(True)
        fig.tight_layout(pad=1.4)

        cv = FigureCanvasTkAgg(fig, master=chart_card)
        cv.draw()
        cv.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Side insight panel
        side = self._glass_card(row2, accent=C_ORANGE, row=0, column=1, sticky="nsew", pady=4)
        ctk.CTkLabel(side, text="System Insights", font=ctk.CTkFont(size=12, weight="bold"), text_color=C_TEXT).pack(anchor="w", padx=12, pady=(10, 2))
        tk.Canvas(side, height=1, bg=C_ORANGE, highlightthickness=0).pack(fill="x", padx=12)

        latest = self.data.tail(1).iloc[0]
        insights = [
            ("Last Reading",   f"{latest['Consumption']:.1f} kWh", C_BLUE),
            ("Temperature",    f"{latest['Temperature']:.1f} °C",  C_TEAL),
            ("Current",        f"{latest['Current']:.1f} A",       C_ORANGE),
            ("Total Records",  f"{len(self.data):,}",              C_MUTED),
            ("Date Range",     "2024 – 2025",                       C_MUTED),
        ]
        for label, val, col in insights:
            row_f = ctk.CTkFrame(side, fg_color=C_SURFACE, corner_radius=8)
            row_f.pack(fill="x", padx=10, pady=4)
            ctk.CTkLabel(row_f, text=label, font=ctk.CTkFont(size=9, weight="bold"), text_color=C_MUTED).pack(anchor="w", padx=10, pady=(6, 0))
            ctk.CTkLabel(row_f, text=val, font=ctk.CTkFont(size=14, weight="bold"), text_color=col).pack(anchor="w", padx=10, pady=(0, 6))

    # ── Forecast Center ────────────────────────────────────────────────────────

    def show_prediction(self) -> None:
        self.clear_main_frame()
        self._page_header("Forecast Center", "AI-powered consumption predictions for any planning horizon.")

        # Controls bar
        ctrl = ctk.CTkFrame(
            self.main_frame,
            fg_color=C_SURFACE2,
            corner_radius=14,
            border_color=C_ORANGE, border_width=1,
        )
        ctrl.pack(fill="x", padx=24, pady=(4, 10))

        ctk.CTkLabel(
            ctrl, text="Horizon",
            font=ctk.CTkFont(size=11, weight="bold"), text_color=C_MUTED,
        ).pack(side="left", padx=(16, 6), pady=16)

        self.horizon_var = tk.StringVar(value="Daily")
        ctk.CTkOptionMenu(
            ctrl,
            values=["Daily", "Weekly", "Monthly", "Quarterly"],
            variable=self.horizon_var,
            fg_color=C_SURFACE,
            button_color=C_BLUE,
            button_hover_color=C_BLUE_H,
            text_color=C_TEXT,
            dropdown_fg_color=C_SURFACE2,
            dropdown_text_color=C_TEXT,
            dropdown_hover_color=C_SURFACE,
            font=ctk.CTkFont(size=13, weight="bold"),
            width=160,
        ).pack(side="left", padx=6, pady=16)

        # Model badge strip
        badge_wrap = ctk.CTkFrame(ctrl, fg_color="transparent")
        badge_wrap.pack(side="left", padx=14, pady=16)
        ctk.CTkLabel(badge_wrap, text="Models:", font=ctk.CTkFont(size=10), text_color=C_FAINT).pack(side="left", padx=(0, 6))
        for name, color in [("RF", C_BLUE), ("XGB", C_ORANGE), ("LSTM", C_TEAL), ("Transformer", C_PURPLE)]:
            self._model_badge(badge_wrap, name, color).pack(side="left", padx=3)

        ctk.CTkButton(
            ctrl,
            text="  ▶  RUN PREDICTION",
            command=self.run_prediction,
            fg_color=C_ORANGE,
            hover_color=C_ORANGE_H,
            text_color=C_BG,
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=10,
            height=38,
        ).pack(side="right", padx=16, pady=16)

        # Per-horizon manual model assignment.
        mapper = ctk.CTkFrame(
            self.main_frame,
            fg_color=C_SURFACE2,
            corner_radius=14,
            border_color=C_BLUE,
            border_width=1,
        )
        mapper.pack(fill="x", padx=24, pady=(0, 10))
        ctk.CTkLabel(
            mapper,
            text="Manual model assignment by horizon",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=C_TEXT,
        ).pack(anchor="w", padx=14, pady=(10, 2))

        assignments = ctk.CTkFrame(mapper, fg_color="transparent")
        assignments.pack(fill="x", padx=10, pady=(4, 10))
        assignments.grid_columnconfigure((0, 1, 2, 3), weight=1)

        for idx, horizon_name in enumerate(HORIZONS):
            cell = ctk.CTkFrame(assignments, fg_color=C_SURFACE, corner_radius=8)
            cell.grid(row=0, column=idx, sticky="ew", padx=4, pady=4)
            ctk.CTkLabel(
                cell,
                text=horizon_name,
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=C_MUTED,
            ).pack(anchor="w", padx=8, pady=(8, 2))
            ctk.CTkOptionMenu(
                cell,
                values=MODEL_NAMES,
                variable=self.horizon_model_vars[horizon_name],
                fg_color=C_SURFACE2,
                button_color=C_BLUE,
                button_hover_color=C_BLUE_H,
                text_color=C_TEXT,
                dropdown_fg_color=C_SURFACE2,
                dropdown_text_color=C_TEXT,
                dropdown_hover_color=C_SURFACE,
                font=ctk.CTkFont(size=12, weight="bold"),
            ).pack(fill="x", padx=8, pady=(0, 8))

        # Result panel — use plain frame; vertical scroll comes from outer main_frame scroll.
        self.pred_result_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color=C_SURFACE2,
            corner_radius=14,
            border_color=C_BORDER, border_width=1,
        )
        self.pred_result_frame.pack(fill="both", expand=True, padx=24, pady=(0, 22))

        ctk.CTkLabel(
            self.pred_result_frame,
            text="Select a horizon above and click  ▶  RUN PREDICTION",
            font=ctk.CTkFont(size=14),
            text_color=C_FAINT,
        ).pack(expand=True)

    def run_prediction(self) -> None:
        if self.data is None:
            messagebox.showerror(
                "Missing Data",
                "Data is not loaded. Please check sonelgaz_consumption_data.csv.",
            )
            return

        if not hasattr(self, "pred_result_frame") or self.pred_result_frame is None:
            messagebox.showerror(
                "UI Error",
                "Open Forecast Center first, then click Run Prediction.",
            )
            return

        try:
            pred_frame = self.pred_result_frame
            for w in pred_frame.winfo_children():
                w.destroy()

            horizon = self.horizon_var.get()
            selected_model_name = self.horizon_model_vars[horizon].get()
            X_input = self.scaler_X.transform(self.data.tail(24)[FEATURES])
            model, accent, is_seq = self._get_model_bundle(selected_model_name)
            model_name = selected_model_name
            steps = {"Daily": 24, "Weekly": 168, "Monthly": 720, "Quarterly": 2160}[horizon]

            if is_seq:
                pred_scaled = model.predict(X_input.reshape(1, 24, len(FEATURES)), verbose=0)
            else:
                pred_scaled = model.predict(X_input[-1:])

            pred = float(self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0])

            # ── Result header ──
            hdr = ctk.CTkFrame(pred_frame, fg_color=C_BG, corner_radius=12)
            hdr.pack(fill="x", padx=14, pady=(14, 8))

            left = ctk.CTkFrame(hdr, fg_color="transparent")
            left.pack(side="left", padx=18, pady=16)
            ctk.CTkLabel(
                left, text="PREDICTED  NEXT-HOUR  LOAD",
                font=ctk.CTkFont(size=10, weight="bold"), text_color=C_MUTED,
            ).pack(anchor="w")
            val_row = ctk.CTkFrame(left, fg_color="transparent")
            val_row.pack(anchor="w", pady=(4, 0))
            ctk.CTkLabel(
                val_row, text=f"{pred:,.2f}",
                font=ctk.CTkFont(size=42, weight="bold"), text_color=accent,
            ).pack(side="left")
            ctk.CTkLabel(
                val_row, text=" kWh",
                font=ctk.CTkFont(size=16), text_color=C_MUTED,
            ).pack(side="left", anchor="s", pady=(0, 8))

            right = ctk.CTkFrame(hdr, fg_color="transparent")
            right.pack(side="right", padx=18, pady=16)
            self._model_badge(right, model_name, accent).pack(anchor="e")
            ctk.CTkLabel(
                right, text=f"Horizon: {horizon}", font=ctk.CTkFont(size=11), text_color=C_MUTED
            ).pack(anchor="e", pady=(6, 0))

            # ── Forecast summary + chart ──
            steps_clamped = min(steps, 720)
            x_full = np.arange(steps_clamped)
            future_full = np.array([
                pred * (1 + 0.08 * np.sin(i / 12) + 0.015 * np.cos(i / 7)) for i in x_full
            ])

            max_points = 220
            stride = max(1, math.ceil(len(x_full) / max_points))
            x = x_full[::stride]
            future = future_full[::stride]

            stats_row = ctk.CTkFrame(pred_frame, fg_color="transparent")
            stats_row.pack(fill="x", padx=10, pady=(0, 6))
            stats_row.grid_columnconfigure((0, 1, 2), weight=1)
            self._kpi_card(
                stats_row, "⏱", "DISPLAY WINDOW", f"{steps_clamped}", "h", C_MUTED,
                row=0, column=0, sticky="ew", padx=4, pady=4,
            )
            self._kpi_card(
                stats_row, "📉", "MIN ESTIMATE", f"{future_full.min():,.1f}", "kWh", C_TEAL,
                row=0, column=1, sticky="ew", padx=4, pady=4,
            )
            self._kpi_card(
                stats_row, "📈", "MAX ESTIMATE", f"{future_full.max():,.1f}", "kWh", C_ORANGE,
                row=0, column=2, sticky="ew", padx=4, pady=4,
            )

            chart_container = self._chart_panel(
                pred_frame,
                title=f"{horizon} Forecast  ·  {model_name}",
                subtitle=f"displaying {len(x):,} points (downsampled from {steps_clamped:,})",
                fill="both",
                expand=True,
                padx=10,
                pady=(0, 10),
            )

            figure_width = max(10.0, min(28.0, 10.0 + steps_clamped / 70.0))
            fig, ax = plt.subplots(figsize=(figure_width, 3.4))
            ax.fill_between(x, future, alpha=0.18, color=accent)
            ax.plot(x, future, color=accent, linewidth=2.2, label=f"{horizon} projection")
            ax.axhline(pred, color=C_BLUE, linestyle="--", linewidth=1.4, label="Baseline")
            ax.set_xlabel("Hours from now")
            ax.set_ylabel("Consumption (kWh)")
            ax.legend(loc="upper right")
            ax.grid(True)

            tick_count = 7
            tick_positions = np.linspace(0, max(1, len(x) - 1), num=tick_count, dtype=int)
            tick_labels = [f"{int(x[pos])}h" for pos in tick_positions]
            ax.set_xticks(x[tick_positions])
            ax.set_xticklabels(tick_labels)

            fig.tight_layout(pad=1.2)

            chart_viewport = ctk.CTkFrame(chart_container, fg_color="transparent")
            chart_viewport.pack(fill="both", expand=True, padx=10, pady=10)

            x_scroll = ctk.CTkScrollbar(chart_viewport, orientation="horizontal")
            x_scroll.pack(side="bottom", fill="x")

            chart_canvas = tk.Canvas(
                chart_viewport,
                bg=C_SURFACE2,
                highlightthickness=0,
                xscrollcommand=x_scroll.set,
            )
            chart_canvas.pack(side="top", fill="both", expand=True)
            x_scroll.configure(command=chart_canvas.xview)

            # Plain tk.Frame: more reliable inside tk.Canvas than CTk widgets.
            chart_inner = tk.Frame(chart_canvas, bg=C_SURFACE2)
            canvas_window = chart_canvas.create_window((0, 0), window=chart_inner, anchor="nw")

            cv = FigureCanvasTkAgg(fig, master=chart_inner)
            cv.draw()
            plot_widget = cv.get_tk_widget()
            plot_widget.pack(side="left")

            def _refresh_scroll_region(_event=None):
                chart_canvas.update_idletasks()
                chart_canvas.configure(scrollregion=chart_canvas.bbox("all"))
                iw = max(chart_canvas.winfo_width(), chart_inner.winfo_reqwidth())
                chart_canvas.itemconfigure(canvas_window, width=iw)

            def _on_canvas_configure(event):
                chart_canvas.itemconfigure(canvas_window, height=event.height)
                _refresh_scroll_region()

            chart_inner.bind("<Configure>", lambda _e: _refresh_scroll_region())
            chart_canvas.bind("<Configure>", _on_canvas_configure)
            _refresh_scroll_region()

        except Exception as exc:
            print("[run_prediction]", traceback.format_exc())
            messagebox.showerror("Prediction failed", f"{exc}\n\nSee terminal for traceback.")
            try:
                ctk.CTkLabel(
                    self.pred_result_frame,
                    text=f"Prediction failed: {exc}",
                    text_color=C_DANGER,
                    font=ctk.CTkFont(size=14, weight="bold"),
                ).pack(pady=20)
            except Exception:
                pass

    # ── Telemetry Data ─────────────────────────────────────────────────────────

    def show_data(self) -> None:
        self.clear_main_frame()
        self._page_header("Telemetry Data", "Interactive telemetry console for quick inspection and validation.")

        if self.data is None:
            ctk.CTkLabel(self.main_frame, text="⚠  Data unavailable.", text_color=C_DANGER).pack(pady=32)
            return

        # Top KPI strip.
        kpi_wrap = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        kpi_wrap.pack(fill="x", padx=24, pady=(4, 8))
        kpi_wrap.grid_columnconfigure((0, 1, 2, 3), weight=1)

        tail_24 = self.data.tail(24)
        self._kpi_card(kpi_wrap, "⏱", "LATEST HOUR", f"{int(tail_24.iloc[-1]['Hour'])}", "h", C_MUTED, row=0, column=0, sticky="ew", padx=4, pady=4)
        self._kpi_card(kpi_wrap, "🌡", "AVG TEMP (24h)", f"{tail_24['Temperature'].mean():.1f}", "°C", C_TEAL, row=0, column=1, sticky="ew", padx=4, pady=4)
        self._kpi_card(kpi_wrap, "⚡", "AVG CURRENT (24h)", f"{tail_24['Current'].mean():.1f}", "A", C_BLUE, row=0, column=2, sticky="ew", padx=4, pady=4)
        self._kpi_card(kpi_wrap, "📊", "AVG LOAD (24h)", f"{tail_24['Consumption'].mean():.1f}", "kWh", C_ORANGE, row=0, column=3, sticky="ew", padx=4, pady=4)

        wrap = ctk.CTkFrame(
            self.main_frame,
            fg_color=C_SURFACE2,
            corner_radius=14,
            border_color=C_ORANGE, border_width=1,
        )
        wrap.pack(fill="both", expand=True, padx=24, pady=(4, 22))

        # Control row.
        controls = ctk.CTkFrame(wrap, fg_color=C_SURFACE, corner_radius=10)
        controls.pack(fill="x", padx=10, pady=(10, 8))
        ctk.CTkLabel(
            controls,
            text="Rows",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=C_MUTED,
        ).pack(side="left", padx=(10, 6), pady=8)

        rows_var = tk.StringVar(value="120")
        ctk.CTkOptionMenu(
            controls,
            variable=rows_var,
            values=["40", "80", "120", "240", "480"],
            fg_color=C_SURFACE2,
            button_color=C_BLUE,
            button_hover_color=C_BLUE_H,
            text_color=C_TEXT,
            width=100,
        ).pack(side="left", padx=(0, 12), pady=8)

        ctk.CTkLabel(
            controls,
            text="Search timestamp (YYYY-MM-DD)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=C_MUTED,
        ).pack(side="left", padx=(4, 6), pady=8)

        search_var = tk.StringVar(value="")
        search_entry = ctk.CTkEntry(
            controls,
            textvariable=search_var,
            width=220,
            placeholder_text="e.g. 2025-03",
            fg_color=C_BG,
            text_color=C_TEXT,
            border_color=C_BORDER,
        )
        search_entry.pack(side="left", padx=(0, 8), pady=8)

        table_host = ctk.CTkFrame(wrap, fg_color=C_BG, corner_radius=10)
        table_host.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Telemetry.Treeview",
            background=C_BG,
            foreground=C_TEXT,
            fieldbackground=C_BG,
            rowheight=28,
            borderwidth=0,
        )
        style.configure(
            "Telemetry.Treeview.Heading",
            background=C_SURFACE,
            foreground=C_TEXT,
            relief="flat",
            font=("Segoe UI", 10, "bold"),
        )
        style.map("Telemetry.Treeview", background=[("selected", C_BLUE_H)], foreground=[("selected", C_TEXT)])

        columns = ("Timestamp", "Hour", "Temperature", "Current", "Consumption", "Holiday")
        tree = ttk.Treeview(
            table_host,
            columns=columns,
            show="headings",
            style="Telemetry.Treeview",
        )
        tree.pack(side="left", fill="both", expand=True)

        v_scroll = ctk.CTkScrollbar(table_host, orientation="vertical", command=tree.yview)
        v_scroll.pack(side="right", fill="y")
        h_scroll = ctk.CTkScrollbar(wrap, orientation="horizontal", command=tree.xview)
        h_scroll.pack(fill="x", padx=10, pady=(0, 10))
        tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        headings = {
            "Timestamp": "Timestamp",
            "Hour": "Hour",
            "Temperature": "Temperature (°C)",
            "Current": "Current (A)",
            "Consumption": "Consumption (kWh)",
            "Holiday": "Holiday",
        }
        widths = {"Timestamp": 220, "Hour": 80, "Temperature": 140, "Current": 140, "Consumption": 170, "Holiday": 90}
        for col in columns:
            tree.heading(col, text=headings[col])
            tree.column(col, width=widths[col], anchor="center", stretch=True)

        def refresh_table(*_args):
            tree.delete(*tree.get_children())
            rows_to_show = int(rows_var.get())
            df_view = self.data.tail(rows_to_show).copy()
            token = search_var.get().strip()
            if token:
                mask = df_view["Timestamp"].astype(str).str.contains(token, case=False, regex=False)
                df_view = df_view[mask]

            for _, row in df_view.iterrows():
                tree.insert(
                    "",
                    "end",
                    values=(
                        str(row["Timestamp"])[:19],
                        int(row["Hour"]),
                        f"{row['Temperature']:.1f}",
                        f"{row['Current']:.1f}",
                        f"{row['Consumption']:.2f}",
                        int(row["IsHoliday"]),
                    ),
                )

        rows_var.trace_add("write", refresh_table)
        search_var.trace_add("write", refresh_table)
        refresh_table()


if __name__ == "__main__":
    app = SonelgazApp()
    app.mainloop()
