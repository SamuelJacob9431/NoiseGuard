import sys
from dataclasses import dataclass
from typing import Optional, List

import torch

from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QPixmap, QFont, QColor, QImage
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QStatusBar,
    QProgressBar,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib

from src.phase1.device import get_device
from src.phase1.vae import (
    load_vae,
    freeze_vae,
    encode_image,
    decode_latent,
)
from src.phase1.image_utils import load_image
from src.phase1.output_utils import reconstruction_error
from src.phase2.attack import run_attack


matplotlib.rcParams["font.family"] = "sans-serif"


# ==========================================================
# THEME
# ==========================================================

class Theme:
    BG = "#0f1115"
    PANEL = "#171a21"
    PANEL_ALT = "#1e222b"
    BORDER = "#2a2f3a"
    ACCENT = "#6c8cff"
    ACCENT_HOVER = "#8aa4ff"
    SUCCESS = "#4ade80"
    WARN = "#fbbf24"
    DANGER = "#f87171"
    TEXT_PRIMARY = "#e8eaf0"
    TEXT_SECOND = "#9aa3b5"
    TEXT_MUTED = "#5c6479"


STYLESHEET = f"""
QMainWindow {{
    background-color: {Theme.BG};
}}

QWidget {{
    color: {Theme.TEXT_PRIMARY};
    font-family: 'Segoe UI', sans-serif;
}}

QLabel#appTitle {{
    font-size: 20px;
    font-weight: 700;
    color: {Theme.TEXT_PRIMARY};
}}

QLabel#appSubtitle {{
    font-size: 11px;
    color: {Theme.TEXT_MUTED};
    letter-spacing: 1px;
}}

QFrame#panel {{
    background-color: {Theme.PANEL};
    border: 1px solid {Theme.BORDER};
    border-radius: 14px;
}}

QFrame#imageFrame {{
    background-color: {Theme.PANEL_ALT};
    border: 1px solid {Theme.BORDER};
    border-radius: 10px;
}}

QLabel#imageCaption {{
    font-size: 11px;
    font-weight: 600;
    color: {Theme.TEXT_SECOND};
    letter-spacing: 1.5px;
}}

QLabel#imagePlaceholder {{
    color: {Theme.TEXT_MUTED};
    font-size: 12px;
}}

QLabel#sectionTitle {{
    font-size: 12px;
    font-weight: 700;
    color: {Theme.TEXT_SECOND};
    letter-spacing: 1.5px;
}}

QLabel#statValue {{
    font-size: 18px;
    font-weight: 700;
    color: {Theme.TEXT_PRIMARY};
}}

QLabel#statLabel {{
    font-size: 10px;
    color: {Theme.TEXT_MUTED};
}}

QPushButton#primaryBtn {{
    background-color: {Theme.ACCENT};
    color: #0b0d12;
    font-weight: 700;
    font-size: 13px;
    border: none;
    border-radius: 10px;
    padding: 11px 22px;
}}

QPushButton#primaryBtn:hover {{
    background-color: {Theme.ACCENT_HOVER};
}}

QPushButton#primaryBtn:disabled {{
    background-color: {Theme.BORDER};
    color: {Theme.TEXT_MUTED};
}}

QPushButton#secondaryBtn {{
    background-color: transparent;
    color: {Theme.TEXT_PRIMARY};
    font-weight: 600;
    font-size: 13px;
    border: 1px solid {Theme.BORDER};
    border-radius: 10px;
    padding: 11px 22px;
}}

QPushButton#secondaryBtn:hover {{
    background-color: {Theme.PANEL_ALT};
    border: 1px solid {Theme.ACCENT};
}}

QStatusBar {{
    background-color: {Theme.PANEL};
    color: {Theme.TEXT_SECOND};
    border-top: 1px solid {Theme.BORDER};
    font-size: 11px;
    padding: 4px 10px;
}}

QProgressBar {{
    background-color: {Theme.PANEL_ALT};
    border: 1px solid {Theme.BORDER};
    border-radius: 6px;
    height: 8px;
}}

QProgressBar::chunk {{
    background-color: {Theme.ACCENT};
    border-radius: 6px;
}}
"""


# ==========================================================
# PIPELINE RESULT
# ==========================================================

@dataclass
class PipelineResult:
    reconstruction_pixmap: Optional[QPixmap] = None
    protected_pixmap: Optional[QPixmap] = None

    latent_mean: Optional[float] = None
    latent_std: Optional[float] = None
    latent_shape: Optional[str] = None

    reconstruction_mse: Optional[float] = None
    final_latent_distance: Optional[float] = None

    pgd_loss_curve: Optional[List[float]] = None


# ==========================================================
# TENSOR -> QPIXMAP
# ==========================================================

def tensor_to_pixmap(tensor: torch.Tensor) -> QPixmap:

    image = tensor.detach().float().cpu()

    if image.dim() == 4:
        image = image[0]

    image = image.clamp(-1.0, 1.0)

    image = (image + 1.0) / 2.0
    image = image.mul(255).byte()

    image = image.permute(1, 2, 0).contiguous()

    height, width, channels = image.shape

    data = image.numpy()

    qimage = QImage(
        data,
        width,
        height,
        channels * width,
        QImage.Format_RGB888,
    )

    return QPixmap.fromImage(qimage.copy())


# ==========================================================
# REAL PIPELINE
# ==========================================================

def run_real_pipeline(image_path: str) -> PipelineResult:

    device = get_device()

    vae = load_vae(device)
    freeze_vae(vae)

    image = load_image(
        image_path,
        device
    )

    # ------------------------------------------------------
    # Encode
    # ------------------------------------------------------

    latent = encode_image(
        vae,
        image
    )

    latent_mean = latent.mean().item()
    latent_std = latent.std().item()
    latent_shape = str(tuple(latent.shape))

    # ------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------

    reconstructed = decode_latent(
        vae,
        latent
    )

    reconstruction_mse = reconstruction_error(
        image,
        reconstructed
    )

    # ------------------------------------------------------
    # PGD
    # ------------------------------------------------------

    (
        protected_image,
        loss_history,
        original_latent,
        protected_latent,
        final_latent_distance,
    ) = run_attack(
        vae,
        image,
        epsilon=8 / 255,
        alpha=2 / 255,
        steps=50,
    )

    # ------------------------------------------------------
    # Result
    # ------------------------------------------------------

    return PipelineResult(
        reconstruction_pixmap=tensor_to_pixmap(
            reconstructed
        ),

        protected_pixmap=tensor_to_pixmap(
            protected_image
        ),

        latent_mean=latent_mean,
        latent_std=latent_std,
        latent_shape=latent_shape,

        reconstruction_mse=reconstruction_mse,

        final_latent_distance=final_latent_distance,

        pgd_loss_curve=loss_history,
    )


# ==========================================================
# WORKER
# ==========================================================

class PipelineWorker(QObject):

    finished = Signal(object)
    progress = Signal(int, str)
    failed = Signal(str)

    def __init__(self, image_path: str):
        super().__init__()
        self.image_path = image_path

    def run(self):

        try:

            self.progress.emit(
                5,
                "Loading VAE..."
            )

            self.progress.emit(
                15,
                "Encoding image..."
            )

            self.progress.emit(
                30,
                "Creating reconstruction..."
            )

            self.progress.emit(
                40,
                "Running PGD attack..."
            )

            result = run_real_pipeline(
                self.image_path
            )

            self.progress.emit(
                100,
                "Pipeline complete"
            )

            self.finished.emit(result)

        except Exception as exc:

            self.failed.emit(
                f"{type(exc).__name__}: {exc}"
            )


# ==========================================================
# IMAGE PANEL
# ==========================================================

class ImagePanel(QFrame):

    def __init__(
        self,
        caption: str,
        accent: Optional[str] = None
    ):

        super().__init__()

        self.setObjectName("panel")

        self._accent = accent or Theme.ACCENT

        outer = QVBoxLayout(self)

        outer.setContentsMargins(
            16,
            16,
            16,
            16
        )

        outer.setSpacing(10)

        cap_row = QHBoxLayout()

        dot = QLabel("●")

        dot.setStyleSheet(
            f"color: {self._accent};"
        )

        label = QLabel(
            caption.upper()
        )

        label.setObjectName(
            "imageCaption"
        )

        cap_row.addWidget(dot)
        cap_row.addWidget(label)
        cap_row.addStretch()

        outer.addLayout(cap_row)

        self.image_frame = QFrame()

        self.image_frame.setObjectName(
            "imageFrame"
        )

        self.image_frame.setMinimumSize(
            260,
            260
        )

        img_layout = QVBoxLayout(
            self.image_frame
        )

        img_layout.setContentsMargins(
            0,
            0,
            0,
            0
        )

        self.image_label = QLabel(
            "No image loaded"
        )

        self.image_label.setObjectName(
            "imagePlaceholder"
        )

        self.image_label.setAlignment(
            Qt.AlignCenter
        )

        img_layout.addWidget(
            self.image_label
        )

        outer.addWidget(
            self.image_frame
        )

    def set_pixmap(
        self,
        pixmap: Optional[QPixmap]
    ):

        if (
            pixmap is None
            or pixmap.isNull()
        ):

            self.image_label.setText(
                "No image loaded"
            )

            return

        scaled = pixmap.scaled(
            self.image_frame.width() - 20,
            self.image_frame.height() - 20,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.image_label.setPixmap(
            scaled
        )

    def set_pending(self):

        self.image_label.setText(
            "Processing…"
        )

        self.image_label.setStyleSheet(
            f"color: {Theme.WARN};"
            "font-size: 12px;"
        )


# ==========================================================
# STAT TILE
# ==========================================================

class StatTile(QFrame):

    def __init__(self, label: str):

        super().__init__()

        self.setObjectName(
            "panel"
        )

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            16,
            14,
            16,
            14
        )

        layout.setSpacing(4)

        self.value_label = QLabel(
            "—"
        )

        self.value_label.setObjectName(
            "statValue"
        )

        name_label = QLabel(
            label.upper()
        )

        name_label.setObjectName(
            "statLabel"
        )

        layout.addWidget(
            self.value_label
        )

        layout.addWidget(
            name_label
        )

    def set_value(
        self,
        text: str
    ):

        self.value_label.setText(
            text
        )


# ==========================================================
# LOSS GRAPH
# ==========================================================

class LossCurveCanvas(
    FigureCanvasQTAgg
):

    def __init__(self):

        fig = Figure(
            figsize=(5, 2.4),
            dpi=100
        )

        fig.patch.set_facecolor(
            Theme.PANEL
        )

        super().__init__(
            fig
        )

        self.ax = fig.add_subplot(
            111
        )

        self.fig = fig

        self._style_axes()

        fig.tight_layout(
            pad=1.2
        )

    def _style_axes(self):

        self.ax.set_facecolor(
            Theme.PANEL
        )

        for spine in self.ax.spines.values():

            spine.set_color(
                Theme.BORDER
            )

        self.ax.tick_params(
            colors=Theme.TEXT_MUTED,
            labelsize=8
        )

        self.ax.set_xlabel(
            "PGD Iteration",
            color=Theme.TEXT_SECOND,
            fontsize=9
        )

        self.ax.set_ylabel(
            "Latent Distance",
            color=Theme.TEXT_SECOND,
            fontsize=9
        )

        self.ax.grid(
            True,
            color=Theme.BORDER,
            linewidth=0.6,
            alpha=0.5
        )

    def update_curve(
        self,
        values: Optional[List[float]]
    ):

        self.ax.clear()

        self._style_axes()

        if values:

            xs = list(
                range(len(values))
            )

            self.ax.plot(
                xs,
                values,
                color=Theme.ACCENT,
                linewidth=2.2
            )

            self.ax.fill_between(
                xs,
                values,
                min(values),
                color=Theme.ACCENT,
                alpha=0.08
            )

        else:

            self.ax.text(
                0.5,
                0.5,
                "No run yet",
                transform=self.ax.transAxes,
                ha="center",
                va="center",
                color=Theme.TEXT_MUTED,
                fontsize=10,
            )

        self.fig.tight_layout(
            pad=1.2
        )

        self.draw()


# ==========================================================
# MAIN WINDOW
# ==========================================================

class NoiseGuardWindow(
    QMainWindow
):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "NoiseGuard"
        )

        self.resize(
            1180,
            820
        )

        self.setMinimumSize(
            980,
            720
        )

        self.current_image_path = None

        self._thread = None
        self._worker = None

        self._build_ui()

        self._apply_shadows()

    # ------------------------------------------------------
    # UI
    # ------------------------------------------------------

    def _build_ui(self):

        central = QWidget()

        self.setCentralWidget(
            central
        )

        root = QVBoxLayout(
            central
        )

        root.setContentsMargins(
            24,
            20,
            24,
            12
        )

        root.setSpacing(
            18
        )

        root.addLayout(
            self._build_header()
        )

        root.addLayout(
            self._build_image_row()
        )

        root.addLayout(
            self._build_stats_row()
        )

        root.addWidget(
            self._build_loss_panel()
        )

        self.progress_bar = QProgressBar()

        self.progress_bar.setRange(
            0,
            100
        )

        self.progress_bar.setValue(
            0
        )

        self.progress_bar.setVisible(
            False
        )

        root.addWidget(
            self.progress_bar
        )

        self._build_status_bar()

    def _build_header(self):

        row = QHBoxLayout()

        title_col = QVBoxLayout()

        title = QLabel(
            "NoiseGuard"
        )

        title.setObjectName(
            "appTitle"
        )

        subtitle = QLabel(
            "VAE ENCODER ADVERSARIAL PROTECTION"
        )

        subtitle.setObjectName(
            "appSubtitle"
        )

        title_col.addWidget(
            title
        )

        title_col.addWidget(
            subtitle
        )

        row.addLayout(
            title_col
        )

        row.addStretch()

        self.load_btn = QPushButton(
            "Load Image"
        )

        self.load_btn.setObjectName(
            "secondaryBtn"
        )

        self.load_btn.clicked.connect(
            self.on_load_image
        )

        self.run_btn = QPushButton(
            "Run Pipeline"
        )

        self.run_btn.setObjectName(
            "primaryBtn"
        )

        self.run_btn.setEnabled(
            False
        )

        self.run_btn.clicked.connect(
            self.on_run_pipeline
        )

        row.addWidget(
            self.load_btn
        )

        row.addWidget(
            self.run_btn
        )

        return row

    def _build_image_row(self):

        row = QHBoxLayout()

        row.setSpacing(
            16
        )

        self.panel_original = ImagePanel(
            "Original",
            Theme.TEXT_SECOND
        )

        self.panel_reconstruction = ImagePanel(
            "Reconstruction",
            Theme.WARN
        )

        self.panel_protected = ImagePanel(
            "Protected",
            Theme.SUCCESS
        )

        row.addWidget(
            self.panel_original
        )

        row.addWidget(
            self.panel_reconstruction
        )

        row.addWidget(
            self.panel_protected
        )

        return row

    def _build_stats_row(self):

        row = QHBoxLayout()

        row.setSpacing(
            16
        )

        self.stat_mean = StatTile(
            "Latent Mean"
        )

        self.stat_std = StatTile(
            "Latent Std"
        )

        self.stat_shape = StatTile(
            "Latent Shape"
        )

        self.stat_mse = StatTile(
            "Reconstruction MSE"
        )

        self.stat_dist = StatTile(
            "Final Latent Distance"
        )

        for tile in (
            self.stat_mean,
            self.stat_std,
            self.stat_shape,
            self.stat_mse,
            self.stat_dist,
        ):

            row.addWidget(
                tile
            )

        return row

    def _build_loss_panel(self):

        panel = QFrame()

        panel.setObjectName(
            "panel"
        )

        layout = QVBoxLayout(
            panel
        )

        title = QLabel(
            "PGD LOSS CURVE"
        )

        title.setObjectName(
            "sectionTitle"
        )

        layout.addWidget(
            title
        )

        self.loss_canvas = (
            LossCurveCanvas()
        )

        self.loss_canvas.setMinimumHeight(
            220
        )

        layout.addWidget(
            self.loss_canvas
        )

        return panel

    def _build_status_bar(self):

        bar = QStatusBar()

        self.setStatusBar(
            bar
        )

        self.status_dot = QLabel(
            "●"
        )

        self.status_dot.setStyleSheet(
            f"color: {Theme.TEXT_MUTED};"
        )

        self.status_text = QLabel(
            "Idle — load an image to begin"
        )

        bar.addWidget(
            self.status_dot
        )

        bar.addWidget(
            self.status_text
        )

    def _apply_shadows(self):

        widgets = (
            self.panel_original,
            self.panel_reconstruction,
            self.panel_protected,
            self.stat_mean,
            self.stat_std,
            self.stat_shape,
            self.stat_mse,
            self.stat_dist,
        )

        for widget in widgets:

            self._add_shadow(
                widget
            )

    @staticmethod
    def _add_shadow(widget):

        effect = QGraphicsDropShadowEffect(
            widget
        )

        effect.setBlurRadius(
            24
        )

        effect.setOffset(
            0,
            6
        )

        effect.setColor(
            QColor(
                0,
                0,
                0,
                110
            )
        )

        widget.setGraphicsEffect(
            effect
        )

    # ------------------------------------------------------
    # STATUS
    # ------------------------------------------------------

    def _set_status(
        self,
        text,
        color=Theme.TEXT_MUTED
    ):

        self.status_text.setText(
            text
        )

        self.status_dot.setStyleSheet(
            f"color: {color};"
        )

    # ------------------------------------------------------
    # LOAD IMAGE
    # ------------------------------------------------------

    def on_load_image(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )

        if not path:

            return

        self.current_image_path = path

        pixmap = QPixmap(
            path
        )

        self.panel_original.set_pixmap(
            pixmap
        )

        self.panel_reconstruction.set_pixmap(
            None
        )

        self.panel_protected.set_pixmap(
            None
        )

        self._reset_stats()

        self.loss_canvas.update_curve(
            None
        )

        self.run_btn.setEnabled(
            True
        )

        filename = path.replace(
            "\\",
            "/"
        ).split("/")[-1]

        self._set_status(
            f"Loaded {filename}",
            Theme.SUCCESS
        )

    # ------------------------------------------------------
    # RUN
    # ------------------------------------------------------

    def on_run_pipeline(self):

        if not self.current_image_path:

            return

        self.run_btn.setEnabled(
            False
        )

        self.load_btn.setEnabled(
            False
        )

        self.progress_bar.setVisible(
            True
        )

        self.progress_bar.setValue(
            0
        )

        self.panel_reconstruction.set_pending()
        self.panel_protected.set_pending()

        self._set_status(
            "Starting NoiseGuard pipeline...",
            Theme.WARN
        )

        self._thread = QThread(
            self
        )

        self._worker = PipelineWorker(
            self.current_image_path
        )

        self._worker.moveToThread(
            self._thread
        )

        self._thread.started.connect(
            self._worker.run
        )

        self._worker.progress.connect(
            self._on_progress
        )

        self._worker.finished.connect(
            self._on_pipeline_finished
        )

        self._worker.failed.connect(
            self._on_pipeline_failed
        )

        self._worker.finished.connect(
            self._thread.quit
        )

        self._worker.failed.connect(
            self._thread.quit
        )

        self._thread.finished.connect(
            self._cleanup_thread
        )

        self._thread.start()

    # ------------------------------------------------------
    # PROGRESS
    # ------------------------------------------------------

    def _on_progress(
        self,
        value,
        message
    ):

        self.progress_bar.setValue(
            value
        )

        self._set_status(
            message,
            Theme.WARN
        )

    # ------------------------------------------------------
    # FINISHED
    # ------------------------------------------------------

    def _on_pipeline_finished(
        self,
        result: PipelineResult
    ):

        self.panel_reconstruction.set_pixmap(
            result.reconstruction_pixmap
        )

        self.panel_protected.set_pixmap(
            result.protected_pixmap
        )

        self.stat_mean.set_value(
            self._fmt(
                result.latent_mean
            )
        )

        self.stat_std.set_value(
            self._fmt(
                result.latent_std
            )
        )

        self.stat_shape.set_value(
            result.latent_shape or "—"
        )

        self.stat_mse.set_value(
            self._fmt(
                result.reconstruction_mse
            )
        )

        self.stat_dist.set_value(
            self._fmt(
                result.final_latent_distance
            )
        )

        self.loss_canvas.update_curve(
            result.pgd_loss_curve
        )

        self.progress_bar.setValue(
            100
        )

        self.progress_bar.setVisible(
            False
        )

        self.run_btn.setEnabled(
            True
        )

        self.load_btn.setEnabled(
            True
        )

        self._set_status(
            "NoiseGuard pipeline complete",
            Theme.SUCCESS
        )

    # ------------------------------------------------------
    # ERROR
    # ------------------------------------------------------

    def _on_pipeline_failed(
        self,
        message
    ):

        self.progress_bar.setVisible(
            False
        )

        self.run_btn.setEnabled(
            True
        )

        self.load_btn.setEnabled(
            True
        )

        self._set_status(
            f"Pipeline error: {message}",
            Theme.DANGER
        )

        print(
            "\nPIPELINE ERROR:"
        )

        print(
            message
        )

    # ------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------

    def _cleanup_thread(self):

        self._thread = None
        self._worker = None

    # ------------------------------------------------------
    # RESET
    # ------------------------------------------------------

    def _reset_stats(self):

        for tile in (
            self.stat_mean,
            self.stat_std,
            self.stat_shape,
            self.stat_mse,
            self.stat_dist,
        ):

            tile.set_value(
                "—"
            )

    @staticmethod
    def _fmt(value):

        if value is None:

            return "—"

        if isinstance(
            value,
            float
        ):

            return f"{value:.4f}"

        return str(value)


# ==========================================================
# ENTRY POINT
# ==========================================================

def main():

    app = QApplication(
        sys.argv
    )

    app.setStyleSheet(
        STYLESHEET
    )

    app.setFont(
        QFont(
            "Segoe UI",
            10
        )
    )

    window = NoiseGuardWindow()

    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":

    main()