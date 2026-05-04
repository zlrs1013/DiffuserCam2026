"""Small visualization helpers for saved simulation outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont


FloatArray = NDArray[np.floating]


def normalize_for_display(image: FloatArray, percentile_clip: tuple[float, float] = (1.0, 99.0)) -> NDArray[np.uint8]:
    """Map an array to uint8 using robust percentile clipping."""

    lo, hi = np.percentile(image, percentile_clip)
    if hi <= lo:
        hi = lo + 1e-9
    scaled = (image - lo) / (hi - lo)
    return np.uint8(np.clip(scaled, 0.0, 1.0) * 255.0)


def add_title(tile: Image.Image, title: str, height: int = 26) -> Image.Image:
    """Add a compact title strip above an image tile."""

    out = Image.new("RGB", (tile.width, tile.height + height), "white")
    out.paste(tile.convert("RGB"), (0, height))
    draw = ImageDraw.Draw(out)
    draw.text((8, 6), title, fill=(20, 20, 20), font=ImageFont.load_default())
    return out


def save_montage(images: list[tuple[str, FloatArray]], path: Path, columns: int = 3) -> None:
    """Save a labeled montage."""

    tiles = []
    for title, array in images:
        if title.lower().startswith("psf"):
            display = normalize_for_display(np.log1p(array / max(float(np.max(array)), 1e-12)))
        else:
            display = normalize_for_display(array)
        tiles.append(add_title(Image.fromarray(display, mode="L"), title))

    rows = int(np.ceil(len(tiles) / columns))
    tile_width, tile_height = tiles[0].size
    montage = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    for index, tile in enumerate(tiles):
        row, col = divmod(index, columns)
        montage.paste(tile, (col * tile_width, row * tile_height))

    montage.save(path)


def save_convergence_plot(histories: list[tuple[str, list[object]]], path: Path) -> None:
    """Save a compact objective/PSNR convergence plot using Pillow only."""

    width, height = 900, 360
    margin_left, margin_right = 58, 18
    margin_top, margin_bottom = 34, 42
    panel_gap = 42
    panel_width = (width - margin_left - margin_right - panel_gap) // 2
    panel_height = height - margin_top - margin_bottom
    colors = [
        (31, 119, 180),
        (214, 39, 40),
        (44, 160, 44),
        (148, 103, 189),
        (255, 127, 14),
        (23, 190, 207),
    ]
    font = ImageFont.load_default()

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def record_value(record: object, field_name: str) -> float:
        return float(getattr(record, field_name))

    def draw_panel(origin_x: int, title: str, field_name: str, log10_y: bool) -> None:
        all_iterations = []
        all_values = []
        for _, records in histories:
            all_iterations.extend(record_value(record, "iteration") for record in records)
            for record in records:
                value = record_value(record, field_name)
                if log10_y:
                    value = float(np.log10(max(value, 1e-12)))
                all_values.append(value)

        x_min, x_max = min(all_iterations), max(all_iterations)
        y_min, y_max = min(all_values), max(all_values)
        if y_max <= y_min:
            y_max = y_min + 1.0

        left = origin_x
        right = origin_x + panel_width
        top = margin_top
        bottom = margin_top + panel_height

        draw.rectangle((left, top, right, bottom), outline=(60, 60, 60), width=1)
        draw.text((left, 10), title, fill=(20, 20, 20), font=font)
        draw.text((left, bottom + 18), "iteration", fill=(20, 20, 20), font=font)
        draw.text((left, bottom + 4), f"{y_min:.2f}", fill=(80, 80, 80), font=font)
        draw.text((left, top - 14), f"{y_max:.2f}", fill=(80, 80, 80), font=font)

        def map_point(iteration: float, value: float) -> tuple[int, int]:
            x_frac = 0.0 if x_max == x_min else (iteration - x_min) / (x_max - x_min)
            y_frac = (value - y_min) / (y_max - y_min)
            x = int(left + x_frac * panel_width)
            y = int(bottom - y_frac * panel_height)
            return x, y

        for index, (label, records) in enumerate(histories):
            points = []
            for record in records:
                value = record_value(record, field_name)
                if log10_y:
                    value = float(np.log10(max(value, 1e-12)))
                points.append(map_point(record_value(record, "iteration"), value))
            if len(points) > 1:
                draw.line(points, fill=colors[index % len(colors)], width=2)
            draw.text(
                (left + 8, top + 10 + 14 * index),
                label,
                fill=colors[index % len(colors)],
                font=font,
            )

    draw_panel(margin_left, "log10 objective", "objective", log10_y=True)
    draw_panel(margin_left + panel_width + panel_gap, "PSNR (dB)", "psnr_db", log10_y=False)
    image.save(path)


def save_objective_residual_plot(histories: list[tuple[str, list[object]]], path: Path) -> None:
    """Save objective and residual convergence for data without ground truth."""

    width, height = 900, 360
    margin_left, margin_right = 58, 18
    margin_top, margin_bottom = 34, 42
    panel_gap = 42
    panel_width = (width - margin_left - margin_right - panel_gap) // 2
    panel_height = height - margin_top - margin_bottom
    colors = [
        (31, 119, 180),
        (214, 39, 40),
        (44, 160, 44),
        (148, 103, 189),
        (255, 127, 14),
        (23, 190, 207),
    ]
    font = ImageFont.load_default()
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def record_value(record: object, field_name: str) -> float:
        return float(getattr(record, field_name))

    def draw_panel(origin_x: int, title: str, field_name: str, log10_y: bool) -> None:
        all_iterations = []
        all_values = []
        for _, records in histories:
            all_iterations.extend(record_value(record, "iteration") for record in records)
            for record in records:
                value = record_value(record, field_name)
                if log10_y:
                    value = float(np.log10(max(value, 1e-12)))
                all_values.append(value)

        x_min, x_max = min(all_iterations), max(all_iterations)
        y_min, y_max = min(all_values), max(all_values)
        if y_max <= y_min:
            y_max = y_min + 1.0

        left = origin_x
        right = origin_x + panel_width
        top = margin_top
        bottom = margin_top + panel_height
        draw.rectangle((left, top, right, bottom), outline=(60, 60, 60), width=1)
        draw.text((left, 10), title, fill=(20, 20, 20), font=font)
        draw.text((left, bottom + 18), "iteration", fill=(20, 20, 20), font=font)
        draw.text((left, bottom + 4), f"{y_min:.3g}", fill=(80, 80, 80), font=font)
        draw.text((left, top - 14), f"{y_max:.3g}", fill=(80, 80, 80), font=font)

        def map_point(iteration: float, value: float) -> tuple[int, int]:
            x_frac = 0.0 if x_max == x_min else (iteration - x_min) / (x_max - x_min)
            y_frac = (value - y_min) / (y_max - y_min)
            x = int(left + x_frac * panel_width)
            y = int(bottom - y_frac * panel_height)
            return x, y

        for index, (label, records) in enumerate(histories):
            points = []
            for record in records:
                value = record_value(record, field_name)
                if log10_y:
                    value = float(np.log10(max(value, 1e-12)))
                points.append(map_point(record_value(record, "iteration"), value))
            if len(points) > 1:
                draw.line(points, fill=colors[index % len(colors)], width=2)
            draw.text(
                (left + 8, top + 10 + 14 * index),
                label,
                fill=colors[index % len(colors)],
                font=font,
            )

    draw_panel(margin_left, "log10 objective", "objective", log10_y=True)
    draw_panel(
        margin_left + panel_width + panel_gap,
        "relative residual",
        "relative_residual_l2",
        log10_y=False,
    )
    image.save(path)


def save_two_panel_metric_plot(
    *,
    x_values: list[float],
    left_series: list[tuple[str, list[float]]],
    right_series: list[tuple[str, list[float]]],
    path: Path,
    left_title: str,
    right_title: str,
    x_label: str,
    left_y_label: str,
    right_y_label: str,
    log_x: bool = False,
    log_left_y: bool = False,
    log_right_y: bool = False,
) -> None:
    """Save a polished two-panel line plot for metric sweeps."""

    width, height = 980, 400
    margin_left, margin_right = 70, 24
    margin_top, margin_bottom = 42, 58
    panel_gap = 54
    panel_width = (width - margin_left - margin_right - panel_gap) // 2
    panel_height = height - margin_top - margin_bottom
    colors = [
        (31, 119, 180),
        (214, 39, 40),
        (44, 160, 44),
        (148, 103, 189),
        (255, 127, 14),
        (23, 190, 207),
    ]
    font = ImageFont.load_default()
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def transform(values: list[float], use_log: bool) -> list[float]:
        if use_log:
            return [float(np.log10(max(value, 1e-18))) for value in values]
        return [float(value) for value in values]

    x_plot_values = transform(x_values, log_x)

    def draw_panel(
        *,
        origin_x: int,
        title: str,
        y_label: str,
        series: list[tuple[str, list[float]]],
        log_y: bool,
    ) -> None:
        transformed_series = [(label, transform(values, log_y)) for label, values in series]
        all_y = [value for _, values in transformed_series for value in values]

        x_min, x_max = min(x_plot_values), max(x_plot_values)
        y_min, y_max = min(all_y), max(all_y)
        if x_max <= x_min:
            x_max = x_min + 1.0
        if y_max <= y_min:
            y_max = y_min + 1.0

        x_padding = 0.05 * (x_max - x_min)
        y_padding = 0.08 * (y_max - y_min)
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        left = origin_x
        right = origin_x + panel_width
        top = margin_top
        bottom = margin_top + panel_height
        draw.rectangle((left, top, right, bottom), outline=(70, 70, 70), width=1)
        draw.text((left, 14), title, fill=(15, 15, 15), font=font)
        draw.text((left, bottom + 28), x_label, fill=(30, 30, 30), font=font)
        draw.text((left, top - 12), y_label, fill=(30, 30, 30), font=font)

        draw.text((left, bottom + 4), f"{x_values[0]:.1e}" if log_x else f"{x_values[0]:.3g}", fill=(80, 80, 80), font=font)
        draw.text((right - 46, bottom + 4), f"{x_values[-1]:.1e}" if log_x else f"{x_values[-1]:.3g}", fill=(80, 80, 80), font=font)
        draw.text((left + 3, bottom - 14), f"{y_min:.3g}", fill=(90, 90, 90), font=font)
        draw.text((left + 3, top + 3), f"{y_max:.3g}", fill=(90, 90, 90), font=font)

        def map_point(x_value: float, y_value: float) -> tuple[int, int]:
            x_frac = (x_value - x_min) / (x_max - x_min)
            y_frac = (y_value - y_min) / (y_max - y_min)
            return int(left + x_frac * panel_width), int(bottom - y_frac * panel_height)

        for index, (label, values) in enumerate(transformed_series):
            points = [map_point(x_value, y_value) for x_value, y_value in zip(x_plot_values, values)]
            if len(points) > 1:
                draw.line(points, fill=colors[index % len(colors)], width=2)
            for point in points:
                x, y = point
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=colors[index % len(colors)])
            draw.text(
                (left + 8, top + 12 + 14 * index),
                label,
                fill=colors[index % len(colors)],
                font=font,
            )

    draw_panel(
        origin_x=margin_left,
        title=left_title,
        y_label=left_y_label,
        series=left_series,
        log_y=log_left_y,
    )
    draw_panel(
        origin_x=margin_left + panel_width + panel_gap,
        title=right_title,
        y_label=right_y_label,
        series=right_series,
        log_y=log_right_y,
    )
    image.save(path)


def save_table_image(
    *,
    rows: list[dict[str, object]],
    columns: list[tuple[str, str]],
    path: Path,
    title: str,
) -> None:
    """Render a small table as an image.

    ``columns`` is a list of ``(key, label)`` pairs.
    """

    font = ImageFont.load_default()
    title_height = 30
    row_height = 24
    padding_x = 12
    col_widths = []
    for key, label in columns:
        max_chars = len(label)
        for row in rows:
            max_chars = max(max_chars, len(str(row.get(key, ""))))
        col_widths.append(max(86, 7 * max_chars + 2 * padding_x))

    width = int(sum(col_widths))
    height = title_height + row_height * (len(rows) + 1) + 8
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((padding_x, 9), title, fill=(15, 15, 15), font=font)

    y = title_height
    x = 0
    for (key, label), col_width in zip(columns, col_widths):
        draw.rectangle((x, y, x + col_width, y + row_height), fill=(235, 238, 242), outline=(180, 180, 180))
        draw.text((x + padding_x, y + 7), label, fill=(20, 20, 20), font=font)
        x += col_width

    for row_index, row in enumerate(rows):
        y = title_height + row_height * (row_index + 1)
        fill = (255, 255, 255) if row_index % 2 == 0 else (248, 248, 248)
        x = 0
        for (key, _), col_width in zip(columns, col_widths):
            draw.rectangle((x, y, x + col_width, y + row_height), fill=fill, outline=(220, 220, 220))
            draw.text((x + padding_x, y + 7), str(row.get(key, "")), fill=(35, 35, 35), font=font)
            x += col_width

    image.save(path)
