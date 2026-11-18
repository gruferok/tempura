"""Small explicit chart-catalog editor for the AI Studio dataset."""
from __future__ import annotations
from pathlib import Path
import pygame
from .chart import Chart
from .storage import CHART_DIR
from .ui import App, CYAN, MUTED, RED, TEXT


def _is_enabled(chart: Chart) -> bool:
    metadata = chart.metadata
    return metadata.get("include_in_training") is True or (
        "include_in_training" not in metadata and not metadata.get("auto_quantized")
        and not metadata.get("ai_generated"))


def run_dataset_manager(app: App, charts_dir=CHART_DIR) -> None:
    directory = Path(charts_dir)
    while app.running:
        entries = []
        for path in sorted(directory.glob("*.json")):
            try:
                entries.append((path, Chart.load(path)))
            except (OSError, ValueError, TypeError):
                continue
        selected = min(getattr(run_dataset_manager, "selected", 0), max(0, len(entries) - 1))
        run_dataset_manager.selected = selected
        app.begin("AI Dataset Manager", "Выберите карту; T переключает include_in_training, изменения атомарны")
        for i, (path, chart) in enumerate(entries[:14]):
            enabled = _is_enabled(chart)
            color = CYAN if enabled else MUTED
            app.fit_text(("> " if i == selected else "  ") + path.name + ("  [IN]" if enabled else "  [OUT]"),
                         48, 165 + i * 32, app.width - 96, 16, color)
        if not entries:
            app.text("В data/charts пока нет карт.", 48, 180, 18, MUTED)
        app.fit_text("AI-карты требуют human_reviewed=true; Beat-Snap можно включить явным IN.",
                     48, 635, app.width - 96, 14, RED)
        app.footer("↑/↓ выбрать   T — включить/исключить   Esc — назад")
        for event in app.events():
            if event.type != pygame.KEYDOWN or getattr(event, "repeat", False):
                continue
            if event.key == pygame.K_ESCAPE:
                return
            if event.key == pygame.K_UP and entries:
                selected = (selected - 1) % len(entries)
            elif event.key == pygame.K_DOWN and entries:
                selected = (selected + 1) % len(entries)
            elif event.key == pygame.K_t and entries:
                path, chart = entries[selected]
                chart.metadata["include_in_training"] = not _is_enabled(chart)
                chart.save(path)
                app.toast = f"{path.name}: include_in_training={chart.metadata['include_in_training']}"
            run_dataset_manager.selected = selected
        app.frame(fast=False)
