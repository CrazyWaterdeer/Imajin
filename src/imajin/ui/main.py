from __future__ import annotations

from typing import Any


_SCALE_CHOICES: tuple[tuple[str, str], ...] = (
    ("Auto", "auto"),
    ("1.0×", "1.0"),
    ("1.25×", "1.25"),
    ("1.5×", "1.5"),
    ("2.0×", "2.0"),
)


def _register_imajin_theme() -> None:
    from napari.utils.theme import Theme, available_themes, register_theme

    if "imajin" in available_themes():
        return

    theme = Theme(
        id="imajin",
        label="Imajin",
        background="#0A0A0A",
        canvas="black",
        console="#0A0A0A",
        current="#DA4E42",
        error="#99121f",
        foreground="#121212",
        highlight="#DA4E42",
        icon="#FFFFFF",
        primary="#636867",
        secondary="#7A7F7E",
        syntax_style="native",
        text="#FFFFFF",
        warning="#e3b617",
        font_size="9pt",
    )
    register_theme("imajin", theme, "imajin")


def _add_imajin_menu(viewer: Any, settings: Any, chat_dock: Any) -> None:
    from qtpy.QtGui import QAction, QActionGroup

    qmain = viewer.window._qt_window
    menubar = qmain.menuBar()
    menu = menubar.addMenu("Imajin")

    new_project_action = QAction("New Project…", qmain)
    new_project_action.triggered.connect(lambda: _new_project(qmain))
    menu.addAction(new_project_action)

    open_project_action = QAction("Open Project…", qmain)
    open_project_action.triggered.connect(lambda: _open_project(qmain))
    menu.addAction(open_project_action)

    save_project_action = QAction("Save Project", qmain)
    save_project_action.triggered.connect(lambda: _save_project(qmain))
    menu.addAction(save_project_action)

    project_status_action = QAction("Project Status", qmain)
    project_status_action.triggered.connect(lambda: _show_project_status(qmain))
    menu.addAction(project_status_action)

    menu.addSeparator()

    panels_menu = menu.addMenu("Panels")
    jobs_action = QAction("Show Jobs", qmain)
    jobs_action.triggered.connect(lambda: _show_jobs_panel(viewer))
    panels_menu.addAction(jobs_action)

    tables_action = QAction("Show Tables", qmain)
    tables_action.triggered.connect(lambda: _show_tables_panel(viewer))
    panels_menu.addAction(tables_action)

    qc_action = QAction("Show QC", qmain)
    qc_action.triggered.connect(lambda: _show_qc_panel(viewer))
    panels_menu.addAction(qc_action)

    menu.addSeparator()

    keys_action = QAction("API Keys…", qmain)
    keys_action.triggered.connect(lambda: _open_settings(qmain, settings, chat_dock))
    menu.addAction(keys_action)

    menu.addSeparator()

    scale_menu = menu.addMenu("UI Scale")
    group = QActionGroup(qmain)
    group.setExclusive(True)

    current = (settings.ui_scale or "auto").strip().lower()
    for label, value in _SCALE_CHOICES:
        action = QAction(label, qmain)
        action.setCheckable(True)
        action.setChecked(current == value)
        action.triggered.connect(
            lambda _checked=False, v=value: _set_ui_scale(qmain, settings, v)
        )
        group.addAction(action)
        scale_menu.addAction(action)


def _set_ui_scale(parent: Any, settings: Any, value: str) -> None:
    from qtpy.QtWidgets import QMessageBox

    if settings.ui_scale == value:
        return
    settings.ui_scale = value
    settings.save_secrets()
    QMessageBox.information(
        parent,
        "Imajin — UI Scale",
        f"UI scale set to <b>{value}</b>.<br>Restart imajin to apply.",
    )


def _open_settings(parent: Any, settings: Any, chat_dock: Any) -> None:
    from imajin.ui.settings_dialog import SettingsDialog

    dialog = SettingsDialog(settings, parent=parent)
    if dialog.exec():
        chat_dock.invalidate_runner()


def _new_project(parent: Any) -> None:
    from qtpy.QtWidgets import QFileDialog, QMessageBox

    from imajin.project import create_project

    path = QFileDialog.getExistingDirectory(parent, "New Imajin Project Folder")
    if not path:
        return
    try:
        result = create_project(path)
    except Exception as exc:  # noqa: BLE001
        QMessageBox.critical(parent, "Imajin Project", str(exc))
        return
    QMessageBox.information(
        parent,
        "Imajin Project",
        f"Project created:<br><b>{result['name']}</b>",
    )


def _open_project(parent: Any) -> None:
    from qtpy.QtWidgets import QFileDialog, QMessageBox

    from imajin.project import load_project

    path = QFileDialog.getExistingDirectory(parent, "Open Imajin Project Folder")
    if not path:
        return
    try:
        result = load_project(path)
    except Exception as exc:  # noqa: BLE001
        QMessageBox.critical(parent, "Imajin Project", str(exc))
        return
    warnings = result.get("warnings") or []
    suffix = f"<br>{len(warnings)} warning(s)." if warnings else ""
    QMessageBox.information(
        parent,
        "Imajin Project",
        f"Project loaded:<br><b>{result['name']}</b>{suffix}",
    )


def _save_project(parent: Any) -> None:
    from qtpy.QtWidgets import QFileDialog, QMessageBox

    from imajin.project import current_project, save_project

    project = current_project()
    if project is None:
        path = QFileDialog.getExistingDirectory(parent, "Save Imajin Project Folder")
        if not path:
            return
    else:
        path = None
    try:
        result = save_project(path)
    except Exception as exc:  # noqa: BLE001
        QMessageBox.critical(parent, "Imajin Project", str(exc))
        return
    QMessageBox.information(
        parent,
        "Imajin Project",
        f"Project saved:<br><b>{result['name']}</b>",
    )


def _show_project_status(parent: Any) -> None:
    from qtpy.QtWidgets import QMessageBox

    from imajin.project import project_status

    status = project_status()
    if not status.get("open"):
        QMessageBox.information(parent, "Imajin Project", "No project is open.")
        return
    missing = int(status.get("n_missing_files") or 0)
    autosave_error = status.get("last_autosave_error")
    details = [
        f"Path: {status.get('path')}",
        f"Files: {status.get('n_files')} ({missing} missing)",
        f"Samples: {status.get('n_samples')}",
        f"Recipes: {status.get('n_recipes')}",
        f"Runs: {status.get('n_runs')}",
        f"Tables: {status.get('n_tables')}",
    ]
    if autosave_error:
        details.append(f"Autosave error: {autosave_error}")
    QMessageBox.information(
        parent,
        "Imajin Project",
        f"<b>{status.get('name')}</b><br>" + "<br>".join(details),
    )


def _show_optional_panel(
    viewer: Any,
    attr: str,
    widget_factory: Any,
    *,
    area: str,
    name: str,
) -> None:
    qmain = viewer.window._qt_window
    dock = getattr(qmain, attr, None)
    if dock is not None:
        try:
            dock.show()
            dock.raise_()
            return
        except RuntimeError:
            setattr(qmain, attr, None)

    widget = widget_factory()
    dock = viewer.window.add_dock_widget(widget, area=area, name=name)
    setattr(qmain, attr, dock)
    dock.show()
    dock.raise_()


def _show_jobs_panel(viewer: Any) -> None:
    from imajin.ui.job_dock import JobDock

    _show_optional_panel(
        viewer,
        "_imajin_jobs_dock",
        JobDock,
        area="bottom",
        name="Jobs",
    )


def _show_tables_panel(viewer: Any) -> None:
    from imajin.ui.table_dock import TableDock

    _show_optional_panel(
        viewer,
        "_imajin_tables_dock",
        lambda: TableDock(viewer=viewer),
        area="bottom",
        name="Tables",
    )


def _show_qc_panel(viewer: Any) -> None:
    from imajin.ui.qc_dock import QCDock

    _show_optional_panel(
        viewer,
        "_imajin_qc_dock",
        lambda: QCDock(viewer=viewer),
        area="right",
        name="QC",
    )


def launch(settings: Any | None = None) -> int:
    from imajin.cli import _setup_input_method_env, _setup_wsl_env

    _setup_wsl_env()
    _setup_input_method_env()

    import napari

    from imajin.agent.state import set_viewer
    from imajin.config import Settings, ensure_dirs
    from imajin.ui.chat_dock import ChatDock
    from imajin.ui.manual_dock import ManualDock
    from imajin.ui.fonts import register_cjk_font
    from imajin.ui.theme import apply_dark_app_palette

    import imajin.tools  # noqa: F401  (registers @tool functions)

    if settings is None:
        settings = Settings.from_env()
    ensure_dirs(settings)

    _register_imajin_theme()

    viewer = napari.Viewer(title="imajin")
    viewer.theme = "imajin"
    set_viewer(viewer)

    # Apply dark palette to QApplication so Qt's client-side decorations
    # (used under Wayland) draw a dark titlebar. Must run after Viewer
    # construction (which creates the QApplication) and before showing docks.
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        apply_dark_app_palette(app)
        register_cjk_font(app)

    chat = ChatDock(viewer=viewer, settings=settings)
    manual = ManualDock(viewer=viewer)

    chat_dw = viewer.window.add_dock_widget(chat, area="right", name="Chat")
    manual_dw = viewer.window.add_dock_widget(manual, area="right", name="Manual")

    viewer.window._qt_window.tabifyDockWidget(chat_dw, manual_dw)
    chat_dw.raise_()

    _add_imajin_menu(viewer, settings, chat)

    napari.run()
    return 0
