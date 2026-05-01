from __future__ import annotations


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


def launch() -> int:
    import napari

    from imajin.agent.state import set_viewer
    from imajin.config import Settings, ensure_dirs
    from imajin.ui.chat_dock import ChatDock
    from imajin.ui.manual_dock import ManualDock
    from imajin.ui.table_dock import TableDock

    import imajin.tools  # noqa: F401  (registers @tool functions)

    settings = Settings.from_env()
    ensure_dirs(settings)

    _register_imajin_theme()

    viewer = napari.Viewer(title="imajin")
    viewer.theme = "imajin"
    set_viewer(viewer)

    chat = ChatDock(viewer=viewer, settings=settings)
    manual = ManualDock(viewer=viewer)
    table = TableDock(viewer=viewer)

    viewer.window.add_dock_widget(chat, area="right", name="Chat")
    viewer.window.add_dock_widget(manual, area="right", name="Manual")
    viewer.window.add_dock_widget(table, area="bottom", name="Tables")

    napari.run()
    return 0
