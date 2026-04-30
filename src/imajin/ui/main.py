from __future__ import annotations


def launch() -> int:
    import napari

    from imajin.agent.state import set_viewer
    from imajin.config import Settings, ensure_dirs
    from imajin.ui.chat_dock import ChatDock
    from imajin.ui.manual_dock import ManualDock
    from imajin.ui.table_dock import TableDock

    # Importing tools here ensures all @tool-decorated functions register.
    import imajin.tools  # noqa: F401

    settings = Settings.from_env()
    ensure_dirs(settings)

    viewer = napari.Viewer(title="imajin")
    set_viewer(viewer)

    chat = ChatDock(viewer=viewer, settings=settings)
    manual = ManualDock(viewer=viewer)
    table = TableDock(viewer=viewer)

    viewer.window.add_dock_widget(chat, area="right", name="Chat")
    viewer.window.add_dock_widget(manual, area="right", name="Manual")
    viewer.window.add_dock_widget(table, area="bottom", name="Tables")

    napari.run()
    return 0
