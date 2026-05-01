"""Register a CJK-capable font so Korean/Japanese/Chinese render correctly.

WSL ships with no CJK fonts by default. The Windows host always has
Malgun Gothic (`malgun.ttf`) and usually NotoSansKR — accessible via
`/mnt/c/Windows/Fonts/`. On native Linux, `fonts-noto-cjk` (or distro
equivalent) provides the same coverage.
"""
from __future__ import annotations

import os
import sys


_CANDIDATES: tuple[tuple[str, str], ...] = (
    # path, family-name to put first in the QApplication font stack
    ("/mnt/c/Windows/Fonts/malgun.ttf", "Malgun Gothic"),
    ("/mnt/c/Windows/Fonts/NotoSansKR-VF.ttf", "Noto Sans KR"),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK KR"),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK KR"),
    ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "NanumGothic"),
)


def find_cjk_font() -> tuple[str, str] | None:
    """Return (path, family) of the first existing CJK font, or None."""
    for path, family in _CANDIDATES:
        if os.path.exists(path):
            return path, family
    return None


def register_cjk_font(app) -> str | None:
    """Register a CJK font with the running QApplication, set it as the
    primary family. Returns the family that was set, or None on failure.
    """
    found = find_cjk_font()
    if found is None:
        print(
            "imajin: no CJK font found; Korean/Japanese/Chinese text may "
            "render as boxes. Install fonts-noto-cjk or run on WSL where "
            "Windows fonts are accessible.",
            file=sys.stderr,
        )
        return None
    path, family = found

    from qtpy.QtGui import QFont, QFontDatabase

    font_id = QFontDatabase.addApplicationFont(path)
    if font_id == -1:
        # Registration failed — Qt couldn't parse the file. Fall through
        # and still try to set the family in case it's installed elsewhere.
        pass
    families = QFontDatabase.applicationFontFamilies(font_id) if font_id != -1 else []
    primary = families[0] if families else family

    current = app.font()
    new_font = QFont(current)
    # Put CJK family first; keep current point size and weight. Qt falls
    # back to subsequent families for glyphs the primary doesn't cover.
    fallbacks = [primary, current.family(), "sans-serif"]
    new_font.setFamilies([f for f in fallbacks if f])
    app.setFont(new_font)
    return primary
