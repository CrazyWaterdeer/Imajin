from __future__ import annotations


def test_user_message_creates_user_bubble(qtbot) -> None:
    from imajin.ui.chat_transcript import ChatTranscript

    t = ChatTranscript()
    qtbot.addWidget(t)
    t.append_user("안녕하세요")

    assert len(t._bubbles) == 1
    assert t._bubbles[0].role == "user"
    assert t._bubbles[0].text() == "안녕하세요"


def test_assistant_streaming_updates_single_bubble(qtbot) -> None:
    from imajin.ui.chat_transcript import ChatTranscript

    t = ChatTranscript()
    qtbot.addWidget(t)
    t.begin_assistant()
    t.append_assistant_delta("Hello ")
    t.append_assistant_delta("world!")

    assistants = [b for b in t._bubbles if b.role == "assistant"]
    assert len(assistants) == 1
    assert assistants[0].text() == "Hello world!"


def test_user_after_assistant_starts_new_assistant_on_next_delta(qtbot) -> None:
    from imajin.ui.chat_transcript import ChatTranscript

    t = ChatTranscript()
    qtbot.addWidget(t)
    t.begin_assistant()
    t.append_assistant_delta("first turn")
    t.append_user("second prompt")
    # New assistant turn must not bleed back into the previous one.
    t.append_assistant_delta("second answer")

    assistants = [b for b in t._bubbles if b.role == "assistant"]
    assert len(assistants) == 2
    assert assistants[0].text() == "first turn"
    assert assistants[1].text() == "second answer"


def test_system_message_creates_muted_bubble(qtbot) -> None:
    from imajin.ui.chat_transcript import ChatTranscript

    t = ChatTranscript()
    qtbot.addWidget(t)
    t.append_system("→ tool_x running…")

    assert len(t._bubbles) == 1
    assert t._bubbles[0].role == "system"


def test_clear_removes_all_bubbles(qtbot) -> None:
    from imajin.ui.chat_transcript import ChatTranscript

    t = ChatTranscript()
    qtbot.addWidget(t)
    t.append_user("hi")
    t.begin_assistant()
    t.append_assistant_delta("hi back")
    t.append_system("done")

    t.clear()
    assert t._bubbles == []
    assert t._current_assistant is None


def test_to_plain_text_shim_joins_messages(qtbot) -> None:
    from imajin.ui.chat_transcript import ChatTranscript

    t = ChatTranscript()
    qtbot.addWidget(t)
    t.append_user("first")
    t.append_system("middle")
    out = t.toPlainText()
    assert "first" in out and "middle" in out


def test_append_shim_routes_to_system(qtbot) -> None:
    from imajin.ui.chat_transcript import ChatTranscript

    t = ChatTranscript()
    qtbot.addWidget(t)
    t.append("legacy hello")
    assert len(t._bubbles) == 1
    assert t._bubbles[0].role == "system"
