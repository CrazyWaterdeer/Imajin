import numpy as np

from imajin import session as state


def test_put_get_array_roundtrip():
    state.put_array("m", np.zeros((3, 4)))
    assert state.get_array("m").shape == (3, 4)
    state.reset_tables()           # also clears arrays
    assert "m" not in state.current_session().arrays
