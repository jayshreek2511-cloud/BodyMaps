import numpy as np

from services.interactive_mock import FakeInteractiveSession


def make_session(shape=(20, 24, 16)):
    session = FakeInteractiveSession()
    session.set_image(np.zeros((1, *shape), dtype=np.int16))
    session.set_target_buffer(np.zeros(shape, dtype=np.uint8))
    return session


def test_mock_point_produces_preview_and_undo_restores_previous_mask():
    session = make_session()
    changed = session.add_point_interaction((10, 12, 8))
    assert changed == [[8, 13], [10, 15], [6, 11]]
    assert session.target_buffer.sum() > 0
    assert session.undo() is True
    assert not session.target_buffer.any()


def test_mock_negative_point_removes_region_and_reset_clears_prompts():
    session = make_session()
    session.add_bbox_interaction([[3, 8], [4, 9], [6, 7]])
    assert session.target_buffer.sum() > 0
    session.add_point_interaction((5, 6, 6), include_interaction=False)
    assert session.target_buffer.sum() < 30
    session.reset_interactions()
    assert not session.target_buffer.any()
    assert session._prompts == []


def test_mock_scribble_and_lasso_apply_only_the_crop():
    session = make_session()
    box = [[2, 6], [3, 8], [5, 6]]
    crop = np.zeros((4, 5, 1), dtype=np.uint8)
    crop[1:3, 2:4, 0] = 1
    session.add_scribble_interaction(crop, interaction_bbox=box)
    assert int(session.target_buffer.sum()) == 4
    session.add_lasso_interaction(crop, include_interaction=False, interaction_bbox=box)
    assert not session.target_buffer.any()


def test_mock_replay_reconstructs_prompts():
    session = make_session()
    prompt = {"type": "point", "coordinates": [9, 11, 7], "include": True}
    session.replay([prompt])
    assert session.target_buffer.sum() > 0
    assert len(session._prompts) == 1
