import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from skills.builtin.segmentation_modification.session_manager import EditorSession
from skills.builtin.segmentation_modification import medsam2_wrapper


def _make_session(tmp_path, mask_data, mask_affine):
    shape = (3, 4, 2)
    ct_path = tmp_path / "ct.nii.gz"
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4)), ct_path)
    nib.save(nib.Nifti1Image(mask_data, mask_affine), mask_dir / "liver.nii.gz")
    return EditorSession(
        session_id="test-session",
        case_id="test-case",
        ct_nifti_path=str(ct_path),
        mask_dir=str(mask_dir),
        output_dir=str(tmp_path),
        mask_name="liver",
        mask_names=["liver"],
        ct_shape=shape,
        affine=np.eye(4),
    )


def test_mask_is_resampled_to_ct_grid_when_affine_flips_axis(tmp_path):
    mask = np.zeros((3, 4, 2), dtype=np.uint8)
    mask[0, 1, 0] = 1
    flipped_x_affine = np.array(
        [[-1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=float,
    )
    session = _make_session(tmp_path, mask, flipped_x_affine)

    aligned = session.get_current_mask()

    assert aligned.shape == (3, 4, 2)
    assert aligned[2, 1, 0] == 1
    assert aligned[0, 1, 0] == 0


def test_pop_last_click_only_affects_current_mask_and_slice(tmp_path):
    session = _make_session(tmp_path, np.zeros((3, 4, 2), dtype=np.uint8), np.eye(4))
    session.add_click(0, 1, 1, 1)
    session.add_click(1, 2, 2, 0)
    session.add_click(0, 3, 2, 0)

    removed = session.pop_last_click(slice_idx=0)

    assert removed[:4] == (0, 3, 2, 0)
    assert session.get_clicks_for_slice(0) == [(1, 1, 1)]
    assert session.get_clicks_for_slice(1) == [(2, 2, 0)]
    assert session.get_clicks_for_current_mask() == [
        (0, 1, 1, 1),
        (1, 2, 2, 0),
    ]


def test_refined_slice_tracking_excludes_propagated_updates(tmp_path):
    session = _make_session(tmp_path, np.zeros((3, 4, 2), dtype=np.uint8), np.eye(4))

    session.update_mask_slice(0, np.ones((3, 4), dtype=np.uint8))
    assert session.get_refined_slices() == []

    session.mark_refined_slice(1)
    session.mark_refined_slice(1)
    assert session.get_refined_slices() == [1]


def test_session_fallback_prefers_liver(tmp_path):
    shape = (3, 4, 2)
    ct_path = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4)), ct_path)

    session = EditorSession(
        session_id="default-mask-session",
        case_id="test-case",
        ct_nifti_path=str(ct_path),
        mask_dir=str(tmp_path),
        output_dir=str(tmp_path),
        mask_name="all",
        mask_names=["tumor", "liver"],
        ct_shape=shape,
        affine=np.eye(4),
    )

    assert session.current_mask_name == "liver"


def test_propagation_only_returns_slices_inside_anchor_radius(monkeypatch):
    torch = pytest.importorskip("torch")

    class FakePredictor:
        image_size = 2

        def __init__(self):
            self.states = []
            self.anchor_calls = []

        def init_state(self, images, video_height, video_width):
            state = {"index": len(self.states), "num_frames": len(images)}
            self.states.append(state)
            return state

        def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
            self.anchor_calls.append(
                (inference_state["index"], frame_idx, np.asarray(mask).copy())
            )

        def propagate_in_video(self, state):
            for frame_idx in range(state["num_frames"]):
                yield frame_idx, [1], torch.ones((1, 1, 2, 2))

    predictor = FakePredictor()
    fake_model = type(
        "FakeModel",
        (),
        {"predictor": predictor, "device": "cpu"},
    )()
    monkeypatch.setattr(
        medsam2_wrapper.MedSAM2Model,
        "get_instance",
        classmethod(lambda cls: fake_model),
    )

    ct = np.zeros((2, 2, 9), dtype=np.float32)
    mask = np.zeros_like(ct, dtype=np.uint8)
    mask[0, 1, 2] = 1

    result = medsam2_wrapper.propagate_3d(
        ct_volume=ct,
        mask_volume=mask,
        refined_indices=[2],
        video_height=2,
        video_width=2,
        max_propagation_slices=2,
    )

    assert set(result) == {0, 1, 2, 3, 4}
    # 反向视频中的锚点帧是 9 - 1 - 2 = 6，但注入内容必须仍来自原始第 2 层。
    assert predictor.anchor_calls[1][1] == 6
    np.testing.assert_array_equal(predictor.anchor_calls[1][2], mask[:, :, 2])
