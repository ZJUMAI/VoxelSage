import numpy as np

from Tool_Box.vessel_fragment_filter import (
    VesselFragmentFilterConfig,
    filter_cross_class_fragments,
)


def _base():
    hepatic = np.zeros((40, 20, 20), dtype=bool)
    portal = np.zeros_like(hepatic)
    hepatic[5:35, 8, 8] = True
    portal[5:35, 14, 14] = True
    return hepatic, portal


def test_removes_small_fragment_isolated_from_both_trees():
    hepatic, portal = _base()
    portal[20:23, 2, 2] = True

    result = filter_cross_class_fragments(
        hepatic,
        portal,
        (1, 1, 1),
        config=VesselFragmentFilterConfig(max_class_volume_fraction=0.2),
    )

    assert not result.portal_mask[20:23, 2, 2].any()
    assert result.audit["removed_noise_components"] == 1


def test_reclassifies_detached_portal_fragment_along_hepatic_tree():
    hepatic, portal = _base()
    portal[18:23, 9, 8] = True

    result = filter_cross_class_fragments(
        hepatic,
        portal,
        (1, 1, 1),
        config=VesselFragmentFilterConfig(max_class_volume_fraction=0.2),
    )

    assert not result.portal_mask[18:23, 9, 8].any()
    assert result.hepatic_mask[18:23, 9, 8].all()
    assert result.audit["reclassified_components"] == 1


def test_keeps_large_detached_component_for_review():
    hepatic, portal = _base()
    portal[5:30, 2, 2] = True

    result = filter_cross_class_fragments(hepatic, portal, (1, 1, 1))

    assert result.portal_mask[5:30, 2, 2].all()
    assert result.audit["review_required_components"] >= 1


def test_tumor_proximity_prevents_automatic_reclassification():
    hepatic, portal = _base()
    portal[18:23, 9, 8] = True
    tumor = np.zeros_like(hepatic)
    tumor[20, 10, 8] = True

    result = filter_cross_class_fragments(
        hepatic,
        portal,
        (1, 1, 1),
        tumor_mask=tumor,
        config=VesselFragmentFilterConfig(max_class_volume_fraction=0.2),
    )

    assert result.portal_mask[18:23, 9, 8].all()
    assert result.audit["review_required_components"] >= 1
