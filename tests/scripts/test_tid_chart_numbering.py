"""``--tID`` under ``--ref_select quality`` is read in the numbered chart's labels.

Persistence filtering keeps a subset of the anchor frame's detections. The
numbered chart labels every kept source with its anchor index, so the labels
can skip values. ``--tID`` must be translated from the chart numbering into
the kept-source numbering, including the value that force-adds the target,
which must never name a detected star (issue #93).
"""

import pytest

from prose.scripts.run_photometry import map_tid_to_kept

# 22 anchor detections (labels 0..21); persistence dropped label 5.
KEPT = [i for i in range(22) if i != 5]
DETECTED = 22


def test_kept_label_maps_to_its_position_in_the_kept_list():
    assert map_tid_to_kept(21, KEPT, DETECTED, has_target_coord=True) == 20
    assert map_tid_to_kept(3, KEPT, DETECTED, has_target_coord=True) == 3


def test_detected_count_force_adds_the_target():
    # The kept list has 21 entries, so index 21 is the force-add slot.
    assert map_tid_to_kept(DETECTED, KEPT, DETECTED, has_target_coord=True) == len(KEPT)


def test_issue_93_force_add_value_is_never_a_detected_star():
    """The run in #93 used --tID 21 to add the target and measured label 21."""
    for tid in KEPT:
        assert map_tid_to_kept(tid, KEPT, DETECTED, True) != len(KEPT)


def test_dropped_label_is_rejected_with_a_message_in_chart_numbering():
    with pytest.raises(ValueError, match=r"--tID 5 .*dropped") as err:
        map_tid_to_kept(5, KEPT, DETECTED, has_target_coord=True)
    assert "--tID 22" in str(err.value)


@pytest.mark.parametrize("tid", [23, -1])
def test_out_of_range_label_offers_only_the_chart_force_add_value(tid):
    with pytest.raises(ValueError) as err:
        map_tid_to_kept(tid, KEPT, DETECTED, has_target_coord=True)
    message = str(err.value)
    assert "0..21" in message
    assert "--tID 22" in message
    assert "--tID 21 " not in message


def test_force_add_without_a_target_coordinate_is_rejected():
    with pytest.raises(ValueError, match="--target_coord"):
        map_tid_to_kept(DETECTED, KEPT, DETECTED, has_target_coord=False)


def test_without_target_coord_the_message_does_not_offer_force_add():
    with pytest.raises(ValueError) as err:
        map_tid_to_kept(30, KEPT, DETECTED, has_target_coord=False)
    assert "--tID 22" not in str(err.value)
