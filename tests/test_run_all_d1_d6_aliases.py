from scripts.run_all_d1_d6 import expand_task_ids


def test_expands_d4_d6_alias_before_validation():
    assert expand_task_ids(["d4_d6"]) == [
        "d4_without",
        "d4_with",
        "d5_without",
        "d5_with",
        "d6_without",
        "d6_with",
    ]


def test_expands_single_dataset_aliases():
    assert expand_task_ids(["d4", "d6"]) == [
        "d4_without",
        "d4_with",
        "d6_without",
        "d6_with",
    ]


def test_rejects_unknown_task_after_alias_expansion():
    try:
        expand_task_ids(["d4", "d7"])
    except ValueError as exc:
        assert "Unknown task id" in str(exc)
        assert "d7" in str(exc)
    else:
        raise AssertionError("Expected unknown task id to be rejected")
