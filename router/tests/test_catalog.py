from shellwise_router.buckets import BUCKET_IDS, ROUTABLE
from shellwise_router.build_catalog import assign_bucket, first_util
from shellwise_router.curated import CURATED


def test_curated_buckets_exist():
    assert set(CURATED.values()) <= set(BUCKET_IDS)


def test_internal_is_not_routable():
    assert "internal" not in {b.id for b in ROUTABLE}


def test_curated_wins_over_rules():
    assert assign_bucket("git", "1", "the stupid content tracker", False) == "dev"
    assert assign_bucket("tar", "1", "manipulate tape archives", False) == "archive"


def test_rules_and_defaults():
    assert assign_bucket("fooctl", "8", "configure the network interface", False) == "network"
    assert assign_bucket("bard", "8", "background daemon for bar", False) == "internal"
    assert assign_bucket("mystery", None, None, False) == "internal"
    assert assign_bucket("compadd", None, None, True) == "shell"


def test_first_util_skips_sudo_and_handles_bad_quotes():
    assert first_util("sudo lsof -i :3000") == "lsof"
    assert first_util("echo 'unterminated") is None
