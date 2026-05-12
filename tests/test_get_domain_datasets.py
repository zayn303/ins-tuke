import pytest
from types import SimpleNamespace
from src.data.multi_domain import get_domain_datasets


def _make_datasets(n: int):
    return [SimpleNamespace(domain_id=i) for i in range(n)]


def cfg(val):
    return SimpleNamespace(held_out_domain=val)


class TestGetDomainDatasetsValidation:
    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="missing or empty"):
            get_domain_datasets(cfg(""), _make_datasets(4))

    def test_none_raises(self):
        with pytest.raises(ValueError, match="missing or empty"):
            get_domain_datasets(cfg(None), _make_datasets(4))

    def test_non_integer_string_raises(self):
        with pytest.raises(ValueError, match="must be integer"):
            get_domain_datasets(cfg("foo"), _make_datasets(4))

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            get_domain_datasets(cfg(7), _make_datasets(4))

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            get_domain_datasets(cfg(-1), _make_datasets(4))

    def test_valid_index_0(self):
        datasets = _make_datasets(4)
        source, test_ds = get_domain_datasets(cfg(0), datasets)
        assert test_ds.domain_id == 0
        assert len(source) == 3
        assert all(ds.domain_id != 0 for ds in source)

    def test_valid_index_3(self):
        datasets = _make_datasets(4)
        source, test_ds = get_domain_datasets(cfg(3), datasets)
        assert test_ds.domain_id == 3
        assert len(source) == 3

    def test_string_int_coerced(self):
        datasets = _make_datasets(4)
        source, test_ds = get_domain_datasets(cfg("2"), datasets)
        assert test_ds.domain_id == 2
