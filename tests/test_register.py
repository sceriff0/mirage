import pytest

pytest.importorskip("valis")
from register import merge_first_file


def test_merge_first_file(tmp_path):
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    content = "hello"
    src.write_text(content)

    rc = merge_first_file([str(src)], str(dst))
    assert rc == 0
    assert dst.exists()
    assert dst.read_text() == content
