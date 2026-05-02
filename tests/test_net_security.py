import os

from net_security import is_public_http_url


def test_blocks_private_ip_literal():
    assert is_public_http_url("http://127.0.0.1:5000/") is False


def test_onion_requires_flag():
    assert is_public_http_url("http://example.onion/path") is False
    os.environ["ALLOW_ONION"] = "1"
    try:
        assert is_public_http_url("http://example.onion/path") is True
    finally:
        os.environ.pop("ALLOW_ONION", None)


