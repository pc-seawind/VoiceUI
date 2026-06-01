from __future__ import annotations


def get_aliyun_nls_token(access_key_id: str, access_key_secret: str) -> str:
    try:
        from nls.token import getToken  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "Aliyun NLS SDK is not installed. Install with: "
            "pip install aliyun-python-sdk-core websocket-client "
            "git+https://github.com/aliyun/alibabacloud-nls-python-sdk.git"
        ) from exc

    return str(getToken(access_key_id, access_key_secret))
