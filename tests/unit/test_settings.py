from local_lke.settings import Settings


def test_settings_load_prefixed_environment(monkeypatch: object) -> None:
    monkeypatch.setenv("LKE_PORT", "9001")
    monkeypatch.setenv("LKE_CHAT_API_KEY", "super-secret")
    settings = Settings(_env_file=None)

    assert settings.port == 9001
    assert "super-secret" not in str(settings.redacted_summary)


def test_server_defaults_to_loopback() -> None:
    assert Settings(_env_file=None).host == "127.0.0.1"

