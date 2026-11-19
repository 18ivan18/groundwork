from pathlib import Path

from api_helpers import as_user

from groundwork.config import Settings
from groundwork.main import create_app


def test_config_reports_the_upload_limits(tmp_path: Path) -> None:
    app = create_app(Settings(_env_file=None, data_dir=tmp_path, max_upload_mb=7))

    response = as_user(app).get("/api/config")

    assert response.status_code == 200
    assert response.json() == {
        "upload_types": [".htm", ".html", ".md", ".pdf", ".txt"],
        "max_upload_mb": 7,
    }
