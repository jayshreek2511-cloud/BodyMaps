from __future__ import annotations

import json
import importlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from flask import Flask

from api.api_blueprint import api_blueprint
from api.interactive import interactive_blueprint
from constants import Constants
from services.mesh_generation import ensure_case_meshes


def mesh_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    pants_path = tmp_path / "pants"
    display_id = "PanTS_00000035"
    label_path = pants_path / "mask_only" / display_id / "combined_labels.nii.gz"
    label_path.parent.mkdir(parents=True)
    labels = np.zeros((8, 8, 8), dtype=np.uint8)
    labels[1:3, 1:3, 1:3] = 1
    labels[4:6, 4:6, 4:6] = 14
    nib.save(nib.Nifti1Image(labels, np.eye(4)), label_path)
    return pants_path, tmp_path / "mesh-cache", label_path


def test_missing_mesh_cache_is_generated_once_and_served(tmp_path, monkeypatch):
    pants_path, mesh_path, _ = mesh_dataset(tmp_path)
    monkeypatch.setattr(Constants, "PANTS_PATH", str(pants_path))
    monkeypatch.setattr(Constants, "MESH_PATH", str(mesh_path))

    app = Flask(__name__)
    app.register_blueprint(api_blueprint, url_prefix="/api")
    client = app.test_client()

    first = client.get("/api/cases/35/mesh-manifest")
    assert first.status_code == 200
    manifest = first.get_json()
    assert [organ["id"] for organ in manifest["organs"]] == [1, 14]
    assert manifest["center"] == [3.0, 3.0, -3.0]
    assert manifest["bounds"]["min"] == [-3.5, -3.5, -3.5]
    assert manifest["bounds"]["max"] == [3.5, 3.5, 3.5]
    assert all(organ["url"].startswith("/api/cases/PanTS_00000035/") for organ in manifest["organs"])

    manifest_path = mesh_path / "PanTS_00000035" / "manifest.json"
    first_mtime = manifest_path.stat().st_mtime_ns
    second = client.get("/api/cases/35/mesh-manifest")
    assert second.status_code == 200
    assert manifest_path.stat().st_mtime_ns == first_mtime

    asset = client.get(manifest["organs"][0]["url"])
    assert asset.status_code == 200
    assert asset.data[:4] == b"glTF"


def test_parallel_manifest_requests_share_generation(tmp_path):
    _, mesh_path, label_path = mesh_dataset(tmp_path)
    args = ("PanTS_00000035", str(label_path), str(mesh_path))
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda _: ensure_case_meshes(*args), range(2)))
    assert paths[0] == paths[1]
    assert json.loads(paths[0].read_text())["organs"]
    assert not list(mesh_path.glob(".*.tmp"))


def test_mesh_endpoints_reject_invalid_or_missing_cases(tmp_path, monkeypatch):
    pants_path, mesh_path, _ = mesh_dataset(tmp_path)
    monkeypatch.setattr(Constants, "PANTS_PATH", str(pants_path))
    monkeypatch.setattr(Constants, "MESH_PATH", str(mesh_path))
    app = Flask(__name__)
    app.register_blueprint(api_blueprint, url_prefix="/api")
    client = app.test_client()
    api_module = importlib.import_module("api.api_blueprint")

    assert client.get("/api/cases/not-numeric/mesh-manifest").status_code == 400
    monkeypatch.setattr(api_module, "_ai_local_mask_path", lambda _case_id: None)
    assert client.get("/api/cases/36/mesh-manifest").status_code == 404
    assert client.get("/api/cases/not-a-case/render_only/liver.glb").status_code == 400
    assert client.get("/api/cases/PanTS_00000035/render_only/not-a-mesh.txt").status_code == 400

    monkeypatch.setattr(Constants, "PANTS_PATH", None)
    assert client.get("/api/cases/37/mesh-manifest").status_code == 404


def test_viewer_case_volumes_load_with_interactive_disabled_or_mock(tmp_path, monkeypatch):
    pants_path = tmp_path / "pants"
    display_id = "PanTS_00000001"
    ct_path = pants_path / "image_only" / display_id / "ct.nii.gz"
    mask_path = pants_path / "mask_only" / display_id / "combined_labels.nii.gz"
    ct_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    ct = np.arange(4 * 5 * 6, dtype=np.int16).reshape((4, 5, 6))
    labels = np.zeros(ct.shape, dtype=np.int16)
    labels[1:3, 2:4, 2:5] = 3
    nib.save(nib.Nifti1Image(ct, np.eye(4)), ct_path)
    nib.save(nib.Nifti1Image(labels, np.eye(4)), mask_path)
    monkeypatch.setattr(Constants, "PANTS_PATH", str(pants_path))
    monkeypatch.setattr("api.api_blueprint.tempfile.tempdir", str(tmp_path / "temp"))
    app = Flask(__name__)
    app.register_blueprint(api_blueprint, url_prefix="/api")
    app.register_blueprint(interactive_blueprint, url_prefix="/api")
    client = app.test_client()

    for enabled, mock in (("false", "false"), ("true", "true")):
        monkeypatch.setenv("NNINTERACTIVE_ENABLED", enabled)
        monkeypatch.setenv("NNINTERACTIVE_MOCK", mock)
        config = client.get("/api/interactive/config")
        assert config.status_code == 200
        assert config.get_json()["enabled"] is (enabled == "true")

        ct_response = client.get("/api/get-main-nifti/1.nii.gz?res=low")
        mask_response = client.get("/api/get-segmentations/1.nii.gz?res=low")
        assert ct_response.status_code == 200
        assert mask_response.status_code == 200
        converted_path = tmp_path / f"converted-{enabled}.nii.gz"
        converted_path.write_bytes(mask_response.data)
        assert nib.load(str(converted_path)).get_data_dtype() == np.uint8
