from __future__ import annotations

import json
import asyncio
import shutil
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.live_room_store import (
    MEASUREMENT_TOOLS,
    LiveRoomError,
    LiveRoomStore,
    RoomExpired,
    RoomNotFound,
    RoomUnauthorized,
)
from live_rooms_ws import LiveRoomWebSocketService
import live_rooms_ws
import api.live_rooms as live_rooms_api
import services.live_room_store as live_room_store_module
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve


class AlwaysAllow:
    def allow(self, *_args, **_kwargs):
        return True


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value


@pytest.fixture
def room_store(tmp_path: Path):
    pants = tmp_path / "pants"
    case = "PanTS_00000035"
    image_dir = pants / "image_only" / case
    mask_dir = pants / "mask_only" / case
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    affine = np.diag([1.5, 1.5, 2.0, 1.0])
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 2), dtype=np.int16), affine), image_dir / "ct.nii.gz")
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 2), dtype=np.uint8), affine), mask_dir / "combined_labels.nii.gz")
    clock = Clock()
    store = LiveRoomStore(tmp_path / "sessions", pants, now=clock)
    return store, clock


def create(room_store):
    store, _ = room_store
    metadata, key = store.create_room("35", "low")
    return store, metadata, key


def test_fast_room_keeps_full_resolution_mask(room_store):
    store, _ = room_store
    case = "PanTS_00000035"
    image_dir = store.pants_path / "image_only" / case
    mask_dir = store.pants_path / "mask_only" / case
    low_affine = np.diag([3.0, 3.0, 4.0, 1.0])
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 1), dtype=np.int16), low_affine), image_dir / "ct_lowres.nii.gz")
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 1), dtype=np.uint8), low_affine), mask_dir / "combined_labels_lowres.nii.gz")

    metadata, _ = store.create_room("35", "low")
    stored = json.loads((store.root / metadata["room_id"] / "metadata.json").read_text())

    assert stored["base_ct_path"].endswith("ct_lowres.nii.gz")
    assert stored["base_mask_path"].endswith("combined_labels.nii.gz")
    assert metadata["dimensions"] == [4, 4, 2]


def test_websocket_origin_allowlist_is_exact(monkeypatch):
    monkeypatch.setenv(
        "LIVE_ROOMS_ALLOWED_ORIGINS",
        "https://bodymaps.example, http://localhost:5173",
    )
    assert live_rooms_ws._allowed_origins() == [
        "https://bodymaps.example", "http://localhost:5173",
    ]
    monkeypatch.setenv("LIVE_ROOMS_ALLOWED_ORIGINS", "*")
    with pytest.raises(RuntimeError, match="non-wildcard"):
        live_rooms_ws._allowed_origins()


def commit_chat(store: LiveRoomStore, room_id: str, key: str, index: int):
    return store.commit_event(
        room_id,
        key,
        event_id=f"event-{index}",
        event_type="chat.add",
        participant_id="participant-1",
        name="Reviewer",
        payload={"message": {"id": f"message-{index}", "author": "Reviewer", "text": f"message {index}"}},
    )


def test_server_owns_chat_and_note_author_names(room_store):
    store, metadata, key = create(room_store)
    chat, _ = store.commit_event(
        metadata["room_id"], key, event_id="spoof-chat", event_type="chat.add",
        participant_id="participant-1", name="Reviewer",
        payload={"message": {"id": "message-1", "author": "Someone else", "text": "hello"}},
    )
    note, _ = store.commit_event(
        metadata["room_id"], key, event_id="spoof-note", event_type="note.upsert",
        participant_id="participant-1", name="Reviewer",
        payload={"note": {
            "id": "note-1", "author": "Someone else", "text": "finding",
            "world": [1, 2, 3], "plane": "axial",
        }},
    )
    assert chat["payload"]["message"]["author"] == "Reviewer"
    assert note["payload"]["note"]["author"] == "Reviewer"


@pytest.mark.skipif(sys.platform == "win32", reason="Live Room store uses POSIX fsync/chmod semantics")
def test_interactive_prompt_delta_is_persisted_as_export_provenance(room_store):
    store, metadata, key = create(room_store)
    event, replayed = store.commit_event(
        metadata["room_id"], key, event_id="interactive-prompt:0", event_type="interactive.prompt",
        participant_id="participant-1", name="Reviewer",
        payload={
            "prompt_id": "interactive-prompt", "prompt_type": "point",
            "prompt": {"type": "point", "coordinates": [1, 1, 1]},
            "slice_axis": 2, "slice_index": 1, "include": True,
            "bbox": [[0, 2], [0, 2], [0, 1]], "shape": [2, 2, 1],
            "encoding": "zlib-base64-uint8", "chunk_index": 0, "chunk_count": 1,
            "data_chunk": "eA==", "label_id": 23, "organ_label": "Liver", "prompt_count": 2,
        },
    )
    assert not replayed
    assert event["payload"]["prompt"]["coordinates"] == [1, 1, 1]
    assert event["payload"]["label_id"] == 23
    assert event["payload"]["organ_label"] == "Liver"
    assert event["payload"]["prompt_count"] == 2
    accepted, _ = store.commit_event(
        metadata["room_id"], key, event_id="interactive-accept:0", event_type="interactive.accept",
        participant_id="participant-1", name="Reviewer",
        payload={"accept_id": "interactive-accept", "bbox": [[0, 1], [0, 1], [0, 1]],
                 "shape": [1, 1, 1], "chunk_index": 0, "chunk_count": 1, "data_chunk": "eA==",
                 "label_id": 23, "organ_label": "Liver", "prompt_count": 2,
                 "prompt_log": ["point (+)", "box (+)"], "allow_overwrite": False},
    )
    assert accepted["payload"]["prompt_log"] == ["point (+)", "box (+)"]
    with zipfile.ZipFile(store.build_export(metadata["room_id"], key)) as archive:
        saved = [json.loads(line) for line in archive.read("events.jsonl").decode().splitlines()]
    assert saved[-2]["type"] == "interactive.prompt"
    assert saved[-1]["type"] == "interactive.accept"
    assert saved[-1]["payload"]["data_chunk"] == "eA=="


def test_interactive_accept_event_sanitizes_organ_provenance_without_disk_io():
    store = object.__new__(LiveRoomStore)
    payload = {"accept_id": "accept-1", "bbox": [[0, 1], [0, 1], [0, 1]], "shape": [1, 1, 1],
               "chunk_index": 0, "chunk_count": 1, "data_chunk": "eA==", "label_id": 23,
               "organ_label": "Liver", "prompt_count": 2, "prompt_log": ["point (+)"],
               "allow_overwrite": False}
    normalized, before = store._apply_event(
        Path("."), {"dimensions": [2, 2, 2]}, {}, "interactive.accept", payload, 1, "Reviewer"
    )
    assert before is None
    assert normalized["organ_label"] == "Liver"
    assert normalized["prompt_log"] == ["point (+)"]
    with pytest.raises(LiveRoomError, match="accepted organ provenance"):
        store._apply_event(Path("."), {"dimensions": [2, 2, 2]}, {}, "interactive.accept",
                           {**payload, "label_id": True}, 2, "Reviewer")


def test_current_metadata_does_not_rescan_complete_event_log(room_store, monkeypatch):
    store, metadata, key = create(room_store)
    commit_chat(store, metadata["room_id"], key, 1)

    def fail_if_scanned(_room_dir):
        raise AssertionError("normal metadata load scanned complete event log")

    monkeypatch.setattr(store, "_iter_events", fail_if_scanned)
    assert store.get_metadata(metadata["room_id"], key)["latest_seq"] == 1
    commit_chat(store, metadata["room_id"], key, 2)


def test_chat_state_is_bounded_but_export_keeps_full_history(room_store, monkeypatch):
    store, metadata, key = create(room_store)
    monkeypatch.setattr(live_room_store_module, "MAX_CHAT_MESSAGES", 3)
    for index in range(5):
        commit_chat(store, metadata["room_id"], key, index)
    snapshot = store.get_snapshot(metadata["room_id"], key)
    assert [item["text"] for item in snapshot["state"]["chat"]] == ["message 2", "message 3", "message 4"]
    with zipfile.ZipFile(store.build_export(metadata["room_id"], key)) as archive:
        exported = json.loads(archive.read("chat.json"))
    assert [item["text"] for item in exported] == [f"message {index}" for index in range(5)]


def test_new_mask_writer_map_uses_uint32(room_store):
    store, metadata, key = create(room_store)
    store.commit_event(
        metadata["room_id"], key, event_id="mask-size", event_type="mask.patch",
        participant_id="participant-1", name="Reviewer",
        payload={
            "operation_id": "mask-size", "geometry_hash": metadata["geometry_hash"],
            "resolution": "low", "segment_label": 1,
            "ranges": [{"start": 0, "length": 1, "before": 0, "after": 1}],
        },
    )
    writer_path = store.root / metadata["room_id"] / "mask_writers.bin"
    assert writer_path.stat().st_size == int(np.prod(metadata["dimensions"])) * np.dtype(np.uint32).itemsize


def test_mask_patch_rejects_excessive_range_fragmentation(room_store, monkeypatch):
    store, metadata, key = create(room_store)
    monkeypatch.setattr(live_room_store_module, "MAX_MASK_RANGES_PER_EVENT", 2)
    with pytest.raises(live_room_store_module.LiveRoomError, match="too many ranges"):
        store.commit_event(
            metadata["room_id"], key, event_id="fragmented-mask", event_type="mask.patch",
            participant_id="participant-1", name="Reviewer",
            payload={
                "operation_id": "fragmented-mask", "geometry_hash": metadata["geometry_hash"],
                "resolution": "low", "segment_label": 1,
                "ranges": [
                    {"start": 0, "length": 1, "before": 0, "after": 1},
                    {"start": 2, "length": 1, "before": 0, "after": 1},
                    {"start": 4, "length": 1, "before": 0, "after": 1},
                ],
            },
        )


def test_capability_hash_and_path_rejection(room_store):
    store, metadata, key = create(room_store)
    room_dir = store.root / metadata["room_id"]
    stored = json.loads((room_dir / "metadata.json").read_text())
    assert key not in stored
    assert len(stored["key_hash"]) == 64
    with pytest.raises(RoomUnauthorized):
        store.get_metadata(metadata["room_id"], "wrong")
    with pytest.raises(RoomNotFound):
        store.get_metadata("../../etc/passwd", key)


def test_atomic_sequences_and_event_id_idempotency(room_store):
    store, metadata, key = create(room_store)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda index: commit_chat(store, metadata["room_id"], key, index), range(24)))
    sequences = sorted(event["seq"] for event, _ in results)
    assert sequences == list(range(1, 25))
    original = next(event for event, _ in results if event["event_id"] == "event-3")
    first, duplicate = commit_chat(store, metadata["room_id"], key, 3)
    assert duplicate is True
    assert first["seq"] == original["seq"]
    assert store.get_metadata(metadata["room_id"], key)["latest_seq"] == 24
    restarted = LiveRoomStore(store.root.parent, store.pants_path, now=room_store[1])
    first, duplicate = commit_chat(restarted, metadata["room_id"], key, 3)
    assert duplicate is True
    assert first["seq"] == original["seq"]


def test_replay_is_bounded_and_last_sequence_is_validated(room_store):
    store, metadata, key = create(room_store)
    for index in range(3):
        commit_chat(store, metadata["room_id"], key, index)

    events, resync = store.replay_after(metadata["room_id"], key, 0, budget=2)
    assert events == []
    assert resync is True
    events, resync = store.replay_after(metadata["room_id"], key, 1, budget=2)
    assert [event["seq"] for event in events] == [2, 3]
    assert resync is False
    for invalid in (-1, True, 4):
        with pytest.raises(live_room_store_module.LiveRoomError) as error:
            store.replay_after(metadata["room_id"], key, invalid)
        assert error.value.code == "invalid_last_seq"


def test_replay_aggregate_byte_budget_requires_resync_below_event_limit(room_store):
    store, metadata, key = create(room_store)
    room_dir = store.root / metadata["room_id"]
    events = [
        {
            "seq": index,
            "event_id": f"large-{index}",
            "type": "chat.add",
            "participant_id": "participant-1",
            "name": "Reviewer",
            "created_at": "2026-07-12T12:00:00Z",
            "payload": {"padding": "x" * (300 * 1024)},
        }
        for index in range(1, 9)
    ]
    (room_dir / "events.jsonl").write_bytes(
        b"".join(
            json.dumps(event, separators=(",", ":")).encode("utf-8") + b"\n"
            for event in events
        )
    )
    stored_metadata = json.loads((room_dir / "metadata.json").read_text())
    stored_metadata["latest_seq"] = len(events)
    (room_dir / "metadata.json").write_text(json.dumps(stored_metadata))

    replay, resync = store.replay_after(metadata["room_id"], key, 0)

    assert len(events) < live_room_store_module.MAX_REPLAY_EVENTS
    assert replay == []
    assert resync is True


def test_room_permissions_and_undo_share_event_quota(room_store, monkeypatch):
    store, metadata, key = create(room_store)
    room_dir = store.root / metadata["room_id"]
    assert store.root.stat().st_mode & 0o777 == 0o700
    assert room_dir.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in room_dir.iterdir() if path.is_file())

    store.commit_event(
        metadata["room_id"], key, event_id="note-quota", event_type="note.upsert",
        participant_id="participant-1", name="Reviewer",
        payload={"note": {"id": "note-1", "text": "finding", "world": [1, 2, 3]}},
    )
    monkeypatch.setattr(live_room_store_module, "MAX_EVENTS", 1)
    with pytest.raises(live_room_store_module.RoomFull):
        store.undo_latest(metadata["room_id"], key, "participant-1", "Reviewer")
    assert "note-1" in store.get_snapshot(metadata["room_id"], key)["state"]["notes"]


def test_export_csv_escapes_spreadsheet_formulas(room_store):
    store, metadata, key = create(room_store)
    store.commit_event(
        metadata["room_id"], key, event_id="formula-measurement",
        event_type="measurement.upsert", participant_id="participant-1", name="Reviewer",
        payload={"measurement": {
            "id": "measurement-1", "tool": "Length", "points": [[1, 2, 3], [2, 3, 4]],
            "polyline": [], "label": "=HYPERLINK(\"https://example.invalid\")", "text": "+cmd",
        }},
    )
    with zipfile.ZipFile(store.build_export(metadata["room_id"], key)) as archive:
        exported = archive.read("measurements.csv").decode("utf-8")
    assert "'=HYPERLINK" in exported
    assert "'+cmd" in exported


@pytest.mark.parametrize("tool", sorted(MEASUREMENT_TOOLS))
def test_measurement_round_trip_for_allowlisted_tools(room_store, tool):
    store, metadata, key = create(room_store)
    event, duplicate = store.commit_event(
        metadata["room_id"],
        key,
        event_id=f"measurement-{tool}",
        event_type="measurement.upsert",
        participant_id="participant-1",
        name="Reviewer",
        payload={
            "measurement": {
                "id": f"annotation-{tool}",
                "tool": tool,
                "points": [[1, 2, 3], [4, 5, 6]],
                "polyline": [],
                "text": "plain text",
                "label": "finding",
                "frame_of_reference": "frame-1",
            }
        },
    )
    snapshot = store.get_snapshot(metadata["room_id"], key)
    assert duplicate is False
    assert event["payload"]["measurement"]["tool"] == tool
    assert snapshot["state"]["measurements"][f"annotation-{tool}"]["points"] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_overlapping_mask_patch_has_conflict_aware_partial_undo(room_store):
    store, metadata, key = create(room_store)
    first, _ = store.commit_event(
        metadata["room_id"], key,
        event_id="stroke-one", event_type="mask.patch", participant_id="p1", name="One",
        payload={
            "operation_id": "stroke-one", "geometry_hash": metadata["geometry_hash"],
            "resolution": "low", "segment_label": 1,
            "ranges": [{"start": 0, "length": 4, "before": 77, "after": 1}],
        },
    )
    second, _ = store.commit_event(
        metadata["room_id"], key,
        event_id="stroke-two", event_type="mask.patch", participant_id="p2", name="Two",
        payload={
            "operation_id": "stroke-two", "geometry_hash": metadata["geometry_hash"],
            "resolution": "low", "segment_label": 2,
            "ranges": [{"start": 2, "length": 2, "before": 88, "after": 2}],
        },
    )
    assert (first["seq"], second["seq"]) == (1, 2)
    assert first["payload"]["ranges"] == [{"start": 0, "length": 4, "before": 0, "after": 1}]
    assert second["payload"]["ranges"] == [{"start": 2, "length": 2, "before": 1, "after": 2}]
    result = store.undo_latest(metadata["room_id"], key, "p1", "One")
    assert result["ok"] is True
    assert result["partial"] is True
    assert (result["reverted"], result["total"]) == (2, 4)
    mask_path, sequence = store.get_mask_snapshot(metadata["room_id"], key)
    flat = np.asarray(nib.load(mask_path).dataobj).reshape(-1, order="F")
    assert sequence == 3
    assert flat[:4].tolist() == [0, 0, 2, 2]


def test_unknown_room_ids_do_not_grow_lock_map(room_store):
    store, _ = room_store
    initial = len(store._locks)
    for index in range(100):
        room_id = f"00000000-0000-4000-8000-{index:012d}"
        with pytest.raises(RoomNotFound):
            store.get_metadata(room_id, "unused")
    assert len(store._locks) == initial


def test_export_contains_required_artifacts(room_store):
    store, metadata, key = create(room_store)
    commit_chat(store, metadata["room_id"], key, 1)
    export = store.build_export(metadata["room_id"], key)
    with zipfile.ZipFile(export) as archive:
        names = set(archive.namelist())
        assert {
            "edited_labelmap.nii.gz", "measurements.csv", "measurements.json",
            "notes.json", "chat.json", "events.json", "report.html", "report.pdf",
            "manifest.json",
        } <= names
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["case_id"] == "35"
        assert "research and education use only" in manifest["disclaimer"].lower()


def test_restart_replays_event_after_stale_snapshot_files(room_store):
    _, clock = room_store
    store, metadata, key = create(room_store)
    commit_chat(store, metadata["room_id"], key, 7)
    room_dir = store.root / metadata["room_id"]
    stale_metadata = json.loads((room_dir / "metadata.json").read_text())
    stale_metadata["latest_seq"] = 0
    (room_dir / "metadata.json").write_text(json.dumps(stale_metadata))
    (room_dir / "state.json").write_text(json.dumps({
        "measurements": {}, "notes": {}, "chat": [], "undone_event_ids": [],
    }))
    with (room_dir / "events.jsonl").open("a") as stream:
        stream.write('{"seq":2')
    restarted = LiveRoomStore(store.root.parent, store.pants_path, now=clock)
    snapshot = restarted.get_snapshot(metadata["room_id"], key)
    assert snapshot["latest_seq"] == 1
    assert snapshot["state"]["chat"][0]["text"] == "message 7"


@pytest.mark.parametrize("failed_file", ["state.json", "metadata.json"])
def test_event_transaction_recovers_after_derived_or_final_metadata_failure(
    room_store, monkeypatch, failed_file
):
    store, metadata, key = create(room_store)
    room_dir = store.root / metadata["room_id"]
    atomic_json = live_room_store_module._atomic_json
    failed = False

    def fail_once(path, value):
        nonlocal failed
        if (
            not failed
            and path.name == failed_file
            and (room_dir / "events.jsonl").stat().st_size > 0
        ):
            failed = True
            raise OSError("injected persistence failure")
        return atomic_json(path, value)

    monkeypatch.setattr(live_room_store_module, "_atomic_json", fail_once)
    with pytest.raises(OSError, match="injected"):
        commit_chat(store, metadata["room_id"], key, 41)
    assert json.loads((room_dir / "metadata.json").read_text())["latest_seq"] == 0
    assert (room_dir / ".pending_event.json").is_file()

    monkeypatch.setattr(live_room_store_module, "_atomic_json", atomic_json)
    restarted = LiveRoomStore(store.root.parent, store.pants_path, now=room_store[1])
    snapshot = restarted.get_snapshot(metadata["room_id"], key)
    assert snapshot["latest_seq"] == 1
    assert snapshot["state"]["chat"][0]["text"] == "message 41"
    assert not (room_dir / ".pending_event.json").exists()


def test_recovery_discards_incomplete_final_append_before_retry(room_store, monkeypatch):
    store, metadata, key = create(room_store)
    room_dir = store.root / metadata["room_id"]
    atomic_json = live_room_store_module._atomic_json

    def fail_state(path, value):
        if path.name == "state.json" and (room_dir / "events.jsonl").stat().st_size:
            raise OSError("injected persistence failure")
        return atomic_json(path, value)

    monkeypatch.setattr(live_room_store_module, "_atomic_json", fail_state)
    with pytest.raises(OSError):
        commit_chat(store, metadata["room_id"], key, 42)
    event_bytes = (room_dir / "events.jsonl").read_bytes()
    (room_dir / "events.jsonl").write_bytes(event_bytes[: len(event_bytes) // 2])

    monkeypatch.setattr(live_room_store_module, "_atomic_json", atomic_json)
    restarted = LiveRoomStore(store.root.parent, store.pants_path, now=room_store[1])
    assert restarted.get_snapshot(metadata["room_id"], key)["latest_seq"] == 0
    event, duplicate = commit_chat(restarted, metadata["room_id"], key, 42)
    assert duplicate is False
    assert event["seq"] == 1


def test_join_queues_commit_between_replay_and_ready(room_store, monkeypatch):
    store, metadata, key = create(room_store)

    async def scenario():
        service = LiveRoomWebSocketService(store)
        async with serve(service.handler, "127.0.0.1", 0, max_size=512 * 1024) as server:
            port = server.sockets[0].getsockname()[1]
            uri = f"ws://127.0.0.1:{port}/ws/live-rooms/{metadata['room_id']}"
            async with connect(uri) as first, connect(uri) as joining:
                await first.send(json.dumps({
                    "type": "hello", "protocol": 1, "room_key": key,
                    "name": "First", "last_seq": 0,
                }))
                assert json.loads(await first.recv())["type"] == "room.ready"
                await first.send(json.dumps({
                    "type": "chat.add", "event_id": "before-join",
                    "payload": {"message": {"id": "before-join-message", "text": "before"}},
                }))
                assert json.loads(await first.recv())["event"]["seq"] == 1

                replay_captured = threading.Event()
                release_replay = threading.Event()
                replay_after = store.replay_after

                def delayed_replay(*args, **kwargs):
                    result = replay_after(*args, **kwargs)
                    replay_captured.set()
                    assert release_replay.wait(timeout=5)
                    return result

                monkeypatch.setattr(store, "replay_after", delayed_replay)
                await joining.send(json.dumps({
                    "type": "hello", "protocol": 1, "room_key": key,
                    "name": "Joining", "last_seq": 1,
                }))
                assert await asyncio.to_thread(replay_captured.wait, 5)
                await first.send(json.dumps({
                    "type": "chat.add", "event_id": "join-race",
                    "payload": {"message": {"id": "join-race-message", "text": "not missed"}},
                }))
                committed = json.loads(await asyncio.wait_for(first.recv(), timeout=2))
                assert committed["type"] == "event.committed"
                release_replay.set()

                ready = json.loads(await asyncio.wait_for(joining.recv(), timeout=2))
                queued = json.loads(await asyncio.wait_for(joining.recv(), timeout=2))
                assert ready["type"] == "room.ready"
                assert ready["events"] == []
                assert queued["type"] == "event.committed"
                assert queued["event"]["event_id"] == "join-race"

    asyncio.run(scenario())


def test_expiry_and_cleanup(room_store):
    store, clock = room_store
    metadata, key = store.create_room("35", "low")
    clock.value += timedelta(hours=24, seconds=1)
    with pytest.raises(RoomExpired):
        store.get_metadata(metadata["room_id"], key)
    assert store.cleanup_expired() == [metadata["room_id"]]
    assert not (store.root / metadata["room_id"]).exists()


def test_two_websocket_clients_converge_and_reconnect(room_store):
    store, metadata, key = create(room_store)

    async def scenario():
        service = LiveRoomWebSocketService(store)
        async with serve(service.handler, "127.0.0.1", 0, max_size=512 * 1024) as server:
            port = server.sockets[0].getsockname()[1]
            uri = f"ws://127.0.0.1:{port}/ws/live-rooms/{metadata['room_id']}"
            hello = {
                "type": "hello", "protocol": 1, "room_key": key,
                "name": "One", "last_seq": 0,
            }
            async with connect(uri) as first:
                await first.send(json.dumps(hello))
                ready_one = json.loads(await first.recv())
                assert ready_one["type"] == "room.ready"
                await first.send(json.dumps({
                    "type": "presence.update",
                    "payload": {"cursor": {"pane": "axial", "x": float("nan"), "y": 0.5}},
                }))
                invalid_presence = json.loads(await first.recv())
                assert invalid_presence["type"] == "error"
                assert invalid_presence["fatal"] is False
                async with connect(uri) as second:
                    await second.send(json.dumps({
                        **hello,
                        "name": "Two",
                    }))
                    ready_two = json.loads(await second.recv())
                    joined = json.loads(await first.recv())
                    assert ready_two["type"] == "room.ready"
                    assert joined["type"] == "presence.changed"
                    await first.send(json.dumps({
                        "type": "chat.add",
                        "event_id": "socket-event-1",
                        "payload": {"message": {"id": "socket-message-1", "author": "One", "text": "review together"}},
                    }))
                    committed_one = json.loads(await first.recv())
                    committed_two = json.loads(await second.recv())
                    assert committed_one["event"]["seq"] == committed_two["event"]["seq"] == 1
            # Same participant resumes at acknowledged sequence without replay duplicates.
            async with connect(uri) as resumed:
                await resumed.send(json.dumps({
                    **hello,
                    "participant_id": ready_one["self"]["participant_id"],
                    "resume_credential": ready_one["resume_credential"],
                    "last_seq": 1,
                }))
                ready = json.loads(await resumed.recv())
                assert ready["type"] == "room.ready"
                assert ready["events"] == []
                assert ready["latest_seq"] == 1

    asyncio.run(scenario())


def test_reconnecting_same_participant_replaces_stale_socket(room_store):
    store, metadata, key = create(room_store)

    async def scenario():
        service = LiveRoomWebSocketService(store)
        async with serve(service.handler, "127.0.0.1", 0, max_size=512 * 1024) as server:
            port = server.sockets[0].getsockname()[1]
            uri = f"ws://127.0.0.1:{port}/ws/live-rooms/{metadata['room_id']}"
            hello = {
                "type": "hello", "protocol": 1, "room_key": key,
                "name": "One", "last_seq": 0,
            }
            async with connect(uri) as first:
                await first.send(json.dumps(hello))
                first_ready = json.loads(await first.recv())
                assert first_ready["type"] == "room.ready"
                resume = {
                    **hello,
                    "participant_id": first_ready["self"]["participant_id"],
                    "resume_credential": first_ready["resume_credential"],
                }
                async with connect(uri) as attacker:
                    await attacker.send(json.dumps({
                        **hello,
                        "participant_id": first_ready["self"]["participant_id"],
                        "resume_credential": "wrong-credential",
                    }))
                    rejected = json.loads(await attacker.recv())
                    assert rejected["type"] == "error"
                    assert rejected["code"] == "invalid_participant_resume"
                    assert rejected["fatal"] is True
                await first.send(json.dumps({"type": "ping"}))
                assert json.loads(await first.recv())["type"] == "pong"
                async with connect(uri) as replacement:
                    await replacement.send(json.dumps(resume))
                    ready = json.loads(await asyncio.wait_for(replacement.recv(), timeout=2))
                    assert ready["type"] == "room.ready"
                    await asyncio.wait_for(first.wait_closed(), timeout=2)
                    assert first.close_code == 4000
                    await replacement.send(json.dumps({"type": "ping"}))
                    assert json.loads(await replacement.recv())["type"] == "pong"

    asyncio.run(scenario())


def test_websocket_capacity_rejects_before_minting_identity(room_store):
    store, metadata, key = create(room_store)

    async def scenario():
        service = LiveRoomWebSocketService(store)
        sockets = []
        try:
            async with serve(service.handler, "127.0.0.1", 0, max_size=512 * 1024) as server:
                port = server.sockets[0].getsockname()[1]
                uri = f"ws://127.0.0.1:{port}/ws/live-rooms/{metadata['room_id']}"
                for index in range(8):
                    websocket = await connect(uri)
                    sockets.append(websocket)
                    await websocket.send(json.dumps({
                        "type": "hello", "protocol": 1, "room_key": key,
                        "name": f"Peer {index}", "last_seq": 0,
                    }))
                    assert json.loads(await websocket.recv())["type"] == "room.ready"
                rejected = await connect(uri)
                sockets.append(rejected)
                await rejected.send(json.dumps({
                    "type": "hello", "protocol": 1, "room_key": key,
                    "name": "Ninth", "last_seq": 0,
                }))
                error = json.loads(await rejected.recv())
                assert error["code"] == "participant_limit"
                room_dir = store.root / metadata["room_id"]
                stored = json.loads((room_dir / "metadata.json").read_text())
                assert len(stored["participant_credentials"]) == 8
        finally:
            await asyncio.gather(*(websocket.close() for websocket in sockets), return_exceptions=True)

    asyncio.run(scenario())


def test_websocket_expiry_check_survives_external_cleanup(room_store):
    store, metadata, key = create(room_store)

    async def scenario():
        service = LiveRoomWebSocketService(store)
        async with serve(service.handler, "127.0.0.1", 0, max_size=512 * 1024) as server:
            port = server.sockets[0].getsockname()[1]
            uri = f"ws://127.0.0.1:{port}/ws/live-rooms/{metadata['room_id']}"
            async with connect(uri) as websocket:
                await websocket.send(json.dumps({
                    "type": "hello", "protocol": 1, "room_key": key,
                    "name": "One", "last_seq": 0,
                }))
                assert json.loads(await websocket.recv())["type"] == "room.ready"
                shutil.rmtree(store.root / metadata["room_id"])
                await service.expire_rooms_once()
                assert json.loads(await websocket.recv())["type"] == "room.expired"

    asyncio.run(scenario())


def test_rest_creation_and_capability_header(room_store, monkeypatch):
    store, _ = room_store
    monkeypatch.setattr(live_rooms_api, "_store", store)
    monkeypatch.setattr(live_rooms_api, "_creation_limiter", AlwaysAllow())
    app = Flask(__name__)
    app.register_blueprint(live_rooms_api.live_rooms_blueprint, url_prefix="/api")
    client = app.test_client()
    created = client.post("/api/live-rooms", json={"case_id": "35", "resolution": "low"})
    assert created.status_code == 201
    body = created.get_json()
    assert body["share_url"] == f"/live/{body['room_id']}#{body['room_key']}"
    assert "key_hash" not in body
    assert client.get(f"/api/live-rooms/{body['room_id']}").status_code == 401
    metadata = client.get(
        f"/api/live-rooms/{body['room_id']}",
        headers={"X-Room-Key": body["room_key"]},
    )
    assert metadata.status_code == 200
    assert metadata.headers["Referrer-Policy"] == "no-referrer"
