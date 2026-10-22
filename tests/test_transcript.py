"""Test konverter transkrip (Fase 7): label prompt, sensor data kelompok lain, pesan Claude."""

import json

from scripts.transcript_to_md import convert


def _write(tmp_path, entries):
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf8")
    return path


def test_convert_labels_redacts_and_keeps_claude(tmp_path):
    entries = [
        {"type": "user", "origin": {"kind": "human"}, "timestamp": "2026-10-22T05:00:00Z",
         "message": {"content": "prompt asli kelompok<system-reminder>rahasia</system-reminder>"}},
        {"type": "assistant", "timestamp": "2026-10-22T05:00:01Z", "message": {"content": [
            {"type": "thinking", "thinking": "pikiran internal"},
            {"type": "text", "text": "jawaban Claude"},
            {"type": "tool_use", "id": "t1", "name": "Read",
             "input": {"file_path": "D:/x/Daftar Kelompok 2026-1.pdf"}},
            {"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "ls"}},
        ]}},
        {"type": "user", "timestamp": "2026-10-22T05:00:02Z", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "Nama Mahasiswa Lain 123"},
            {"type": "tool_result", "tool_use_id": "t2", "content": "file.txt"},
        ]}},
        {"type": "user", "isMeta": True, "timestamp": "2026-10-22T05:00:03Z",
         "message": {"content": "Your response above was cut off mid-stream."}},
        {"type": "user", "origin": {"kind": "task-notification"},
         "timestamp": "2026-10-22T05:00:04Z", "message": {"content": "<task-notification>x"}},
        {"type": "attachment", "timestamp": "2026-10-22T05:00:05Z", "attachment": {
            "type": "queued_command", "prompt": "pesan saat bekerja", "origin": {"kind": "human"}}},
    ]
    md = convert(_write(tmp_path, entries))
    assert "prompt asli kelompok" in md and "rahasia" not in md
    assert "jawaban Claude" in md  # pesan Claude tidak boleh hilang
    assert "pikiran internal" not in md  # blok thinking tidak disertakan
    assert "Nama Mahasiswa Lain" not in md and "DISENSOR" in md
    assert "file.txt" in md  # output tool lain tetap ada
    assert "cut off" not in md  # pesan meta otomatis dibuang
    assert "notifikasi sistem" in md
    assert "pesan saat bekerja" in md
    assert md.count("## Kelompok") == 2
