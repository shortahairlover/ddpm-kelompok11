"""Ubah transkrip sesi Claude Code (JSONL hasil Export) menjadi Markdown yang mudah dibaca (T7.3).

Jalankan dari root repo:
    python -m scripts.transcript_to_md <transcript.jsonl> --out logs/raw/sesi_<tanggal>.md

Isi Markdown: prompt kelompok (apa adanya), jawaban Claude, ringkasan pemanggilan tool, dan
output tool (dipotong). Blok "thinking" internal, gambar, dan metadata aplikasi tidak disertakan.
Output tool yang membaca dokumen berisi data mahasiswa kelompok lain (Daftar Kelompok) disensor.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REDACT_MARKERS = ["Daftar Kelompok"]  # input tool yang output-nya wajib disensor
REDACTED = "[DISENSOR: berisi nama/NIM mahasiswa kelompok lain dari Daftar Kelompok]"
SYSTEM_TAG = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
PASTE_TAG = re.compile(r"</?pasted_content[^>]*>")


def clean_user_text(text: str) -> str:
    """Hapus blok pengingat sistem & tag tempelan; sisakan teks yang diketik kelompok."""
    text = SYSTEM_TAG.sub("", text)
    return PASTE_TAG.sub("", text).strip()


def truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + f"\n... [dipotong, {len(text)} karakter]"


def tool_input_summary(name: str, data: dict, limit: int = 600) -> str:
    """Ringkasan input tool: perintah shell, path file, atau JSON singkat."""
    if "command" in data:
        body = data["command"]
    elif "file_path" in data:
        body = data["file_path"]
    else:
        body = json.dumps(data, ensure_ascii=False)
    desc = data.get("description")
    head = f"**Tool `{name}`**" + (f" — {desc}" if desc else "")
    return f"{head}\n```\n{truncate(body, limit)}\n```"


def result_text(content) -> str:
    """Ambil teks dari isi tool_result (string atau list blok); gambar diganti penanda."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "image":
            parts.append("[gambar]")
    return "\n".join(parts)


def convert(jsonl: Path, result_limit: int = 1500) -> str:
    note = ("_Dibuat otomatis dari ekspor sesi (scripts/transcript_to_md.py). Prompt kelompok "
            "ditulis apa adanya; output tool dipotong; data mahasiswa kelompok lain disensor._")
    lines = ["# Transkrip Sesi Claude Code — Kelompok 11", "", note, ""]
    redact_ids: set[str] = set()
    for raw in jsonl.read_text(encoding="utf8").splitlines():
        entry = json.loads(raw)
        kind = entry.get("type")
        msg = entry.get("message") or {}
        stamp = (entry.get("timestamp") or "")[:19].replace("T", " ")
        content = msg.get("content")
        origin = (entry.get("origin") or {}).get("kind")

        if kind == "attachment":
            att = entry.get("attachment") or {}
            # pesan yang dikirim kelompok saat Claude masih bekerja (diantrikan)
            if att.get("type") == "queued_command" and (att.get("origin") or {}).get(
                    "kind") == "human":
                lines += [f"## Kelompok ({stamp} UTC, dikirim saat Claude bekerja)", "",
                          clean_user_text(att.get("prompt", "")), ""]
            continue
        if kind == "assistant" and isinstance(content, list):
            for block in content:
                if block.get("type") == "text" and block.get("text", "").strip():
                    lines += [f"### Claude ({stamp} UTC)", "", block["text"].strip(), ""]
                elif block.get("type") == "tool_use":
                    data = block.get("input") or {}
                    if any(m in json.dumps(data, ensure_ascii=False) for m in REDACT_MARKERS):
                        redact_ids.add(block.get("id"))
                    lines += [tool_input_summary(block.get("name", "?"), data), ""]
        elif kind != "user" or entry.get("isMeta"):
            pass  # pesan meta otomatis (lanjutan respons, metadata gambar) tidak ditampilkan
        elif origin == "task-notification":
            lines += [f"_[{stamp} UTC] notifikasi sistem: proses background selesai_", ""]
        elif isinstance(content, str) and origin == "human":
            text = clean_user_text(content)
            if text:
                lines += [f"## Kelompok ({stamp} UTC)", "", text, ""]
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result":
                    body = (REDACTED if block.get("tool_use_id") in redact_ids
                            else truncate(result_text(block.get("content")), result_limit))
                    if body:
                        lines += ["<details><summary>output tool</summary>", "", "```",
                                  body, "```", "</details>", ""]
                elif block.get("type") == "text" and origin == "human":
                    text = clean_user_text(block.get("text", ""))
                    if text:
                        lines += [f"## Kelompok ({stamp} UTC)", "", text, ""]
                elif block.get("type") == "image" and origin == "human":
                    lines += [f"## Kelompok ({stamp} UTC)", "", "[mengirim gambar/screenshot]", ""]
    return "\n".join(lines) + "\n"


def redact_terms(md: str, terms: list[str]) -> tuple[str, int]:
    """Ganti setiap istilah (nama/NIM, tidak peka huruf besar-kecil, utuh per kata)."""
    total = 0
    for term in sorted({t.strip() for t in terms if t.strip()}, key=len, reverse=True):
        pattern = re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)
        md, n = pattern.subn("[DISENSOR]", md)
        total += n
    return md, total


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Konversi transkrip JSONL ke Markdown")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--result-limit", type=int, default=1500)
    parser.add_argument("--redact-file", type=Path,
                        help="berkas teks (di LUAR repo) berisi nama/NIM yang wajib disensor, "
                             "satu per baris")
    args = parser.parse_args(argv)
    md = convert(args.jsonl, args.result_limit)
    if args.redact_file:
        md, n = redact_terms(md, args.redact_file.read_text(encoding="utf8").splitlines())
        print(f"istilah disensor: {n} kemunculan")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf8")
    print(f"transkrip Markdown: {args.out} ({len(md) / 1024:.0f} KB)")
    return args.out


if __name__ == "__main__":
    main()
