"""Bangun slide presentasi Proyek 1 (T7.7) -> reports/slides_presentasi.pptx.

Jalankan dari root repo:
    python -m scripts.build_slides

Angka hasil dibaca dari reports/fid_results.csv agar slide selalu konsisten dengan evaluasi.
Gambar diambil dari reports/figures/. Setiap slide punya catatan pembicara (speaker notes).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"

# Palet: indigo malam (dominan, "noise gelap") + koral (aksen, "sinyal") + teal (pendukung)
INDIGO = RGBColor(0x1B, 0x1F, 0x3B)
INDIGO_2 = RGBColor(0x2D, 0x33, 0x5E)
CORAL = RGBColor(0xF2, 0x5F, 0x5C)
TEAL = RGBColor(0x24, 0x7B, 0xA0)
INK = RGBColor(0x22, 0x22, 0x2E)
MUTED = RGBColor(0x5F, 0x63, 0x7A)
CARD = RGBColor(0xF1, 0xF2, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEAD_FONT, BODY_FONT = "Cambria", "Calibri"
W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)


# ------------------------------------------------------------------ helper
def text(slide, x, y, w, h, runs, size=16, color=INK, bold=False, font=BODY_FONT,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False):
    """Kotak teks. runs: str atau list paragraf (str / list[(teks, opsi)])."""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paragraphs = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        pieces = para if isinstance(para, list) else [(para, {})]
        for piece, opts in pieces:
            r = p.add_run()
            r.text = piece
            f = r.font
            f.name = opts.get("font", font)
            f.size = Pt(opts.get("size", size))
            f.bold = opts.get("bold", bold)
            f.italic = opts.get("italic", italic)
            f.color.rgb = opts.get("color", color)
        p.space_after = Pt(opts.get("space_after", 6) if pieces else 6)
    return box


def bullets(slide, x, y, w, h, items, size=16, color=INK):
    """Daftar berbutir (bullet asli PowerPoint lewat XML buChar)."""
    from pptx.oxml.ns import qn

    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        pieces = item if isinstance(item, list) else [(item, {})]
        for piece, opts in pieces:
            r = p.add_run()
            r.text = piece
            r.font.name = BODY_FONT
            r.font.size = Pt(size)
            r.font.bold = opts.get("bold", False)
            r.font.color.rgb = opts.get("color", color)
        p.space_after = Pt(8)
        ppr = p._p.get_or_add_pPr()
        ppr.set("marL", str(Inches(0.28)))
        ppr.set("indent", str(-Inches(0.22)))
        bu_font = ppr.makeelement(qn("a:buFont"), {"typeface": "Arial"})
        bu = ppr.makeelement(qn("a:buChar"), {"char": "•"})
        ppr.append(bu_font)
        ppr.append(bu)
    return box


def rect(slide, x, y, w, h, fill, rounded=True, line=None):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE, x, y, w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
    if rounded:
        shape.adjustments[0] = 0.08
    shape.shadow.inherit = False
    return shape


def circle_number(slide, x, y, d, label, fill=CORAL, color=WHITE, size=16):
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, x, y, d, d)
    c.fill.solid()
    c.fill.fore_color.rgb = fill
    c.line.fill.background()
    c.shadow.inherit = False
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.size, r.font.bold, r.font.name = Pt(size), True, BODY_FONT
    r.font.color.rgb = color
    return c


def picture(slide, path, x, y, max_w, max_h):
    """Sisipkan gambar sebesar mungkin di dalam kotak (x, y, max_w, max_h), tetap proporsional."""
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(max_w / iw, max_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    return slide.shapes.add_picture(str(path), x + (max_w - w) // 2, y + (max_h - h) // 2, w, h)


def new_slide(prs, title, dark=False, kicker=None):
    """Slide kosong dengan judul di posisi tetap (semua slide konten konsisten)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = INDIGO if dark else WHITE
    if kicker:
        text(slide, MARGIN, Inches(0.35), Inches(10), Inches(0.35), kicker.upper(), size=12,
             bold=True, color=CORAL)
    text(slide, MARGIN, Inches(0.62), W - 2 * MARGIN, Inches(0.8), title, size=34, bold=True,
         font=HEAD_FONT, color=WHITE if dark else INDIGO)
    return slide


def stat(slide, x, y, w, value, label, value_color=CORAL, label_color=MUTED, size=44):
    text(slide, x, y, w, Inches(0.8), value, size=size, bold=True, font=HEAD_FONT,
         color=value_color)
    text(slide, x, y + Inches(0.85), w, Inches(0.6), label, size=13, color=label_color)


def style_chart(chart, title):
    chart.has_title = True
    chart.chart_title.text_frame.text = title
    tp = chart.chart_title.text_frame.paragraphs[0]
    tp.runs[0].font.size, tp.runs[0].font.bold = Pt(14), True
    tp.runs[0].font.color.rgb = INK
    chart.has_legend = False
    for axis in (chart.category_axis, chart.value_axis):
        axis.tick_labels.font.size = Pt(12)
        axis.tick_labels.font.color.rgb = MUTED
        axis.format.line.color.rgb = RGBColor(0xC8, 0xCA, 0xD6)
    chart.value_axis.has_major_gridlines = True
    chart.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xE3, 0xE4, 0xEC)
    chart.category_axis.has_major_gridlines = False


def read_fid(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def fid_value(rows, **conds) -> float:
    for r in rows:
        if all(str(r[k]) == str(v) for k, v in conds.items()):
            return float(r["fid"])
    raise KeyError(f"FID tidak ditemukan untuk {conds}")


def fmt(v: float) -> str:
    return f"{v:.2f}".replace(".", ",")


# ------------------------------------------------------------------ slide
def build(out: Path) -> Path:
    fid = read_fid(ROOT / "reports" / "fid_results.csv")
    exps = ["E1", "E2", "E3", "E4", "E5"]
    exp_fid = [fid_value(fid, label=e, epoch=30, guidance="3.0", num_samples=5000) for e in exps]
    ws = ["0.0", "1.0", "2.0", "3.0", "5.0"]
    w_fid = [fid_value(fid, label="E1", epoch=100, guidance=w, sampler="ddim",
                       num_samples=5000) for w in ws]
    best = min(w_fid)
    best_w = ws[w_fid.index(best)].removesuffix(".0")
    ddim = fid_value(fid, label="E1", sampler="ddim", num_samples=2000)
    ddpm = fid_value(fid, label="E1", sampler="ddpm", num_samples=2000)

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    prs.core_properties.title = "Mini DDPM Class-Conditional — Kelompok 11"
    prs.core_properties.author = "Kelompok 11"

    # 1. Judul ----------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = INDIGO
    text(s, MARGIN, Inches(0.9), Inches(6.6), Inches(0.4), "PROYEK 1 · CAK4KBB3 2026-1",
         size=13, bold=True, color=CORAL)
    text(s, MARGIN, Inches(1.4), Inches(6.6), Inches(2.2),
         "Mini DDPM: dari Noise menjadi Gambar", size=44, bold=True, font=HEAD_FONT,
         color=WHITE)
    text(s, MARGIN, Inches(3.55), Inches(6.4), Inches(0.9),
         "Diffusion model class-conditional + classifier-free guidance, dibangun from scratch "
         "dengan PyTorch dan Claude Code", size=17, color=RGBColor(0xCF, 0xD3, 0xEA))
    members = [
        [("Naufal Rafif Hidayat", {"bold": True, "color": WHITE}),
         ("  103012300403 · Ketua", {"color": RGBColor(0xAE, 0xB3, 0xD1)})],
        [("Ardhian Calwa Nugraha", {"bold": True, "color": WHITE}),
         ("  103012300404", {"color": RGBColor(0xAE, 0xB3, 0xD1)})],
        [("Ali Hizqil Syauqani", {"bold": True, "color": WHITE}),
         ("  103012300036", {"color": RGBColor(0xAE, 0xB3, 0xD1)})],
        [("Muhammad Akhdan Ghifari", {"bold": True, "color": WHITE}),
         ("  103012300148", {"color": RGBColor(0xAE, 0xB3, 0xD1)})],
    ]
    text(s, MARGIN, Inches(4.75), Inches(6.4), Inches(1.9), members, size=15)
    text(s, MARGIN, Inches(6.75), Inches(6.4), Inches(0.4), "Kelompok 11", size=13,
         color=CORAL, bold=True)
    picture(s, FIG / "cifar10_final_w2.png", Inches(7.6), Inches(0.6), Inches(5.1), Inches(6.3))
    s.notes_slide.notes_text_frame.text = (
        "Pembuka (Naufal). Kami membangun model difusi kecil yang bisa membuat gambar sesuai "
        "kelas yang diminta. Gambar di kanan adalah hasil model kami: satu baris per kelas "
        "CIFAR-10.")

    # 2. Ringkasan hasil ------------------------------------------------
    s = new_slide(prs, "Ringkasan: semua target tercapai", kicker="Hasil utama")
    stats = [
        (fmt(best), "FID terbaik CIFAR-10\n(lebih kecil lebih baik)"),
        ("8,95 jt", "parameter U-Net\ndilatih di GPU laptop 6 GB"),
        ("3,8 jam", "training CIFAR-10\n100 epoch (39.000 step)"),
        ("9 fase", "branch + audit + PR\nsebelum setiap merge"),
    ]
    cw = (W - 2 * MARGIN - Inches(0.9)) / 4
    for i, (v, lab) in enumerate(stats):
        x = MARGIN + i * (cw + Inches(0.3))
        rect(s, x, Inches(1.75), cw, Inches(2.1), CARD)
        stat(s, x + Inches(0.3), Inches(1.95), cw - Inches(0.5), v, lab)
    bullets(s, MARGIN, Inches(4.35), Inches(12), Inches(2.6), [
        [("MNIST: ", {"bold": True}), ("angka 0–9 tergenerate sesuai label (sanity check lolos).",
                                        {})],
        [("CIFAR-10: ", {"bold": True}),
         ("10 kelas dikenali jelas; guidance w = 2 memberi kualitas terbaik.", {})],
        [("Eksperimen: ", {"bold": True}),
         ("lr, batch size, noise schedule, guidance, EMA, dan sampler dibandingkan dengan FID.",
          {})],
        [("Proses: ", {"bold": True}),
         ("±80 task terlacak, 9 laporan audit, Log Interaksi LLM tiap fase.", {})],
    ], size=17)
    s.notes_slide.notes_text_frame.text = (
        "Ringkasan sebelum detail. FID 16,43 dihitung dengan 5.000 sampel; angka ini untuk "
        "perbandingan internal, bukan dibandingkan langsung dengan paper (FID 50k).")

    # 3. Cara kerja DDPM ------------------------------------------------
    s = new_slide(prs, "Cara kerja: rusak dengan noise, lalu belajar membalik",
                  kicker="Konsep DDPM")
    picture(s, FIG / "forward_cifar10.png", MARGIN, Inches(1.7), Inches(6.2), Inches(4.4))
    text(s, MARGIN, Inches(6.2), Inches(6.2), Inches(0.4),
         "Forward process pada t = 0, 100, 250, 500, 750, 999", size=12, color=MUTED,
         italic=True)
    steps = [
        ("1", "Forward", "x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε  — 1000 langkah, β linear 1e-4 → 0,02"),
        ("2", "Training", "U-Net menebak noise ε yang ditambahkan; loss = MSE(ε, ε̂)"),
        ("3", "Sampling", "Mulai dari noise murni, hapus noise 1000 langkah (DDPM) "
                          "atau 100 langkah (DDIM)"),
    ]
    for i, (n, head, body) in enumerate(steps):
        y = Inches(1.8) + i * Inches(1.55)
        circle_number(s, Inches(7.3), y, Inches(0.55), n)
        text(s, Inches(8.05), y - Inches(0.02), Inches(4.7), Inches(0.4), head, size=20,
             bold=True, color=INDIGO)
        text(s, Inches(8.05), y + Inches(0.42), Inches(4.7), Inches(1.0), body, size=15)
    s.notes_slide.notes_text_frame.text = (
        "Kenapa memprediksi noise, bukan gambar? Targetnya selalu N(0,1) di setiap t, sehingga "
        "skala loss stabil dan training mudah. Data dinormalisasi ke [-1, 1] agar sejajar "
        "dengan skala noise.")

    # 4. Arsitektur -----------------------------------------------------
    s = new_slide(prs, "Arsitektur U-Net dengan embedding waktu dan kelas",
                  kicker="Model (Prompt Engineer)")
    boxes = [
        ("x_t\n3×32×32", INDIGO_2), ("Encoder\n32→16→8→4", TEAL),
        ("Bottleneck\nResBlock · Attention", CORAL), ("Decoder\n4→8→16→32", TEAL),
        ("ε̂\n3×32×32", INDIGO_2),
    ]
    bw, gap = Inches(2.0), Inches(0.42)
    total = 5 * bw + 4 * gap
    x0 = (W - total) // 2
    for i, (lab, col) in enumerate(boxes):
        x = x0 + i * (bw + gap)
        b = rect(s, x, Inches(1.95), bw, Inches(1.25), col)
        tf = b.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        lines = lab.split("\n")
        for j, line in enumerate(lines):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = line
            r.font.name, r.font.size = BODY_FONT, Pt(16 if j == 0 else 13)
            r.font.bold = j == 0
            r.font.color.rgb = WHITE
        if i < 4:
            a = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x + bw, Inches(2.575),
                                       x + bw + gap, Inches(2.575))
            a.line.color.rgb = MUTED
            a.line.width = Pt(2)
            a.line._get_or_add_ln().append(_arrow_head(a))
    skip = rect(s, x0 + bw + gap, Inches(3.4), 3 * bw + 2 * gap, Inches(0.5), CARD)
    skip.text_frame.text = "skip connection: fitur encoder digabung ke decoder di resolusi sama"
    _style_shape_text(skip, 13, INK)
    emb = rect(s, x0 + bw + gap, Inches(4.05), 3 * bw + 2 * gap, Inches(0.5), CARD)
    emb.text_frame.text = ("emb = MLP(sinusoidal(t)) + Embedding(kelas 0–9, null = 10) "
                           "→ setiap ResBlock")
    _style_shape_text(emb, 13, INK)
    cards = [
        ("ResBlock", "GroupNorm → SiLU → Conv, embedding ditambahkan per channel; conv akhir "
                     "di-init nol agar blok mulai sebagai identitas."),
        ("Self-attention", "4 head, ditulis manual (einsum + softmax) pada resolusi 16×16 "
                           "dan bottleneck."),
        ("Ukuran", "MNIST: 1,62 jt parameter (base 32).\nCIFAR-10: 8,95 jt parameter "
                   "(base 64, 4 level)."),
    ]
    cw = (W - 2 * MARGIN - Inches(0.6)) / 3
    for i, (head, body) in enumerate(cards):
        x = MARGIN + i * (cw + Inches(0.3))
        rect(s, x, Inches(4.85), cw, Inches(2.1), CARD)
        text(s, x + Inches(0.25), Inches(5.0), cw - Inches(0.5), Inches(0.4), head, size=17,
             bold=True, color=CORAL)
        text(s, x + Inches(0.25), Inches(5.45), cw - Inches(0.5), Inches(1.4), body, size=14)
    s.notes_slide.notes_text_frame.text = (
        "Fungsi timestep embedding: memberi tahu jaringan seberapa banyak noise di input. "
        "Embedding kelas punya satu token ekstra 'null' untuk classifier-free guidance.")

    # 5. Classifier-free guidance ----------------------------------------
    s = new_slide(prs, f"Classifier-free guidance: w = {best_w} paling seimbang",
                  kicker="Mengontrol kelas")
    picture(s, FIG / "s1_guidance_cifar10.png", MARGIN, Inches(1.5), W - 2 * MARGIN,
            Inches(3.2))
    cd = CategoryChartData()
    cd.categories = ["w = 0", "w = 1", "w = 2", "w = 3", "w = 5"]
    cd.add_series("FID", w_fid)
    gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(8.2), Inches(4.75),
                            Inches(4.55), Inches(2.45), cd)
    ch = gf.chart
    style_chart(ch, "FID vs guidance (E1 epoch 100)")
    plot = ch.plots[0]
    plot.has_data_labels = True
    plot.data_labels.font.size = Pt(11)
    plot.data_labels.font.color.rgb = INK
    plot.data_labels.number_format = "0.00"
    plot.data_labels.number_format_is_linked = False
    plot.data_labels.position = XL_LABEL_POSITION.ABOVE
    ser = plot.series[0]
    ser.format.line.color.rgb = CORAL
    ser.format.line.width = Pt(2.5)
    ser.smooth = False
    ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = 10, 30
    ch.value_axis.major_unit = 5
    text(s, MARGIN, Inches(4.9), Inches(7.3), Inches(0.6),
         "ε = ε_uncond + w · (ε_cond − ε_uncond)", size=20, bold=True, font=HEAD_FONT,
         color=INDIGO)
    bullets(s, MARGIN, Inches(5.6), Inches(7.3), Inches(1.6), [
        "Label di-drop 10 % saat training → satu model, dua mode",
        "w = 0 kelas acak · w = 2–3 tegas · w = 5 jenuh, variasi turun",
        "Kurva U = trade-off kualitas–keragaman",
    ], size=15)
    s.notes_slide.notes_text_frame.text = (
        "Satu model yang sama belajar versi kondisional dan tanpa kondisi. Saat sampling kami "
        "menggabungkan keduanya. Nilai FID: " + ", ".join(
            f"w={w[:-2]}: {fmt(v)}" for w, v in zip(ws, w_fid, strict=True)) + ".")

    # 6. Data pipeline ---------------------------------------------------
    s = new_slide(prs, "Data: MNIST untuk uji, CIFAR-10 untuk hasil utama",
                  kicker="Data (Data Engineer)")
    for i, (img, head, items) in enumerate([
        (FIG / "batch_mnist.png", "MNIST · 60.000 gambar",
         ["28×28 di-pad 2 px → 32×32 (1 channel)", "Tanpa flip: angka bisa berubah makna"]),
        (FIG / "batch_cifar10.png", "CIFAR-10 · 50.000 gambar",
         ["32×32 RGB, 10 kelas", "Random horizontal flip saat training"]),
    ]):
        x = MARGIN + i * Inches(6.2)
        picture(s, img, x, Inches(1.7), Inches(2.9), Inches(2.9))
        text(s, x + Inches(3.15), Inches(1.75), Inches(2.85), Inches(0.5), head, size=17,
             bold=True, color=INDIGO)
        bullets(s, x + Inches(3.15), Inches(2.35), Inches(2.85), Inches(2.3), items, size=14)
    rect(s, MARGIN, Inches(5.05), W - 2 * MARGIN, Inches(1.85), CARD)
    bullets(s, MARGIN + Inches(0.3), Inches(5.25), W - 2 * MARGIN - Inches(0.6),
            Inches(1.6), [
                [("Normalisasi ke [−1, 1] ", {"bold": True}),
                 ("dengan Normalize(0.5, 0.5) — bukan Lambda, karena worker DataLoader di "
                  "Windows (spawn) harus bisa mem-pickle transform.", {})],
                [("Unduh otomatis ", {"bold": True}),
                 ("via torchvision; mode --debug memakai subset 512 gambar tetap.", {})],
            ], size=15)
    s.notes_slide.notes_text_frame.text = (
        "Kenapa [-1, 1]? Agar skala data sama dengan noise Gaussian dan agar clamp x0 di "
        "sampling punya batas yang jelas.")

    # 7. Training --------------------------------------------------------
    s = new_slide(prs, "Training: stabil, terukur, dan bisa dilanjutkan",
                  kicker="Training (Training Engineer)")
    picture(s, FIG / "loss_cifar10_e1.png", MARGIN, Inches(1.65), Inches(6.6), Inches(3.9))
    bullets(s, Inches(7.55), Inches(1.75), Inches(5.2), Inches(4.0), [
        "AdamW lr 2e-4 + warmup, batch 128, grad clip 1,0",
        "Mixed precision (AMP fp16): VRAM puncak 3,7 GB",
        "EMA 0,9999 — sampel jauh lebih halus",
        "Checkpoint atomik + --resume, log CSV & TensorBoard",
        "Tes overfit 1 batch: loss 1,0 → 0,0001",
    ], size=16)
    for i, (v, lab) in enumerate([("0,0167", "loss akhir MNIST (20 epoch)"),
                                  ("0,0302", "loss akhir CIFAR-10 (100 epoch)")]):
        x = MARGIN + i * Inches(3.4)
        stat(s, x, Inches(5.75), Inches(3.2), v, lab, size=36)
    s.notes_slide.notes_text_frame.text = (
        "Kenapa EMA? Rata-rata bobot dari banyak step meredam fluktuasi optimizer sehingga "
        "sampel lebih bersih (lihat perbandingan EMA vs non-EMA di laporan).")

    # 8. Eksperimen (chart FID) -----------------------------------------
    s = new_slide(prs, "Eksperimen: batch kecil membantu, cosine tidak",
                  kicker="Hyperparameter tuning")
    cd = CategoryChartData()
    cd.categories = ["E1 baseline", "E2 lr 1e-4", "E3 lr 4e-4", "E4 batch 64", "E5 cosine"]
    cd.add_series("FID", exp_fid)
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, MARGIN, Inches(1.6), Inches(7.2),
                            Inches(5.3), cd)
    ch = gf.chart
    style_chart(ch, "FID pada epoch 30 (w = 3, 5.000 sampel)")
    plot = ch.plots[0]
    plot.gap_width = 70
    plot.has_data_labels = True
    plot.data_labels.font.size = Pt(12)
    plot.data_labels.font.color.rgb = INK
    plot.data_labels.number_format = "0.00"
    plot.data_labels.number_format_is_linked = False
    plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    best_i = exp_fid.index(min(exp_fid))
    for i, point in enumerate(plot.series[0].points):
        point.format.fill.solid()
        point.format.fill.fore_color.rgb = CORAL if i == best_i else INDIGO_2
    findings = [
        ("Batch 64", "FID terbaik per epoch: 2× jumlah update, VRAM ±separuh."),
        ("Learning rate", "2e-4 titik tengah; loss kecil (4e-4) ≠ sampel lebih baik."),
        ("Cosine", "Terburuk di 32×32; skala loss-nya pun tidak sebanding."),
    ]
    for i, (head, body) in enumerate(findings):
        y = Inches(1.75) + i * Inches(1.7)
        rect(s, Inches(8.2), y, Inches(4.55), Inches(1.45), CARD)
        text(s, Inches(8.45), y + Inches(0.15), Inches(4.1), Inches(0.4), head, size=17,
             bold=True, color=CORAL)
        text(s, Inches(8.45), y + Inches(0.55), Inches(4.1), Inches(0.85), body, size=14)
    s.notes_slide.notes_text_frame.text = (
        "Semua eksperimen dibandingkan pada epoch yang sama dengan seed yang sama. "
        "Satu variabel per eksperimen (config mewarisi baseline). Nilai FID: " + ", ".join(
            f"{e}: {fmt(v)}" for e, v in zip(exps, exp_fid, strict=True)) + ".")

    # 9. Hasil kualitatif -----------------------------------------------
    s = new_slide(prs, "Kualitas naik bertahap selama training", kicker="Hasil visual")
    picture(s, FIG / "progress_cifar10_e1.png", MARGIN, Inches(1.55), W - 2 * MARGIN,
            Inches(4.6))
    bullets(s, MARGIN, Inches(6.3), Inches(12), Inches(0.9), [
        "Seed sampling sama di setiap epoch: bentuk kelas muncul sejak epoch 10, detail dan "
        "kontras membaik sampai epoch 100",
    ], size=15)
    s.notes_slide.notes_text_frame.text = (
        "Gambar per kelas dari model EMA. Detail halus masih buram — wajar untuk model 9 juta "
        "parameter di resolusi 32×32.")

    # 10. DDIM vs DDPM ----------------------------------------------------
    s = new_slide(prs, "DDIM: kualitas setara, 10× lebih cepat", kicker="Sampler")
    for i, (v, lab, col) in enumerate([
        (fmt(ddim), "FID DDIM-100\n8,9 menit / 2.000 sampel", CORAL),
        (fmt(ddpm), "FID DDPM-1000\n85,7 menit / 2.000 sampel", TEAL),
        ("≈10×", "lebih cepat\nmodel sama, tanpa training ulang", CORAL),
    ]):
        x = MARGIN + i * Inches(4.15)
        rect(s, x, Inches(1.6), Inches(3.85), Inches(1.75), CARD)
        stat(s, x + Inches(0.3), Inches(1.72), Inches(3.3), v, lab, value_color=col)
    bullets(s, MARGIN, Inches(3.8), Inches(6.5), Inches(3.3), [
        "FID 5.000 sampel × 1000 langkah DDPM ±2,3 jam per evaluasi → tidak realistis",
        "DDIM melompati timestep (100 dari 1000) memakai model yang sama",
        "Dibuktikan dengan test: η = 1 dan 1000 langkah → hasil identik dengan DDPM",
        "Demo langsung: 12 gambar dalam ±2,5 detik",
    ], size=16)
    picture(s, FIG / "s2_ema_vs_raw.png", Inches(7.45), Inches(3.6), Inches(5.3),
            Inches(3.5))
    s.notes_slide.notes_text_frame.text = (
        "Gambar kanan: EMA vs non-EMA pada w = 3 — EMA lebih halus. DDIM dan DDPM memberi FID "
        "yang praktis sama pada 2.000 sampel.")

    # 11. Vibe coding workflow ------------------------------------------
    s = new_slide(prs, "Vibe coding: Claude Code dalam proses yang diaudit",
                  kicker="Metodologi")
    flow = ["CLAUDE.md\nsystem prompt", "PLANNING.md\n±80 task", "Task → verifikasi\n→ commit",
            "Audit 9 poin\n0 error/bug", "Push → PR\n→ merge"]
    fw, fg = Inches(2.2), Inches(0.3)
    fx0 = (W - (5 * fw + 4 * fg)) // 2
    for i, lab in enumerate(flow):
        x = fx0 + i * (fw + fg)
        b = rect(s, x, Inches(1.8), fw, Inches(1.25), INDIGO if i % 2 == 0 else TEAL)
        b.text_frame.text = lab
        _style_shape_text(b, 15, WHITE, bold_first=True)
    cards = [
        ("Log Interaksi", "Prompt kelompok apa adanya, eror utuh, analisis & solusi per fase "
                          "(template dosen) + transkrip lengkap."),
        ("Audit sebelum merge", "ruff, pytest (semua test lolos), kriteria fase, smoke test, "
                                "scope, review diff, kebersihan repo."),
        ("Contoh debugging", "Override 1e-4 terbaca string · Lambda tak bisa di-pickle · "
                             "toleransi test salah · rantai PowerShell berhenti karena stderr."),
    ]
    cw = (W - 2 * MARGIN - Inches(0.6)) / 3
    for i, (head, body) in enumerate(cards):
        x = MARGIN + i * (cw + Inches(0.3))
        rect(s, x, Inches(3.45), cw, Inches(2.7), CARD)
        circle_number(s, x + Inches(0.25), Inches(3.65), Inches(0.5), str(i + 1), fill=CORAL)
        text(s, x + Inches(0.9), Inches(3.7), cw - Inches(1.1), Inches(0.45), head, size=17,
             bold=True, color=INDIGO)
        text(s, x + Inches(0.25), Inches(4.35), cw - Inches(0.5), Inches(1.7), body, size=16)
    s.notes_slide.notes_text_frame.text = (
        "Rubrik: kualitas vibe coding 30 %, log interaksi 25 %. Setiap fase punya branch, "
        "laporan audit di folder audits/, dan Pull Request di GitHub.")

    # 12. Pembagian peran ------------------------------------------------
    s = new_slide(prs, "Pembagian peran", kicker="Tim")
    roles = [
        ("Naufal Rafif Hidayat", "Prompt Engineer & LLM Navigator (Ketua)",
         "CLAUDE.md, planning, arsitektur U-Net, schedule, sampling & CFG"),
        ("Ardhian Calwa Nugraha", "Data Pipeline & Preprocessing",
         "Loader MNIST & CIFAR-10, normalisasi, augmentasi, cek shape"),
        ("Ali Hizqil Syauqani", "Training Loop & Hyperparameter Tuning",
         "Training loop, EMA, AMP, eksperimen E1–E5, FID"),
        ("Muhammad Akhdan Ghifari", "Lead Documentarian & Error Handler",
         "Log Interaksi, analisis eror, laporan akhir, slide"),
    ]
    cw = (W - 2 * MARGIN - Inches(0.3)) / 2
    for i, (name, role, task) in enumerate(roles):
        x = MARGIN + (i % 2) * (cw + Inches(0.3))
        y = Inches(1.7) + (i // 2) * Inches(2.65)
        rect(s, x, y, cw, Inches(2.35), CARD)
        circle_number(s, x + Inches(0.3), y + Inches(0.3), Inches(0.7), str(i + 1),
                      fill=INDIGO, size=20)
        text(s, x + Inches(1.25), y + Inches(0.3), cw - Inches(1.5), Inches(0.45), name,
             size=19, bold=True, color=INDIGO)
        text(s, x + Inches(1.25), y + Inches(0.8), cw - Inches(1.5), Inches(0.45), role,
             size=14, bold=True, color=CORAL)
        text(s, x + Inches(1.25), y + Inches(1.3), cw - Inches(1.5), Inches(0.9), task,
             size=14)
    s.notes_slide.notes_text_frame.text = (
        "Setiap anggota mempresentasikan bagiannya sendiri (penguasaan materi individu).")

    # 13. Demo -----------------------------------------------------------
    s = new_slide(prs, "Demo langsung", dark=True, kicker="Demo")
    rect(s, MARGIN, Inches(1.75), Inches(7.2), Inches(2.6), INDIGO_2)
    code = {"font": "Consolas", "size": 16, "color": WHITE}
    text(s, MARGIN + Inches(0.3), Inches(2.0), Inches(6.8), Inches(2.3), [
        [("python -m scripts.demo --kelas cat dog horse", code)],
        [("python -m scripts.demo --dataset mnist --kelas 3 7", code)],
        [("python -m scripts.demo --sampler ddpm", code)],
    ], size=16)
    bullets(s, MARGIN, Inches(4.75), Inches(7.2), Inches(2.3), [
        "Pilih kelas dari audiens, generate dalam ±3 detik (DDIM-100, w = 2)",
        "Seed acak setiap kali: gambar selalu baru, bukan hafalan",
        "Kanan: frame akhir GIF noise → gambar (CIFAR-10 & MNIST)",
    ], size=16, color=RGBColor(0xDD, 0xE0, 0xF0))
    picture(s, _gif_last_frame(FIG / "denoise_cifar10.gif"), Inches(8.2), Inches(1.75),
            Inches(4.55), Inches(2.6))
    picture(s, _gif_last_frame(FIG / "denoise_mnist.gif"), Inches(8.2), Inches(4.6),
            Inches(4.55), Inches(2.4))
    s.notes_slide.notes_text_frame.text = (
        "Buka reports/figures/denoise_cifar10.gif di browser untuk menunjukkan animasinya. "
        "Demo memakai checkpoint terbaik (E1 epoch 100).")

    # 14. Kesimpulan -----------------------------------------------------
    s = new_slide(prs, "Kesimpulan", dark=True, kicker="Penutup")
    points = [
        ("DDPM from scratch berhasil", "Pipeline lengkap terbukti benar di MNIST dan "
                                       "menghasilkan 10 kelas CIFAR-10 yang dikenali."),
        ("Evaluasi kuantitatif", f"FID terbaik {fmt(best)} (w = 2); lr 2e-4 & schedule linear "
                                 "cocok; batch kecil efisien per epoch."),
        ("Proses vibe coding teraudit", "Branch per fase, audit 0 bug, log interaksi lengkap — "
                                        "kode bersih dan dapat direproduksi."),
    ]
    for i, (head, body) in enumerate(points):
        y = Inches(1.8) + i * Inches(1.45)
        circle_number(s, MARGIN, y, Inches(0.6), str(i + 1))
        text(s, MARGIN + Inches(0.85), y - Inches(0.02), Inches(11), Inches(0.45), head,
             size=21, bold=True, color=WHITE)
        text(s, MARGIN + Inches(0.85), y + Inches(0.48), Inches(11), Inches(0.8), body,
             size=15, color=RGBColor(0xCF, 0xD3, 0xEA))
    text(s, MARGIN, Inches(6.25), Inches(12), Inches(0.8),
         "Pengembangan: model lebih besar & resolusi 64×64, learned variance, dataset lokal "
         "(mis. batik).", size=14, italic=True, color=RGBColor(0xAE, 0xB3, 0xD1))
    s.notes_slide.notes_text_frame.text = "Terima kasih — sesi tanya jawab."

    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    return out


def _style_shape_text(shape, size, color, bold_first=False):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.1)
    for j, p in enumerate(tf.paragraphs):
        p.alignment = PP_ALIGN.CENTER
        for r in p.runs:
            r.font.name, r.font.size, r.font.color.rgb = BODY_FONT, Pt(size), color
            r.font.bold = bold_first and j == 0


def _arrow_head(connector):
    from pptx.oxml.ns import qn

    ln = connector.line._get_or_add_ln()
    return ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})


def _gif_last_frame(gif: Path) -> Path:
    """Simpan frame terakhir GIF sebagai PNG sementara (python-pptx menyisipkan gambar statis)."""
    out = ROOT / "runs" / "slides_cache" / f"{gif.stem}_last.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(gif) as im:
        im.seek(im.n_frames - 1)
        im.convert("RGB").save(out)
    return out


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Bangun slide presentasi .pptx")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "slides_presentasi.pptx")
    args = parser.parse_args(argv)
    out = build(args.out)
    print(f"slide disimpan: {out} ({out.stat().st_size / 1024:.0f} KB)")
    return out


if __name__ == "__main__":
    main()
