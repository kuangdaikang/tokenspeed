"""Aggregate hw2 JSON results and write SVG figures. No third-party packages."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "hw2_results"
FIG = Path("/home/u2601213479/hw requirement/figures")


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    avg = sum(values) / len(values)
    var = sum((v - avg) ** 2 for v in values) / len(values)
    return avg, math.sqrt(var)


def collect_matrix():
    cells = defaultdict(lambda: defaultdict(list))
    for path in sorted(OUT.glob("m08_tp*_s*_b*_r*.json")):
        if ".rank" in path.name:
            continue
        data = load(path)
        if data is None or "prefill_ms" not in data:
            continue
        # m08_tp1_s32_b1_r0
        parts = path.stem.split("_")
        tp = 1 if "tp1" in path.stem else 2
        seq = int(parts[2][1:])
        batch = int(parts[3][1:])
        key = (tp, seq, batch)
        rank_path = path.with_name(path.stem + ".rank1.json")
        rank1 = load(rank_path) if tp == 2 else None
        cells[key]["prefill"].append(data["prefill_ms"])
        cells[key]["tpot"].append(data["tpot_ms"])
        cells[key]["e2e"].append(data["e2e_tokens_per_s"])
        per_sequence = data["decode_tokens_per_s"]
        aggregate = data.get("aggregate_decode_tokens_per_s")
        if aggregate is None:
            aggregate = per_sequence * batch
        cells[key]["decode_tps"].append(per_sequence)
        cells[key]["agg_decode"].append(aggregate)
        cells[key]["prefill_tps"].append(data["prefill_tokens_per_s"])
        cells[key]["hbm0"].append(data["allocated_bytes"] / 2**20)
        if rank1:
            cells[key]["hbm1"].append(rank1["allocated_bytes"] / 2**20)
        cells[key]["param"].append(data["parameter_bytes"] / 2**20)
        cells[key]["kv"].append(data.get("kv_bytes", 0) / 2**20)
        cells[key]["gdn"].append(data.get("gdn_state_bytes", 0) / 2**20)
    table = []
    for key, series in sorted(cells.items()):
        tp, seq, batch = key
        row = {"tp": tp, "seq": seq, "batch": batch, "n": len(series["prefill"])}
        for name, values in series.items():
            avg, std = mean_std(values)
            row[name] = avg
            row[name + "_std"] = std
        table.append(row)
    return table


def _series_values(item):
    return item[1]


def _series_std(item):
    return item[2] if len(item) > 2 else None


def svg_grouped_bars(title, categories, series, ylabel, path: Path):
    width, height = 760, 440
    left, right, top, bottom = 70, 20, 50, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    flat = []
    for item in series:
        values = _series_values(item)
        stds = _series_std(item)
        for i, value in enumerate(values):
            if value != value:
                continue
            err = 0.0 if stds is None or stds[i] != stds[i] else stds[i]
            flat.append(value + err)
    ymax = max(flat) * 1.18 if flat else 1
    colors = ["#3b6d11", "#185fa5", "#993c1d"]
    n = len(categories)
    k = len(series)
    gap = plot_w / max(n, 1)
    bar_w = gap / (k + 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="16">{title}</text>',
        f'<text x="18" y="{top + plot_h/2}" transform="rotate(-90 18 {top + plot_h/2})" font-family="sans-serif" font-size="12">{ylabel}</text>',
    ]
    for tick in range(5):
        value = ymax * tick / 4
        y = top + plot_h - plot_h * tick / 4
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.1f}</text>')
    for i, cat in enumerate(categories):
        for j, item in enumerate(series):
            values = _series_values(item)
            stds = _series_std(item)
            value = values[i]
            if value != value:
                continue
            h = plot_h * value / ymax
            x = left + i * gap + (j + 0.5) * bar_w
            bw = bar_w * 0.85
            y = top + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{colors[j % len(colors)]}"/>')
            if stds is not None and stds[i] == stds[i] and stds[i] > 0:
                err = stds[i]
                cx = x + bw / 2
                y_hi = top + plot_h - plot_h * min(value + err, ymax) / ymax
                y_lo = top + plot_h - plot_h * max(value - err, 0) / ymax
                parts.append(f'<line x1="{cx:.1f}" y1="{y_hi:.1f}" x2="{cx:.1f}" y2="{y_lo:.1f}" stroke="#222" stroke-width="1.2"/>')
                parts.append(f'<line x1="{cx-4:.1f}" y1="{y_hi:.1f}" x2="{cx+4:.1f}" y2="{y_hi:.1f}" stroke="#222" stroke-width="1.2"/>')
                parts.append(f'<line x1="{cx-4:.1f}" y1="{y_lo:.1f}" x2="{cx+4:.1f}" y2="{y_lo:.1f}" stroke="#222" stroke-width="1.2"/>')
        parts.append(f'<text x="{left + i*gap + gap/2:.1f}" y="{height-36}" text-anchor="middle" font-family="sans-serif" font-size="11">{cat}</text>')
    for j, item in enumerate(series):
        x = left + j * 160
        parts.append(f'<rect x="{x}" y="{height-20}" width="12" height="12" fill="{colors[j % len(colors)]}"/>')
        parts.append(f'<text x="{x+18}" y="{height-10}" font-family="sans-serif" font-size="12">{item[0]}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def svg_layer_lines(title, series, path: Path):
    """series items are (label, nrmse values). X axis is relative depth."""
    width, height = 780, 480
    left, right, top, bottom = 72, 36, 44, 112
    plot_w = width - left - right
    plot_h = height - top - bottom
    plot_bottom = top + plot_h
    flat = [v for _, ys in series for v in ys]
    ymax = max(flat) * 1.12 if flat else 1
    colors = ["#185fa5", "#993c1d", "#3b6d11"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="26" text-anchor="middle" font-family="sans-serif" font-size="16">{title}</text>',
        f'<text x="16" y="{top + plot_h/2}" transform="rotate(-90 16 {top + plot_h/2})" font-family="sans-serif" font-size="12">NRMSE</text>',
        f'<text x="{left + plot_w/2}" y="{plot_bottom + 40}" text-anchor="middle" font-family="sans-serif" font-size="12">相对深度</text>',
    ]
    for tick in range(5):
        value = ymax * tick / 4
        y = top + plot_h - plot_h * tick / 4
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.3f}</text>')
        depth = tick / 4
        x = left + plot_w * depth
        parts.append(f'<text x="{x:.1f}" y="{plot_bottom + 18}" text-anchor="middle" font-family="sans-serif" font-size="11">{depth:.2f}</text>')
    legend_y = plot_bottom + 72
    legend_x = left
    for j, (label, ys) in enumerate(series):
        n = max(len(ys) - 1, 1)
        coords = []
        for i, value in enumerate(ys):
            x = left + plot_w * (i / n)
            y = top + plot_h - plot_h * value / ymax
            coords.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{colors[j % len(colors)]}" stroke-width="2"/>')
        parts.append(f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 22}" y2="{legend_y - 4}" stroke="{colors[j % len(colors)]}" stroke-width="2"/>')
        parts.append(f'<text x="{legend_x + 28}" y="{legend_y}" font-family="sans-serif" font-size="12">{label}</text>')
        legend_x += 250
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def timeline_svg(spans, path: Path, title: str, wall_ms: float):
    shown = [s for s in spans if "start_ms" in s and "end_ms" in s]
    if not shown or wall_ms <= 0:
        return
    width, height = 920, 300
    left, top = 118, 42
    plot_w = width - left - 24
    lane_names = ["未打点计算", "行并行 GEMM", "AllReduce", "等待"]
    kind_lane = {"compute": 1, "comm": 2, "wait": 3}
    colors = {"compute": "#3b6d11", "comm": "#185fa5", "wait": "#993c1d", "gap": "#c5c1b7"}
    intervals = sorted((s["start_ms"], s["end_ms"]) for s in shown)
    merged = []
    for start, end in intervals:
        start = max(0.0, min(start, wall_ms))
        end = max(0.0, min(end, wall_ms))
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    gaps = []
    cursor = 0.0
    for start, end in merged:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < wall_ms:
        gaps.append((cursor, wall_ms))
    gap_ms = sum(end - start for start, end in gaps)

    def xpos(ms):
        return left + plot_w * ms / wall_ms

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="16" y="22" font-family="sans-serif" font-size="14">{title}</text>',
        f'<text x="16" y="38" font-family="sans-serif" font-size="12">未打点计算 {gap_ms:.1f} ms，横轴 0 到 {wall_ms:.1f} ms</text>',
    ]
    for lane, name in enumerate(lane_names):
        y = top + 16 + lane * 52
        parts.append(f'<text x="8" y="{y+14}" font-family="sans-serif" font-size="12">{name}</text>')
    for start, end in gaps:
        x = xpos(start)
        w = max(xpos(end) - x, 0.6)
        y = top + 16
        parts.append(f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="16" fill="{colors["gap"]}"/>')
    for span in shown:
        lane = kind_lane.get(span["kind"])
        if lane is None:
            continue
        start = max(0.0, min(span["start_ms"], wall_ms))
        end = max(0.0, min(span["end_ms"], wall_ms))
        x = xpos(start)
        w = max(xpos(end) - x, 0.6)
        y = top + 16 + lane * 52
        parts.append(f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="16" fill="{colors[span["kind"]]}"/>')
    for tick in range(5):
        ms = wall_ms * tick / 4
        x = xpos(ms)
        parts.append(f'<text x="{x:.1f}" y="{height-16}" text-anchor="middle" font-family="sans-serif" font-size="11">{ms:.0f} ms</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))
    return gap_ms


def span_sums(data: dict) -> dict[str, float]:
    totals = {"comm": 0.0, "compute": 0.0, "wait": 0.0}
    for span in data.get("timeline") or []:
        totals[span["kind"]] = totals.get(span["kind"], 0.0) + span["ms"]
    return totals


def collect_models():
    rows = []
    for size in ("4B", "9B", "27B"):
        tp2 = load(OUT / f"m{size}_s32_b1.json")
        tp1 = load(OUT / f"m{size}_tp1_try.json")
        rank1 = load(OUT / f"m{size}_s32_b1.rank1.json")
        if tp2 is None:
            continue
        rows.append(
            {
                "size": size,
                "prefill": tp2["prefill_ms"],
                "tpot": tp2["tpot_ms"],
                "decode_tps": tp2["decode_tokens_per_s"],
                "hbm0": tp2["allocated_bytes"] / 2**20,
                "hbm1": None if rank1 is None else rank1["allocated_bytes"] / 2**20,
                "param_tp2": tp2["parameter_bytes"] / 2**20,
                "param_tp1": None if tp1 is None else tp1["parameter_bytes"] / 2**20,
                "hbm_tp1": None if tp1 is None else tp1["allocated_bytes"] / 2**20,
                "tokens": tp2.get("tokens"),
                "text": tp2.get("text"),
            }
        )
    return rows


def write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    """Write an RGB PNG. pixels is width*height*3, row-major."""
    import struct
    import zlib

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        raw.extend(pixels[y * stride : (y + 1) * stride])
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


FONT = {
    "0": [" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
    "1": ["  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],
    "2": [" ### ", "#   #", "    #", "  ## ", " #   ", "#    ", "#####"],
    "3": [" ### ", "#   #", "    #", "  ## ", "    #", "#   #", " ### "],
    "4": ["   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "],
    "5": ["#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "],
    "6": [" ### ", "#    ", "#    ", "#### ", "#   #", "#   #", " ### "],
    "7": ["#####", "    #", "   # ", "  #  ", "  #  ", "  #  ", "  #  "],
    "8": [" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "],
    "9": [" ### ", "#   #", "#   #", " ####", "    #", "    #", " ### "],
    "x": ["     ", "     ", "#   #", " # # ", "  #  ", " # # ", "#   #"],
    ".": ["     ", "     ", "     ", "     ", "     ", "     ", "  #  "],
    "-": ["     ", "     ", "     ", " ### ", "     ", "     ", "     "],
    " ": ["     ", "     ", "     ", "     ", "     ", "     ", "     "],
}


def png_grouped_bars(title, categories, series, ylabel, path: Path):
    width, height = 900, 520
    left, right, top, bottom = 78, 24, 56, 88
    plot_w = width - left - right
    plot_h = height - top - bottom
    flat = [v for item in series for v in item[1] if v == v]
    ymax = max(flat) * 1.18 if flat else 1
    colors = [(59, 109, 17), (24, 95, 165), (153, 60, 29)]
    pixels = bytearray([255]) * (width * height * 3)

    def put(x, y, rgb):
        x = int(x)
        y = int(y)
        if 0 <= x < width and 0 <= y < height:
            i = (y * width + x) * 3
            pixels[i : i + 3] = bytes(rgb)

    def fill_rect(x0, y0, x1, y1, rgb):
        for y in range(int(y0), int(y1)):
            for x in range(int(x0), int(x1)):
                put(x, y, rgb)

    def text(x, y, message, rgb=(30, 30, 30)):
        cursor = int(x)
        for ch in message:
            bits = FONT.get(ch, FONT[" "])
            for row, line in enumerate(bits):
                for col, pixel in enumerate(line):
                    if pixel == "#":
                        put(cursor + col, y + row, rgb)
            cursor += 6

    fill_rect(0, 0, width, height, (255, 255, 255))
    for tick in range(5):
        y = top + plot_h - plot_h * tick / 4
        fill_rect(left, y, left + plot_w, y + 1, (230, 230, 230))
    n = max(len(categories), 1)
    k = max(len(series), 1)
    gap = plot_w / n
    bar_w = gap / (k + 1)
    for i in range(n):
        for j, item in enumerate(series):
            values = item[1]
            value = values[i]
            if value != value:
                continue
            h = plot_h * value / ymax
            x = left + i * gap + (j + 0.55) * bar_w
            fill_rect(x, top + plot_h - h, x + bar_w * 0.8, top + plot_h, colors[j % len(colors)])
            label = f"{value:.0f}" if value >= 10 else f"{value:.1f}"
            text(x, max(top, top + plot_h - h - 10), label)
        text(left + i * gap + 8, height - 52, categories[i])
    for tick in range(5):
        value = ymax * tick / 4
        y = top + plot_h - plot_h * tick / 4
        text(8, y - 3, f"{value:.0f}")
    for j, item in enumerate(series):
        x = left + j * 220
        fill_rect(x, height - 22, x + 14, height - 10, colors[j % len(colors)])
        text(x + 18, height - 24, str(j + 1))
    write_png(path, width, height, pixels)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    table = collect_matrix()
    (OUT / "matrix.json").write_text(json.dumps(table, indent=2))
    if not table:
        print("no matrix rows")
        return
    cats = []
    prefill = {1: [], 2: []}
    tpot = {1: [], 2: []}
    decode = {1: [], 2: []}
    hbm = {1: [], 2: []}
    prefill_std = {1: [], 2: []}
    tpot_std = {1: [], 2: []}
    decode_std = {1: [], 2: []}
    hbm_std = {1: [], 2: []}
    keys = []
    for seq in (32, 512):
        for batch in (1, 4):
            keys.append((seq, batch))
            cats.append(f"{seq}x{batch}")

    def cell(row, name):
        if row is None:
            return float("nan")
        return row[name]

    for seq, batch in keys:
        for tp in (1, 2):
            row = next((r for r in table if r["tp"] == tp and r["seq"] == seq and r["batch"] == batch), None)
            prefill[tp].append(cell(row, "prefill"))
            tpot[tp].append(cell(row, "tpot"))
            decode[tp].append(cell(row, "agg_decode"))
            hbm[tp].append(cell(row, "hbm0"))
            prefill_std[tp].append(cell(row, "prefill_std"))
            tpot_std[tp].append(cell(row, "tpot_std"))
            decode_std[tp].append(cell(row, "agg_decode_std"))
            hbm_std[tp].append(cell(row, "hbm0_std"))
    charts = [
        ("Qwen3.5-0.8B prefill latency", cats, [("TP=1", prefill[1], prefill_std[1]), ("TP=2", prefill[2], prefill_std[2])], "ms", "prefill"),
        ("Qwen3.5-0.8B decode time per token", cats, [("TP=1", tpot[1], tpot_std[1]), ("TP=2", tpot[2], tpot_std[2])], "ms / token", "tpot"),
        ("Qwen3.5-0.8B aggregate decode throughput", cats, [("TP=1", decode[1], decode_std[1]), ("TP=2", decode[2], decode_std[2])], "tokens / s", "decode_tps"),
        ("Qwen3.5-0.8B HBM allocated per chip", cats, [("TP=1", hbm[1], hbm_std[1]), ("TP=2", hbm[2], hbm_std[2])], "MiB", "hbm"),
    ]
    for title, categories, series, ylabel, stem in charts:
        svg_grouped_bars(title, categories, series, ylabel, FIG / f"{stem}.svg")
        png_grouped_bars(title, categories, series, ylabel, FIG / f"{stem}.png")

    ocats = []
    owall = {"sync": [], "overlap": []}
    ocomm = {"sync": [], "overlap": []}
    otpot = {"sync": [], "overlap": []}
    for seq, batch in keys:
        ocats.append(f"{seq}x{batch}")
        for kind in ("sync", "overlap"):
            data = load(OUT / f"m08_{kind}_s{seq}_b{batch}.json")
            owall[kind].append(float("nan") if data is None else data["prefill_ms"])
            otpot[kind].append(float("nan") if data is None else data["tpot_ms"])
            ocomm[kind].append(float("nan") if data is None else span_sums(data)["comm"])
    svg_grouped_bars(
        "0.8B overlap experiment: prefill wall time",
        ocats,
        [("sync", owall["sync"]), ("overlap", owall["overlap"])],
        "ms",
        FIG / "overlap_prefill.svg",
    )
    png_grouped_bars(
        "0.8B overlap experiment: prefill wall time",
        ocats,
        [("sync", owall["sync"]), ("overlap", owall["overlap"])],
        "ms",
        FIG / "overlap_prefill.png",
    )
    svg_grouped_bars(
        "0.8B recorded AllReduce span time (not wall clock)",
        ocats,
        [("sync", ocomm["sync"]), ("overlap", ocomm["overlap"])],
        "sum of span ms",
        FIG / "overlap_comm.svg",
    )
    png_grouped_bars(
        "0.8B recorded AllReduce span time",
        ocats,
        [("sync", ocomm["sync"]), ("overlap", ocomm["overlap"])],
        "sum of span ms",
        FIG / "overlap_comm.png",
    )
    svg_grouped_bars(
        "0.8B overlap experiment: decode time per token",
        ocats,
        [("sync", otpot["sync"]), ("overlap", otpot["overlap"])],
        "ms / token",
        FIG / "overlap_tpot.svg",
    )
    png_grouped_bars(
        "0.8B overlap experiment: decode time per token",
        ocats,
        [("sync", otpot["sync"]), ("overlap", otpot["overlap"])],
        "ms / token",
        FIG / "overlap_tpot.png",
    )

    models = collect_models()
    small_tp1 = load(OUT / "m08_tp1_s32_b1_r0.json")
    small_tp2 = load(OUT / "m08_tp2_s32_b1_r0.json")
    if small_tp1 and small_tp2:
        models = [
            {
                "size": "0.8B",
                "prefill": small_tp2["prefill_ms"],
                "tpot": small_tp2["tpot_ms"],
                "param_tp1": small_tp1["parameter_bytes"] / 2**20,
                "param_tp2": small_tp2["parameter_bytes"] / 2**20,
            }
        ] + models
    (OUT / "models.json").write_text(json.dumps(models, indent=2))
    if models:
        latency = [row for row in models if row["size"] != "0.8B" and "prefill" in row]
        svg_grouped_bars(
            "TP=2 prefill and TPOT, seq 32 batch 1",
            [row["size"] for row in latency],
            [
                ("prefill", [row["prefill"] for row in latency]),
                ("TPOT", [row["tpot"] for row in latency]),
            ],
            "ms",
            FIG / "models_latency.svg",
        )
        png_grouped_bars(
            "TP=2 prefill and TPOT, seq 32 batch 1",
            [row["size"] for row in latency],
            [
                ("prefill", [row["prefill"] for row in latency]),
                ("TPOT", [row["tpot"] for row in latency]),
            ],
            "ms",
            FIG / "models_latency.png",
        )
        svg_grouped_bars(
            "每卡参数量，含整份词表",
            [row["size"] for row in models],
            [
                ("TP=1", [row["param_tp1"] if row["param_tp1"] is not None else float("nan") for row in models]),
                ("TP=2", [row["param_tp2"] for row in models]),
            ],
            "MiB",
            FIG / "models_params.svg",
        )
        png_grouped_bars(
            "params per chip, full vocabulary included",
            [row["size"] for row in models],
            [
                ("TP=1", [row["param_tp1"] if row["param_tp1"] is not None else float("nan") for row in models]),
                ("TP=2", [row["param_tp2"] for row in models]),
            ],
            "MiB",
            FIG / "models_params.png",
        )
    curves = load(OUT / "layer_error.json") or []
    by_seq = defaultdict(list)
    for row in curves:
        by_seq[row["seq_len"]].append(row)
    for seq, rows in by_seq.items():
        svg_layer_lines(
            f"逐层 hidden NRMSE，序列长度 {seq}",
            [(f'{row["model"]}，序列长度 {seq}', [layer["nrmse"] for layer in row["layers"]]) for row in rows],
            FIG / f"layer_nrmse_seq{seq}.svg",
        )
    overlap = load(OUT / "m08_overlap_s32_b1.json")
    if overlap and overlap.get("timeline"):
        gap_ms = timeline_svg(
            overlap["timeline"],
            FIG / "timeline.svg",
            "0.8B overlap prefill, 32 x 1, full wall clock",
            overlap["prefill_ms"],
        )
        print(f"timeline gap_ms {gap_ms:.1f} wall {overlap['prefill_ms']:.1f}")
    print(f"rows {len(table)} models {len(models)} figures in {FIG}")


if __name__ == "__main__":
    main()
