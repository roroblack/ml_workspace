from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PNG_PATH = REPORTS / "19_mysql_erd_v6_reflected.png"
SVG_PATH = REPORTS / "19_mysql_erd_v6_reflected.svg"


W, H = 2500, 1700
BG = "#fbfbfa"
TEXT = "#1f2328"
MUTED = "#6b7280"
LINE = "#111827"
LOGICAL = "#b6c0cc"


@dataclass
class Column:
    name: str
    dtype: str
    mark: str = ""


@dataclass
class Table:
    name: str
    x: int
    y: int
    w: int
    color: str
    columns: list[Column]

    @property
    def row_h(self) -> int:
        return 34

    @property
    def header_h(self) -> int:
        return 48

    @property
    def h(self) -> int:
        return self.header_h + len(self.columns) * self.row_h + 18

    @property
    def left(self) -> tuple[int, int]:
        return (self.x, self.y + self.h // 2)

    @property
    def right(self) -> tuple[int, int]:
        return (self.x + self.w, self.y + self.h // 2)

    @property
    def top(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y)

    @property
    def bottom(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "malgunbd.ttf" if bold else "malgun.ttf"
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


F_TITLE = font(34, True)
F_SUB = font(19)
F_HEAD = font(24, True)
F_COL = font(19)
F_TYPE = font(17)
F_MARK = font(15, True)
F_SMALL = font(16)


def lighten(hex_color: str, amount: float = 0.82) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


tables = {
    "face_user": Table(
        "face_user",
        60,
        145,
        420,
        "#7e57c2",
        [
            Column("user_id", "VARCHAR(64)", "PK"),
            Column("display_name", "VARCHAR(100)"),
            Column("role", "VARCHAR(30)"),
            Column("created_at", "TIMESTAMP"),
        ],
    ),
    "model_registry": Table(
        "model_registry",
        825,
        115,
        520,
        "#2e7d32",
        [
            Column("model_id", "BIGINT", "PK"),
            Column("model_name", "VARCHAR(128)", "UK"),
            Column("model_type", "VARCHAR(32)"),
            Column("feature_schema_version", "VARCHAR(16)"),
            Column("preprocessing_config", "JSON"),
            Column("dataset_path", "TEXT"),
            Column("artifact_path", "TEXT"),
            Column("train_period", "VARCHAR(64)"),
            Column("metric_cv", "DOUBLE"),
            Column("metric_oot", "DOUBLE"),
            Column("metrics_json", "JSON"),
            Column("is_active", "TINYINT(1)"),
            Column("created_at", "TIMESTAMP"),
        ],
    ),
    "recommendation": Table(
        "recommendation",
        1910,
        120,
        500,
        "#b7791f",
        [
            Column("rec_id", "BIGINT", "PK"),
            Column("user_id", "VARCHAR(64)"),
            Column("model_id", "BIGINT", "FK"),
            Column("rec_items_json", "JSON"),
            Column("rec_categories_json", "JSON"),
            Column("created_at", "TIMESTAMP"),
        ],
    ),
    "prediction_log": Table(
        "prediction_log",
        60,
        545,
        500,
        "#2b6cb0",
        [
            Column("prediction_id", "BIGINT", "PK"),
            Column("model_id", "BIGINT", "FK"),
            Column("user_id", "VARCHAR(64)"),
            Column("session_id", "VARCHAR(128)"),
            Column("churn_probability", "DOUBLE"),
            Column("risk_level", "VARCHAR(16)"),
            Column("top_factors_json", "JSON"),
            Column("recommended_action", "TEXT"),
            Column("created_at", "TIMESTAMP"),
        ],
    ),
    "feature_user_snapshot": Table(
        "feature_user_snapshot",
        610,
        655,
        520,
        "#805ad5",
        [
            Column("snapshot_id", "BIGINT", "PK"),
            Column("user_id", "VARCHAR(64)"),
            Column("snapshot_time", "TIMESTAMP"),
            Column("cohort_flag", "TINYINT(1)"),
            Column("churn", "TINYINT(1)"),
            Column("churn_no_purchase", "TINYINT(1)"),
            Column("obs_period", "VARCHAR(64)"),
            Column("outcome_period", "VARCHAR(64)"),
            Column("feature_json", "JSON"),
            Column("created_at", "TIMESTAMP"),
        ],
    ),
    "sequence_snapshot": Table(
        "sequence_snapshot",
        1190,
        655,
        500,
        "#6b7280",
        [
            Column("snapshot_id", "BIGINT", "PK"),
            Column("user_id", "VARCHAR(64)"),
            Column("dataset_tag", "VARCHAR(32)"),
            Column("seq_len", "INT"),
            Column("n_features", "INT"),
            Column("storage_format", "VARCHAR(16)"),
            Column("artifact_path", "TEXT"),
            Column("row_index", "INT"),
            Column("label", "INT"),
            Column("created_at", "TIMESTAMP"),
        ],
    ),
    "ensemble_result": Table(
        "ensemble_result",
        1910,
        570,
        500,
        "#0f766e",
        [
            Column("ensemble_id", "BIGINT", "PK"),
            Column("user_id", "VARCHAR(64)"),
            Column("prob_ensemble", "DOUBLE"),
            Column("risk_level", "VARCHAR(16)"),
            Column("improvement_json", "JSON"),
            Column("created_at", "TIMESTAMP"),
        ],
    ),
    "face_login_log": Table(
        "face_login_log",
        60,
        1210,
        430,
        "#64748b",
        [
            Column("login_id", "BIGINT", "PK"),
            Column("user_id", "VARCHAR(64)"),
            Column("success", "TINYINT(1)"),
            Column("similarity", "DOUBLE"),
            Column("created_at", "TIMESTAMP"),
        ],
    ),
    "user_interest": Table(
        "user_interest",
        610,
        1220,
        500,
        "#c2410c",
        [
            Column("user_id", "VARCHAR(64)", "PK"),
            Column("top_category_id", "VARCHAR(64)"),
            Column("top_brand", "VARCHAR(128)"),
            Column("interest_json", "JSON"),
            Column("updated_at", "TIMESTAMP"),
        ],
    ),
    "retention_action_log": Table(
        "retention_action_log",
        1190,
        1210,
        500,
        "#be123c",
        [
            Column("action_id", "BIGINT", "PK"),
            Column("prediction_id", "BIGINT", "FK"),
            Column("action_type", "VARCHAR(50)"),
            Column("message", "TEXT"),
            Column("status", "VARCHAR(30)"),
            Column("created_at", "TIMESTAMP"),
        ],
    ),
    "ensemble_member": Table(
        "ensemble_member",
        1910,
        1175,
        500,
        "#1d4ed8",
        [
            Column("member_id", "BIGINT", "PK"),
            Column("ensemble_id", "BIGINT", "FK"),
            Column("model_id", "BIGINT", "FK"),
            Column("weight", "DOUBLE"),
            Column("prob", "DOUBLE"),
        ],
    ),
}


solid_relations = [
    ("model_registry", "prediction_log", "model_id FK", "left", "right"),
    ("model_registry", "recommendation", "model_id FK", "right", "left"),
    ("prediction_log", "retention_action_log", "prediction_id FK", "bottom", "top"),
    ("ensemble_result", "ensemble_member", "ensemble_id FK", "bottom", "top"),
]

logical_relations = [
    ("face_user", "prediction_log", "user_id logical", "bottom", "top"),
    ("face_user", "recommendation", "user_id logical", "right", "left"),
    ("face_user", "feature_user_snapshot", "user_id logical", "bottom", "top"),
    ("face_user", "sequence_snapshot", "user_id logical", "bottom", "top"),
    ("face_user", "ensemble_result", "user_id logical", "right", "left"),
    ("face_user", "user_interest", "user_id logical", "bottom", "top"),
    ("face_user", "face_login_log", "user_id logical", "bottom", "top"),
]


def anchor(table: Table, side: str) -> tuple[int, int]:
    return getattr(table, side)


def draw_marker(draw: ImageDraw.ImageDraw, xy: tuple[int, int], kind: str, color: str = LINE) -> None:
    x, y = xy
    if kind == "one":
        draw.line([(x - 8, y - 12), (x - 8, y + 12)], fill=color, width=3)
        draw.line([(x - 2, y - 12), (x - 2, y + 12)], fill=color, width=3)
    else:
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), outline=color, width=3)
        draw.line([(x + 8, y - 14), (x + 8, y + 14)], fill=color, width=3)


def route(p1: tuple[int, int], p2: tuple[int, int]) -> list[tuple[int, int]]:
    x1, y1 = p1
    x2, y2 = p2
    if abs(x1 - x2) > abs(y1 - y2):
        mid = (x1 + x2) // 2
        return [(x1, y1), (mid, y1), (mid, y2), (x2, y2)]
    mid = (y1 + y2) // 2
    return [(x1, y1), (x1, mid), (x2, mid), (x2, y2)]


def dashed_line(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], fill: str, width: int = 2, dash: int = 14) -> None:
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        dx, dy = x2 - x1, y2 - y1
        dist = max((dx * dx + dy * dy) ** 0.5, 1)
        steps = int(dist // dash)
        for i in range(steps + 1):
            if i % 2 == 0:
                a = i / max(steps, 1)
                b = min((i + 1) / max(steps, 1), 1)
                draw.line(
                    [(x1 + dx * a, y1 + dy * a), (x1 + dx * b, y1 + dy * b)],
                    fill=fill,
                    width=width,
                )


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt, fill: str = LINE) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=fnt)
    pad = 5
    draw.rounded_rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        radius=6,
        fill=BG,
        outline="#e5e7eb",
        width=1,
    )
    draw.text((x, y), text, font=fnt, fill=fill)


def draw_text_center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fnt, fill: str = TEXT) -> None:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - tw) // 2
    y = box[1] + (box[3] - box[1] - th) // 2 - 2
    draw.text((x, y), text, font=fnt, fill=fill)


def draw_table(draw: ImageDraw.ImageDraw, t: Table) -> None:
    draw.rounded_rectangle((t.x, t.y, t.x + t.w, t.y + t.h), radius=16, fill="#ffffff", outline=t.color, width=3)
    draw.rounded_rectangle(
        (t.x, t.y, t.x + t.w, t.y + t.header_h),
        radius=16,
        fill=lighten(t.color, 0.78),
        outline=t.color,
        width=3,
    )
    draw.line([(t.x, t.y + t.header_h), (t.x + t.w, t.y + t.header_h)], fill=t.color, width=3)
    draw_text_center(draw, (t.x, t.y, t.x + t.w, t.y + t.header_h), t.name, F_HEAD)

    y = t.y + t.header_h + 14
    for c in t.columns:
        if c.mark:
            fill = "#f59e0b" if c.mark in {"PK", "UK"} else "#8b5cf6"
            draw.rounded_rectangle((t.x + 18, y + 3, t.x + 58, y + 28), radius=8, fill=fill)
            draw_text_center(draw, (t.x + 18, y + 3, t.x + 58, y + 28), c.mark, F_MARK, fill="#ffffff")
            name_x = t.x + 70
        else:
            name_x = t.x + 28
        draw.text((name_x, y), c.name, font=F_COL, fill=TEXT)
        draw.text((t.x + t.w - 185, y + 2), c.dtype, font=F_TYPE, fill=TEXT)
        y += t.row_h


def draw_relations(draw: ImageDraw.ImageDraw) -> None:
    for src, dst, label, src_side, dst_side in solid_relations:
        p1 = anchor(tables[src], src_side)
        p2 = anchor(tables[dst], dst_side)
        pts = route(p1, p2)
        draw.line(pts, fill=LINE, width=4, joint="curve")
        draw_marker(draw, p1, "one", LINE)
        draw_marker(draw, p2, "many", LINE)
        lx = (pts[len(pts) // 2][0] + pts[-1][0]) // 2
        ly = (pts[len(pts) // 2][1] + pts[-1][1]) // 2
        draw_label(draw, (lx + 8, ly - 24), label, F_SMALL, LINE)

    for src, dst, label, src_side, dst_side in logical_relations:
        p1 = anchor(tables[src], src_side)
        p2 = anchor(tables[dst], dst_side)
        pts = route(p1, p2)
        dashed_line(draw, pts, LOGICAL, width=2)
        draw_marker(draw, p1, "one", LOGICAL)
        draw_marker(draw, p2, "many", LOGICAL)

    # Keep the optional model_registry -> ensemble_member.model_id relation away
    # from snapshot tables so it cannot be mistaken as their FK.
    model = tables["model_registry"]
    member = tables["ensemble_member"]
    p1 = (model.x + model.w, model.y + model.h - 76)
    p2 = (member.x, member.y + member.header_h + member.row_h * 2 + 18)
    elbow_x = member.x - 155
    pts = [p1, (elbow_x, p1[1]), (elbow_x, p2[1]), p2]
    draw.line(pts, fill=LINE, width=4, joint="curve")
    draw_marker(draw, p1, "one", LINE)
    draw_marker(draw, p2, "many", LINE)
    draw_label(draw, (elbow_x + 10, p2[1] - 34), "model_id FK 권장", F_SMALL, LINE)


def draw_legend(draw: ImageDraw.ImageDraw) -> None:
    x, y, w, h = 60, 1510, 900, 110
    draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill="#ffffff", outline="#d1d5db", width=2)
    draw.text((x + 24, y + 18), "Legend", font=F_HEAD, fill=TEXT)
    draw.line((x + 150, y + 35, x + 260, y + 35), fill=LINE, width=4)
    draw.text((x + 280, y + 23), "강제 FK 관계", font=F_SMALL, fill=TEXT)
    dashed_line(draw, [(x + 430, y + 35), (x + 540, y + 35)], fill=LOGICAL, width=3)
    draw.text((x + 560, y + 23), "face_user.user_id 논리참조(강제 FK 아님)", font=F_SMALL, fill=TEXT)
    draw.rounded_rectangle((x + 150, y + 65, x + 190, y + 90), radius=8, fill="#f59e0b")
    draw.text((x + 198, y + 65), "PK/UK", font=F_SMALL, fill=TEXT)
    draw.rounded_rectangle((x + 310, y + 65, x + 350, y + 90), radius=8, fill="#8b5cf6")
    draw.text((x + 358, y + 65), "FK 컬럼", font=F_SMALL, fill=TEXT)


def make_png() -> None:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.text((60, 40), "MySQL 운영 ERD v6 - DDL 반영본", font=F_TITLE, fill=TEXT)
    draw.text(
        (60, 86),
        "실선=강제 FK, 점선=face_user.user_id 논리참조 / model_registry는 모델·전처리·파일 경로의 단일 기준",
        font=F_SUB,
        fill=MUTED,
    )
    draw_label(draw, (500, 178), "user_id 논리참조: FK 아이콘/제약 없음", F_SMALL, LOGICAL)

    draw_relations(draw)
    for table in tables.values():
        draw_table(draw, table)
    draw_legend(draw)
    img.save(PNG_PATH)


def svg_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_line(points: list[tuple[int, int]], color: str, width: int, dashed: bool = False) -> str:
    pts = " ".join(f"{x},{y}" for x, y in points)
    dash = ' stroke-dasharray="14 12"' if dashed else ""
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{dash}/>'


def make_svg() -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{BG}"/>',
        '<style>text{font-family:"Malgun Gothic","Arial",sans-serif;fill:#1f2328}.muted{fill:#6b7280}.small{font-size:16px}.col{font-size:19px}.typ{font-size:17px}.head{font-size:24px;font-weight:700}.title{font-size:34px;font-weight:700}</style>',
        '<text x="60" y="68" class="title">MySQL 운영 ERD v6 - DDL 반영본</text>',
        '<text x="60" y="104" class="muted" font-size="19">실선=강제 FK, 점선=face_user.user_id 논리참조 / model_registry는 모델·전처리·파일 경로의 단일 기준</text>',
    ]

    for src, dst, label, src_side, dst_side in solid_relations:
        p1 = anchor(tables[src], src_side)
        p2 = anchor(tables[dst], dst_side)
        pts = route(p1, p2)
        parts.append(svg_line(pts, LINE, 4))
        parts.append(f'<text x="{pts[-1][0] + 10}" y="{pts[-1][1] - 14}" class="small">{svg_escape(label)}</text>')

    for src, dst, label, src_side, dst_side in logical_relations:
        p1 = anchor(tables[src], src_side)
        p2 = anchor(tables[dst], dst_side)
        pts = route(p1, p2)
        parts.append(svg_line(pts, LOGICAL, 3, dashed=True))

    model = tables["model_registry"]
    member = tables["ensemble_member"]
    p1 = (model.x + model.w, model.y + model.h - 76)
    p2 = (member.x, member.y + member.header_h + member.row_h * 2 + 18)
    elbow_x = member.x - 155
    pts = [p1, (elbow_x, p1[1]), (elbow_x, p2[1]), p2]
    parts.append(svg_line(pts, LINE, 4))
    parts.append(f'<text x="{elbow_x + 10}" y="{p2[1] - 18}" class="small">model_id FK 권장</text>')

    for t in tables.values():
        parts.append(f'<rect x="{t.x}" y="{t.y}" width="{t.w}" height="{t.h}" rx="16" fill="#fff" stroke="{t.color}" stroke-width="3"/>')
        parts.append(f'<rect x="{t.x}" y="{t.y}" width="{t.w}" height="{t.header_h}" rx="16" fill="{lighten(t.color, 0.78)}" stroke="{t.color}" stroke-width="3"/>')
        parts.append(f'<line x1="{t.x}" y1="{t.y + t.header_h}" x2="{t.x + t.w}" y2="{t.y + t.header_h}" stroke="{t.color}" stroke-width="3"/>')
        parts.append(f'<text x="{t.x + t.w / 2}" y="{t.y + 32}" text-anchor="middle" class="head">{svg_escape(t.name)}</text>')
        y = t.y + t.header_h + 34
        for c in t.columns:
            if c.mark:
                fill = "#f59e0b" if c.mark in {"PK", "UK"} else "#8b5cf6"
                parts.append(f'<rect x="{t.x + 18}" y="{y - 24}" width="40" height="25" rx="8" fill="{fill}"/>')
                parts.append(f'<text x="{t.x + 38}" y="{y - 6}" text-anchor="middle" font-size="15" font-weight="700" fill="#fff">{c.mark}</text>')
                nx = t.x + 70
            else:
                nx = t.x + 28
            parts.append(f'<text x="{nx}" y="{y}" class="col">{svg_escape(c.name)}</text>')
            parts.append(f'<text x="{t.x + t.w - 185}" y="{y}" class="typ">{svg_escape(c.dtype)}</text>')
            y += t.row_h

    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    REPORTS.mkdir(parents=True, exist_ok=True)
    make_png()
    make_svg()
    print(PNG_PATH)
    print(SVG_PATH)
