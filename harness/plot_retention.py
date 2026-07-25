"""Render the retention-ratio study (claim 3) as Figure-3-style heatmaps.

One figure per dataset: five 5x5 NDCG@20 heatmaps (aggregate + UU/II/UI/IU) over
the (q, q') retention grid, single-hue sequential ramp, per-dataset shared scale,
best cell of each panel outlined. Emits plotly HTML (plotly.js via CDN) plus the
raw values as CSV for the figure cell's --raw attachment.

Run:  .venv/bin/python harness/plot_retention.py harness/retention_ml1m.json out.html out.csv
"""

import csv
import json
import sys

import plotly.graph_objects as go
from plotly.subplots import make_subplots

RAMP = [
    [0.0, "#f2f6fb"], [0.25, "#c4d7ec"], [0.5, "#8fb3d9"],
    [0.75, "#5585bd"], [1.0, "#1f4e8c"],
]
PANELS = ["ALL", "UU", "II", "UI", "IU"]
TITLES = {
    "ALL": "Aggregate (all pair types)", "UU": "User-User", "II": "Item-Item",
    "UI": "User-Item", "IU": "Item-User",
}


def main(json_path, html_path, csv_path):
    blob = json.load(open(json_path))
    grid = blob["grid"]
    labels = [f"{g:g}%" for g in grid]
    res = blob["results"]

    vmin = min(min(d.values()) for d in res.values())
    vmax = max(max(d.values()) for d in res.values())
    fig = make_subplots(
        rows=1, cols=5, subplot_titles=[TITLES[p] for p in PANELS],
        horizontal_spacing=0.035,
    )
    for c, panel in enumerate(PANELS, start=1):
        z = [[res[panel][f"{q}/{qp}"] for qp in grid] for q in grid]
        best = max(res[panel], key=res[panel].get)
        bq, bqp = (float(x) for x in best.split("/"))
        fig.add_trace(
            go.Heatmap(
                z=z, x=labels, y=labels, zmin=vmin, zmax=vmax, colorscale=RAMP,
                showscale=(c == 5), colorbar=dict(title="NDCG@20", thickness=12),
                text=[[f"{v:.3f}" for v in row] for row in z],
                texttemplate="%{text}", textfont=dict(size=8),
                hovertemplate=("user q=%{y}, item q'=%{x}<br>"
                               "NDCG@20=%{z:.5f}<extra>" + panel + "</extra>"),
            ),
            row=1, col=c,
        )
        fig.add_shape(
            type="rect", row=1, col=c,
            x0=grid.index(bqp) - 0.5, x1=grid.index(bqp) + 0.5,
            y0=grid.index(bq) - 0.5, y1=grid.index(bq) + 0.5,
            line=dict(color="#b3261e", width=2),
        )
        fig.update_xaxes(title_text="item retention q'" if c == 3 else None,
                         row=1, col=c, tickfont=dict(size=9))
        fig.update_yaxes(title_text="user retention q" if c == 1 else None,
                         row=1, col=c, tickfont=dict(size=9),
                         showticklabels=(c == 1))
    fig.update_layout(
        title=(f"Retention-ratio study, {blob['dataset']}: NDCG@20 of the Eq. 4 "
               "heuristic over (q, q'); red box = panel optimum"),
        height=340, width=1500, margin=dict(t=80, b=60),
        font=dict(size=11), plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.write_html(html_path, include_plotlyjs="cdn")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "user_q_percent", "item_q_percent", "ndcg_at_20"])
        for panel in PANELS:
            for q in grid:
                for qp in grid:
                    w.writerow([panel, q, qp, res[panel][f"{q}/{qp}"]])
    print(f"written {html_path} and {csv_path}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
