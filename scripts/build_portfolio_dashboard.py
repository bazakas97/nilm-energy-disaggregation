#!/usr/bin/env python3
import argparse
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>REEFLEX NILM Demo</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
    <link rel="stylesheet" href="./styles.css" />
  </head>
  <body>
    <div class="page-shell">
      <header class="hero">
        <div class="hero-copy">
          <p class="eyebrow">Portfolio Demo</p>
          <h1>Interactive NILM Showcase</h1>
          <p class="hero-text">
            Explore disaggregation results across multiple homes, dates, and devices using
            the current REEFLEX NILM model. Each chart is a real interactive Plotly export.
          </p>
        </div>
        <div class="hero-stats">
          <div class="stat-card">
            <span class="stat-label">Model</span>
            <strong id="hero-model">-</strong>
          </div>
          <div class="stat-card">
            <span class="stat-label">Dates</span>
            <strong id="hero-dates">-</strong>
          </div>
          <div class="stat-card">
            <span class="stat-label">Homes</span>
            <strong id="hero-houses">-</strong>
          </div>
        </div>
      </header>

      <main class="dashboard-grid">
        <section class="control-panel panel">
          <div class="panel-header">
            <p class="eyebrow">Controls</p>
            <h2>Choose a view</h2>
          </div>
          <label class="field">
            <span>Date</span>
            <select id="date-select"></select>
          </label>
          <label class="field">
            <span>House</span>
            <select id="house-select"></select>
          </label>
          <label class="field">
            <span>Device</span>
            <select id="device-select"></select>
          </label>
          <div class="meta-block">
            <div>
              <span class="meta-label">Available devices</span>
              <p id="available-devices" class="meta-value">-</p>
            </div>
            <div>
              <span class="meta-label">Plot path</span>
              <p id="plot-path" class="meta-value codeish">-</p>
            </div>
          </div>
          <a id="open-plot-link" class="link-button" target="_blank" rel="noreferrer">Open plot in new tab</a>
        </section>

        <section class="metrics-panel panel">
          <div class="panel-header">
            <p class="eyebrow">Metrics</p>
            <h2>Current selection</h2>
          </div>
          <div class="metric-grid" id="summary-metrics"></div>
          <div class="device-metrics">
            <h3>Device-level metrics</h3>
            <div class="metric-grid metric-grid-compact" id="device-metrics"></div>
          </div>
        </section>

        <section class="featured-panel panel">
          <div class="panel-header">
            <p class="eyebrow">Highlights</p>
            <h2>Featured examples</h2>
          </div>
          <div id="featured-cards" class="featured-cards"></div>
        </section>

        <section class="plot-panel panel plot-panel-wide">
          <div class="plot-head">
            <div>
              <p class="eyebrow">Interactive Plot</p>
              <h2 id="plot-title">-</h2>
            </div>
            <p id="plot-subtitle" class="plot-subtitle">-</p>
          </div>
          <iframe id="plot-frame" title="Plotly dashboard chart"></iframe>
        </section>
      </main>
    </div>

    <script src="./app.js"></script>
  </body>
</html>
"""


STYLES_CSS = """:root {
  --bg: #f4efe4;
  --paper: rgba(255, 251, 245, 0.88);
  --ink: #1b1b18;
  --muted: #5b594f;
  --accent: #c44f2a;
  --accent-deep: #7d2617;
  --line: rgba(27, 27, 24, 0.12);
  --shadow: 0 18px 50px rgba(76, 51, 24, 0.14);
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  font-family: "Space Grotesk", sans-serif;
  color: var(--ink);
  background:
    radial-gradient(circle at top left, rgba(250, 221, 177, 0.85), transparent 28%),
    radial-gradient(circle at top right, rgba(211, 97, 63, 0.22), transparent 24%),
    linear-gradient(180deg, #f8f3ea 0%, #efe4d5 100%);
}

.page-shell {
  max-width: 1520px;
  margin: 0 auto;
  padding: 28px;
}

.hero {
  display: grid;
  grid-template-columns: 1.8fr 1fr;
  gap: 20px;
  margin-bottom: 20px;
}

.panel,
.hero-copy,
.hero-stats {
  background: var(--paper);
  backdrop-filter: blur(10px);
  border: 1px solid var(--line);
  border-radius: 24px;
  box-shadow: var(--shadow);
}

.hero-copy {
  padding: 28px;
}

.hero-copy h1 {
  margin: 0;
  font-size: clamp(2rem, 4vw, 4rem);
  line-height: 0.95;
  letter-spacing: -0.04em;
}

.hero-text {
  max-width: 62ch;
  font-size: 1.02rem;
  line-height: 1.65;
  color: var(--muted);
}

.hero-stats {
  display: grid;
  grid-template-columns: 1fr;
  gap: 12px;
  padding: 18px;
}

.stat-card {
  padding: 16px 18px;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.55);
  border: 1px solid rgba(196, 79, 42, 0.12);
}

.stat-label,
.eyebrow,
.field span,
.meta-label {
  font-family: "IBM Plex Mono", monospace;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.72rem;
}

.eyebrow {
  color: var(--accent-deep);
  margin: 0 0 10px;
}

.dashboard-grid {
  display: grid;
  grid-template-columns: 320px 360px 1fr;
  gap: 20px;
}

.panel {
  padding: 22px;
}

.panel-header h2,
.plot-head h2 {
  margin: 0;
  font-size: 1.35rem;
}

.field {
  display: grid;
  gap: 8px;
  margin-bottom: 16px;
}

select {
  width: 100%;
  padding: 14px 16px;
  border-radius: 14px;
  border: 1px solid rgba(27, 27, 24, 0.12);
  background: rgba(255, 255, 255, 0.9);
  color: var(--ink);
  font: inherit;
}

.meta-block {
  display: grid;
  gap: 12px;
  margin: 18px 0;
  padding: 14px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.55);
}

.meta-value {
  margin: 4px 0 0;
  color: var(--muted);
  line-height: 1.45;
}

.codeish {
  font-family: "IBM Plex Mono", monospace;
  word-break: break-all;
  font-size: 0.8rem;
}

.link-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  padding: 14px 16px;
  border-radius: 14px;
  background: linear-gradient(135deg, #c44f2a 0%, #7d2617 100%);
  color: #fff;
  text-decoration: none;
  font-weight: 700;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.metric-grid-compact {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.metric-card,
.featured-card {
  padding: 14px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(27, 27, 24, 0.08);
}

.metric-card strong {
  display: block;
  margin-top: 6px;
  font-size: 1.25rem;
}

.metric-card small,
.featured-card small {
  color: var(--muted);
}

.featured-cards {
  display: grid;
  gap: 12px;
}

.featured-card button {
  margin-top: 10px;
  padding: 10px 12px;
  border: 0;
  border-radius: 12px;
  background: rgba(196, 79, 42, 0.12);
  color: var(--accent-deep);
  font: inherit;
  font-weight: 700;
  cursor: pointer;
}

.plot-panel-wide {
  grid-column: 2 / 4;
}

.plot-head {
  display: flex;
  justify-content: space-between;
  align-items: end;
  gap: 16px;
  margin-bottom: 14px;
}

.plot-subtitle {
  margin: 0;
  color: var(--muted);
}

#plot-frame {
  width: 100%;
  min-height: 78vh;
  border: 1px solid rgba(27, 27, 24, 0.08);
  border-radius: 18px;
  background: #fff;
}

@media (max-width: 1180px) {
  .hero,
  .dashboard-grid {
    grid-template-columns: 1fr;
  }

  .plot-panel-wide {
    grid-column: auto;
  }

  .metric-grid-compact {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 680px) {
  .page-shell {
    padding: 16px;
  }

  .panel,
  .hero-copy,
  .hero-stats {
    border-radius: 18px;
  }

  .metric-grid,
  .metric-grid-compact {
    grid-template-columns: 1fr;
  }

  #plot-frame {
    min-height: 62vh;
  }
}
"""


APP_JS = """async function main() {
  const response = await fetch("./manifest.json");
  const manifest = await response.json();

  const dateSelect = document.getElementById("date-select");
  const houseSelect = document.getElementById("house-select");
  const deviceSelect = document.getElementById("device-select");
  const frame = document.getElementById("plot-frame");
  const plotTitle = document.getElementById("plot-title");
  const plotSubtitle = document.getElementById("plot-subtitle");
  const availableDevices = document.getElementById("available-devices");
  const plotPath = document.getElementById("plot-path");
  const openPlotLink = document.getElementById("open-plot-link");
  const summaryMetrics = document.getElementById("summary-metrics");
  const deviceMetrics = document.getElementById("device-metrics");
  const featuredCards = document.getElementById("featured-cards");

  document.getElementById("hero-model").textContent = manifest.model_label;
  document.getElementById("hero-dates").textContent = String(manifest.dates.length);
  document.getElementById("hero-houses").textContent = String(manifest.house_count);

  const state = {
    date: manifest.default_selection.date,
    house: manifest.default_selection.house,
    device: manifest.default_selection.device,
  };

  function fmt(value, digits = 3) {
    if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
    const num = Number(value);
    if (!Number.isFinite(num)) return "n/a";
    return num.toFixed(digits);
  }

  function populateFeatured() {
    featuredCards.innerHTML = "";
    manifest.featured_examples.forEach((item) => {
      const card = document.createElement("article");
      card.className = "featured-card";
      card.innerHTML = `
        <small>${item.date} · ${item.house}</small>
        <h3>${item.device_label}</h3>
        <p>F1 ${fmt(item.on_off_f1)} · Precision ${fmt(item.on_off_precision)} · Recall ${fmt(item.on_off_recall)}</p>
      `;
      const button = document.createElement("button");
      button.textContent = "Open this example";
      button.addEventListener("click", () => {
        state.date = item.date;
        state.house = item.house;
        state.device = item.device;
        syncControls();
        render();
      });
      card.appendChild(button);
      featuredCards.appendChild(card);
    });
  }

  function setOptions(select, values, currentValue) {
    select.innerHTML = "";
    values.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      if (value === currentValue) option.selected = true;
      select.appendChild(option);
    });
  }

  function syncControls() {
    setOptions(dateSelect, manifest.dates, state.date);
    const houses = manifest.by_date[state.date]?.houses || [];
    if (!houses.includes(state.house)) state.house = houses[0];
    setOptions(houseSelect, houses, state.house);

    const devices = manifest.by_date[state.date]?.plots?.[state.house]?.devices || [];
    if (!devices.includes(state.device)) state.device = devices[0];
    setOptions(deviceSelect, devices, state.device);
  }

  function renderMetricCards(target, entries) {
    target.innerHTML = "";
    entries.forEach((entry) => {
      const card = document.createElement("div");
      card.className = "metric-card";
      card.innerHTML = `<small>${entry.label}</small><strong>${entry.value}</strong>`;
      target.appendChild(card);
    });
  }

  function render() {
    const dateInfo = manifest.by_date[state.date];
    const houseInfo = dateInfo.plots[state.house];
    const deviceInfo = houseInfo.plot_files[state.device];
    const houseMetrics = dateInfo.metrics[state.house] || {};
    const deviceMetric = (houseMetrics.overall || {})[state.device] || {};
    const houseSummary = houseMetrics.house_summary || {};

    frame.src = deviceInfo.relative_path;
    plotTitle.textContent = `${state.house} · ${deviceInfo.device_label}`;
    plotSubtitle.textContent = `${state.date} · ${manifest.model_label}`;
    availableDevices.textContent = houseInfo.devices.map((name) => houseInfo.plot_files[name].device_label).join(", ");
    plotPath.textContent = deviceInfo.relative_path;
    openPlotLink.href = deviceInfo.relative_path;

    renderMetricCards(summaryMetrics, [
      { label: "House F1 micro", value: fmt(houseSummary.on_off_f1_micro) },
      { label: "House precision", value: fmt(houseSummary.on_off_precision_micro) },
      { label: "House recall", value: fmt(houseSummary.on_off_recall_micro) },
      { label: "Weighted MAE", value: fmt(houseSummary.mae_weighted_active, 1) },
      { label: "Weighted SAE", value: fmt(houseSummary.sae_weighted_active) },
      { label: "Devices in plot set", value: String(houseInfo.devices.length) },
    ]);

    renderMetricCards(deviceMetrics, [
      { label: "Known points", value: deviceMetric.known_points ?? "n/a" },
      { label: "Active points", value: deviceMetric.active_points ?? "n/a" },
      { label: "TECA", value: fmt(deviceMetric.teca, 1) },
      { label: "R²", value: fmt(deviceMetric.r2, 2) },
      { label: "MAE", value: fmt(deviceMetric.mae, 1) },
      { label: "SAE", value: fmt(deviceMetric.sae, 3) },
      { label: "ON/OFF F1", value: fmt(deviceMetric.on_off_f1) },
      { label: "ON/OFF precision", value: fmt(deviceMetric.on_off_precision) },
      { label: "ON/OFF recall", value: fmt(deviceMetric.on_off_recall) },
    ]);
  }

  dateSelect.addEventListener("change", () => {
    state.date = dateSelect.value;
    syncControls();
    render();
  });

  houseSelect.addEventListener("change", () => {
    state.house = houseSelect.value;
    syncControls();
    render();
  });

  deviceSelect.addEventListener("change", () => {
    state.device = deviceSelect.value;
    render();
  });

  populateFeatured();
  syncControls();
  render();
}

main();
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a static portfolio dashboard from generated daily NILM plots."
    )
    parser.add_argument(
        "--plots-root",
        default="results/plots_nilmformer_sel_full_rebuilt_60_20_20",
        help="Root directory containing per-date/per-house Plotly HTML outputs.",
    )
    parser.add_argument(
        "--metrics-dir",
        default="results/csv",
        help="Directory containing per-house metrics JSON files.",
    )
    parser.add_argument(
        "--dates",
        required=True,
        help="Comma-separated dates in YYYYMMDD form to include in the demo.",
    )
    parser.add_argument(
        "--output-dir",
        default="demo/portfolio_dashboard",
        help="Output directory for the generated static dashboard.",
    )
    parser.add_argument(
        "--model-label",
        default="NILMFormer SEL Full Rebuilt",
        help="Human-readable model label for the demo page.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the existing output directory before rebuilding.",
    )
    return parser.parse_args()


def resolve_project_dir():
    return Path(__file__).resolve().parent.parent


def resolve_path(project_dir, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_dir / path).resolve()


def normalize_dates(text):
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("At least one date is required.")
    return values


def safe_float(value):
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(num):
        return num
    return None


def device_label(device_name):
    text = str(device_name).replace("energy_", "").replace("_", " ")
    return text.title()


def copy_plot_html(src_path, dst_path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def load_metrics(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_metrics_path(metrics_dir, house, date_tag):
    pattern = f"*_{house}_{date_tag}.json"
    matches = sorted(metrics_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing metrics JSON for {house} {date_tag} in {metrics_dir}")
    return matches[0]


def collect_demo_data(plots_root, metrics_dir, output_dir, dates):
    manifest = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dates": [],
        "by_date": {},
        "featured_examples": [],
    }
    featured_pool = []

    for date_tag in dates:
        date_root = plots_root / date_tag
        if not date_root.is_dir():
            raise FileNotFoundError(f"Missing plots for date {date_tag}: {date_root}")

        houses = sorted([path.name for path in date_root.iterdir() if path.is_dir()])
        if not houses:
            raise ValueError(f"No house directories found under {date_root}")

        date_entry = {"houses": houses, "plots": {}, "metrics": {}}

        for house in houses:
            house_root = date_root / house
            plot_files = sorted(house_root.glob("*_test_plot.html"))
            if not plot_files:
                continue

            house_plot_entry = {"devices": [], "plot_files": {}}
            for plot_path in plot_files:
                device_name = plot_path.name.replace("_test_plot.html", "")
                dst_path = output_dir / "assets" / "plots" / date_tag / house / plot_path.name
                copy_plot_html(plot_path, dst_path)
                relative_path = dst_path.relative_to(output_dir).as_posix()
                house_plot_entry["devices"].append(device_name)
                house_plot_entry["plot_files"][device_name] = {
                    "device_label": device_label(device_name),
                    "relative_path": relative_path,
                }

            metrics_path = find_metrics_path(metrics_dir, house, date_tag)
            metrics = load_metrics(metrics_path)
            house_summary = (metrics.get("per_participant_summary") or {}).get(house) or {}
            date_entry["plots"][house] = house_plot_entry
            date_entry["metrics"][house] = {
                "house_summary": house_summary,
                "overall": metrics.get("overall") or {},
            }

            for device_name, device_metrics in (metrics.get("overall") or {}).items():
                f1 = safe_float(device_metrics.get("on_off_f1"))
                captured = bool(device_metrics.get("on_off_captured"))
                if f1 is None or not captured:
                    continue
                featured_pool.append(
                    {
                        "date": date_tag,
                        "house": house,
                        "device": device_name,
                        "device_label": device_label(device_name),
                        "on_off_f1": f1,
                        "on_off_precision": safe_float(device_metrics.get("on_off_precision")),
                        "on_off_recall": safe_float(device_metrics.get("on_off_recall")),
                    }
                )

        manifest["dates"].append(date_tag)
        manifest["by_date"][date_tag] = date_entry

    manifest["featured_examples"] = sorted(
        featured_pool,
        key=lambda item: (
            safe_float(item.get("on_off_f1")) or -1.0,
            safe_float(item.get("on_off_precision")) or -1.0,
        ),
        reverse=True,
    )[:6]
    return manifest


def pick_default_selection(manifest):
    if not manifest["dates"]:
        raise ValueError("Manifest does not contain any dates.")
    first_date = manifest["dates"][0]
    date_entry = manifest["by_date"][first_date]
    first_house = date_entry["houses"][0]
    first_device = date_entry["plots"][first_house]["devices"][0]
    return {"date": first_date, "house": first_house, "device": first_device}


def main():
    args = parse_args()
    project_dir = resolve_project_dir()
    plots_root = resolve_path(project_dir, args.plots_root)
    metrics_dir = resolve_path(project_dir, args.metrics_dir)
    output_dir = resolve_path(project_dir, args.output_dir)
    dates = normalize_dates(args.dates)

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = collect_demo_data(
        plots_root=plots_root,
        metrics_dir=metrics_dir,
        output_dir=output_dir,
        dates=dates,
    )
    manifest["model_label"] = args.model_label
    manifest["house_count"] = len(
        sorted(
            {
                house
                for date_info in manifest["by_date"].values()
                for house in date_info.get("houses", [])
            }
        )
    )
    manifest["default_selection"] = pick_default_selection(manifest)

    (output_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (output_dir / "styles.css").write_text(STYLES_CSS, encoding="utf-8")
    (output_dir / "app.js").write_text(APP_JS, encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Portfolio dashboard written to: {output_dir}")
    print(f"Dates included: {', '.join(dates)}")
    print(f"Houses covered: {manifest['house_count']}")


if __name__ == "__main__":
    main()
