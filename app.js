const HOUSE_LABELS = {
  certh15crq5w: "Household A",
  certh7zcqwmc: "Household B",
  certh97dgl14: "Household C",
  certhckoz1h4: "Household D",
  certhr5fwl7p: "Household E",
  certhtwo505o: "Household F",
};

const DEVICE_LABELS = {
  energy_dish_washer:     "Dishwasher",
  energy_fridge_freezer:  "Fridge / Freezer",
  energy_induction_hob:   "Induction Hob",
  energy_oven:            "Oven",
  energy_washing_machine: "Washing Machine",
  energy_ac:              "Air Conditioning",
  energy_ewh:             "Electric Water Heater",
  energy_ev:              "EV Charger",
  energy_pv:              "Solar PV",
  energy_dryer:           "Dryer",
};

function houseLabel(id)  { return HOUSE_LABELS[id]  || id; }
function deviceLabel(key){ return DEVICE_LABELS[key] || key.replace("energy_","").replace(/_/g," "); }

function fmtDate(d) {
  const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  return `${parseInt(d.slice(6,8))} ${months[parseInt(d.slice(4,6),10)-1]} ${d.slice(0,4)}`;
}

function fmt(value, digits=3) {
  if (value===null||value===undefined||Number.isNaN(value)) return "n/a";
  const n = Number(value);
  if (!Number.isFinite(n)) return "n/a";
  return n.toFixed(digits);
}

function qualityColor(teca) {
  if (!teca && teca!==0) return "#93c5fd";
  return teca>=88 ? "#34d399" : teca>=75 ? "#fbbf24" : "#f87171";
}
function f1Color(f1) {
  if (!f1 && f1!==0) return "#93c5fd";
  return f1>=0.85 ? "#34d399" : f1>=0.65 ? "#fbbf24" : "#f87171";
}

function isDeviceGood(manifest, date, house, device) {
  const m = manifest.by_date[date]?.metrics?.[house]?.overall?.[device];
  if (!m) return false;
  // hide only if the device was genuinely never active (nothing to show)
  if (!m.active_points) return false;
  return true;
}

// ── Chart.js setup ────────────────────────────────────────────────────────────

const DATA_CACHE = {};
let chartInstance = null;
let loadedDataKey = null;
let _resetZoomBtn = null; // set after DOM ready

async function loadHouseData(date, house) {
  const key = `${date}_${house}`;
  if (DATA_CACHE[key]) return DATA_CACHE[key];
  const resp = await fetch(`./assets/data/${key}.json`);
  if (!resp.ok) throw new Error(`Failed to load ${key}.json`);
  DATA_CACHE[key] = await resp.json();
  return DATA_CACHE[key];
}

function buildChart(data, device) {
  const dev = data.devices[device];
  if (!dev) return;

  // Mask true values where not known
  const trueVals = dev.true.map((v, i) => (dev.known && dev.known[i]===0) ? null : v);
  const predVals = dev.pred;
  const mainsVals = data.mains;

  const datasets = [
    {
      label: "Ground Truth",
      data: trueVals,
      borderColor: "#3b82f6",
      backgroundColor: "rgba(59,130,246,0.07)",
      borderWidth: 2.5,
      tension: 0.35,
      pointRadius: 0,
      spanGaps: false,
      fill: true,
      order: 1,
    },
    {
      label: "Prediction",
      data: predVals,
      borderColor: "#34d399",
      backgroundColor: "transparent",
      borderWidth: 2,
      tension: 0.35,
      pointRadius: 0,
      spanGaps: true,
      order: 2,
    },
    {
      label: "Total Mains",
      data: mainsVals,
      borderColor: "rgba(148,163,184,0.22)",
      backgroundColor: "transparent",
      borderWidth: 1,
      tension: 0.2,
      pointRadius: 0,
      spanGaps: true,
      order: 3,
    },
  ];

  const commonScaleOpts = {
    grid: { color: "rgba(147,197,253,0.07)", drawBorder: false },
    ticks: { color: "#93c5fd", font: { family: "'IBM Plex Mono', monospace", size: 11 } },
    border: { color: "rgba(147,197,253,0.12)" },
  };

  const chartOpts = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 350 },
    interaction: { mode: "index", intersect: false },
    plugins: {
      zoom: {
        zoom: {
          wheel: { enabled: true },
          pinch: { enabled: true },
          mode: "x",
          onZoom: () => { if (_resetZoomBtn) _resetZoomBtn.style.display = "flex"; },
        },
        pan: {
          enabled: true,
          mode: "x",
          onPan: () => { if (_resetZoomBtn) _resetZoomBtn.style.display = "flex"; },
        },
      },
      legend: {
        position: "top",
        align: "end",
        labels: {
          color: "#dbeafe",
          font: { family: "'Space Grotesk', sans-serif", size: 13 },
          padding: 20,
          usePointStyle: true,
          pointStyleWidth: 24,
          boxHeight: 3,
        },
      },
      tooltip: {
        backgroundColor: "rgba(6,14,30,0.96)",
        borderColor: "rgba(147,197,253,0.2)",
        borderWidth: 1,
        titleColor: "#93c5fd",
        bodyColor: "#dbeafe",
        padding: 12,
        titleFont: { family: "'IBM Plex Mono', monospace", size: 12 },
        bodyFont: { family: "'IBM Plex Mono', monospace", size: 12 },
        callbacks: {
          label: (ctx) => {
            const v = ctx.parsed.y;
            if (v===null||v===undefined) return `${ctx.dataset.label}: n/a`;
            return `${ctx.dataset.label}: ${v.toFixed(1)} W`;
          },
        },
      },
    },
    scales: {
      x: {
        ...commonScaleOpts,
        ticks: {
          ...commonScaleOpts.ticks,
          maxTicksLimit: 13,
          maxRotation: 0,
        },
      },
      y: {
        ...commonScaleOpts,
        title: {
          display: true,
          text: "Power (W)",
          color: "#6b8ab5",
          font: { family: "'IBM Plex Mono', monospace", size: 11 },
        },
      },
    },
  };

  if (chartInstance) {
    chartInstance.data.labels = data.timestamps;
    chartInstance.data.datasets = datasets;
    chartInstance.options = chartOpts;
    chartInstance.update("none");
  } else {
    const canvas = document.getElementById("plot-chart");
    chartInstance = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: { labels: data.timestamps, datasets },
      options: chartOpts,
    });
  }
}

// ── App logic ─────────────────────────────────────────────────────────────────

async function main() {
  const manifest = await (await fetch("./manifest.json")).json();

  document.getElementById("hero-model").textContent  = manifest.model_label;
  document.getElementById("hero-dates").textContent  = String(manifest.dates.length);
  document.getElementById("hero-houses").textContent = String(manifest.house_count);

  const dateSelect    = document.getElementById("date-select");
  const houseSelect   = document.getElementById("house-select");
  const deviceSelect  = document.getElementById("device-select");
  const plotTitle     = document.getElementById("plot-title");
  const plotSubtitle  = document.getElementById("plot-subtitle");
  const deviceMetrics = document.getElementById("device-metrics");
  const featuredCards = document.getElementById("featured-cards");
  const tecaBadge     = document.getElementById("teca-badge");
  const tecaVal       = document.getElementById("teca-val");
  const f1Badge       = document.getElementById("f1-badge");
  const f1Val         = document.getElementById("f1-val");
  const chartLoading  = document.getElementById("chart-loading");
  _resetZoomBtn = document.getElementById("reset-zoom");
  _resetZoomBtn.addEventListener("click", () => {
    chartInstance?.resetZoom();
    _resetZoomBtn.style.display = "none";
  });

  const state = {
    date:   manifest.default_selection.date,
    house:  manifest.default_selection.house,
    device: manifest.default_selection.device,
  };

  function getGoodDevices(date, house) {
    return (manifest.by_date[date]?.plots?.[house]?.devices || [])
      .filter(d => isDeviceGood(manifest, date, house, d));
  }

  function getGoodHouses(date) {
    return (manifest.by_date[date]?.houses || [])
      .filter(h => getGoodDevices(date, h).length > 0);
  }

  function setOptions(select, items, current, labelFn) {
    select.innerHTML = "";
    items.forEach(item => {
      const opt = document.createElement("option");
      opt.value = item;
      opt.textContent = labelFn(item);
      if (item===current) opt.selected = true;
      select.appendChild(opt);
    });
  }

  function syncControls() {
    setOptions(dateSelect, manifest.dates, state.date, fmtDate);
    const houses = getGoodHouses(state.date);
    if (!houses.includes(state.house)) state.house = houses[0];
    setOptions(houseSelect, houses, state.house, houseLabel);
    const devices = getGoodDevices(state.date, state.house);
    if (!devices.includes(state.device)) state.device = devices[0];
    setOptions(deviceSelect, devices, state.device, deviceLabel);
  }

  function setBadge(el, valEl, show, text, color) {
    el.style.display = show ? "flex" : "none";
    if (!show) return;
    valEl.textContent = text;
    valEl.style.color = color;
    el.style.borderColor = color + "44";
    el.style.background   = color + "18";
  }

  function renderMetrics() {
    const dm = (manifest.by_date[state.date]?.metrics?.[state.house]?.overall || {})[state.device] || {};
    const hasTECA = dm.teca!=null;
    const hasF1   = dm.on_off_f1!=null;

    setBadge(tecaBadge, tecaVal, hasTECA, fmt(dm.teca,1)+"%", qualityColor(dm.teca));
    setBadge(f1Badge,   f1Val,   hasF1,   fmt(dm.on_off_f1,3), f1Color(dm.on_off_f1));

    const entries = [
      { label: "TECA",      value: hasTECA ? fmt(dm.teca,1)+"%" : "n/a", color: hasTECA ? qualityColor(dm.teca) : "",
        tooltip: "Total Energy Correctly Assigned — fraction of the appliance's true energy correctly attributed by the model. 100% = perfect, 50% = random." },
      { label: "MAE (W)",   value: fmt(dm.mae, 1) },
      { label: "ON/OFF F1", value: fmt(dm.on_off_f1, 3), color: hasF1 ? f1Color(dm.on_off_f1) : "" },
      { label: "Precision", value: fmt(dm.on_off_precision, 3) },
      { label: "Recall",    value: fmt(dm.on_off_recall, 3) },
    ];

    deviceMetrics.innerHTML = "";
    entries.forEach(({ label, value, color, tooltip }) => {
      const card = document.createElement("div");
      card.className = "metric-card";
      const tip = tooltip ? ` <span class="metric-tip" title="${tooltip}">?</span>` : "";
      const sty = color ? `color:${color}` : "";
      card.innerHTML = `<small>${label}${tip}</small><strong style="${sty}">${value}</strong>`;
      deviceMetrics.appendChild(card);
    });
  }

  function renderHeader() {
    plotTitle.textContent    = `${houseLabel(state.house)} · ${deviceLabel(state.device)}`;
    plotSubtitle.textContent = `${fmtDate(state.date)} · ${manifest.model_label}`;
  }

  async function render() {
    renderHeader();
    renderMetrics();

    const dataKey = `${state.date}_${state.house}`;
    const needsLoad = dataKey !== loadedDataKey;

    if (needsLoad) {
      chartLoading.style.display = "flex";
      _resetZoomBtn.style.display = "none";
      if (chartInstance) { chartInstance.resetZoom?.(); }
    }

    try {
      const data = await loadHouseData(state.date, state.house);
      loadedDataKey = dataKey;
      chartLoading.style.display = "none";
      buildChart(data, state.device);
    } catch (e) {
      chartLoading.textContent = "Failed to load data";
    }
  }

  function populateFeatured() {
    featuredCards.innerHTML = "";
    manifest.featured_examples.forEach(item => {
      const card = document.createElement("article");
      card.className = "featured-card";
      card.innerHTML = `
        <small>${fmtDate(item.date)} · ${houseLabel(item.house)}</small>
        <h3>${deviceLabel(item.device)}</h3>
        <p>F1 ${fmt(item.on_off_f1)} · Precision ${fmt(item.on_off_precision)} · Recall ${fmt(item.on_off_recall)}</p>
      `;
      const btn = document.createElement("button");
      btn.textContent = "View this example →";
      btn.addEventListener("click", async () => {
        state.date   = item.date;
        state.house  = item.house;
        state.device = item.device;
        syncControls();
        await render();
      });
      card.appendChild(btn);
      featuredCards.appendChild(card);
    });
  }

  dateSelect.addEventListener("change", async () => {
    state.date = dateSelect.value;
    syncControls();
    await render();
  });
  houseSelect.addEventListener("change", async () => {
    state.house = houseSelect.value;
    syncControls();
    await render();
  });
  deviceSelect.addEventListener("change", async () => {
    state.device = deviceSelect.value;
    await render();
  });

  populateFeatured();
  syncControls();
  await render();
}

main();
