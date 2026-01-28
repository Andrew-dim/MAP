/**
 * Multi-Sensor SAR Analysis Platform
 * Frontend JavaScript - Phase 1 (NISAR Integration)
 */

function setStatus(text) {
  const el = document.getElementById("status");
  if (el) el.textContent = text;
}

function pad2(n) {
  return String(n).padStart(2, "0");
}

function toDatetimeLocalValue(d) {
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}T${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
}

function showInfoBanner(message, type = 'info') {
  const banner = document.getElementById('info-banner');
  const text = document.getElementById('info-text');
  if (banner && text) {
    text.textContent = message;
    banner.className = `info-banner ${type}`;
    banner.style.display = 'flex';
  }
}

function hideInfoBanner() {
  const banner = document.getElementById('info-banner');
  if (banner) banner.style.display = 'none';
}

const fromInput = document.getElementById("from-dt");
const toInput = document.getElementById("to-dt");
const cloudMaxInput = document.getElementById("cloud-max");
const cloudCard = document.getElementById("cloud-card");
const satelliteSelect = document.getElementById("satellite-select");
const nisarOptions = document.getElementById("nisar-options");
const nisarAnalysis = document.getElementById("nisar-analysis");
const militaryPanel = document.getElementById("military-panel");
const panelToggle = document.getElementById("panel-toggle");
const panelContent = document.getElementById("panel-content");
const layerSelect = document.getElementById("layer-select");
const nisarFrequency = document.getElementById("nisar-frequency");
const nisarLevel = document.getElementById("nisar-level");
const cfarThreshold = document.getElementById("cfar-threshold");
const cfarValue = document.getElementById("cfar-value");
const minTargetSize = document.getElementById("min-target-size");
const sizeValue = document.getElementById("size-value");

let currentSatellite = "Sentinel-2";
let currentMapType = "Satellite";
let currentResolutionMode = "original";

const now = new Date();
if (fromInput) fromInput.value = toDatetimeLocalValue(now);
if (toInput) toInput.value = toDatetimeLocalValue(new Date(now.getTime() + 60 * 60 * 1000));

let panelExpanded = false;

if (panelToggle && panelContent) {
  panelToggle.addEventListener("click", () => {
    panelExpanded = !panelExpanded;
    if (panelExpanded) {
      panelContent.classList.remove("collapsed");
      panelToggle.textContent = "▼";
    } else {
      panelContent.classList.add("collapsed");
      panelToggle.textContent = "▲";
    }
  });
}

const osmLayer = new ol.layer.Tile({
  source: new ol.source.OSM(),
  visible: false
});

const satelliteLayer = new ol.layer.Tile({
  source: new ol.source.XYZ({
    url: "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    attributions: "© Google"
  }),
  visible: true
});

const map = new ol.Map({
  target: "map",
  layers: [osmLayer, satelliteLayer],
  view: new ol.View({
    center: ol.proj.fromLonLat([0, 0]),
    zoom: 2
  }),
  controls: ol.control.defaults(),
  interactions: ol.interaction.defaults({ shiftDragZoom: false })
});

if (layerSelect) {
  layerSelect.addEventListener("change", function () {
    osmLayer.setVisible(this.value === "osm");
    satelliteLayer.setVisible(this.value === "satellite");
    currentMapType = this.value === "osm" ? "OSM" : "Satellite";
  });
}

document.querySelectorAll('.resolution-option').forEach(option => {
  option.addEventListener('click', function() {
    document.querySelectorAll('.resolution-option').forEach(opt => opt.classList.remove('selected'));
    this.classList.add('selected');
    this.querySelector('input[type="radio"]').checked = true;
    currentResolutionMode = this.dataset.mode;
    const warning = document.getElementById('resolution-warning');
    if (warning) warning.style.display = currentResolutionMode === 'enhanced' ? 'block' : 'none';
  });
});

if (satelliteSelect) {
  satelliteSelect.addEventListener("change", function () {
    currentSatellite = this.value;
    updateSatelliteUI(currentSatellite);
  });
}

function updateSatelliteUI(satellite) {
  if (nisarOptions) nisarOptions.style.display = 'none';
  if (militaryPanel) militaryPanel.style.display = 'none';
  if (cloudCard) cloudCard.style.display = 'none';

  switch (satellite) {
    case 'NISAR':
      if (nisarOptions) nisarOptions.style.display = 'block';
      showInfoBanner('🛰️ NISAR: Quad-pol L+S band SAR.', 'nisar');
      if (nisarAnalysis && (nisarAnalysis.value === 'military' || nisarAnalysis.value === 'target-detection')) {
        if (militaryPanel) militaryPanel.style.display = 'block';
      }
      break;
    case 'Sentinel-1':
      showInfoBanner('📡 Sentinel-1: C-band dual-pol SAR.', 'info');
      break;
    case 'Sentinel-2':
    default:
      if (cloudCard) cloudCard.style.display = 'block';
      hideInfoBanner();
      break;
  }
}

if (nisarAnalysis) {
  nisarAnalysis.addEventListener("change", function () {
    const showMilitary = this.value === 'military' || this.value === 'target-detection';
    if (militaryPanel) militaryPanel.style.display = showMilitary ? 'block' : 'none';
  });
}

if (cfarThreshold && cfarValue) {
  cfarThreshold.addEventListener("input", function () {
    cfarValue.textContent = this.value + ' σ';
  });
}

if (minTargetSize && sizeValue) {
  minTargetSize.addEventListener("input", function () {
    sizeValue.textContent = this.value + ' px';
  });
}

document.getElementById('info-close')?.addEventListener('click', hideInfoBanner);

let lastPointerCoord = null;

map.on("pointermove", (evt) => {
  lastPointerCoord = evt.coordinate;
  const [lon, lat] = ol.proj.toLonLat(evt.coordinate);
  const el = document.getElementById("mouse-coords");
  if (el) el.textContent = `lon: ${lon.toFixed(4)}, lat: ${lat.toFixed(4)}`;
});

const polySource = new ol.source.Vector();
const polyLayer = new ol.layer.Vector({
  source: polySource,
  style: new ol.style.Style({
    stroke: new ol.style.Stroke({ color: "rgba(200,0,0,0.95)", width: 2 }),
    fill: new ol.style.Fill({ color: "rgba(200,0,0,0.18)" })
  })
});
map.addLayer(polyLayer);

const labelSource = new ol.source.Vector();
const labelLayer = new ol.layer.Vector({
  source: labelSource,
  style: (f) => new ol.style.Style({
    text: new ol.style.Text({
      text: f.get("label") || "",
      font: "bold 14px Segoe UI",
      fill: new ol.style.Fill({ color: "#0a3d62" }),
      stroke: new ol.style.Stroke({ color: "white", width: 3 }),
      backgroundFill: new ol.style.Fill({ color: "rgba(255,255,255,0.92)" }),
      backgroundStroke: new ol.style.Stroke({ color: "#ccc", width: 1 }),
      padding: [4, 6, 4, 6]
    })
  })
});
map.addLayer(labelLayer);

let polygonsStack = [];
let polygonFeature = null;
let selectedCoords = [];
let currentVertexLabels = [];

function addVertexLabelAt(coordProjected) {
  const [lon, lat] = ol.proj.toLonLat(coordProjected);
  const labelFeature = new ol.Feature({
    geometry: new ol.geom.Point(coordProjected),
    label: `(${lon.toFixed(4)}, ${lat.toFixed(4)})`
  });
  labelSource.addFeature(labelFeature);
  currentVertexLabels.push(labelFeature);
}

function updatePreviewPolygon() {
  const pts = selectedCoords.map((c) => ol.proj.fromLonLat(c));
  if (pts.length >= 2) {
    polygonFeature.setGeometry(new ol.geom.Polygon([[...pts, pts[0]]]));
  } else {
    polygonFeature.setGeometry(null);
  }
}

let isSelecting = false;

document.addEventListener("keydown", (e) => {
  if (e.key === "Shift" && !isSelecting) {
    isSelecting = true;
    selectedCoords = [];
    currentVertexLabels = [];

    if (polygonFeature) polySource.removeFeature(polygonFeature);
    polygonFeature = new ol.Feature(new ol.geom.Polygon([]));
    polySource.addFeature(polygonFeature);

    setStatus("Selecting: 0 vertices\n(none)");
    return;
  }

  const key = e.key.toLowerCase();
  const isC = (key === "c" || e.key === "ψ");
  if (isSelecting && isC && lastPointerCoord) {
    const [lon, lat] = ol.proj.toLonLat(lastPointerCoord);
    selectedCoords.push([lon, lat]);
    addVertexLabelAt(lastPointerCoord);
    updatePreviewPolygon();

    const lines = selectedCoords.map((c, i) => `[${i+1}] ${c[0].toFixed(4)}, ${c[1].toFixed(4)}`).join("\n");
    setStatus(`Selecting: ${selectedCoords.length} vertices\n${lines || "(none)"}`);
  }
});

document.addEventListener("keyup", (e) => {
  if (e.key === "Shift" && isSelecting) {
    isSelecting = false;

    if (selectedCoords.length >= 3) {
      polygonsStack.push({
        feature: polygonFeature,
        labels: [...currentVertexLabels],
        coords: [...selectedCoords]
      });

      const lines = selectedCoords.map((c, i) => `[${i+1}] ${c[0].toFixed(4)}, ${c[1].toFixed(4)}`).join("\n");
      setStatus(`Polygon finalized: ${selectedCoords.length} vertices\n${lines}`);
    } else {
      if (polygonFeature) polySource.removeFeature(polygonFeature);
      currentVertexLabels.forEach((f) => labelSource.removeFeature(f));
      setStatus("Selection canceled (need at least 3 vertices).");
    }

    polygonFeature = null;
    selectedCoords = [];
    currentVertexLabels = [];
  }
});

function removeLast() {
  if (polygonsStack.length > 0) {
    const last = polygonsStack.pop();
    if (last.feature) polySource.removeFeature(last.feature);
    last.labels.forEach((f) => labelSource.removeFeature(f));
    setStatus(`Removed last polygon. Remaining: ${polygonsStack.length}`);
  } else {
    setStatus("Nothing to remove.");
  }
}

function clearAll() {
  polySource.clear();
  labelSource.clear();
  polygonsStack = [];
  setStatus("Cleared all polygons.");
}

function goToProfile() {
  window.location.href = "/profile";
}

function getAllSelections() {
  return polygonsStack.map((p, idx) => ({
    number: idx + 1,
    coords: p.coords.map(([lon, lat]) => ({ lon, lat }))
  }));
}

function getNisarConfig() {
  if (currentSatellite !== 'NISAR') return null;
  return {
    frequency: nisarFrequency ? nisarFrequency.value : 'L-band',
    level: nisarLevel ? nisarLevel.value : 'L2-GCOV',
    analysis: nisarAnalysis ? nisarAnalysis.value : 'basic',
    military_features: {}
  };
}

async function submitSelections() {
  if (!polygonsStack.length) {
    alert("Please draw at least one polygon before submitting.");
    return;
  }
  const fromVal = fromInput ? fromInput.value : "";
  const toVal = toInput ? toInput.value : "";
  if (!fromVal || !toVal) {
    alert("Please select a date range.");
    return;
  }
  const cloudMax = (currentSatellite === 'Sentinel-2' && cloudMaxInput) ? Number(cloudMaxInput.value || 30) : 100;
  const nisarConfig = getNisarConfig();
  const submissionData = {
    satellite: currentSatellite,
    selections: getAllSelections(),
    date_from: fromVal,
    date_to: toVal,
    cloud_max: cloudMax,
    map_type: currentMapType,
    resolution_mode: currentResolutionMode
  };
  if (currentSatellite === 'NISAR' && nisarConfig) {
    submissionData.frequency = nisarConfig.frequency;
    submissionData.level = nisarConfig.level;
    submissionData.analysis = nisarConfig.analysis;
    submissionData.nisar_config = nisarConfig;
  }
  setStatus("Submitting...");
  try {
    const res = await fetch("/api/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify(submissionData)
    });
    const resp = await res.json().catch(() => ({}));
    if (!res.ok) {
      alert(resp.error || "Failed to submit.");
      return;
    }
    setStatus(`✓ Submitted! Job ID: ${resp.job_id || "N/A"}`);
    showInfoBanner(`✓ Job submitted! ID: ${resp.job_id}`, 'success');
  } catch (err) {
    alert("Server unreachable.");
    setStatus("Network error.");
  }
}

document.getElementById("clear-all")?.addEventListener("click", clearAll);
document.getElementById("remove-last")?.addEventListener("click", removeLast);
document.getElementById("submit")?.addEventListener("click", submitSelections);
document.getElementById("profile")?.addEventListener("click", goToProfile);

console.log("Platform initialized.");
setStatus("Hold Shift to start. Press C/Ψ to add vertices. Release Shift to finish.");
