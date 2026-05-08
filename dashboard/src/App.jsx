import { useState, useEffect, useRef } from "react";
import * as THREE from "three";

// ── constants ──────────────────────────────────────────────────────────────────

const STAGE_COLORS = {
  operational: "#00e87a", active_construction: "#ff9f1c", undisturbed: "#64748b",
};
const SEVERITY_COLORS = { high: "#ff5577", medium: "#ffaa00", low: "#00e87a" };
const STAGE_RANK = {
  undisturbed: 0, active_construction: 1, operational: 2,
};
const DETECTION_FIELDS = [
  "site_class", "construction_stage", "roof_bright_membrane", "bare_soil_present", "reasoning",
];

function latLonToVec3(lat, lon, r) {
  const phi = (90 - lat) * Math.PI / 180, theta = (lon + 180) * Math.PI / 180;
  return new THREE.Vector3(-(r * Math.sin(phi) * Math.cos(theta)), r * Math.cos(phi), r * Math.sin(phi) * Math.sin(theta));
}

const CONTINENTS = {
  na: [[60,-140],[65,-168],[72,-168],[71,-156],[60,-148],[60,-140],[58,-137],[55,-133],[52,-128],[48,-124],[40,-124],[35,-120],[33,-117],[30,-115],[25,-110],[20,-105],[15,-92],[15,-87],[18,-88],[21,-87],[21,-90],[19,-96],[26,-97],[29,-95],[30,-88],[29,-83],[27,-80],[25,-80],[28,-82],[30,-81],[33,-79],[35,-76],[38,-75],[40,-74],[41,-72],[43,-70],[45,-67],[47,-67],[45,-61],[47,-53],[52,-56],[55,-60],[60,-64],[62,-75],[58,-78],[52,-80],[55,-85],[50,-88],[48,-89],[46,-84],[43,-82],[45,-82],[48,-88],[50,-95],[52,-95],[55,-100],[58,-110],[60,-120],[60,-140]],
  sa: [[12,-72],[10,-62],[7,-60],[5,-52],[0,-50],[-5,-35],[-10,-37],[-15,-39],[-20,-40],[-23,-42],[-28,-48],[-33,-52],[-35,-57],[-40,-62],[-45,-65],[-50,-70],[-55,-68],[-55,-70],[-50,-75],[-45,-75],[-40,-73],[-35,-72],[-30,-71],[-25,-70],[-18,-70],[-15,-75],[-10,-77],[-5,-80],[0,-80],[5,-77],[8,-77],[10,-72],[12,-72]],
  eu: [[36,-10],[38,-8],[43,-9],[44,-2],[47,-2],[48,5],[44,3],[43,5],[46,8],[47,15],[45,14],[42,18],[40,20],[38,24],[35,25],[37,28],[40,29],[42,28],[42,32],[44,34],[46,30],[48,22],[51,14],[54,10],[56,8],[54,12],[58,12],[60,5],[62,5],[64,14],[68,16],[70,20],[72,28],[70,32],[68,45],[65,40],[60,30],[56,28],[54,20],[52,16],[50,4],[48,-2],[47,-2],[44,-2],[43,-9],[38,-8],[36,-10]],
  af: [[37,10],[32,10],[30,10],[25,33],[22,36],[15,43],[12,44],[12,50],[5,42],[0,42],[-5,40],[-10,40],[-15,35],[-20,35],[-25,35],[-30,32],[-35,20],[-34,18],[-30,16],[-20,12],[-15,12],[-10,14],[-5,12],[0,10],[5,2],[5,-5],[8,-15],[15,-17],[20,-17],[25,-15],[30,-10],[32,-5],[35,-2],[37,10]],
  as: [[42,28],[44,34],[46,30],[48,22],[51,14],[54,20],[56,28],[60,30],[65,40],[68,45],[70,50],[68,60],[72,72],[72,100],[72,130],[70,140],[68,170],[65,170],[60,163],[56,140],[50,142],[46,143],[44,145],[40,130],[35,130],[32,130],[30,120],[25,120],[22,114],[20,107],[10,105],[7,103],[0,104],[-5,105],[-8,115],[-8,120],[-5,130],[0,128],[-8,140],[-8,132],[-5,120],[-8,114],[-5,105],[0,104],[7,103],[10,100],[5,98],[10,80],[8,77],[15,75],[25,68],[25,60],[30,50],[25,45],[22,36],[25,33],[30,35],[35,35],[38,42],[40,50],[35,51],[30,48],[28,58],[25,57],[22,59],[16,52],[12,44],[12,50],[5,42],[0,42],[-5,40]],
  au: [[-12,136],[-12,142],[-15,145],[-20,149],[-25,153],[-28,153],[-33,152],[-35,150],[-38,145],[-38,140],[-35,137],[-33,134],[-32,132],[-35,130],[-35,117],[-30,115],[-25,113],[-22,114],[-20,118],[-15,125],[-14,130],[-12,136]],
};

// ── helpers ────────────────────────────────────────────────────────────────────

function imageUrl(dbPath) {
  if (!dbPath) return null;
  const m = dbPath.match(/ground_assets\/([^/]+)\/(rgb|swir|index|mapbox)/);
  return m ? `/api/images/${m[1]}/${m[2]}` : null;
}

function siteName(tileId, siteId) {
  if (!tileId) return `site_${siteId}`;
  return tileId.split("/")[0];
}

function fmtTs(v) {
  if (!v) return "n/a";
  try { return new Date(v.replace("Z", "+00:00")).toISOString().replace("T", " ").slice(0, 19) + " UTC"; }
  catch { return v; }
}

// ── StageTimeline (SVG, same pattern as reference NDBIChart) ────────────────

function StageTimeline({ history }) {
  if (!history || history.length < 2) return <div style={{ fontSize: 10, color: "#607898", padding: 8 }}>Insufficient stage data</div>;
  const w = 300, h = 100, p = { t: 10, r: 10, b: 20, l: 45 };
  const iw = w - p.l - p.r, ih = h - p.t - p.b;
  const sorted = [...history].sort((a, b) => new Date(a.observed_at) - new Date(b.observed_at));
  const times = sorted.map(d => new Date(d.observed_at).getTime());
  // once operational, always operational
  let peak = 0;
  const ranks = sorted.map(d => {
    const r = STAGE_RANK[d.construction_stage] ?? 0;
    peak = Math.max(peak, r);
    return peak;
  });
  const tMin = Math.min(...times), tMax = Math.max(...times), rMax = 2;
  const tRange = tMax - tMin || 1;
  const pts = sorted.map((_, i) =>
    `${p.l + (((times[i] - tMin) / tRange) * iw)},${p.t + ih - ((ranks[i] / rMax) * ih)}`
  ).join(" ");
  return (
    <svg width={w} height={h} style={{ display: "block" }}>
      <defs><linearGradient id="sg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#ff5577" stopOpacity="0.25" /><stop offset="100%" stopColor="#ff5577" stopOpacity="0.02" /></linearGradient></defs>
      <line x1={p.l} x2={w - p.r} y1={p.t + ih} y2={p.t + ih} stroke="#2a3648" strokeDasharray="3,3" />
      <polygon points={`${p.l},${p.t + ih} ${pts} ${p.l + iw},${p.t + ih}`} fill="url(#sg)" />
      <polyline points={pts} fill="none" stroke="#ff5577" strokeWidth="1.5" />
      {sorted.map((d, i) => (
        <circle key={i} cx={p.l + ((times[i] - tMin) / tRange) * iw} cy={p.t + ih - (ranks[i] / rMax) * ih} r="3" fill="#ffaa00" />
      ))}
      <text x={2} y={p.t + 7} fill="#7088a0" fontSize="6" fontFamily="monospace">operational</text>
      <text x={2} y={p.t + ih} fill="#7088a0" fontSize="6" fontFamily="monospace">undisturbed</text>
      <text x={p.l} y={h - 2} fill="#7088a0" fontSize="7" fontFamily="monospace">{new Date(tMin).toISOString().slice(0, 10)}</text>
      <text x={w - p.r} y={h - 2} fill="#7088a0" fontSize="7" textAnchor="end" fontFamily="monospace">{new Date(tMax).toISOString().slice(0, 10)}</text>
    </svg>
  );
}

// ── SitePanel ──────────────────────────────────────────────────────────────────

function SitePanel({ site, detail, onClose, enrichment, enrichLoading, onEnrich }) {
  const stage = site.current_construction_stage || "unknown";
  const color = STAGE_COLORS[stage] || "#64748b";
  const detection = site.detection_json || {};
  const labelStyle = { fontSize: 8, color: "#8098b8", letterSpacing: 2, marginBottom: 5 };
  const cardStyle = { background: "#10182a", borderRadius: 5, padding: 10, border: "1px solid #1e2e44" };

  return (
    <div style={{ position: "absolute", top: 0, right: 0, width: 365, height: "100%", background: "#0e1625f0", borderLeft: `1px solid ${color}30`, zIndex: 20, padding: "20px 16px", overflowY: "auto", backdropFilter: "blur(20px)", fontFamily: "'JetBrains Mono', monospace" }}>
      <button onClick={onClose} style={{ position: "absolute", top: 12, right: 14, background: "none", border: "none", color: "#8098b8", fontSize: 18, cursor: "pointer", lineHeight: 1 }}>&#x2715;</button>

      <div style={{ fontSize: 8, color: "#8098b8", letterSpacing: 3, marginBottom: 3 }}>SITE ANALYSIS</div>
      <div style={{ fontSize: 16, color: "#e8eef8", fontWeight: 700, lineHeight: 1.3 }}>{siteName(site.tile_id, site.site_id)}</div>
      <div style={{ fontSize: 11, color: "#90a8c4", marginBottom: 4 }}>{site.current_site_class || "unknown"}</div>
      <span style={{ display: "inline-block", padding: "3px 8px", borderRadius: 3, background: color + "18", color, fontSize: 8, letterSpacing: 2, fontWeight: 600, border: `1px solid ${color}35`, marginBottom: 16 }}>{stage.toUpperCase()}</span>

      <div style={labelStyle}>SENTINEL-2 COMPOSITES</div>
      <div style={{ display: "flex", gap: 5, marginBottom: 14 }}>
        {["rgb", "swir", "index"].map(type => {
          const url = imageUrl(site[`${type}_path`]);
          return (
            <div key={type} style={{ flex: 1 }}>
              {url
                ? <img src={url} alt={type} style={{ width: "100%", aspectRatio: "1", borderRadius: 4, objectFit: "cover", border: "1px solid #1e2e44", display: "block" }} />
                : <div style={{ width: "100%", aspectRatio: "1", borderRadius: 4, background: "#142028", border: "1px solid #1e2e44" }} />
              }
              <div style={{ fontSize: 7, color: "#607898", textAlign: "center", marginTop: 3, letterSpacing: 1 }}>{type.toUpperCase()}</div>
            </div>
          );
        })}
      </div>

      {detail?.stage_history?.length >= 2 && (
        <>
          <div style={labelStyle}>STAGE TIMELINE</div>
          <div style={{ ...cardStyle, padding: 6, marginBottom: 14 }}>
            <StageTimeline history={detail.stage_history} />
          </div>
        </>
      )}

      <div style={labelStyle}>LFM2.5-VL OUTPUT</div>
      <div style={{ ...cardStyle, fontSize: 10, lineHeight: 1.9, marginBottom: 14 }}>
        {DETECTION_FIELDS.map(field => (
          <div key={field} style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid #182438", paddingBottom: 2, marginBottom: 2 }}>
            <span style={{ color: "#7090b0" }}>{field.replace(/_/g, " ")}</span>
            <span style={{ color: "#d0d8e8" }}>{String(detection[field] ?? "n/a")}</span>
          </div>
        ))}
        <div style={{ display: "flex", justifyContent: "space-between", paddingBottom: 2 }}>
          <span style={{ color: "#7090b0" }}>coords</span>
          <span style={{ color: "#d0d8e8" }}>{site.lat?.toFixed(3)}, {site.lon?.toFixed(3)}</span>
        </div>
      </div>

      {detail?.alerts?.length > 0 && (
        <>
          <div style={labelStyle}>ALERT HISTORY</div>
          <div style={{ ...cardStyle, marginBottom: 14 }}>
            {detail.alerts.slice(0, 5).map((a, i) => (
              <div key={i} style={{ borderBottom: i < detail.alerts.length - 1 ? "1px solid #182438" : "none", paddingBottom: 6, marginBottom: 6 }}>
                <div style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 3 }}>
                  <span style={{ fontSize: 8, color: SEVERITY_COLORS[a.severity] || "#7fa7d8", letterSpacing: 1, fontWeight: 600 }}>{a.severity?.toUpperCase()}</span>
                  <span style={{ fontSize: 8, color: "#607898" }}>{a.alert_type}</span>
                </div>
                <AlertSummary summary={a.summary} />
                <div style={{ fontSize: 7, color: "#607898", marginTop: 2 }}>{fmtTs(a.created_at)}</div>
              </div>
            ))}
          </div>
        </>
      )}


      <div style={labelStyle}>GROUND ENRICHMENT</div>
      <div style={{ fontSize: 8, color: "#607898", marginBottom: 8 }}>Gemini 2.5 Flash + Google Search &rarr; site ID, demographics, water, grid, health, community</div>
      {enrichment ? (
        <div style={{ ...cardStyle }}>
          <MiniMarkdown text={enrichment} />
        </div>
      ) : (
        <button
          onClick={onEnrich}
          disabled={enrichLoading}
          style={{
            width: "100%", padding: 10, border: `1px solid ${enrichLoading ? "#1e3050" : "#a855f7"}44`,
            borderRadius: 6, background: enrichLoading ? "#10182a" : "#1a1230",
            color: enrichLoading ? "#607898" : "#a855f7", fontSize: 10, fontWeight: 700,
            letterSpacing: 2, textTransform: "uppercase", cursor: enrichLoading ? "default" : "pointer",
            fontFamily: "inherit",
          }}
        >
          {enrichLoading ? "RESEARCHING ..." : "RESEARCH THIS SITE"}
        </button>
      )}
    </div>
  );
}

// ── MiniMarkdown ──────────────────────────────────────────────────────────────

function inlineBold(text) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((p, i) =>
    p.startsWith("**") && p.endsWith("**")
      ? <strong key={i} style={{ color: "#e0e8f4", fontWeight: 700 }}>{p.slice(2, -2)}</strong>
      : p
  );
}

function MiniMarkdown({ text }) {
  if (!text) return null;
  const lines = text.split("\n");
  const nodes = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // blank line
    if (!line.trim()) { i++; continue; }

    // heading
    if (/^#{1,3} /.test(line)) {
      nodes.push(
        <div key={i} style={{ fontSize: 9, color: "#8098b8", letterSpacing: 2, textTransform: "uppercase", marginTop: 10, marginBottom: 4, fontWeight: 700 }}>
          {line.replace(/^#+\s*/, "")}
        </div>
      );
      i++; continue;
    }

    // table — collect all consecutive pipe lines
    if (line.trim().startsWith("|")) {
      const tableLines = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        tableLines.push(lines[i]);
        i++;
      }
      const rows = tableLines
        .filter(l => !/^\|[-| :]+\|$/.test(l.trim()))
        .map(l => l.replace(/^\||\|$/g, "").split("|").map(c => c.trim()));
      nodes.push(
        <table key={`tbl-${i}`} style={{ width: "100%", borderCollapse: "collapse", marginTop: 6, marginBottom: 6 }}>
          <tbody>
            {rows.map((cols, ri) => (
              <tr key={ri} style={{ borderBottom: "1px solid #182438" }}>
                {cols.map((cell, ci) => (
                  <td key={ci} style={{ fontSize: 9, padding: "3px 4px", color: ci === 0 ? "#7090b0" : "#d0d8e8", verticalAlign: "top" }}>
                    {inlineBold(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      );
      continue;
    }

    // bullet
    if (/^[-*] /.test(line.trim())) {
      nodes.push(
        <div key={i} style={{ fontSize: 10, color: "#90a8c4", lineHeight: 1.6, paddingLeft: 10, marginBottom: 2, display: "flex", gap: 6 }}>
          <span style={{ color: "#4a6080", flexShrink: 0 }}>·</span>
          <span>{inlineBold(line.replace(/^[-*] /, ""))}</span>
        </div>
      );
      i++; continue;
    }

    // paragraph line
    nodes.push(
      <div key={i} style={{ fontSize: 10, color: "#90a8c4", lineHeight: 1.7, marginBottom: 4 }}>
        {inlineBold(line)}
      </div>
    );
    i++;
  }

  return <>{nodes}</>;
}

// ── AlertSummary ───────────────────────────────────────────────────────────────

function AlertSummary({ summary }) {
  let parsed = null;
  try {
    const jsonStart = summary?.indexOf("{");
    if (jsonStart !== -1) parsed = JSON.parse(summary.slice(jsonStart));
  } catch {}

  if (parsed && typeof parsed === "object") {
    const changes = (parsed.detections || []).flatMap(d => {
      if (d.type !== "changed") return [];
      return (d.fields || []).map(f => ({ field: f, prev: d.previous?.[f], curr: d.current?.[f] }));
    });
    const ctxFlags = parsed.tile_context_fields || [];

    if (changes.length === 0 && ctxFlags.length === 0) {
      return <div style={{ fontSize: 10, color: "#607898" }}>No changes recorded.</div>;
    }

    return (
      <div>
        {changes.slice(0, 4).map(({ field, prev, curr }) => (
          <div key={field} style={{ fontSize: 9, lineHeight: 1.5, display: "flex", gap: 4, flexWrap: "wrap" }}>
            <span style={{ color: "#607898" }}>{field.replace(/_/g, " ")}:</span>
            <span style={{ color: "#ff5577", textDecoration: "line-through" }}>{String(prev)}</span>
            <span style={{ color: "#8098b8" }}>→</span>
            <span style={{ color: "#00e87a" }}>{String(curr)}</span>
          </div>
        ))}
        {ctxFlags.length > 0 && (
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4 }}>
            {ctxFlags.map(f => (
              <span key={f} style={{ fontSize: 7, padding: "2px 5px", borderRadius: 3, background: "#1a2a3a", color: "#7090b0", border: "1px solid #1e3050", letterSpacing: 1 }}>
                {f.replace(/_/g, " ").toUpperCase()}
              </span>
            ))}
          </div>
        )}
      </div>
    );
  }

  return <div style={{ fontSize: 10, color: "#d0dae8", lineHeight: 1.4 }}>{summary?.slice(0, 120)}</div>;
}

// ── App ────────────────────────────────────────────────────────────────────────

export default function App() {
  const mountRef = useRef(null);
  const sceneRef = useRef({});
  const [selectedSite, setSelectedSite] = useState(null);
  const [siteDetail, setSiteDetail] = useState(null);
  const [data, setData] = useState({ sites: [], alerts: [], satellite: null, stats: {} });
  const [enrichments, setEnrichments] = useState({});
  const [enrichLoading, setEnrichLoading] = useState(false);
  const [newestAlertId, setNewestAlertId] = useState(null);
  const newestAlertIdRef = useRef(null);
  const dataRef = useRef(data);
  const mouseRef = useRef({ isDown: false, lastX: 0, lastY: 0 });
  const rotRef = useRef({ x: 0.35, y: -1.85 });
  const autoRotateRef = useRef(true);
  const [autoRotate, setAutoRotate] = useState(true);

  // ── polling ──
  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch("/api/state");
        const json = await res.json();
        setData(json);
        dataRef.current = json;
        const newest = json.alerts?.[0]?.alert_id ?? null;
        if (newest !== null && newest !== newestAlertIdRef.current) {
          newestAlertIdRef.current = newest;
          setNewestAlertId(newest);
        }
      } catch (e) { console.warn("poll failed", e); }
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, []);

  // ── site detail fetch ──
  useEffect(() => {
    if (!selectedSite) { setSiteDetail(null); return; }
    fetch(`/api/sites/${selectedSite.site_id}`)
      .then(r => r.json()).then(setSiteDetail).catch(() => setSiteDetail(null));
  }, [selectedSite]);

  // ── three.js scene (runs once) ──
  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;
    const W = el.clientWidth, H = el.clientHeight;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 1000);
    camera.position.set(0, 0, 3.2);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x080e1c, 1);
    el.appendChild(renderer.domElement);

    const globe = new THREE.Group();
    scene.add(globe);

    // ocean
    globe.add(new THREE.Mesh(new THREE.SphereGeometry(0.995, 80, 80), new THREE.MeshBasicMaterial({ color: 0x0e2040 })));

    // grid
    const gridMat = new THREE.LineBasicMaterial({ color: 0x1c3458, transparent: true, opacity: 0.55 });
    for (let lat = -80; lat <= 80; lat += 20) {
      const pts = []; for (let lon = -180; lon <= 180; lon += 3) pts.push(latLonToVec3(lat, lon, 1.001));
      globe.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), gridMat));
    }
    for (let lon = -180; lon < 180; lon += 30) {
      const pts = []; for (let lat = -90; lat <= 90; lat += 3) pts.push(latLonToVec3(lat, lon, 1.001));
      globe.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), gridMat));
    }

    // continents
    const landMat = new THREE.LineBasicMaterial({ color: 0x3a7898, transparent: true, opacity: 0.75 });
    Object.values(CONTINENTS).forEach(coords => {
      globe.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(coords.map(([la, lo]) => latLonToVec3(la, lo, 1.003))), landMat));
    });

    // atmosphere
    [1.04, 1.08, 1.13].forEach((r, i) => {
      globe.add(new THREE.Mesh(new THREE.SphereGeometry(r, 64, 64), new THREE.MeshBasicMaterial({ color: 0x2060a0, transparent: true, opacity: 0.06 - i * 0.015, side: THREE.BackSide })));
    });

    // stars
    const sPos = new Float32Array(1800);
    for (let i = 0; i < 1800; i++) sPos[i] = (Math.random() - 0.5) * 70;
    const sGeo = new THREE.BufferGeometry();
    sGeo.setAttribute("position", new THREE.BufferAttribute(sPos, 3));
    scene.add(new THREE.Points(sGeo, new THREE.PointsMaterial({ color: 0x5a7090, size: 0.06, transparent: true, opacity: 0.5 })));

    const markers = new THREE.Group();
    const satGroup = new THREE.Group();
    globe.add(markers);
    globe.add(satGroup);
    sceneRef.current = { scene, camera, renderer, globe, markers, satGroup };

    const raycaster = new THREE.Raycaster();
    const mv = new THREE.Vector2();
    let fid;
    const clk = new THREE.Clock();

    const animate = () => {
      fid = requestAnimationFrame(animate);
      const t = clk.getElapsedTime();
      if (!mouseRef.current.isDown && autoRotateRef.current) rotRef.current.y -= 0.0006;
      globe.rotation.x = rotRef.current.x;
      globe.rotation.y = rotRef.current.y;
      markers.children.forEach((c, i) => { if (c.userData.pulse) c.scale.setScalar(1 + 0.3 * Math.sin(t * 2.5 + i * 0.6)); });
      satGroup.children.forEach((c, i) => { if (c.userData.pulse) c.scale.setScalar(1 + 0.4 * Math.sin(t * 3 + i)); });
      renderer.render(scene, camera);
    };
    animate();

    const onDown = e => { mouseRef.current = { isDown: true, lastX: e.clientX, lastY: e.clientY }; };
    const onMove = e => {
      if (!mouseRef.current.isDown) return;
      rotRef.current.y += (e.clientX - mouseRef.current.lastX) * 0.005;
      rotRef.current.x += (e.clientY - mouseRef.current.lastY) * 0.005;
      rotRef.current.x = Math.max(-1.2, Math.min(1.2, rotRef.current.x));
      mouseRef.current.lastX = e.clientX;
      mouseRef.current.lastY = e.clientY;
    };
    const onUp = () => { mouseRef.current.isDown = false; };
    const onClick = e => {
      const r = el.getBoundingClientRect();
      mv.x = ((e.clientX - r.left) / r.width) * 2 - 1;
      mv.y = -((e.clientY - r.top) / r.height) * 2 + 1;
      raycaster.setFromCamera(mv, camera);
      const hits = raycaster.intersectObjects(markers.children.filter(c => c.userData.sid));
      if (hits.length) {
        const site = dataRef.current.sites.find(s => s.site_id === hits[0].object.userData.sid);
        if (site) setSelectedSite(site);
      }
    };
    const onWheel = e => { camera.position.z = Math.max(1.6, Math.min(6, camera.position.z + e.deltaY * 0.002)); };
    const onResize = () => { const w2 = el.clientWidth, h2 = el.clientHeight; camera.aspect = w2 / h2; camera.updateProjectionMatrix(); renderer.setSize(w2, h2); };

    el.addEventListener("mousedown", onDown);
    el.addEventListener("mousemove", onMove);
    el.addEventListener("mouseup", onUp);
    el.addEventListener("click", onClick);
    el.addEventListener("wheel", onWheel);
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(fid);
      el.removeEventListener("mousedown", onDown);
      el.removeEventListener("mousemove", onMove);
      el.removeEventListener("mouseup", onUp);
      el.removeEventListener("click", onClick);
      el.removeEventListener("wheel", onWheel);
      window.removeEventListener("resize", onResize);
      if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  // ── update site markers when data changes ──
  useEffect(() => {
    const { markers } = sceneRef.current;
    if (!markers) return;
    while (markers.children.length) markers.remove(markers.children[0]);

    const severityBySite = {};
    data.alerts.forEach(a => { if (!severityBySite[a.site_id]) severityBySite[a.site_id] = a.severity; });

    data.sites.forEach(site => {
      const severity = severityBySite[site.site_id] || "low";
      const col = new THREE.Color(SEVERITY_COLORS[severity] || "#00e87a");
      const pos = latLonToVec3(site.lat, site.lon, 1.012);
      const sz = severity === "high" ? 1.4 : severity === "medium" ? 1.1 : 0.9;
      const dot = new THREE.Mesh(new THREE.SphereGeometry(0.009 * sz, 10, 10), new THREE.MeshBasicMaterial({ color: col }));
      dot.position.copy(pos);
      dot.userData = { sid: site.site_id };
      markers.add(dot);
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.014 * sz, 0.02 * sz, 20), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.25, side: THREE.DoubleSide }));
      ring.position.copy(pos);
      ring.lookAt(0, 0, 0);
      ring.userData = { pulse: true };
      markers.add(ring);
      const outer = latLonToVec3(site.lat, site.lon, 1.012 + 0.02 + 0.016 * sz);
      markers.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([pos, outer]), new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.45 })));
    });
  }, [data.sites, data.alerts]);

  // ── update satellite marker ──
  useEffect(() => {
    const { satGroup } = sceneRef.current;
    if (!satGroup) return;
    while (satGroup.children.length) satGroup.remove(satGroup.children[0]);
    if (!data.satellite) return;
    const pos = latLonToVec3(data.satellite.lat, data.satellite.lon, 1.03);
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.014, 12, 12), new THREE.MeshBasicMaterial({ color: 0x4cc9f0 }));
    dot.position.copy(pos);
    dot.userData = { pulse: true };
    satGroup.add(dot);
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.02, 0.028, 24), new THREE.MeshBasicMaterial({ color: 0x4cc9f0, transparent: true, opacity: 0.3, side: THREE.DoubleSide }));
    ring.position.copy(pos);
    ring.lookAt(0, 0, 0);
    ring.userData = { pulse: true };
    satGroup.add(ring);
  }, [data.satellite]);

  // ── styles ──
  const panelBg = "#0c1628e8";
  const panelBorder = "1px solid #1e3050";
  const labelColor = "#8098b8";
  const brightText = "#e0e8f4";
  const font = "'JetBrains Mono', 'Fira Code', monospace";
  const stats = data.stats || {};

  return (
    <div style={{ width: "100%", height: "100vh", background: "#080e1c", position: "relative", fontFamily: font, overflow: "hidden", color: brightText }}>
      <div ref={mountRef} style={{ width: "100%", height: "100%", cursor: "grab" }} />

      {/* title */}
      <div style={{ position: "absolute", top: 20, left: 22, zIndex: 10 }}>
        <div style={{ fontSize: 8, letterSpacing: 5, color: labelColor }}>SATELLITE INTELLIGENCE</div>
        <div style={{ fontSize: 20, fontWeight: 800, letterSpacing: 1, background: "linear-gradient(90deg, #f0f4ff, #55aadd)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>DataCenterWatch</div>
        <div style={{ fontSize: 9, color: labelColor, marginTop: 1 }}>LFM2.5-VL-450M &times; Sentinel-2 &times; DPhi SimSat</div>
      </div>

      {/* stats */}
      <div style={{ position: "absolute", top: 20, right: selectedSite ? 385 : 22, zIndex: 10, display: "flex", gap: 10, transition: "right 0.3s" }}>
        <div style={{ background: panelBg, border: panelBorder, borderRadius: 5, padding: "6px 12px", backdropFilter: "blur(12px)" }}>
          <div style={{ fontSize: 7, color: labelColor, letterSpacing: 2 }}>ALERTS</div>
          <div style={{ fontSize: 16, fontWeight: 700, color: "#ff5577" }}>{stats.total_alerts ?? 0}</div>
        </div>
      </div>

      {/* satellite info */}
      {data.satellite && (
        <div style={{ position: "absolute", top: 80, left: 22, zIndex: 10, background: panelBg, border: panelBorder, borderRadius: 5, padding: "6px 12px", backdropFilter: "blur(12px)" }}>
          <div style={{ fontSize: 7, color: labelColor, letterSpacing: 2, marginBottom: 2 }}>SIMSAT LIVE</div>
          <div style={{ fontSize: 10, color: "#4cc9f0" }}>{data.satellite.lat?.toFixed(3)}, {data.satellite.lon?.toFixed(3)}</div>
          <div style={{ fontSize: 8, color: "#607898", marginTop: 1 }}>{fmtTs(data.satellite.timestamp)}</div>
        </div>
      )}

      {/* alert inbox */}
      <div style={{ position: "absolute", bottom: 14, left: 22, width: 340, maxHeight: "45vh", overflowY: "auto", zIndex: 10, background: panelBg, border: panelBorder, borderRadius: 6, padding: "10px 14px", backdropFilter: "blur(12px)" }}>
        <div style={{ fontSize: 7, color: labelColor, letterSpacing: 3, marginBottom: 8 }}>ALERT INBOX</div>
        {data.alerts.length === 0 && <div style={{ fontSize: 10, color: "#607898" }}>No alerts yet.</div>}
        {data.alerts.slice(0, 15).map(alert => {
          const isNewest = alert.alert_id === newestAlertId;
          const accentColor = isNewest ? "#4cc9f0" : (SEVERITY_COLORS[alert.severity] || "#7fa7d8");
          return (
            <div key={alert.alert_id}
              onClick={() => { const s = data.sites.find(s => s.site_id === alert.site_id); if (s) setSelectedSite(s); }}
              style={{ borderLeft: `3px solid ${accentColor}`, background: isNewest ? "#0a1e30" : "#10182a", borderRadius: 6, padding: "8px 10px", marginBottom: 6, cursor: "pointer", transition: "background 0.4s, border-color 0.4s" }}>
              <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 3 }}>
                <span style={{ fontSize: 8, color: isNewest ? "#4cc9f0" : SEVERITY_COLORS[alert.severity], letterSpacing: 1, fontWeight: 600 }}>{alert.severity?.toUpperCase()}</span>
                <span style={{ fontSize: 8, color: "#607898" }}>{alert.alert_type}</span>
                {isNewest && <span style={{ fontSize: 7, color: "#4cc9f0", letterSpacing: 2, marginLeft: "auto" }}>NEW</span>}
              </div>
              <AlertSummary summary={alert.summary} />
              <div style={{ fontSize: 7, color: "#607898", marginTop: 3 }}>{fmtTs(alert.created_at)}</div>
            </div>
          );
        })}
      </div>

      {/* legend */}
      <div style={{ position: "absolute", bottom: 14, right: selectedSite ? 385 : 22, zIndex: 10, background: panelBg, border: panelBorder, borderRadius: 5, padding: "8px 12px", backdropFilter: "blur(12px)", transition: "right 0.3s" }}>
        <div style={{ fontSize: 7, color: labelColor, letterSpacing: 2, marginBottom: 4 }}>SEVERITY</div>
        {Object.entries(SEVERITY_COLORS).map(([s, c]) => (
          <div key={s} style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 2 }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: c, boxShadow: `0 0 6px ${c}66` }} />
            <span style={{ fontSize: 8, color: "#a0b4cc" }}>{s.toUpperCase()}</span>
          </div>
        ))}
        <div style={{ display: "flex", alignItems: "center", gap: 7, marginTop: 4 }}>
          <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#4cc9f0", boxShadow: "0 0 6px #4cc9f066" }} />
          <span style={{ fontSize: 8, color: "#a0b4cc" }}>SATELLITE</span>
        </div>
      </div>

      {/* help */}
      <div style={{ position: "absolute", top: 20, right: selectedSite ? 385 : 22, zIndex: 5, fontSize: 8, color: labelColor, textAlign: "right", transition: "right 0.3s", marginTop: 70 }}>
        <div>DRAG to rotate &middot; SCROLL to zoom</div>
        <div>CLICK marker for analysis</div>
        <button
          onClick={() => { autoRotateRef.current = !autoRotateRef.current; setAutoRotate(autoRotateRef.current); }}
          style={{
            marginTop: 6, padding: "4px 10px", border: `1px solid ${autoRotate ? "#1e3050" : "#55aadd"}`,
            borderRadius: 4, background: autoRotate ? "#0c1628e8" : "#0e2040",
            color: autoRotate ? labelColor : "#55aadd", fontSize: 8, letterSpacing: 2,
            textTransform: "uppercase", cursor: "pointer", fontFamily: font,
          }}
        >
          {autoRotate ? "STOP ROTATION" : "START ROTATION"}
        </button>
      </div>

      {selectedSite && (
        <SitePanel
          site={selectedSite}
          detail={siteDetail}
          onClose={() => setSelectedSite(null)}
          enrichment={enrichments[selectedSite.site_id]}
          enrichLoading={enrichLoading}
          onEnrich={async () => {
            if (enrichLoading || enrichments[selectedSite.site_id]) return;
            setEnrichLoading(true);
            try {
              const controller = new AbortController();
              const timer = setTimeout(() => controller.abort(), 90000);
              const res = await fetch(`/api/enrich/${selectedSite.site_id}`, { method: "POST", signal: controller.signal });
              clearTimeout(timer);
              const text = await res.text();
              let result;
              try { const json = JSON.parse(text); result = json.enrichment || json.error || "No data."; }
              catch { result = text || "Unknown error"; }
              setEnrichments(prev => ({ ...prev, [selectedSite.site_id]: result }));
            } catch (e) {
              const msg = e.name === "AbortError" ? "Request timed out (90s). Try again." : `Error: ${e.message}`;
              setEnrichments(prev => ({ ...prev, [selectedSite.site_id]: msg }));
            }
            setEnrichLoading(false);
          }}
        />
      )}
    </div>
  );
}
