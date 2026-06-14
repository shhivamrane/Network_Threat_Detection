import { useEffect, useState, useMemo } from "react";
import {
  LineChart, Line, BarChart, Bar, Cell, XAxis, YAxis, Tooltip,
  CartesianGrid, ResponsiveContainer,
} from "recharts";

const API = "http://127.0.0.1:8000";
const WS_URL = "ws://127.0.0.1:8000/stream";

const COLORS = {
  Normal: "#22c55e", DoS: "#ef4444", Probe: "#f59e0b",
  R2L: "#a855f7", U2R: "#ec4899",
};

function normalize(d) {
  return {
    attack_class: d.attack_class,
    confidence: Number(d.confidence) || 0,
    is_attack: !!d.is_attack,
    ts: d.ts || Date.now() / 1000,
  };
}

export default function App() {
  const [detections, setDetections] = useState([]);
  const [info, setInfo] = useState(null);
  const [live, setLive] = useState(false);

  // Initial load from REST.
  useEffect(() => {
    fetch(`${API}/model/info`).then(r => r.json()).then(setInfo).catch(() => {});
    fetch(`${API}/detections?limit=200`)
      .then(r => r.json())
      .then(rows => setDetections(rows.map(normalize)))
      .catch(() => {});
  }, []);

  // Live feed over WebSocket.
  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    ws.onopen = () => setLive(true);
    ws.onclose = () => setLive(false);
    ws.onerror = () => setLive(false);
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (!msg.attack_class) return; // ignore keepalive pings
      setDetections(prev =>
        [normalize({ ...msg, ts: Date.now() / 1000 }), ...prev].slice(0, 300)
      );
    };
    return () => ws.close();
  }, []);

  const byType = useMemo(() => {
    const c = {};
    detections.forEach(d => { c[d.attack_class] = (c[d.attack_class] || 0) + 1; });
    return Object.entries(c).map(([name, count]) => ({ name, count }));
  }, [detections]);

  const overTime = useMemo(() => {
    const b = {};
    detections.forEach(d => {
      const k = Math.floor((d.ts || 0) / 5) * 5; // 5-second buckets
      if (!b[k]) b[k] = { attacks: 0, total: 0 };
      b[k].total += 1;
      if (d.is_attack) b[k].attacks += 1;
    });
    return Object.entries(b)
      .sort((a, bb) => a[0] - bb[0])
      .slice(-30)
      .map(([k, v]) => ({
        time: new Date(k * 1000).toLocaleTimeString(),
        attacks: v.attacks, total: v.total,
      }));
  }, [detections]);

  const attackCount = detections.filter(d => d.is_attack).length;

  return (
    <div style={S.page}>
      <header style={S.header}>
        <h1 style={S.h1}>Network Threat Detection</h1>
        <div style={S.meta}>
          <span style={{ ...S.dot, background: live ? "#22c55e" : "#ef4444" }} />
          {live ? "Live" : "Offline"}
          {info && <span> · {info.model} · {info.n_features} features</span>}
        </div>
      </header>

      <div style={S.cards}>
        <Card label="Total flows" value={detections.length} />
        <Card label="Attacks" value={attackCount} accent="#ef4444" />
        <Card label="Benign" value={detections.length - attackCount} accent="#22c55e" />
      </div>

      <div style={S.grid}>
        <div style={S.panel}>
          <h3 style={S.h3}>Attacks over time</h3>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={overTime}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="time" tick={{ fontSize: 11, fill: "#9ca3af" }} />
              <YAxis tick={{ fontSize: 11, fill: "#9ca3af" }} allowDecimals={false} />
              <Tooltip contentStyle={{ background: "#111827", border: "1px solid #1f2937" }} />
              <Line type="monotone" dataKey="attacks" stroke="#ef4444" dot={false} />
              <Line type="monotone" dataKey="total" stroke="#3b82f6" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={S.panel}>
          <h3 style={S.h3}>By type</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={byType}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#9ca3af" }} />
              <YAxis tick={{ fontSize: 11, fill: "#9ca3af" }} allowDecimals={false} />
              <Tooltip contentStyle={{ background: "#111827", border: "1px solid #1f2937" }} />
              <Bar dataKey="count">
                {byType.map((e) => (
                  <Cell key={e.name} fill={COLORS[e.name] || "#3b82f6"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={S.panel}>
        <h3 style={S.h3}>Live feed</h3>
        <table style={S.table}>
          <thead>
            <tr>
              <th style={S.th}>Class</th>
              <th style={S.th}>Confidence</th>
              <th style={S.th}>Status</th>
            </tr>
          </thead>
          <tbody>
            {detections.slice(0, 50).map((d, i) => (
              <tr key={i} style={{ background: d.is_attack ? "#2a1416" : "transparent" }}>
                <td style={{ ...S.td, color: COLORS[d.attack_class] || "#e5e7eb" }}>
                  {d.attack_class}
                </td>
                <td style={S.td}>{(d.confidence * 100).toFixed(1)}%</td>
                <td style={S.td}>{d.is_attack ? "⚠ Attack" : "OK"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Card({ label, value, accent = "#e5e7eb" }) {
  return (
    <div style={S.card}>
      <div style={S.cardLabel}>{label}</div>
      <div style={{ ...S.cardValue, color: accent }}>{value}</div>
    </div>
  );
}

const S = {
  page: { maxWidth: 1100, margin: "0 auto", padding: 24, fontFamily: "system-ui, sans-serif", color: "#e5e7eb" },
  header: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 },
  h1: { fontSize: 22, margin: 0 },
  meta: { fontSize: 13, color: "#9ca3af", display: "flex", alignItems: "center", gap: 6 },
  dot: { width: 9, height: 9, borderRadius: "50%", display: "inline-block" },
  cards: { display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginBottom: 16 },
  card: { background: "#111827", border: "1px solid #1f2937", borderRadius: 10, padding: 16 },
  cardLabel: { fontSize: 12, color: "#9ca3af" },
  cardValue: { fontSize: 28, fontWeight: 700, marginTop: 4 },
  grid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 },
  panel: { background: "#111827", border: "1px solid #1f2937", borderRadius: 10, padding: 16, marginBottom: 16 },
  h3: { fontSize: 14, margin: "0 0 12px", color: "#d1d5db" },
  table: { width: "100%", borderCollapse: "collapse", fontSize: 13 },
  th: { textAlign: "left", padding: "8px 10px", borderBottom: "1px solid #1f2937", color: "#9ca3af", fontWeight: 600 },
  td: { padding: "8px 10px", borderBottom: "1px solid #1f2937" },
};