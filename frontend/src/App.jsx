// frontend/src/App.jsx
import { useMemo, useState } from "react";
import axios from "axios";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

const API_BASE = "http://127.0.0.1:8000"; // FastAPI

function toIntSafe(v, fallback = 0) {
  const n = Number.parseInt(String(v), 10);
  return Number.isFinite(n) ? n : fallback;
}

function toFloatOrNull(v) {
  if (v === "" || v === null || v === undefined) return null;
  const n = Number.parseFloat(String(v));
  return Number.isFinite(n) ? n : null;
}

function computeQuarter(month) {
  const m = toIntSafe(month, 1);
  return Math.floor((m - 1) / 3) + 1;
}

function computeWeekOfYear(dateStr) {
  // ISO-ish week calc (good enough for UI). Backend also derives week if you use date.
  const d = new Date(dateStr);
  if (Number.isNaN(d.getTime())) return 1;

  const date = new Date(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()));
  const dayNum = date.getUTCDay() || 7;
  date.setUTCDate(date.getUTCDate() + 4 - dayNum);
  const yearStart = new Date(Date.UTC(date.getUTCFullYear(), 0, 1));
  const weekNo = Math.ceil((((date - yearStart) / 86400000) + 1) / 7);
  return weekNo;
}

function cx(...classes) {
  return classes.filter(Boolean).join(" ");
}

export default function App() {
  // ✅ Simple/Advanced toggle:
  // simple => user enters region, vegetable, date (API auto-fills climate baselines)
  // advanced => user can optionally override climate fields
  const [mode, setMode] = useState("simple"); // "simple" | "advanced"

  const [form, setForm] = useState({
    region: "Colombo",
    vegetable_commodity: "Carrot",
    date: "2026-12-30",

    // advanced inputs (optional)
    temperature_c: "",
    rainfall_mm: "",
    humidity_pct: "",
    crop_yield_impact_score: "",
  });

  const derived = useMemo(() => {
    const d = new Date(form.date);
    const year = Number.isNaN(d.getTime()) ? 2024 : d.getFullYear();
    const month = Number.isNaN(d.getTime()) ? 5 : d.getMonth() + 1;
    const weekofyear = computeWeekOfYear(form.date);
    const quarter = computeQuarter(month);
    return { year, month, weekofyear, quarter };
  }, [form.date]);

  // ✅ Payload rules:
  // Always send region, commodity, date.
  // Only send climate values in Advanced mode (and only if user typed them).
  const payload = useMemo(() => {
    const base = {
      region: String(form.region).trim(),
      vegetable_commodity: String(form.vegetable_commodity).trim(),
      date: form.date,
      // (Optional) If your API still accepts month/year, it can ignore these when date exists.
      // Keeping them doesn't hurt, but simplest is just date.
      // month: derived.month, weekofyear: derived.weekofyear, quarter: derived.quarter, year: derived.year,
    };

    if (mode === "advanced") {
      const t = toFloatOrNull(form.temperature_c);
      const r = toFloatOrNull(form.rainfall_mm);
      const h = toFloatOrNull(form.humidity_pct);
      const c = toFloatOrNull(form.crop_yield_impact_score);

      // Only include if not null, so backend can baseline-fill missing ones
      if (t !== null) base.temperature_c = t;
      if (r !== null) base.rainfall_mm = r;
      if (h !== null) base.humidity_pct = h;
      if (c !== null) base.crop_yield_impact_score = c;
    }

    return base;
  }, [form, mode]);

  const [loading, setLoading] = useState(false);
  const [predicted, setPredicted] = useState(null);
  const [explainData, setExplainData] = useState([]);
  const [error, setError] = useState("");

  // ✅ new: show if baseline was used and show final features returned by API
  const [usedBaseline, setUsedBaseline] = useState(false);
  const [finalFeatures, setFinalFeatures] = useState(null);

  function onChange(name, value) {
    setForm((p) => ({ ...p, [name]: value }));
  }

  async function callPredict() {
    setError("");
    setLoading(true);
    setExplainData([]);
    setUsedBaseline(false);
    setFinalFeatures(null);

    try {
      const res = await axios.post(`${API_BASE}/predict`, payload);
      setPredicted(res.data?.predicted_price_lkr_per_kg ?? null);
      setUsedBaseline(Boolean(res.data?.used_climate_baseline));
      setFinalFeatures(res.data?.final_features ?? null);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  async function callExplain() {
    setError("");
    setLoading(true);
    setExplainData([]);
    setUsedBaseline(false);
    setFinalFeatures(null);

    try {
      const res = await axios.post(`${API_BASE}/explain`, payload);
      setPredicted(res.data?.predicted_price_lkr_per_kg ?? null);
      setUsedBaseline(Boolean(res.data?.used_climate_baseline));
      setFinalFeatures(res.data?.final_features ?? null);

      const contributions = res.data?.feature_contributions || {};
      const arr = Object.entries(contributions).map(([feature, shap]) => ({
        feature,
        shap: Number(shap),
        abs: Math.abs(Number(shap)),
      }));
      arr.sort((a, b) => b.abs - a.abs);
      setExplainData(arr.slice(0, 10));
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || "Explain failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="max-w-6xl mx-auto px-4 py-10">
        <header className="mb-8">
          <h1 className="text-3xl font-semibold tracking-tight">
            Sri Lanka Vegetable Price Predictor
          </h1>
          <p className="text-slate-300 mt-2">
            Predict price (LKR/kg) and explain results using SHAP (CatBoost).
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Form */}
          <div className="rounded-2xl bg-slate-900/60 border border-slate-800 p-5">
            <div className="flex items-center justify-between gap-4 mb-4">
              <h2 className="text-lg font-medium">Inputs</h2>

              {/* ✅ Toggle */}
              <div className="flex items-center gap-2">
                <span className={cx("text-xs", mode === "simple" ? "text-slate-100" : "text-slate-400")}>
                  Simple
                </span>

                <button
                  type="button"
                  onClick={() => setMode((m) => (m === "simple" ? "advanced" : "simple"))}
                  className={cx(
                    "relative inline-flex h-6 w-11 items-center rounded-full border transition",
                    mode === "advanced"
                      ? "bg-sky-600 border-sky-500"
                      : "bg-slate-800 border-slate-700"
                  )}
                  aria-label="Toggle Advanced Mode"
                >
                  <span
                    className={cx(
                      "inline-block h-5 w-5 transform rounded-full bg-white transition",
                      mode === "advanced" ? "translate-x-5" : "translate-x-1"
                    )}
                  />
                </button>

                <span className={cx("text-xs", mode === "advanced" ? "text-slate-100" : "text-slate-400")}>
                  Advanced
                </span>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Field label="Region">
                <input
                  className="w-full rounded-xl bg-slate-950 border border-slate-800 px-3 py-2 text-slate-100 outline-none focus:ring-2 focus:ring-sky-500"
                  value={form.region}
                  onChange={(e) => onChange("region", e.target.value)}
                  placeholder="e.g., Colombo"
                />
              </Field>

              <Field label="Vegetable Commodity">
                <input
                  className="w-full rounded-xl bg-slate-950 border border-slate-800 px-3 py-2 text-slate-100 outline-none focus:ring-2 focus:ring-sky-500"
                  value={form.vegetable_commodity}
                  onChange={(e) => onChange("vegetable_commodity", e.target.value)}
                  placeholder="e.g., Carrot"
                />
              </Field>

              <Field label="Date">
                <input
                  type="date"
                  className="w-full rounded-xl bg-slate-950 border border-slate-800 px-3 py-2 text-slate-100 outline-none focus:ring-2 focus:ring-sky-500"
                  value={form.date}
                  onChange={(e) => onChange("date", e.target.value)}
                />
                <div className="text-xs text-slate-400 mt-2">
                  Derived from date: year={derived.year}, month={derived.month}, week={derived.weekofyear}, quarter={derived.quarter}
                </div>
              </Field>

              <div className="hidden md:block" />

              {/* ✅ Advanced fields */}
              {mode === "advanced" && (
                <>
                  <Field label="Temperature (°C) — optional override">
                    <input
                      type="number"
                      step="0.1"
                      className="w-full rounded-xl bg-slate-950 border border-slate-800 px-3 py-2 text-slate-100 outline-none focus:ring-2 focus:ring-sky-500"
                      value={form.temperature_c}
                      onChange={(e) => onChange("temperature_c", e.target.value)}
                      placeholder="leave blank to auto-fill"
                    />
                  </Field>

                  <Field label="Rainfall (mm) — optional override">
                    <input
                      type="number"
                      step="0.1"
                      className="w-full rounded-xl bg-slate-950 border border-slate-800 px-3 py-2 text-slate-100 outline-none focus:ring-2 focus:ring-sky-500"
                      value={form.rainfall_mm}
                      onChange={(e) => onChange("rainfall_mm", e.target.value)}
                      placeholder="leave blank to auto-fill"
                    />
                  </Field>

                  <Field label="Humidity (%) — optional override">
                    <input
                      type="number"
                      step="0.1"
                      className="w-full rounded-xl bg-slate-950 border border-slate-800 px-3 py-2 text-slate-100 outline-none focus:ring-2 focus:ring-sky-500"
                      value={form.humidity_pct}
                      onChange={(e) => onChange("humidity_pct", e.target.value)}
                      placeholder="leave blank to auto-fill"
                    />
                  </Field>

                  <Field label="Crop Yield Impact Score — optional override">
                    <input
                      type="number"
                      step="0.01"
                      className="w-full rounded-xl bg-slate-950 border border-slate-800 px-3 py-2 text-slate-100 outline-none focus:ring-2 focus:ring-sky-500"
                      value={form.crop_yield_impact_score}
                      onChange={(e) => onChange("crop_yield_impact_score", e.target.value)}
                      placeholder="leave blank to auto-fill"
                    />
                  </Field>

                  <div className="md:col-span-2 text-xs text-slate-400">
                    Tip: Leave any climate field blank to use historical averages for the selected region + month.
                  </div>
                </>
              )}
            </div>

            {error && (
              <div className="mt-4 rounded-xl border border-red-900/60 bg-red-950/30 text-red-200 px-4 py-3">
                {error}
              </div>
            )}

            <div className="mt-5 flex flex-wrap gap-3">
              <button
                onClick={callPredict}
                disabled={loading}
                className="rounded-xl bg-sky-600 hover:bg-sky-500 disabled:opacity-60 px-4 py-2 font-medium"
              >
                {loading ? "Working..." : "Predict"}
              </button>

              <button
                onClick={callExplain}
                disabled={loading}
                className="rounded-xl bg-slate-800 hover:bg-slate-700 disabled:opacity-60 px-4 py-2 font-medium border border-slate-700"
              >
                {loading ? "Working..." : "Predict + Explain"}
              </button>
            </div>

            {/* ✅ Show what is being sent */}
            <div className="mt-5 rounded-2xl bg-slate-950 border border-slate-800 p-4">
              <div className="text-xs text-slate-400 mb-2">Request payload (what frontend sends)</div>
              <pre className="text-xs text-slate-200 overflow-auto">
                {JSON.stringify(payload, null, 2)}
              </pre>
            </div>
          </div>

          {/* Results */}
          <div className="rounded-2xl bg-slate-900/60 border border-slate-800 p-5">
            <h2 className="text-lg font-medium mb-4">Results</h2>

            <div className="rounded-2xl bg-slate-950 border border-slate-800 p-5">
              <div className="text-slate-400 text-sm">Predicted Price</div>
              <div className="text-4xl font-semibold mt-2">
                {predicted === null ? "—" : `${predicted} LKR/kg`}
              </div>

              {/* ✅ Auto-fill message */}
              {predicted !== null && (
                <div className="mt-3">
                  {usedBaseline ? (
                    <div className="inline-flex items-center gap-2 rounded-xl border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-amber-200 text-sm">
                      <span className="h-2 w-2 rounded-full bg-amber-300" />
                      Using historical climate baseline (region + month averages)
                    </div>
                  ) : (
                    <div className="inline-flex items-center gap-2 rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-emerald-200 text-sm">
                      <span className="h-2 w-2 rounded-full bg-emerald-300" />
                      Using user-provided climate inputs (Advanced mode)
                    </div>
                  )}
                </div>
              )}

              <div className="text-xs text-slate-500 mt-3">
                Model: CatBoost Regression • Explainability: SHAP
              </div>
            </div>

            {/* ✅ Show final features used by the model */}
            <div className="mt-5 rounded-2xl bg-slate-950 border border-slate-800 p-4">
              <div className="text-sm font-medium mb-2">Final features used for prediction</div>
              {!finalFeatures ? (
                <div className="text-sm text-slate-400">
                  Run Predict / Predict + Explain to see the final feature values used by the model.
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                  {Object.entries(finalFeatures).map(([k, v]) => (
                    <div
                      key={k}
                      className="flex items-center justify-between gap-3 rounded-xl border border-slate-800 bg-slate-900/40 px-3 py-2"
                    >
                      <span className="text-slate-300">{k}</span>
                      <span className="text-slate-100 font-medium">
                        {typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(3)) : String(v)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* SHAP Chart */}
            <div className="mt-6">
              <div className="flex items-center justify-between">
                <h3 className="font-medium">Top SHAP Contributions</h3>
                <span className="text-xs text-slate-400">
                  (+ increases prediction, − decreases)
                </span>
              </div>

              {explainData.length === 0 ? (
                <div className="mt-3 text-slate-400 text-sm">
                  Click <span className="text-slate-200">Predict + Explain</span> to see feature contributions.
                </div>
              ) : (
                <div className="mt-3 h-80 rounded-2xl bg-slate-950 border border-slate-800 p-3">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={explainData} layout="vertical" margin={{ left: 20, right: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.25)" />
                      <XAxis
                        type="number"
                        tick={{ fill: "#cbd5e1", fontSize: 12 }}
                        axisLine={{ stroke: "rgba(148,163,184,0.35)" }}
                        tickLine={{ stroke: "rgba(148,163,184,0.35)" }}
                      />
                      <YAxis
                        type="category"
                        dataKey="feature"
                        width={170}
                        tick={{ fill: "#cbd5e1", fontSize: 12 }}
                        axisLine={{ stroke: "rgba(148,163,184,0.35)" }}
                        tickLine={{ stroke: "rgba(148,163,184,0.35)" }}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "rgba(15, 23, 42, 0.95)",
                          border: "1px solid rgba(148, 163, 184, 0.25)",
                          borderRadius: 12,
                          color: "#e2e8f0",
                        }}
                        labelStyle={{ color: "#e2e8f0" }}
                        itemStyle={{ color: "#e2e8f0" }}
                        cursor={{ fill: "rgba(148, 163, 184, 0.08)" }}
                      />
                      <Bar
                        dataKey="shap"
                        stroke="rgba(148,163,184,0.35)"
                        strokeWidth={1}
                        radius={[8, 8, 8, 8]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>

            
          </div>
        </div>

        <footer className="mt-10 text-xs text-slate-500">
          Demo stack: React + Tailwind + FastAPI + CatBoost + SHAP
        </footer>
      </div>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <label className="block">
      <div className="text-sm text-slate-300 mb-1">{label}</div>
      {children}
    </label>
  );
}