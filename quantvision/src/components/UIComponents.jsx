import { useState } from "react";
import { C } from "../utils/data";

/* ─── Custom Tooltip ─────────────────────────────────────────── */
export const ChartTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    return (
        <div className="tooltip-card">
            <div style={{ color: C.amber, marginBottom: 4, fontSize: 10, letterSpacing: 1 }}>{label}</div>
            {payload.map((p, i) => (
                <div key={i} style={{ color: p.color || C.text, display: "flex", justifyContent: "space-between", gap: 16 }}>
                    <span style={{ color: C.textMid }}>{p.name}</span>
                    <span>{typeof p.value === "number" ? p.value.toFixed(2) : p.value}</span>
                </div>
            ))}
        </div>
    );
};

/* ─── Tooltip hint ───────────────────────────────────────────── */
export const Hint = ({ text }) => {
    const [show, setShow] = useState(false);
    return (
        <span style={{ position: "relative", display: "inline-flex", alignItems: "center", marginLeft: 4 }}>
            <span
                onMouseEnter={() => setShow(true)}
                onMouseLeave={() => setShow(false)}
                style={{ color: C.textDim, cursor: "help", fontSize: 11, border: `1px solid ${C.border}`, borderRadius: "50%", width: 14, height: 14, display: "inline-flex", alignItems: "center", justifyContent: "center", lineHeight: 1 }}
            >?</span>
            {show && (
                <span style={{
                    position: "absolute", bottom: "120%", left: "50%", transform: "translateX(-50%)",
                    // bg3, not bg1 — Section panels are bg1, and the tooltip has to lift off them.
                    background: C.bg3, border: `1px solid ${C.border}`, borderRadius: 6, padding: "6px 10px",
                    width: 200, fontSize: 11, color: C.textMid, fontFamily: "'DM Mono',monospace", zIndex: 99,
                    pointerEvents: "none", lineHeight: 1.5, textTransform: "none", letterSpacing: 0,
                }}>{text}</span>
            )}
        </span>
    );
};

/* ─── Stat card ─────────────────────────────────────────────── */
/** `color` tints the value and the left accent — call sites use it semantically
 *  (green = return, red = risk). `positive` still overrides the sub-line colour. */
export const StatCard = ({ label, value, sub, positive, icon, color, hint }) => (
    <div style={{
        background: C.bg2, border: `1px solid ${C.border}`,
        borderLeft: `3px solid ${color || C.border}`, borderRadius: 10,
        padding: "14px 18px", flex: 1, minWidth: 130,
    }}>
        <div style={{ color: C.textDim, fontSize: 10, letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 6, fontFamily: "'Syne',sans-serif", display: "flex", alignItems: "center" }}>
            {icon && <span style={{ marginRight: 4 }}>{icon}</span>}{label}
            {hint && <Hint text={hint} />}
        </div>
        <div style={{ color: color || C.text, fontSize: 22, fontWeight: 700, fontFamily: "'DM Mono',monospace", lineHeight: 1 }}>{value}</div>
        {sub && <div style={{ color: positive === undefined ? C.textMid : positive ? C.green : C.red, fontSize: 11, marginTop: 4, fontFamily: "'DM Mono',monospace" }}>{sub}</div>}
    </div>
);

/* ─── Badge ──────────────────────────────────────────────────── */
export const Badge = ({ children, color = C.amber }) => (
    <span style={{
        background: color + "22", color, border: `1px solid ${color}44`,
        borderRadius: 4, padding: "2px 8px", fontSize: 10,
        fontFamily: "'DM Mono',monospace", letterSpacing: .5,
    }}>{children}</span>
);

/* ─── Tab button ─────────────────────────────────────────────── */
export const Tab = ({ active, onClick, children }) => (
    <button onClick={onClick} style={{
        background: active ? C.amberDim : "transparent",
        border: "none", borderBottom: `2px solid ${active ? C.amber : "transparent"}`,
        color: active ? C.amber : C.textDim, padding: "10px 20px",
        cursor: "pointer", fontFamily: "'Syne',sans-serif",
        fontWeight: active ? 700 : 400, fontSize: 13,
        letterSpacing: .5, transition: "all .2s", whiteSpace: "nowrap",
    }}>{children}</button>
);

/* ─── Section wrapper ────────────────────────────────────────── */
/** Titled panel. `hint` adds a `?` tooltip beside the heading; `right` is a
 *  slot for actions (buttons, toggles) on the heading row. */
export const Section = ({ title, hint, right, children, style = {} }) => (
    <div className="fade-up" style={{
        animation: "fadeUp .35s ease both",
        background: C.bg1, border: `1px solid ${C.border}`, borderRadius: 10,
        padding: 16, ...style,
    }}>
        {title && (
            <div style={{
                display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12,
                marginBottom: 14, paddingBottom: 10, borderBottom: `1px solid ${C.border}`,
            }}>
                <span style={{ display: "flex", alignItems: "center" }}>
                    <span style={{
                        color: C.textMid, fontSize: 10, letterSpacing: 1.5, textTransform: "uppercase",
                        fontFamily: "'Syne',sans-serif", fontWeight: 700,
                    }}>{title}</span>
                    {hint && <Hint text={hint} />}
                </span>
                {right}
            </div>
        )}
        {children}
    </div>
);

