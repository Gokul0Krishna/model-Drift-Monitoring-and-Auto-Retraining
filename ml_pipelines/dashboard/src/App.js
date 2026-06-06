import React, { useState, useEffect, useRef } from 'react';

function App() {
  const [metrics, setMetrics] = useState({ total_inferences: 0, drift_share: 0, status: "Idle" });
  const [systemState, setSystemState] = useState("DISCONNECTED");
  const wsRef = useRef(null);

  useEffect(() => {
    let reconnectTimeout;

    const connectWebSocket = () => {
      console.log("🔌 Attempting to connect to MLOps WebSocket...");

      // Explicitly targeting localhost loopback
      const ws = new WebSocket("ws://127.0.0.1:8000/ws/metrics");
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("✅ WebSocket Connected to FastAPI Server!");
        setSystemState("CONNECTED");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setMetrics(data);
        } catch (err) {
          console.error("❌ Error parsing live socket metrics:", err);
        }
      };

      ws.onclose = (e) => {
        console.log(`🔌 WebSocket connection closed (${e.reason}). Reconnecting in 3s...`);
        setSystemState("DISCONNECTED");
        // Auto-retry connection loop
        reconnectTimeout = setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = (err) => {
        console.error("⚠️ WebSocket caught a handshake error:", err);
        ws.close(); // Force clean close to trigger the reconnect loop
      };
    };

    connectWebSocket();

    // Cleanup listeners on component unmount
    return () => {
      if (wsRef.current) wsRef.current.close();
      clearTimeout(reconnectTimeout);
    };
  }, []);

  // UI Status Badges Dynamic Styling
  const getStatusColor = () => {
    if (systemState.status === "HEALTHY") return "#10B981"; // Emerald Green
    if (systemState.status === "RETRAINING_IN_PROGRESS") return "#EF4444"; // Vivid Amber/Red
    return "#6B7280"; // Gray
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>MLOps Live Control Center</h1>
        <div style={{ ...styles.badge, backgroundColor: getStatusColor() }}>
          SYSTEM STATE: {systemState.status}
        </div>
      </header>

      <div style={styles.grid}>
        <div style={styles.card}>
          <h3>Production Metrics</h3>
          <p style={styles.metricBig}>{systemState.total_predictions}</p>
          <p style={styles.label}>Total Logged Inferences</p>
        </div>

        <div style={styles.card}>
          <h3>Statistical Drift Status</h3>
          <p style={styles.metricBig}>{(systemState.drift_score * 100).toFixed(1)}%</p>
          <p style={styles.label}>Dataset Feature Drift Share</p>
        </div>

        <div style={styles.card}>
          <h3>Automation Engine</h3>
          <p style={styles.metricText}>
            {systemState.drift_detected ? "Celery Retraining Running" : "Pipeline Idle / Stable"}
          </p>
          <p style={styles.label}>Active Workers Logs Timestamp: {systemState.last_updated}</p>
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: { fontFamily: 'Segoe UI, sans-serif', padding: '40px', backgroundColor: '#F3F4F6', minHeight: '100vh' },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '2px solid #E5E7EB', paddingBottom: '20px' },
  title: { color: '#1F2937', margin: 0 },
  badge: { padding: '10px 20px', borderRadius: '20px', color: '#FFF', fontWeight: 'bold', fontSize: '14px' },
  grid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px', marginTop: '40px' },
  card: { backgroundColor: '#FFF', padding: '30px', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.05)', textAlign: 'center' },
  metricBig: { fontSize: '48px', fontWeight: 'bold', margin: '10px 0', color: '#111827' },
  metricText: { fontSize: '22px', fontWeight: 'bold', margin: '32px 0', color: '#374151' },
  label: { color: '#6B7280', margin: 0, fontSize: '14px' }
};

export default App;