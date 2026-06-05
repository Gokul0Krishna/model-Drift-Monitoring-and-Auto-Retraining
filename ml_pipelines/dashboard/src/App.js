import React, { useState, useEffect } from 'react';

function App() {
  const [systemState, setSystemState] = useState({
    status: "CONNECTING",
    total_predictions: 0,
    drift_score: 0.0,
    drift_detected: false,
    last_updated: "Never"
  });

  useEffect(() => {
    // Establish a live pipeline over a native WebSocket connection
    const wsUrl = process.env.REACT_APP_WS_URL || "ws://localhost:8000/ws/metrics";
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log("WebSocket connection established successfully.");
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setSystemState({
        status: data.drift_detected ? "RETRAINING_IN_PROGRESS" : "HEALTHY",
        total_predictions: data.total_predictions,
        drift_score: data.drift_score,
        drift_detected: data.drift_detected,
        last_updated: new Date().toLocaleTimeString()
      });
    };

    ws.onerror = (error) => {
      console.error("WebSocket Error: ", error);
    };

    ws.onclose = () => {
      setSystemState((prev) => ({ ...prev, status: "DISCONNECTED" }));
    };

    return () => ws.close();
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