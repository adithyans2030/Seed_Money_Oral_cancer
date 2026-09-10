import { useState, useEffect } from 'react';
import { Activity, Users, FileText, CheckCircle, ShieldAlert } from 'lucide-react';
import { API_BASE_URL } from '../config';
import './AdminDashboard.css';

export default function AdminDashboard() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [password, setPassword] = useState('');
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    try {
      const res = await fetch(`${API_BASE_URL}/metrics`, {
        headers: { "x-admin-key": password }
      });
      
      if (res.status === 403) {
        setError('Invalid admin key');
        setLoading(false);
        return;
      }
      if (!res.ok) throw new Error("Server error");
      
      const data = await res.json();
      setMetrics(data);
      setIsAuthenticated(true);
      // store in session storage for simple persistence
      sessionStorage.setItem('admin_key', password);
    } catch (err) {
      setError('Could not connect to backend API');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const savedKey = sessionStorage.getItem('admin_key');
    if (savedKey) {
      setPassword(savedKey);
      // Auto trigger login if we have a saved key, but safely inside an async IIFE
      const autoLogin = async () => {
        try {
          const res = await fetch(`${API_BASE_URL}/metrics`, {
            headers: { "x-admin-key": savedKey }
          });
          if (res.ok) {
            const data = await res.json();
            setMetrics(data);
            setIsAuthenticated(true);
          } else {
            sessionStorage.removeItem('admin_key');
          }
        } catch (e) {
          // silent fail on auto login
        }
      };
      autoLogin();
    }
  }, []);

  const handleLogout = () => {
    sessionStorage.removeItem('admin_key');
    setIsAuthenticated(false);
    setPassword('');
    setMetrics(null);
  };

  if (!isAuthenticated) {
    return (
      <div className="admin-login-wrap">
        <div className="admin-login-box">
          <h2><ShieldAlert size={24} style={{verticalAlign:'middle', marginRight:'8px'}}/>Admin Access</h2>
          <p>Please enter the admin key to view live validation metrics.</p>
          <form onSubmit={handleLogin}>
            <input 
              type="password" 
              placeholder="Admin Key"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            {error && <div className="admin-error">{error}</div>}
            <button type="submit" disabled={loading}>
              {loading ? 'Authenticating...' : 'View Dashboard'}
            </button>
          </form>
        </div>
      </div>
    );
  }

  if (!metrics) return <div style={{padding:'40px', textAlign:'center'}}>Loading metrics...</div>;

  const renderConfusionMatrix = (title, cm) => {
    if (!cm) return null;
    return (
      <div className="cm-card">
        <h3>{title} Confusion Matrix</h3>
        <table className="cm-table">
          <thead>
            <tr>
              <th></th>
              <th>Predicted Benign</th>
              <th>Predicted Malig</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>True Benign</th>
              <td className="cm-cell tn">TN: {cm.TN}</td>
              <td className="cm-cell fp">FP: {cm.FP}</td>
            </tr>
            <tr>
              <th>True Malig</th>
              <td className="cm-cell fn">FN: {cm.FN}</td>
              <td className="cm-cell tp">TP: {cm.TP}</td>
            </tr>
          </tbody>
        </table>
      </div>
    );
  };

  const renderMetricsBlock = (title, data) => {
    return (
      <div className="metrics-block">
        <h3>{title} Performance</h3>
        <div className="kpi-grid">
          <div className="kpi-card">
            <div className="kpi-label">Sensitivity (Recall)</div>
            <div className="kpi-val">{(data.sensitivity * 100).toFixed(1)}%</div>
            <div className="kpi-ci">95% CI: {(data.sensitivity_ci_lower * 100).toFixed(1)}% - {(data.sensitivity_ci_upper * 100).toFixed(1)}%</div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">Specificity</div>
            <div className="kpi-val">{(data.specificity * 100).toFixed(1)}%</div>
            <div className="kpi-ci">95% CI: {(data.specificity_ci_lower * 100).toFixed(1)}% - {(data.specificity_ci_upper * 100).toFixed(1)}%</div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">Accuracy</div>
            <div className="kpi-val">{(data.accuracy * 100).toFixed(1)}%</div>
          </div>
        </div>
        {renderConfusionMatrix(title, data.confusion_matrix)}
      </div>
    );
  };

  return (
    <div className="admin-dash-wrap">
      <div className="admin-header">
        <div className="admin-header-left">
          <h1>OralGuard Validation Dashboard</h1>
          <span className="live-badge"><div className="pulse"></div> LIVE</span>
        </div>
        <button className="logout-btn" onClick={handleLogout}>Logout</button>
      </div>

      <div className="overview-stats">
        <div className="stat-card">
          <Activity size={24} color="#3b82f6"/>
          <div className="stat-content">
            <div className="stat-title">Total Sessions Logged</div>
            <div className="stat-num">{metrics.total_sessions}</div>
          </div>
        </div>
        <div className="stat-card">
          <CheckCircle size={24} color="#10b981"/>
          <div className="stat-content">
            <div className="stat-title">Clinician Ground Truths</div>
            <div className="stat-num">{metrics.labelled_sessions}</div>
          </div>
        </div>
      </div>

      <div className="metrics-container">
        {renderMetricsBlock("Image Matcher (CBIR)", metrics.image_metrics)}
        {renderMetricsBlock("Questionnaire (XGBoost)", metrics.questionnaire_metrics)}
      </div>
    </div>
  );
}
