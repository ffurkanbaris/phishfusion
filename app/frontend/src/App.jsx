import React, { useState } from 'react';
import ResultCard from './ResultCard';
import { Camera, CameraResultType, CameraSource } from '@capacitor/camera';

function App() {
  const [url, setUrl] = useState('');
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
  // For mobile app, localhost won't work. Allow user to input server IP.
  const [serverUrl, setServerUrl] = useState('https://ffeo-ff.hf.space');

  const takePicture = async () => {
    try {
      const image = await Camera.getPhoto({
        quality: 90,
        allowEditing: false,
        resultType: CameraResultType.Uri,
        source: CameraSource.Camera
      });

      // Convert webPath to a File object
      const response = await fetch(image.webPath);
      const blob = await response.blob();
      const fileObj = new File([blob], 'camera-qr.jpeg', { type: 'image/jpeg' });
      setFile(fileObj);
      setError(null);
    } catch (err) {
      console.error("Camera error:", err);
      // User cancelled or permissions denied
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!url && !file) {
      setError("Please provide either a URL or a QR code image.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    if (url) formData.append('url', url);
    if (file) formData.append('file', file);

    try {
      const targetEndpoint = `${serverUrl.replace(/\/$/, '')}/predict`;
      
      const response = await fetch(targetEndpoint, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || "Prediction failed. Check server connection.");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message + " (Is the server running and accessible?)");
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1>PhishFusion</h1>
        <p>Mobile Security & QR Scanner</p>
      </div>

      <div className="main-content">
        <form className="glass-card compact-form" onSubmit={handleSubmit}>
          <div className="form-row">
            <div className="form-group flex-1">
              <label htmlFor="serverUrl">Server API</label>
              <input
                type="text"
                id="serverUrl"
                className="input-field h-full"
                value={serverUrl}
                onChange={(e) => setServerUrl(e.target.value)}
                style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}
              />
            </div>

            <div className="form-group flex-1">
              <label htmlFor="url">Analyze Link</label>
              <input
                type="text"
                id="url"
                className="input-field h-full"
                placeholder="https://example.com"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
              />
            </div>
          </div>

          <div className="or-divider">
            OR
          </div>

          <div className="form-row">
            <div className="form-group flex-1" style={{ marginBottom: 0 }}>
              <label>Scan QR Code</label>
              <button 
                type="button" 
                className="btn secondary h-full" 
                onClick={takePicture}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"></path><circle cx="12" cy="13" r="3"></circle></svg>
                Camera
              </button>
            </div>
            
            <div className="form-group flex-1" style={{ marginBottom: 0 }}>
              <label>Upload Image</label>
              <div className="file-upload h-full">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
                <div style={{ fontSize: '0.9rem' }}>{file ? file.name : "Choose from gallery"}</div>
                <input 
                  type="file" 
                  accept="image/*" 
                  onChange={handleFileChange} 
                />
              </div>
            </div>
          </div>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          <button type="submit" className="btn submit-btn" disabled={loading}>
            {loading ? <span className="spinner"></span> : "Analyze Risk"}
          </button>
        </form>

        <div className="result-wrapper">
          {result ? (
            <ResultCard result={result} />
          ) : (
            <div className="glass-card empty-result">
              <div className="empty-icon">🛡️</div>
              <h3>Ready to Analyze</h3>
              <p>Enter a URL or upload a QR code image to get started.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
