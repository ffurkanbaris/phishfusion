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
  const [serverUrl, setServerUrl] = useState('http://localhost:8000');

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
        <p>Mobile Phishing & QR Scanner</p>
      </div>

      <form className="glass-card" onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="serverUrl">Backend Server API URL</label>
          <input
            type="text"
            id="serverUrl"
            className="input-field"
            value={serverUrl}
            onChange={(e) => setServerUrl(e.target.value)}
            style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}
          />
        </div>

        <div className="form-group">
          <label htmlFor="url">Check Text URL</label>
          <input
            type="text"
            id="url"
            className="input-field"
            placeholder="https://example.com"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
        </div>

        <div className="form-group" style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
          - OR -
        </div>

        <div className="form-group">
          <label>Check QR Code</label>
          <button 
            type="button" 
            className="btn" 
            style={{ marginBottom: '1rem', backgroundColor: '#4f46e5' }}
            onClick={takePicture}
          >
            📸 Scan with Camera
          </button>
          
          <div className="file-upload">
            {file ? file.name : "Select image from gallery"}
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleFileChange} 
            />
          </div>
        </div>

        {error && (
          <div style={{ color: 'var(--danger-color)', marginBottom: '1rem', textAlign: 'center' }}>
            {error}
          </div>
        )}

        <button type="submit" className="btn" disabled={loading}>
          {loading ? <span className="spinner"></span> : "Analyze Risk"}
        </button>
      </form>

      <ResultCard result={result} />
    </div>
  );
}

export default App;
