import React from 'react';

export default function ResultCard({ result }) {
  if (!result) return null;

  const { risk_score, is_phishing, decoded_url } = result;
  
  const scorePercent = (risk_score * 100).toFixed(1);
  const statusClass = is_phishing ? 'malicious' : 'safe';
  const statusText = is_phishing ? 'Malicious' : 'Safe';
  
  // Calculate color dynamically
  // 0% = green (#10b981), 100% = red (#ef4444)
  const hue = ((1 - risk_score) * 120).toString(10);
  const barColor = `hsl(${hue}, 80%, 50%)`;

  return (
    <div className="glass-card result-card">
      <h2 className={`result-header ${statusClass}`}>
        {statusText}
      </h2>
      
      <div className="score-text">
        {scorePercent}%
      </div>
      <div style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '1.5rem' }}>
        Risk Score
      </div>
      
      <div className="progress-container">
        <div 
          className="progress-bar" 
          style={{ 
            width: `${scorePercent}%`,
            backgroundColor: barColor 
          }}
        />
      </div>

      {decoded_url && (
        <div className="detail-text">
          <strong>Analyzed URL:</strong> <br/>
          {decoded_url}
        </div>
      )}
    </div>
  );
}
