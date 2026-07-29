import { useState, useRef } from 'react';
import { Camera, Upload, Trash2, X } from 'lucide-react';
import SelfExamModal from '../components/SelfExamModal';
import './Matcher.css';

export default function Matcher() {
  const [isExamOpen, setIsExamOpen] = useState(false);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  
  const [results, setResults] = useState(null);
  const [decision, setDecision] = useState(null);
  const [combinedRisk, setCombinedRisk] = useState(null);
  
  const [lightboxData, setLightboxData] = useState(null);

  const fileInputRef = useRef(null);
  const cameraInputRef = useRef(null);
  const resultsRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      processFile(selectedFile);
    }
  };

  const processFile = (selectedFile) => {
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setResults(null);
    setDecision(null);
    setCombinedRisk(null);
    
    // Automatically submit after a short delay
    setTimeout(() => submitImage(selectedFile), 800);
  };

  const submitImage = async (imgFile) => {
    if (!imgFile) return;

    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", imgFile);
    
    // Use the same session logic
    let sid = localStorage.getItem('cbir_session_id');
    if (!sid || sid.startsWith('sess_')) {
      sid = crypto.randomUUID();
      localStorage.setItem('cbir_session_id', sid);
    }
    formData.append("session_id", sid);

    try {
      const res = await fetch("http://127.0.0.1:8000/search", {
        method: "POST",
        body: formData
      });
      
      if (!res.ok) throw new Error("API Error");
      const data = await res.json();
      
      setResults(data);
      computeDecision(data);
      
      const hasQuestionnaire = localStorage.getItem('questionnaire_done') === 'true';
      if (hasQuestionnaire) {
        await checkCombinedRisk(sid);
      }

      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);

    } catch (e) {
      console.error(e);
      alert("Error contacting the API. Please ensure the backend is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const checkCombinedRisk = async (sessionId) => {
    try {
      const res = await fetch('http://127.0.0.1:8000/combined-risk', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
      });
      if (res.ok) {
        const crData = await res.json();
        if (crData.combined_risk_label) {
          setCombinedRisk(crData);
        }
      }
    } catch (e) {
      console.error(e);
    }
  };

  const computeDecision = (data) => {
    if (!data.results) return;
    const combined = [
      ...(data.results.benign || []).map(m => ({ ...m, type: 'benign' })),
      ...(data.results.malignant || []).map(m => ({ ...m, type: 'malignant' }))
    ];
    combined.sort((a, b) => b.similarity - a.similarity);
    
    const top3 = combined.slice(0, 3);
    const malignantCount = top3.filter(m => m.type === 'malignant').length;
    
    if (malignantCount >= 2) {
      setDecision({
        type: 'urgent',
        title: 'High Visual Similarity to Malignant Cases',
        desc: 'The visual characteristics strongly align with malignant cases in our database. Urgent clinical evaluation is recommended.'
      });
    } else if (malignantCount === 1) {
      setDecision({
        type: 'refer',
        title: 'Mixed Visual Indicators',
        desc: 'The lesion shares features with both benign and malignant cases. A dentist should evaluate this in person.'
      });
    } else {
      setDecision({
        type: 'monitor',
        title: 'Predominantly Benign Visual Features',
        desc: 'The image mostly matches benign conditions. However, any lesion persisting over 3 weeks requires professional assessment.'
      });
    }
  };

  const openLightbox = (match, type) => {
    setLightboxData({ match, type });
    document.body.style.overflow = 'hidden';
  };

  const closeLightbox = () => {
    setLightboxData(null);
    document.body.style.overflow = '';
  };

  const formatSize = (bytes) => {
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  };

  return (
    <div className="page">
      <div className="page-header">
        <div className="page-badge">Visual Triage</div>
        <h1>Image Matcher</h1>
        <p>Upload a clear photo of an oral lesion. Our AI compares it against a validated clinical database to find the most visually similar cases.</p>
      </div>

      <div className="upload-section">
        <div className="upload-section-head">
          <h2>Upload or capture image</h2>
          <button className="self-exam-link" onClick={() => setIsExamOpen(true)}>
            🔍 Self-Exam Guide
          </button>
        </div>
        <div className="upload-body">
          {!previewUrl && (
            <div className="upload-btn-row">
              <input 
                type="file" 
                accept="image/*" 
                ref={fileInputRef} 
                style={{ display: 'none' }} 
                onChange={handleFileChange} 
              />
              <button className="upload-btn upload-btn-file" onClick={() => fileInputRef.current.click()}>
                <Upload size={18} /> Upload Photo
              </button>
              
              <input 
                type="file" 
                accept="image/*" 
                capture="environment" 
                ref={cameraInputRef} 
                style={{ display: 'none' }} 
                onChange={handleFileChange} 
              />
              <button className="upload-btn upload-btn-camera" onClick={() => cameraInputRef.current.click()}>
                <Camera size={18} /> Take Photo
              </button>
            </div>
          )}

          {previewUrl && (
            <div className="image-preview-area visible">
              <div className="image-preview-bar">
                <div className="image-preview-bar-left">
                  <div className="image-preview-filename">{file?.name || 'Image'}</div>
                  <div className="image-preview-size">{file ? formatSize(file.size) : ''}</div>
                </div>
                <div className="image-preview-actions">
                  <button className="preview-action-btn remove" onClick={() => setPreviewUrl(null)}>
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
              <div className="image-preview-inner">
                <img src={previewUrl} alt="Preview" />
              </div>
            </div>
          )}

          {isLoading && (
            <div className="loading-bar visible">
              <div className="loading-spinner"></div>
              <span>Analyzing image and retrieving clinical matches...</span>
            </div>
          )}

          {results && results.blur_score && results.blur_score < 100 && (
            <div className="quality-warning visible">
              <h4>Low Image Quality Detected</h4>
              <p>The image appears blurry or poorly lit, which may reduce matching accuracy.</p>
              <ul className="quality-tips">
                <li>Use flash or move to a well-lit area</li>
                <li>Tap your screen to focus before capturing</li>
                <li>Ask someone to help take the photo</li>
              </ul>
            </div>
          )}
        </div>
      </div>

      {results && (
        <div className="results-section visible" ref={resultsRef}>
          {decision && (
            <div className={`decision-banner visible banner-${decision.type}`}>
              <h3>{decision.title}</h3>
              <p>{decision.desc}</p>
            </div>
          )}

          {combinedRisk && (
            <div className={`combined-risk-card visible`}>
              <h3>Combined Triage Assessment</h3>
              <div className={`combined-risk-label ${combinedRisk.urgency_color === 'red' ? 'cr-red' : combinedRisk.urgency_color === 'green' ? 'cr-green' : 'cr-amber'}`}>
                {combinedRisk.combined_risk_label}
              </div>
              <p>{combinedRisk.recommendation}</p>
            </div>
          )}

          <div className="results-header">
            <h2>Clinical Matches</h2>
          </div>

          <div className="results-grid-wrap">
            <div className="benign-col">
              <div className="results-col-label benign-label">
                <div className="dot"></div>
                Top Benign Matches
              </div>
              <div className="img-grid">
                {(results.results?.benign || []).slice(0, 6).map((match, i) => (
                  <div className="img-card" key={i} onClick={() => openLightbox(match, 'benign')}>
                    <img src={match.image_path} alt={`Benign ${i}`} />
                    <div className="img-card-info">
                      <div className="img-card-label">{match.label || 'Benign'}</div>
                      <div className="img-card-sim">{(match.similarity * 100).toFixed(1)}% match</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="results-divider"></div>

            <div className="malignant-col">
              <div className="results-col-label malignant-label">
                <div className="dot"></div>
                Top Malignant Matches
              </div>
              <div className="img-grid">
                {(results.results?.malignant || []).slice(0, 6).map((match, i) => (
                  <div className="img-card" key={i} onClick={() => openLightbox(match, 'malignant')}>
                    <img src={match.image_path} alt={`Malignant ${i}`} />
                    <div className="img-card-info">
                      <div className="img-card-label">{match.label || 'Malignant'}</div>
                      <div className="img-card-sim">{(match.similarity * 100).toFixed(1)}% match</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {lightboxData && (
        <div className="lightbox" onClick={(e) => { if (e.target === e.currentTarget) closeLightbox(); }}>
          <div className="lb-box">
            <div className="lb-head">
              <div className="lb-head-info">
                <div className="lb-head-title">Visual Comparison</div>
                <div className={`lb-badge lb-badge-${lightboxData.type}`}>
                  {lightboxData.type.toUpperCase()} MATCH — {(lightboxData.match.similarity * 100).toFixed(1)}% SIMILAR
                </div>
              </div>
              <button className="close-x" onClick={closeLightbox}><X size={16} /></button>
            </div>
            <div className="lb-body">
              <div className="lb-col">
                <div className="lb-col-title">Your Photo</div>
                <div className="lb-img-wrap">
                  <img src={previewUrl} alt="User Upload" />
                </div>
              </div>
              <div className="lb-col">
                <div className="lb-col-title">Clinical Reference</div>
                <div className="lb-img-wrap">
                  <img src={lightboxData.match.image_path} alt="Reference Match" />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <SelfExamModal isOpen={isExamOpen} onClose={() => setIsExamOpen(false)} />
    </div>
  );
}
