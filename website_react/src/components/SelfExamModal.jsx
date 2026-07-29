import { useEffect } from 'react';

export default function SelfExamModal({ isOpen, onClose }) {
  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  // Prevent background scrolling when open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => { document.body.style.overflow = ''; };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="exam-modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="exam-box">
        <div className="exam-head">
          <h3>Oral Self-Examination Guide</h3>
          <button className="close-x" onClick={onClose}>✕</button>
        </div>
        <div className="exam-img-wrap">
          <img src="/img.png" alt="Oral self-exam diagram" onError={(e) => e.target.style.display='none'} />
        </div>
        <div className="exam-caption">Check all highlighted areas monthly — takes 5 minutes</div>
        <div className="exam-body">
          <div className="exam-grid">
            <div className="exam-step">
              <div className="en">01</div><div className="et">Lips &amp; Corners</div>
              <div className="ed">Pull lips away from teeth. Look for sores or colour changes on the lips and corners.</div>
            </div>
            <div className="exam-step">
              <div className="en">02</div><div className="et">Inner Cheeks &amp; Gums</div>
              <div className="ed">Pull each cheek sideways. Look for red, white, or mixed patches. Feel for lumps.</div>
            </div>
            <div className="exam-step">
              <div className="en">03</div><div className="et">Tongue — All Sides</div>
              <div className="ed">Stick out tongue. Check top, then inspect sides and underside — the most common cancer site.</div>
            </div>
            <div className="exam-step">
              <div className="en">04</div><div className="et">Floor of Mouth</div>
              <div className="ed">Press tongue to roof of mouth. Check underneath for swelling, sores, or colour changes.</div>
            </div>
            <div className="exam-step">
              <div className="en">05</div><div className="et">Palate (Roof)</div>
              <div className="ed">Tilt head back, open wide. Look for lumps or changes on the hard and soft palate.</div>
            </div>
            <div className="exam-step">
              <div className="en">06</div><div className="et">Throat &amp; Neck</div>
              <div className="ed">Say "Aah" and check the back of your throat. Feel along your neck for enlarged lymph nodes.</div>
            </div>
          </div>
          <div className="exam-signs">
            <h4>Report any of these to a dentist:</h4>
            <div className="exam-signs-list">
              <div className="es-item"><div className="es-dot"></div>Sore not healed in 3 weeks</div>
              <div className="es-item"><div className="es-dot"></div>Red or white patch that won't wipe off</div>
              <div className="es-item"><div className="es-dot"></div>Lump or rough thickening</div>
              <div className="es-item"><div className="es-dot"></div>Persistent pain or numbness</div>
              <div className="es-item"><div className="es-dot"></div>Difficulty chewing or swallowing</div>
              <div className="es-item"><div className="es-dot"></div>Hoarse voice lasting 3+ weeks</div>
              <div className="es-item"><div className="es-dot"></div>Swollen neck node 3+ weeks</div>
              <div className="es-item"><div className="es-dot"></div>Unexplained weight loss</div>
            </div>
          </div>
        </div>
        <div className="exam-foot">
          <button className="exam-close-btn" onClick={onClose}>Close Guide</button>
        </div>
      </div>
    </div>
  );
}
