import { useState } from 'react';
import { Link } from 'react-router-dom';
import SelfExamModal from '../components/SelfExamModal';
import './Home.css';

export default function Home() {
  const [isExamOpen, setIsExamOpen] = useState(false);

  return (
    <>
      {/* HERO */}
      <div className="hero">
        <div className="hero-inner">
          <div>
            <div className="hero-badge">● Oral Cancer Screening Platform</div>
            <h1>Detect early.<br /><em>Act sooner.</em><br />Survive.</h1>
            <p className="hero-copy">Oral cancer is highly treatable when caught early. Use our advanced AI-driven risk assessment to evaluate symptoms, and visually compare lesion images for a comprehensive combined triage.</p>
            <div className="hero-stat-row">
              <div className="hero-stat"><div className="num">84%</div><div className="lbl">5-yr survival<br />at Stage I</div></div>
              <div className="hero-stat"><div className="num">20%</div><div className="lbl">5-yr survival<br />at Stage IV</div></div>
            </div>
          </div>
          <div className="hero-visual">
            <div className="hero-card" onClick={() => setIsExamOpen(true)}>
              <div className="hc-icon ic-red">🔍</div>
              <div className="hc-info"><div className="hc-title">Self-Exam Guide</div><div className="hc-desc">Check your own mouth in 5 minutes</div></div>
              <div className="hc-arrow">→</div>
            </div>
            <Link className="hero-card" to="/questionnaire">
              <div className="hc-icon ic-teal">📋</div>
              <div className="hc-info"><div className="hc-title">Risk Screener</div><div className="hc-desc">18 questions — get a personalised risk result</div></div>
              <div className="hc-arrow">→</div>
            </Link>
            <Link className="hero-card" to="/matcher">
              <div className="hc-icon ic-red">📷</div>
              <div className="hc-info"><div className="hc-title">Image Matcher</div><div className="hc-desc">Compare a lesion photo to our clinical database</div></div>
              <div className="hc-arrow">→</div>
            </Link>
            <Link className="hero-card" to="/combined-risk">
              <div className="hc-icon ic-teal">🔬</div>
              <div className="hc-info"><div className="hc-title">Combined Triage</div><div className="hc-desc">Fuse questionnaire & image matcher for a final risk score</div></div>
              <div className="hc-arrow">→</div>
            </Link>
          </div>
        </div>
      </div>

      {/* FEATURES */}
      <div className="features-wrap">
        <div className="feature-card" onClick={() => setIsExamOpen(true)}>
          <div className="fc-badge">Start Here</div>
          <h3>Self-Exam Guide</h3>
          <p>A step-by-step visual guide to checking your own mouth at home. Know what to look for — lips, tongue, cheeks, floor of mouth, and throat.</p>
          <button className="fc-btn">Open Guide →</button>
        </div>
        <Link className="feature-card" to="/questionnaire">
          <div className="fc-badge">Risk Assessment</div>
          <h3>Risk Screener</h3>
          <p>A clinical questionnaire covering tobacco, alcohol, HPV, and 9 key symptoms. Intelligent follow-up questions help differentiate oral cancer from benign conditions.</p>
          <span className="fc-btn">Start Screener →</span>
        </Link>
        <Link className="feature-card" to="/matcher">
          <div className="fc-badge">Visual Comparison</div>
          <h3>Image Matcher</h3>
          <p>Upload a photo of an oral lesion. AI finds the most visually similar benign and malignant cases from our indexed clinical image database.</p>
          <span className="fc-btn">Open Matcher →</span>
        </Link>
        <Link className="feature-card" to="/combined-risk">
          <div className="fc-badge" style={{ color: 'var(--primary-rust)' }}>Synthesis</div>
          <h3>Combined Triage</h3>
          <p>Have you completed both the screener and image matcher? View your unified, AI-fused clinical risk assessment.</p>
          <span className="fc-btn">View Synthesis →</span>
        </Link>
        <div className="feature-card" style={{ cursor: 'default' }}>
          <div className="fc-badge" style={{ color: 'var(--sage)' }}>Always Remember</div>
          <h3>See a Clinician</h3>
          <p>These tools help you become aware — they do not diagnose. Any persistent or unexplained change lasting over 3 weeks should be evaluated by a dentist or specialist.</p>
        </div>
      </div>

      {/* HOW IT WORKS */}
      <div className="how-wrap">
        <div className="how-inner">
          <div className="section-label">How It Works</div>
          <h2>Three tools, one platform</h2>
          <div className="how-steps">
            <div className="how-step">
              <div className="hs-num">01</div>
              <div className="hs-title">Learn to self-examine</div>
              <div className="hs-desc">Follow the illustrated popup guide to inspect your mouth systematically.</div>
            </div>
            <div className="how-step">
              <div className="hs-num">02</div>
              <div className="hs-title">Answer the screener</div>
              <div className="hs-desc">Our questionnaire weighs risk factors and symptoms to assess urgency.</div>
            </div>
            <div className="how-step">
              <div className="hs-num">03</div>
              <div className="hs-title">Match a lesion photo</div>
              <div className="hs-desc">Upload an image and our AI finds the most visually similar cases.</div>
            </div>
            <div className="how-step">
              <div className="hs-num">04</div>
              <div className="hs-title">See a clinician</div>
              <div className="hs-desc">These tools inform — they don't diagnose. Always consult a dentist or specialist.</div>
            </div>
          </div>
        </div>
      </div>

      <SelfExamModal isOpen={isExamOpen} onClose={() => setIsExamOpen(false)} />
    </>
  );
}
