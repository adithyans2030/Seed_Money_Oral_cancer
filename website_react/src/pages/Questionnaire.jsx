import { useState, useRef, useEffect } from 'react';
import SelfExamModal from '../components/SelfExamModal';
import './Questionnaire.css';

export default function Questionnaire() {
  const [isExamOpen, setIsExamOpen] = useState(false);
  const [answers, setAnswers] = useState({});
  const [demographics, setDemographics] = useState({ age: '', gender: '1', region: '' });
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [resultData, setResultData] = useState(null);
  const [combinedRisk, setCombinedRisk] = useState(null);
  
  const resultCardRef = useRef(null);

  // Constants
  const totalQuestions = 21; // Estimate based on original logic
  const answeredCount = Object.keys(answers).length;
  const progressPct = Math.min((answeredCount / totalQuestions) * 100, 100);

  const setAnswer = (qId, val) => {
    setAnswers(prev => ({ ...prev, [qId]: val }));
  };

  const resetForm = () => {
    setAnswers({});
    setDemographics({ age: '', gender: '1', region: '' });
    setResultData(null);
    setCombinedRisk(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const evaluate = async () => {
    const { age, gender, region } = demographics;
    if (!age) {
      alert("Please enter your age.");
      return;
    }

    setIsEvaluating(true);

    const payload = {
      answers: {
        age: parseInt(age),
        gender: parseInt(gender),
        region: region || 'Unknown',
        q_tobacco: answers.c1 === 'yes' ? 1 : 0,
        q_alcohol: answers.c4 === 'yes' ? 1 : 0,
        q_hpv: answers.c5 === 'yes' ? 1 : 0,
        q_betel: answers.c3 === 'yes' ? 1 : 0,
        q_sun: answers.c6 === 'yes' ? 1 : 0,
        q_hygiene: 0,
        q_diet: 0,
        q_family: answers.c7 === 'yes' ? 1 : 0,
        q_immune: answers.c9 === 'yes' ? 1 : 0,
        q_lesions: (answers.s1 === 'yes' || answers.c8 === 'yes') ? 1 : 0,
        q_bleeding: 0,
        q_swallowing: answers.s5 === 'yes' ? 1 : 0,
        q_patches: answers.s2 === 'yes' ? 1 : 0,
        questionnaire_raw: answers
      },
      session_id: (() => {
        let sid = localStorage.getItem('cbir_session_id');
        if (sid && sid.startsWith('sess_')) return null;
        return sid || null;
      })()
    };

    try {
      const res = await fetch('http://127.0.0.1:8000/predict-risk', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error('API Error');
      const data = await res.json();
      
      setResultData({
        label: data.risk_label,
        score: data.risk_score,
        features: data.top_features
      });
      
      if (data.session_id) {
        localStorage.setItem('cbir_session_id', data.session_id);
      }
      
      localStorage.setItem('questionnaire_done', 'true');

      const hasCBIR = localStorage.getItem('cbir_session_id') !== null;
      if (hasCBIR) {
        await checkCombinedRisk(localStorage.getItem('cbir_session_id'));
      }
    } catch (e) {
      console.error("Full /predict-risk error:", e);
      alert("Risk assessment service is temporarily unavailable. Please try again.");
      throw e;
    } finally {
      setIsEvaluating(false);
    }
  };

  const fallbackEvaluate = () => {
    const causeKeys = ['c1','c2','c3','c4','c5','c6','c7','c8','c9'];
    const symptomKeys = ['s1','s2','s3','s4','s5','s6','s7','s8','s9'];
    const diffKeys = ['d1','d2','d3','d4','d5','d6','d7','d8'];

    const hasCause = causeKeys.some(k => answers[k] === 'yes');
    const hasSymptom = symptomKeys.some(k => answers[k] === 'yes');
    const diffPresent = diffKeys.some(k => answers[k] === 'yes');

    let label, score = 0, features = [];

    if (hasCause && hasSymptom) {
      label = 'High';
      score = 0.9;
    } else if (hasCause && !hasSymptom) {
      label = 'Moderate';
      score = 0.6;
    } else if (!hasCause && hasSymptom && !diffPresent) {
      label = 'Moderate';
      score = 0.5;
    } else if (!hasCause && hasSymptom && diffPresent) {
      label = 'Moderate';
      score = 0.4;
    } else {
      label = 'Low';
      score = 0.1;
    }

    setResultData({ label, score, features, isFallback: true, diffPresent });
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

  useEffect(() => {
    if (resultData && resultCardRef.current) {
      resultCardRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [resultData]);

  const renderQuestion = (id, text, hint = null) => (
    <div className="question-item">
      <div className="q-text">
        {text}
        {hint && <small>{hint}</small>}
      </div>
      <div className="toggle-group">
        <button 
          className={`toggle-btn yes ${answers[id] === 'yes' ? 'active' : ''}`} 
          onClick={() => setAnswer(id, 'yes')}
        >Yes</button>
        <button 
          className={`toggle-btn no ${answers[id] === 'no' ? 'active' : ''}`} 
          onClick={() => {
            setAnswer(id, 'no');
            // reset diff if any
            if (id === 's1') setAnswer('d1', null);
            if (id === 's2') setAnswer('d2', null);
            if (id === 's3') setAnswer('d3', null);
            if (id === 's4') setAnswer('d4', null);
            if (id === 's5') setAnswer('d5', null);
            if (id === 's6') setAnswer('d6', null);
            if (id === 's7') setAnswer('d7', null);
            if (id === 's8') setAnswer('d8', null);
          }}
        >No</button>
      </div>
    </div>
  );

  const renderDiff = (triggerId, diffId, text, hint = null) => (
    <div className={`diff-block ${answers[triggerId] === 'yes' ? 'visible' : ''}`}>
      <div className="diff-block-header">🔬 Follow-up — Differentiating the symptom</div>
      <div className="diff-question">
        <div className="q-text">
          {text}
          {hint && <small>{hint}</small>}
        </div>
        <div className="toggle-group">
          <button 
            className={`toggle-btn yes ${answers[diffId] === 'yes' ? 'active' : ''}`}
            onClick={() => setAnswer(diffId, 'yes')}
          >Yes</button>
          <button 
            className={`toggle-btn no ${answers[diffId] === 'no' ? 'active' : ''}`}
            onClick={() => setAnswer(diffId, 'no')}
          >No</button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="container">
      <div className="masthead">
        <div className="label">Clinical Screening Tool</div>
        <h1>Oral Cancer<br />Risk Questionnaire</h1>
        <p>This screener evaluates risk factors and symptoms associated with oral cancer. It dynamically adjusts questions based on your responses.</p>
        <div className="disclaimer">
          ⚠ This tool is for educational purposes and is not a substitute for professional clinical evaluation.
        </div>
      </div>

      <div className="exam-trigger" onClick={() => setIsExamOpen(true)}>
        <div className="et-icon">🔍</div>
        <div className="et-info">
          <div className="et-title">Before you begin — do a quick self-exam</div>
          <div className="et-desc">Open the visual guide to check your mouth first, then answer the questions below.</div>
        </div>
        <div className="et-arrow">→</div>
      </div>

      <div className="progress-bar-wrap">
        <div className="progress-bar" style={{ width: `${progressPct}%` }}></div>
      </div>

      <div className="section" id="demographicsSection">
        <div className="section-header">
          <div className="section-icon icon-cause">&#128100;</div>
          <div>
            <h2>Demographics</h2>
            <div className="subtitle">Basic information to improve risk assessment accuracy.</div>
          </div>
        </div>
        
        <div className="question-item">
          <div className="q-text">Age:</div>
          <div style={{ flexShrink: 0 }}>
            <input 
              type="number" 
              placeholder="e.g. 45" 
              value={demographics.age} 
              onChange={e => setDemographics({...demographics, age: e.target.value})}
              style={{ padding: '8px', border: '1px solid var(--border)', borderRadius: '4px', width: '100px', fontFamily: 'inherit' }}
            />
          </div>
        </div>
        <div className="question-item">
          <div className="q-text">Gender:</div>
          <div style={{ flexShrink: 0 }}>
            <select 
              value={demographics.gender}
              onChange={e => setDemographics({...demographics, gender: e.target.value})}
              style={{ padding: '8px', border: '1px solid var(--border)', borderRadius: '4px', width: '140px', fontFamily: 'inherit' }}
            >
              <option value="1">Male</option>
              <option value="0">Female/Other</option>
            </select>
          </div>
        </div>
        <div className="question-item">
          <div className="q-text">Region (Optional):</div>
          <div style={{ flexShrink: 0 }}>
            <input 
              type="text" 
              placeholder="e.g. India" 
              value={demographics.region}
              onChange={e => setDemographics({...demographics, region: e.target.value})}
              style={{ padding: '8px', border: '1px solid var(--border)', borderRadius: '4px', width: '140px', fontFamily: 'inherit' }}
            />
          </div>
        </div>
      </div>

      <div className="section">
        <div className="section-header">
          <div className="section-icon icon-cause">⚡</div>
          <div>
            <h2>Section A: Risk Factors</h2>
            <div className="subtitle">Habits and history that increase susceptibility</div>
          </div>
        </div>
        
        {renderQuestion('c1', '1. Do you currently use, or have you ever regularly used, smoked tobacco?', 'Includes cigarettes, cigars, pipes, or bidi.')}
        {renderQuestion('c2', '2. Do you currently use, or have you ever regularly used, smokeless tobacco?', 'Includes chewing tobacco, snuff, dip, snus, khaini, or gutka.')}
        {renderQuestion('c3', '3. Do you regularly chew areca nut (betel nut) or paan?', 'Highly common in South and Southeast Asia.')}
        {renderQuestion('c4', '4. Do you consume alcohol heavily or frequently?', 'Defined as 3+ drinks per day for men, 2+ for women, or frequent binge drinking.')}
        {renderQuestion('c5', '5. Have you ever been diagnosed with HPV (Human Papillomavirus)?', 'Specifically high-risk strains like HPV-16.')}
        {renderQuestion('c6', '6. Do you have a history of frequent, unprotected sun exposure to your lips?', 'Relevant for lip cancer risk.')}
        {renderQuestion('c7', '7. Do you have a family history of oral or other head and neck cancers?')}
        {renderQuestion('c8', '8. Have you previously been diagnosed with a potentially malignant oral disorder?', 'Examples: Leukoplakia, Erythroplakia, Oral Submucous Fibrosis.')}
        {renderQuestion('c9', '9. Do you have a compromised immune system?', 'Due to HIV/AIDS, immunosuppressive drugs, or organ transplant.')}
      </div>

      <div className="section">
        <div className="section-header">
          <div className="section-icon icon-symptom">🩺</div>
          <div>
            <h2>Section B: Active Symptoms</h2>
            <div className="subtitle">Physical changes you have noticed recently</div>
          </div>
        </div>

        {renderQuestion('s1', '10. Do you have a sore, ulcer, or lump in your mouth that has not healed within 3 weeks?')}
        {renderDiff('s1', 'd1', 'Did this sore appear immediately after a minor injury?', 'e.g., biting your cheek, sharp food, or a broken tooth rubbing the area.')}

        {renderQuestion('s2', '11. Do you have a white or red patch in your mouth that you cannot wipe off?')}
        {renderDiff('s2', 'd2', 'Is the patch painful, burning, or sensitive to spicy foods?', 'Some benign conditions like thrush or lichen planus can present this way.')}

        {renderQuestion('s3', '12. Have you experienced unexplained bleeding in your mouth?')}
        {renderDiff('s3', 'd3', 'Does the bleeding only occur when brushing or flossing?', 'May indicate gum disease rather than malignancy.')}

        {renderQuestion('s4', '13. Have you noticed any loose teeth with no obvious dental cause?')}
        {renderDiff('s4', 'd4', 'Have you previously been diagnosed with severe gum disease (periodontitis)?')}

        {renderQuestion('s5', '14. Are you experiencing difficulty or pain when chewing or swallowing?')}
        {renderDiff('s5', 'd5', 'Did this difficulty start suddenly alongside a cold or sore throat?')}

        {renderQuestion('s6', '15. Do you have a persistent sore throat or feel like something is caught in your throat?')}
        {renderDiff('s6', 'd6', 'Do you suffer from frequent acid reflux (heartburn)?')}

        {renderQuestion('s7', '16. Is your voice persistently hoarse or has it changed significantly over the last 3 weeks?')}
        {renderDiff('s7', 'd7', 'Have you recently had a severe respiratory infection or used your voice excessively?')}

        {renderQuestion('s8', '17. Do you have a lump or swelling in your neck that has lasted more than 3 weeks?')}
        {renderDiff('s8', 'd8', 'Is the lump tender, and did it appear at the same time as an infection or toothache?')}

        {renderQuestion('s9', '18. Have you experienced numbness in your tongue, lips, or mouth?')}
      </div>

      <div className="submit-row">
        <button className="submit-btn" onClick={evaluate} disabled={isEvaluating}>
          {isEvaluating ? 'Analyzing...' : 'Evaluate My Risk'}
        </button>
      </div>

      <div className={`result-card ${resultData ? 'visible' : ''}`} ref={resultCardRef}>
        {resultData && (
          <>
            <div className={`result-header ${resultData.label === 'High' ? 'risk-high' : resultData.label === 'Moderate' ? 'risk-caution' : 'risk-none'}`}>
              <div className="risk-label">ASSESSMENT RESULT</div>
              <h3>
                {resultData.label === 'High' ? 'High Risk — Seek Immediate Consultation' : 
                 resultData.label === 'Moderate' ? 'Elevated Risk — Caution Advised' : 
                 'No Immediate Concerns Identified'}
              </h3>
            </div>
            <div className="result-body">
              <p>Your risk score is <strong>{(resultData.score * 100).toFixed(1)}%</strong>. 
                {resultData.label === 'High' ? ' Please consult a dentist or oncologist as soon as possible.' : 
                 resultData.label === 'Moderate' ? ' Schedule a routine oral cancer screening with your dentist.' : 
                 ' Maintain a healthy diet and limit alcohol consumption.'}
              </p>
              
              {resultData.features && resultData.features.length > 0 && (
                <p><strong>Top driving factors:</strong> {resultData.features.join(', ')}</p>
              )}

              {resultData.isFallback && resultData.diffPresent && (
                <div style={{ marginTop: '12px', padding: '12px', background: 'var(--sage-light)', borderLeft: '3px solid var(--sage)', color: 'var(--sage)', fontSize: '13px' }}>
                  Some of your symptoms may overlap with other conditions based on your follow-up answers. A clinician will rule out alternatives through examination.
                </div>
              )}

              {combinedRisk && (
                <div style={{ 
                  marginTop: '16px', padding: '16px', border: '1px solid #d9cec4', borderRadius: '8px',
                  backgroundColor: combinedRisk.urgency_color === 'red' ? 'var(--rust-light)' : combinedRisk.urgency_color === 'green' ? 'var(--sage-light)' : '#fff8e6'
                }}>
                  <h4 style={{ marginBottom: '8px', fontFamily: '"Playfair Display", serif', fontSize: '18px', fontWeight: 'normal' }}>Combined Triage Assessment</h4>
                  <div style={{ fontWeight: 600, marginBottom: '8px', fontSize: '14px' }}>{combinedRisk.combined_risk_label}</div>
                  <p style={{ fontSize: '14px', color: 'var(--ink)' }}>{combinedRisk.recommendation}</p>
                </div>
              )}

              <button className="reset-btn" onClick={resetForm}>↻ Start Over</button>
            </div>
          </>
        )}
      </div>

      <SelfExamModal isOpen={isExamOpen} onClose={() => setIsExamOpen(false)} />
    </div>
  );
}
