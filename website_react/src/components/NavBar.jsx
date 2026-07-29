import { Link, useLocation } from 'react-router-dom';

export default function NavBar() {
  const location = useLocation();
  const path = location.pathname;

  return (
    <nav>
      <Link className="nav-logo" to="/">
        <div className="logo-mark">+</div>
        <div className="logo-text">Oral<span>Guard</span></div>
      </Link>
      <div className="nav-links">
        <Link className={`nav-link ${path === '/' ? 'active' : ''}`} to="/">Home</Link>
        <Link className={`nav-link ${path === '/questionnaire' ? 'active' : ''}`} to="/questionnaire">Risk Screener</Link>
        <Link className={`nav-link ${path === '/matcher' ? 'active' : ''}`} to="/matcher">Image Matcher</Link>
      </div>
    </nav>
  );
}
