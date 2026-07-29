import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import NavBar from './components/NavBar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Questionnaire from './pages/Questionnaire';
import Matcher from './pages/Matcher';
import CombinedRisk from './pages/CombinedRisk';

function App() {
  return (
    <Router>
      <NavBar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/questionnaire" element={<Questionnaire />} />
        <Route path="/matcher" element={<Matcher />} />
        <Route path="/combined-risk" element={<CombinedRisk />} />
      </Routes>
      <Footer />
    </Router>
  );
}

export default App;
