import { NavLink, Route, Routes } from 'react-router-dom'

import Cases from './pages/Cases'
import CaseTrace from './pages/CaseTrace'
import SessionView from './pages/SessionView'
import TherapistView from './pages/TherapistView'
import Insights from './pages/Insights'
import Multimodal from './pages/Multimodal'
import Bench from './pages/Bench'
import Method from './pages/Method'

export default function App() {
  return (
    <div className="shell">
      <aside className="rail">
        <div className="rail-mark">Therapy<span>Trace</span></div>
        <div className="rail-sub">Process monitoring</div>
        <nav>
          <NavLink to="/" end>Cases</NavLink>
          <NavLink to="/multimodal">Multimodal</NavLink>
          <NavLink to="/bench">Bench</NavLink>
          <NavLink to="/method">How the index works</NavLink>
        </nav>
        <div className="rail-foot">
          Decision support for reflection and supervision. Not a diagnostic
          device. Every score is relative to the client's own baseline.
        </div>
      </aside>

      <main className="main">
        <Routes>
          <Route path="/" element={<Cases />} />
          <Route path="/case/:id" element={<CaseTrace />} />
          <Route path="/case/:id/therapist" element={<TherapistView />} />
          <Route path="/case/:id/insights" element={<Insights />} />
          <Route path="/session/:sid" element={<SessionView />} />
          <Route path="/bench" element={<Bench />} />
          <Route path="/multimodal" element={<Multimodal />} />
          <Route path="/method" element={<Method />} />
        </Routes>
      </main>
    </div>
  )
}
