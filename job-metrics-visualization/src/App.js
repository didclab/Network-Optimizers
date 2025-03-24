import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import JobMetricsViewer from './components/JobMetricsViewer';
import './App.css';


function App() {
  return (
    <Router>
      <div className="App">
        <nav className="nav-header">
          <div className="nav-content">
            <h1>Job Metrics Visualization for One Data Share</h1>
          </div>
        </nav>
        <div className="main-content">
          <Routes>
            <Route path="/metrics/:ownerId" element={<JobMetricsViewer />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;