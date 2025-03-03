import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import JobMetricsViewer from './components/JobMetricsViewer';

function App() {
  return (
    <Router>
    <div className="App">
      <header className="App-header">
        <h1>Job Metrics Visualization</h1>
        <Routes>
            <Route path="/metrics/:ownerId" element={<JobMetricsViewer />} />
        </Routes>
      </header>
    </div>
    </Router>
  );
}

export default App;