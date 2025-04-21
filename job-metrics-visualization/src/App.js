import React from 'react';
import { useLocation } from 'react-router-dom';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import JobMetricsViewer from './components/JobMetricsViewer';
import './App.css';

function ProtectedRoute({ children }) {
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const ownerId = queryParams.get('ownerId');
  const email = queryParams.get('email');
  const storedToken = localStorage.getItem('ATOKEN');
  let emailFromToken = null;

  if (!ownerId || !email || !storedToken) {
    console.error('Access Denied: Missing ownerId, email, or token');
    return (
      <div className="error-message">
        <h2>Access Denied</h2>
        <p>Metrics cannot be accessed. Please log in to view your metrics.</p>
      </div>
    );
  }
  try {
    const payload = JSON.parse(window.atob(storedToken.split('.')[1]));
    emailFromToken = payload.sub;
    const currentTime = Math.floor(Date.now() / 1000);
    if (payload.exp && payload.exp < currentTime) {
      console.error('Token is expired');
      return (
        <div className="error-message">
          <h2>Access Denied</h2>
          <p>Metrics cannot be accessed. Please log in to view your metrics.</p>
        </div>
      );
    }

  } catch (error) {
    console.error('Error decoding token:', error);
    return (
      <div className="error-message">
        <h2>Access Denied</h2>
        <p>Metrics cannot be accessed. Please log in to view your metrics.</p>
      </div>
    );
  }
  if (emailFromToken === email) {
    return children;
  } else {
    return (
      <div className="error-message">
        <h2>Access Denied</h2>
        <p>Metrics cannot be accessed. Please log in to view your metrics.</p>
      </div>
    );
  }
}

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
            <Route path="/metrics"
              element={
                <ProtectedRoute>
                  <JobMetricsViewer />
                </ProtectedRoute>
              } />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;