import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import JobMetricsViewer from './components/JobMetricsViewer';
import './App.css';
import { getCookie } from './utils/CookieUtils';
import NavbarComponent from './components/NavbarComponent';
function ProtectedRoute({ children }) {
  const ownerId = getCookie('email');
  const storedToken = getCookie("ATOKEN");
  console.log(ownerId + " " + storedToken);
  let emailFromToken = null;

  if (!ownerId || !storedToken) {
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
  if (emailFromToken === ownerId) {
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
        <NavbarComponent />
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