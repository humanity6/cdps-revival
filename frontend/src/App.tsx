import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { AppLayout } from './components/Layout/AppLayout';
import Dashboard from './pages/Dashboard';
import LiveFeeds from './pages/LiveFeeds';
import FaceRecognition from './pages/FaceRecognition';
import WeaponDetection from './pages/WeaponDetection';
import ViolenceDetection from './pages/ViolenceDetection';
import ANPR from './pages/ANPR';
import Alerts from './pages/Alerts';
import Search from './pages/Search';
import Settings from './pages/Settings';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <Router>
        <AppLayout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/live" element={<LiveFeeds />} />
            <Route path="/face" element={<FaceRecognition />} />
            <Route path="/weapon" element={<WeaponDetection />} />
            <Route path="/violence" element={<ViolenceDetection />} />
            <Route path="/anpr" element={<ANPR />} />
            <Route path="/alerts" element={<Alerts />} />
            <Route path="/search" element={<Search />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </AppLayout>
      </Router>
    </ThemeProvider>
  );
}

export default App;
