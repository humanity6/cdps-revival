import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Switch,
  FormControlLabel,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Alert,
  CircularProgress,
} from '@mui/material';
import { PlayArrow, Stop, Settings } from '@mui/icons-material';
import { useWebSocket } from '../hooks/useWebSocket';
import { liveApi } from '../services/api';

const LiveFeeds: React.FC = () => {
  const {
    isConnected,
    videoFrame,
    startVideoStream,
    stopVideoStream,
    updateStreamSettings,
  } = useWebSocket();

  const [isStreaming, setIsStreaming] = useState(false);
  const [cameraStatus, setCameraStatus] = useState<any>(null);
  const [modules, setModules] = useState({
    face: true,
    weapon: true,
    violence: true,
    anpr: true,
  });
  const [resolution, setResolution] = useState('640x480');
  const [fps, setFps] = useState(30);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchCameraStatus();
  }, []);

  const fetchCameraStatus = async () => {
    try {
      const status = await liveApi.getCameraStatus();
      setCameraStatus(status);
      setIsStreaming(status.is_active);
    } catch (error) {
      console.error('Failed to fetch camera status:', error);
    }
  };

  const handleStartStream = async () => {
    setLoading(true);
    try {
      await liveApi.startCamera();
      startVideoStream();
      setIsStreaming(true);
      await fetchCameraStatus();
    } catch (error) {
      console.error('Failed to start stream:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleStopStream = async () => {
    setLoading(true);
    try {
      await liveApi.stopCamera();
      stopVideoStream();
      setIsStreaming(false);
      await fetchCameraStatus();
    } catch (error) {
      console.error('Failed to stop stream:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleModuleToggle = (module: string) => {
    const newModules = { ...modules, [module]: !modules[module as keyof typeof modules] };
    setModules(newModules);
    updateStreamSettings({ modules: newModules });
  };

  const handleSettingsUpdate = () => {
    updateStreamSettings({
      resolution,
      fps,
      modules,
    });
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Live Video Feeds
      </Typography>

      <Grid container spacing={3}>
        {/* Video Stream Panel */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Live Camera Feed
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Chip
                    label={isConnected ? 'WebSocket Connected' : 'WebSocket Disconnected'}
                    color={isConnected ? 'success' : 'error'}
                    size="small"
                  />
                  <Chip
                    label={isStreaming ? 'Streaming' : 'Stopped'}
                    color={isStreaming ? 'success' : 'default'}
                    size="small"
                  />
                </Box>
              </Box>

              {/* Video Display */}
              <Box
                sx={{
                  width: '100%',
                  height: '480px',
                  backgroundColor: '#000',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: '2px solid #ccc',
                  borderRadius: 1,
                  mb: 2,
                }}
              >
                {videoFrame ? (
                  <img
                    src={`data:image/jpeg;base64,${videoFrame}`}
                    alt="Live Feed"
                    style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                  />
                ) : isStreaming ? (
                  <Box sx={{ textAlign: 'center', color: 'white' }}>
                    <CircularProgress color="inherit" />
                    <Typography variant="body2" sx={{ mt: 1 }}>
                      Waiting for video feed...
                    </Typography>
                  </Box>
                ) : (
                  <Typography variant="body1" color="white">
                    Camera feed not active
                  </Typography>
                )}
              </Box>

              {/* Stream Controls */}
              <Box sx={{ display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  startIcon={<PlayArrow />}
                  onClick={handleStartStream}
                  disabled={isStreaming || loading}
                >
                  Start Stream
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<Stop />}
                  onClick={handleStopStream}
                  disabled={!isStreaming || loading}
                >
                  Stop Stream
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<Settings />}
                  onClick={handleSettingsUpdate}
                >
                  Update Settings
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Controls Panel */}
        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Detection Modules
              </Typography>
              
              <Box sx={{ mb: 3 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={modules.face}
                      onChange={() => handleModuleToggle('face')}
                    />
                  }
                  label="Face Recognition"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={modules.weapon}
                      onChange={() => handleModuleToggle('weapon')}
                    />
                  }
                  label="Weapon Detection"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={modules.violence}
                      onChange={() => handleModuleToggle('violence')}
                    />
                  }
                  label="Violence Detection"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={modules.anpr}
                      onChange={() => handleModuleToggle('anpr')}
                    />
                  }
                  label="ANPR"
                />
              </Box>

              <Typography variant="h6" gutterBottom>
                Stream Settings
              </Typography>

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Resolution</InputLabel>
                <Select
                  value={resolution}
                  label="Resolution"
                  onChange={(e) => setResolution(e.target.value)}
                >
                  <MenuItem value="320x240">320x240</MenuItem>
                  <MenuItem value="640x480">640x480</MenuItem>
                  <MenuItem value="1280x720">1280x720</MenuItem>
                  <MenuItem value="1920x1080">1920x1080</MenuItem>
                </Select>
              </FormControl>

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>FPS</InputLabel>
                <Select
                  value={fps}
                  label="FPS"
                  onChange={(e) => setFps(e.target.value as number)}
                >
                  <MenuItem value={15}>15 FPS</MenuItem>
                  <MenuItem value={30}>30 FPS</MenuItem>
                  <MenuItem value={60}>60 FPS</MenuItem>
                </Select>
              </FormControl>

              {cameraStatus && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Camera Status
                  </Typography>
                  <Typography variant="body2">
                    Status: {cameraStatus.is_active ? 'Active' : 'Inactive'}
                  </Typography>
                  <Typography variant="body2">
                    Resolution: {cameraStatus.resolution}
                  </Typography>
                  <Typography variant="body2">
                    FPS: {cameraStatus.fps}
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>

          {!isConnected && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              WebSocket connection lost. Real-time features may not work properly.
            </Alert>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default LiveFeeds;