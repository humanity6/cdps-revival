import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Chip,
  List,
  ListItem,
  ListItemText,
  CircularProgress,
  Alert,
} from '@mui/material';
import { CloudUpload, PhotoCamera, Security, Warning } from '@mui/icons-material';
import { detectionsApi } from '../services/api';
import { WeaponDetection } from '../types';

const WeaponDetectionPage: React.FC = () => {
  const [detections, setDetections] = useState<WeaponDetection[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [annotatedImageUrl, setAnnotatedImageUrl] = useState<string | null>(null);
  const [webcamActive, setWebcamActive] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    fetchRecentDetections();
  }, []);

  const fetchRecentDetections = async () => {
    try {
      const data = await detectionsApi.searchDetections({ type: 'weapon', limit: 20 });
      setDetections(data as WeaponDetection[]);
    } catch (error) {
      console.error('Failed to fetch weapon detections:', error);
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setAnnotatedImageUrl(null); // Clear previous annotations
    }
  };

  const handleDetectWeapons = async () => {
    if (!selectedFile) return;

    setLoading(true);
    try {
      const result = await detectionsApi.uploadImage(selectedFile, ['weapon']);
      const weaponDetections = result.weapon_results?.detections || [];
      setDetections(weaponDetections);
      
      // Set annotated image if available
      if (result.annotated_image) {
        setAnnotatedImageUrl(`data:image/jpeg;base64,${result.annotated_image}`);
      } else {
        setAnnotatedImageUrl(null);
      }
    } catch (error) {
      console.error('Weapon detection failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const startWebcam = async () => {
    try {
      // Stop any existing streams first
      stopWebcam();
      
      // Try different video constraints if the first one fails
      const constraints = [
        { video: { width: 640, height: 480, facingMode: 'user' } },
        { video: { width: 640, height: 480 } },
        { video: true }
      ];
      
      let stream = null;
      let lastError = null;
      
      for (const constraint of constraints) {
        try {
          stream = await navigator.mediaDevices.getUserMedia(constraint);
          break;
        } catch (err) {
          lastError = err;
          console.warn('Failed with constraint:', constraint, err);
        }
      }
      
      if (!stream) {
        throw lastError || new Error('Could not access webcam with any constraints');
      }
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
        };
        setWebcamActive(true);
      }
    } catch (error) {
      console.error('Failed to start webcam:', error);
      alert(`Failed to access webcam: ${error.message}. Please ensure no other application is using the camera and permissions are granted.`);
    }
  };

  const stopWebcam = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setWebcamActive(false);
    }
  };

  const captureFromWebcam = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const context = canvas.getContext('2d');

    if (context) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0);
      
      canvas.toBlob(async (blob) => {
        if (blob) {
          const file = new File([blob], 'webcam-capture.jpg', { type: 'image/jpeg' });
          setSelectedFile(file);
          setPreviewUrl(URL.createObjectURL(file));
          setAnnotatedImageUrl(null); // Clear previous annotations
          await handleDetectWeapons();
        }
      }, 'image/jpeg');
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'default';
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Weapon Detection
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Weapon Detection System
              </Typography>

              <Box sx={{ mb: 3 }}>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  style={{ display: 'none' }}
                />
                <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                  <Button
                    variant="outlined"
                    startIcon={<CloudUpload />}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    Upload Image
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<PhotoCamera />}
                    onClick={webcamActive ? stopWebcam : startWebcam}
                  >
                    {webcamActive ? 'Stop' : 'Start'} Webcam
                  </Button>
                  {webcamActive && (
                    <Button
                      variant="contained"
                      onClick={captureFromWebcam}
                    >
                      Capture & Detect
                    </Button>
                  )}
                </Box>

                {webcamActive && (
                  <Box sx={{ mb: 2 }}>
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      style={{ width: '100%', maxWidth: '640px', height: 'auto' }}
                    />
                    <canvas ref={canvasRef} style={{ display: 'none' }} />
                  </Box>
                )}

                {(annotatedImageUrl || previewUrl) && (
                  <Box sx={{ mb: 2 }}>
                    <img
                      src={annotatedImageUrl || previewUrl}
                      alt={annotatedImageUrl ? "Detection Results" : "Preview"}
                      style={{ width: '100%', maxWidth: '640px', height: 'auto' }}
                    />
                    {annotatedImageUrl && (
                      <Typography variant="caption" display="block" sx={{ mt: 1, textAlign: 'center' }}>
                        Image with detected weapons highlighted
                      </Typography>
                    )}
                  </Box>
                )}

                {selectedFile && (
                  <Button
                    variant="contained"
                    onClick={handleDetectWeapons}
                    disabled={loading}
                    startIcon={loading ? <CircularProgress size={20} /> : <Security />}
                  >
                    {loading ? 'Detecting...' : 'Detect Weapons'}
                  </Button>
                )}
              </Box>

              {detections.length > 0 && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Detection Results ({detections.length} weapons detected)
                  </Typography>
                  <Grid container spacing={2}>
                    {detections.map((detection, index) => (
                      <Grid item xs={12} sm={6} md={4} key={index}>
                        <Card variant="outlined">
                          <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                              <Warning color="error" sx={{ mr: 1 }} />
                              <Typography variant="subtitle1">
                                {detection.weapon_type}
                              </Typography>
                            </Box>
                            <Box sx={{ mb: 1 }}>
                              <Chip
                                size="small"
                                label={`${(detection.confidence * 100).toFixed(1)}%`}
                                color={detection.confidence > 0.8 ? 'success' : 'warning'}
                              />
                              <Chip
                                size="small"
                                label={detection.severity}
                                color={getSeverityColor(detection.severity)}
                                sx={{ ml: 1 }}
                              />
                            </Box>
                            {detection.location && (
                              <Typography variant="caption" color="text.secondary">
                                Location: ({detection.location.x}, {detection.location.y})
                              </Typography>
                            )}
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Weapon Detections
              </Typography>
              
              <List sx={{ maxHeight: 400, overflow: 'auto' }}>
                {detections.map((detection, index) => (
                  <ListItem key={index}>
                    <ListItemText
                      primary={detection.weapon_type}
                      secondary={
                        <>
                          <Typography variant="caption" display="block">
                            {new Date(detection.timestamp).toLocaleString()}
                          </Typography>
                          <Chip
                            size="small"
                            label={detection.severity}
                            color={getSeverityColor(detection.severity)}
                          />
                          <Chip
                            size="small"
                            label={`${(detection.confidence * 100).toFixed(1)}%`}
                            color={detection.confidence > 0.8 ? 'success' : 'warning'}
                            sx={{ ml: 1 }}
                          />
                        </>
                      }
                    />
                  </ListItem>
                ))}
              </List>

              <Button
                variant="outlined"
                fullWidth
                onClick={fetchRecentDetections}
                sx={{ mt: 2 }}
              >
                Refresh List
              </Button>
            </CardContent>
          </Card>

          <Alert severity="warning" sx={{ mt: 2 }}>
            <Typography variant="subtitle2">Security Alert</Typography>
            This system detects various types of weapons including firearms, knives, and other dangerous objects.
          </Alert>
        </Grid>
      </Grid>
    </Box>
  );
};

export default WeaponDetectionPage;