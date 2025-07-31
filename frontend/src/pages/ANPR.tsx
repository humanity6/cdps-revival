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
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import { CloudUpload, PhotoCamera, DirectionsCar, Add, Block } from '@mui/icons-material';
import { detectionsApi } from '../services/api';
import { ANPRDetection } from '../types';

const ANPRPage: React.FC = () => {
  const [detections, setDetections] = useState<ANPRDetection[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [webcamActive, setWebcamActive] = useState(false);
  const [addPlateDialog, setAddPlateDialog] = useState(false);
  const [newPlateNumber, setNewPlateNumber] = useState('');
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    fetchRecentDetections();
  }, []);

  const fetchRecentDetections = async () => {
    try {
      const data = await detectionsApi.searchDetections({ type: 'anpr', limit: 20 });
      setDetections(data as ANPRDetection[]);
    } catch (error) {
      console.error('Failed to fetch ANPR detections:', error);
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleDetectPlates = async () => {
    if (!selectedFile) return;

    setLoading(true);
    try {
      const result = await detectionsApi.uploadImage(selectedFile, ['anpr']);
      const anprDetections = result.detections?.anpr || [];
      setDetections(anprDetections);
    } catch (error) {
      console.error('ANPR detection failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setWebcamActive(true);
      }
    } catch (error) {
      console.error('Failed to start webcam:', error);
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
          await handleDetectPlates();
        }
      }, 'image/jpeg');
    }
  };

  const handleAddToRedList = async () => {
    if (!newPlateNumber.trim()) return;

    try {
      // Here you would typically send the plate number to the backend
      // await anprApi.addToRedList({ plate_number: newPlateNumber });
      
      setAddPlateDialog(false);
      setNewPlateNumber('');
      await fetchRecentDetections();
    } catch (error) {
      console.error('Failed to add plate to red list:', error);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Automatic Number Plate Recognition (ANPR)
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                License Plate Detection & Recognition
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
                  <Button
                    variant="outlined"
                    startIcon={<Add />}
                    onClick={() => setAddPlateDialog(true)}
                  >
                    Add to Red List
                  </Button>
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

                {previewUrl && (
                  <Box sx={{ mb: 2 }}>
                    <img
                      src={previewUrl}
                      alt="Preview"
                      style={{ width: '100%', maxWidth: '640px', height: 'auto' }}
                    />
                  </Box>
                )}

                {selectedFile && (
                  <Button
                    variant="contained"
                    onClick={handleDetectPlates}
                    disabled={loading}
                    startIcon={loading ? <CircularProgress size={20} /> : <DirectionsCar />}
                  >
                    {loading ? 'Detecting...' : 'Detect License Plates'}
                  </Button>
                )}
              </Box>

              {detections.length > 0 && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Detection Results ({detections.length} plates detected)
                  </Typography>
                  <Grid container spacing={2}>
                    {detections.map((detection, index) => (
                      <Grid item xs={12} sm={6} md={4} key={index}>
                        <Card variant="outlined">
                          <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                              <DirectionsCar sx={{ mr: 1 }} />
                              <Typography variant="h6">
                                {detection.plate_number}
                              </Typography>
                            </Box>
                            <Box sx={{ mb: 1 }}>
                              <Chip
                                size="small"
                                label={`${(detection.confidence * 100).toFixed(1)}%`}
                                color={detection.confidence > 0.8 ? 'success' : 'warning'}
                              />
                              {detection.is_red_listed && (
                                <Chip
                                  size="small"
                                  label="RED LISTED"
                                  color="error"
                                  icon={<Block />}
                                  sx={{ ml: 1 }}
                                />
                              )}
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
                Recent ANPR Detections
              </Typography>
              
              <List sx={{ maxHeight: 400, overflow: 'auto' }}>
                {detections.map((detection, index) => (
                  <ListItem key={index}>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                            {detection.plate_number}
                          </Typography>
                          {detection.is_red_listed && (
                            <Chip
                              size="small"
                              label="RED LISTED"
                              color="error"
                              sx={{ ml: 1 }}
                            />
                          )}
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography variant="caption" display="block">
                            {new Date(detection.timestamp).toLocaleString()}
                          </Typography>
                          <Chip
                            size="small"
                            label={`${(detection.confidence * 100).toFixed(1)}%`}
                            color={detection.confidence > 0.8 ? 'success' : 'warning'}
                          />
                        </Box>
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

          <Alert severity="info" sx={{ mt: 2 }}>
            <Typography variant="subtitle2">ANPR System</Typography>
            Automatically detects and recognizes license plates. Red-listed vehicles trigger immediate alerts.
          </Alert>
        </Grid>
      </Grid>

      {/* Add to Red List Dialog */}
      <Dialog open={addPlateDialog} onClose={() => setAddPlateDialog(false)}>
        <DialogTitle>Add Plate to Red List</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="License Plate Number"
            fullWidth
            variant="outlined"
            value={newPlateNumber}
            onChange={(e) => setNewPlateNumber(e.target.value.toUpperCase())}
            placeholder="e.g., ABC123"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddPlateDialog(false)}>Cancel</Button>
          <Button
            onClick={handleAddToRedList}
            variant="contained"
            disabled={!newPlateNumber.trim()}
          >
            Add to Red List
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ANPRPage;