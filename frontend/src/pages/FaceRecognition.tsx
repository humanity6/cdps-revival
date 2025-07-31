import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Avatar,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  CircularProgress,
  Alert,
} from '@mui/material';
import { CloudUpload, PhotoCamera, Person, PersonAdd } from '@mui/icons-material';
import { detectionsApi } from '../services/api';
import { FaceDetection } from '../types';

const FaceRecognition: React.FC = () => {
  const [detections, setDetections] = useState<FaceDetection[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [webcamActive, setWebcamActive] = useState(false);
  const [addPersonDialog, setAddPersonDialog] = useState(false);
  const [newPersonName, setNewPersonName] = useState('');
  const [selectedDetection, setSelectedDetection] = useState<FaceDetection | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    fetchRecentDetections();
  }, []);

  const fetchRecentDetections = async () => {
    try {
      const data = await detectionsApi.searchDetections({ type: 'face', limit: 20 });
      setDetections(data as FaceDetection[]);
    } catch (error) {
      console.error('Failed to fetch face detections:', error);
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

  const handleDetectFaces = async () => {
    if (!selectedFile) return;

    setLoading(true);
    try {
      const result = await detectionsApi.uploadImage(selectedFile, ['face']);
      const faceDetections = result.detections?.face || [];
      setDetections(faceDetections);
    } catch (error) {
      console.error('Face detection failed:', error);
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
          await handleDetectFaces();
        }
      }, 'image/jpeg');
    }
  };

  const handleAddPerson = (detection: FaceDetection) => {
    setSelectedDetection(detection);
    setAddPersonDialog(true);
  };

  const savePerson = async () => {
    if (!selectedDetection || !newPersonName.trim()) return;

    try {
      // Here you would typically send the face encoding and name to the backend
      // await faceApi.addPerson({ name: newPersonName, encoding: selectedDetection.face_encoding });
      
      setAddPersonDialog(false);
      setNewPersonName('');
      setSelectedDetection(null);
      await fetchRecentDetections();
    } catch (error) {
      console.error('Failed to add person:', error);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.9) return 'success';
    if (confidence > 0.7) return 'warning';
    return 'error';
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Face Recognition
      </Typography>

      <Grid container spacing={3}>
        {/* Upload and Detection Panel */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Face Detection & Recognition
              </Typography>

              {/* Upload Section */}
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

                {/* Webcam View */}
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

                {/* Image Preview */}
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
                    onClick={handleDetectFaces}
                    disabled={loading}
                    startIcon={loading ? <CircularProgress size={20} /> : <Person />}
                  >
                    {loading ? 'Detecting...' : 'Detect Faces'}
                  </Button>
                )}
              </Box>

              {/* Detection Results */}
              {detections.length > 0 && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Detection Results ({detections.length} faces)
                  </Typography>
                  <Grid container spacing={2}>
                    {detections.map((detection, index) => (
                      <Grid item xs={12} sm={6} md={4} key={index}>
                        <Card variant="outlined">
                          <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                              <Avatar sx={{ mr: 2 }}>
                                <Person />
                              </Avatar>
                              <Box>
                                <Typography variant="subtitle1">
                                  {detection.person_name || 'Unknown Person'}
                                </Typography>
                                <Chip
                                  size="small"
                                  label={`${(detection.confidence * 100).toFixed(1)}%`}
                                  color={getConfidenceColor(detection.confidence)}
                                />
                              </Box>
                            </Box>
                            
                            {detection.location && (
                              <Typography variant="caption" color="text.secondary">
                                Location: ({detection.location.x}, {detection.location.y})
                              </Typography>
                            )}
                            
                            {detection.is_unknown && (
                              <Box sx={{ mt: 1 }}>
                                <Button
                                  size="small"
                                  startIcon={<PersonAdd />}
                                  onClick={() => handleAddPerson(detection)}
                                >
                                  Add Person
                                </Button>
                              </Box>
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

        {/* Recent Detections Panel */}
        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Face Detections
              </Typography>
              
              <List sx={{ maxHeight: 400, overflow: 'auto' }}>
                {detections.map((detection, index) => (
                  <ListItem key={index}>
                    <ListItemAvatar>
                      <Avatar>
                        <Person />
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary={detection.person_name || 'Unknown'}
                      secondary={
                        <>
                          <Typography variant="caption" display="block">
                            {new Date(detection.timestamp).toLocaleString()}
                          </Typography>
                          <Chip
                            size="small"
                            label={`${(detection.confidence * 100).toFixed(1)}%`}
                            color={getConfidenceColor(detection.confidence)}
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

          <Alert severity="info" sx={{ mt: 2 }}>
            Upload an image or use webcam to detect and recognize faces in real-time.
          </Alert>
        </Grid>
      </Grid>

      {/* Add Person Dialog */}
      <Dialog open={addPersonDialog} onClose={() => setAddPersonDialog(false)}>
        <DialogTitle>Add New Person</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Person Name"
            fullWidth
            variant="outlined"
            value={newPersonName}
            onChange={(e) => setNewPersonName(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddPersonDialog(false)}>Cancel</Button>
          <Button
            onClick={savePerson}
            variant="contained"
            disabled={!newPersonName.trim()}
          >
            Add Person
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default FaceRecognition;