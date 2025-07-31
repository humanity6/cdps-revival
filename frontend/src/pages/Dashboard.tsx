import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  Alert,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import { Bar, Line, Doughnut } from 'react-chartjs-2';
import { useWebSocket } from '../hooks/useWebSocket';
import { analyticsApi, systemApi, detectionsApi } from '../services/api';
import { Detection, SystemStatus, AnalyticsData } from '../types';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const Dashboard: React.FC = () => {
  const { isConnected, latestDetection, latestAlert } = useWebSocket();
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [recentDetections, setRecentDetections] = useState<Detection[]>([]);
  const [timeRange, setTimeRange] = useState('24h');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [timeRange]);

  const fetchDashboardData = async () => {
    try {
      const [statusData, analyticsData, detectionsData] = await Promise.all([
        systemApi.getStatus(),
        analyticsApi.getAnalytics(timeRange),
        detectionsApi.getRecentDetections(10),
      ]);
      
      setSystemStatus(statusData);
      setAnalytics(analyticsData);
      setRecentDetections(detectionsData);
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const detectionTrendsData = {
    labels: analytics?.detections_by_hour.map(d => d.hour) || [],
    datasets: [
      {
        label: 'Detections per Hour',
        data: analytics?.detections_by_hour.map(d => d.count) || [],
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1,
      },
    ],
  };

  const detectionTypeData = {
    labels: Object.keys(analytics?.detections_by_type || {}),
    datasets: [
      {
        data: Object.values(analytics?.detections_by_type || {}),
        backgroundColor: [
          '#FF6384',
          '#36A2EB',
          '#FFCE56',
          '#4BC0C0',
        ],
      },
    ],
  };

  const confidenceData = {
    labels: analytics?.confidence_distribution.map(d => d.range) || [],
    datasets: [
      {
        label: 'Detection Count',
        data: analytics?.confidence_distribution.map(d => d.count) || [],
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1,
      },
    ],
  };

  const getStatusColor = (status: boolean) => status ? 'success' : 'error';
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'default';
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <Typography>Loading dashboard...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          Dashboard
        </Typography>
        <FormControl size="small">
          <InputLabel>Time Range</InputLabel>
          <Select
            value={timeRange}
            label="Time Range"
            onChange={(e) => setTimeRange(e.target.value)}
          >
            <MenuItem value="1h">Last Hour</MenuItem>
            <MenuItem value="24h">Last 24 Hours</MenuItem>
            <MenuItem value="7d">Last 7 Days</MenuItem>
            <MenuItem value="30d">Last 30 Days</MenuItem>
          </Select>
        </FormControl>
      </Box>

      <Grid container spacing={3}>
        {/* System Status */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Status
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Chip
                    label={`System: ${systemStatus?.is_healthy ? 'Healthy' : 'Unhealthy'}`}
                    color={getStatusColor(systemStatus?.is_healthy || false)}
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Chip
                    label={`Camera: ${systemStatus?.camera_connected ? 'Connected' : 'Disconnected'}`}
                    color={getStatusColor(systemStatus?.camera_connected || false)}
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Chip
                    label={`WebSocket: ${isConnected ? 'Connected' : 'Disconnected'}`}
                    color={getStatusColor(isConnected)}
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Button variant="outlined" size="small" onClick={fetchDashboardData}>
                    Refresh
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Module Status */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Detection Modules
              </Typography>
              <Grid container spacing={1}>
                <Grid item xs={6}>
                  <Chip
                    label="Face Recognition"
                    color={getStatusColor(systemStatus?.modules_active.face || false)}
                    size="small"
                  />
                </Grid>
                <Grid item xs={6}>
                  <Chip
                    label="Weapon Detection"
                    color={getStatusColor(systemStatus?.modules_active.weapon || false)}
                    size="small"
                  />
                </Grid>
                <Grid item xs={6}>
                  <Chip
                    label="Violence Detection"
                    color={getStatusColor(systemStatus?.modules_active.violence || false)}
                    size="small"
                  />
                </Grid>
                <Grid item xs={6}>
                  <Chip
                    label="ANPR"
                    color={getStatusColor(systemStatus?.modules_active.anpr || false)}
                    size="small"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Detection Summary */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Detection Summary
              </Typography>
              <Typography variant="h3" color="primary">
                {analytics?.total_detections || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total detections in {timeRange}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Detection Trends Chart */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Detection Trends
              </Typography>
              <Line
                data={detectionTrendsData}
                options={{
                  responsive: true,
                  plugins: {
                    legend: {
                      position: 'top' as const,
                    },
                    title: {
                      display: true,
                      text: 'Detections Over Time',
                    },
                  },
                }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Detection Types Chart */}
        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Detection Types
              </Typography>
              <Doughnut
                data={detectionTypeData}
                options={{
                  responsive: true,
                  plugins: {
                    legend: {
                      position: 'bottom' as const,
                    },
                  },
                }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Confidence Distribution */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Confidence Distribution
              </Typography>
              <Bar
                data={confidenceData}
                options={{
                  responsive: true,
                  plugins: {
                    legend: {
                      display: false,
                    },
                  },
                }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Detections */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Detections
              </Typography>
              <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
                {recentDetections.map((detection) => (
                  <Box
                    key={detection.id}
                    sx={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      p: 1,
                      borderBottom: '1px solid #eee',
                    }}
                  >
                    <Box>
                      <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                        {detection.type.toUpperCase()}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {new Date(detection.timestamp).toLocaleString()}
                      </Typography>
                    </Box>
                    <Chip
                      label={`${(detection.confidence * 100).toFixed(1)}%`}
                      size="small"
                      color={detection.confidence > 0.8 ? 'success' : 'warning'}
                    />
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Latest Alert */}
        {latestAlert && (
          <Grid item xs={12}>
            <Alert severity={getSeverityColor(latestAlert.severity) as any}>
              <Typography variant="subtitle2">
                Latest Alert: {latestAlert.message}
              </Typography>
              <Typography variant="caption">
                {new Date(latestAlert.timestamp).toLocaleString()}
              </Typography>
            </Alert>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default Dashboard;