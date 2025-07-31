import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  Button,
  IconButton,
  Badge,
  Alert as MuiAlert,
} from '@mui/material';
import { Notifications, CheckCircle, Warning, Error, Info } from '@mui/icons-material';
import { useWebSocket } from '../hooks/useWebSocket';
import { alertsApi } from '../services/api';
import { Alert } from '../types';

const AlertsPage: React.FC = () => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const { latestAlert } = useWebSocket();

  useEffect(() => {
    fetchAlerts();
  }, []);

  useEffect(() => {
    if (latestAlert) {
      setAlerts(prev => [latestAlert, ...prev]);
    }
  }, [latestAlert]);

  const fetchAlerts = async () => {
    try {
      const data = await alertsApi.getRecentAlerts(50);
      setAlerts(data);
    } catch (error) {
      console.error('Failed to fetch alerts:', error);
    }
  };

  const markAsRead = async (alertId: string) => {
    try {
      await alertsApi.markAlertAsRead(alertId);
      setAlerts(prev => prev.map(alert => 
        alert.id === alertId ? { ...alert, is_read: true } : alert
      ));
    } catch (error) {
      console.error('Failed to mark alert as read:', error);
    }
  };

  const clearAllAlerts = async () => {
    try {
      await alertsApi.clearAllAlerts();
      setAlerts([]);
    } catch (error) {
      console.error('Failed to clear alerts:', error);
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'high': return <Error color="error" />;
      case 'medium': return <Warning color="warning" />;
      case 'low': return <Info color="info" />;
      default: return <Notifications />;
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

  const unreadCount = alerts.filter(alert => !alert.is_read).length;

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          Real-Time Alerts
          {unreadCount > 0 && (
            <Badge badgeContent={unreadCount} color="error" sx={{ ml: 2 }} />
          )}
        </Typography>
        <Box>
          <Button variant="outlined" onClick={fetchAlerts} sx={{ mr: 1 }}>
            Refresh
          </Button>
          <Button variant="contained" color="error" onClick={clearAllAlerts}>
            Clear All
          </Button>
        </Box>
      </Box>

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Alert Timeline ({alerts.length} total)
          </Typography>
          
          {alerts.length === 0 ? (
            <MuiAlert severity="info">
              No alerts at this time. The system is monitoring for any security incidents.
            </MuiAlert>
          ) : (
            <List>
              {alerts.map((alert) => (
                <ListItem
                  key={alert.id}
                  sx={{
                    backgroundColor: alert.is_read ? 'transparent' : 'rgba(25, 118, 210, 0.04)',
                    borderRadius: 1,
                    mb: 1,
                    border: alert.is_read ? 'none' : '1px solid rgba(25, 118, 210, 0.12)',
                  }}
                >
                  <ListItemIcon>
                    {getSeverityIcon(alert.severity)}
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography
                          variant="subtitle1"
                          sx={{ fontWeight: alert.is_read ? 'normal' : 'bold' }}
                        >
                          {alert.message}
                        </Typography>
                        <Chip
                          label={alert.type.toUpperCase()}
                          size="small"
                          variant="outlined"
                        />
                        <Chip
                          label={alert.severity}
                          size="small"
                          color={getSeverityColor(alert.severity) as any}
                        />
                      </Box>
                    }
                    secondary={
                      <Typography variant="caption" color="text.secondary">
                        {new Date(alert.timestamp).toLocaleString()}
                      </Typography>
                    }
                  />
                  {!alert.is_read && (
                    <IconButton
                      onClick={() => markAsRead(alert.id)}
                      title="Mark as read"
                    >
                      <CheckCircle color="primary" />
                    </IconButton>
                  )}
                </ListItem>
              ))}
            </List>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default AlertsPage;