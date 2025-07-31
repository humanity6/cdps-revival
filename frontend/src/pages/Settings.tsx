import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Grid,
  Slider,
  Switch,
  FormControlLabel,
  Divider,
  Alert,
  Tabs,
  Tab,
} from '@mui/material';
import { Save, RestoreFromTrash } from '@mui/icons-material';
import { settingsApi } from '../services/api';
import { ModuleSettings } from '../types';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div role="tabpanel" hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const SettingsPage: React.FC = () => {
  const [settings, setSettings] = useState<ModuleSettings | null>(null);
  const [loading, setLoading] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [currentTab, setCurrentTab] = useState(0);

  useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      const data = await settingsApi.getSettings();
      setSettings(data);
    } catch (error) {
      console.error('Failed to fetch settings:', error);
    }
  };

  const handleSaveSettings = async (module: string) => {
    if (!settings) return;
    
    setLoading(true);
    try {
      await settingsApi.updateSettings(module, (settings as any)[module]);
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (error) {
      console.error('Failed to save settings:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleResetSettings = async (module: string) => {
    setLoading(true);
    try {
      await settingsApi.resetSettings(module);
      await fetchSettings();
    } catch (error) {
      console.error('Failed to reset settings:', error);
    } finally {
      setLoading(false);
    }
  };

  const updateSetting = (module: string, key: string, value: any) => {
    setSettings(prev => {
      if (!prev) return null;
      return {
        ...prev,
        [module]: {
          ...(prev as any)[module],
          [key]: value,
        },
      };
    });
  };

  if (!settings) {
    return (
      <Box>
        <Typography>Loading settings...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        System Settings
      </Typography>

      {saveSuccess && (
        <Alert severity="success" sx={{ mb: 2 }}>
          Settings saved successfully!
        </Alert>
      )}

      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={currentTab} onChange={(_, newValue) => setCurrentTab(newValue)}>
            <Tab label="Face Recognition" />
            <Tab label="Weapon Detection" />
            <Tab label="Violence Detection" />
            <Tab label="ANPR" />
          </Tabs>
        </Box>

        {/* Face Recognition Settings */}
        <TabPanel value={currentTab} index={0}>
          <Typography variant="h6" gutterBottom>
            Face Recognition Configuration
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Detection Threshold</Typography>
              <Slider
                value={settings.face.detection_threshold}
                onChange={(_, value) => updateSetting('face', 'detection_threshold', value)}
                min={0.1}
                max={1.0}
                step={0.05}
                marks
                valueLabelDisplay="auto"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Recognition Threshold</Typography>
              <Slider
                value={settings.face.recognition_threshold}
                onChange={(_, value) => updateSetting('face', 'recognition_threshold', value)}
                min={0.1}
                max={1.0}
                step={0.05}
                marks
                valueLabelDisplay="auto"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Max Faces Per Frame"
                type="number"
                value={settings.face.max_faces_per_frame}
                onChange={(e) => updateSetting('face', 'max_faces_per_frame', Number(e.target.value))}
                inputProps={{ min: 1, max: 20 }}
              />
            </Grid>
          </Grid>
          
          <Divider sx={{ my: 3 }} />
          
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              variant="contained"
              startIcon={<Save />}
              onClick={() => handleSaveSettings('face')}
              disabled={loading}
            >
              Save Settings
            </Button>
            <Button
              variant="outlined"
              startIcon={<RestoreFromTrash />}
              onClick={() => handleResetSettings('face')}
              disabled={loading}
            >
              Reset to Defaults
            </Button>
          </Box>
        </TabPanel>

        {/* Weapon Detection Settings */}
        <TabPanel value={currentTab} index={1}>
          <Typography variant="h6" gutterBottom>
            Weapon Detection Configuration
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Detection Threshold</Typography>
              <Slider
                value={settings.weapon.detection_threshold}
                onChange={(_, value) => updateSetting('weapon', 'detection_threshold', value)}
                min={0.1}
                max={1.0}
                step={0.05}
                marks
                valueLabelDisplay="auto"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>NMS Threshold</Typography>
              <Slider
                value={settings.weapon.nms_threshold}
                onChange={(_, value) => updateSetting('weapon', 'nms_threshold', value)}
                min={0.1}
                max={1.0}
                step={0.05}
                marks
                valueLabelDisplay="auto"
              />
            </Grid>
          </Grid>
          
          <Divider sx={{ my: 3 }} />
          
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              variant="contained"
              startIcon={<Save />}
              onClick={() => handleSaveSettings('weapon')}
              disabled={loading}
            >
              Save Settings
            </Button>
            <Button
              variant="outlined"
              startIcon={<RestoreFromTrash />}
              onClick={() => handleResetSettings('weapon')}
              disabled={loading}
            >
              Reset to Defaults
            </Button>
          </Box>
        </TabPanel>

        {/* Violence Detection Settings */}
        <TabPanel value={currentTab} index={2}>
          <Typography variant="h6" gutterBottom>
            Violence Detection Configuration
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Detection Threshold</Typography>
              <Slider
                value={settings.violence.detection_threshold}
                onChange={(_, value) => updateSetting('violence', 'detection_threshold', value)}
                min={0.1}
                max={1.0}
                step={0.05}
                marks
                valueLabelDisplay="auto"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Frame Buffer Size"
                type="number"
                value={settings.violence.frame_buffer_size}
                onChange={(e) => updateSetting('violence', 'frame_buffer_size', Number(e.target.value))}
                inputProps={{ min: 5, max: 30 }}
              />
            </Grid>
          </Grid>
          
          <Divider sx={{ my: 3 }} />
          
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              variant="contained"
              startIcon={<Save />}
              onClick={() => handleSaveSettings('violence')}
              disabled={loading}
            >
              Save Settings
            </Button>
            <Button
              variant="outlined"
              startIcon={<RestoreFromTrash />}
              onClick={() => handleResetSettings('violence')}
              disabled={loading}
            >
              Reset to Defaults
            </Button>
          </Box>
        </TabPanel>

        {/* ANPR Settings */}
        <TabPanel value={currentTab} index={3}>
          <Typography variant="h6" gutterBottom>
            ANPR Configuration
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Detection Threshold</Typography>
              <Slider
                value={settings.anpr.detection_threshold}
                onChange={(_, value) => updateSetting('anpr', 'detection_threshold', value)}
                min={0.1}
                max={1.0}
                step={0.05}
                marks
                valueLabelDisplay="auto"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>OCR Threshold</Typography>
              <Slider
                value={settings.anpr.ocr_threshold}
                onChange={(_, value) => updateSetting('anpr', 'ocr_threshold', value)}
                min={0.1}
                max={1.0}
                step={0.05}
                marks
                valueLabelDisplay="auto"
              />
            </Grid>
          </Grid>
          
          <Divider sx={{ my: 3 }} />
          
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              variant="contained"
              startIcon={<Save />}
              onClick={() => handleSaveSettings('anpr')}
              disabled={loading}
            >
              Save Settings
            </Button>
            <Button
              variant="outlined"
              startIcon={<RestoreFromTrash />}
              onClick={() => handleResetSettings('anpr')}
              disabled={loading}
            >
              Reset to Defaults
            </Button>
          </Box>
        </TabPanel>
      </Card>
    </Box>
  );
};

export default SettingsPage;