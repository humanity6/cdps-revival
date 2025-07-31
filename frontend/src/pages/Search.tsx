import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Grid,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  List,
  ListItem,
  ListItemText,
  Pagination,
} from '@mui/material';
import { Search as SearchIcon, Download, FilterList } from '@mui/icons-material';
import { detectionsApi } from '../services/api';
import { Detection } from '../types';

const SearchPage: React.FC = () => {
  const [searchResults, setSearchResults] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchParams, setSearchParams] = useState({
    type: '',
    startDate: '',
    endDate: '',
    minConfidence: 0,
  });
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const resultsPerPage = 20;

  const handleSearch = async (page = 1) => {
    setLoading(true);
    try {
      const params: any = {
        limit: resultsPerPage,
        offset: (page - 1) * resultsPerPage,
      };
      
      if (searchParams.type) params.type = searchParams.type;
      if (searchParams.startDate) params.start_time = searchParams.startDate;
      if (searchParams.endDate) params.end_time = searchParams.endDate;
      if (searchParams.minConfidence > 0) params.min_confidence = searchParams.minConfidence / 100;

      const results = await detectionsApi.searchDetections(params);
      setSearchResults(results);
      setTotalPages(Math.ceil(results.length / resultsPerPage));
      setCurrentPage(page);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleExport = () => {
    const csvContent = [
      ['Type', 'Timestamp', 'Confidence', 'Details'].join(','),
      ...searchResults.map(detection => [
        detection.type,
        detection.timestamp,
        detection.confidence,
        JSON.stringify(detection.metadata || {})
      ].join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `detections_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'face': return 'primary';
      case 'weapon': return 'error';
      case 'violence': return 'error';
      case 'anpr': return 'info';
      default: return 'default';
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Smart Search
      </Typography>

      <Grid container spacing={3}>
        {/* Search Filters */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <FilterList sx={{ mr: 1 }} />
                Search Filters
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <FormControl fullWidth>
                    <InputLabel>Detection Type</InputLabel>
                    <Select
                      value={searchParams.type}
                      label="Detection Type"
                      onChange={(e) => setSearchParams(prev => ({ ...prev, type: e.target.value }))}
                    >
                      <MenuItem value="">All Types</MenuItem>
                      <MenuItem value="face">Face Recognition</MenuItem>
                      <MenuItem value="weapon">Weapon Detection</MenuItem>
                      <MenuItem value="violence">Violence Detection</MenuItem>
                      <MenuItem value="anpr">ANPR</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <TextField
                    fullWidth
                    label="Start Date"
                    type="datetime-local"
                    value={searchParams.startDate}
                    onChange={(e) => setSearchParams(prev => ({ ...prev, startDate: e.target.value }))}
                    InputLabelProps={{ shrink: true }}
                  />
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <TextField
                    fullWidth
                    label="End Date"
                    type="datetime-local"
                    value={searchParams.endDate}
                    onChange={(e) => setSearchParams(prev => ({ ...prev, endDate: e.target.value }))}
                    InputLabelProps={{ shrink: true }}
                  />
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <TextField
                    fullWidth
                    label="Min Confidence (%)"
                    type="number"
                    value={searchParams.minConfidence}
                    onChange={(e) => setSearchParams(prev => ({ ...prev, minConfidence: Number(e.target.value) }))}
                    inputProps={{ min: 0, max: 100 }}
                  />
                </Grid>
              </Grid>
              
              <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  startIcon={<SearchIcon />}
                  onClick={() => handleSearch(1)}
                  disabled={loading}
                >
                  {loading ? 'Searching...' : 'Search'}
                </Button>
                
                <Button
                  variant="outlined"
                  startIcon={<Download />}
                  onClick={handleExport}
                  disabled={searchResults.length === 0}
                >
                  Export Results
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Search Results */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Search Results ({searchResults.length} found)
              </Typography>
              
              {searchResults.length === 0 ? (
                <Typography color="text.secondary">
                  No results found. Try adjusting your search criteria.
                </Typography>
              ) : (
                <>
                  <List>
                    {searchResults.map((detection) => (
                      <ListItem key={detection.id} divider>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Chip
                                label={detection.type.toUpperCase()}
                                size="small"
                                color={getTypeColor(detection.type) as any}
                              />
                              <Typography variant="subtitle1">
                                Detection ID: {detection.id}
                              </Typography>
                              <Chip
                                label={`${(detection.confidence * 100).toFixed(1)}%`}
                                size="small"
                                color={detection.confidence > 0.8 ? 'success' : 'warning'}
                              />
                            </Box>
                          }
                          secondary={
                            <Box>
                              <Typography variant="body2" color="text.secondary">
                                {new Date(detection.timestamp).toLocaleString()}
                              </Typography>
                              {detection.location && (
                                <Typography variant="caption" color="text.secondary">
                                  Location: ({detection.location.x}, {detection.location.y})
                                </Typography>
                              )}
                              {detection.metadata && Object.keys(detection.metadata).length > 0 && (
                                <Typography variant="caption" color="text.secondary" display="block">
                                  Additional data available
                                </Typography>
                              )}
                            </Box>
                          }
                        />
                      </ListItem>
                    ))}
                  </List>
                  
                  {totalPages > 1 && (
                    <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
                      <Pagination
                        count={totalPages}
                        page={currentPage}
                        onChange={(_, page) => handleSearch(page)}
                        color="primary"
                      />
                    </Box>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SearchPage;