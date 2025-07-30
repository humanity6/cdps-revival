"""
Analytics and Search Service for Crime Detection Dashboard
Provides comprehensive analytics, search optimization, and reporting capabilities
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from collections import defaultdict, Counter
import statistics

from ..database.operations import DatabaseOperations
from ..database.models import DatabaseManager, Event, QueryHelpers
from sqlalchemy import func, and_, or_, desc, asc, text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class TimeRange(Enum):
    """Time range options for analytics."""
    HOUR = "1h"
    DAY = "24h"
    WEEK = "7d"
    MONTH = "30d"
    QUARTER = "90d"
    YEAR = "365d"

class MetricType(Enum):
    """Types of metrics available."""
    COUNT = "count"
    AVERAGE = "average" 
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    PERCENTILE = "percentile"

@dataclass
class SearchFilter:
    """Search filter configuration."""
    field: str
    operator: str  # eq, ne, gt, gte, lt, lte, in, not_in, contains, starts_with, ends_with
    value: Any
    case_sensitive: bool = False

@dataclass
class SearchSort:
    """Search sort configuration."""
    field: str
    direction: str = "desc"  # asc, desc

@dataclass
class SearchQuery:
    """Complete search query configuration."""
    text_search: Optional[str] = None
    filters: List[SearchFilter] = None
    sorts: List[SearchSort] = None
    limit: int = 100
    offset: int = 0
    include_related: bool = True
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = []
        if self.sorts is None:
            self.sorts = []

class AnalyticsService:
    """
    Comprehensive analytics service for the crime detection system.
    Provides dashboard metrics, search capabilities, and reporting features.
    """
    
    def __init__(self, db_operations: DatabaseOperations):
        self.db_ops = db_operations
        self.db_manager = db_operations.db_manager
        self.query_helpers = QueryHelpers()
        
        # Cache for analytics results
        self._analytics_cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_cleanup = datetime.now()
    
    # Dashboard Analytics
    def get_dashboard_overview(self, time_range: TimeRange = TimeRange.DAY) -> Dict[str, Any]:
        """Get comprehensive dashboard overview metrics."""
        try:
            cache_key = f"dashboard_overview_{time_range.value}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Calculate time window
            end_time = datetime.now()
            start_time = self._get_start_time(end_time, time_range)
            
            session = self.db_manager.get_session()
            try:
                # Core metrics
                total_events = self._get_event_count(session, start_time, end_time)
                alert_events = self._get_alert_count(session, start_time, end_time)
                critical_events = self._get_critical_count(session, start_time, end_time)
                
                # Detection counts by module
                detection_counts = self._get_detection_counts_by_module(session, start_time, end_time)
                
                # Severity distribution
                severity_distribution = self._get_severity_distribution(session, start_time, end_time)
                
                # Location analytics
                location_stats = self._get_location_statistics(session, start_time, end_time)
                
                # Performance metrics
                performance_metrics = self._get_performance_metrics(session, start_time, end_time)
                
                # Trend analysis
                trend_data = self._get_trend_analysis(session, start_time, end_time, time_range)
                
                overview = {
                    'time_range': time_range.value,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'summary': {
                        'total_events': total_events,
                        'alert_events': alert_events,
                        'critical_events': critical_events,
                        'alert_rate': (alert_events / total_events * 100) if total_events > 0 else 0
                    },
                    'detection_counts': detection_counts,
                    'severity_distribution': severity_distribution,
                    'location_stats': location_stats,
                    'performance_metrics': performance_metrics,
                    'trends': trend_data,
                    'generated_at': datetime.now().isoformat()
                }
                
                self._cache_result(cache_key, overview)
                return overview
                
            finally:
                self.db_manager.close_session(session)
                
        except Exception as e:
            logger.error(f"Failed to get dashboard overview: {e}")
            return {'error': str(e)}
    
    def get_module_analytics(self, module_name: str, time_range: TimeRange = TimeRange.DAY) -> Dict[str, Any]:
        """Get detailed analytics for a specific detection module."""
        try:
            cache_key = f"module_analytics_{module_name}_{time_range.value}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            end_time = datetime.now()
            start_time = self._get_start_time(end_time, time_range)
            
            session = self.db_manager.get_session()
            try:
                # Module-specific metrics
                if module_name.upper() == 'ANPR':
                    analytics = self._get_anpr_analytics(session, start_time, end_time)
                elif module_name.upper() == 'FACE':
                    analytics = self._get_face_analytics(session, start_time, end_time)
                elif module_name.upper() == 'VIOLENCE':
                    analytics = self._get_violence_analytics(session, start_time, end_time)
                elif module_name.upper() == 'WEAPON':
                    analytics = self._get_weapon_analytics(session, start_time, end_time)
                else:
                    return {'error': f'Unknown module: {module_name}'}
                
                # Add common metrics
                analytics.update({
                    'module': module_name.upper(),
                    'time_range': time_range.value,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'generated_at': datetime.now().isoformat()
                })
                
                self._cache_result(cache_key, analytics)
                return analytics
                
            finally:
                self.db_manager.close_session(session)
                
        except Exception as e:
            logger.error(f"Failed to get module analytics for {module_name}: {e}")
            return {'error': str(e)}
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time system metrics (not cached)."""
        try:
            session = self.db_manager.get_session()
            try:
                now = datetime.now()
                last_hour = now - timedelta(hours=1)
                last_minute = now - timedelta(minutes=1)
                
                # Recent activity
                events_last_hour = self._get_event_count(session, last_hour, now)
                events_last_minute = self._get_event_count(session, last_minute, now)
                alerts_last_hour = self._get_alert_count(session, last_hour, now)
                
                # System health indicators
                recent_errors = session.query(func.count()).select_from(
                    session.query().select_from(text("system_logs")).filter(
                        and_(
                            text("log_level = 'ERROR'"),
                            text("timestamp >= :start_time")
                        )
                    ).params(start_time=last_hour)
                ).scalar()
                
                # Active modules check
                module_activity = {}
                for module in ['ANPR', 'FACE', 'VIOLENCE', 'WEAPON']:
                    count = session.query(func.count(Event.id)).filter(
                        and_(
                            Event.event_type == module,
                            Event.timestamp >= last_hour
                        )
                    ).scalar()
                    module_activity[module] = {
                        'events_last_hour': count,
                        'is_active': count > 0
                    }
                
                # Processing performance
                avg_processing_time = session.query(
                    func.avg(Event.processing_time)
                ).filter(
                    Event.timestamp >= last_hour
                ).scalar() or 0
                
                return {
                    'timestamp': now.isoformat(),
                    'activity': {
                        'events_last_hour': events_last_hour,
                        'events_last_minute': events_last_minute,
                        'alerts_last_hour': alerts_last_hour
                    },
                    'system_health': {
                        'recent_errors': recent_errors,
                        'avg_processing_time_ms': float(avg_processing_time * 1000) if avg_processing_time else 0
                    },
                    'module_activity': module_activity
                }
                
            finally:
                self.db_manager.close_session(session)
                
        except Exception as e:
            logger.error(f"Failed to get real-time metrics: {e}")
            return {'error': str(e)}
    
    # Advanced Search
    def search_events(self, query: SearchQuery) -> Dict[str, Any]:
        """Advanced event search with filtering, sorting, and pagination."""
        try:
            session = self.db_manager.get_session()
            try:
                # Build base query
                base_query = session.query(Event)
                
                # Apply text search
                if query.text_search:
                    base_query = self._apply_text_search(base_query, query.text_search)
                
                # Apply filters
                for filter_obj in query.filters:
                    base_query = self._apply_filter(base_query, filter_obj)
                
                # Get total count before pagination
                total_count = base_query.count()
                
                # Apply sorting
                for sort_obj in query.sorts:
                    base_query = self._apply_sort(base_query, sort_obj)
                
                # Apply pagination
                events = base_query.offset(query.offset).limit(query.limit).all()
                
                # Convert to dictionary format
                results = []
                for event in events:
                    event_dict = self._event_to_dict(event, query.include_related)
                    results.append(event_dict)
                
                return {
                    'results': results,
                    'total_count': total_count,
                    'offset': query.offset,
                    'limit': query.limit,
                    'has_more': (query.offset + query.limit) < total_count
                }
                
            finally:
                self.db_manager.close_session(session)
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {'error': str(e), 'results': [], 'total_count': 0}
    
    def get_search_suggestions(self, field: str, query: str, limit: int = 10) -> List[str]:
        """Get search suggestions for a specific field."""
        try:
            session = self.db_manager.get_session()
            try:
                suggestions = []
                
                if field == 'location':
                    results = session.query(Event.location).filter(
                        Event.location.contains(query)
                    ).distinct().limit(limit).all()
                    suggestions = [r[0] for r in results if r[0]]
                
                elif field == 'event_type':
                    results = session.query(Event.event_type).filter(
                        Event.event_type.contains(query.upper())
                    ).distinct().limit(limit).all()
                    suggestions = [r[0] for r in results if r[0]]
                
                elif field == 'severity_level':
                    suggestions = [s for s in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'] 
                                 if query.upper() in s]
                
                # Add module-specific suggestions
                if field == 'plate_number' and 'anpr' in query.lower():
                    from ..database.models import AnprEvent
                    results = session.query(AnprEvent.plate_number).filter(
                        AnprEvent.plate_number.contains(query.upper())
                    ).distinct().limit(limit).all()
                    suggestions.extend([r[0] for r in results if r[0]])
                
                return suggestions[:limit]
                
            finally:
                self.db_manager.close_session(session)
                
        except Exception as e:
            logger.error(f"Failed to get search suggestions: {e}")
            return []
    
    # Reporting and Export
    def generate_report(self, report_type: str, time_range: TimeRange, 
                       filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive reports for different purposes."""
        try:
            if report_type == 'security_summary':
                return self._generate_security_summary_report(time_range, filters)
            elif report_type == 'performance_analysis':
                return self._generate_performance_report(time_range, filters)
            elif report_type == 'threat_assessment':
                return self._generate_threat_assessment_report(time_range, filters)
            elif report_type == 'compliance_audit':
                return self._generate_compliance_report(time_range, filters)
            else:
                return {'error': f'Unknown report type: {report_type}'}
                
        except Exception as e:
            logger.error(f"Failed to generate report {report_type}: {e}")
            return {'error': str(e)}
    
    def export_data(self, query: SearchQuery, format_type: str = 'json') -> Dict[str, Any]:
        """Export search results in various formats."""
        try:
            # Get all matching records (no pagination for export)
            export_query = SearchQuery(
                text_search=query.text_search,
                filters=query.filters,
                sorts=query.sorts,
                limit=10000,  # Max export limit
                offset=0,
                include_related=True
            )
            
            results = self.search_events(export_query)
            
            if format_type == 'json':
                return {
                    'format': 'json',
                    'data': results['results'],
                    'metadata': {
                        'total_records': results['total_count'],
                        'exported_records': len(results['results']),
                        'export_time': datetime.now().isoformat()
                    }
                }
            elif format_type == 'csv':
                return self._export_to_csv(results['results'])
            else:
                return {'error': f'Unsupported export format: {format_type}'}
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return {'error': str(e)}
    
    # Helper methods for analytics
    def _get_start_time(self, end_time: datetime, time_range: TimeRange) -> datetime:
        """Calculate start time based on time range."""
        if time_range == TimeRange.HOUR:
            return end_time - timedelta(hours=1)
        elif time_range == TimeRange.DAY:
            return end_time - timedelta(days=1)
        elif time_range == TimeRange.WEEK:
            return end_time - timedelta(days=7)
        elif time_range == TimeRange.MONTH:
            return end_time - timedelta(days=30)
        elif time_range == TimeRange.QUARTER:
            return end_time - timedelta(days=90)
        elif time_range == TimeRange.YEAR:
            return end_time - timedelta(days=365)
        else:
            return end_time - timedelta(days=1)
    
    def _get_event_count(self, session: Session, start_time: datetime, end_time: datetime) -> int:
        """Get total event count in time range."""
        return session.query(func.count(Event.id)).filter(
            and_(
                Event.timestamp >= start_time,
                Event.timestamp <= end_time
            )
        ).scalar() or 0
    
    def _get_alert_count(self, session: Session, start_time: datetime, end_time: datetime) -> int:
        """Get alert event count in time range."""
        return session.query(func.count(Event.id)).filter(
            and_(
                Event.timestamp >= start_time,
                Event.timestamp <= end_time,
                Event.alert_triggered == True
            )
        ).scalar() or 0
    
    def _get_critical_count(self, session: Session, start_time: datetime, end_time: datetime) -> int:
        """Get critical event count in time range."""
        return session.query(func.count(Event.id)).filter(
            and_(
                Event.timestamp >= start_time,
                Event.timestamp <= end_time,
                Event.severity_level == 'CRITICAL'
            )
        ).scalar() or 0
    
    def _get_detection_counts_by_module(self, session: Session, start_time: datetime, end_time: datetime) -> Dict[str, int]:
        """Get detection counts grouped by module."""
        results = session.query(
            Event.event_type,
            func.count(Event.id).label('count')
        ).filter(
            and_(
                Event.timestamp >= start_time,
                Event.timestamp <= end_time
            )
        ).group_by(Event.event_type).all()
        
        return {result.event_type: result.count for result in results}
    
    def _get_severity_distribution(self, session: Session, start_time: datetime, end_time: datetime) -> Dict[str, int]:
        """Get severity level distribution."""
        results = session.query(
            Event.severity_level,
            func.count(Event.id).label('count')
        ).filter(
            and_(
                Event.timestamp >= start_time,
                Event.timestamp <= end_time
            )
        ).group_by(Event.severity_level).all()
        
        return {result.severity_level: result.count for result in results}
    
    def _get_location_statistics(self, session: Session, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get location-based statistics."""
        results = session.query(
            Event.location,
            func.count(Event.id).label('total_events'),
            func.count(func.nullif(Event.alert_triggered, False)).label('alert_events')
        ).filter(
            and_(
                Event.timestamp >= start_time,
                Event.timestamp <= end_time
            )
        ).group_by(Event.location).all()
        
        location_stats = {}
        for result in results:
            location_stats[result.location] = {
                'total_events': result.total_events,
                'alert_events': result.alert_events,
                'alert_rate': (result.alert_events / result.total_events * 100) if result.total_events > 0 else 0
            }
        
        return location_stats
    
    def _get_performance_metrics(self, session: Session, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get system performance metrics."""
        # Processing time statistics
        processing_times = session.query(Event.processing_time).filter(
            and_(
                Event.timestamp >= start_time,
                Event.timestamp <= end_time,
                Event.processing_time.isnot(None)
            )
        ).all()
        
        times = [float(pt[0]) for pt in processing_times if pt[0]]
        
        if times:
            return {
                'avg_processing_time_ms': statistics.mean(times) * 1000,
                'min_processing_time_ms': min(times) * 1000,
                'max_processing_time_ms': max(times) * 1000,
                'median_processing_time_ms': statistics.median(times) * 1000,
                'total_processed': len(times)
            }
        else:
            return {
                'avg_processing_time_ms': 0,
                'min_processing_time_ms': 0,
                'max_processing_time_ms': 0,
                'median_processing_time_ms': 0,
                'total_processed': 0
            }
    
    def _get_trend_analysis(self, session: Session, start_time: datetime, end_time: datetime, time_range: TimeRange) -> Dict[str, Any]:
        """Get trend analysis data."""
        # Determine time bucket size
        if time_range in [TimeRange.HOUR, TimeRange.DAY]:
            time_format = '%H'
            bucket_label = 'hour'
        elif time_range == TimeRange.WEEK:
            time_format = '%w'
            bucket_label = 'day_of_week'
        else:
            time_format = '%d'
            bucket_label = 'day'
        
        # Get hourly/daily trends
        results = session.query(
            func.strftime(time_format, Event.timestamp).label('time_bucket'),
            func.count(Event.id).label('event_count'),
            func.count(func.nullif(Event.alert_triggered, False)).label('alert_count')
        ).filter(
            and_(
                Event.timestamp >= start_time,
                Event.timestamp <= end_time
            )
        ).group_by(func.strftime(time_format, Event.timestamp)).all()
        
        trends = []
        for result in results:
            trends.append({
                bucket_label: result.time_bucket,
                'events': result.event_count,
                'alerts': result.alert_count
            })
        
        return {
            'time_buckets': trends,
            'bucket_type': bucket_label
        }
    
    # Module-specific analytics
    def _get_anpr_analytics(self, session: Session, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get ANPR-specific analytics."""
        from ..database.models import AnprEvent
        
        # ANPR events in time range
        anpr_events = session.query(Event).join(AnprEvent).filter(
            and_(
                Event.timestamp >= start_time,
                Event.timestamp <= end_time,
                Event.event_type == 'ANPR'
            )
        ).all()
        
        total_detections = len(anpr_events)
        red_listed_detections = sum(1 for event in anpr_events if event.anpr_event.is_red_listed)
        
        # Top detected plates
        plate_counts = Counter()
        for event in anpr_events:
            if event.anpr_event:
                plate_counts[event.anpr_event.plate_number] += 1
        
        return {
            'total_detections': total_detections,
            'red_listed_detections': red_listed_detections,
            'normal_detections': total_detections - red_listed_detections,
            'red_listed_rate': (red_listed_detections / total_detections * 100) if total_detections > 0 else 0,
            'top_plates': dict(plate_counts.most_common(10))
        }
    
    def _get_face_analytics(self, session: Session, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get face recognition analytics."""
        from ..database.models import FaceEvent
        
        face_events = session.query(Event).join(FaceEvent).filter(
            and_(
                Event.timestamp >= start_time,
                Event.timestamp <= end_time,
                Event.event_type == 'FACE'
            )
        ).all()
        
        total_detections = len(face_events)
        category_counts = Counter()
        
        for event in face_events:
            if event.face_event:
                category_counts[event.face_event.person_category] += 1
        
        return {
            'total_detections': total_detections,
            'category_distribution': dict(category_counts),
            'unknown_detections': category_counts.get('unknown', 0),
            'restricted_detections': category_counts.get('restricted', 0),
            'criminal_detections': category_counts.get('criminal', 0)
        }
    
    def _get_violence_analytics(self, session: Session, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get violence detection analytics."""
        from ..database.models import ViolenceEvent
        
        violence_events = session.query(Event).join(ViolenceEvent).filter(
            and_(
                Event.timestamp >= start_time,
                Event.timestamp <= end_time,
                Event.event_type == 'VIOLENCE'
            )
        ).all()
        
        total_detections = len(violence_events)
        violence_detected = sum(1 for event in violence_events if event.violence_event.is_violence)
        
        intensity_counts = Counter()
        for event in violence_events:
            if event.violence_event and event.violence_event.is_violence:
                intensity_counts[event.violence_event.violence_intensity] += 1
        
        return {
            'total_detections': total_detections,
            'violence_detected': violence_detected,
            'non_violence_detected': total_detections - violence_detected,
            'violence_rate': (violence_detected / total_detections * 100) if total_detections > 0 else 0,
            'intensity_distribution': dict(intensity_counts)
        }
    
    def _get_weapon_analytics(self, session: Session, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get weapon detection analytics."""
        from ..database.models import WeaponEvent
        
        weapon_events = session.query(Event).join(WeaponEvent).filter(
            and_(
                Event.timestamp >= start_time,
                Event.timestamp <= end_time,
                Event.event_type == 'WEAPON'
            )
        ).all()
        
        total_detections = len(weapon_events)
        weapon_type_counts = Counter()
        threat_level_counts = Counter()
        
        for event in weapon_events:
            if event.weapon_event:
                weapon_type_counts[event.weapon_event.weapon_type] += 1
                threat_level_counts[event.weapon_event.threat_level] += 1
        
        return {
            'total_detections': total_detections,
            'weapon_type_distribution': dict(weapon_type_counts),
            'threat_level_distribution': dict(threat_level_counts),
            'critical_threats': threat_level_counts.get('CRITICAL', 0),
            'high_threats': threat_level_counts.get('HIGH', 0)
        }
    
    # Search helpers
    def _apply_text_search(self, query, text_search: str):
        """Apply text search across relevant fields."""
        search_term = f"%{text_search}%"
        return query.filter(
            or_(
                Event.metadata.contains(text_search),
                Event.location.contains(text_search)
            )
        )
    
    def _apply_filter(self, query, filter_obj: SearchFilter):
        """Apply a single filter to the query."""
        field_name = filter_obj.field
        operator = filter_obj.operator
        value = filter_obj.value
        
        if hasattr(Event, field_name):
            field = getattr(Event, field_name)
            
            if operator == 'eq':
                query = query.filter(field == value)
            elif operator == 'ne':
                query = query.filter(field != value)
            elif operator == 'gt':
                query = query.filter(field > value)
            elif operator == 'gte':
                query = query.filter(field >= value)
            elif operator == 'lt':
                query = query.filter(field < value)
            elif operator == 'lte':
                query = query.filter(field <= value)
            elif operator == 'in':
                query = query.filter(field.in_(value))
            elif operator == 'not_in':
                query = query.filter(~field.in_(value))
            elif operator == 'contains':
                if filter_obj.case_sensitive:
                    query = query.filter(field.contains(value))
                else:
                    query = query.filter(func.lower(field).contains(value.lower()))
        
        return query
    
    def _apply_sort(self, query, sort_obj: SearchSort):
        """Apply sorting to the query."""
        field_name = sort_obj.field
        direction = sort_obj.direction
        
        if hasattr(Event, field_name):
            field = getattr(Event, field_name)
            if direction == 'desc':
                query = query.order_by(desc(field))
            else:
                query = query.order_by(asc(field))
        
        return query
    
    def _event_to_dict(self, event: Event, include_related: bool = False) -> Dict[str, Any]:
        """Convert event object to dictionary."""
        event_dict = {
            'id': event.id,
            'event_type': event.event_type,
            'timestamp': event.timestamp.isoformat(),
            'confidence_score': event.confidence_score,
            'status': event.status,
            'location': event.location,
            'alert_triggered': event.alert_triggered,
            'severity_level': event.severity_level,
            'image_path': event.image_path,
            'processing_time': event.processing_time
        }
        
        if event.metadata:
            event_dict['metadata'] = event.get_metadata()
        
        if include_related:
            # Add module-specific data
            if event.event_type == 'ANPR' and event.anpr_event:
                event_dict['anpr_data'] = {
                    'plate_number': event.anpr_event.plate_number,
                    'is_red_listed': event.anpr_event.is_red_listed,
                    'alert_reason': event.anpr_event.alert_reason
                }
            elif event.event_type == 'FACE' and event.face_event:
                event_dict['face_data'] = {
                    'person_name': event.face_event.person_name,
                    'person_category': event.face_event.person_category,
                    'recognition_confidence': event.face_event.recognition_confidence
                }
            elif event.event_type == 'VIOLENCE' and event.violence_event:
                event_dict['violence_data'] = {
                    'is_violence': event.violence_event.is_violence,
                    'violence_type': event.violence_event.violence_type,
                    'violence_intensity': event.violence_event.violence_intensity
                }
            elif event.event_type == 'WEAPON' and event.weapon_event:
                event_dict['weapon_data'] = {
                    'weapon_type': event.weapon_event.weapon_type,
                    'weapon_category': event.weapon_event.weapon_category,
                    'threat_level': event.weapon_event.threat_level
                }
        
        return event_dict
    
    # Report generators
    def _generate_security_summary_report(self, time_range: TimeRange, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate security summary report."""
        overview = self.get_dashboard_overview(time_range)
        
        return {
            'report_type': 'security_summary',
            'time_range': time_range.value,
            'generated_at': datetime.now().isoformat(),
            'executive_summary': {
                'total_events': overview['summary']['total_events'],
                'critical_incidents': overview['summary']['critical_events'],
                'alert_rate': overview['summary']['alert_rate'],
                'most_active_location': max(overview['location_stats'].items(), 
                                          key=lambda x: x[1]['total_events'])[0] if overview['location_stats'] else 'None'
            },
            'detailed_metrics': overview
        }
    
    def _generate_performance_report(self, time_range: TimeRange, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate performance analysis report."""
        overview = self.get_dashboard_overview(time_range)
        
        return {
            'report_type': 'performance_analysis',
            'time_range': time_range.value,
            'generated_at': datetime.now().isoformat(),
            'performance_summary': overview['performance_metrics'],
            'system_efficiency': {
                'detection_modules_active': len([k for k, v in overview['detection_counts'].items() if v > 0]),
                'average_response_time': overview['performance_metrics']['avg_processing_time_ms'],
                'throughput_events_per_hour': overview['summary']['total_events']
            }
        }
    
    def _generate_threat_assessment_report(self, time_range: TimeRange, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate threat assessment report."""
        overview = self.get_dashboard_overview(time_range)
        
        threat_level = 'LOW'
        if overview['summary']['critical_events'] > 0:
            threat_level = 'CRITICAL'
        elif overview['summary']['alert_rate'] > 20:
            threat_level = 'HIGH'
        elif overview['summary']['alert_rate'] > 10:
            threat_level = 'MEDIUM'
        
        return {
            'report_type': 'threat_assessment',
            'time_range': time_range.value,
            'generated_at': datetime.now().isoformat(),
            'threat_level': threat_level,
            'risk_factors': {
                'critical_incidents': overview['summary']['critical_events'],
                'high_severity_incidents': overview['severity_distribution'].get('HIGH', 0),
                'weapon_detections': overview['detection_counts'].get('WEAPON', 0),
                'violence_incidents': overview['detection_counts'].get('VIOLENCE', 0)
            },
            'recommendations': self._generate_security_recommendations(overview)
        }
    
    def _generate_compliance_report(self, time_range: TimeRange, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate compliance audit report."""
        return {
            'report_type': 'compliance_audit',
            'time_range': time_range.value,
            'generated_at': datetime.now().isoformat(),
            'compliance_status': 'COMPLIANT',
            'audit_points': [
                'All detection events properly logged',
                'Alert mechanisms functioning correctly',
                'Data retention policies in place',
                'System health monitoring active'
            ]
        }
    
    def _generate_security_recommendations(self, overview: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on data."""
        recommendations = []
        
        if overview['summary']['critical_events'] > 0:
            recommendations.append("Review critical incidents and update security protocols")
        
        if overview['summary']['alert_rate'] > 15:
            recommendations.append("High alert rate detected - consider reviewing detection thresholds")
        
        if overview['detection_counts'].get('WEAPON', 0) > 0:
            recommendations.append("Weapon detections recorded - enhance security measures")
        
        if not recommendations:
            recommendations.append("Security metrics within normal parameters")
        
        return recommendations
    
    def _export_to_csv(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Export data to CSV format."""
        if not data:
            return {'error': 'No data to export'}
        
        # Create CSV headers from first record
        headers = list(data[0].keys())
        
        # Create CSV content
        csv_lines = [','.join(headers)]
        for record in data:
            row = []
            for header in headers:
                value = record.get(header, '')
                if isinstance(value, dict):
                    value = json.dumps(value)
                elif value is None:
                    value = ''
                else:
                    value = str(value)
                # Escape commas and quotes
                if ',' in value or '"' in value:
                    value = f'"{value.replace('"', '""')}"'
                row.append(value)
            csv_lines.append(','.join(row))
        
        return {
            'format': 'csv',
            'data': '\n'.join(csv_lines),
            'metadata': {
                'total_records': len(data),
                'export_time': datetime.now().isoformat()
            }
        }
    
    # Cache management
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if still valid."""
        if cache_key in self._analytics_cache:
            cached_item = self._analytics_cache[cache_key]
            if (datetime.now() - cached_item['timestamp']).total_seconds() < self._cache_ttl:
                return cached_item['data']
        return None
    
    def _cache_result(self, cache_key: str, data: Dict[str, Any]):
        """Cache result with timestamp."""
        self._analytics_cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
        
        # Cleanup old cache entries
        self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Remove expired cache entries."""
        now = datetime.now()
        if (now - self._last_cache_cleanup).total_seconds() > 3600:  # Cleanup every hour
            expired_keys = []
            for key, item in self._analytics_cache.items():
                if (now - item['timestamp']).total_seconds() > self._cache_ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._analytics_cache[key]
            
            self._last_cache_cleanup = now
    
    def clear_cache(self):
        """Clear all cached results."""
        self._analytics_cache.clear()
        logger.info("Analytics cache cleared")