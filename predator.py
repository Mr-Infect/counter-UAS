import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import json
import warnings
import re
from scipy import stats
warnings.filterwarnings('ignore')

class GPSDataPoint:
    def __init__(self, timestamp: datetime, latitude: float, longitude: float,
                 altitude: float = 0, speed: float = 0, course: float = 0, 
                 hdop: float = None, satellites: int = None, snr: float = None):
        self.timestamp = timestamp
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.speed = speed
        self.course = course
        self.hdop = hdop
        self.satellites = satellites
        self.snr = snr

class EnhancedGPSSpoofingDetector:
    def __init__(self):
        self.data_points = []
        self.raw_data = None
        self.column_mapping = {}
        self.detection_results = []
        
        # Enhanced thresholds for better detection
        self.thresholds = {
            'max_speed': 100,  # m/s (360 km/h)
            'max_acceleration': 15,  # m/sÂ²
            'min_satellites': 4,
            'max_hdop': 10.0,
            'position_jump_threshold': 500,  # meters
            'time_gap_threshold': 60,  # seconds
            'min_snr': 20.0,
            'max_altitude_change': 200,  # meters per second
            'course_change_threshold': 120,  # degrees
            'cluster_distance_threshold': 50,  # meters
            'statistical_z_threshold': 2.5
        }

    def identify_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Intelligently identify GPS-related columns"""
        columns = df.columns.tolist()
        mapping = {}
        
        # Common column name patterns
        patterns = {
            'timestamp': ['time', 'timestamp', 'datetime', 'date', 'utc', 'gps_time'],
            'latitude': ['lat', 'latitude', 'y', 'lat_deg', 'lat_dd'],
            'longitude': ['lon', 'lng', 'long', 'longitude', 'x', 'lon_deg', 'lon_dd'],
            'altitude': ['alt', 'altitude', 'elevation', 'height', 'z'],
            'speed': ['speed', 'velocity', 'vel', 'ground_speed', 'gs'],
            'course': ['course', 'heading', 'track', 'bearing', 'direction'],
            'hdop': ['hdop', 'h_dop', 'horizontal_dilution'],
            'satellites': ['sat', 'satellites', 'sat_count', 'num_sat', 'sv'],
            'snr': ['snr', 'signal', 'c_n0', 'cn0', 'signal_strength']
        }
        
        for field, possible_names in patterns.items():
            for col in columns:
                col_lower = col.lower().strip()
                if any(pattern in col_lower for pattern in possible_names):
                    mapping[field] = col
                    break
        
        return mapping

    def parse_timestamp(self, timestamp_str) -> Optional[datetime]:
        """Enhanced timestamp parsing with multiple format support"""
        if pd.isna(timestamp_str) or timestamp_str is None:
            return None
            
        # Convert to string if not already
        timestamp_str = str(timestamp_str).strip()
        
        # Common timestamp formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%d/%m/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
            '%Y%m%d %H%M%S',
            '%Y-%m-%d',
            '%d-%m-%Y',
            '%m-%d-%Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        # Try pandas parsing as fallback
        try:
            return pd.to_datetime(timestamp_str, errors='coerce')
        except:
            return None

    def load_from_csv(self, filepath: str):
        """Enhanced CSV loading with intelligent column detection"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    print(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                print("Failed to load CSV with any encoding")
                return
                
            self.raw_data = df
            print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
            
            # Identify columns
            self.column_mapping = self.identify_columns(df)
            print(f"Column mapping: {self.column_mapping}")
            
            if 'latitude' not in self.column_mapping or 'longitude' not in self.column_mapping:
                print("ERROR: Could not identify latitude and longitude columns")
                return
            
            # Process data
            valid_points = 0
            for idx, row in df.iterrows():
                try:
                    # Extract timestamp
                    timestamp = None
                    if 'timestamp' in self.column_mapping:
                        timestamp = self.parse_timestamp(row[self.column_mapping['timestamp']])
                    
                    if timestamp is None:
                        timestamp = datetime.now() + timedelta(seconds=idx)  # Fallback
                    
                    # Extract coordinates
                    lat = float(row[self.column_mapping['latitude']])
                    lon = float(row[self.column_mapping['longitude']])
                    
                    # Validate coordinates
                    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                        continue
                    
                    # Extract optional fields
                    altitude = 0
                    if 'altitude' in self.column_mapping:
                        altitude = float(row.get(self.column_mapping['altitude'], 0))
                    
                    speed = 0
                    if 'speed' in self.column_mapping:
                        speed = float(row.get(self.column_mapping['speed'], 0))
                    
                    course = 0
                    if 'course' in self.column_mapping:
                        course = float(row.get(self.column_mapping['course'], 0))
                    
                    hdop = None
                    if 'hdop' in self.column_mapping:
                        hdop_val = row.get(self.column_mapping['hdop'])
                        if hdop_val is not None and not pd.isna(hdop_val):
                            hdop = float(hdop_val)
                    
                    satellites = None
                    if 'satellites' in self.column_mapping:
                        sat_val = row.get(self.column_mapping['satellites'])
                        if sat_val is not None and not pd.isna(sat_val):
                            satellites = int(sat_val)
                    
                    snr = None
                    if 'snr' in self.column_mapping:
                        snr_val = row.get(self.column_mapping['snr'])
                        if snr_val is not None and not pd.isna(snr_val):
                            snr = float(snr_val)
                    
                    # Create data point
                    data_point = GPSDataPoint(
                        timestamp=timestamp,
                        latitude=lat,
                        longitude=lon,
                        altitude=altitude,
                        speed=speed,
                        course=course,
                        hdop=hdop,
                        satellites=satellites,
                        snr=snr
                    )
                    
                    self.data_points.append(data_point)
                    valid_points += 1
                    
                except Exception as e:
                    if idx < 10:  # Only print first 10 errors
                        print(f"Skipping row {idx}: {e}")
                    continue
            
            print(f"Successfully processed {valid_points} valid GPS data points")
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return

    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS coordinates"""
        R = 6371000  # Earth's radius in meters
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c

    def velocity_consistency_check(self) -> List[Dict]:
        """Enhanced velocity and acceleration analysis"""
        anomalies = []
        
        if len(self.data_points) < 2:
            return anomalies
        
        for i in range(1, len(self.data_points)):
            current = self.data_points[i]
            previous = self.data_points[i-1]
            
            time_diff = (current.timestamp - previous.timestamp).total_seconds()
            if time_diff <= 0:
                continue
            
            # Calculate distance and speed
            distance = self.haversine_distance(
                previous.latitude, previous.longitude,
                current.latitude, current.longitude
            )
            
            calculated_speed = distance / time_diff
            
            # Check for excessive speed
            if calculated_speed > self.thresholds['max_speed']:
                anomalies.append({
                    'type': 'excessive_speed',
                    'timestamp': current.timestamp,
                    'calculated_speed': calculated_speed,
                    'reported_speed': current.speed,
                    'threshold': self.thresholds['max_speed'],
                    'severity': 'high',
                    'confidence': min(calculated_speed / self.thresholds['max_speed'], 5.0)
                })
            
            # Check for acceleration anomalies
            if i > 1:
                prev_distance = self.haversine_distance(
                    self.data_points[i-2].latitude, self.data_points[i-2].longitude,
                    previous.latitude, previous.longitude
                )
                prev_time_diff = (previous.timestamp - self.data_points[i-2].timestamp).total_seconds()
                
                if prev_time_diff > 0:
                    prev_speed = prev_distance / prev_time_diff
                    acceleration = abs(calculated_speed - prev_speed) / time_diff
                    
                    if acceleration > self.thresholds['max_acceleration']:
                        anomalies.append({
                            'type': 'excessive_acceleration',
                            'timestamp': current.timestamp,
                            'acceleration': acceleration,
                            'threshold': self.thresholds['max_acceleration'],
                            'severity': 'high',
                            'confidence': min(acceleration / self.thresholds['max_acceleration'], 5.0)
                        })
        
        return anomalies

    def position_jump_detection(self) -> List[Dict]:
        """Enhanced position jump detection"""
        anomalies = []
        
        for i in range(1, len(self.data_points)):
            current = self.data_points[i]
            previous = self.data_points[i-1]
            
            distance = self.haversine_distance(
                previous.latitude, previous.longitude,
                current.latitude, current.longitude
            )
            
            time_diff = (current.timestamp - previous.timestamp).total_seconds()
            
            # Check for instantaneous jumps
            if (distance > self.thresholds['position_jump_threshold'] and 
                time_diff < self.thresholds['time_gap_threshold']):
                
                anomalies.append({
                    'type': 'position_jump',
                    'timestamp': current.timestamp,
                    'distance': distance,
                    'time_diff': time_diff,
                    'threshold': self.thresholds['position_jump_threshold'],
                    'severity': 'high',
                    'confidence': min(distance / self.thresholds['position_jump_threshold'], 5.0)
                })
        
        return anomalies

    def signal_quality_analysis(self) -> List[Dict]:
        """Enhanced signal quality analysis"""
        anomalies = []
        
        for point in self.data_points:
            # Low satellite count
            if point.satellites is not None and point.satellites < self.thresholds['min_satellites']:
                anomalies.append({
                    'type': 'low_satellite_count',
                    'timestamp': point.timestamp,
                    'satellites': point.satellites,
                    'threshold': self.thresholds['min_satellites'],
                    'severity': 'medium',
                    'confidence': (self.thresholds['min_satellites'] - point.satellites) / self.thresholds['min_satellites']
                })
            
            # High HDOP
            if point.hdop is not None and point.hdop > self.thresholds['max_hdop']:
                anomalies.append({
                    'type': 'high_hdop',
                    'timestamp': point.timestamp,
                    'hdop': point.hdop,
                    'threshold': self.thresholds['max_hdop'],
                    'severity': 'medium',
                    'confidence': min(point.hdop / self.thresholds['max_hdop'], 3.0)
                })
            
            # Low SNR
            if point.snr is not None and point.snr < self.thresholds['min_snr']:
                anomalies.append({
                    'type': 'low_snr',
                    'timestamp': point.timestamp,
                    'snr': point.snr,
                    'threshold': self.thresholds['min_snr'],
                    'severity': 'medium',
                    'confidence': (self.thresholds['min_snr'] - point.snr) / self.thresholds['min_snr']
                })
        
        return anomalies

    def temporal_consistency_check(self) -> List[Dict]:
        """Check for temporal anomalies"""
        anomalies = []
        
        if len(self.data_points) < 3:
            return anomalies
        
        timestamps = [p.timestamp for p in self.data_points]
        time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                      for i in range(1, len(timestamps))]
        
        if len(time_diffs) < 2:
            return anomalies
        
        # Check for unusual time gaps
        median_diff = np.median(time_diffs)
        
        for i, diff in enumerate(time_diffs):
            if diff > median_diff * 10 or diff < 0:
                anomalies.append({
                    'type': 'temporal_anomaly',
                    'timestamp': timestamps[i+1],
                    'time_gap': diff,
                    'expected_gap': median_diff,
                    'severity': 'medium',
                    'confidence': min(abs(diff - median_diff) / median_diff, 5.0)
                })
        
        return anomalies

    def clustering_analysis(self) -> List[Dict]:
        """Detect suspicious clustering patterns"""
        anomalies = []
        
        if len(self.data_points) < 10:
            return anomalies
        
        # Check for repeated identical positions
        position_counts = {}
        for point in self.data_points:
            pos_key = (round(point.latitude, 6), round(point.longitude, 6))
            position_counts[pos_key] = position_counts.get(pos_key, 0) + 1
        
        # Flag positions that appear too frequently
        for pos, count in position_counts.items():
            if count > len(self.data_points) * 0.1:  # More than 10% of points
                anomalies.append({
                    'type': 'position_clustering',
                    'position': pos,
                    'count': count,
                    'percentage': (count / len(self.data_points)) * 100,
                    'severity': 'medium',
                    'confidence': min(count / (len(self.data_points) * 0.1), 3.0)
                })
        
        return anomalies

    def statistical_anomaly_detection(self) -> List[Dict]:
        """Enhanced statistical analysis"""
        anomalies = []
        
        if len(self.data_points) < 10:
            return anomalies
        
        # Speed analysis
        speeds = [p.speed for p in self.data_points if p.speed is not None and p.speed > 0]
        if len(speeds) > 5:
            z_scores = np.abs(stats.zscore(speeds))
            for i, z in enumerate(z_scores):
                if z > self.thresholds['statistical_z_threshold']:
                    anomalies.append({
                        'type': 'speed_statistical_anomaly',
                        'timestamp': self.data_points[i].timestamp,
                        'speed': speeds[i],
                        'z_score': z,
                        'severity': 'low',
                        'confidence': min(z / self.thresholds['statistical_z_threshold'], 3.0)
                    })
        
        # Altitude analysis
        altitudes = [p.altitude for p in self.data_points if p.altitude is not None]
        if len(altitudes) > 5:
            altitude_changes = [abs(altitudes[i] - altitudes[i-1]) 
                              for i in range(1, len(altitudes))]
            if altitude_changes:
                mean_change = np.mean(altitude_changes)
                std_change = np.std(altitude_changes)
                
                for i, change in enumerate(altitude_changes):
                    if change > mean_change + 3 * std_change:
                        anomalies.append({
                            'type': 'altitude_anomaly',
                            'timestamp': self.data_points[i+1].timestamp,
                            'altitude_change': change,
                            'mean_change': mean_change,
                            'severity': 'low',
                            'confidence': min(change / (mean_change + 3 * std_change), 3.0)
                        })
        
        return anomalies

    def detect_spoofing(self) -> Dict:
        """Comprehensive spoofing detection"""
        if not self.data_points:
            return {'error': 'No data points available for analysis'}
        
        print("Running spoofing detection analysis...")
        
        # Run all detection methods
        velocity_anomalies = self.velocity_consistency_check()
        position_anomalies = self.position_jump_detection()
        signal_anomalies = self.signal_quality_analysis()
        temporal_anomalies = self.temporal_consistency_check()
        clustering_anomalies = self.clustering_analysis()
        statistical_anomalies = self.statistical_anomaly_detection()
        
        # Combine all anomalies
        all_anomalies = (velocity_anomalies + position_anomalies + 
                        signal_anomalies + temporal_anomalies + 
                        clustering_anomalies + statistical_anomalies)
        
        # Count by severity
        high_severity = sum(1 for a in all_anomalies if a['severity'] == 'high')
        medium_severity = sum(1 for a in all_anomalies if a['severity'] == 'medium')
        low_severity = sum(1 for a in all_anomalies if a['severity'] == 'low')
        
        # Enhanced risk calculation
        confidence_weighted_score = sum(a.get('confidence', 1.0) for a in all_anomalies)
        base_risk = (high_severity * 10 + medium_severity * 5 + low_severity * 2)
        risk_score = min((base_risk + confidence_weighted_score) / len(self.data_points) * 100, 100)
        
        # Determine spoofing likelihood
        if risk_score > 70:
            likelihood = 'VERY HIGH'
        elif risk_score > 40:
            likelihood = 'HIGH'
        elif risk_score > 15:
            likelihood = 'MEDIUM'
        elif risk_score > 5:
            likelihood = 'LOW'
        else:
            likelihood = 'MINIMAL'
        
        return {
            'total_data_points': len(self.data_points),
            'total_anomalies': len(all_anomalies),
            'risk_score': risk_score,
            'spoofing_likelihood': likelihood,
            'anomalies_by_severity': {
                'high': high_severity,
                'medium': medium_severity,
                'low': low_severity
            },
            'anomalies_by_type': {
                'velocity': len(velocity_anomalies),
                'position': len(position_anomalies),
                'signal': len(signal_anomalies),
                'temporal': len(temporal_anomalies),
                'clustering': len(clustering_anomalies),
                'statistical': len(statistical_anomalies)
            },
            'detailed_anomalies': all_anomalies,
            'column_mapping': self.column_mapping
        }

    def generate_report(self, output_file: str = None) -> str:
        """Generate comprehensive detection report"""
        results = self.detect_spoofing()
        
        if 'error' in results:
            error_msg = f"Error: {results['error']}"
            print(error_msg)
            return error_msg
        
        report = f"""
GPS SPOOFING DETECTION REPORT
============================

Data Summary:
- Total GPS data points analyzed: {results['total_data_points']}
- Column mapping used: {results['column_mapping']}

Detection Results:
- Total anomalies detected: {results['total_anomalies']}
- Risk Score: {results['risk_score']:.2f}%
- Spoofing Likelihood: {results['spoofing_likelihood']}

Anomalies by Severity:
- High severity: {results['anomalies_by_severity']['high']}
- Medium severity: {results['anomalies_by_severity']['medium']}
- Low severity: {results['anomalies_by_severity']['low']}

Anomalies by Category:
- Velocity anomalies: {results['anomalies_by_type']['velocity']}
- Position anomalies: {results['anomalies_by_type']['position']}
- Signal quality anomalies: {results['anomalies_by_type']['signal']}
- Temporal anomalies: {results['anomalies_by_type']['temporal']}
- Clustering anomalies: {results['anomalies_by_type']['clustering']}
- Statistical anomalies: {results['anomalies_by_type']['statistical']}

Detailed Anomalies:
"""
        
        # Group anomalies by type for better reporting
        anomaly_groups = {}
        for anomaly in results['detailed_anomalies']:
            atype = anomaly['type']
            if atype not in anomaly_groups:
                anomaly_groups[atype] = []
            anomaly_groups[atype].append(anomaly)
        
        for atype, anomalies in anomaly_groups.items():
            report += f"\n{atype.upper().replace('_', ' ')} ({len(anomalies)} instances):\n"
            report += "-" * 50 + "\n"
            
            for i, anomaly in enumerate(anomalies[:5]):  # Show first 5 of each type
                #report += f"{i+1}. Timestamp: {anomaly['timestamp']}\n"
                report += f"   Severity: {anomaly['severity'].upper()}\n"
                report += f"   Confidence: {anomaly.get('confidence', 'N/A'):.2f}\n"
                
                for key, value in anomaly.items():
                    if key not in ['type', 'severity', 'timestamp', 'confidence']:
                        report += f"   {key}: {value}\n"
                report += "\n"
            
            if len(anomalies) > 5:
                report += f"   ... and {len(anomalies) - 5} more instances\n\n"
        
        # Add recommendations
        report += "\nRECOMMENDATIONS:\n"
        report += "=" * 50 + "\n"
        
        if results['risk_score'] > 70:
            report += "âš ï¸  CRITICAL: Very high probability of GPS spoofing detected!\n"
            report += "   - Immediately verify GPS source and integrity\n"
            report += "   - Consider using alternative navigation methods\n"
            report += "   - Check for interference or jamming devices\n"
        elif results['risk_score'] > 40:
            report += "âš ï¸  WARNING: High probability of GPS spoofing detected!\n"
            report += "   - Investigate anomalous readings\n"
            report += "   - Cross-reference with other sensors\n"
            report += "   - Monitor for continued anomalies\n"
        elif results['risk_score'] > 15:
            report += "âš ï¸  CAUTION: Moderate anomalies detected\n"
            report += "   - Monitor GPS performance\n"
            report += "   - Check for environmental interference\n"
        else:
            report += "âœ… GPS data appears normal\n"
            report += "   - Continue regular monitoring\n"
        
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report)
                print(f"Report saved to {output_file}")
            except Exception as e:
                print(f"Error saving report: {e}")
        else:
            print(report)
        
        return report

    def visualize_data(self):
        """Enhanced data visualization"""
        if not self.data_points:
            print("No data to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Extract data for plotting
        times = [p.timestamp for p in self.data_points]
        lats = [p.latitude for p in self.data_points]
        lons = [p.longitude for p in self.data_points]
        speeds = [p.speed for p in self.data_points if p.speed is not None]
        altitudes = [p.altitude for p in self.data_points if p.altitude is not None]
        
        # 1. GPS Track
        axes[0, 0].plot(lons, lats, 'b-', alpha=0.7, linewidth=2)
        axes[0, 0].scatter(lons[0], lats[0], color='green', s=100, label='Start', zorder=5)
        axes[0, 0].scatter(lons[-1], lats[-1], color='red', s=100, label='End', zorder=5)
        axes[0, 0].set_title('GPS Track')
        axes[0, 0].set_xlabel('Longitude')
        axes[0, 0].set_ylabel('Latitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Speed over time
        if speeds:
            speed_times = [p.timestamp for p in self.data_points if p.speed is not None]
            axes[0, 1].plot(speed_times, speeds, 'g-', linewidth=2)
            axes[0, 1].axhline(y=self.thresholds['max_speed'], color='r', linestyle='--', 
                              label=f'Threshold ({self.thresholds["max_speed"]} m/s)')
            axes[0, 1].set_title('Speed Over Time')
            axes[0, 1].set_ylabel('Speed (m/s)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Altitude profile
        if altitudes:
            axes[0, 2].plot(times, altitudes, 'purple', linewidth=2)
            axes[0, 2].set_title('Altitude Profile')
            axes[0, 2].set_ylabel('Altitude (m)')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Signal quality metrics
        satellites = [p.satellites for p in self.data_points if p.satellites is not None]
        hdops = [p.hdop for p in self.data_points if p.hdop is not None]
        
        if satellites:
            sat_times = [p.timestamp for p in self.data_points if p.satellites is not None]
            axes[1, 0].plot(sat_times, satellites, 'orange', linewidth=2, label='Satellites')
            axes[1, 0].axhline(y=self.thresholds['min_satellites'], color='r', linestyle='--', 
                              label=f'Min threshold ({self.thresholds["min_satellites"]})')
            axes[1, 0].set_title('Satellite Count')
            axes[1, 0].set_ylabel('Number of Satellites')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. HDOP values
        if hdops:
            hdop_times = [p.timestamp for p in self.data_points if p.hdop is not None]
            axes[1, 1].plot(hdop_times, hdops, 'brown', linewidth=2, label='HDOP')
            axes[1, 1].axhline(y=self.thresholds['max_hdop'], color='r', linestyle='--', 
                              label=f'Max threshold ({self.thresholds["max_hdop"]})')
            axes[1, 1].set_title('HDOP Values')
            axes[1, 1].set_ylabel('HDOP')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Speed distribution
        if speeds:
            axes[1, 2].hist(speeds, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 2].axvline(x=self.thresholds['max_speed'], color='r', linestyle='--', 
                              label=f'Speed threshold ({self.thresholds["max_speed"]} m/s)')
            axes[1, 2].set_title('Speed Distribution')
            axes[1, 2].set_xlabel('Speed (m/s)')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def export_anomalies_to_csv(self, filepath: str):
        """Export detected anomalies to CSV file"""
        results = self.detect_spoofing()
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        if not results['detailed_anomalies']:
            print("No anomalies to export")
            return
        
        # Convert anomalies to DataFrame
        anomaly_data = []
        for anomaly in results['detailed_anomalies']:
            row = {
                #'timestamp': anomaly['timestamp'],
                'type': anomaly['type'],
                'severity': anomaly['severity'],
                'confidence': anomaly.get('confidence', 'N/A')
            }
            
            # Add type-specific fields
            for key, value in anomaly.items():
                if key not in ['timestamp', 'type', 'severity', 'confidence']:
                    row[key] = value
            
            anomaly_data.append(row)
        
        df = pd.DataFrame(anomaly_data)
        df.to_csv(filepath, index=False)
        print(f"Anomalies exported to {filepath}")

def generate_sample_spoofed_data():
    """Generate sample data with spoofing scenarios"""
    detector = EnhancedGPSSpoofingDetector()
    base_time = datetime.now()
    base_lat, base_lon = 40.7128, -74.0060  # New York City
    
    print("Generating sample data with spoofing scenarios...")
    
    # Normal GPS data
    for i in range(30):
        lat = base_lat + (i * 0.0001) + np.random.normal(0, 0.00001)
        lon = base_lon + (i * 0.0001) + np.random.normal(0, 0.00001)
        speed = 5 + np.random.normal(0, 1)
        
        data_point = GPSDataPoint(
            timestamp=base_time + timedelta(seconds=i * 10),
            latitude=lat,
            longitude=lon,
            altitude=100 + np.random.normal(0, 5),
            speed=max(0, speed),
            course=45 + np.random.normal(0, 5),
            hdop=1.2 + np.random.normal(0, 0.2),
            satellites=8 + int(np.random.normal(0, 1)),
            snr=35 + np.random.normal(0, 3)
        )
        detector.add_data_point(data_point)
    
    # Add spoofing scenarios
    spoofing_scenarios = [
        # Scenario 1: Sudden position jump
        GPSDataPoint(
            timestamp=base_time + timedelta(seconds=300),
            latitude=base_lat + 0.01,  # ~1km jump
            longitude=base_lon + 0.01,
            altitude=100,
            speed=5,
            course=45,
            hdop=1.2,
            satellites=8,
            snr=35
        ),
        
        # Scenario 2: Excessive speed
        GPSDataPoint(
            timestamp=base_time + timedelta(seconds=310),
            latitude=base_lat + 0.02,
            longitude=base_lon + 0.02,
            altitude=100,
            speed=200,  # Impossible speed
            course=45,
            hdop=1.2,
            satellites=8,
            snr=35
        ),
        
        # Scenario 3: Poor signal quality
        GPSDataPoint(
            timestamp=base_time + timedelta(seconds=320),
            latitude=base_lat + 0.021,
            longitude=base_lon + 0.021,
            altitude=100,
            speed=5,
            course=45,
            hdop=15.0,  # Very poor HDOP
            satellites=2,  # Too few satellites
            snr=15  # Poor signal strength
        ),
        
        # Scenario 4: Repeated identical positions (clustering)
        *[GPSDataPoint(
            timestamp=base_time + timedelta(seconds=330 + i * 10),
            latitude=base_lat + 0.022,  # Same position
            longitude=base_lon + 0.022,  # Same position
            altitude=100,
            speed=0,
            course=45,
            hdop=1.2,
            satellites=8,
            snr=35
        ) for i in range(5)]
    ]
    
    for scenario in spoofing_scenarios:
        detector.add_data_point(scenario)
    
    print(f"Generated {len(detector.data_points)} data points including spoofing scenarios")
    return detector

def main():
    """Main function to run the GPS spoofing detection system"""
    print("=== Enhanced GPS Spoofing Detection System ===")
    print("This system analyzes GPS data for potential spoofing attacks.")
    print()
    
    detector = EnhancedGPSSpoofingDetector()
    
    # Ask user for input
    while True:
        choice = input("Choose an option:\n1. Load CSV file\n2. Use sample data with spoofing\n3. Exit\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            filepath = input("Enter the path to your GPS data CSV file: ").strip()
            if filepath:
                detector.load_from_csv(filepath)
                if detector.data_points:
                    break
                else:
                    print("Failed to load valid GPS data. Please try again.")
            else:
                print("Please provide a valid file path.")
        
        elif choice == '2':
            detector = generate_sample_spoofed_data()
            break
        
        elif choice == '3':
            print("Exiting...")
            return
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    if not detector.data_points:
        print("No GPS data available for analysis.")
        return
    
    print(f"\nAnalyzing {len(detector.data_points)} GPS data points...")
    
    # Run detection
    results = detector.detect_spoofing()
    
    # Generate report
    print("\n" + "="*50)
    detector.generate_report()
    
    # Ask if user wants to save results
    save_choice = input("\nSave results? (y/n): ").strip().lower()
    if save_choice == 'y':
        report_file = input("Enter filename for report (default: gps_spoofing_report.txt): ").strip()
        if not report_file:
            report_file = "gps_spoofing_report.txt"
        
        detector.generate_report(report_file)
        
        # Ask about exporting anomalies
        export_choice = input("Export anomalies to CSV? (y/n): ").strip().lower()
        if export_choice == 'y':
            anomaly_file = input("Enter filename for anomalies (default: gps_anomalies.csv): ").strip()
            if not anomaly_file:
                anomaly_file = "gps_anomalies.csv"
            
            detector.export_anomalies_to_csv(anomaly_file)
    
    # Ask about visualization
    viz_choice = input("Show visualizations? (y/n): ").strip().lower()
    if viz_choice == 'y':
        detector.visualize_data()
    
    print("\nAnalysis complete!")
    
    # Print summary
    if 'spoofing_likelihood' in results:
        print(f"\nSUMMARY:")
        print(f"- Spoofing Likelihood: {results['spoofing_likelihood']}")
        print(f"- Risk Score: {results['risk_score']:.2f}%")
        print(f"- Total Anomalies: {results['total_anomalies']}")
        
        if results['spoofing_likelihood'] in ['HIGH', 'VERY HIGH']:
            print("\nâš ï¸  WARNING: High probability of GPS spoofing detected!")
            print("   Consider investigating the GPS source and data integrity.")
    
if __name__ == "__main__":
    main()
