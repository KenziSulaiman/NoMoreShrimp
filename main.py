import cv2
import mediapipe as mp
import time
import math
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import logging
from datetime import datetime
from pathlib import Path
from collections import deque
import argparse

# ================= LOGGING SETUP =================
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('posture_monitor.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ================= ENHANCED CONFIG =================
@dataclass
class Config:
    """Enhanced configuration with validation and defaults."""

    # Display settings
    WINDOW_WIDTH: int = 1280
    WINDOW_HEIGHT: int = 720
    FPS_TARGET: int = 30

    # Pose detection
    DETECTION_CONFIDENCE: float = 0.8
    TRACKING_CONFIDENCE: float = 0.7
    MODEL_COMPLEXITY: int = 1

    # Posture analysis - More lenient for side movements
    SMOOTHING_ALPHA: float = 0.4
    STABILITY_THRESHOLD: float = 15.0
    POSE_STABILITY_FRAMES: int = 3

    # Adaptive thresholds for different viewing angles
    SIDE_VIEW_THRESHOLD: float = 0.3  # When to switch to side view analysis
    FRONT_VIEW_WEIGHT: float = 1.0
    SIDE_VIEW_WEIGHT: float = 0.7

    # Stricter thresholds for harsh posture monitoring
    GOOD_POSTURE_SCORE_THRESHOLD: float = 85.0  # Much higher requirement for "good"
    POOR_POSTURE_SCORE_THRESHOLD: float = 65.0  # Higher threshold for "poor"
    SLOUCH_ALERT_THRESHOLD: float = 8.0  # Much more sensitive to slouching
    FORWARD_HEAD_THRESHOLD: float = 12.0  # More sensitive to forward head
    SHOULDER_ALIGNMENT_THRESHOLD: float = 6.0  # More sensitive to shoulder misalignment
    NECK_FORWARD_THRESHOLD: float = 10.0  # New stricter neck threshold
    HEAD_TILT_THRESHOLD: float = 8.0  # New head tilt threshold

    # UI settings with expanded text area
    HUD_MARGIN: int = 15
    HUD_PANEL_WIDTH: int = 520
    HUD_PANEL_HEIGHT: int = 380
    TEXT_SCALE: float = 0.6
    TEXT_THICKNESS: int = 1

    # Mini pose guide
    GUIDE_SIZE: int = 200
    GUIDE_MARGIN: int = 20

    # Session settings
    POSTURE_CHECK_INTERVAL: int = 60
    SAVE_SESSION_JSON: bool = True
    SESSION_DIR: str = "sessions"
    MAX_HISTORY_LENGTH: int = 200

    # Notification settings - More frequent alerts for strict monitoring
    ALERT_COOLDOWN_SECONDS: int = 120  # Reduced from 180 for stricter monitoring
    ENABLE_VISUAL_ALERTS: bool = True
    ALERT_FLASH_DURATION: float = 3.0  # Longer flash duration for visibility

    # Performance settings
    LANDMARK_HISTORY_SIZE: int = 15
    FPS_SMOOTHING_FACTOR: float = 0.85

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._create_directories()

    def _validate_config(self):
        """Validate configuration parameters."""
        if not (0.1 <= self.DETECTION_CONFIDENCE <= 1.0):
            raise ValueError("DETECTION_CONFIDENCE must be between 0.1 and 1.0")
        if not (0.1 <= self.TRACKING_CONFIDENCE <= 1.0):
            raise ValueError("TRACKING_CONFIDENCE must be between 0.1 and 1.0")
        if not (0 <= self.MODEL_COMPLEXITY <= 2):
            raise ValueError("MODEL_COMPLEXITY must be 0, 1, or 2")

    def _create_directories(self):
        """Create necessary directories."""
        if self.SAVE_SESSION_JSON:
            Path(self.SESSION_DIR).mkdir(exist_ok=True)


class AppMode(Enum):
    """Application mode - simplified to posture monitoring only."""
    POSTURE_MONITOR = "posture_monitor"


class PostureState(Enum):
    """Posture quality states."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very_poor"


class AlertType(Enum):
    """Types of posture alerts."""
    FORWARD_HEAD = "forward_head"
    SLOUCHING = "slouching"
    UNEVEN_SHOULDERS = "uneven_shoulders"
    POOR_OVERALL = "poor_overall"


class ViewAngle(Enum):
    """Viewing angles for adaptive analysis."""
    FRONT = "front"
    SIDE_LEFT = "side_left"
    SIDE_RIGHT = "side_right"
    BACK = "back"
    UNKNOWN = "unknown"


# ================= ENHANCED POSE DETECTOR =================
class PoseDetector:
    """Enhanced pose detection with better error handling and view angle detection."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=config.MODEL_COMPLEXITY,
                smooth_landmarks=True,
                min_detection_confidence=config.DETECTION_CONFIDENCE,
                min_tracking_confidence=config.TRACKING_CONFIDENCE
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
        except Exception as e:
            self.logger.error(f"Failed to initialize MediaPipe Pose: {e}")
            raise

        self.results = None
        self.landmark_history = deque(maxlen=config.LANDMARK_HISTORY_SIZE)
        self.last_detection_time = time.time()
        self.detection_fps = 0
        self.current_view_angle = ViewAngle.FRONT
        self.view_angle_history = deque(maxlen=10)

    def process(self, frame_bgr: np.ndarray) -> Tuple[List[Tuple[int, int, int]], bool]:
        """Process frame and return landmarks with detection success status."""
        if frame_bgr is None or frame_bgr.size == 0:
            return [], False

        start_time = time.time()
        h, w, _ = frame_bgr.shape

        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            self.results = self.pose.process(rgb)
            rgb.flags.writeable = True
        except Exception as e:
            self.logger.warning(f"Pose processing failed: {e}")
            return [], False

        landmarks = []
        detection_success = False

        if self.results.pose_landmarks:
            detection_success = True
            current_frame_landmarks = {}

            for i, lm in enumerate(self.results.pose_landmarks.landmark):
                if lm.visibility > 0.5:  # Only use visible landmarks
                    x, y = int(lm.x * w), int(lm.y * h)
                    if 0 <= x < w and 0 <= y < h:
                        landmarks.append((i, x, y))
                        current_frame_landmarks[i] = (x, y, lm.visibility)

            if current_frame_landmarks:
                self.landmark_history.append(current_frame_landmarks)
                self._update_view_angle(current_frame_landmarks)

        # Update detection FPS
        detection_time = time.time() - start_time
        if detection_time > 0:
            current_fps = 1.0 / detection_time
            self.detection_fps = (0.8 * self.detection_fps + 0.2 * current_fps
                                  if self.detection_fps > 0 else current_fps)

        return landmarks, detection_success

    def _update_view_angle(self, landmarks: Dict[int, Tuple[int, int, float]]):
        """Determine current viewing angle based on landmark positions."""
        try:
            # Get key landmarks for view angle detection
            nose = landmarks.get(0)
            left_eye = landmarks.get(2)
            right_eye = landmarks.get(5)
            left_shoulder = landmarks.get(11)
            right_shoulder = landmarks.get(12)

            if not all([nose, left_eye, right_eye, left_shoulder, right_shoulder]):
                return

            # Calculate face orientation
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            nose_offset = nose[0] - eye_center_x
            face_width = abs(left_eye[0] - right_eye[0])

            # Calculate shoulder visibility ratio
            left_vis = landmarks[11][2] if 11 in landmarks else 0
            right_vis = landmarks[12][2] if 12 in landmarks else 0
            shoulder_ratio = (left_vis - right_vis) if face_width > 0 else 0

            # Determine view angle
            if abs(nose_offset) < face_width * 0.1 and abs(shoulder_ratio) < 0.3:
                view_angle = ViewAngle.FRONT
            elif nose_offset > face_width * 0.15 or shoulder_ratio > 0.3:
                view_angle = ViewAngle.SIDE_RIGHT
            elif nose_offset < -face_width * 0.15 or shoulder_ratio < -0.3:
                view_angle = ViewAngle.SIDE_LEFT
            else:
                view_angle = ViewAngle.FRONT

            self.view_angle_history.append(view_angle)

            # Smooth view angle detection
            if len(self.view_angle_history) >= 5:
                most_common = max(set(self.view_angle_history),
                                  key=list(self.view_angle_history).count)
                self.current_view_angle = most_common

        except Exception as e:
            self.logger.debug(f"View angle detection failed: {e}")

    def get_view_angle(self) -> ViewAngle:
        """Get current view angle."""
        return self.current_view_angle

    def is_pose_stable(self, threshold: float = None) -> bool:
        """Check if pose is stable based on landmark variance."""
        if threshold is None:
            threshold = self.config.STABILITY_THRESHOLD

        if len(self.landmark_history) < self.config.POSE_STABILITY_FRAMES:
            return False

        # Key points for stability analysis
        key_points = [0, 11, 12, 23, 24]

        for point_id in key_points:
            positions = []
            for frame_landmarks in self.landmark_history:
                if point_id in frame_landmarks:
                    x, y, _ = frame_landmarks[point_id]
                    positions.append((x, y))

            if len(positions) < 2:
                continue

            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]

            x_var = np.var(x_coords) if len(x_coords) > 1 else 0
            y_var = np.var(y_coords) if len(y_coords) > 1 else 0

            if math.sqrt(x_var + y_var) > threshold:
                return False

        return True

    @staticmethod
    def angle_between_points(a: Tuple[int, int], b: Tuple[int, int],
                             c: Tuple[int, int]) -> Optional[float]:
        """Calculate angle between three points."""
        try:
            (ax, ay), (bx, by), (cx, cy) = a, b, c
            v1 = np.array([ax - bx, ay - by])
            v2 = np.array([cx - bx, cy - by])

            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)

            if norms < 1e-6:
                return None

            cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
            angle = np.arccos(cos_angle)

            return math.degrees(angle)
        except (ValueError, ZeroDivisionError):
            return None

    @staticmethod
    def distance_between_points(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ================= ADAPTIVE POSTURE ANALYZER =================
class PostureAnalyzer:
    """Enhanced posture analysis with view-angle adaptation."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

        self.posture_session_data = {
            'start_time': time.time(),
            'good_posture_time': 0,
            'poor_posture_time': 0,
            'alerts_triggered': 0,
            'posture_scores': deque(maxlen=config.MAX_HISTORY_LENGTH),
            'alert_history': deque(maxlen=50),
            'view_angle_distribution': {},
            'exercise_sessions': []
        }

        self.last_alert_time = 0
        self.baseline_measurements = None
        self.calibrated = False
        self.smoothed_angles = {}
        self.alert_callbacks: List[Callable] = []

        # Adaptive weights for different view angles
        self.view_weights = {
            ViewAngle.FRONT: 1.0,
            ViewAngle.SIDE_LEFT: 0.8,
            ViewAngle.SIDE_RIGHT: 0.8,
            ViewAngle.BACK: 0.3,
            ViewAngle.UNKNOWN: 0.5
        }

    def add_alert_callback(self, callback: Callable[[AlertType, str], None]):
        """Add callback for posture alerts."""
        self.alert_callbacks.append(callback)

    def analyze_posture(self, landmark_points: Dict[int, Tuple[int, int]],
                        pose_stable: bool = True, view_angle: ViewAngle = ViewAngle.FRONT) -> Dict:
        """Adaptive posture analysis based on viewing angle."""

        if not landmark_points or len(landmark_points) < 8:
            return self._create_empty_analysis()

        # Update view angle distribution
        angle_name = view_angle.value
        self.posture_session_data['view_angle_distribution'][angle_name] = \
            self.posture_session_data['view_angle_distribution'].get(angle_name, 0) + 1

        # Get view-specific weight
        view_weight = self.view_weights.get(view_angle, 0.5)

        issues = []
        recommendations = []
        angles = {}
        measurements = {}
        scores = []
        alerts = []

        # Extract key landmarks
        landmarks = self._extract_key_landmarks(landmark_points)

        # Analyze based on view angle
        if view_angle == ViewAngle.FRONT:
            analyses = [
                self._analyze_head_posture_front(landmarks, angles, measurements),
                self._analyze_shoulder_alignment(landmarks, angles, measurements),
                self._analyze_spine_alignment_front(landmarks, angles, measurements)
            ]
        elif view_angle in [ViewAngle.SIDE_LEFT, ViewAngle.SIDE_RIGHT]:
            analyses = [
                self._analyze_head_posture_side(landmarks, angles, measurements, view_angle),
                self._analyze_spine_alignment_side(landmarks, angles, measurements, view_angle),
                self._analyze_shoulder_posture_side(landmarks, angles, measurements, view_angle)
            ]
        else:
            # For back/unknown views, use simplified analysis
            analyses = [
                self._analyze_basic_alignment(landmarks, angles, measurements)
            ]

        # Combine analysis results
        for analysis in analyses:
            if analysis:
                scores.append(analysis['score'] * view_weight)
                issues.extend(analysis['issues'])
                recommendations.extend(analysis['recommendations'])
                alerts.extend(analysis['alerts'])

        # Calculate overall score
        overall_score = sum(scores) / len(scores) if scores else 0

        # Adjust score based on pose stability and view angle
        if not pose_stable:
            overall_score *= 0.9

        if view_angle == ViewAngle.UNKNOWN:
            overall_score *= 0.7

        # Determine posture state
        state = self._determine_posture_state(overall_score)

        # Update session data
        self._update_session_data(overall_score, state, alerts)

        # Trigger alerts if necessary
        for alert_type, message in alerts:
            self._trigger_alert(alert_type, message)

        return {
            'overall_score': overall_score,
            'state': state,
            'issues': issues,
            'recommendations': recommendations,
            'angles': angles,
            'measurements': measurements,
            'session_data': self.posture_session_data,
            'pose_stable': pose_stable,
            'view_angle': view_angle.value,
            'view_weight': view_weight,
            'individual_scores': self._calculate_individual_scores(analyses),
            'confidence': self._calculate_confidence(landmark_points),
            'calibrated': self.calibrated
        }

    def _analyze_head_posture_front(self, landmarks: Dict, angles: Dict, measurements: Dict) -> Dict:
        """Analyze head posture from front view - STRICT MONITORING."""
        issues = []
        recommendations = []
        alerts = []
        score = 95  # Start higher, deduct more severely

        nose = landmarks['nose']
        left_ear = landmarks['ears']['left']
        right_ear = landmarks['ears']['right']
        left_shoulder = landmarks['shoulders']['left']
        right_shoulder = landmarks['shoulders']['right']

        if nose and left_shoulder and right_shoulder:
            shoulder_center = (
                (left_shoulder[0] + right_shoulder[0]) // 2,
                (left_shoulder[1] + right_shoulder[1]) // 2
            )

            # STRICT Head tilt analysis
            if left_ear and right_ear:
                head_tilt = abs(left_ear[1] - right_ear[1])
                ear_distance = abs(left_ear[0] - right_ear[0])

                if ear_distance > 0:
                    tilt_ratio = head_tilt / ear_distance
                    tilt_angle = math.degrees(math.atan(tilt_ratio))
                    angles['head_tilt'] = tilt_angle

                    if tilt_angle > self.config.HEAD_TILT_THRESHOLD + 8:  # >16 degrees
                        issues.append('SEVERE head tilt - major imbalance detected')
                        recommendations.append('URGENT: Level your head immediately')
                        alerts.append((AlertType.FORWARD_HEAD, 'Severe head tilt'))
                        score -= 45
                    elif tilt_angle > self.config.HEAD_TILT_THRESHOLD + 3:  # >11 degrees
                        issues.append('SIGNIFICANT head tilt detected')
                        recommendations.append('Straighten your head position')
                        score -= 30
                    elif tilt_angle > self.config.HEAD_TILT_THRESHOLD:  # >8 degrees
                        issues.append('Noticeable head tilt')
                        recommendations.append('Keep your head level')
                        score -= 20
                    elif tilt_angle > 4:  # Even slight tilt
                        issues.append('Slight head tilt detected')
                        recommendations.append('Minor head position adjustment needed')
                        score -= 12

            # STRICT Forward head position
            head_forward = abs(nose[0] - shoulder_center[0])
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])

            if shoulder_width > 0:
                forward_ratio = head_forward / shoulder_width
                smoothed_ratio = self._smooth_angle('head_forward_front', forward_ratio)
                measurements['head_forward_ratio'] = smoothed_ratio

                if smoothed_ratio > 0.15:  # Much stricter threshold
                    issues.append('CRITICAL forward head posture - head too far forward')
                    recommendations.append('URGENT: Pull head back over shoulders')
                    alerts.append((AlertType.FORWARD_HEAD, 'Critical forward head posture'))
                    score -= 50
                elif smoothed_ratio > 0.10:
                    issues.append('SIGNIFICANT forward head positioning')
                    recommendations.append('Move head back towards shoulder line')
                    score -= 35
                elif smoothed_ratio > 0.06:
                    issues.append('Moderate forward head posture')
                    recommendations.append('Align head better over shoulders')
                    score -= 25
                elif smoothed_ratio > 0.03:  # Very sensitive
                    issues.append('Slight forward head detected')
                    recommendations.append('Minor head position correction needed')
                    score -= 15

        return {'score': score, 'issues': issues, 'recommendations': recommendations, 'alerts': alerts}

    def _analyze_head_posture_side(self, landmarks: Dict, angles: Dict, measurements: Dict,
                                   view_angle: ViewAngle) -> Dict:
        """Analyze head posture from side view - STRICT MONITORING."""
        issues = []
        recommendations = []
        alerts = []
        score = 95

        nose = landmarks['nose']
        visible_ear = landmarks['ears']['left'] if view_angle == ViewAngle.SIDE_LEFT else landmarks['ears']['right']
        visible_shoulder = landmarks['shoulders']['left'] if view_angle == ViewAngle.SIDE_LEFT else \
        landmarks['shoulders']['right']

        if nose and visible_ear and visible_shoulder:
            # STRICT neck angle analysis
            neck_angle = self.angle_between_points(visible_ear, nose, visible_shoulder)
            if neck_angle:
                smoothed_angle = self._smooth_angle('neck_angle_side', neck_angle)
                angles['neck_angle'] = smoothed_angle

                # Much stricter ideal angle expectations
                ideal_angle = 90
                angle_deviation = abs(smoothed_angle - ideal_angle)

                if angle_deviation > 18:  # Stricter than before (was 25)
                    issues.append('SEVERE forward head posture - major neck strain')
                    recommendations.append('URGENT: Perform chin tucks, fix workstation')
                    alerts.append((AlertType.FORWARD_HEAD, 'Severe forward head posture'))
                    score -= 55
                elif angle_deviation > 12:  # Stricter than before (was 15)
                    issues.append('SIGNIFICANT forward head posture detected')
                    recommendations.append('Pull chin back, elongate neck')
                    score -= 40
                elif angle_deviation > 8:
                    issues.append('Moderate forward head posture')
                    recommendations.append('Improve head position alignment')
                    score -= 25
                elif angle_deviation > 5:  # Very sensitive
                    issues.append('Slight forward head tendency')
                    recommendations.append('Minor neck position adjustment needed')
                    score -= 15

        return {'score': score, 'issues': issues, 'recommendations': recommendations, 'alerts': alerts}

    def _analyze_shoulder_alignment(self, landmarks: Dict, angles: Dict, measurements: Dict) -> Dict:
        """Analyze shoulder alignment from front view - STRICT MONITORING."""
        issues = []
        recommendations = []
        alerts = []
        score = 95

        left_shoulder = landmarks['shoulders']['left']
        right_shoulder = landmarks['shoulders']['right']

        if left_shoulder and right_shoulder:
            height_diff = abs(left_shoulder[1] - right_shoulder[1])
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])

            if shoulder_width > 0:
                shoulder_slope = height_diff / shoulder_width
                angle = math.degrees(math.atan(shoulder_slope))
                smoothed_angle = self._smooth_angle('shoulder_alignment', angle)
                angles['shoulder_alignment'] = smoothed_angle

                # Much stricter shoulder alignment requirements
                if smoothed_angle > self.config.SHOULDER_ALIGNMENT_THRESHOLD + 6:  # >12 degrees
                    issues.append('SEVERE shoulder misalignment - major imbalance')
                    recommendations.append('URGENT: Check workstation, see healthcare provider')
                    alerts.append((AlertType.UNEVEN_SHOULDERS, 'Severe shoulder misalignment'))
                    score -= 50
                elif smoothed_angle > self.config.SHOULDER_ALIGNMENT_THRESHOLD + 2:  # >8 degrees
                    issues.append('SIGNIFICANT shoulder height difference')
                    recommendations.append('Level shoulders, check desk setup')
                    score -= 35
                elif smoothed_angle > self.config.SHOULDER_ALIGNMENT_THRESHOLD:  # >6 degrees
                    issues.append('Noticeable shoulder misalignment')
                    recommendations.append('Consciously level your shoulders')
                    score -= 25
                elif smoothed_angle > 3:  # Very sensitive
                    issues.append('Slight shoulder imbalance detected')
                    recommendations.append('Minor shoulder position adjustment needed')
                    score -= 15
                elif smoothed_angle > 1.5:  # Extremely sensitive
                    issues.append('Minimal shoulder height difference')
                    recommendations.append('Maintain shoulder awareness')
                    score -= 8

        return {'score': score, 'issues': issues, 'recommendations': recommendations, 'alerts': alerts}

    def _analyze_spine_alignment_front(self, landmarks: Dict, angles: Dict, measurements: Dict) -> Dict:
        """Analyze spine alignment from front view - STRICT MONITORING."""
        issues = []
        recommendations = []
        alerts = []
        score = 95

        left_shoulder = landmarks['shoulders']['left']
        right_shoulder = landmarks['shoulders']['right']
        left_hip = landmarks['hips']['left']
        right_hip = landmarks['hips']['right']

        if all([left_shoulder, right_shoulder, left_hip, right_hip]):
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
            hip_center = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)

            # STRICT lateral spine deviation
            lateral_deviation = abs(shoulder_center[0] - hip_center[0])
            torso_height = abs(shoulder_center[1] - hip_center[1])

            if torso_height > 0:
                deviation_ratio = lateral_deviation / torso_height
                smoothed_ratio = self._smooth_angle('spine_lateral', deviation_ratio)
                measurements['lateral_deviation'] = lateral_deviation

                # Much stricter spine alignment requirements
                if smoothed_ratio > 0.08:  # Stricter threshold
                    issues.append('SEVERE lateral spine curvature - major alignment issue')
                    recommendations.append('URGENT: Center torso, see posture specialist')
                    alerts.append((AlertType.SLOUCHING, 'Severe spine misalignment'))
                    score -= 45
                elif smoothed_ratio > 0.05:
                    issues.append('SIGNIFICANT lateral spine deviation')
                    recommendations.append('Center your torso over hips immediately')
                    score -= 30
                elif smoothed_ratio > 0.03:
                    issues.append('Noticeable lateral lean detected')
                    recommendations.append('Straighten your posture alignment')
                    score -= 20
                elif smoothed_ratio > 0.015:  # Very sensitive
                    issues.append('Slight lateral spine deviation')
                    recommendations.append('Minor torso centering needed')
                    score -= 12
                elif smoothed_ratio > 0.008:  # Extremely sensitive
                    issues.append('Minimal lateral lean tendency')
                    recommendations.append('Maintain spine awareness')
                    score -= 6

        return {'score': score, 'issues': issues, 'recommendations': recommendations, 'alerts': alerts}

    def _analyze_spine_alignment_side(self, landmarks: Dict, angles: Dict, measurements: Dict,
                                      view_angle: ViewAngle) -> Dict:
        """Analyze spine alignment from side view - STRICT MONITORING."""
        issues = []
        recommendations = []
        alerts = []
        score = 95

        visible_shoulder = landmarks['shoulders']['left'] if view_angle == ViewAngle.SIDE_LEFT else \
        landmarks['shoulders']['right']
        visible_hip = landmarks['hips']['left'] if view_angle == ViewAngle.SIDE_LEFT else landmarks['hips']['right']

        if visible_shoulder and visible_hip:
            # STRICT forward lean analysis
            forward_lean = abs(visible_shoulder[0] - visible_hip[0])
            torso_height = abs(visible_shoulder[1] - visible_hip[1])

            if torso_height > 0:
                lean_angle = math.degrees(math.atan(forward_lean / torso_height))
                smoothed_angle = self._smooth_angle('spine_lean_side', lean_angle)
                angles['spine_lean'] = smoothed_angle

                # Much stricter slouching detection
                if smoothed_angle > self.config.SLOUCH_ALERT_THRESHOLD + 10:  # >18 degrees
                    issues.append('CRITICAL slouching - severe spine misalignment')
                    recommendations.append('URGENT: Sit up straight immediately')
                    alerts.append((AlertType.SLOUCHING, 'Critical slouching detected'))
                    score -= 60
                elif smoothed_angle > self.config.SLOUCH_ALERT_THRESHOLD + 5:  # >13 degrees
                    issues.append('SEVERE forward slouching detected')
                    recommendations.append('Straighten spine, align shoulders over hips')
                    alerts.append((AlertType.SLOUCHING, 'Severe slouching'))
                    score -= 45
                elif smoothed_angle > self.config.SLOUCH_ALERT_THRESHOLD:  # >8 degrees
                    issues.append('SIGNIFICANT forward lean detected')
                    recommendations.append('Improve sitting/standing posture')
                    score -= 30
                elif smoothed_angle > 5:
                    issues.append('Moderate forward posture')
                    recommendations.append('Straighten your back')
                    score -= 20
                elif smoothed_angle > 3:  # Very sensitive
                    issues.append('Slight forward lean tendency')
                    recommendations.append('Minor posture correction needed')
                    score -= 12
                elif smoothed_angle > 1.5:  # Extremely sensitive
                    issues.append('Minimal forward lean detected')
                    recommendations.append('Maintain upright awareness')
                    score -= 6

        return {'score': score, 'issues': issues, 'recommendations': recommendations, 'alerts': alerts}

    def _analyze_shoulder_posture_side(self, landmarks: Dict, angles: Dict, measurements: Dict,
                                       view_angle: ViewAngle) -> Dict:
        """Analyze shoulder posture from side view."""
        issues = []
        recommendations = []
        alerts = []
        score = 85

        visible_shoulder = landmarks['shoulders']['left'] if view_angle == ViewAngle.SIDE_LEFT else \
        landmarks['shoulders']['right']
        visible_elbow = landmarks['elbows']['left'] if view_angle == ViewAngle.SIDE_LEFT else landmarks['elbows'][
            'right']

        if visible_shoulder and visible_elbow:
            # Check for rounded shoulders
            arm_angle = self.angle_between_points(visible_shoulder, visible_elbow,
                                                  (visible_elbow[0], visible_elbow[1] + 100))
            if arm_angle and arm_angle < 80:  # Arms hanging too far forward
                issues.append('Rounded shoulders detected')
                recommendations.append('Pull shoulders back, open chest')
                score -= 20

        return {'score': score, 'issues': issues, 'recommendations': recommendations, 'alerts': alerts}

    def _analyze_basic_alignment(self, landmarks: Dict, angles: Dict, measurements: Dict) -> Dict:
        """Basic alignment analysis for unclear views."""
        return {'score': 70, 'issues': ['View angle unclear'], 'recommendations': ['Face camera for better analysis'],
                'alerts': []}

    def _extract_key_landmarks(self, landmark_points: Dict[int, Tuple[int, int]]) -> Dict:
        """Extract and organize key landmarks."""
        return {
            'nose': landmark_points.get(0),
            'ears': {
                'left': landmark_points.get(7),
                'right': landmark_points.get(8)
            },
            'shoulders': {
                'left': landmark_points.get(11),
                'right': landmark_points.get(12)
            },
            'elbows': {
                'left': landmark_points.get(13),
                'right': landmark_points.get(14)
            },
            'hips': {
                'left': landmark_points.get(23),
                'right': landmark_points.get(24)
            }
        }

    def _smooth_angle(self, key: str, value: float) -> float:
        """Apply smoothing to angle measurements."""
        if key not in self.smoothed_angles:
            self.smoothed_angles[key] = value
        else:
            alpha = self.config.SMOOTHING_ALPHA
            self.smoothed_angles[key] = alpha * value + (1 - alpha) * self.smoothed_angles[key]
        return self.smoothed_angles[key]

    def _calculate_individual_scores(self, analyses: List[Dict]) -> Dict:
        """Calculate individual component scores."""
        scores = {'head': 0, 'shoulders': 0, 'spine': 0}
        if analyses:
            avg_score = sum(a.get('score', 0) for a in analyses) / len(analyses)
            scores = {'head': avg_score, 'shoulders': avg_score, 'spine': avg_score}
        return scores

    def _determine_posture_state(self, overall_score: float) -> PostureState:
        """Determine posture state based on STRICT scoring system."""
        if overall_score >= 92:  # Much stricter - was 85
            return PostureState.EXCELLENT
        elif overall_score >= 82:  # Much stricter - was 70
            return PostureState.GOOD
        elif overall_score >= 70:  # Stricter - was 55
            return PostureState.FAIR
        elif overall_score >= 55:  # Stricter - was 35
            return PostureState.POOR
        else:
            return PostureState.VERY_POOR

    def _update_session_data(self, overall_score: float, state: PostureState, alerts: List):
        """Update session tracking data."""
        current_time = time.time()

        if overall_score >= self.config.GOOD_POSTURE_SCORE_THRESHOLD:
            self.posture_session_data['good_posture_time'] += 1
        elif overall_score <= self.config.POOR_POSTURE_SCORE_THRESHOLD:
            self.posture_session_data['poor_posture_time'] += 1

        self.posture_session_data['posture_scores'].append({
            'timestamp': current_time,
            'score': overall_score,
            'state': state.value
        })

    def _trigger_alert(self, alert_type: AlertType, message: str):
        """Trigger posture alert with callbacks."""
        current_time = time.time()

        if current_time - self.last_alert_time < self.config.ALERT_COOLDOWN_SECONDS:
            return

        self.last_alert_time = current_time
        self.posture_session_data['alerts_triggered'] += 1
        self.posture_session_data['alert_history'].append({
            'timestamp': current_time,
            'type': alert_type.value,
            'message': message
        })

        for callback in self.alert_callbacks:
            try:
                callback(alert_type, message)
            except Exception as e:
                self.logger.warning(f"Alert callback failed: {e}")

    def _calculate_confidence(self, landmark_points: Dict[int, Tuple[int, int]]) -> float:
        """Calculate confidence based on landmark detection."""
        key_landmarks = [0, 11, 12, 23, 24]  # nose, shoulders, hips
        detected = sum(1 for lm in key_landmarks if lm in landmark_points)
        return detected / len(key_landmarks)

    def _create_empty_analysis(self) -> Dict:
        """Create empty analysis result."""
        return {
            'overall_score': 0,
            'state': PostureState.VERY_POOR,
            'issues': ['Insufficient pose data'],
            'recommendations': ['Ensure you are visible in the camera'],
            'angles': {},
            'measurements': {},
            'session_data': self.posture_session_data,
            'pose_stable': False,
            'view_angle': ViewAngle.UNKNOWN.value,
            'view_weight': 0.0,
            'individual_scores': {'head': 0, 'shoulders': 0, 'spine': 0},
            'confidence': 0.0,
            'calibrated': self.calibrated
        }

    @staticmethod
    def angle_between_points(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> Optional[float]:
        """Calculate angle between three points."""
        try:
            (ax, ay), (bx, by), (cx, cy) = a, b, c
            v1 = np.array([ax - bx, ay - by])
            v2 = np.array([cx - bx, cy - by])

            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)

            if norms < 1e-6:
                return None

            cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
            return math.degrees(np.arccos(cos_angle))
        except (ValueError, ZeroDivisionError):
            return None


# ================= ENHANCED VISUALIZER =================
class PostureVisualizer:
    """Enhanced visualization with better text positioning and adaptive display."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

        # Enhanced color schemes using the provided palette
        # Palette: #003049 (navy), #D62828 (red), #F77F00 (orange), #FCBF49 (yellow), #EAE2B7 (cream)
        self.colors = {
            PostureState.EXCELLENT: (183, 226, 234),  # Light cream/beige (EAE2B7)
            PostureState.GOOD: (73, 191, 252),  # Yellow (FCBF49)
            PostureState.FAIR: (0, 127, 247),  # Orange (F77F00)
            PostureState.POOR: (40, 40, 214),  # Red (D62828)
            PostureState.VERY_POOR: (0, 0, 255),  # Bright red for critical alerts
        }

        self.view_angle_colors = {
            ViewAngle.FRONT: (73, 191, 252),  # Yellow (FCBF49)
            ViewAngle.SIDE_LEFT: (0, 127, 247),  # Orange (F77F00)
            ViewAngle.SIDE_RIGHT: (0, 127, 247),  # Orange (F77F00)
            ViewAngle.BACK: (40, 40, 214),  # Red (D62828)
            ViewAngle.UNKNOWN: (128, 128, 128),  # Gray
        }

        # Text and background colors using the palette
        self.text_color = (183, 226, 234)  # Light cream for text
        self.bg_color = (73, 48, 0)  # Navy blue background (003049)
        self.accent_color = (73, 191, 252)  # Yellow for accents
        # Text properties
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.last_alert_time = 0
        self.alert_flash_active = False

    def _get_text_size(self, text: str, scale: float = 0.6) -> Tuple[int, int]:
        """Get text size for proper positioning."""
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, scale, self.config.TEXT_THICKNESS
        )
        return text_width, text_height + baseline

    def _draw_text_with_background(self, frame: np.ndarray, text: str,
                                   pos: Tuple[int, int], scale: float = 0.6,
                                   text_color: Tuple[int, int, int] = None,
                                   bg_color: Tuple[int, int, int] = None,
                                   padding: int = 5):
        """Draw text with background for better readability."""
        if text_color is None:
            text_color = self.text_color
        if bg_color is None:
            bg_color = self.bg_color

        text_width, text_height = self._get_text_size(text, scale)

        # Draw background rectangle
        bg_start = (pos[0] - padding, pos[1]+10 - text_height - padding+8)
        bg_end = (pos[0] + text_width + padding, pos[1]+10 + padding)
        cv2.rectangle(frame, bg_start, bg_end, bg_color, cv2.FILLED)

        # Draw text
        cv2.putText(frame, text, (pos[0], pos[1] + 10), self.font, scale, text_color, self.config.TEXT_THICKNESS)

        return text_height + 2 * padding

    def draw_adaptive_skeleton(self, frame: np.ndarray, posture_analysis: Dict,
                               landmark_points: Dict[int, Tuple[int, int]]):
        """Draw skeleton that adapts to view angle and emphasizes posture issues dramatically."""
        if not landmark_points:
            return

        state = posture_analysis['state']
        view_angle = ViewAngle(posture_analysis.get('view_angle', 'front'))
        main_color = self.colors.get(state, (255, 255, 255))

        # DRAMATIC visual effects for poor posture
        current_time = time.time()

        # Flash effect for alerts (more intense)
        flash_active = False
        if current_time - self.last_alert_time < self.config.ALERT_FLASH_DURATION:
            if int(current_time * 8) % 2:  # Flash at 4Hz
                main_color = (0, 0, 255)  # Red flash
                flash_active = True

        # Pulsing effect for very poor posture
        pulse_factor = 1.0
        if state == PostureState.VERY_POOR:
            pulse_factor = 0.7 + 0.3 * math.sin(time.time() * 6)  # Pulsing effect
            if not flash_active:
                main_color = tuple(int(c * pulse_factor + (255 - c) * (1 - pulse_factor) * 0.3) for c in main_color)

        # Draw connections based on view angle with enhanced thickness for poor posture
        base_thickness = 4 if state in [PostureState.POOR, PostureState.VERY_POOR] else 3

        if view_angle == ViewAngle.FRONT:
            self._draw_front_view_skeleton(frame, landmark_points, posture_analysis, main_color, base_thickness)
        elif view_angle in [ViewAngle.SIDE_LEFT, ViewAngle.SIDE_RIGHT]:
            self._draw_side_view_skeleton(frame, landmark_points, posture_analysis, main_color, view_angle,
                                          base_thickness)
        else:
            self._draw_basic_skeleton(frame, landmark_points, main_color, base_thickness)

        # Draw landmarks with different sizes based on confidence and state
        confidence = posture_analysis.get('confidence', 0.5)
        base_size = max(4, int(8 * confidence))

        # Larger, more prominent landmarks for poor posture
        if state in [PostureState.POOR, PostureState.VERY_POOR]:
            base_size += 2

        key_points = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
        for point_id in key_points:
            if point_id in landmark_points:
                pos = landmark_points[point_id]

                # Special highlighting for problem areas
                point_color = main_color
                issues = posture_analysis.get('issues', [])

                # Highlight specific problem points
                if point_id == 0 and any('head' in issue.lower() or 'neck' in issue.lower() for issue in issues):
                    point_color = (0, 0, 255)  # Red for head issues
                    base_size += 2
                elif point_id in [11, 12] and any('shoulder' in issue.lower() for issue in issues):
                    point_color = (0, 100, 255)  # Orange for shoulder issues
                    base_size += 1

                cv2.circle(frame, pos, base_size, point_color, cv2.FILLED)
                cv2.circle(frame, pos, base_size + 2, (255, 255, 255), 2)

                # Add warning indicators for critical points
                if state == PostureState.VERY_POOR and point_id in [0, 11, 12]:  # Head and shoulders
                    cv2.circle(frame, pos, base_size + 6, (0, 0, 255), 2)

    def _draw_front_view_skeleton(self, frame: np.ndarray, landmark_points: Dict,
                                  posture_analysis: Dict, main_color: Tuple[int, int, int], base_thickness: int = 3):
        """Draw skeleton optimized for front view with enhanced issue highlighting."""
        issues = posture_analysis.get('issues', [])

        # Head and neck with enhanced issue visualization
        if 0 in landmark_points and 11 in landmark_points and 12 in landmark_points:
            nose = landmark_points[0]
            left_shoulder = landmark_points[11]
            right_shoulder = landmark_points[12]
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) // 2,
                               (left_shoulder[1] + right_shoulder[1]) // 2)

            # Enhanced neck line based on issues
            has_head_issue = any('head' in issue.lower() or 'neck' in issue.lower() for issue in issues)
            neck_color = (0, 0, 255) if has_head_issue else main_color
            neck_thickness = base_thickness + 2 if has_head_issue else base_thickness

            cv2.line(frame, nose, shoulder_center, neck_color, neck_thickness)

            # Draw deviation line if head is off-center
            if has_head_issue:
                cv2.circle(frame, nose, 18, (0, 0, 255), 3)
                # Draw ideal position indicator
                cv2.circle(frame, (shoulder_center[0], nose[1]), 8, (0, 255, 0), 2)
                cv2.line(frame, nose, (shoulder_center[0], nose[1]), (255, 255, 0), 2)

        # Enhanced shoulder line
        if 11 in landmark_points and 12 in landmark_points:
            has_shoulder_issue = any('shoulder' in issue.lower() for issue in issues)
            shoulder_color = (0, 100, 255) if has_shoulder_issue else main_color
            shoulder_thickness = base_thickness + 2 if has_shoulder_issue else base_thickness + 1

            cv2.line(frame, landmark_points[11], landmark_points[12], shoulder_color, shoulder_thickness)

            # Draw level indicator if shoulders are uneven
            if has_shoulder_issue:
                left_y = landmark_points[11][1]
                right_y = landmark_points[12][1]
                avg_y = (left_y + right_y) // 2

                # Draw ideal level line
                cv2.line(frame, (landmark_points[11][0], avg_y), (landmark_points[12][0], avg_y), (0, 255, 0), 2)

        # Enhanced torso with spine alignment visualization
        connections = [(11, 23), (12, 24), (23, 24)]
        has_spine_issue = any(
            'slouch' in issue.lower() or 'spine' in issue.lower() or 'lean' in issue.lower() for issue in issues)
        torso_color = (0, 50, 255) if has_spine_issue else main_color
        torso_thickness = base_thickness + 1 if has_spine_issue else base_thickness

        for start_id, end_id in connections:
            if start_id in landmark_points and end_id in landmark_points:
                cv2.line(frame, landmark_points[start_id], landmark_points[end_id], torso_color, torso_thickness)

        # Draw center line for spine alignment reference
        if has_spine_issue and 11 in landmark_points and 12 in landmark_points and 23 in landmark_points and 24 in landmark_points:
            shoulder_center = ((landmark_points[11][0] + landmark_points[12][0]) // 2,
                               (landmark_points[11][1] + landmark_points[12][1]) // 2)
            hip_center = ((landmark_points[23][0] + landmark_points[24][0]) // 2,
                          (landmark_points[23][1] + landmark_points[24][1]) // 2)

            # Draw actual spine line
            cv2.line(frame, shoulder_center, hip_center, torso_color, torso_thickness)

            # Draw ideal spine line (vertical)
            ideal_hip = (shoulder_center[0], hip_center[1])
            cv2.line(frame, shoulder_center, ideal_hip, (0, 255, 0), 2)

        # Arms with normal thickness
        arm_connections = [(11, 13), (13, 15), (12, 14), (14, 16)]
        for start_id, end_id in arm_connections:
            if start_id in landmark_points and end_id in landmark_points:
                cv2.line(frame, landmark_points[start_id], landmark_points[end_id], main_color, base_thickness - 1)

    def _draw_side_view_skeleton(self, frame: np.ndarray, landmark_points: Dict,
                                 posture_analysis: Dict, main_color: Tuple[int, int, int],
                                 view_angle: ViewAngle, base_thickness: int = 3):
        """Draw skeleton optimized for side view with enhanced issue highlighting."""
        issues = posture_analysis.get('issues', [])

        # Determine visible side
        visible_side = 'left' if view_angle == ViewAngle.SIDE_LEFT else 'right'
        shoulder_id = 11 if visible_side == 'left' else 12
        elbow_id = 13 if visible_side == 'left' else 14
        hip_id = 23 if visible_side == 'left' else 24

        # Enhanced head to shoulder line (critical for side view posture)
        if 0 in landmark_points and shoulder_id in landmark_points:
            has_head_issue = any('head' in issue.lower() or 'neck' in issue.lower() for issue in issues)
            head_color = (0, 0, 255) if has_head_issue else main_color
            head_thickness = base_thickness + 3 if has_head_issue else base_thickness + 1

            cv2.line(frame, landmark_points[0], landmark_points[shoulder_id], head_color, head_thickness)

            # Draw ideal head position indicator
            if has_head_issue:
                ideal_head_x = landmark_points[shoulder_id][0]
                ideal_head_pos = (ideal_head_x, landmark_points[0][1])
                cv2.circle(frame, ideal_head_pos, 12, (0, 255, 0), 2)
                cv2.line(frame, landmark_points[0], ideal_head_pos, (255, 255, 0), 2)

        # Enhanced spine line (shoulder to hip) - most important for side view
        if shoulder_id in landmark_points and hip_id in landmark_points:
            has_spine_issue = any('slouch' in issue.lower() or 'lean' in issue.lower() for issue in issues)
            spine_color = (0, 50, 255) if has_spine_issue else main_color
            spine_thickness = base_thickness + 3 if has_spine_issue else base_thickness + 2

            cv2.line(frame, landmark_points[shoulder_id], landmark_points[hip_id], spine_color, spine_thickness)

            # Draw ideal spine position (more vertical)
            if has_spine_issue:
                ideal_shoulder_x = landmark_points[hip_id][0]  # Shoulder should be over hip
                ideal_shoulder_pos = (ideal_shoulder_x, landmark_points[shoulder_id][1])
                cv2.line(frame, ideal_shoulder_pos, landmark_points[hip_id], (0, 255, 0), 3)
                cv2.circle(frame, ideal_shoulder_pos, 8, (0, 255, 0), 2)

        # Visible arm
        if shoulder_id in landmark_points and elbow_id in landmark_points:
            cv2.line(frame, landmark_points[shoulder_id], landmark_points[elbow_id], main_color, base_thickness)

    def _draw_basic_skeleton(self, frame: np.ndarray, landmark_points: Dict,
                             main_color: Tuple[int, int, int], base_thickness: int = 3):
        """Draw basic skeleton for unclear views."""
        # Just draw main body structure with enhanced thickness
        connections = [(11, 12), (11, 23), (12, 24), (23, 24)]
        for start_id, end_id in connections:
            if start_id in landmark_points and end_id in landmark_points:
                cv2.line(frame, landmark_points[start_id], landmark_points[end_id], main_color, base_thickness)

    def draw_comprehensive_hud(self, frame: np.ndarray, posture_analysis: Dict,
                               fps: float, detection_fps: float = 0):
        """Draw comprehensive HUD with expanded text area for strict monitoring."""
        h, w = frame.shape[:2]

        # Calculate HUD position (top-left) with expanded size
        hud_x = self.config.HUD_MARGIN
        hud_y = self.config.HUD_MARGIN
        hud_w = min(self.config.HUD_PANEL_WIDTH, w - 2 * self.config.HUD_MARGIN)
        hud_h = min(self.config.HUD_PANEL_HEIGHT, h - 2 * self.config.HUD_MARGIN)

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h),
                      self.bg_color, cv2.FILLED)
        cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)

        # Border color based on posture state
        state = posture_analysis.get('state', PostureState.VERY_POOR)
        border_color = self.colors.get(state, (255, 255, 255))

        # Dynamic border thickness and style based on severity
        if state == PostureState.VERY_POOR:
            cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), border_color, 4)
            # Add warning border flash
            if int(time.time() * 4) % 2:  # Flash at 2Hz for very poor posture
                cv2.rectangle(frame, (hud_x - 2, hud_y - 2), (hud_x + hud_w + 2, hud_y + hud_h + 2), (0, 0, 255), 2)
        elif state == PostureState.POOR:
            cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), border_color, 3)
        else:
            cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), border_color, 2)

        # Start text layout with proper spacing
        current_y = hud_y + 20
        line_spacing = 18
        section_spacing = 8

        # Title
        title_text = "POSTURE MONITOR"
        title_color = border_color if state != PostureState.VERY_POOR else (183, 226, 234)
        current_y += self._draw_text_with_background(frame, title_text, (hud_x + 15, current_y),
                                                     0.7, title_color, (73, 48, 0)) + section_spacing

        # === SCORE SECTION ===
        score = posture_analysis.get('overall_score', 0)
        confidence = posture_analysis.get('confidence', 0.0)
        view_angle = posture_analysis.get('view_angle', 'unknown')

        # Main score with severity indicator
        score_text = f"POSTURE SCORE: {int(score)}%"
        score_color = border_color
        if score < 70:
            score_text += " - POOR!"
            score_color = (0, 0, 255)  # Bright red for critical
        elif score < 82:
            score_text += " - NEEDS IMPROVEMENT"
            score_color = (40, 40, 214)  # Red from palette

        current_y += self._draw_text_with_background(frame, score_text, (hud_x + 15, current_y),
                                                     0.65, score_color) + 3

        # Status with emphasis
        status_text = f"STATUS: {state.value.upper()}"
        if state in [PostureState.POOR, PostureState.VERY_POOR]:
            status_text += " [WARNING]"
        current_y += self._draw_text_with_background(frame, status_text, (hud_x + 15, current_y),
                                                     0.6, border_color) + 5

        # Progress bar with strict thresholds
        self._draw_enhanced_progress_bar(frame, hud_x + 15, current_y, hud_w - 30, 20,
                                         score, border_color)
        current_y += 28

        # === METRICS SECTION ===
        metrics_header = "=== DETECTION METRICS ==="
        current_y += self._draw_text_with_background(frame, metrics_header, (hud_x + 15, current_y),
                                                     0.5, (73, 191, 252)) + 2

        confidence_text = f"Detection Confidence: {confidence:.0%}"
        conf_color = (183, 226, 234) if confidence > 0.8 else (73, 191, 252) if confidence > 0.6 else (40, 40, 214)
        current_y += self._draw_text_with_background(frame, confidence_text, (hud_x + 20, current_y),
                                                     0.45, conf_color) + line_spacing - 3

        view_text = f"View Angle: {view_angle.replace('_', ' ').title()}"
        view_weight = posture_analysis.get('view_weight', 1.0)
        if view_weight < 0.8:
            view_text += f" (Weight: {view_weight:.1f})"
        current_y += self._draw_text_with_background(frame, view_text, (hud_x + 20, current_y),
                                                     0.45) + section_spacing

        # === COMPONENT SCORES SECTION ===
        individual_scores = posture_analysis.get('individual_scores', {})
        components_header = "=== COMPONENT ANALYSIS ==="
        current_y += self._draw_text_with_background(frame, components_header, (hud_x + 15, current_y),
                                                     0.5, (0, 127, 247)) + 2

        # Detailed component breakdown
        components = [
            ("Head/Neck", individual_scores.get('head', 0)),
            ("Shoulders", individual_scores.get('shoulders', 0)),
            ("Spine", individual_scores.get('spine', 0))
        ]

        for comp_name, comp_score in components:
            comp_color = (183, 226, 234) if comp_score >= 85 else (73, 191, 252) if comp_score >= 70 else (40, 40, 214)
            comp_text = f"{comp_name}: {comp_score:.0f}%"
            if comp_score < 70:
                comp_text += " [LOW]"
            current_y += self._draw_text_with_background(frame, comp_text, (hud_x + 20, current_y),
                                                         0.45, comp_color) + line_spacing - 5

        current_y += section_spacing

        # === SESSION INFO SECTION ===
        session_data = posture_analysis.get('session_data', {})
        session_time = time.time() - session_data.get('start_time', time.time())
        minutes, seconds = divmod(int(session_time), 60)

        session_header = "=== SESSION TRACKING ==="
        current_y += self._draw_text_with_background(frame, session_header, (hud_x + 15, current_y),
                                                     0.5, (73, 191, 252)) + 2

        session_text = f"Duration: {minutes:02d}:{seconds:02d}"
        current_y += self._draw_text_with_background(frame, session_text, (hud_x + 20, current_y),
                                                     0.45) + line_spacing - 5

        alerts_count = session_data.get('alerts_triggered', 0)
        alert_text = f"Alerts Triggered: {alerts_count}"
        alert_color = (40, 40, 214) if alerts_count > 5 else (0, 127, 247) if alerts_count > 0 else (183, 226, 234)
        current_y += self._draw_text_with_background(frame, alert_text, (hud_x + 20, current_y),
                                                     0.45, alert_color) + section_spacing

        # === ISSUES SECTION ===
        issues = posture_analysis.get('issues', [])
        if issues:
            issues_header = "=== CURRENT ISSUES ==="
            current_y += self._draw_text_with_background(frame, issues_header, (hud_x + 15, current_y),
                                                         0.5, (40, 40, 214)) + 2

            # Show top 4 issues with severity indicators using new palette
            for i, issue in enumerate(issues[:4]):
                issue_color = (40, 40, 214)  # Default red
                if "CRITICAL" in issue or "SEVERE" in issue:
                    issue_color = (0, 0, 255)  # Bright red for critical
                    issue = "[CRITICAL] " + issue
                elif "SIGNIFICANT" in issue:
                    issue_color = (40, 40, 214)  # Red from palette
                    issue = "[WARNING] " + issue
                elif "Slight" in issue or "Minor" in issue:
                    issue_color = (0, 127, 247)  # Orange from palette
                    issue = "[MINOR] " + issue

                # Truncate long issues
                display_issue = issue[:50] + "..." if len(issue) > 50 else issue
                current_y += self._draw_text_with_background(frame, f"- {display_issue}",
                                                             (hud_x + 20, current_y), 0.42, issue_color) + 14

            current_y += section_spacing

        # === RECOMMENDATIONS SECTION ===
        recommendations = posture_analysis.get('recommendations', [])
        if recommendations:
            rec_header = "=== RECOMMENDATIONS ==="
            current_y += self._draw_text_with_background(frame, rec_header, (hud_x + 15, current_y),
                                                         0.5, (183, 226, 234)) + 2

            # Show top 2 recommendations using new palette
            for rec in recommendations[:2]:
                rec_short = rec[:48] + "..." if len(rec) > 48 else rec
                current_y += self._draw_text_with_background(frame, f"> {rec_short}",
                                                             (hud_x + 20, current_y), 0.42, (183, 226, 234)) + 14

        # === PERFORMANCE INFO (Bottom) ===
        perf_y = hud_y + hud_h - 55
        perf_text = f"Performance: Render {int(fps)}fps | Detection {int(detection_fps)}fps"
        self._draw_text_with_background(frame, perf_text, (hud_x + 15, perf_y),
                                        0.4, (73, 191, 252))

        # === CONTROLS (Bottom) ===
        controls_y = hud_y + hud_h - 35
        controls_text = "Controls: C=Calibrate | R=Reset | S=Save | Q=Quit"
        self._draw_text_with_background(frame, controls_text, (hud_x + 15, controls_y),
                                        0.4, (183, 226, 234))

        # === STRICT MODE INDICATOR ===
        strict_y = hud_y + hud_h - 15
        strict_text = "[STRICT MODE] High sensitivity posture monitoring active"
        self._draw_text_with_background(frame, strict_text, (hud_x + 15, strict_y),
                                        0.35, (40, 40, 214))

    def _draw_enhanced_progress_bar(self, frame: np.ndarray, x: int, y: int,
                                    width: int, height: int, progress: float,
                                    color: Tuple[int, int, int]):
        """Draw enhanced progress bar with strict thresholds and clear zones."""
        # Background with gradient zones
        cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), cv2.FILLED)

        # Draw threshold zones with new palette colors
        excellent_pos = int(width * 0.92)  # 92% threshold
        good_pos = int(width * 0.82)  # 82% threshold
        fair_pos = int(width * 0.70)  # 70% threshold
        poor_pos = int(width * 0.55)  # 55% threshold

        # Color zones using the new palette (from right to left)
        cv2.rectangle(frame, (x + excellent_pos, y), (x + width, y + height), (183, 226, 234),
                      cv2.FILLED)  # Excellent - cream
        cv2.rectangle(frame, (x + good_pos, y), (x + excellent_pos, y + height), (73, 191, 252),
                      cv2.FILLED)  # Good - yellow
        cv2.rectangle(frame, (x + fair_pos, y), (x + good_pos, y + height), (0, 127, 247), cv2.FILLED)  # Fair - orange
        cv2.rectangle(frame, (x + poor_pos, y), (x + fair_pos, y + height), (40, 40, 214), cv2.FILLED)  # Poor - red
        cv2.rectangle(frame, (x, y), (x + poor_pos, y + height), (73, 48, 0), cv2.FILLED)  # Very poor - navy

        # Progress fill overlay
        fill_width = int(width * (progress / 100))
        if fill_width > 0:
            # Dynamic color based on progress using new palette
            if progress >= 92:
                fill_color = (183, 226, 234)  # Cream for excellent
            elif progress >= 82:
                fill_color = (73, 191, 252)  # Yellow for good
            elif progress >= 70:
                fill_color = (0, 127, 247)  # Orange for fair
            elif progress >= 55:
                fill_color = (40, 40, 214)  # Red for poor
            else:
                fill_color = (0, 0, 255)  # Bright red for very poor

            # Draw progress with pulsing effect for poor scores
            alpha = 1.0
            if progress < 70:
                alpha = 0.7 + 0.3 * math.sin(time.time() * 5)  # Pulsing effect

            # Draw filled portion
            progress_overlay = frame[y:y + height, x:x + fill_width].copy()
            cv2.rectangle(progress_overlay, (0, 0), (fill_width, height), fill_color, cv2.FILLED)
            frame[y:y + height, x:x + fill_width] = cv2.addWeighted(
                frame[y:y + height, x:x + fill_width], 1 - alpha, progress_overlay, alpha, 0
            )

        # Border using accent color
        cv2.rectangle(frame, (x, y), (x + width, y + height), (73, 191, 252), 2)

        # Threshold markers with labels using new palette
        threshold_markers = [
            (excellent_pos, "EXC", (183, 226, 234)),
            (good_pos, "GOOD", (73, 191, 252)),
            (fair_pos, "FAIR", (0, 127, 247)),
            (poor_pos, "POOR", (40, 40, 214))
        ]

        for marker_x, label, label_color in threshold_markers:
            cv2.line(frame, (x + marker_x, y), (x + marker_x, y + height), (183, 226, 234), 1)
            # Label above bar
            cv2.putText(frame, label, (x + marker_x - 10, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, label_color, 1)

        # Progress percentage text
        progress_text = f"{int(progress)}%"
        text_x = x + width + 8
        text_y = y + height - 2

        # Color code the percentage text using new palette
        text_color = color
        if progress < 55:
            text_color = (0, 0, 255)  # Bright red for critical
            progress_text += " CRITICAL"
        elif progress < 70:
            text_color = (40, 40, 214)  # Red for poor
            progress_text += " POOR"
        elif progress < 82:
            text_color = (0, 127, 247)  # Orange for needs work
            progress_text += " NEEDS WORK"

        cv2.putText(frame, progress_text, (text_x, text_y), self.font, 0.45, text_color, 2)

    def draw_view_angle_indicator(self, frame: np.ndarray, view_angle: ViewAngle,
                                  confidence: float):
        """Draw view angle indicator in top-right corner with proper spacing."""
        h, w = frame.shape[:2]

        # Position in top-right with better spacing
        indicator_size = 80
        x = w - indicator_size - 30  # More margin from edge
        y = 30  # More margin from top

        # Background circle using navy from palette
        center = (x + indicator_size // 2, y + indicator_size // 2)
        cv2.circle(frame, center, indicator_size // 2, (73, 48, 0), cv2.FILLED)
        cv2.circle(frame, center, indicator_size // 2, (183, 226, 234), 2)

        # View angle color
        view_color = self.view_angle_colors.get(view_angle, (128, 128, 128))

        # Draw direction indicator
        if view_angle == ViewAngle.FRONT:
            cv2.circle(frame, center, 15, view_color, cv2.FILLED)
        elif view_angle == ViewAngle.SIDE_LEFT:
            points = np.array([[center[0] - 15, center[1]],
                               [center[0] + 10, center[1] - 10],
                               [center[0] + 10, center[1] + 10]], np.int32)
            cv2.fillPoly(frame, [points], view_color)
        elif view_angle == ViewAngle.SIDE_RIGHT:
            points = np.array([[center[0] + 15, center[1]],
                               [center[0] - 10, center[1] - 10],
                               [center[0] - 10, center[1] + 10]], np.int32)
            cv2.fillPoly(frame, [points], view_color)

        # View angle text with better positioning
        text = view_angle.value.replace('_', ' ').title()
        text_size = cv2.getTextSize(text, self.font, 0.4, 1)[0]

        # Center the text under the indicator
        text_x = center[0] - text_size[0] // 2
        text_y = y + indicator_size + 20

        # Draw text with background for better readability using new palette
        self._draw_text_with_background(frame, text, (text_x, text_y), 0.4, view_color, (73, 48, 0))

    def draw_posture_guide(self, frame: np.ndarray, view_angle: ViewAngle = ViewAngle.FRONT):
        """Draw adaptive posture guide based on current view angle."""
        h, w = frame.shape[:2]

        guide_size = self.config.GUIDE_SIZE
        guide_x = w - guide_size - self.config.GUIDE_MARGIN
        guide_y = h - guide_size - self.config.GUIDE_MARGIN

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (guide_x, guide_y), (guide_x + guide_size, guide_y + guide_size),
                      (50, 50, 50), cv2.FILLED)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        cv2.rectangle(frame, (guide_x, guide_y), (guide_x + guide_size, guide_y + guide_size),
                      (0, 255, 0), 2)

        # Title
        title = f"Ideal Posture ({view_angle.value.replace('_', ' ').title()} View)"
        self._draw_text_with_background(frame, title, (guide_x + 10, guide_y + 20), 0.45, (0, 255, 0))

        # Draw ideal posture figure based on view angle
        center_x = guide_x + guide_size // 2
        center_y = guide_y + 60

        if view_angle == ViewAngle.FRONT:
            self._draw_front_view_guide(frame, center_x, center_y)
        elif view_angle in [ViewAngle.SIDE_LEFT, ViewAngle.SIDE_RIGHT]:
            self._draw_side_view_guide(frame, center_x, center_y)
        else:
            self._draw_basic_guide(frame, center_x, center_y)

        # Key points
        points_y = guide_y + guide_size - 60
        key_points = self._get_key_points_for_view(view_angle)

        for i, point in enumerate(key_points[:4]):  # Show top 4 points
            point_y = points_y + i * 12
            self._draw_text_with_background(frame, f"- {point}", (guide_x + 10, point_y), 0.35, (200, 255, 200))

    def _draw_front_view_guide(self, frame: np.ndarray, center_x: int, center_y: int):
        """Draw front view ideal posture guide."""
        # Head
        cv2.circle(frame, (center_x, center_y), 12, (0, 255, 255), 2)

        # Neck
        cv2.line(frame, (center_x, center_y + 12), (center_x, center_y + 25), (0, 255, 0), 3)

        # Shoulders (level)
        shoulder_y = center_y + 25
        cv2.line(frame, (center_x - 20, shoulder_y), (center_x + 20, shoulder_y), (0, 255, 0), 4)

        # Arms
        cv2.line(frame, (center_x - 20, shoulder_y), (center_x - 20, shoulder_y + 30), (0, 255, 0), 2)
        cv2.line(frame, (center_x + 20, shoulder_y), (center_x + 20, shoulder_y + 30), (0, 255, 0), 2)

        # Spine (straight)
        cv2.line(frame, (center_x, shoulder_y), (center_x, center_y + 70), (0, 255, 0), 4)

        # Hips
        hip_y = center_y + 70
        cv2.line(frame, (center_x - 15, hip_y), (center_x + 15, hip_y), (0, 255, 0), 4)

    def _draw_side_view_guide(self, frame: np.ndarray, center_x: int, center_y: int):
        """Draw side view ideal posture guide."""
        # Head
        cv2.circle(frame, (center_x, center_y), 12, (0, 255, 255), 2)

        # Neck (slight curve)
        cv2.line(frame, (center_x + 2, center_y + 12), (center_x, center_y + 25), (0, 255, 0), 3)

        # Shoulder
        shoulder_point = (center_x, center_y + 25)
        cv2.circle(frame, shoulder_point, 5, (0, 255, 0), cv2.FILLED)

        # Spine (natural curve)
        spine_points = [(center_x, center_y + 25), (center_x - 3, center_y + 45),
                        (center_x + 2, center_y + 65), (center_x, center_y + 70)]
        for i in range(len(spine_points) - 1):
            cv2.line(frame, spine_points[i], spine_points[i + 1], (0, 255, 0), 4)

        # Hip
        cv2.circle(frame, (center_x, center_y + 70), 5, (0, 255, 0), cv2.FILLED)

        # Arm
        cv2.line(frame, (center_x, center_y + 25), (center_x + 15, center_y + 45), (0, 255, 0), 2)

    def _draw_basic_guide(self, frame: np.ndarray, center_x: int, center_y: int):
        """Draw basic posture guide for unclear views."""
        cv2.putText(frame, "Position yourself", (center_x - 40, center_y),
                    self.font, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, "facing the camera", (center_x - 45, center_y + 20),
                    self.font, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, "for better analysis", (center_x - 45, center_y + 40),
                    self.font, 0.4, (255, 255, 0), 1)

    def _get_key_points_for_view(self, view_angle: ViewAngle) -> List[str]:
        """Get key posture points for specific view angle."""
        if view_angle == ViewAngle.FRONT:
            return [
                "Head centered over shoulders",
                "Level shoulder line",
                "Straight spine alignment",
                "Even weight distribution",
                "Relaxed arms at sides"
            ]
        elif view_angle in [ViewAngle.SIDE_LEFT, ViewAngle.SIDE_RIGHT]:
            return [
                "Ears over shoulders",
                "Natural neck curve",
                "Chest open and lifted",
                "Shoulders over hips",
                "Natural spinal curves"
            ]
        else:
            return [
                "Face the camera",
                "Stand/sit naturally",
                "Maintain good posture",
                "Stay in camera view"
            ]

    def trigger_alert_flash(self):
        """Trigger visual alert flash."""
        self.last_alert_time = time.time()


# ================= ENHANCED APPLICATION =================
class PostureApp:
    """Enhanced main application with adaptive analysis and better UI."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging()

        try:
            self.detector = PoseDetector(config, self.logger)
            self.analyzer = PostureAnalyzer(config, self.logger)
            self.visualizer = PostureVisualizer(config, self.logger)
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise

        self.mode = AppMode.POSTURE_MONITOR

        # Performance tracking
        self.prev_time = time.time()
        self.fps = 0
        self.smooth_fps = 0

        # Setup alert callbacks
        self.analyzer.add_alert_callback(self._handle_posture_alert)

        self._create_session_directory()

    def _create_session_directory(self):
        """Create session directory if needed."""
        if self.config.SAVE_SESSION_JSON:
            try:
                Path(self.config.SESSION_DIR).mkdir(exist_ok=True)
            except Exception as e:
                self.logger.warning(f"Could not create session directory: {e}")

    def _draw_critical_posture_warning(self, frame: np.ndarray, posture_analysis: Dict):
        """Draw critical posture warning overlay for very poor posture."""
        h, w = frame.shape[:2]

        # Pulsing warning overlay
        pulse = 0.3 + 0.4 * math.sin(time.time() * 8)  # Fast pulsing

        # Semi-transparent red overlay using palette red
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (40, 40, 214), cv2.FILLED)  # Red from palette
        cv2.addWeighted(overlay, pulse * 0.15, frame, 1 - pulse * 0.15, 0, frame)

        # Central warning message
        warning_w, warning_h = 400, 150
        warning_x = (w - warning_w) // 2
        warning_y = (h - warning_h) // 2

        # Warning background using navy from palette
        warning_overlay = frame.copy()
        cv2.rectangle(warning_overlay, (warning_x, warning_y),
                      (warning_x + warning_w, warning_y + warning_h), (73, 48, 0), cv2.FILLED)
        cv2.addWeighted(warning_overlay, 0.9, frame, 0.1, 0, frame)

        # Warning border using red from palette
        cv2.rectangle(frame, (warning_x, warning_y), (warning_x + warning_w, warning_y + warning_h),
                      (40, 40, 214), 4)

        # Warning text using palette colors
        warning_texts = [
            ("[CRITICAL POSTURE ALERT]", 0.8, (183, 226, 234)),  # Cream text
            ("IMMEDIATE CORRECTION NEEDED", 0.6, (73, 191, 252)),  # Yellow text
            ("Take a posture break now!", 0.5, (0, 127, 247))  # Orange text
        ]

        text_y = warning_y + 40
        for text, scale, color in warning_texts:
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
            text_x = warning_x + (warning_w - text_size[0]) // 2
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
            text_y += int(30 * scale) + 10

    def _handle_posture_alert(self, alert_type: AlertType, message: str):
        """Handle posture alerts with visual feedback."""
        self.logger.info(f"Posture Alert - {alert_type.value}: {message}")

        # Trigger visual alert
        if self.config.ENABLE_VISUAL_ALERTS:
            self.visualizer.trigger_alert_flash()

    def reset_session(self):
        """Reset current session data."""
        try:
            self.analyzer = PostureAnalyzer(self.config, self.logger)
            self.analyzer.add_alert_callback(self._handle_posture_alert)
            self.logger.info("Session reset successfully")
        except Exception as e:
            self.logger.error(f"Failed to reset session: {e}")

    def save_session(self):
        """Save session data to file."""
        if not self.config.SAVE_SESSION_JSON:
            return

        try:
            data = dict(self.analyzer.posture_session_data)
            data['end_time'] = time.time()
            data['duration_sec'] = data['end_time'] - data['start_time']

            # Convert deque to list for JSON serialization
            data['posture_scores'] = list(data['posture_scores'])
            data['alert_history'] = list(data['alert_history'])

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"posture_session_{timestamp}.json"
            filepath = Path(self.config.SESSION_DIR) / filename

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            self.logger.info(f"Session saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save session: {e}")

    @staticmethod
    def landmarks_to_dict(landmark_list: List[Tuple[int, int, int]]) -> Dict[int, Tuple[int, int]]:
        """Convert landmark list to dictionary."""
        return {idx: (x, y) for idx, x, y in landmark_list}

    def update_fps(self):
        """Update FPS calculations."""
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time

        if dt > 0:
            self.fps = 1.0 / dt
            if self.smooth_fps == 0:
                self.smooth_fps = self.fps
            else:
                alpha = self.config.FPS_SMOOTHING_FACTOR
                self.smooth_fps = alpha * self.smooth_fps + (1 - alpha) * self.fps

    def run(self, camera_index: int = 0):
        """Main application loop with enhanced error handling."""
        cap = None
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera {camera_index}")

            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.WINDOW_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.WINDOW_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, self.config.FPS_TARGET)

            # Set window properties
            cv2.namedWindow("Posture Monitor", cv2.WINDOW_AUTOSIZE)

            self.logger.info("Starting enhanced posture monitoring application")
            self.logger.info("Controls: Q=Quit, C=Calibrate, R=Reset, S=Save")

            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read camera frame")
                    break

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # Process pose detection
                landmarks_list, detection_success = self.detector.process(frame)
                landmark_points = self.landmarks_to_dict(landmarks_list)
                pose_stable = self.detector.is_pose_stable()
                view_angle = self.detector.get_view_angle()

                # Analyze posture with view angle adaptation
                posture_analysis = self.analyzer.analyze_posture(
                    landmark_points, pose_stable, view_angle
                )

                # Draw visualizations
                if detection_success:
                    self.visualizer.draw_adaptive_skeleton(frame, posture_analysis, landmark_points)

                # Always draw HUD and guides
                self.update_fps()
                self.visualizer.draw_comprehensive_hud(
                    frame, posture_analysis, self.smooth_fps, self.detector.detection_fps
                )

                # Draw view angle indicator
                confidence = posture_analysis.get('confidence', 0.0)
                self.visualizer.draw_view_angle_indicator(frame, view_angle, confidence)

                # CRITICAL POSTURE WARNING OVERLAY
                if posture_analysis.get('state') == PostureState.VERY_POOR:
                    self._draw_critical_posture_warning(frame, posture_analysis)

                # Display frame
                cv2.imshow("Posture Monitor", frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # Q or ESC
                    self.logger.info("Shutting down application")
                    break
                elif key == ord('r'):
                    self.reset_session()
                elif key == ord('s'):  # Save session manually
                    self.save_session()
                elif key == ord('c'):
                    self.logger.info("Calibration feature coming soon")

        except Exception as e:
            self.logger.error(f"Application error: {e}")
        finally:
            # Cleanup
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            self.save_session()
            self.logger.info("Application shutdown complete")


# ================= MAIN ENTRY POINT =================
def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(description="Strict Posture Monitoring System")

    parser.add_argument('--camera', '-c', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging level')
    parser.add_argument('--window-width', type=int, default=1280,
                        help='Window width')
    parser.add_argument('--window-height', type=int, default=720,
                        help='Window height')

    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    try:
        # Create configuration
        config = Config()
        config.WINDOW_WIDTH = args.window_width
        config.WINDOW_HEIGHT = args.window_height

        # Create and run application
        app = PostureApp(config)
        app.run(args.camera)

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()