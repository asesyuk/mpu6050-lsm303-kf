#!/usr/bin/env python3
"""
Extended Kalman Filter for Inertial Navigation System.

This module implements a 15-state EKF for attitude, velocity, and position estimation:
State vector: [q0, q1, q2, q3, vN, vE, vD, pN, pE, pD, bgx, bgy, bgz, bax, bay, baz]

- q0-q3: Unit quaternion (attitude)
- vN,vE,vD: Velocity in NED frame [m/s]
- pN,pE,pD: Position in NED frame [m]
- bgx,bgy,bgz: Gyroscope biases [rad/s]
- bax,bay,baz: Accelerometer biases [m/s²]

Author: Embedded Systems Engineer
"""

import time
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy.linalg import block_diag
from dataclasses import dataclass

# Physical constants
GRAVITY = 9.80665  # Standard gravity [m/s²]
EARTH_ROTATION_RATE = 7.2921159e-5  # Earth rotation rate [rad/s]
EARTH_RADIUS = 6378137.0  # Earth radius [m]

# Default noise parameters
DEFAULT_GYRO_NOISE_DENSITY = 1e-4  # [rad/s/√Hz]
DEFAULT_ACCEL_NOISE_DENSITY = 1e-3  # [m/s²/√Hz]
DEFAULT_GYRO_BIAS_STABILITY = 1e-6  # [rad/s/√Hz]
DEFAULT_ACCEL_BIAS_STABILITY = 1e-5  # [m/s²/√Hz]

# Measurement noise
DEFAULT_ACCEL_GRAVITY_NOISE = 0.1  # [m/s²]
DEFAULT_MAG_NOISE = 1e-6  # [Tesla]


@dataclass
class EKFState:
    """Container for EKF state and covariance."""
    
    # State vector [15x1]
    quaternion: np.ndarray        # [4x1] q = [qw, qx, qy, qz]
    velocity_ned: np.ndarray      # [3x1] velocity in NED frame [m/s]
    position_ned: np.ndarray      # [3x1] position in NED frame [m]
    gyro_bias: np.ndarray         # [3x1] gyroscope bias [rad/s]
    accel_bias: np.ndarray        # [3x1] accelerometer bias [m/s²]
    
    # Covariance matrix [15x15]
    covariance: np.ndarray
    
    # Timestamp
    timestamp: float
    
    def __post_init__(self):
        """Ensure quaternion is normalized."""
        self.quaternion = self.quaternion / np.linalg.norm(self.quaternion)
    
    def to_vector(self) -> np.ndarray:
        """Convert state to 15x1 vector."""
        return np.concatenate([
            self.quaternion,
            self.velocity_ned,
            self.position_ned,
            self.gyro_bias,
            self.accel_bias
        ])
    
    @classmethod
    def from_vector(cls, state_vector: np.ndarray, covariance: np.ndarray, timestamp: float):
        """Create EKFState from state vector."""
        return cls(
            quaternion=state_vector[0:4],
            velocity_ned=state_vector[4:7],
            position_ned=state_vector[7:10],
            gyro_bias=state_vector[10:13],
            accel_bias=state_vector[13:16],
            covariance=covariance,
            timestamp=timestamp
        )


class QuaternionMath:
    """Quaternion mathematics utilities."""
    
    @staticmethod
    def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions: q1 * q2."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @staticmethod
    def conjugate(q: np.ndarray) -> np.ndarray:
        """Quaternion conjugate."""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    @staticmethod
    def to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        q = q / np.linalg.norm(q)  # Normalize
        w, x, y, z = q
        
        return np.array([
            [1-2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1-2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x + y*y)]
        ])
    
    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """Create quaternion from axis-angle representation."""
        axis = axis / np.linalg.norm(axis)
        half_angle = angle / 2.0
        sin_half = np.sin(half_angle)
        
        return np.array([
            np.cos(half_angle),
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half
        ])
    
    @staticmethod
    def to_euler(q: np.ndarray) -> np.ndarray:
        """Convert quaternion to Euler angles (roll, pitch, yaw) in radians."""
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])


class InertialEKF:
    """
    Extended Kalman Filter for inertial navigation.
    
    Implements a 15-state EKF with quaternion-based attitude representation.
    """
    
    def __init__(self, initial_state: Optional[EKFState] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize state
        if initial_state is None:
            self.state = self._create_initial_state()
        else:
            self.state = initial_state
        
        # Process noise parameters
        self.gyro_noise_density = DEFAULT_GYRO_NOISE_DENSITY
        self.accel_noise_density = DEFAULT_ACCEL_NOISE_DENSITY
        self.gyro_bias_stability = DEFAULT_GYRO_BIAS_STABILITY
        self.accel_bias_stability = DEFAULT_ACCEL_BIAS_STABILITY
        
        # Measurement noise parameters
        self.accel_gravity_noise = DEFAULT_ACCEL_GRAVITY_NOISE
        self.mag_noise = DEFAULT_MAG_NOISE
        
        # ZUPT parameters
        self.zupt_enabled = True
        self.zupt_velocity_threshold = 0.1  # m/s
        self.zupt_acceleration_threshold = 0.5  # m/s²
        self.zupt_angular_rate_threshold = 0.05  # rad/s
        self.zupt_noise = 0.01  # m/s
        
        # Statistics
        self.prediction_count = 0
        self.update_count = 0
        self.zupt_count = 0
        
        self.logger.info("InertialEKF initialized")
    
    def _create_initial_state(self) -> EKFState:
        """Create initial state with reasonable defaults."""
        # Initial state vector
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        velocity_ned = np.zeros(3)
        position_ned = np.zeros(3)
        gyro_bias = np.zeros(3)
        accel_bias = np.zeros(3)
        
        # Initial covariance matrix
        P = np.zeros((15, 15))
        
        # Attitude uncertainty (roll, pitch, yaw) - 10 degrees std
        P[0:3, 0:3] = np.eye(3) * (10 * np.pi / 180)**2
        
        # Velocity uncertainty - 1 m/s std
        P[3:6, 3:6] = np.eye(3) * 1.0**2
        
        # Position uncertainty - 10 m std
        P[6:9, 6:9] = np.eye(3) * 10.0**2
        
        # Gyro bias uncertainty - 0.1 deg/s std
        P[9:12, 9:12] = np.eye(3) * (0.1 * np.pi / 180)**2
        
        # Accel bias uncertainty - 0.1 m/s² std
        P[12:15, 12:15] = np.eye(3) * 0.1**2
        
        return EKFState(
            quaternion=quaternion,
            velocity_ned=velocity_ned,
            position_ned=position_ned,
            gyro_bias=gyro_bias,
            accel_bias=accel_bias,
            covariance=P,
            timestamp=time.time()
        )
    
    def predict(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None:
        """
        EKF prediction step using IMU measurements.
        
        Args:
            gyro: Gyroscope measurements [rad/s] in body frame
            accel: Accelerometer measurements [m/s²] in body frame
            dt: Time step [s]
        """
        if dt <= 0 or dt > 1.0:  # Sanity check
            self.logger.warning(f"Invalid dt: {dt}, skipping prediction")
            return
        
        # Current state
        q = self.state.quaternion
        v = self.state.velocity_ned
        p = self.state.position_ned
        bg = self.state.gyro_bias
        ba = self.state.accel_bias
        
        # Corrected measurements
        gyro_corrected = gyro - bg
        accel_corrected = accel - ba
        
        # Rotation matrix from body to NED
        R_bn = QuaternionMath.to_rotation_matrix(q)
        
        # State propagation
        # 1. Quaternion kinematics
        omega = gyro_corrected
        omega_norm = np.linalg.norm(omega)
        
        if omega_norm > 1e-8:
            # Finite rotation
            dq = QuaternionMath.from_axis_angle(omega, omega_norm * dt)
            q_new = QuaternionMath.multiply(q, dq)
        else:
            # Small angle approximation
            q_new = q + 0.5 * dt * np.array([
                -omega[0]*q[1] - omega[1]*q[2] - omega[2]*q[3],
                omega[0]*q[0] + omega[2]*q[2] - omega[1]*q[3],
                omega[1]*q[0] - omega[2]*q[1] + omega[0]*q[3],
                omega[2]*q[0] + omega[1]*q[1] - omega[0]*q[2]
            ])
        
        q_new = q_new / np.linalg.norm(q_new)  # Normalize
        
        # 2. Velocity dynamics
        accel_ned = R_bn @ accel_corrected
        gravity_ned = np.array([0, 0, GRAVITY])
        v_new = v + dt * (accel_ned - gravity_ned)
        
        # 3. Position dynamics
        p_new = p + dt * v + 0.5 * dt**2 * (accel_ned - gravity_ned)
        
        # 4. Bias dynamics (random walk)
        bg_new = bg  # Constant bias model
        ba_new = ba  # Constant bias model
        
        # Update state
        self.state.quaternion = q_new
        self.state.velocity_ned = v_new
        self.state.position_ned = p_new
        self.state.gyro_bias = bg_new
        self.state.accel_bias = ba_new
        self.state.timestamp += dt
        
        # Covariance propagation
        F = self._compute_state_transition_matrix(gyro_corrected, accel_corrected, dt)
        G = self._compute_noise_matrix(dt)
        Q = self._compute_process_noise_matrix(dt)
        
        P_new = F @ self.state.covariance @ F.T + G @ Q @ G.T
        
        # Ensure positive definite covariance
        self.state.covariance = 0.5 * (P_new + P_new.T)
        
        self.prediction_count += 1
    
    def _compute_state_transition_matrix(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> np.ndarray:
        """Compute the state transition matrix F."""
        F = np.eye(15)
        
        q = self.state.quaternion
        R_bn = QuaternionMath.to_rotation_matrix(q)
        
        # Skew-symmetric matrix for cross product
        def skew(v):
            return np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
        
        # Fill in the non-trivial blocks
        # Attitude error propagation
        F[0:3, 0:3] = np.eye(3) - skew(gyro) * dt
        F[0:3, 9:12] = -np.eye(3) * dt
        
        # Velocity propagation
        F[3:6, 0:3] = -skew(R_bn @ accel) * dt
        F[3:6, 12:15] = -R_bn * dt
        
        # Position propagation
        F[6:9, 0:3] = -0.5 * skew(R_bn @ accel) * dt**2
        F[6:9, 3:6] = np.eye(3) * dt
        F[6:9, 12:15] = -0.5 * R_bn * dt**2
        
        return F
    
    def _compute_noise_matrix(self, dt: float) -> np.ndarray:
        """Compute the noise input matrix G."""
        G = np.zeros((15, 12))
        
        q = self.state.quaternion
        R_bn = QuaternionMath.to_rotation_matrix(q)
        
        # Gyroscope noise affects attitude
        G[0:3, 0:3] = -np.eye(3) * dt
        
        # Accelerometer noise affects velocity and position
        G[3:6, 3:6] = -R_bn * dt
        G[6:9, 3:6] = -0.5 * R_bn * dt**2
        
        # Bias noise affects biases directly
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)
        
        return G
    
    def _compute_process_noise_matrix(self, dt: float) -> np.ndarray:
        """Compute the process noise covariance matrix Q."""
        # White noise PSD
        gyro_psd = self.gyro_noise_density**2
        accel_psd = self.accel_noise_density**2
        gyro_bias_psd = self.gyro_bias_stability**2
        accel_bias_psd = self.accel_bias_stability**2
        
        Q = block_diag(
            np.eye(3) * gyro_psd,      # Gyroscope white noise
            np.eye(3) * accel_psd,     # Accelerometer white noise
            np.eye(3) * gyro_bias_psd * dt,   # Gyro bias random walk
            np.eye(3) * accel_bias_psd * dt   # Accel bias random walk
        )
        
        return Q
    
    def update_gravity(self, accel_measurement: np.ndarray) -> None:
        """
        Update using gravity vector observation.
        
        Args:
            accel_measurement: Accelerometer measurement [m/s²] in body frame
        """
        # Expected gravity in body frame
        q = self.state.quaternion
        R_nb = QuaternionMath.to_rotation_matrix(q).T  # NED to body
        gravity_ned = np.array([0, 0, GRAVITY])
        expected_accel = R_nb @ gravity_ned
        
        # Innovation
        innovation = accel_measurement - expected_accel
        
        # Measurement matrix (linearized)
        H = np.zeros((3, 15))
        
        # ∂h/∂θ = -R_nb * [g]×
        def skew(v):
            return np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
        
        H[0:3, 0:3] = -R_nb @ skew(gravity_ned)
        H[0:3, 12:15] = -np.eye(3)  # Accelerometer bias
        
        # Measurement noise
        R = np.eye(3) * self.accel_gravity_noise**2
        
        # Kalman update
        self._kalman_update(innovation, H, R)
        self.update_count += 1
    
    def update_magnetometer(self, mag_measurement: np.ndarray, mag_reference: np.ndarray) -> None:
        """
        Update using magnetometer measurement.
        
        Args:
            mag_measurement: Magnetometer measurement [Tesla] in body frame
            mag_reference: Reference magnetic field [Tesla] in NED frame
        """
        # Expected magnetic field in body frame
        q = self.state.quaternion
        R_nb = QuaternionMath.to_rotation_matrix(q).T  # NED to body
        expected_mag = R_nb @ mag_reference
        
        # Innovation
        innovation = mag_measurement - expected_mag
        
        # Measurement matrix
        H = np.zeros((3, 15))
        
        # ∂h/∂θ = -R_nb * [mag_ref]×
        def skew(v):
            return np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
        
        H[0:3, 0:3] = -R_nb @ skew(mag_reference)
        
        # Measurement noise
        R = np.eye(3) * self.mag_noise**2
        
        # Kalman update
        self._kalman_update(innovation, H, R)
        self.update_count += 1
    
    def update_zupt(self, accel: np.ndarray, gyro: np.ndarray) -> bool:
        """
        Zero Velocity Update (ZUPT) when stationary.
        
        Args:
            accel: Accelerometer measurement [m/s²]
            gyro: Gyroscope measurement [rad/s]
            
        Returns:
            True if ZUPT was applied
        """
        if not self.zupt_enabled:
            return False
        
        # Check if vehicle is stationary
        accel_magnitude = np.linalg.norm(accel)
        gyro_magnitude = np.linalg.norm(gyro)
        velocity_magnitude = np.linalg.norm(self.state.velocity_ned)
        
        # Stationary detection criteria
        accel_stationary = abs(accel_magnitude - GRAVITY) < self.zupt_acceleration_threshold
        gyro_stationary = gyro_magnitude < self.zupt_angular_rate_threshold
        velocity_stationary = velocity_magnitude < self.zupt_velocity_threshold
        
        if accel_stationary and gyro_stationary:
            # Apply zero velocity constraint
            innovation = -self.state.velocity_ned  # Innovation = 0 - v
            
            # Measurement matrix
            H = np.zeros((3, 15))
            H[0:3, 3:6] = np.eye(3)  # ∂v/∂v = I
            
            # Measurement noise
            R = np.eye(3) * self.zupt_noise**2
            
            # Kalman update
            self._kalman_update(innovation, H, R)
            self.zupt_count += 1
            
            return True
        
        return False
    
    def _kalman_update(self, innovation: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        """
        Generic Kalman filter update step.
        
        Args:
            innovation: Measurement innovation
            H: Measurement matrix
            R: Measurement noise covariance
        """
        P = self.state.covariance
        
        # Innovation covariance
        S = H @ P @ H.T + R
        
        # Kalman gain
        try:
            K = P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.logger.warning("Singular innovation covariance, skipping update")
            return
        
        # State update (error state formulation)
        delta_x = K @ innovation
        
        # Apply state corrections
        self._apply_state_corrections(delta_x)
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(15) - K @ H
        P_new = I_KH @ P @ I_KH.T + K @ R @ K.T
        
        # Ensure positive definite
        self.state.covariance = 0.5 * (P_new + P_new.T)
    
    def _apply_state_corrections(self, delta_x: np.ndarray) -> None:
        """Apply error state corrections to the nominal state."""
        # Attitude correction (quaternion)
        delta_theta = delta_x[0:3]
        if np.linalg.norm(delta_theta) > 1e-8:
            delta_q = QuaternionMath.from_axis_angle(delta_theta, np.linalg.norm(delta_theta))
            self.state.quaternion = QuaternionMath.multiply(self.state.quaternion, delta_q)
            self.state.quaternion = self.state.quaternion / np.linalg.norm(self.state.quaternion)
        
        # Other state corrections
        self.state.velocity_ned += delta_x[3:6]
        self.state.position_ned += delta_x[6:9]
        self.state.gyro_bias += delta_x[9:12]
        self.state.accel_bias += delta_x[12:15]
    
    def get_attitude_euler(self) -> np.ndarray:
        """Get attitude as Euler angles (roll, pitch, yaw) in radians."""
        return QuaternionMath.to_euler(self.state.quaternion)
    
    def get_rotation_matrix(self) -> np.ndarray:
        """Get rotation matrix from body to NED frame."""
        return QuaternionMath.to_rotation_matrix(self.state.quaternion)
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get current state as dictionary for logging/networking."""
        euler = self.get_attitude_euler()
        
        return {
            'timestamp': self.state.timestamp,
            'quaternion': self.state.quaternion.tolist(),
            'euler_rad': euler.tolist(),
            'euler_deg': np.degrees(euler).tolist(),
            'velocity_ned': self.state.velocity_ned.tolist(),
            'position_ned': self.state.position_ned.tolist(),
            'gyro_bias': self.state.gyro_bias.tolist(),
            'accel_bias': self.state.accel_bias.tolist(),
            'covariance_trace': np.trace(self.state.covariance),
            'statistics': {
                'predictions': self.prediction_count,
                'updates': self.update_count,
                'zupts': self.zupt_count
            }
        }
    
    def reset_statistics(self) -> None:
        """Reset performance statistics."""
        self.prediction_count = 0
        self.update_count = 0
        self.zupt_count = 0
    
    def set_noise_parameters(self, gyro_noise: float = None, accel_noise: float = None,
                           gyro_bias_stability: float = None, accel_bias_stability: float = None) -> None:
        """Set process noise parameters."""
        if gyro_noise is not None:
            self.gyro_noise_density = gyro_noise
        if accel_noise is not None:
            self.accel_noise_density = accel_noise
        if gyro_bias_stability is not None:
            self.gyro_bias_stability = gyro_bias_stability
        if accel_bias_stability is not None:
            self.accel_bias_stability = accel_bias_stability
        
        self.logger.info(f"Noise parameters updated: gyro={self.gyro_noise_density:.2e}, "
                        f"accel={self.accel_noise_density:.2e}")


class MotionSimulator:
    """Simple motion simulator for testing EKF consistency."""
    
    def __init__(self, dt: float = 0.005):
        self.dt = dt
        self.time = 0.0
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.attitude = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        self.angular_velocity = np.zeros(3)
        
        # Noise parameters
        self.gyro_noise_std = 1e-4
        self.accel_noise_std = 1e-3
        self.mag_noise_std = 1e-6
    
    def step(self) -> Dict[str, np.ndarray]:
        """Simulate one time step and return sensor measurements."""
        # Simple circular motion for testing
        radius = 10.0
        frequency = 0.1  # Hz
        
        # Update true state
        self.time += self.dt
        angle = 2 * np.pi * frequency * self.time
        
        self.position = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            0.0
        ])
        
        self.velocity = np.array([
            -radius * 2 * np.pi * frequency * np.sin(angle),
            radius * 2 * np.pi * frequency * np.cos(angle),
            0.0
        ])
        
        # Generate sensor measurements
        R_bn = QuaternionMath.to_rotation_matrix(self.attitude)
        R_nb = R_bn.T
        
        # Accelerometer: specific force in body frame
        acceleration_ned = np.array([
            -radius * (2 * np.pi * frequency)**2 * np.cos(angle),
            -radius * (2 * np.pi * frequency)**2 * np.sin(angle),
            0.0
        ])
        gravity_ned = np.array([0, 0, GRAVITY])
        specific_force_ned = acceleration_ned + gravity_ned
        accel_body = R_nb @ specific_force_ned
        accel_measured = accel_body + np.random.normal(0, self.accel_noise_std, 3)
        
        # Gyroscope: angular velocity in body frame
        angular_velocity_ned = np.array([0, 0, 2 * np.pi * frequency])
        gyro_body = R_nb @ angular_velocity_ned
        gyro_measured = gyro_body + np.random.normal(0, self.gyro_noise_std, 3)
        
        # Magnetometer: Earth's field in body frame
        mag_ned = np.array([20e-6, 0, 45e-6])  # Typical field
        mag_body = R_nb @ mag_ned
        mag_measured = mag_body + np.random.normal(0, self.mag_noise_std, 3)
        
        return {
            'timestamp': self.time,
            'dt': self.dt,
            'gyroscope': gyro_measured,
            'accelerometer_mpu': accel_measured,
            'accelerometer_lsm': accel_measured,  # Same for simulation
            'magnetometer': mag_measured,
            'temperature': 25.0,
            'truth': {
                'position': self.position.copy(),
                'velocity': self.velocity.copy(),
                'attitude': self.attitude.copy()
            }
        }


if __name__ == "__main__":
    """Test EKF with motion simulator."""
    import matplotlib.pyplot as plt
    
    # Create EKF and simulator
    ekf = InertialEKF()
    simulator = MotionSimulator(dt=0.005)
    
    # Reference magnetic field (typical values)
    mag_reference = np.array([20e-6, 0, 45e-6])  # Tesla, NED frame
    
    # Simulation parameters
    duration = 60.0  # seconds
    steps = int(duration / simulator.dt)
    
    # Storage for results
    results = {
        'time': [],
        'position_est': [],
        'position_true': [],
        'velocity_est': [],
        'velocity_true': [],
        'attitude_est': [],
        'attitude_true': []
    }
    
    print(f"Running EKF simulation for {duration} seconds...")
    
    for i in range(steps):
        # Generate sensor data
        sensor_data = simulator.step()
        
        # EKF prediction
        ekf.predict(
            sensor_data['gyroscope'],
            sensor_data['accelerometer_mpu'],
            sensor_data['dt']
        )
        
        # EKF updates (every 10th step to simulate lower rate)
        if i % 10 == 0:
            ekf.update_gravity(sensor_data['accelerometer_mpu'])
            ekf.update_magnetometer(sensor_data['magnetometer'], mag_reference)
        
        # ZUPT (when stationary - not in this circular motion)
        ekf.update_zupt(sensor_data['accelerometer_mpu'], sensor_data['gyroscope'])
        
        # Store results
        if i % 20 == 0:  # Reduce storage frequency
            state = ekf.get_state_dict()
            results['time'].append(sensor_data['timestamp'])
            results['position_est'].append(state['position_ned'])
            results['position_true'].append(sensor_data['truth']['position'])
            results['velocity_est'].append(state['velocity_ned'])
            results['velocity_true'].append(sensor_data['truth']['velocity'])
            results['attitude_est'].append(state['euler_deg'])
    
    # Convert to numpy arrays
    for key in ['position_est', 'position_true', 'velocity_est', 'velocity_true', 'attitude_est']:
        results[key] = np.array(results[key])
    
    # Print final statistics
    final_state = ekf.get_state_dict()
    print(f"\nEKF Performance:")
    print(f"  Predictions: {final_state['statistics']['predictions']}")
    print(f"  Updates: {final_state['statistics']['updates']}")
    print(f"  ZUPTs: {final_state['statistics']['zupts']}")
    print(f"  Final position error: {np.linalg.norm(results['position_est'][-1] - results['position_true'][-1]):.3f} m")
    
    # Simple plots if matplotlib is available
    try:
        plt.figure(figsize=(12, 8))
        
        # Position plot
        plt.subplot(2, 2, 1)
        plt.plot(results['position_true'][:, 0], results['position_true'][:, 1], 'g-', label='True')
        plt.plot(results['position_est'][:, 0], results['position_est'][:, 1], 'r--', label='Estimated')
        plt.xlabel('North [m]')
        plt.ylabel('East [m]')
        plt.title('Trajectory')
        plt.legend()
        plt.grid(True)
        
        # Position error
        plt.subplot(2, 2, 2)
        pos_error = np.linalg.norm(results['position_est'] - results['position_true'], axis=1)
        plt.plot(results['time'], pos_error)
        plt.xlabel('Time [s]')
        plt.ylabel('Position Error [m]')
        plt.title('Position Error')
        plt.grid(True)
        
        # Velocity
        plt.subplot(2, 2, 3)
        vel_error = np.linalg.norm(results['velocity_est'] - results['velocity_true'], axis=1)
        plt.plot(results['time'], vel_error)
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity Error [m/s]')
        plt.title('Velocity Error')
        plt.grid(True)
        
        # Attitude (yaw only)
        plt.subplot(2, 2, 4)
        plt.plot(results['time'], results['attitude_est'][:, 2])
        plt.xlabel('Time [s]')
        plt.ylabel('Yaw [deg]')
        plt.title('Estimated Yaw')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('ekf_test_results.png', dpi=150)
        print("Test results saved to ekf_test_results.png")
        
    except ImportError:
        print("Matplotlib not available, skipping plots")
    
    print("EKF test completed successfully!") 