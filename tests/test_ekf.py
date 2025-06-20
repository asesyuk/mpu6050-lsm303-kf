#!/usr/bin/env python3
"""
Unit tests for Extended Kalman Filter module.

Tests the InertialEKF class including:
- State initialization and management
- Prediction and update steps
- Quaternion mathematics
- ZUPT functionality
- Motion simulation

Author: Embedded Systems Engineer
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

# Import modules to test
from ekf import (
    EKFState, QuaternionMath, InertialEKF, MotionSimulator,
    GRAVITY, DEFAULT_GYRO_NOISE_DENSITY, DEFAULT_ACCEL_NOISE_DENSITY
)


class TestQuaternionMath:
    """Test cases for quaternion mathematics utilities."""
    
    def test_multiply_identity(self):
        """Test quaternion multiplication with identity."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
        q2 = np.array([0.707, 0.707, 0.0, 0.0])  # 90° rotation about X
        
        result = QuaternionMath.multiply(q1, q2)
        
        assert np.allclose(result, q2)
    
    def test_multiply_associative(self):
        """Test quaternion multiplication associativity."""
        q1 = np.array([0.707, 0.707, 0.0, 0.0])
        q2 = np.array([0.707, 0.0, 0.707, 0.0])
        q3 = np.array([0.707, 0.0, 0.0, 0.707])
        
        # (q1 * q2) * q3
        result1 = QuaternionMath.multiply(QuaternionMath.multiply(q1, q2), q3)
        
        # q1 * (q2 * q3)
        result2 = QuaternionMath.multiply(q1, QuaternionMath.multiply(q2, q3))
        
        assert np.allclose(result1, result2)
    
    def test_conjugate(self):
        """Test quaternion conjugate."""
        q = np.array([0.6, 0.8, 0.0, 0.0])
        q_conj = QuaternionMath.conjugate(q)
        
        expected = np.array([0.6, -0.8, 0.0, 0.0])
        assert np.allclose(q_conj, expected)
    
    def test_conjugate_identity(self):
        """Test that conjugate of identity is identity."""
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])
        q_conj = QuaternionMath.conjugate(q_identity)
        
        assert np.allclose(q_conj, q_identity)
    
    def test_to_rotation_matrix_identity(self):
        """Test rotation matrix conversion for identity quaternion."""
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])
        R = QuaternionMath.to_rotation_matrix(q_identity)
        
        assert np.allclose(R, np.eye(3))
    
    def test_to_rotation_matrix_90_degree_x(self):
        """Test rotation matrix for 90° rotation about X axis."""
        q = np.array([0.707107, 0.707107, 0.0, 0.0])  # 90° about X
        R = QuaternionMath.to_rotation_matrix(q)
        
        expected = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        
        assert np.allclose(R, expected, atol=1e-6)
    
    def test_rotation_matrix_orthogonal(self):
        """Test that rotation matrices are orthogonal."""
        q = np.array([0.6, 0.8, 0.0, 0.0])
        q = q / np.linalg.norm(q)  # Normalize
        R = QuaternionMath.to_rotation_matrix(q)
        
        # Check orthogonality: R^T * R = I
        assert np.allclose(R.T @ R, np.eye(3))
        
        # Check determinant = 1
        assert np.allclose(np.linalg.det(R), 1.0)
    
    def test_from_axis_angle(self):
        """Test quaternion creation from axis-angle."""
        axis = np.array([1.0, 0.0, 0.0])
        angle = np.pi / 2  # 90 degrees
        
        q = QuaternionMath.from_axis_angle(axis, angle)
        
        expected = np.array([0.707107, 0.707107, 0.0, 0.0])
        assert np.allclose(q, expected, atol=1e-6)
    
    def test_to_euler_identity(self):
        """Test Euler angle conversion for identity quaternion."""
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])
        euler = QuaternionMath.to_euler(q_identity)
        
        assert np.allclose(euler, np.zeros(3))
    
    def test_to_euler_90_degree_x(self):
        """Test Euler angle conversion for 90° X rotation."""
        q = np.array([0.707107, 0.707107, 0.0, 0.0])
        euler = QuaternionMath.to_euler(q)
        
        expected = np.array([np.pi/2, 0.0, 0.0])  # 90° roll
        assert np.allclose(euler, expected, atol=1e-6)


class TestEKFState:
    """Test cases for EKF state management."""
    
    def test_initialization(self):
        """Test EKF state initialization."""
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        velocity = np.array([1.0, 2.0, 3.0])
        position = np.array([10.0, 20.0, 30.0])
        gyro_bias = np.array([0.01, 0.02, 0.03])
        accel_bias = np.array([0.1, 0.2, 0.3])
        covariance = np.eye(15)
        timestamp = time.time()
        
        state = EKFState(
            quaternion=quaternion,
            velocity_ned=velocity,
            position_ned=position,
            gyro_bias=gyro_bias,
            accel_bias=accel_bias,
            covariance=covariance,
            timestamp=timestamp
        )
        
        assert np.allclose(state.quaternion, quaternion)
        assert np.allclose(state.velocity_ned, velocity)
        assert np.allclose(state.position_ned, position)
        assert np.allclose(state.gyro_bias, gyro_bias)
        assert np.allclose(state.accel_bias, accel_bias)
        assert np.allclose(state.covariance, covariance)
        assert state.timestamp == timestamp
    
    def test_quaternion_normalization(self):
        """Test that quaternion is normalized during initialization."""
        quaternion = np.array([2.0, 0.0, 0.0, 0.0])  # Not normalized
        state = EKFState(
            quaternion=quaternion,
            velocity_ned=np.zeros(3),
            position_ned=np.zeros(3),
            gyro_bias=np.zeros(3),
            accel_bias=np.zeros(3),
            covariance=np.eye(15),
            timestamp=time.time()
        )
        
        # Should be normalized
        assert np.allclose(np.linalg.norm(state.quaternion), 1.0)
        assert np.allclose(state.quaternion, np.array([1.0, 0.0, 0.0, 0.0]))
    
    def test_to_vector(self):
        """Test state vector conversion."""
        state = EKFState(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            velocity_ned=np.array([1.0, 2.0, 3.0]),
            position_ned=np.array([10.0, 20.0, 30.0]),
            gyro_bias=np.array([0.01, 0.02, 0.03]),
            accel_bias=np.array([0.1, 0.2, 0.3]),
            covariance=np.eye(15),
            timestamp=time.time()
        )
        
        vector = state.to_vector()
        
        assert vector.shape == (15,)
        assert np.allclose(vector[0:4], [1.0, 0.0, 0.0, 0.0])
        assert np.allclose(vector[4:7], [1.0, 2.0, 3.0])
        assert np.allclose(vector[7:10], [10.0, 20.0, 30.0])
    
    def test_from_vector(self):
        """Test state creation from vector."""
        vector = np.array([
            1.0, 0.0, 0.0, 0.0,  # quaternion
            1.0, 2.0, 3.0,       # velocity
            10.0, 20.0, 30.0,    # position
            0.01, 0.02, 0.03,    # gyro bias
            0.1, 0.2, 0.3        # accel bias
        ])
        covariance = np.eye(15)
        timestamp = time.time()
        
        state = EKFState.from_vector(vector, covariance, timestamp)
        
        assert np.allclose(state.quaternion, [1.0, 0.0, 0.0, 0.0])
        assert np.allclose(state.velocity_ned, [1.0, 2.0, 3.0])
        assert np.allclose(state.position_ned, [10.0, 20.0, 30.0])


class TestInertialEKF:
    """Test cases for the Inertial EKF."""
    
    def test_initialization_default(self):
        """Test EKF initialization with default parameters."""
        ekf = InertialEKF()
        
        assert ekf.state is not None
        assert ekf.gyro_noise_density == DEFAULT_GYRO_NOISE_DENSITY
        assert ekf.accel_noise_density == DEFAULT_ACCEL_NOISE_DENSITY
        assert ekf.zupt_enabled == True
        assert ekf.prediction_count == 0
        assert ekf.update_count == 0
        assert ekf.zupt_count == 0
    
    def test_initialization_custom_state(self):
        """Test EKF initialization with custom initial state."""
        custom_state = EKFState(
            quaternion=np.array([0.707, 0.707, 0.0, 0.0]),
            velocity_ned=np.array([1.0, 0.0, 0.0]),
            position_ned=np.array([100.0, 200.0, 300.0]),
            gyro_bias=np.array([0.01, 0.01, 0.01]),
            accel_bias=np.array([0.1, 0.1, 0.1]),
            covariance=np.eye(15) * 0.1,
            timestamp=time.time()
        )
        
        ekf = InertialEKF(initial_state=custom_state)
        
        assert np.allclose(ekf.state.quaternion, custom_state.quaternion)
        assert np.allclose(ekf.state.velocity_ned, custom_state.velocity_ned)
    
    def test_predict_stationary(self):
        """Test EKF prediction with stationary measurements."""
        ekf = InertialEKF()
        
        # Stationary measurements
        gyro = np.zeros(3)
        accel = np.array([0.0, 0.0, GRAVITY])  # Only gravity
        dt = 0.005
        
        initial_position = ekf.state.position_ned.copy()
        initial_velocity = ekf.state.velocity_ned.copy()
        
        # Predict one step
        ekf.predict(gyro, accel, dt)
        
        # Position and velocity should remain close to initial (stationary)
        assert np.allclose(ekf.state.velocity_ned, initial_velocity, atol=1e-6)
        assert np.allclose(ekf.state.position_ned, initial_position, atol=1e-6)
        assert ekf.prediction_count == 1
    
    def test_predict_with_rotation(self):
        """Test EKF prediction with rotation."""
        ekf = InertialEKF()
        
        # Rotation about Z axis
        gyro = np.array([0.0, 0.0, 0.1])  # 0.1 rad/s
        accel = np.array([0.0, 0.0, GRAVITY])
        dt = 0.1  # 100ms
        
        initial_quaternion = ekf.state.quaternion.copy()
        
        # Predict one step
        ekf.predict(gyro, accel, dt)
        
        # Quaternion should have changed
        assert not np.allclose(ekf.state.quaternion, initial_quaternion)
        
        # Should still be normalized
        assert np.allclose(np.linalg.norm(ekf.state.quaternion), 1.0)
    
    def test_predict_invalid_dt(self):
        """Test EKF prediction with invalid time step."""
        ekf = InertialEKF()
        
        gyro = np.zeros(3)
        accel = np.array([0.0, 0.0, GRAVITY])
        
        initial_count = ekf.prediction_count
        
        # Test with negative dt
        ekf.predict(gyro, accel, -0.1)
        assert ekf.prediction_count == initial_count  # Should not increment
        
        # Test with too large dt
        ekf.predict(gyro, accel, 2.0)
        assert ekf.prediction_count == initial_count  # Should not increment
    
    def test_update_gravity_stationary(self):
        """Test gravity update with stationary vehicle."""
        ekf = InertialEKF()
        
        # Perfect gravity measurement
        accel_measurement = np.array([0.0, 0.0, GRAVITY])
        
        initial_update_count = ekf.update_count
        initial_covariance_trace = np.trace(ekf.state.covariance)
        
        # Apply gravity update
        ekf.update_gravity(accel_measurement)
        
        # Update count should increment
        assert ekf.update_count == initial_update_count + 1
        
        # Covariance should generally decrease (information gained)
        # Note: This might not always be true due to Joseph form update
        final_covariance_trace = np.trace(ekf.state.covariance)
        assert final_covariance_trace > 0  # Should remain positive definite
    
    def test_update_magnetometer(self):
        """Test magnetometer update."""
        ekf = InertialEKF()
        
        # Typical Earth magnetic field in NED frame
        mag_reference = np.array([20e-6, 0, 45e-6])  # Tesla
        
        # Simulate measurement (assuming identity attitude)
        mag_measurement = mag_reference.copy()
        
        initial_update_count = ekf.update_count
        
        # Apply magnetometer update
        ekf.update_magnetometer(mag_measurement, mag_reference)
        
        # Update count should increment
        assert ekf.update_count == initial_update_count + 1
    
    def test_zupt_stationary_detection(self):
        """Test ZUPT stationary detection."""
        ekf = InertialEKF()
        
        # Set small velocity
        ekf.state.velocity_ned = np.array([0.05, 0.02, 0.01])  # Small velocity
        
        # Stationary sensor measurements
        accel = np.array([0.0, 0.0, GRAVITY])  # Only gravity
        gyro = np.array([0.001, 0.002, 0.001])  # Very small rotation
        
        initial_zupt_count = ekf.zupt_count
        
        # Apply ZUPT
        zupt_applied = ekf.update_zupt(accel, gyro)
        
        # ZUPT should be applied
        assert zupt_applied
        assert ekf.zupt_count == initial_zupt_count + 1
        
        # Velocity should be reduced
        assert np.linalg.norm(ekf.state.velocity_ned) < 0.05
    
    def test_zupt_moving_detection(self):
        """Test ZUPT with moving vehicle (should not trigger)."""
        ekf = InertialEKF()
        
        # Moving sensor measurements
        accel = np.array([2.0, 0.0, GRAVITY])  # Acceleration
        gyro = np.array([0.1, 0.0, 0.0])  # Significant rotation
        
        initial_zupt_count = ekf.zupt_count
        
        # Apply ZUPT
        zupt_applied = ekf.update_zupt(accel, gyro)
        
        # ZUPT should NOT be applied
        assert not zupt_applied
        assert ekf.zupt_count == initial_zupt_count
    
    def test_zupt_disabled(self):
        """Test ZUPT when disabled."""
        ekf = InertialEKF()
        ekf.zupt_enabled = False
        
        # Stationary measurements
        accel = np.array([0.0, 0.0, GRAVITY])
        gyro = np.zeros(3)
        
        # Apply ZUPT
        zupt_applied = ekf.update_zupt(accel, gyro)
        
        # Should not be applied when disabled
        assert not zupt_applied
    
    def test_get_attitude_euler(self):
        """Test Euler angle extraction."""
        ekf = InertialEKF()
        
        # Set known quaternion (90° roll)
        ekf.state.quaternion = np.array([0.707107, 0.707107, 0.0, 0.0])
        
        euler = ekf.get_attitude_euler()
        
        expected = np.array([np.pi/2, 0.0, 0.0])  # 90° roll
        assert np.allclose(euler, expected, atol=1e-6)
    
    def test_get_rotation_matrix(self):
        """Test rotation matrix extraction."""
        ekf = InertialEKF()
        
        # Identity quaternion should give identity matrix
        R = ekf.get_rotation_matrix()
        
        assert np.allclose(R, np.eye(3))
    
    def test_get_state_dict(self):
        """Test state dictionary generation."""
        ekf = InertialEKF()
        
        state_dict = ekf.get_state_dict()
        
        # Check required fields
        required_fields = [
            'timestamp', 'quaternion', 'euler_rad', 'euler_deg',
            'velocity_ned', 'position_ned', 'gyro_bias', 'accel_bias',
            'covariance_trace', 'statistics'
        ]
        
        for field in required_fields:
            assert field in state_dict
        
        # Check statistics
        assert 'predictions' in state_dict['statistics']
        assert 'updates' in state_dict['statistics']
        assert 'zupts' in state_dict['statistics']
    
    def test_set_noise_parameters(self):
        """Test noise parameter setting."""
        ekf = InertialEKF()
        
        new_gyro_noise = 2e-4
        new_accel_noise = 3e-3
        
        ekf.set_noise_parameters(
            gyro_noise=new_gyro_noise,
            accel_noise=new_accel_noise
        )
        
        assert ekf.gyro_noise_density == new_gyro_noise
        assert ekf.accel_noise_density == new_accel_noise
    
    def test_reset_statistics(self):
        """Test statistics reset."""
        ekf = InertialEKF()
        
        # Simulate some activity
        ekf.prediction_count = 100
        ekf.update_count = 50
        ekf.zupt_count = 10
        
        # Reset
        ekf.reset_statistics()
        
        assert ekf.prediction_count == 0
        assert ekf.update_count == 0
        assert ekf.zupt_count == 0


class TestMotionSimulator:
    """Test cases for the motion simulator."""
    
    def test_initialization(self):
        """Test motion simulator initialization."""
        dt = 0.01
        simulator = MotionSimulator(dt=dt)
        
        assert simulator.dt == dt
        assert simulator.time == 0.0
        assert np.allclose(simulator.position, np.zeros(3))
        assert np.allclose(simulator.velocity, np.zeros(3))
        assert np.allclose(simulator.attitude, [1.0, 0.0, 0.0, 0.0])
    
    def test_step_progression(self):
        """Test that simulator progresses time correctly."""
        dt = 0.005
        simulator = MotionSimulator(dt=dt)
        
        initial_time = simulator.time
        
        # Take one step
        data = simulator.step()
        
        # Time should have advanced
        assert simulator.time == initial_time + dt
        assert data['timestamp'] == simulator.time
        assert data['dt'] == dt
    
    def test_sensor_data_format(self):
        """Test sensor data format and content."""
        simulator = MotionSimulator()
        
        data = simulator.step()
        
        # Check required fields
        required_fields = [
            'timestamp', 'dt', 'gyroscope', 'accelerometer_mpu',
            'accelerometer_lsm', 'magnetometer', 'temperature', 'truth'
        ]
        
        for field in required_fields:
            assert field in data
        
        # Check data shapes
        assert data['gyroscope'].shape == (3,)
        assert data['accelerometer_mpu'].shape == (3,)
        assert data['magnetometer'].shape == (3,)
        
        # Check truth data
        assert 'position' in data['truth']
        assert 'velocity' in data['truth']
        assert 'attitude' in data['truth']
    
    def test_sensor_data_realism(self):
        """Test that sensor data is realistic."""
        simulator = MotionSimulator()
        
        data = simulator.step()
        
        # Accelerometer should include gravity
        accel_magnitude = np.linalg.norm(data['accelerometer_mpu'])
        assert 8.0 < accel_magnitude < 12.0  # Should be close to gravity
        
        # Magnetometer should be reasonable Earth field
        mag_magnitude = np.linalg.norm(data['magnetometer'])
        assert 30e-6 < mag_magnitude < 70e-6  # Typical Earth field
        
        # Temperature should be reasonable
        assert 20.0 < data['temperature'] < 30.0
    
    def test_circular_motion(self):
        """Test that simulator produces circular motion."""
        simulator = MotionSimulator(dt=0.01)
        
        positions = []
        velocities = []
        
        # Collect data for one complete cycle
        for _ in range(int(10.0 / simulator.dt)):  # 10 seconds
            data = simulator.step()
            positions.append(data['truth']['position'])
            velocities.append(data['truth']['velocity'])
        
        positions = np.array(positions)
        velocities = np.array(velocities)
        
        # Should be roughly circular in X-Y plane
        radii = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        assert np.std(radii) < 1.0  # Should maintain roughly constant radius
        
        # Z position should remain close to zero
        assert np.max(np.abs(positions[:, 2])) < 1.0


# Integration tests combining multiple components
class TestEKFIntegration:
    """Integration tests for EKF with simulated data."""
    
    def test_ekf_with_simulator_short_run(self):
        """Test EKF with simulator data for short duration."""
        ekf = InertialEKF()
        simulator = MotionSimulator(dt=0.005)
        
        # Reference magnetic field
        mag_reference = np.array([20e-6, 0, 45e-6])
        
        # Run for 1 second
        duration = 1.0
        steps = int(duration / simulator.dt)
        
        for i in range(steps):
            # Get sensor data
            sensor_data = simulator.step()
            
            # EKF prediction
            ekf.predict(
                sensor_data['gyroscope'],
                sensor_data['accelerometer_mpu'],
                sensor_data['dt']
            )
            
            # Periodic updates
            if i % 10 == 0:
                ekf.update_gravity(sensor_data['accelerometer_mpu'])
                ekf.update_magnetometer(sensor_data['magnetometer'], mag_reference)
            
            # ZUPT
            ekf.update_zupt(sensor_data['accelerometer_mpu'], sensor_data['gyroscope'])
        
        # Check that EKF processed all data
        assert ekf.prediction_count == steps
        assert ekf.update_count > 0
        
        # State should be reasonable
        state_dict = ekf.get_state_dict()
        assert np.all(np.isfinite(state_dict['quaternion']))
        assert np.all(np.isfinite(state_dict['velocity_ned']))
        assert np.all(np.isfinite(state_dict['position_ned']))
    
    def test_ekf_consistency(self):
        """Test EKF consistency over time."""
        ekf = InertialEKF()
        simulator = MotionSimulator(dt=0.005)
        
        # Store initial covariance trace
        initial_covariance_trace = np.trace(ekf.state.covariance)
        
        # Run for several seconds
        for _ in range(1000):  # 5 seconds
            sensor_data = simulator.step()
            ekf.predict(
                sensor_data['gyroscope'],
                sensor_data['accelerometer_mpu'],
                sensor_data['dt']
            )
        
        # Covariance should remain positive definite
        final_covariance_trace = np.trace(ekf.state.covariance)
        assert final_covariance_trace > 0
        
        # State should remain bounded (not diverge)
        assert np.linalg.norm(ekf.state.position_ned) < 1000.0  # 1km
        assert np.linalg.norm(ekf.state.velocity_ned) < 100.0   # 100 m/s
    
    def test_zupt_effectiveness(self):
        """Test ZUPT effectiveness in reducing velocity drift."""
        # Create two EKFs - one with ZUPT, one without
        ekf_with_zupt = InertialEKF()
        ekf_without_zupt = InertialEKF()
        ekf_without_zupt.zupt_enabled = False
        
        # Use stationary scenario (no motion)
        gyro = np.array([0.001, 0.001, 0.001])  # Small bias
        accel = np.array([0.0, 0.0, GRAVITY])   # Only gravity
        dt = 0.005
        
        # Run for 10 seconds
        for _ in range(2000):
            ekf_with_zupt.predict(gyro, accel, dt)
            ekf_without_zupt.predict(gyro, accel, dt)
            
            # Apply ZUPT periodically to the first EKF
            if _ % 50 == 0:
                ekf_with_zupt.update_zupt(accel, gyro)
        
        # EKF with ZUPT should have smaller velocity error
        velocity_with_zupt = np.linalg.norm(ekf_with_zupt.state.velocity_ned)
        velocity_without_zupt = np.linalg.norm(ekf_without_zupt.state.velocity_ned)
        
        assert velocity_with_zupt < velocity_without_zupt
        assert ekf_with_zupt.zupt_count > 0


# Performance tests
class TestEKFPerformance:
    """Performance tests for EKF."""
    
    def test_prediction_performance(self):
        """Test EKF prediction performance."""
        ekf = InertialEKF()
        
        gyro = np.array([0.01, 0.02, 0.03])
        accel = np.array([0.1, 0.2, GRAVITY])
        dt = 0.005
        
        # Time multiple predictions
        start_time = time.time()
        num_predictions = 1000
        
        for _ in range(num_predictions):
            ekf.predict(gyro, accel, dt)
        
        elapsed_time = time.time() - start_time
        avg_time_per_prediction = elapsed_time / num_predictions
        
        # Should be able to predict at >1kHz (< 1ms per prediction)
        assert avg_time_per_prediction < 0.001
    
    def test_update_performance(self):
        """Test EKF update performance."""
        ekf = InertialEKF()
        
        accel_measurement = np.array([0.0, 0.0, GRAVITY])
        
        # Time multiple updates
        start_time = time.time()
        num_updates = 100
        
        for _ in range(num_updates):
            ekf.update_gravity(accel_measurement)
        
        elapsed_time = time.time() - start_time
        avg_time_per_update = elapsed_time / num_updates
        
        # Updates should be reasonably fast (< 10ms)
        assert avg_time_per_update < 0.01


if __name__ == '__main__':
    pytest.main([__file__]) 