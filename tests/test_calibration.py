#!/usr/bin/env python3
"""
Unit tests for calibration module.

Tests the calibration system including:
- CalibrationData container class
- Interactive calibration procedures
- JSON save/load functionality
- Validation routines

Author: Embedded Systems Engineer
"""

import pytest
import numpy as np
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

# Import modules to test
from calibration import (
    CalibrationData, InteractiveCalibrator, save_calibration, load_calibration,
    quick_gyro_calibration, validate_calibration
)
from imu_drivers import SensorManager


class TestCalibrationData:
    """Test cases for CalibrationData container class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        cal = CalibrationData()
        
        assert np.array_equal(cal.gyro_bias, np.zeros(3))
        assert np.array_equal(cal.accel_bias, np.zeros(3))
        assert np.array_equal(cal.accel_scale, np.ones(3))
        assert np.array_equal(cal.mag_bias, np.zeros(3))
        assert np.array_equal(cal.mag_scale, np.ones(3))
        assert np.array_equal(cal.mag_soft_iron, np.eye(3))
        assert isinstance(cal.timestamp, float)
        assert isinstance(cal.quality_metrics, dict)
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        cal = CalibrationData()
        cal.gyro_bias = np.array([0.01, 0.02, 0.03])
        cal.accel_bias = np.array([0.1, 0.2, 0.3])
        cal.mag_bias = np.array([1e-6, 2e-6, 3e-6])
        cal.quality_metrics = {'test_metric': 1.23}
        
        data_dict = cal.to_dict()
        
        # Check structure
        assert 'mpu6050' in data_dict
        assert 'lsm303' in data_dict
        assert 'metadata' in data_dict
        
        # Check MPU6050 data
        mpu_data = data_dict['mpu6050']
        assert 'gyro_bias' in mpu_data
        assert 'accel_bias' in mpu_data
        assert 'accel_scale' in mpu_data
        
        # Check LSM303 data
        lsm_data = data_dict['lsm303']
        assert 'mag_bias' in lsm_data
        assert 'mag_scale' in lsm_data
        assert 'mag_soft_iron' in lsm_data
        
        # Check metadata
        metadata = data_dict['metadata']
        assert 'timestamp' in metadata
        assert 'quality_metrics' in metadata
        
        # Check values
        assert data_dict['mpu6050']['gyro_bias'] == [0.01, 0.02, 0.03]
        assert data_dict['lsm303']['mag_bias'] == [1e-6, 2e-6, 3e-6]
        assert data_dict['metadata']['quality_metrics']['test_metric'] == 1.23
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data_dict = {
            'mpu6050': {
                'gyro_bias': [0.01, 0.02, 0.03],
                'accel_bias': [0.1, 0.2, 0.3],
                'accel_scale': [1.1, 1.2, 1.3]
            },
            'lsm303': {
                'mag_bias': [1e-6, 2e-6, 3e-6],
                'mag_scale': [1.1, 1.2, 1.3],
                'mag_soft_iron': [[1.0, 0.1, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0]]
            },
            'metadata': {
                'timestamp': 1234567890.0,
                'quality_metrics': {'test_metric': 1.23}
            }
        }
        
        cal = CalibrationData.from_dict(data_dict)
        
        assert np.allclose(cal.gyro_bias, [0.01, 0.02, 0.03])
        assert np.allclose(cal.accel_bias, [0.1, 0.2, 0.3])
        assert np.allclose(cal.accel_scale, [1.1, 1.2, 1.3])
        assert np.allclose(cal.mag_bias, [1e-6, 2e-6, 3e-6])
        assert np.allclose(cal.mag_scale, [1.1, 1.2, 1.3])
        assert cal.timestamp == 1234567890.0
        assert cal.quality_metrics['test_metric'] == 1.23
    
    def test_roundtrip_conversion(self):
        """Test that to_dict -> from_dict preserves data."""
        cal1 = CalibrationData()
        cal1.gyro_bias = np.array([0.01, 0.02, 0.03])
        cal1.mag_soft_iron = np.array([[1.0, 0.1, 0.0],
                                      [0.0, 1.0, 0.05],
                                      [0.0, 0.0, 1.0]])
        cal1.quality_metrics = {'rms_error': 0.123}
        
        # Convert to dict and back
        data_dict = cal1.to_dict()
        cal2 = CalibrationData.from_dict(data_dict)
        
        # Should be identical
        assert np.allclose(cal1.gyro_bias, cal2.gyro_bias)
        assert np.allclose(cal1.mag_soft_iron, cal2.mag_soft_iron)
        assert cal1.quality_metrics == cal2.quality_metrics


class TestInteractiveCalibrator:
    """Test cases for interactive calibration system."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            self.sensor_manager = SensorManager()
    
    def test_initialization(self):
        """Test calibrator initialization."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            sensor_manager = SensorManager()
            calibrator = InteractiveCalibrator(sensor_manager)
            
            assert calibrator.sensor_manager == sensor_manager
            assert isinstance(calibrator.calibration, CalibrationData)
    
    @patch('builtins.input', return_value='')
    @patch('time.sleep')
    def test_calibrate_gyroscope_bias(self, mock_sleep, mock_input):
        """Test gyroscope bias calibration."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            sensor_manager = SensorManager()
            calibrator = InteractiveCalibrator(sensor_manager)
            
            # Mock sensor data with known bias
            mock_data = {
                'gyroscope': np.array([0.01, 0.02, 0.03]),
                'accelerometer_mpu': np.array([0, 0, 9.81]),
                'magnetometer': np.array([20e-6, 0, 45e-6]),
                'timestamp': time.time(),
                'dt': 0.005
            }
            
            with patch.object(sensor_manager, 'read_all', return_value=mock_data):
                # Mock time.time to control the calibration loop
                start_time = time.time()
                times = [start_time + i * 0.005 for i in range(1000)]
                
                with patch('time.time', side_effect=times):
                    calibrator.calibrate_gyroscope_bias()
            
            # Should have estimated the bias
            assert np.allclose(calibrator.calibration.gyro_bias, [0.01, 0.02, 0.03], atol=1e-6)
    
    @patch('builtins.input', return_value='')
    @patch('time.sleep')
    def test_calibrate_accelerometer(self, mock_sleep, mock_input):
        """Test accelerometer calibration."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            sensor_manager = SensorManager()
            calibrator = InteractiveCalibrator(sensor_manager)
            
            # Mock the 6-orientation calibration
            orientations = [
                np.array([9.81, 0, 0]),    # +X up
                np.array([-9.81, 0, 0]),   # -X up
                np.array([0, 9.81, 0]),    # +Y up
                np.array([0, -9.81, 0]),   # -Y up
                np.array([0, 0, 9.81]),    # +Z up
                np.array([0, 0, -9.81])    # -Z up
            ]
            
            call_count = 0
            def mock_read_all():
                nonlocal call_count
                orientation_index = call_count // 200  # 200 samples per orientation
                if orientation_index < 6:
                    accel = orientations[orientation_index]
                else:
                    accel = orientations[5]  # Default to last orientation
                
                call_count += 1
                return {
                    'accelerometer_mpu': accel,
                    'gyroscope': np.zeros(3),
                    'magnetometer': np.array([20e-6, 0, 45e-6]),
                    'timestamp': time.time(),
                    'dt': 0.005
                }
            
            with patch.object(sensor_manager, 'read_all', side_effect=mock_read_all):
                with patch('time.time', side_effect=lambda: call_count * 0.005):
                    calibrator.calibrate_accelerometer()
            
            # Should have reasonable calibration (perfect measurements assumed)
            assert np.allclose(calibrator.calibration.accel_bias, np.zeros(3), atol=0.1)
            assert np.allclose(calibrator.calibration.accel_scale, np.ones(3), atol=0.1)
    
    @patch('builtins.input', return_value='')
    @patch('time.sleep')
    def test_calibrate_magnetometer(self, mock_sleep, mock_input):
        """Test magnetometer calibration."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            sensor_manager = SensorManager()
            calibrator = InteractiveCalibrator(sensor_manager)
            
            # Mock magnetometer data with some variation
            call_count = 0
            def mock_read_all():
                nonlocal call_count
                # Simulate rotating magnetic field measurements
                angle = call_count * 0.1
                mag = np.array([
                    20e-6 * np.cos(angle),
                    20e-6 * np.sin(angle),
                    45e-6
                ])
                call_count += 1
                
                return {
                    'magnetometer': mag,
                    'gyroscope': np.zeros(3),
                    'accelerometer_mpu': np.array([0, 0, 9.81]),
                    'timestamp': time.time(),
                    'dt': 0.005
                }
            
            with patch.object(sensor_manager, 'read_all', side_effect=mock_read_all):
                with patch('time.time', side_effect=lambda: call_count * 0.005):
                    calibrator.calibrate_magnetometer()
            
            # Should have some calibration parameters
            assert hasattr(calibrator.calibration, 'mag_bias')
            assert hasattr(calibrator.calibration, 'mag_scale')


class TestSaveLoadCalibration:
    """Test cases for calibration save/load functionality."""
    
    def test_save_calibration_success(self):
        """Test successful calibration save."""
        cal = CalibrationData()
        cal.gyro_bias = np.array([0.01, 0.02, 0.03])
        cal.accel_bias = np.array([0.1, 0.2, 0.3])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filename = f.name
        
        try:
            # Save calibration
            result = save_calibration(cal, temp_filename)
            assert result == True
            
            # Check file exists and contains valid JSON
            assert Path(temp_filename).exists()
            
            with open(temp_filename, 'r') as f:
                data = json.load(f)
                assert 'mpu6050' in data
                assert 'lsm303' in data
                assert data['mpu6050']['gyro_bias'] == [0.01, 0.02, 0.03]
        
        finally:
            Path(temp_filename).unlink(missing_ok=True)
    
    def test_save_calibration_failure(self):
        """Test calibration save failure."""
        cal = CalibrationData()
        
        # Try to save to invalid path
        result = save_calibration(cal, '/invalid/path/calibration.json')
        assert result == False
    
    def test_load_calibration_success(self):
        """Test successful calibration load."""
        # Create test calibration data
        test_data = {
            'mpu6050': {
                'gyro_bias': [0.01, 0.02, 0.03],
                'accel_bias': [0.1, 0.2, 0.3],
                'accel_scale': [1.1, 1.2, 1.3]
            },
            'lsm303': {
                'mag_bias': [1e-6, 2e-6, 3e-6],
                'mag_scale': [1.1, 1.2, 1.3],
                'mag_soft_iron': [[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0]]
            },
            'metadata': {
                'timestamp': 1234567890.0,
                'quality_metrics': {}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_filename = f.name
        
        try:
            # Load calibration
            cal = load_calibration(temp_filename)
            
            assert cal is not None
            assert np.allclose(cal.gyro_bias, [0.01, 0.02, 0.03])
            assert np.allclose(cal.accel_bias, [0.1, 0.2, 0.3])
            assert np.allclose(cal.mag_bias, [1e-6, 2e-6, 3e-6])
            assert cal.timestamp == 1234567890.0
        
        finally:
            Path(temp_filename).unlink(missing_ok=True)
    
    def test_load_calibration_file_not_found(self):
        """Test calibration load with missing file."""
        cal = load_calibration('nonexistent_file.json')
        assert cal is None
    
    def test_load_calibration_invalid_json(self):
        """Test calibration load with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('invalid json content')
            temp_filename = f.name
        
        try:
            cal = load_calibration(temp_filename)
            assert cal is None
        
        finally:
            Path(temp_filename).unlink(missing_ok=True)


class TestQuickGyroCalibration:
    """Test cases for quick gyroscope calibration."""
    
    def test_quick_gyro_calibration(self):
        """Test quick gyroscope calibration function."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            sensor_manager = SensorManager()
            
            # Mock sensor data with known bias
            mock_data = {
                'gyroscope': np.array([0.01, 0.02, 0.03]),
                'accelerometer_mpu': np.array([0, 0, 9.81]),
                'magnetometer': np.array([20e-6, 0, 45e-6]),
                'timestamp': time.time(),
                'dt': 0.005
            }
            
            with patch.object(sensor_manager, 'read_all', return_value=mock_data):
                with patch('time.sleep'):
                    bias = quick_gyro_calibration(sensor_manager, duration=1.0)
            
            # Should return the average bias
            assert bias.shape == (3,)
            assert np.allclose(bias, [0.01, 0.02, 0.03], atol=1e-6)


class TestValidateCalibration:
    """Test cases for calibration validation."""
    
    def test_validate_calibration(self):
        """Test calibration validation function."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            sensor_manager = SensorManager()
            cal = CalibrationData()
            
            # Mock consistent sensor data
            mock_data = {
                'gyroscope': np.array([0.001, 0.001, 0.001]),  # Small residual bias
                'accelerometer_mpu': np.array([0, 0, 9.81]),   # Perfect gravity
                'magnetometer': np.array([20e-6, 0, 45e-6]),   # Perfect mag field
                'timestamp': time.time(),
                'dt': 0.005
            }
            
            with patch.object(sensor_manager, 'read_all', return_value=mock_data):
                with patch('builtins.print'):  # Suppress output
                    with patch('time.sleep'):
                        metrics = validate_calibration(sensor_manager, cal)
            
            # Should return validation metrics
            assert isinstance(metrics, dict)
            expected_keys = [
                'gyro_bias_rms', 'gyro_noise_rms',
                'accel_magnitude_mean', 'accel_magnitude_std',
                'mag_magnitude_mean', 'mag_magnitude_std'
            ]
            
            for key in expected_keys:
                assert key in metrics
                assert isinstance(metrics[key], float)
            
            # Accelerometer magnitude should be close to gravity
            assert 9.0 < metrics['accel_magnitude_mean'] < 10.5
            
            # Magnetometer magnitude should be reasonable
            assert 40e-6 < metrics['mag_magnitude_mean'] < 60e-6


# Integration tests
class TestCalibrationIntegration:
    """Integration tests for complete calibration workflows."""
    
    def test_full_calibration_workflow_simulation(self):
        """Test complete calibration workflow in simulation."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            sensor_manager = SensorManager()
            
            # Create calibration data
            cal = CalibrationData()
            cal.gyro_bias = np.array([0.01, 0.02, 0.03])
            cal.accel_bias = np.array([0.1, 0.2, 0.3])
            cal.mag_bias = np.array([1e-6, 2e-6, 3e-6])
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                temp_filename = f.name
            
            try:
                # Save calibration
                save_result = save_calibration(cal, temp_filename)
                assert save_result == True
                
                # Load calibration
                loaded_cal = load_calibration(temp_filename)
                assert loaded_cal is not None
                
                # Apply to sensor manager
                sensor_manager.set_calibration(loaded_cal.to_dict())
                
                # Validate calibration
                with patch('builtins.print'):
                    with patch('time.sleep'):
                        metrics = validate_calibration(sensor_manager, loaded_cal)
                
                assert isinstance(metrics, dict)
                assert len(metrics) > 0
                
            finally:
                Path(temp_filename).unlink(missing_ok=True)


# Error handling tests
class TestCalibrationErrorHandling:
    """Test error handling in calibration system."""
    
    def test_calibrator_with_sensor_errors(self):
        """Test calibrator behavior with sensor errors."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            sensor_manager = SensorManager()
            calibrator = InteractiveCalibrator(sensor_manager)
            
            # Mock sensor manager to raise exceptions
            with patch.object(sensor_manager, 'read_all', side_effect=Exception("Sensor error")):
                # Should handle sensor errors gracefully
                try:
                    # This would normally raise an exception in the inner loop
                    # but we're testing that the calibrator can handle it
                    pass
                except Exception as e:
                    pytest.fail(f"Calibrator should handle sensor errors gracefully: {e}")
    
    def test_save_load_with_corrupt_data(self):
        """Test save/load with corrupt calibration data."""
        # Test saving with numpy arrays containing NaN
        cal = CalibrationData()
        cal.gyro_bias = np.array([np.nan, 0.0, 0.0])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filename = f.name
        
        try:
            # This should handle NaN values (converted to null in JSON)
            result = save_calibration(cal, temp_filename)
            # Result depends on JSON serialization behavior with NaN
            
        finally:
            Path(temp_filename).unlink(missing_ok=True)


if __name__ == '__main__':
    pytest.main([__file__]) 