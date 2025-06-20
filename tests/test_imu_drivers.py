#!/usr/bin/env python3
"""
Unit tests for IMU drivers module.

Tests the MPU-6050 and LSM303DLHC sensor drivers including:
- Initialization and configuration
- Data reading and conversion
- Calibration application
- Error handling

Author: Embedded Systems Engineer
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
from imu_drivers import IMUDriverBase, MPU6050, LSM303DLHC, SensorManager


class TestIMUDriverBase:
    """Test cases for the base IMU driver class."""
    
    def test_initialization_without_hardware(self):
        """Test driver initialization when hardware is not available."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            driver = IMUDriverBase(bus=1, address=0x68)
            assert driver.bus_num == 1
            assert driver.address == 0x68
            assert not driver.is_connected
            assert driver.bus is None
    
    def test_initialization_with_hardware_success(self):
        """Test driver initialization with successful hardware connection."""
        with patch('imu_drivers.HAS_HARDWARE', True):
            with patch('imu_drivers.smbus2.SMBus') as mock_smbus:
                mock_bus = Mock()
                mock_smbus.return_value = mock_bus
                
                driver = IMUDriverBase(bus=1, address=0x68)
                
                assert driver.is_connected
                assert driver.bus == mock_bus
                mock_smbus.assert_called_once_with(1)
    
    def test_initialization_with_hardware_failure(self):
        """Test driver initialization with hardware connection failure."""
        with patch('imu_drivers.HAS_HARDWARE', True):
            with patch('imu_drivers.smbus2.SMBus') as mock_smbus:
                mock_smbus.side_effect = Exception("I2C error")
                
                driver = IMUDriverBase(bus=1, address=0x68)
                
                assert not driver.is_connected
                assert driver.bus is None
    
    def test_read_register_without_connection(self):
        """Test register read without hardware connection."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            driver = IMUDriverBase()
            result = driver.read_register(0x75)
            assert result == 0
    
    def test_read_register_with_connection(self):
        """Test register read with hardware connection."""
        with patch('imu_drivers.HAS_HARDWARE', True):
            with patch('imu_drivers.smbus2.SMBus') as mock_smbus:
                mock_bus = Mock()
                mock_bus.read_byte_data.return_value = 0x68
                mock_smbus.return_value = mock_bus
                
                driver = IMUDriverBase()
                result = driver.read_register(0x75)
                
                assert result == 0x68
                mock_bus.read_byte_data.assert_called_once_with(driver.address, 0x75)
    
    def test_write_register_without_connection(self):
        """Test register write without hardware connection."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            driver = IMUDriverBase()
            result = driver.write_register(0x6B, 0x01)
            assert not result
    
    def test_write_register_with_connection(self):
        """Test register write with hardware connection."""
        with patch('imu_drivers.HAS_HARDWARE', True):
            with patch('imu_drivers.smbus2.SMBus') as mock_smbus:
                mock_bus = Mock()
                mock_smbus.return_value = mock_bus
                
                driver = IMUDriverBase()
                result = driver.write_register(0x6B, 0x01)
                
                assert result
                mock_bus.write_byte_data.assert_called_once_with(driver.address, 0x6B, 0x01)


class TestMPU6050:
    """Test cases for the MPU-6050 driver."""
    
    def test_initialization_simulation_mode(self):
        """Test MPU-6050 initialization in simulation mode."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            mpu = MPU6050()
            
            assert mpu.accel_range == 8
            assert mpu.gyro_range == 1000
            assert mpu.sample_rate == 200
            assert np.array_equal(mpu.gyro_bias, np.zeros(3))
            assert np.array_equal(mpu.accel_bias, np.zeros(3))
            assert np.array_equal(mpu.accel_scale, np.ones(3))
    
    def test_initialization_hardware_mode(self):
        """Test MPU-6050 initialization with hardware."""
        with patch('imu_drivers.HAS_HARDWARE', True):
            with patch('imu_drivers.smbus2.SMBus') as mock_smbus:
                mock_bus = Mock()
                mock_bus.read_byte_data.return_value = 0x68  # WHO_AM_I response
                mock_smbus.return_value = mock_bus
                
                mpu = MPU6050()
                
                # Check that initialization sequence was called
                assert mock_bus.write_byte_data.call_count > 0
    
    def test_who_am_i_check_failure(self):
        """Test WHO_AM_I register check failure."""
        with patch('imu_drivers.HAS_HARDWARE', True):
            with patch('imu_drivers.smbus2.SMBus') as mock_smbus:
                mock_bus = Mock()
                mock_bus.read_byte_data.return_value = 0x00  # Wrong WHO_AM_I
                mock_smbus.return_value = mock_bus
                
                mpu = MPU6050()
                
                # Should still create object but log error
                assert mpu is not None
    
    def test_read_raw_simulation_mode(self):
        """Test raw data reading in simulation mode."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            mpu = MPU6050()
            
            accel, gyro, temp = mpu.read_raw()
            
            assert accel.shape == (3,)
            assert gyro.shape == (3,)
            assert isinstance(temp, float)
            assert 20.0 < temp < 30.0  # Reasonable temperature range
    
    def test_read_raw_hardware_mode(self):
        """Test raw data reading with hardware."""
        with patch('imu_drivers.HAS_HARDWARE', True):
            with patch('imu_drivers.smbus2.SMBus') as mock_smbus:
                mock_bus = Mock()
                mock_bus.read_byte_data.return_value = 0x68
                
                # Mock sensor data (14 bytes)
                sensor_data = [
                    0x20, 0x00,  # Accel X (8192 LSB = +4g with ±8g range)
                    0x00, 0x00,  # Accel Y (0)
                    0x40, 0x00,  # Accel Z (16384 LSB = +8g)
                    0x00, 0x00,  # Temperature
                    0x00, 0x00,  # Gyro X (0)
                    0x00, 0x00,  # Gyro Y (0)
                    0x00, 0x00   # Gyro Z (0)
                ]
                mock_bus.read_i2c_block_data.return_value = sensor_data
                mock_smbus.return_value = mock_bus
                
                mpu = MPU6050()
                accel, gyro, temp = mpu.read_raw()
                
                assert accel.shape == (3,)
                assert gyro.shape == (3,)
                assert isinstance(temp, float)
    
    def test_calibration_application(self):
        """Test calibration parameter application."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            mpu = MPU6050()
            
            # Set calibration parameters
            gyro_bias = np.array([0.01, -0.005, 0.002])
            accel_bias = np.array([0.1, -0.05, 0.02])
            accel_scale = np.array([1.01, 0.99, 1.005])
            
            mpu.set_calibration(gyro_bias, accel_bias, accel_scale)
            
            assert np.array_equal(mpu.gyro_bias, gyro_bias)
            assert np.array_equal(mpu.accel_bias, accel_bias)
            assert np.array_equal(mpu.accel_scale, accel_scale)
    
    def test_read_with_calibration(self):
        """Test calibrated data reading."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            mpu = MPU6050()
            
            # Set some calibration
            mpu.gyro_bias = np.array([0.01, 0.01, 0.01])
            mpu.accel_bias = np.array([0.1, 0.1, 0.1])
            mpu.accel_scale = np.array([1.1, 1.1, 1.1])
            
            accel, gyro, temp = mpu.read()
            
            # Should return calibrated values
            assert accel.shape == (3,)
            assert gyro.shape == (3,)
    
    def test_self_test_simulation_mode(self):
        """Test self-test in simulation mode."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            mpu = MPU6050()
            
            results = mpu.self_test()
            
            assert isinstance(results, dict)
            assert 'accel_x' in results
            assert 'gyro_x' in results
            # In simulation mode, all tests pass
            assert all(results.values())


class TestLSM303DLHC:
    """Test cases for the LSM303DLHC driver."""
    
    def test_initialization_simulation_mode(self):
        """Test LSM303DLHC initialization in simulation mode."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            lsm = LSM303DLHC()
            
            assert lsm.accel_range == 8
            assert lsm.mag_range == 1.3
            assert np.array_equal(lsm.mag_bias, np.zeros(3))
            assert np.array_equal(lsm.mag_scale, np.ones(3))
            assert np.array_equal(lsm.mag_soft_iron, np.eye(3))
    
    def test_read_accelerometer_simulation_mode(self):
        """Test accelerometer reading in simulation mode."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            lsm = LSM303DLHC()
            
            accel = lsm.read_accelerometer()
            
            assert accel.shape == (3,)
            # Should be close to gravity
            assert 8.0 < np.linalg.norm(accel) < 11.0
    
    def test_read_magnetometer_simulation_mode(self):
        """Test magnetometer reading in simulation mode."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            lsm = LSM303DLHC()
            
            mag = lsm.read_magnetometer()
            
            assert mag.shape == (3,)
            # Should be reasonable Earth field strength
            assert 30e-6 < np.linalg.norm(mag) < 70e-6
    
    def test_magnetometer_calibration_application(self):
        """Test magnetometer calibration application."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            lsm = LSM303DLHC()
            
            bias = np.array([1e-6, -0.5e-6, 2e-6])
            scale = np.array([1.1, 0.9, 1.05])
            soft_iron = np.array([[1.0, 0.1, 0.0],
                                 [0.0, 1.0, 0.05],
                                 [0.0, 0.0, 1.0]])
            
            lsm.set_magnetometer_calibration(bias, scale, soft_iron)
            
            assert np.array_equal(lsm.mag_bias, bias)
            assert np.array_equal(lsm.mag_scale, scale)
            assert np.array_equal(lsm.mag_soft_iron, soft_iron)
    
    def test_read_with_calibration(self):
        """Test calibrated data reading."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            lsm = LSM303DLHC()
            
            # Set some calibration
            lsm.mag_bias = np.array([1e-6, 1e-6, 1e-6])
            lsm.mag_scale = np.array([1.1, 1.1, 1.1])
            
            accel, mag = lsm.read()
            
            assert accel.shape == (3,)
            assert mag.shape == (3,)


class TestSensorManager:
    """Test cases for the sensor manager."""
    
    def test_initialization(self):
        """Test sensor manager initialization."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            manager = SensorManager()
            
            assert manager.mpu6050 is not None
            assert manager.lsm303 is not None
            assert manager.last_read_time == 0.0
    
    def test_read_all_simulation_mode(self):
        """Test reading all sensors in simulation mode."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            manager = SensorManager()
            
            data = manager.read_all()
            
            assert 'timestamp' in data
            assert 'dt' in data
            assert 'gyroscope' in data
            assert 'accelerometer_mpu' in data
            assert 'accelerometer_lsm' in data
            assert 'magnetometer' in data
            assert 'temperature' in data
            
            # Check data shapes
            assert data['gyroscope'].shape == (3,)
            assert data['accelerometer_mpu'].shape == (3,)
            assert data['accelerometer_lsm'].shape == (3,)
            assert data['magnetometer'].shape == (3,)
    
    def test_timing_consistency(self):
        """Test that consecutive reads have reasonable timing."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            manager = SensorManager()
            
            # First read
            data1 = manager.read_all()
            time.sleep(0.01)  # 10ms delay
            
            # Second read
            data2 = manager.read_all()
            
            # Check that dt is reasonable
            assert 0.005 < data2['dt'] < 0.02
            assert data2['timestamp'] > data1['timestamp']
    
    def test_self_test(self):
        """Test sensor self-test functionality."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            manager = SensorManager()
            
            results = manager.self_test()
            
            assert 'mpu6050' in results
            assert 'lsm303' in results
    
    def test_calibration_loading(self):
        """Test calibration data loading."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            manager = SensorManager()
            
            # Create mock calibration data
            calibration_data = {
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
                }
            }
            
            manager.set_calibration(calibration_data)
            
            # Check that calibration was applied
            assert np.allclose(manager.mpu6050.gyro_bias, [0.01, 0.02, 0.03])
            assert np.allclose(manager.lsm303.mag_bias, [1e-6, 2e-6, 3e-6])


# Performance and stress tests
class TestPerformance:
    """Performance and stress tests."""
    
    def test_read_performance(self):
        """Test sensor read performance."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            manager = SensorManager()
            
            # Time multiple reads
            start_time = time.time()
            num_reads = 100
            
            for _ in range(num_reads):
                data = manager.read_all()
            
            elapsed_time = time.time() - start_time
            avg_time_per_read = elapsed_time / num_reads
            
            # Should be able to read at >1kHz (< 1ms per read)
            assert avg_time_per_read < 0.001
    
    def test_memory_usage(self):
        """Test that objects don't leak memory."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            import gc
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            # Create and destroy many sensor managers
            for _ in range(100):
                manager = SensorManager()
                for _ in range(10):
                    data = manager.read_all()
                del manager
                gc.collect()
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be minimal (< 10MB)
            assert memory_increase < 10 * 1024 * 1024


# Integration tests
class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_sensor_workflow(self):
        """Test complete sensor reading workflow."""
        with patch('imu_drivers.HAS_HARDWARE', False):
            # Initialize sensor manager
            manager = SensorManager()
            
            # Run self-test
            self_test_results = manager.self_test()
            assert self_test_results is not None
            
            # Set calibration
            calibration_data = {
                'mpu6050': {
                    'gyro_bias': [0.01, 0.01, 0.01],
                    'accel_bias': [0.1, 0.1, 0.1],
                    'accel_scale': [1.1, 1.1, 1.1]
                },
                'lsm303': {
                    'mag_bias': [1e-6, 1e-6, 1e-6],
                    'mag_scale': [1.1, 1.1, 1.1],
                    'mag_soft_iron': [[1.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 1.0]]
                }
            }
            manager.set_calibration(calibration_data)
            
            # Read sensor data multiple times
            for i in range(10):
                data = manager.read_all()
                
                # Verify all expected fields are present
                required_fields = ['timestamp', 'dt', 'gyroscope', 
                                 'accelerometer_mpu', 'accelerometer_lsm', 
                                 'magnetometer', 'temperature']
                for field in required_fields:
                    assert field in data
                
                # Verify data is reasonable
                assert np.all(np.isfinite(data['gyroscope']))
                assert np.all(np.isfinite(data['accelerometer_mpu']))
                assert np.all(np.isfinite(data['magnetometer']))
                
                # Brief delay between reads
                time.sleep(0.001)


if __name__ == '__main__':
    pytest.main([__file__]) 