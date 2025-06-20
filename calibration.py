#!/usr/bin/env python3
"""
Sensor Calibration Module for IMU Navigation System.

This module provides interactive calibration procedures for:
- Gyroscope bias calibration
- Accelerometer bias and scale calibration
- Magnetometer hard/soft iron calibration

The calibration data is saved/loaded in JSON format for persistence.

Author: Embedded Systems Engineer
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from imu_drivers import SensorManager


# Constants for calibration
GYRO_CALIBRATION_TIME = 30.0  # seconds
ACCEL_CALIBRATION_TIME = 5.0  # seconds per orientation
MAG_CALIBRATION_TIME = 60.0   # seconds
SAMPLE_RATE = 200.0           # Hz


class CalibrationData:
    """Container for all sensor calibration parameters."""
    
    def __init__(self):
        # MPU-6050 calibration
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.accel_scale = np.ones(3)
        
        # LSM303DLHC magnetometer calibration
        self.mag_bias = np.zeros(3)
        self.mag_scale = np.ones(3)
        self.mag_soft_iron = np.eye(3)
        
        # Metadata
        self.timestamp = time.time()
        self.quality_metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert calibration data to dictionary for JSON serialization."""
        return {
            'mpu6050': {
                'gyro_bias': self.gyro_bias.tolist(),
                'accel_bias': self.accel_bias.tolist(),
                'accel_scale': self.accel_scale.tolist()
            },
            'lsm303': {
                'mag_bias': self.mag_bias.tolist(),
                'mag_scale': self.mag_scale.tolist(),
                'mag_soft_iron': self.mag_soft_iron.tolist()
            },
            'metadata': {
                'timestamp': self.timestamp,
                'quality_metrics': self.quality_metrics
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationData':
        """Create calibration data from dictionary."""
        cal = cls()
        
        if 'mpu6050' in data:
            mpu_data = data['mpu6050']
            cal.gyro_bias = np.array(mpu_data['gyro_bias'])
            cal.accel_bias = np.array(mpu_data['accel_bias'])
            cal.accel_scale = np.array(mpu_data['accel_scale'])
        
        if 'lsm303' in data:
            lsm_data = data['lsm303']
            cal.mag_bias = np.array(lsm_data['mag_bias'])
            cal.mag_scale = np.array(lsm_data['mag_scale'])
            cal.mag_soft_iron = np.array(lsm_data['mag_soft_iron'])
        
        if 'metadata' in data:
            meta = data['metadata']
            cal.timestamp = meta.get('timestamp', time.time())
            cal.quality_metrics = meta.get('quality_metrics', {})
        
        return cal


class InteractiveCalibrator:
    """Interactive calibration system with user prompts and real-time feedback."""
    
    def __init__(self, sensor_manager: SensorManager):
        self.sensor_manager = sensor_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self.calibration = CalibrationData()
    
    def run_full_calibration(self) -> CalibrationData:
        """Run complete calibration sequence for all sensors."""
        print("\n" + "="*60)
        print("IMU CALIBRATION SEQUENCE")
        print("="*60)
        print("This calibration will take approximately 5-10 minutes.")
        print("Please follow the instructions carefully for best results.")
        
        input("\nPress Enter to start...")
        
        # 1. Gyroscope bias calibration
        print("\n1. Gyroscope Bias Calibration")
        print("-"*40)
        self.calibrate_gyroscope_bias()
        
        # 2. Accelerometer calibration
        print("\n2. Accelerometer Calibration")
        print("-"*40)
        self.calibrate_accelerometer()
        
        # 3. Magnetometer calibration
        print("\n3. Magnetometer Calibration")
        print("-"*40)
        self.calibrate_magnetometer()
        
        # Calculate quality metrics
        self.calculate_quality_metrics()
        
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE!")
        print("="*60)
        self.print_calibration_summary()
        
        return self.calibration
    
    def calibrate_gyroscope_bias(self) -> None:
        """Calibrate gyroscope bias by collecting stationary data."""
        print(f"Place the IMU on a stable, level surface.")
        print(f"The device must remain completely stationary for {GYRO_CALIBRATION_TIME} seconds.")
        input("Press Enter when ready...")
        
        print(f"\nCollecting gyroscope data for {GYRO_CALIBRATION_TIME} seconds...")
        
        gyro_samples = []
        start_time = time.time()
        samples_collected = 0
        
        while time.time() - start_time < GYRO_CALIBRATION_TIME:
            data = self.sensor_manager.read_all()
            gyro_samples.append(data['gyroscope'])
            samples_collected += 1
            
            # Progress indicator
            elapsed = time.time() - start_time
            progress = int(50 * elapsed / GYRO_CALIBRATION_TIME)
            print(f"\rProgress: [{'='*progress}{' '*(50-progress)}] "
                  f"{elapsed:.1f}/{GYRO_CALIBRATION_TIME:.1f}s", end="")
            
            time.sleep(1.0 / SAMPLE_RATE)
        
        print()  # New line after progress bar
        
        # Calculate bias
        gyro_data = np.array(gyro_samples)
        self.calibration.gyro_bias = np.mean(gyro_data, axis=0)
        
        # Quality metrics
        gyro_std = np.std(gyro_data, axis=0)
        max_std = np.max(gyro_std)
        
        print(f"Gyroscope bias: [{self.calibration.gyro_bias[0]:.6f}, "
              f"{self.calibration.gyro_bias[1]:.6f}, {self.calibration.gyro_bias[2]:.6f}] rad/s")
        print(f"Standard deviation: [{gyro_std[0]:.6f}, {gyro_std[1]:.6f}, {gyro_std[2]:.6f}] rad/s")
        
        if max_std > 0.01:  # 0.01 rad/s = ~0.57 degrees/s
            print("WARNING: High noise detected during gyroscope calibration!")
            print("Consider repeating calibration in a more stable environment.")
        else:
            print("✓ Gyroscope calibration successful")
        
        self.calibration.quality_metrics['gyro_std'] = gyro_std.tolist()
    
    def calibrate_accelerometer(self) -> None:
        """Calibrate accelerometer bias and scale using 6-point method."""
        print("The accelerometer will be calibrated using a 6-point method.")
        print("You will need to place the IMU in 6 different orientations:")
        print("1. +X up (X-axis pointing up)")
        print("2. -X up (X-axis pointing down)")
        print("3. +Y up (Y-axis pointing up)")
        print("4. -Y up (Y-axis pointing down)")
        print("5. +Z up (Z-axis pointing up)")
        print("6. -Z up (Z-axis pointing down)")
        print(f"\nEach orientation will be measured for {ACCEL_CALIBRATION_TIME} seconds.")
        
        orientations = [
            ("+X up", np.array([1, 0, 0])),
            ("-X up", np.array([-1, 0, 0])),
            ("+Y up", np.array([0, 1, 0])),
            ("-Y up", np.array([0, -1, 0])),
            ("+Z up", np.array([0, 0, 1])),
            ("-Z up", np.array([0, 0, -1]))
        ]
        
        accel_measurements = []
        expected_vectors = []
        
        for i, (orientation_name, expected_vector) in enumerate(orientations):
            print(f"\nOrientation {i+1}/6: {orientation_name}")
            input("Position the IMU and press Enter...")
            
            print(f"Collecting data for {ACCEL_CALIBRATION_TIME} seconds...")
            
            samples = []
            start_time = time.time()
            
            while time.time() - start_time < ACCEL_CALIBRATION_TIME:
                data = self.sensor_manager.read_all()
                samples.append(data['accelerometer_mpu'])
                
                elapsed = time.time() - start_time
                progress = int(20 * elapsed / ACCEL_CALIBRATION_TIME)
                print(f"\rProgress: [{'='*progress}{' '*(20-progress)}] "
                      f"{elapsed:.1f}/{ACCEL_CALIBRATION_TIME:.1f}s", end="")
                
                time.sleep(1.0 / SAMPLE_RATE)
            
            print()  # New line
            
            # Calculate mean for this orientation
            mean_accel = np.mean(samples, axis=0)
            accel_measurements.append(mean_accel)
            expected_vectors.append(expected_vector * 9.80665)  # Expected in m/s²
            
            print(f"Measured: [{mean_accel[0]:.3f}, {mean_accel[1]:.3f}, {mean_accel[2]:.3f}] m/s²")
        
        # Solve for bias and scale
        self._solve_accelerometer_calibration(
            np.array(accel_measurements),
            np.array(expected_vectors)
        )
        
        print(f"\nAccelerometer calibration complete:")
        print(f"Bias: [{self.calibration.accel_bias[0]:.6f}, "
              f"{self.calibration.accel_bias[1]:.6f}, {self.calibration.accel_bias[2]:.6f}] m/s²")
        print(f"Scale: [{self.calibration.accel_scale[0]:.6f}, "
              f"{self.calibration.accel_scale[1]:.6f}, {self.calibration.accel_scale[2]:.6f}]")
    
    def _solve_accelerometer_calibration(self, measurements: np.ndarray, 
                                       expected: np.ndarray) -> None:
        """Solve for accelerometer bias and scale factors."""
        def residuals(params):
            bias = params[:3]
            scale = params[3:6]
            
            # Apply calibration to measurements
            calibrated = (measurements - bias) * scale
            
            # Calculate residuals
            return (calibrated - expected).flatten()
        
        # Initial guess
        x0 = np.concatenate([np.zeros(3), np.ones(3)])
        
        # Solve optimization problem
        result = least_squares(residuals, x0, method='lm')
        
        if result.success:
            self.calibration.accel_bias = result.x[:3]
            self.calibration.accel_scale = result.x[3:6]
            
            # Calculate RMS error
            final_residuals = residuals(result.x).reshape(-1, 3)
            rms_error = np.sqrt(np.mean(np.sum(final_residuals**2, axis=1)))
            self.calibration.quality_metrics['accel_rms_error'] = float(rms_error)
            
            print(f"✓ Accelerometer calibration successful (RMS error: {rms_error:.3f} m/s²)")
        else:
            print("❌ Accelerometer calibration failed!")
            print("Using default values.")
    
    def calibrate_magnetometer(self) -> None:
        """Calibrate magnetometer using ellipsoid fitting."""
        print("Magnetometer calibration requires moving the IMU through all orientations.")
        print("You will need to:")
        print("1. Rotate the IMU slowly around all three axes")
        print("2. Try to cover as many orientations as possible")
        print("3. Avoid metal objects and electronic devices")
        print(f"\nData collection will last {MAG_CALIBRATION_TIME} seconds.")
        
        input("Press Enter to start magnetometer calibration...")
        
        print(f"\nRotate the IMU through all orientations for {MAG_CALIBRATION_TIME} seconds...")
        print("Try to trace imaginary spheres and figure-8 patterns in all directions.")
        
        mag_samples = []
        start_time = time.time()
        
        while time.time() - start_time < MAG_CALIBRATION_TIME:
            data = self.sensor_manager.read_all()
            mag_samples.append(data['magnetometer'])
            
            elapsed = time.time() - start_time
            progress = int(50 * elapsed / MAG_CALIBRATION_TIME)
            print(f"\rProgress: [{'='*progress}{' '*(50-progress)}] "
                  f"{elapsed:.1f}/{MAG_CALIBRATION_TIME:.1f}s", end="")
            
            time.sleep(1.0 / SAMPLE_RATE)
        
        print()  # New line
        
        # Perform ellipsoid fitting
        mag_data = np.array(mag_samples)
        self._fit_magnetometer_ellipsoid(mag_data)
        
        # Calculate quality metrics
        mag_range = np.max(mag_data, axis=0) - np.min(mag_data, axis=0)
        self.calibration.quality_metrics['mag_range'] = mag_range.tolist()
        
        print(f"\nMagnetometer calibration complete:")
        print(f"Bias: [{self.calibration.mag_bias[0]:.2e}, "
              f"{self.calibration.mag_bias[1]:.2e}, {self.calibration.mag_bias[2]:.2e}] T")
        print(f"Scale: [{self.calibration.mag_scale[0]:.6f}, "
              f"{self.calibration.mag_scale[1]:.6f}, {self.calibration.mag_scale[2]:.6f}]")
        
        if np.min(mag_range) < 20e-6:  # 20 µT minimum range
            print("WARNING: Limited magnetometer range detected!")
            print("Consider repeating calibration with more comprehensive movements.")
        else:
            print("✓ Magnetometer calibration successful")
    
    def _fit_magnetometer_ellipsoid(self, mag_data: np.ndarray) -> None:
        """Fit an ellipsoid to magnetometer data for hard/soft iron correction."""
        # Simplified ellipsoid fitting - for production, use more robust algorithm
        
        # Step 1: Estimate center (hard iron bias)
        self.calibration.mag_bias = np.mean(mag_data, axis=0)
        
        # Step 2: Center the data
        centered_data = mag_data - self.calibration.mag_bias
        
        # Step 3: Estimate scale factors (simplified approach)
        # In a perfect sphere, all axes should have the same magnitude range
        ranges = np.max(centered_data, axis=0) - np.min(centered_data, axis=0)
        avg_range = np.mean(ranges)
        self.calibration.mag_scale = avg_range / ranges
        
        # Step 4: Soft iron correction matrix (simplified - identity for basic cal)
        # For production use, implement full ellipsoid fitting with rotation matrix
        self.calibration.mag_soft_iron = np.eye(3)
        
        # Advanced users could implement proper ellipsoid fitting here:
        # 1. Fit general ellipsoid equation: ax² + by² + cz² + 2fyz + 2gxz + 2hxy + 2px + 2qy + 2rz = 1
        # 2. Extract center, scale, and rotation matrix
        # 3. This requires solving a generalized eigenvalue problem
    
    def calculate_quality_metrics(self) -> None:
        """Calculate overall calibration quality metrics."""
        # Test current calibration
        print("\nCalculating calibration quality...")
        
        # Apply calibration to sensor manager
        self.sensor_manager.set_calibration(self.calibration.to_dict())
        
        # Collect some test data
        test_samples = []
        for _ in range(100):  # 0.5 seconds at 200 Hz
            data = self.sensor_manager.read_all()
            test_samples.append({
                'gyro': data['gyroscope'],
                'accel_mpu': data['accelerometer_mpu'],
                'mag': data['magnetometer']
            })
            time.sleep(1.0 / SAMPLE_RATE)
        
        # Calculate noise levels
        gyro_noise = np.std([s['gyro'] for s in test_samples], axis=0)
        accel_noise = np.std([s['accel_mpu'] for s in test_samples], axis=0)
        mag_noise = np.std([s['mag'] for s in test_samples], axis=0)
        
        self.calibration.quality_metrics.update({
            'gyro_noise_rms': float(np.sqrt(np.mean(gyro_noise**2))),
            'accel_noise_rms': float(np.sqrt(np.mean(accel_noise**2))),
            'mag_noise_rms': float(np.sqrt(np.mean(mag_noise**2)))
        })
    
    def print_calibration_summary(self) -> None:
        """Print a summary of calibration results."""
        print(f"Calibration timestamp: {time.ctime(self.calibration.timestamp)}")
        print("\nQuality Metrics:")
        
        metrics = self.calibration.quality_metrics
        if 'gyro_noise_rms' in metrics:
            print(f"  Gyroscope noise RMS: {metrics['gyro_noise_rms']:.6f} rad/s")
        if 'accel_noise_rms' in metrics:
            print(f"  Accelerometer noise RMS: {metrics['accel_noise_rms']:.6f} m/s²")
        if 'mag_noise_rms' in metrics:
            print(f"  Magnetometer noise RMS: {metrics['mag_noise_rms']:.2e} T")
        if 'accel_rms_error' in metrics:
            print(f"  Accelerometer calibration error: {metrics['accel_rms_error']:.3f} m/s²")


def save_calibration(calibration: CalibrationData, filename: str = "imu_calibration.json") -> bool:
    """
    Save calibration data to JSON file.
    
    Args:
        calibration: CalibrationData object to save
        filename: Output filename
        
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = Path(filename)
        with filepath.open('w') as f:
            json.dump(calibration.to_dict(), f, indent=2)
        
        logging.info(f"Calibration saved to {filepath}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to save calibration: {e}")
        return False


def load_calibration(filename: str = "imu_calibration.json") -> Optional[CalibrationData]:
    """
    Load calibration data from JSON file.
    
    Args:
        filename: Input filename
        
    Returns:
        CalibrationData object if successful, None otherwise
    """
    try:
        filepath = Path(filename)
        if not filepath.exists():
            logging.warning(f"Calibration file {filepath} not found")
            return None
        
        with filepath.open('r') as f:
            data = json.load(f)
        
        calibration = CalibrationData.from_dict(data)
        logging.info(f"Calibration loaded from {filepath}")
        return calibration
        
    except Exception as e:
        logging.error(f"Failed to load calibration: {e}")
        return None


def quick_gyro_calibration(sensor_manager: SensorManager, duration: float = 10.0) -> np.ndarray:
    """
    Quick gyroscope bias calibration for development/testing.
    
    Args:
        sensor_manager: SensorManager instance
        duration: Calibration duration in seconds
        
    Returns:
        Gyroscope bias vector
    """
    logging.info(f"Quick gyro calibration for {duration} seconds...")
    
    samples = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        data = sensor_manager.read_all()
        samples.append(data['gyroscope'])
        time.sleep(1.0 / SAMPLE_RATE)
    
    bias = np.mean(samples, axis=0)
    logging.info(f"Gyro bias: {bias}")
    
    return bias


def validate_calibration(sensor_manager: SensorManager, calibration: CalibrationData) -> Dict[str, float]:
    """
    Validate calibration quality by checking sensor outputs.
    
    Args:
        sensor_manager: SensorManager instance
        calibration: Calibration to validate
        
    Returns:
        Dictionary of validation metrics
    """
    # Apply calibration
    sensor_manager.set_calibration(calibration.to_dict())
    
    # Collect validation data
    print("Validating calibration - keep IMU stationary...")
    samples = []
    
    for _ in range(1000):  # 5 seconds at 200 Hz
        data = sensor_manager.read_all()
        samples.append({
            'gyro': data['gyroscope'],
            'accel': data['accelerometer_mpu'],
            'mag': data['magnetometer']
        })
        time.sleep(1.0 / SAMPLE_RATE)
    
    # Calculate validation metrics
    gyro_data = np.array([s['gyro'] for s in samples])
    accel_data = np.array([s['accel'] for s in samples])
    mag_data = np.array([s['mag'] for s in samples])
    
    metrics = {
        'gyro_bias_rms': float(np.sqrt(np.mean(np.mean(gyro_data, axis=0)**2))),
        'gyro_noise_rms': float(np.sqrt(np.mean(np.std(gyro_data, axis=0)**2))),
        'accel_magnitude_mean': float(np.mean(np.linalg.norm(accel_data, axis=1))),
        'accel_magnitude_std': float(np.std(np.linalg.norm(accel_data, axis=1))),
        'mag_magnitude_mean': float(np.mean(np.linalg.norm(mag_data, axis=1))),
        'mag_magnitude_std': float(np.std(np.linalg.norm(mag_data, axis=1)))
    }
    
    return metrics


if __name__ == "__main__":
    """Interactive calibration script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="IMU Calibration Tool")
    parser.add_argument("--output", "-o", default="imu_calibration.json",
                       help="Output calibration file")
    parser.add_argument("--quick-gyro", action="store_true",
                       help="Run quick gyroscope calibration only")
    parser.add_argument("--validate", "-v", 
                       help="Validate existing calibration file")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Initialize sensor manager
    sensor_manager = SensorManager()
    
    if args.validate:
        # Validate existing calibration
        calibration = load_calibration(args.validate)
        if calibration:
            metrics = validate_calibration(sensor_manager, calibration)
            print("\nValidation Results:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.6f}")
        else:
            print(f"Failed to load calibration file: {args.validate}")
    
    elif args.quick_gyro:
        # Quick gyroscope calibration
        bias = quick_gyro_calibration(sensor_manager)
        print(f"Gyroscope bias: {bias}")
        
        # Save minimal calibration
        cal = CalibrationData()
        cal.gyro_bias = bias
        save_calibration(cal, args.output)
        
    else:
        # Full interactive calibration
        calibrator = InteractiveCalibrator(sensor_manager)
        calibration = calibrator.run_full_calibration()
        
        # Save calibration
        if save_calibration(calibration, args.output):
            print(f"\nCalibration saved to: {args.output}")
        else:
            print("\nFailed to save calibration file!") 