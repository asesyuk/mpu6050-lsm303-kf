#!/usr/bin/env python3
"""
Quick hardware test script for IMU sensors.
Run this to verify your sensors are connected and working properly.
"""

import sys
import time
import numpy as np

def test_i2c_devices():
    """Test if I2C devices are detected."""
    print("🔍 Testing I2C device detection...")
    try:
        import subprocess
        result = subprocess.run(['i2cdetect', '-y', '1'], 
                              capture_output=True, text=True)
        output = result.stdout
        
        # Check for expected addresses
        has_mpu6050 = '68' in output
        has_lsm_accel = '19' in output  
        has_lsm_mag = '1e' in output
        
        print(f"  MPU-6050 (0x68):     {'✓' if has_mpu6050 else '✗'}")
        print(f"  LSM303 Accel (0x19): {'✓' if has_lsm_accel else '✗'}")
        print(f"  LSM303 Mag (0x1E):   {'✓' if has_lsm_mag else '✗'}")
        
        return has_mpu6050 and has_lsm_accel and has_lsm_mag
        
    except Exception as e:
        print(f"  ✗ Error running i2cdetect: {e}")
        return False

def test_sensor_import():
    """Test if sensor modules can be imported."""
    print("\n📦 Testing module imports...")
    try:
        from imu_drivers import SensorManager, MPU6050, LSM303DLHC
        print("  ✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False

def test_sensor_initialization():
    """Test sensor initialization."""
    print("\n🔧 Testing sensor initialization...")
    try:
        from imu_drivers import SensorManager
        manager = SensorManager()
        print("  ✓ SensorManager created successfully")
        return manager
    except Exception as e:
        print(f"  ✗ Initialization error: {e}")
        return None

def test_sensor_reading(manager, num_samples=10):
    """Test sensor data reading."""
    print(f"\n📊 Testing sensor readings ({num_samples} samples)...")
    
    try:
        samples = []
        for i in range(num_samples):
            data = manager.read_all()
            samples.append(data)
            
            if i == 0:  # Show first sample details
                print(f"  Sample format: {list(data.keys())}")
                print(f"  Gyro shape: {data['gyroscope'].shape}")
                print(f"  Accel shape: {data['accelerometer_mpu'].shape}")
                print(f"  Mag shape: {data['magnetometer'].shape}")
            
            time.sleep(0.01)  # 100 Hz sampling
        
        print(f"  ✓ Successfully read {len(samples)} samples")
        return samples
        
    except Exception as e:
        print(f"  ✗ Reading error: {e}")
        return None

def analyze_sensor_data(samples):
    """Analyze sensor data for reasonableness."""
    print("\n🔬 Analyzing sensor data...")
    
    if not samples:
        print("  ✗ No samples to analyze")
        return False
    
    try:
        # Extract data arrays
        gyro_data = np.array([s['gyroscope'] for s in samples])
        accel_data = np.array([s['accelerometer_mpu'] for s in samples])
        mag_data = np.array([s['magnetometer'] for s in samples])
        
        # Check for reasonable values
        gyro_rms = np.sqrt(np.mean(gyro_data**2, axis=0))
        accel_mean = np.mean(accel_data, axis=0)
        accel_magnitude = np.linalg.norm(accel_mean)
        mag_magnitude = np.linalg.norm(np.mean(mag_data, axis=0))
        
        print(f"  Gyro RMS: [{gyro_rms[0]:.4f}, {gyro_rms[1]:.4f}, {gyro_rms[2]:.4f}] rad/s")
        print(f"  Accel magnitude: {accel_magnitude:.2f} m/s² (expect ~9.8)")
        print(f"  Mag magnitude: {mag_magnitude*1e6:.1f} µT (expect 20-60)")
        
        # Check for reasonable ranges
        checks = {
            "Gyro noise reasonable": np.all(gyro_rms < 0.1),  # < 0.1 rad/s
            "Accel magnitude OK": 8.0 < accel_magnitude < 12.0,  # Close to gravity
            "Mag magnitude OK": 20e-6 < mag_magnitude < 80e-6,  # Earth field range
            "No NaN values": np.all(np.isfinite(gyro_data)) and 
                           np.all(np.isfinite(accel_data)) and 
                           np.all(np.isfinite(mag_data))
        }
        
        for check, passed in checks.items():
            print(f"  {check}: {'✓' if passed else '✗'}")
        
        return all(checks.values())
        
    except Exception as e:
        print(f"  ✗ Analysis error: {e}")
        return False

def test_self_diagnostics(manager):
    """Test sensor self-diagnostics."""
    print("\n🩺 Running sensor self-tests...")
    
    try:
        results = manager.self_test()
        
        print(f"  MPU-6050 tests:")
        mpu_results = results.get('mpu6050', {})
        for test, passed in mpu_results.items():
            print(f"    {test}: {'✓' if passed else '✗'}")
        
        print(f"  LSM303 status: {results.get('lsm303', {}).get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Self-test error: {e}")
        return False

def test_timing_performance(manager, duration=5.0):
    """Test timing performance."""
    print(f"\n⏱️  Testing timing performance ({duration}s)...")
    
    try:
        start_time = time.time()
        sample_times = []
        
        while time.time() - start_time < duration:
            sample_start = time.time()
            data = manager.read_all()
            sample_end = time.time()
            
            sample_times.append(sample_end - sample_start)
        
        sample_times = np.array(sample_times)
        
        print(f"  Samples collected: {len(sample_times)}")
        print(f"  Average rate: {len(sample_times)/duration:.1f} Hz")
        print(f"  Read time: {np.mean(sample_times)*1000:.2f} ± {np.std(sample_times)*1000:.2f} ms")
        print(f"  Max read time: {np.max(sample_times)*1000:.2f} ms")
        
        # Check if timing is reasonable for 200 Hz operation
        avg_time = np.mean(sample_times)
        timing_ok = avg_time < 0.004  # Should be < 4ms for 200 Hz headroom
        
        print(f"  Timing suitable for 200 Hz: {'✓' if timing_ok else '✗'}")
        
        return timing_ok
        
    except Exception as e:
        print(f"  ✗ Timing test error: {e}")
        return False

def main():
    """Run all hardware tests."""
    print("🚀 IMU Hardware Test Suite")
    print("=" * 50)
    
    tests = [
        ("I2C Device Detection", test_i2c_devices),
        ("Module Import", test_sensor_import),
    ]
    
    # Run basic tests first
    for test_name, test_func in tests:
        if not test_func():
            print(f"\n❌ {test_name} failed. Check your setup and try again.")
            return False
    
    # Initialize sensors
    manager = test_sensor_initialization()
    if manager is None:
        print(f"\n❌ Sensor initialization failed. Check wiring and I2C setup.")
        return False
    
    # Run sensor-specific tests
    sensor_tests = [
        ("Self Diagnostics", lambda: test_self_diagnostics(manager)),
        ("Data Reading", lambda: test_sensor_reading(manager) is not None),
        ("Data Analysis", lambda: analyze_sensor_data(test_sensor_reading(manager, 20))),
        ("Timing Performance", lambda: test_timing_performance(manager)),
    ]
    
    all_passed = True
    for test_name, test_func in sensor_tests:
        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Your hardware setup is working correctly.")
        print("\nNext steps:")
        print("1. Run calibration: python3 calibration.py")
        print("2. Test navigation: python3 main.py --simulation")
        print("3. Run with real hardware: python3 main.py")
    else:
        print("❌ Some tests failed. Please check:")
        print("1. Sensor wiring connections")
        print("2. I2C configuration (sudo raspi-config)")
        print("3. Power supply (3.3V)")
        print("4. Python dependencies (pip install -r requirements.txt)")
    
    return all_passed

if __name__ == "__main__":
    sys.exit(0 if main() else 1) 