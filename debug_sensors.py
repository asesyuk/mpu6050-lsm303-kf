#!/usr/bin/env python3
"""
Debug script to diagnose accelerometer and sensor reading issues.
"""

import time
import numpy as np
from imu_drivers import SensorManager

def test_raw_sensors():
    """Test raw sensor readings to diagnose accelerometer issues."""
    print("=== RAW SENSOR DIAGNOSTICS ===")
    print("Keep IMU stationary on flat table")
    print("Expected: Accel magnitude ~9.8 m/s², Gyro ~0 rad/s")
    print("-" * 50)
    
    try:
        # Initialize sensor manager
        sensor_mgr = SensorManager()
        
        # Collect samples
        samples = []
        for i in range(100):
            data = sensor_mgr.read_all()
            if data:
                samples.append(data)
                
                if i % 20 == 0:  # Print every 20th sample
                    accel = data['accelerometer_mpu']
                    gyro = data['gyroscope']
                    mag = data['magnetometer']
                    
                    accel_mag = np.linalg.norm(accel)
                    gyro_mag = np.linalg.norm(gyro)
                    mag_mag = np.linalg.norm(mag) if mag is not None else 0
                    
                    print(f"Sample {i+1:3d}:")
                    print(f"  Accel: [{accel[0]:8.3f}, {accel[1]:8.3f}, {accel[2]:8.3f}] |mag| = {accel_mag:6.3f} m/s²")
                    print(f"  Gyro:  [{gyro[0]:8.4f}, {gyro[1]:8.4f}, {gyro[2]:8.4f}] |mag| = {gyro_mag:6.4f} rad/s")
                    if mag is not None:
                        print(f"  Mag:   [{mag[0]:8.1e}, {mag[1]:8.1e}, {mag[2]:8.1e}] |mag| = {mag_mag:8.1e} T")
                    print()
            
            time.sleep(0.01)  # 100 Hz
        
        if not samples:
            print("❌ ERROR: No sensor data received!")
            return
        
        # Analyze results
        print("\n=== ANALYSIS ===")
        
        accels = np.array([s['accelerometer_mpu'] for s in samples])
        gyros = np.array([s['gyroscope'] for s in samples])
        
        accel_mean = np.mean(accels, axis=0)
        accel_std = np.std(accels, axis=0)
        accel_mag_mean = np.mean([np.linalg.norm(a) for a in accels])
        
        gyro_mean = np.mean(gyros, axis=0)
        gyro_std = np.std(gyros, axis=0)
        gyro_mag_mean = np.mean([np.linalg.norm(g) for g in gyros])
        
        print(f"Accelerometer (100 samples):")
        print(f"  Mean:     [{accel_mean[0]:8.3f}, {accel_mean[1]:8.3f}, {accel_mean[2]:8.3f}] m/s²")
        print(f"  Std Dev:  [{accel_std[0]:8.3f}, {accel_std[1]:8.3f}, {accel_std[2]:8.3f}] m/s²")
        print(f"  Magnitude: {accel_mag_mean:.3f} m/s² (should be ~9.8)")
        
        print(f"\nGyroscope (100 samples):")
        print(f"  Mean:     [{gyro_mean[0]:8.4f}, {gyro_mean[1]:8.4f}, {gyro_mean[2]:8.4f}] rad/s")
        print(f"  Std Dev:  [{gyro_std[0]:8.4f}, {gyro_std[1]:8.4f}, {gyro_std[2]:8.4f}] rad/s")
        print(f"  Magnitude: {gyro_mag_mean:.4f} rad/s (should be ~0.0)")
        
        # Diagnostic assessment
        print("\n=== DIAGNOSTIC ASSESSMENT ===")
        
        if accel_mag_mean < 5.0:
            print("❌ CRITICAL: Accelerometer magnitude too low!")
            print("   Expected ~9.8 m/s², got {:.3f} m/s²".format(accel_mag_mean))
            print("   Possible causes:")
            print("   - Wrong coordinate frame transformation")
            print("   - Sensor scaling/calibration issue")
            print("   - Hardware problem")
        elif abs(accel_mag_mean - 9.8) > 2.0:
            print("⚠️  WARNING: Accelerometer magnitude off")
            print("   Expected ~9.8 m/s², got {:.3f} m/s²".format(accel_mag_mean))
        else:
            print("✅ Accelerometer magnitude OK ({:.3f} m/s²)".format(accel_mag_mean))
        
        if gyro_mag_mean > 0.1:
            print("⚠️  WARNING: High gyroscope bias ({:.4f} rad/s)".format(gyro_mag_mean))
            print("   Needs calibration")
        else:
            print("✅ Gyroscope readings OK ({:.4f} rad/s)".format(gyro_mag_mean))
        
        print("\n=== COORDINATE FRAME TEST ===")
        print("Current readings (should match orientation):")
        print("IMU flat on table (X=East, Y=North, Z=Up):")
        print(f"  X-axis (East):  {accel_mean[0]:6.2f} m/s² (should be ~0)")
        print(f"  Y-axis (North): {accel_mean[1]:6.2f} m/s² (should be ~0)")  
        print(f"  Z-axis (Up):    {accel_mean[2]:6.2f} m/s² (should be ~+9.8)")
        
        if abs(accel_mean[2] - 9.8) < abs(accel_mean[2] + 9.8):
            print("✅ Z-axis pointing UP (correct for ENU)")
        else:
            print("⚠️  Z-axis pointing DOWN (incorrect for ENU)")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("Check I2C connections and permissions")

if __name__ == "__main__":
    test_raw_sensors() 