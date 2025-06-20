# Inertial Navigation System (INS) for Raspberry Pi

A production-ready, real-time inertial navigation system using dual IMU sensors and an Extended Kalman Filter for attitude, velocity, and position estimation.

## Features

- **Dual IMU Configuration**: MPU-6050 (gyro + accel) + LSM303DLHC (accel + mag)
- **15-State Extended Kalman Filter**: Real-time attitude (quaternion), velocity, and position estimation
- **Advanced Calibration**: Interactive multi-point calibration with JSON persistence
- **Zero Velocity Updates (ZUPT)**: Automatic stationary detection and drift correction
- **High Performance**: 200 Hz processing, <25% CPU usage, <50 MB RAM
- **Network Publishing**: Real-time state streaming via TCP and WebSocket
- **Comprehensive Logging**: CSV data logging with configurable fields
- **Simulation Mode**: Built-in motion simulator for testing and validation
- **Production Ready**: Extensive unit tests, error handling, and documentation

## Hardware Requirements

### Sensors
- **MPU-6050**: 6-axis IMU (3-axis gyroscope + 3-axis accelerometer)
- **LSM303DLHC**: 6-axis IMU (3-axis accelerometer + 3-axis magnetometer)
- **Raspberry Pi**: Tested on Pi 4B with Bullseye 64-bit

### Wiring Diagram

```
Raspberry Pi 4B                    MPU-6050                    LSM303DLHC
┌─────────────────┐                ┌──────────┐                ┌──────────┐
│                 │                │          │                │          │
│  Pin 1  (3.3V)  ├────────────────┤ VCC      │                │ VCC      ├──┐
│  Pin 3  (SDA)   ├────┬───────────┤ SDA      │                │ SDA      ├──┤
│  Pin 5  (SCL)   ├────┼─────┬─────┤ SCL      │                │ SCL      ├──┤
│  Pin 6  (GND)   ├────┼─────┼─────┤ GND      │                │ GND      ├──┤
│  Pin 9  (GND)   │    │     │     │          │                │          │  │
│                 │    │     │     │ AD0 ──┐  │                └──────────┘  │
│                 │    │     │     │       │  │                              │
│                 │    │     │     │ INT ──┼──┼─── Not Connected            │
│                 │    │     │     └───────┼──┘                              │
└─────────────────┘    │     │             │                                 │
                       │     │             └─── GND (sets I2C addr 0x68)    │
        ┌──────────────┴─────┴─────────────────────────────────────────────┴─┐
        │                                                                     │
        │                     I2C Bus (with pull-up resistors)               │
        │                     SDA: 4.7kΩ to 3.3V                            │
        │                     SCL: 4.7kΩ to 3.3V                            │
        └─────────────────────────────────────────────────────────────────────┘

I2C Addresses:
- MPU-6050: 0x68 (AD0 connected to GND)
- LSM303DLHC Accelerometer: 0x19
- LSM303DLHC Magnetometer: 0x1E
```

### I2C Setup on Raspberry Pi

1. Enable I2C interface:
```bash
sudo raspi-config
# Navigate to: Interfacing Options → I2C → Enable
```

2. Install I2C tools:
```bash
sudo apt update
sudo apt install i2c-tools python3-smbus
```

3. Verify sensor connections:
```bash
# Scan I2C bus
sudo i2cdetect -y 1

# Expected output:
#      0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
# 00:          -- -- -- -- -- -- -- -- -- -- -- -- --
# 10: -- -- -- -- -- -- -- -- -- 19 -- -- -- -- 1e --
# 20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# 30: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# 40: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# 50: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# 60: -- -- -- -- -- -- -- -- 68 -- -- -- -- -- -- --
# 70: -- -- -- -- -- -- -- --
```

## Software Installation

### Dependencies

```bash
# Install Python dependencies
pip3 install numpy scipy smbus2

# Optional: For advanced features
pip3 install matplotlib websockets pytest

# For testing without hardware
pip3 install pytest-mock
```

### Installation

```bash
# Clone or download the project
git clone <repository-url>
cd mpu6250

# Make scripts executable
chmod +x main.py calibration.py

# Install system-wide (optional)
sudo pip3 install -e .
```

## Quick Start

### 1. Hardware Verification

```bash
# Test sensor connections
python3 -c "
from imu_drivers import SensorManager
manager = SensorManager()
data = manager.read_all()
print('Sensors working!' if data else 'Check connections')
"
```

### 2. Sensor Calibration

**Important**: Proper calibration is essential for accurate navigation.

```bash
# Run interactive calibration (5-10 minutes)
python3 calibration.py

# Quick gyroscope-only calibration (30 seconds)
python3 calibration.py --quick-gyro
```

Follow the on-screen instructions for:
- **Gyroscope bias**: Keep IMU stationary for 30 seconds
- **Accelerometer scale/bias**: Position IMU in 6 orientations (±X, ±Y, ±Z up)
- **Magnetometer hard/soft iron**: Rotate IMU through all orientations for 60 seconds

### 3. Run Navigation System

```bash
# Basic operation with default settings
python3 main.py

# With custom configuration
python3 main.py --calibration my_calibration.json --log navigation_data.csv

# Real-time attitude display (shows roll/pitch/yaw at every EKF iteration)
python3 main.py --show-attitude --sample-rate 100

# Reduce drift with relaxed ZUPT thresholds (for uncalibrated sensors)
python3 main.py --zupt-relaxed --show-attitude

# Simulation mode (for testing without hardware)
python3 main.py --simulation
```

## Usage Examples

### Real-time Navigation

```python
from main import INSApplication, create_default_config
import time

# Create configuration
config = create_default_config()
config['calibration_file'] = 'my_calibration.json'
config['log_file'] = 'flight_data.csv'

# Create and start application
app = INSApplication(config)
app.start()  # Runs until Ctrl+C
```

### Custom EKF Usage

```python
from ekf import InertialEKF
from imu_drivers import SensorManager
import numpy as np

# Initialize components
ekf = InertialEKF()
sensors = SensorManager()

# Main processing loop
while True:
    # Read sensors
    data = sensors.read_all()
    
    # EKF prediction
    ekf.predict(
        data['gyroscope'],
        data['accelerometer_mpu'], 
        data['dt']
    )
    
    # EKF updates (lower rate)
    if loop_count % 10 == 0:
        ekf.update_gravity(data['accelerometer_mpu'])
        ekf.update_magnetometer(data['magnetometer'], mag_reference)
    
    # Zero velocity update
    ekf.update_zupt(data['accelerometer_mpu'], data['gyroscope'])
    
    # Get current state
    state = ekf.get_state_dict()
    print(f"Position: {state['position_ned']}")
    print(f"Attitude: {state['euler_deg']}")
```

## Configuration

### Command Line Options

```bash
python3 main.py --help

Options:
  --config CONFIG          Configuration file (JSON)
  --calibration FILE       Calibration file
  --log FILE              Output log file (CSV)
  --sample-rate HZ        Sensor sample rate (default: 200)
  --simulation            Run in simulation mode
  --no-network           Disable network publishing
  --show-attitude        Display roll/pitch/yaw at every EKF iteration
  --zupt-relaxed         Use relaxed ZUPT thresholds for noisy/uncalibrated sensors
  --tcp-port PORT        TCP port for state publishing (default: 8888)
  --websocket-port PORT  WebSocket port (default: 8889)
  --log-level LEVEL      Logging level (DEBUG/INFO/WARNING/ERROR)
```

### Configuration File

Create a `config.json` file:

```json
{
  "i2c_bus": 1,
  "sample_rate": 200.0,
  "gyro_noise_density": 1e-4,
  "accel_noise_density": 1e-3,
  "gyro_bias_stability": 1e-6,
  "accel_bias_stability": 1e-5,
  "magnetic_reference": [20e-6, 0, 45e-6],
  "calibration_file": "imu_calibration.json",
  "log_file": "navigation_data.csv",
  "enable_networking": true,
  "tcp_port": 8888,
  "websocket_port": 8889
}
```

## Network Interface

### TCP Stream

Connect to port 8888 for real-time JSON state updates:

```bash
# View live data
nc localhost 8888

# Example output:
{
  "timestamp": 1234567890.123,
  "quaternion": [1.0, 0.0, 0.0, 0.0],
  "euler_deg": [0.0, 0.0, 0.0],
  "velocity_ned": [0.0, 0.0, 0.0],
  "position_ned": [0.0, 0.0, 0.0],
  "statistics": {"predictions": 12000, "updates": 1200}
}
```

### WebSocket Stream

```javascript
// Connect via WebSocket
const ws = new WebSocket('ws://raspberrypi:8889');
ws.onmessage = function(event) {
    const state = JSON.parse(event.data);
    console.log('Position:', state.position_ned);
    console.log('Attitude:', state.euler_deg);
};
```

## Data Logging

CSV files contain timestamped sensor and navigation data:

```csv
timestamp,dt,q_w,q_x,q_y,q_z,roll_deg,pitch_deg,yaw_deg,vel_n,vel_e,vel_d,pos_n,pos_e,pos_d,...
1234567890.123,0.005,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,...
```

### Post-processing

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load navigation data
data = pd.read_csv('navigation_data.csv')

# Plot trajectory
plt.figure(figsize=(10, 8))
plt.plot(data['pos_n'], data['pos_e'])
plt.xlabel('North [m]')
plt.ylabel('East [m]')
plt.title('Vehicle Trajectory')
plt.grid(True)
plt.show()
```

## Testing

### Unit Tests

```bash
# Run all tests
python3 -m pytest tests/

# Run specific test modules
python3 -m pytest tests/test_ekf.py -v

# Run with coverage
python3 -m pytest tests/ --cov=. --cov-report=html
```

### EKF Validation

```bash
# Test EKF with motion simulator
python3 ekf.py

# This generates ekf_test_results.png with performance plots
```

### Calibration Validation

```bash
# Validate existing calibration
python3 calibration.py --validate imu_calibration.json
```

## Performance Optimization

### Raspberry Pi Settings

```bash
# Increase I2C bus speed (optional)
sudo nano /boot/config.txt
# Add: dtparam=i2c_arm_baudrate=400000

# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable wifi-powersave

# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### Real-time Priority

```bash
# Run with higher priority (use carefully)
sudo nice -n -10 python3 main.py
```

## Troubleshooting

### Common Issues

1. **Sensors not detected**
   ```bash
   # Check I2C connections
   sudo i2cdetect -y 1
   
   # Check permissions
   sudo usermod -a -G i2c $USER
   # Log out and back in
   ```

2. **High CPU usage**
   - Reduce sample rate: `--sample-rate 100`
   - Disable networking: `--no-network`
   - Check for I2C bus errors in logs

3. **Poor navigation accuracy**
   - Recalibrate sensors
   - Check magnetic environment (avoid metal objects)
   - Tune EKF noise parameters

4. **Navigation drift**
   - Ensure ZUPT is working (vehicle should be stationary periodically)
   - Check accelerometer calibration quality
   - Consider GPS integration for absolute position reference

### Debug Mode

```bash
# Enable debug logging
python3 main.py --log-level DEBUG

# Monitor system resources
htop
iotop
```

## Theory of Operation

### Coordinate Frames

- **Body Frame**: Fixed to IMU, right-handed (X-forward, Y-right, Z-down)
- **NED Frame**: Navigation frame (North-East-Down)
- **Sensor Frame**: Individual sensor coordinate systems

### State Vector

The EKF estimates a 15-element state vector:
```
x = [q0, q1, q2, q3, vN, vE, vD, pN, pE, pD, bgx, bgy, bgz, bax, bay, baz]
```

Where:
- `q0-q3`: Unit quaternion representing attitude
- `vN,vE,vD`: Velocity in NED frame [m/s]
- `pN,pE,pD`: Position in NED frame [m]
- `bgx,bgy,bgz`: Gyroscope biases [rad/s]
- `bax,bay,baz`: Accelerometer biases [m/s²]

### Measurement Updates

1. **Gravity Vector**: Uses accelerometer to correct attitude
2. **Magnetic Field**: Uses magnetometer for yaw reference
3. **Zero Velocity**: Constrains velocity when stationary

## Extension Points

### Adding GPS

```python
def update_gps(self, gps_position, gps_velocity):
    """Update EKF with GPS measurements."""
    # Implementation for GPS position/velocity updates
    pass
```

### Adding Barometer

```python
def update_barometer(self, altitude):
    """Update EKF with barometric altitude."""
    # Implementation for altitude constraint
    pass
```

### Custom Sensors

Extend the `SensorManager` class to add additional sensors.

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

- Create an issue on GitHub for bug reports
- Discussion forum for questions and tips
- Wiki for additional documentation

## References

1. "Strapdown Inertial Navigation Technology" - D.H. Titterton
2. "Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems" - P.D. Groves
3. MPU-6050 Product Specification - InvenSense
4. LSM303DLHC Datasheet - STMicroelectronics 