# Raspberry Pi Setup Guide for INS

## Hardware Wiring

### Pin Connections
```
Raspberry Pi 4B → MPU-6050:
Pin 1  (3.3V) → VCC
Pin 3  (SDA)  → SDA  
Pin 5  (SCL)  → SCL
Pin 6  (GND)  → GND
             → AD0 to GND (sets address to 0x68)

Raspberry Pi 4B → LSM303DLHC:
Pin 1  (3.3V) → VCC
Pin 3  (SDA)  → SDA
Pin 5  (SCL)  → SCL  
Pin 6  (GND)  → GND
```

### I2C Pull-up Resistors
Add 4.7kΩ resistors:
- SDA (Pin 3) to 3.3V (Pin 1)
- SCL (Pin 5) to 3.3V (Pin 1)

**Note**: Many breakout boards have built-in pull-ups, so external resistors may not be needed.

## Software Setup

### 1. Update Raspberry Pi
```bash
sudo apt update && sudo apt upgrade -y
sudo reboot
```

### 2. Enable I2C
```bash
# Enable I2C interface
sudo raspi-config
# Navigate: Interface Options → I2C → Enable → Finish

# Install I2C tools
sudo apt install i2c-tools python3-pip python3-venv
```

### 3. Create Project Environment
```bash
# Create project directory
mkdir -p ~/navigation && cd ~/navigation

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy scipy matplotlib
```

### 4. Install smbus2
```bash
# For hardware I2C access
pip install smbus2

# Alternative if smbus2 has issues:
# sudo apt install python3-smbus
```

### 5. Copy Project Files
Transfer your project files to ~/navigation/:
```bash
# Option 1: Copy files directly
scp -r mpu6250/* pi@raspberry-pi-ip:~/navigation/

# Option 2: Clone from repository
git clone <your-repo-url> ~/navigation/
cd ~/navigation
```

## Testing Steps

### 1. Hardware Verification
```bash
# Check I2C devices are detected
sudo i2cdetect -y 1

# Expected output shows:
# 19 (LSM303 accel), 1e (LSM303 mag), 68 (MPU-6050)
```

### 2. Test Basic Connectivity
```bash
cd ~/navigation
source venv/bin/activate

# Quick hardware test
python3 -c "
from imu_drivers import SensorManager
try:
    manager = SensorManager()
    data = manager.read_all()
    print('✓ Sensors connected and working!')
    print(f'  Gyro: {data[\"gyroscope\"]}')
    print(f'  Accel: {data[\"accelerometer_mpu\"]}')
    print(f'  Mag: {data[\"magnetometer\"]}')
except Exception as e:
    print('✗ Error:', e)
"
```

### 3. Run Self-Tests
```bash
# Test sensor self-diagnostics
python3 -c "
from imu_drivers import SensorManager
manager = SensorManager()
results = manager.self_test()
print('Self-test results:', results)
"
```

### 4. Sensor Calibration
```bash
# Interactive calibration (5-10 minutes)
python3 calibration.py

# Follow the prompts:
# 1. Place IMU flat and still for gyro bias
# 2. Position in 6 orientations for accelerometer  
# 3. Rotate through all orientations for magnetometer
```

### 5. Test Navigation System
```bash
# Run in simulation mode first
python3 main.py --simulation --log test_sim.csv

# Then test with real hardware
python3 main.py --calibration imu_calibration.json --log test_real.csv
```

### 6. Monitor Performance
```bash
# Check system resources while running
htop

# View real-time data stream
nc localhost 8888
```

## Verification Checklist

### Hardware Checks
- [ ] All sensors detected by i2cdetect
- [ ] No I2C communication errors in logs
- [ ] Stable power supply (3.3V)
- [ ] Good electrical connections

### Software Checks  
- [ ] All Python dependencies installed
- [ ] No import errors
- [ ] Sensor self-tests pass
- [ ] Calibration completes successfully

### Performance Checks
- [ ] 200 Hz sample rate achieved
- [ ] <25% CPU usage
- [ ] No missed deadlines in logs
- [ ] Network streaming works

## Common Issues & Fixes

### I2C Not Working
```bash
# Check I2C is enabled
ls /dev/i2c*  # Should show /dev/i2c-1

# Check user permissions
sudo usermod -a -G i2c $USER
# Log out and back in

# Check for device tree issues
sudo nano /boot/config.txt
# Ensure: dtparam=i2c_arm=on
```

### Sensors Not Detected
```bash
# Check wiring connections
# Verify power (3.3V, not 5V)
# Check for loose connections
# Try different I2C bus speed:
sudo nano /boot/config.txt
# Add: dtparam=i2c_arm_baudrate=100000
sudo reboot
```

### Performance Issues
```bash
# Reduce sample rate
python3 main.py --sample-rate 100

# Disable networking
python3 main.py --no-network

# Check for I2C errors
dmesg | grep i2c
```

### Import Errors
```bash
# Reinstall dependencies
pip install --force-reinstall numpy scipy smbus2

# Check Python path
python3 -c "import sys; print(sys.path)"
```

## Production Deployment

### Systemd Service
Create `/etc/systemd/system/ins.service`:
```ini
[Unit]
Description=Inertial Navigation System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/navigation
Environment=PATH=/home/pi/navigation/venv/bin
ExecStart=/home/pi/navigation/venv/bin/python main.py --calibration imu_calibration.json --log /var/log/ins/navigation.log
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable ins.service
sudo systemctl start ins.service
sudo systemctl status ins.service
```

### Log Rotation
```bash
# Create log directory
sudo mkdir -p /var/log/ins
sudo chown pi:pi /var/log/ins

# Setup logrotate
sudo nano /etc/logrotate.d/ins
```

Add:
```
/var/log/ins/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

## Remote Monitoring

### SSH Access
```bash
# Connect remotely
ssh pi@raspberry-pi-ip

# Monitor logs
tail -f /var/log/ins/navigation.log
```

### Web Interface (Optional)
```bash
# Simple web viewer for real-time data
python3 -c "
import socket, json, time
s = socket.socket()
s.connect(('localhost', 8888))
while True:
    data = s.recv(1024).decode()
    if data:
        state = json.loads(data)
        print(f'Position: {state[\"position_ned\"]}')
        print(f'Attitude: {state[\"euler_deg\"]}')
    time.sleep(1)
"
``` 