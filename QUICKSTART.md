# Quick Start Reference Card

## 🚀 One-Line Setup
```bash
# On Raspberry Pi, run this to set up everything automatically:
curl -sSL https://raw.githubusercontent.com/your-repo/main/setup_rpi.sh | bash
```

## 🔧 Manual Setup Steps
```bash
# 1. Enable I2C and install dependencies
sudo raspi-config  # Enable I2C in Interface Options
sudo apt install i2c-tools python3-pip python3-venv

# 2. Create project and install dependencies  
mkdir ~/navigation && cd ~/navigation
python3 -m venv venv && source venv/bin/activate
pip install numpy scipy smbus2 matplotlib websockets

# 3. Copy project files to ~/navigation/
```

## ✅ Hardware Testing
```bash
cd ~/navigation && source venv/bin/activate

# Check I2C devices detected
sudo i2cdetect -y 1
# Should show: 19, 1e, 68

# Run comprehensive hardware test
python3 test_hardware.py

# Quick sensor test
python3 -c "from imu_drivers import SensorManager; print('OK' if SensorManager().read_all() else 'FAIL')"
```

## 📐 Calibration Commands
```bash
# Full interactive calibration (5-10 minutes)
python3 calibration.py

# Quick gyro-only calibration (30 seconds)  
python3 calibration.py --quick-gyro

# Validate existing calibration
python3 calibration.py --validate imu_calibration.json
```

## 🧭 Navigation Commands
```bash
# Test in simulation mode (no hardware required)
python3 main.py --simulation

# Run with real hardware
python3 main.py --calibration imu_calibration.json

# Run with logging
python3 main.py --log flight_data.csv

# Custom sample rate
python3 main.py --sample-rate 100

# Display real-time attitude (roll/pitch/yaw at every EKF iteration)
python3 main.py --show-attitude

# Reduce drift with relaxed ZUPT (for uncalibrated/noisy sensors)
python3 main.py --zupt-relaxed --show-attitude

# Disable networking
python3 main.py --no-network
```

## 📊 Monitoring & Data Access
```bash
# View real-time data stream
nc localhost 8888

# Monitor system performance
htop

# Check logs with timestamp
journalctl -f -u ins.service

# View CSV data
head -n 5 flight_data.csv
```

## 🔍 Troubleshooting Commands
```bash
# Check I2C bus
sudo i2cdetect -y 1

# Check I2C permissions
groups $USER  # Should include 'i2c'

# Test Python imports
python3 -c "import numpy, scipy, smbus2; print('All imports OK')"

# Check sensor connections
python3 -c "from imu_drivers import SensorManager; SensorManager().self_test()"

# Monitor I2C errors
dmesg | grep i2c

# Check service status
sudo systemctl status ins.service
```

## 🐛 Common Issues & Fixes

### Sensors Not Detected
```bash
# Check wiring (3.3V not 5V!)
# Ensure AD0 pin on MPU-6050 is connected to GND
# Try slower I2C speed:
echo 'dtparam=i2c_arm_baudrate=100000' | sudo tee -a /boot/config.txt
sudo reboot
```

### Permission Errors
```bash
sudo usermod -a -G i2c $USER
# Log out and back in
```

### High CPU Usage
```bash
# Reduce sample rate
python3 main.py --sample-rate 50

# Check for I2C errors
dmesg | tail
```

### Import Errors
```bash
# Reinstall in virtual environment
cd ~/navigation && source venv/bin/activate
pip install --force-reinstall numpy scipy smbus2
```

## 📡 Network Integration

### TCP Stream (Port 8888)
```bash
# Python client
import socket, json
s = socket.socket()
s.connect(('raspberrypi-ip', 8888))
data = json.loads(s.recv(1024).decode())
print(f"Position: {data['position_ned']}")
```

### WebSocket Stream (Port 8889)
```javascript
// JavaScript client
const ws = new WebSocket('ws://raspberrypi-ip:8889');
ws.onmessage = (event) => {
    const state = JSON.parse(event.data);
    console.log('Attitude:', state.euler_deg);
};
```

## 🔄 Production Deployment
```bash
# Install as system service
sudo cp ins.service /etc/systemd/system/
sudo systemctl enable ins.service
sudo systemctl start ins.service

# Check service status
sudo systemctl status ins.service

# View logs
sudo journalctl -u ins.service -f
```

## 📊 Data Analysis
```python
# Load and plot CSV data
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('flight_data.csv')

# Plot trajectory
plt.plot(data['pos_n'], data['pos_e'])
plt.xlabel('North [m]')
plt.ylabel('East [m]')
plt.show()

# Plot attitude
plt.figure(figsize=(12,4))
plt.subplot(131); plt.plot(data['roll_deg']); plt.title('Roll')
plt.subplot(132); plt.plot(data['pitch_deg']); plt.title('Pitch') 
plt.subplot(133); plt.plot(data['yaw_deg']); plt.title('Yaw')
plt.show()
```

## 🧪 Testing Commands
```bash
# Run unit tests
python3 -m pytest tests/ -v

# Test specific module
python3 -m pytest tests/test_ekf.py -v

# Run with coverage
python3 -m pytest tests/ --cov=. --cov-report=html

# Test EKF performance
python3 ekf.py  # Generates plots
```

## ⚙️ Configuration Files

### config.json
```json
{
  "sample_rate": 200.0,
  "magnetic_reference": [20e-6, 0, 45e-6],
  "log_file": "navigation.csv",
  "tcp_port": 8888
}
```

### Launch with config
```bash
python3 main.py --config config.json
```

## 📚 Useful File Locations
- Project: `~/navigation/`
- Calibration: `~/navigation/imu_calibration.json`
- Logs: `/var/log/ins/navigation.log`
- Service: `/etc/systemd/system/ins.service`
- Config: `/boot/config.txt`

## 🆘 Emergency Reset
```bash
# Stop everything and reset
sudo systemctl stop ins.service
pkill -f main.py
cd ~/navigation
git checkout -- .  # Reset all files
python3 test_hardware.py  # Retest
``` 