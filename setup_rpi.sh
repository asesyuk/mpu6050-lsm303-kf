#!/bin/bash
# Raspberry Pi INS Setup Script
# Run this script to automatically set up the inertial navigation system

set -e  # Exit on any error

echo "🚀 Raspberry Pi Inertial Navigation System Setup"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please don't run this script as root"
    exit 1
fi

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    print_warning "This doesn't appear to be a Raspberry Pi"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

print_status "Step 1: Updating system packages..."
sudo apt update
sudo apt upgrade -y

print_status "Step 2: Installing system dependencies..."
sudo apt install -y \
    i2c-tools \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    htop

print_status "Step 3: Configuring I2C..."
# Enable I2C
if ! grep -q "dtparam=i2c_arm=on" /boot/config.txt; then
    echo "dtparam=i2c_arm=on" | sudo tee -a /boot/config.txt
    print_success "I2C enabled in /boot/config.txt"
else
    print_success "I2C already enabled"
fi

# Add user to i2c group
sudo usermod -a -G i2c $USER
print_success "Added $USER to i2c group"

print_status "Step 4: Setting up project directory..."
PROJECT_DIR="$HOME/navigation"
if [ ! -d "$PROJECT_DIR" ]; then
    mkdir -p "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"

print_status "Step 5: Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

print_status "Step 6: Installing Python dependencies..."
pip install --upgrade pip

# Install core dependencies
pip install numpy>=1.19.0 scipy>=1.7.0 smbus2>=0.4.0

# Install optional dependencies
pip install matplotlib>=3.3.0 websockets>=10.0 pytest>=6.0.0 pytest-mock>=3.6.0

print_success "Python dependencies installed"

print_status "Step 7: Copying project files..."
# Check if files exist in current directory
if [ -f "../imu_drivers.py" ]; then
    cp ../imu_drivers.py .
    cp ../calibration.py .
    cp ../ekf.py .
    cp ../main.py .
    cp ../test_hardware.py .
    cp ../requirements.txt .
    if [ -d "../tests" ]; then
        cp -r ../tests .
    fi
    print_success "Project files copied"
else
    print_warning "Project files not found in parent directory"
    print_warning "Please copy the files manually to $PROJECT_DIR"
fi

# Make scripts executable
chmod +x *.py

print_status "Step 8: Testing I2C setup..."
if command -v i2cdetect >/dev/null 2>&1; then
    print_status "Scanning I2C bus..."
    sudo i2cdetect -y 1
    print_success "I2C scan completed"
else
    print_error "i2cdetect not found"
fi

print_status "Step 9: Creating systemd service (optional)..."
cat > ins.service << EOF
[Unit]
Description=Inertial Navigation System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
ExecStart=$PROJECT_DIR/venv/bin/python main.py --calibration imu_calibration.json --log /var/log/ins/navigation.log
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

print_success "Systemd service file created (ins.service)"

print_status "Step 10: Creating log directory..."
sudo mkdir -p /var/log/ins
sudo chown $USER:$USER /var/log/ins
print_success "Log directory created"

echo
echo "================================================"
print_success "Setup completed successfully!"
echo
echo "📋 Next Steps:"
echo "1. Reboot your Raspberry Pi: sudo reboot"
echo "2. After reboot, activate environment: cd $PROJECT_DIR && source venv/bin/activate"
echo "3. Test hardware: python3 test_hardware.py"
echo "4. Run calibration: python3 calibration.py"
echo "5. Test navigation: python3 main.py --simulation"
echo "6. Run with real hardware: python3 main.py"
echo
echo "📡 Network Access:"
echo "- TCP stream: nc \$(hostname -I | awk '{print \$1}') 8888"
echo "- WebSocket: ws://\$(hostname -I | awk '{print \$1}'):8889"
echo
echo "🔧 Optional Service Setup:"
echo "- Install service: sudo cp ins.service /etc/systemd/system/"
echo "- Enable service: sudo systemctl enable ins.service"
echo "- Start service: sudo systemctl start ins.service"
echo
print_warning "IMPORTANT: You need to reboot for I2C changes to take effect!"
echo
read -p "Reboot now? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Rebooting..."
    sudo reboot
fi 