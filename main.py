#!/usr/bin/env python3
"""
Main Inertial Navigation System Application.

This module orchestrates the complete INS system with:
- Multi-threaded sensor data acquisition
- Real-time EKF processing 
- Network state publishing (TCP/WebSocket)
- CSV data logging
- Command-line interface
- Configuration management

Author: Embedded Systems Engineer
"""

import time
import json
import argparse
import logging
import threading
import queue
import signal
import sys
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

# Network imports (with fallbacks)
try:
    import socket
    import websockets
    import asyncio
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    logging.warning("websockets not available - TCP-only mode")

from imu_drivers import SensorManager
from calibration import load_calibration, CalibrationData
from ekf import InertialEKF, MotionSimulator


class SensorThread(threading.Thread):
    """Thread for continuous sensor data acquisition."""
    
    def __init__(self, sensor_manager: SensorManager, data_queue: queue.Queue, 
                 sample_rate: float = 200.0):
        super().__init__(name="SensorThread", daemon=True)
        self.sensor_manager = sensor_manager
        self.data_queue = data_queue
        self.sample_rate = sample_rate
        self.sample_period = 1.0 / sample_rate
        self.running = False
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Statistics
        self.samples_collected = 0
        self.missed_deadlines = 0
        self.start_time = 0.0
        
    def run(self):
        """Main sensor acquisition loop."""
        self.logger.info(f"Starting sensor acquisition at {self.sample_rate} Hz")
        self.running = True
        self.start_time = time.time()
        next_sample_time = self.start_time
        
        while self.running:
            current_time = time.time()
            
            # Check for missed deadline
            if current_time > next_sample_time + self.sample_period:
                self.missed_deadlines += 1
                if self.missed_deadlines % 100 == 0:
                    self.logger.warning(f"Missed {self.missed_deadlines} sensor deadlines")
            
            try:
                # Read sensor data
                sensor_data = self.sensor_manager.read_all()
                
                # Add to queue (non-blocking)
                try:
                    self.data_queue.put_nowait(sensor_data)
                    self.samples_collected += 1
                except queue.Full:
                    # Drop oldest sample if queue is full
                    try:
                        self.data_queue.get_nowait()
                        self.data_queue.put_nowait(sensor_data)
                    except queue.Empty:
                        pass
                
                # Precise timing
                next_sample_time += self.sample_period
                sleep_time = next_sample_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_sample_time = time.time()  # Reset if we're behind
                    
            except Exception as e:
                self.logger.error(f"Sensor read error: {e}")
                time.sleep(0.001)  # Brief pause on error
    
    def stop(self):
        """Stop sensor acquisition."""
        self.running = False
        self.logger.info(f"Sensor thread stopped - collected {self.samples_collected} samples, "
                        f"missed {self.missed_deadlines} deadlines")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get sensor thread statistics."""
        runtime = time.time() - self.start_time if self.start_time > 0 else 0
        avg_rate = self.samples_collected / runtime if runtime > 0 else 0
        
        return {
            'samples_collected': self.samples_collected,
            'missed_deadlines': self.missed_deadlines,
            'average_rate_hz': avg_rate,
            'runtime_seconds': runtime
        }


class EKFThread(threading.Thread):
    """Thread for EKF processing and state estimation."""
    
    def __init__(self, data_queue: queue.Queue, state_queue: queue.Queue,
                 ekf: InertialEKF, mag_reference: np.ndarray):
        super().__init__(name="EKFThread", daemon=True)
        self.data_queue = data_queue
        self.state_queue = state_queue
        self.ekf = ekf
        self.mag_reference = mag_reference
        self.running = False
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Processing statistics
        self.measurements_processed = 0
        self.processing_times = []
        self.start_time = 0.0
        
    def run(self):
        """Main EKF processing loop."""
        self.logger.info("Starting EKF processing")
        self.running = True
        self.start_time = time.time()
        
        while self.running:
            try:
                # Get sensor data (blocking with timeout)
                sensor_data = self.data_queue.get(timeout=1.0)
                
                process_start = time.time()
                
                # EKF prediction step
                self.ekf.predict(
                    sensor_data['gyroscope'],
                    sensor_data['accelerometer_mpu'],
                    sensor_data['dt']
                )
                
                # EKF update steps (at reduced rate)
                if self.measurements_processed % 10 == 0:  # 20 Hz updates
                    # Gravity update using accelerometer
                    self.ekf.update_gravity(sensor_data['accelerometer_mpu'])
                    
                    # Magnetometer update
                    self.ekf.update_magnetometer(sensor_data['magnetometer'], self.mag_reference)
                
                # Zero velocity update
                zupt_applied = self.ekf.update_zupt(
                    sensor_data['accelerometer_mpu'],
                    sensor_data['gyroscope']
                )
                
                # Get current state
                state_dict = self.ekf.get_state_dict()
                state_dict['sensor_data'] = sensor_data
                state_dict['zupt_applied'] = zupt_applied
                
                # Send to state queue
                try:
                    self.state_queue.put_nowait(state_dict)
                except queue.Full:
                    # Replace oldest state if queue is full
                    try:
                        self.state_queue.get_nowait()
                        self.state_queue.put_nowait(state_dict)
                    except queue.Empty:
                        pass
                
                # Track processing time
                process_time = time.time() - process_start
                self.processing_times.append(process_time)
                
                # Keep only recent processing times (for statistics)
                if len(self.processing_times) > 1000:
                    self.processing_times = self.processing_times[-500:]
                
                self.measurements_processed += 1
                
                # Mark task as done
                self.data_queue.task_done()
                
            except queue.Empty:
                continue  # Timeout - check if still running
            except Exception as e:
                self.logger.error(f"EKF processing error: {e}")
                time.sleep(0.001)
    
    def stop(self):
        """Stop EKF processing."""
        self.running = False
        self.logger.info(f"EKF thread stopped - processed {self.measurements_processed} measurements")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get EKF processing statistics."""
        runtime = time.time() - self.start_time if self.start_time > 0 else 0
        avg_rate = self.measurements_processed / runtime if runtime > 0 else 0
        
        if self.processing_times:
            avg_process_time = np.mean(self.processing_times)
            max_process_time = np.max(self.processing_times)
            cpu_usage = avg_process_time * avg_rate * 100  # Approximate CPU usage %
        else:
            avg_process_time = 0
            max_process_time = 0
            cpu_usage = 0
        
        return {
            'measurements_processed': self.measurements_processed,
            'average_rate_hz': avg_rate,
            'runtime_seconds': runtime,
            'avg_processing_time_ms': avg_process_time * 1000,
            'max_processing_time_ms': max_process_time * 1000,
            'estimated_cpu_usage_percent': cpu_usage
        }


class DataLogger:
    """CSV data logger for sensor and state data."""
    
    def __init__(self, filename: str, log_raw_sensors: bool = True):
        self.filename = filename
        self.log_raw_sensors = log_raw_sensors
        self.file = None
        self.writer = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Statistics
        self.records_written = 0
        
    def start(self):
        """Start logging to CSV file."""
        try:
            self.file = open(self.filename, 'w', newline='')
            
            # Define CSV header
            header = [
                'timestamp', 'dt',
                'q_w', 'q_x', 'q_y', 'q_z',
                'roll_deg', 'pitch_deg', 'yaw_deg',
                'vel_n', 'vel_e', 'vel_d',
                'pos_n', 'pos_e', 'pos_d',
                'gyro_bias_x', 'gyro_bias_y', 'gyro_bias_z',
                'accel_bias_x', 'accel_bias_y', 'accel_bias_z',
                'covariance_trace', 'zupt_applied'
            ]
            
            if self.log_raw_sensors:
                header.extend([
                    'gyro_x', 'gyro_y', 'gyro_z',
                    'accel_mpu_x', 'accel_mpu_y', 'accel_mpu_z',
                    'accel_lsm_x', 'accel_lsm_y', 'accel_lsm_z',
                    'mag_x', 'mag_y', 'mag_z',
                    'temperature'
                ])
            
            self.writer = csv.writer(self.file)
            self.writer.writerow(header)
            self.file.flush()
            
            self.logger.info(f"Started logging to {self.filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to start logging: {e}")
            self.file = None
            self.writer = None
    
    def log_state(self, state_dict: Dict[str, Any]):
        """Log a state dictionary to CSV."""
        if self.writer is None:
            return
        
        try:
            # Extract data
            sensor_data = state_dict.get('sensor_data', {})
            
            row = [
                state_dict['timestamp'],
                sensor_data.get('dt', 0),
                *state_dict['quaternion'],
                *state_dict['euler_deg'],
                *state_dict['velocity_ned'],
                *state_dict['position_ned'],
                *state_dict['gyro_bias'],
                *state_dict['accel_bias'],
                state_dict['covariance_trace'],
                state_dict.get('zupt_applied', False)
            ]
            
            if self.log_raw_sensors:
                row.extend([
                    *sensor_data.get('gyroscope', [0, 0, 0]),
                    *sensor_data.get('accelerometer_mpu', [0, 0, 0]),
                    *sensor_data.get('accelerometer_lsm', [0, 0, 0]),
                    *sensor_data.get('magnetometer', [0, 0, 0]),
                    sensor_data.get('temperature', 0)
                ])
            
            self.writer.writerow(row)
            self.records_written += 1
            
            # Periodic flush
            if self.records_written % 100 == 0:
                self.file.flush()
                
        except Exception as e:
            self.logger.error(f"Logging error: {e}")
    
    def stop(self):
        """Stop logging and close file."""
        if self.file:
            self.file.flush()
            self.file.close()
            self.file = None
            self.writer = None
            self.logger.info(f"Stopped logging - wrote {self.records_written} records")


class StatePublisher:
    """Network state publisher (TCP and WebSocket)."""
    
    def __init__(self, tcp_port: int = 8888, websocket_port: int = 8889):
        self.tcp_port = tcp_port
        self.websocket_port = websocket_port
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # TCP server
        self.tcp_socket = None
        self.tcp_clients = []
        
        # WebSocket server
        self.websocket_server = None
        self.websocket_clients = set()
        
        # Statistics
        self.messages_sent = 0
        
    def start_tcp_server(self):
        """Start TCP server for state publishing."""
        try:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.tcp_socket.bind(('0.0.0.0', self.tcp_port))
            self.tcp_socket.listen(5)
            self.tcp_socket.settimeout(1.0)  # Non-blocking accept
            
            self.logger.info(f"TCP server listening on port {self.tcp_port}")
            
            # Start client handler thread
            threading.Thread(target=self._tcp_client_handler, daemon=True).start()
            
        except Exception as e:
            self.logger.error(f"Failed to start TCP server: {e}")
    
    def _tcp_client_handler(self):
        """Handle TCP client connections."""
        while self.tcp_socket:
            try:
                client_socket, address = self.tcp_socket.accept()
                self.tcp_clients.append(client_socket)
                self.logger.info(f"TCP client connected: {address}")
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.tcp_socket:  # Only log if we're still running
                    self.logger.error(f"TCP accept error: {e}")
                break
    
    def start_websocket_server(self):
        """Start WebSocket server for state publishing."""
        if not HAS_WEBSOCKETS:
            self.logger.warning("WebSocket server not available")
            return
        
        async def websocket_handler(websocket, path):
            self.websocket_clients.add(websocket)
            self.logger.info(f"WebSocket client connected: {websocket.remote_address}")
            try:
                await websocket.wait_closed()
            finally:
                self.websocket_clients.discard(websocket)
                self.logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
        
        def start_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            self.websocket_server = websockets.serve(
                websocket_handler, '0.0.0.0', self.websocket_port
            )
            
            loop.run_until_complete(self.websocket_server)
            self.logger.info(f"WebSocket server listening on port {self.websocket_port}")
            loop.run_forever()
        
        # Start WebSocket server in separate thread
        threading.Thread(target=start_server, daemon=True).start()
    
    def publish_state(self, state_dict: Dict[str, Any]):
        """Publish state to all connected clients."""
        if not self.tcp_clients and not self.websocket_clients:
            return
        
        try:
            # Prepare JSON message
            message = json.dumps(state_dict, default=str) + '\n'
            message_bytes = message.encode('utf-8')
            
            # Send to TCP clients
            failed_clients = []
            for client in self.tcp_clients:
                try:
                    client.send(message_bytes)
                except Exception:
                    failed_clients.append(client)
            
            # Remove failed clients
            for client in failed_clients:
                try:
                    client.close()
                except Exception:
                    pass
                self.tcp_clients.remove(client)
            
            # Send to WebSocket clients
            if HAS_WEBSOCKETS and self.websocket_clients:
                # Create a coroutine to send to all WebSocket clients
                async def send_to_websockets():
                    failed_websockets = set()
                    for websocket in self.websocket_clients.copy():
                        try:
                            await websocket.send(message)
                        except Exception:
                            failed_websockets.add(websocket)
                    
                    # Remove failed WebSocket clients
                    self.websocket_clients -= failed_websockets
                
                # Schedule the coroutine
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(send_to_websockets())
                except RuntimeError:
                    pass  # No event loop running
            
            self.messages_sent += 1
            
        except Exception as e:
            self.logger.error(f"State publishing error: {e}")
    
    def stop(self):
        """Stop state publisher."""
        # Close TCP server
        if self.tcp_socket:
            self.tcp_socket.close()
            self.tcp_socket = None
        
        # Close TCP clients
        for client in self.tcp_clients:
            try:
                client.close()
            except Exception:
                pass
        self.tcp_clients.clear()
        
        # WebSocket server will close when the thread ends
        self.websocket_clients.clear()
        
        self.logger.info(f"State publisher stopped - sent {self.messages_sent} messages")


class INSApplication:
    """Main Inertial Navigation System application."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Components
        self.sensor_manager = None
        self.ekf = None
        self.simulator = None
        
        # Threads
        self.sensor_thread = None
        self.ekf_thread = None
        
        # Queues
        self.sensor_queue = queue.Queue(maxsize=100)
        self.state_queue = queue.Queue(maxsize=50)
        
        # Utilities
        self.data_logger = None
        self.state_publisher = None
        
        # Control
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.start_time = 0.0
        
    def initialize(self) -> bool:
        """Initialize all system components."""
        self.logger.info("Initializing INS application...")
        
        try:
            # Initialize sensor manager or simulator
            if self.config['simulation_mode']:
                self.logger.info("Running in simulation mode")
                self.simulator = MotionSimulator(dt=1.0/self.config['sample_rate'])
            else:
                self.sensor_manager = SensorManager(bus=self.config['i2c_bus'])
                
                # Load calibration if available
                if self.config['calibration_file']:
                    calibration = load_calibration(self.config['calibration_file'])
                    if calibration:
                        self.sensor_manager.set_calibration(calibration.to_dict())
                        self.logger.info("Calibration loaded successfully")
                    else:
                        self.logger.warning("Failed to load calibration - using defaults")
            
            # Initialize EKF
            self.ekf = InertialEKF()
            
            # Set EKF noise parameters from config
            self.ekf.set_noise_parameters(
                gyro_noise=self.config.get('gyro_noise_density', 1e-4),
                accel_noise=self.config.get('accel_noise_density', 1e-3),
                gyro_bias_stability=self.config.get('gyro_bias_stability', 1e-6),
                accel_bias_stability=self.config.get('accel_bias_stability', 1e-5)
            )
            
            # Initialize threads
            if self.config['simulation_mode']:
                # Create a sensor thread that uses the simulator
                self.sensor_thread = SimulatorThread(
                    self.simulator, self.sensor_queue, self.config['sample_rate']
                )
            else:
                self.sensor_thread = SensorThread(
                    self.sensor_manager, self.sensor_queue, self.config['sample_rate']
                )
            
            self.ekf_thread = EKFThread(
                self.sensor_queue, self.state_queue, self.ekf,
                np.array(self.config['magnetic_reference'])  # NED frame
            )
            
            # Initialize data logger
            if self.config['log_file']:
                self.data_logger = DataLogger(
                    self.config['log_file'],
                    log_raw_sensors=self.config['log_raw_sensors']
                )
            
            # Initialize state publisher
            if self.config['enable_networking']:
                self.state_publisher = StatePublisher(
                    tcp_port=self.config['tcp_port'],
                    websocket_port=self.config['websocket_port']
                )
            
            self.logger.info("INS application initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def start(self):
        """Start the INS application."""
        if not self.initialize():
            return False
        
        self.logger.info("Starting INS application...")
        self.start_time = time.time()
        self.running = True
        
        try:
            # Start data logger
            if self.data_logger:
                self.data_logger.start()
            
            # Start state publisher
            if self.state_publisher:
                self.state_publisher.start_tcp_server()
                self.state_publisher.start_websocket_server()
            
            # Start processing threads
            self.sensor_thread.start()
            self.ekf_thread.start()
            
            self.logger.info("INS application started successfully")
            
            # Main loop - process states
            self._main_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Runtime error: {e}")
        finally:
            self.stop()
        
        return True
    
    def _main_loop(self):
        """Main application loop - processes states and manages output."""
        state_count = 0
        last_stats_time = time.time()
        
        while self.running:
            try:
                # Get processed state
                state_dict = self.state_queue.get(timeout=1.0)
                
                # Log to CSV
                if self.data_logger:
                    self.data_logger.log_state(state_dict)
                
                # Publish over network
                if self.state_publisher:
                    self.state_publisher.publish_state(state_dict)
                
                state_count += 1
                
                # Print periodic statistics
                current_time = time.time()
                if current_time - last_stats_time > 10.0:  # Every 10 seconds
                    self._print_statistics()
                    last_stats_time = current_time
                
                # Mark task as done
                self.state_queue.task_done()
                
            except queue.Empty:
                continue  # Timeout - check if still running
            except Exception as e:
                self.logger.error(f"Main loop error: {e}")
    
    def _print_statistics(self):
        """Print system performance statistics."""
        runtime = time.time() - self.start_time
        
        print(f"\n--- INS Statistics (Runtime: {runtime:.1f}s) ---")
        
        # Sensor statistics
        if self.sensor_thread:
            sensor_stats = self.sensor_thread.get_statistics()
            print(f"Sensor: {sensor_stats['average_rate_hz']:.1f} Hz, "
                  f"missed {sensor_stats['missed_deadlines']} deadlines")
        
        # EKF statistics
        if self.ekf_thread:
            ekf_stats = self.ekf_thread.get_statistics()
            print(f"EKF: {ekf_stats['average_rate_hz']:.1f} Hz, "
                  f"~{ekf_stats['estimated_cpu_usage_percent']:.1f}% CPU, "
                  f"{ekf_stats['avg_processing_time_ms']:.2f}ms avg")
        
        # EKF state statistics
        if self.ekf:
            state_dict = self.ekf.get_state_dict()
            stats = state_dict['statistics']
            print(f"EKF counts: {stats['predictions']} pred, "
                  f"{stats['updates']} upd, {stats['zupts']} ZUPT")
        
        # Queue sizes
        print(f"Queues: sensor={self.sensor_queue.qsize()}, "
              f"state={self.state_queue.qsize()}")
        
        # Network statistics
        if self.state_publisher:
            print(f"Network: {len(self.state_publisher.tcp_clients)} TCP, "
                  f"{len(self.state_publisher.websocket_clients)} WS clients")
        
        print("-" * 50)
    
    def stop(self):
        """Stop the INS application."""
        self.logger.info("Stopping INS application...")
        self.running = False
        
        # Stop threads
        if self.sensor_thread:
            self.sensor_thread.stop()
        if self.ekf_thread:
            self.ekf_thread.stop()
        
        # Stop utilities
        if self.data_logger:
            self.data_logger.stop()
        if self.state_publisher:
            self.state_publisher.stop()
        
        # Final statistics
        self._print_statistics()
        
        runtime = time.time() - self.start_time
        self.logger.info(f"INS application stopped after {runtime:.1f} seconds")


class SimulatorThread(SensorThread):
    """Simulator-based sensor thread for testing."""
    
    def __init__(self, simulator: MotionSimulator, data_queue: queue.Queue, sample_rate: float):
        super().__init__(None, data_queue, sample_rate)
        self.simulator = simulator
        self.name = "SimulatorThread"
    
    def run(self):
        """Main simulator loop."""
        self.logger.info(f"Starting motion simulation at {self.sample_rate} Hz")
        self.running = True
        self.start_time = time.time()
        
        while self.running:
            try:
                # Generate simulated sensor data
                sensor_data = self.simulator.step()
                
                # Add to queue
                try:
                    self.data_queue.put_nowait(sensor_data)
                    self.samples_collected += 1
                except queue.Full:
                    try:
                        self.data_queue.get_nowait()
                        self.data_queue.put_nowait(sensor_data)
                    except queue.Empty:
                        pass
                
                # Timing (simulator handles its own timing)
                
            except Exception as e:
                self.logger.error(f"Simulation error: {e}")
                time.sleep(0.001)


def create_default_config() -> Dict[str, Any]:
    """Create default configuration dictionary."""
    return {
        # Hardware settings
        'i2c_bus': 1,
        'sample_rate': 200.0,
        
        # EKF noise parameters
        'gyro_noise_density': 1e-4,
        'accel_noise_density': 1e-3,
        'gyro_bias_stability': 1e-6,
        'accel_bias_stability': 1e-5,
        
        # Magnetic reference field (NED frame, Tesla)
        'magnetic_reference': [20e-6, 0, 45e-6],
        
        # Calibration
        'calibration_file': 'imu_calibration.json',
        
        # Logging
        'log_file': None,
        'log_raw_sensors': True,
        
        # Networking
        'enable_networking': True,
        'tcp_port': 8888,
        'websocket_port': 8889,
        
        # Testing
        'simulation_mode': False
    }


def setup_logging(level: str = 'INFO') -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logging.getLogger('main').info(f"Received signal {signum}")
    sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Inertial Navigation System')
    
    parser.add_argument('--config', '-c', help='Configuration file (JSON)')
    parser.add_argument('--calibration', help='Calibration file')
    parser.add_argument('--log', '-l', help='Output log file (CSV)')
    parser.add_argument('--sample-rate', type=float, default=200.0,
                       help='Sensor sample rate (Hz)')
    parser.add_argument('--simulation', action='store_true',
                       help='Run in simulation mode')
    parser.add_argument('--no-network', action='store_true',
                       help='Disable network publishing')
    parser.add_argument('--tcp-port', type=int, default=8888,
                       help='TCP port for state publishing')
    parser.add_argument('--websocket-port', type=int, default=8889,
                       help='WebSocket port for state publishing')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger('main')
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load configuration
    config = create_default_config()
    
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config.update(json.load(f))
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return 1
    
    # Override with command line arguments
    if args.calibration:
        config['calibration_file'] = args.calibration
    if args.log:
        config['log_file'] = args.log
    if args.simulation:
        config['simulation_mode'] = True
    if args.no_network:
        config['enable_networking'] = False
    
    config['sample_rate'] = args.sample_rate
    config['tcp_port'] = args.tcp_port
    config['websocket_port'] = args.websocket_port
    
    # Create and run application
    app = INSApplication(config)
    
    try:
        success = app.start()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main()) 