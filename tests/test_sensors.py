import pytest
from services.sensor_hub.mock_sensor import MockSensorHub

def test_mock_sensor_hub_initial_state():
    hub = MockSensorHub()
    assert not hub.is_connected
    assert hub.commands_sent == 0
    assert hub.last_command is None

def test_mock_sensor_hub_connect_disconnect():
    hub = MockSensorHub()
    
    # Connection simulation
    conn_res = hub.connect("http://192.168.1.100")
    assert not conn_res["success"]
    assert "Simulated mode active" in conn_res["message"]
    
    # Disconnection simulation
    disc_res = hub.disconnect()
    assert disc_res["success"]
    assert "Simulated sensor disconnected" in disc_res["message"]

def test_mock_sensor_hub_modes():
    hub = MockSensorHub()
    
    # Test setting mode to Assistive (Mode 1)
    assistive_cmd = {
        "cmd": "set_mode",
        "mode_id": 1,
        "mode_name": "Assistive",
        "target_torque": 5.0
    }
    res = hub.send_command(assistive_cmd)
    assert not res["success"]  # mock hub returns false as it doesn't send to real hardware
    assert hub.commands_sent == 1
    assert hub.last_command == assistive_cmd
    
    # Test setting mode to Resistive (Mode 2)
    resistive_cmd = {
        "cmd": "set_mode",
        "mode_id": 2,
        "mode_name": "Resistive",
        "target_torque": 10.0
    }
    hub.send_command(resistive_cmd)
    assert hub.commands_sent == 2
    assert hub.last_command == resistive_cmd
