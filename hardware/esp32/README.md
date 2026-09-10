Install:

* **VS Code**
* **PlatformIO IDE** extension (from the VS Code Extensions Marketplace)

After installing PlatformIO, restart VS Code.

---

# Step 2: Connect Your ESP32

1. Connect the ESP32 to your laptop using a USB cable.
2. Open a terminal and verify that your computer detects the board.

### Windows

Open **Command Prompt**:

```cmd
mode
```

You should see something like:

```
COM3
COM4
COM5
```

The ESP32 is usually one of these ports.

---

# Step 3: Create a New Project

1. Click the **PlatformIO** icon (alien head) on the left.
2. Click **New Project**.
3. Fill in:

```
Project Name: ESP32WiFi
Board: Espressif ESP32 Dev Module
Framework: Arduino
```

Click **Finish**.

PlatformIO will create a project like:

```
ESP32WiFi/
│
├── include/
├── lib/
├── src/
│   └── main.cpp
├── platformio.ini
```

---

# Step 4: Copy the ESP Code 

Replacing src/main.cpp and platformio.ini with respective files in esp and esp now folder.

---

# Step 6: Build the Project

Click the ✔ (Build) button in the PlatformIO toolbar at the bottom.

You should see:

```
SUCCESS
```

---

# Step 7: Upload the Program

Click the → (Upload) button.

If the upload fails with an error like:

```
Failed to connect to ESP32
```

press and hold the **BOOT** button on the ESP32 while the upload starts, then release it once it begins.

---

# Step 8: Open the Serial Monitor

Click the plug icon (Monitor) or run:

```
PlatformIO: Serial Monitor
```

You should see output like:

```
Connecting to WiFi...
.....
Connected!
IP Address:
192.168.1.105
```

Your ESP32 is now connected to Wi-Fi.

---

# Step 9: Verify the Connection

Open a terminal on your computer and ping the ESP32:

```cmd
ping 192.168.1.105
```

If you receive replies, the connection is working.

---

