#!/usr/bin/env python3
"""
eurotherm_furnace_gui.py
NUPyLab - Standalone Eurotherm Furnace Control GUI

Connects to a Eurotherm 2200 / 2400 / 3216 over RS-485 (USB-serial adapter),
reads live temperature and heater power output, ramps the setpoint at a
controlled rate, and optionally logs everything to a CSV file.

Ramping mirrors the approach in NUPyLab's instruments/heater/ layer:
  * 2200 series: setpoint_rate_limit + setpoint2 + program_status = "run"
  * 2400 series: programs/segments API
  * 3216 series: segments API
This hands the ramp off to the controller's built-in programmer instead of
writing the setpoint every second from Python, which was the cause of the
90-second crash seen during testing.

Dependencies: PyQt6 or PyQt5, pyqtgraph, numpy
              + nupylab drivers (eurotherm2200 / 2400 / 3216) for real hardware
"""

import sys
import csv
import time
import threading
from datetime import datetime
from collections import deque

# Try to import the NUPyLab Eurotherm drivers.
# First: installed as part of the nupylab package (normal lab usage).
# Second: driver files sitting alongside this file (standalone dev).
# If neither works, fall back to SimulatedEurotherm defined below.
try:
    try:
        from nupylab.drivers.eurotherm2200 import Eurotherm2200
        from nupylab.drivers.eurotherm2400 import Eurotherm2400
        from nupylab.drivers.eurotherm3216 import Eurotherm3216
    except ImportError:
        from eurotherm2200 import Eurotherm2200  # type: ignore
        from eurotherm2400 import Eurotherm2400  # type: ignore
        from eurotherm3216 import Eurotherm3216  # type: ignore
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    print("[INFO] Eurotherm drivers not found - running in simulation mode.")

# Pull available serial ports for the dropdown.
# Changed from a plain text box to a combo populated by list_resources() so the port
# shows up automatically when the adapter is plugged in - same as the S8 GUI does it.
# list_resources() returns VISA-style names like ASRL5::INSTR; conversion happens in _connect.
# Falls back to serial.tools.list_ports if the nupylab utility is not present.
try:
    from nupylab.utilities import list_resources as _list_resources
    _SERIAL_PORTS = _list_resources()
except Exception:
    try:
        import serial.tools.list_ports as _lp
        _SERIAL_PORTS = [p.device for p in _lp.comports()]
    except Exception:
        _SERIAL_PORTS = []
if not _SERIAL_PORTS:
    _SERIAL_PORTS = ["/dev/cu.usbserial"]

import numpy as np

# NUPyLab runs on PyQt5 on the lab computer but the original code was written with PyQt6.
# This try/except loads whichever version is installed so the same file works on both.
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget,
        QHBoxLayout, QVBoxLayout, QGridLayout,
        QLabel, QLineEdit, QPushButton, QComboBox,
        QGroupBox, QFileDialog, QMessageBox, QFrame,
        QStatusBar,
    )
    from PyQt6.QtCore import pyqtSignal, QObject
    _PYQT6 = True
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget,
        QHBoxLayout, QVBoxLayout, QGridLayout,
        QLabel, QLineEdit, QPushButton, QComboBox,
        QGroupBox, QFileDialog, QMessageBox, QFrame,
        QStatusBar,
    )
    from PyQt5.QtCore import pyqtSignal, QObject
    _PYQT6 = False

# PyQt6 moved enums into nested classes; PyQt5 had them flat.
# Rather than scattering if/else checks throughout the file, set aliases here once.
# _HLINE, _MSGBOX_YES, _MSGBOX_CANCEL then work the same regardless of which Qt is loaded.
if _PYQT6:
    _HLINE         = QFrame.Shape.HLine
    _MSGBOX_YES    = QMessageBox.StandardButton.Yes
    _MSGBOX_CANCEL = QMessageBox.StandardButton.Cancel
else:
    _HLINE         = QFrame.HLine
    _MSGBOX_YES    = QMessageBox.Yes
    _MSGBOX_CANCEL = QMessageBox.Cancel

import pyqtgraph as pg


# ---------------------------------------------------------------------------
# Model list
# ---------------------------------------------------------------------------

EUROTHERM_MODEL_NAMES = ["Eurotherm 2200", "Eurotherm 2400", "Eurotherm 3216"]

if HARDWARE_AVAILABLE:
    EUROTHERM_DRIVER_MAP = {
        "Eurotherm 2200": Eurotherm2200,
        "Eurotherm 2400": Eurotherm2400,
        "Eurotherm 3216": Eurotherm3216,
    }

# Lab safety cap - agreed with Danielle
MAX_RAMP_RATE_C_PER_MIN = 10.0

# program_status values across all three models that mean the ramp has ended
RAMP_DONE_STATES = frozenset({"off", "end", "complete"})

# First reading outside this range almost always means an Eurotherm decimal
# place configuration mismatch rather than a real measurement.
TEMP_SANITY_MIN = -50.0
TEMP_SANITY_MAX = 1800.0

# Color scheme - each trace shares its color with its Y-axis
TEMP_COLOR  = "#ff5555"   # red  - temperature
POWER_COLOR = "#55aaff"   # blue - power output

LIVE_BG    = "#0a0a0a"
LIVE_FG    = "#00e676"
EDIT_STYLE = "background-color: #dff0f7; color: #111111;"


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------

class SimulatedEurotherm:
    """Stand-in for testing without hardware connected.

    Matches the same property interface as the real NUPyLab drivers so everything
    else in the file (DataWorker, RampManager, the GUI) works identically whether
    hardware is connected or not. Also simulates the 2200-series programmer behavior
    so working_setpoint steps realistically during a simulated ramp.
    """

    def __init__(self):
        self._sim_temp   = 22.0
        self._working_sp = 22.0   # intermediate SP, steps toward _target
        self._target     = 22.0   # setpoint2 - where the ramp is headed
        self._direct_sp  = 22.0   # target_setpoint (register 2, manual mode)
        self._rate_limit = 0.0    # C/min; 0 = no rate limit
        self._status     = "off"  # program_status
        self._sp1        = 22.0
        self._sp2        = 22.0
        self._last_t     = time.time()

    def _advance_working_sp(self):
        """Move _working_sp toward _target at _rate_limit C/min."""
        now = time.time()
        dt  = now - self._last_t
        self._last_t = now
        if self._status in ("run", "ramp") and self._rate_limit > 0:
            step = self._rate_limit / 60.0 * dt
            if self._working_sp < self._target:
                self._working_sp = min(self._working_sp + step, self._target)
            else:
                self._working_sp = max(self._working_sp - step, self._target)
            if abs(self._working_sp - self._target) < 0.05:
                self._working_sp = self._target
                self._status = "end"

    @property
    def process_value(self) -> float:
        self._advance_working_sp()
        sp = self._working_sp if self._status in ("run", "ramp") else self._direct_sp
        self._sim_temp += 0.05 * (sp - self._sim_temp)
        self._sim_temp += float(np.random.normal(0, 0.1))
        return round(self._sim_temp, 2)

    @property
    def output_level(self) -> float:
        sp = self._working_sp if self._status in ("run", "ramp") else self._direct_sp
        error = sp - self._sim_temp
        return round(max(0.0, min(100.0, error * 2.0 + float(np.random.normal(0, 0.3)))), 1)

    @property
    def working_setpoint(self) -> float:
        return round(self._working_sp, 2)

    @property
    def program_status(self) -> str:
        return self._status

    @program_status.setter
    def program_status(self, val: str):
        if val == "run":
            self._status     = "ramp"
            self._working_sp = self._sp1
            self._target     = self._sp2
            self._last_t     = time.time()
        elif val == "reset":
            self._status = "off"

    @property
    def target_setpoint(self) -> float:
        return self._direct_sp

    @target_setpoint.setter
    def target_setpoint(self, val: float):
        self._direct_sp  = val
        self._working_sp = val

    @property
    def setpoint_rate_limit(self) -> float:
        return self._rate_limit

    @setpoint_rate_limit.setter
    def setpoint_rate_limit(self, val: float):
        self._rate_limit = val

    @property
    def setpoint1(self) -> float:
        return self._sp1

    @setpoint1.setter
    def setpoint1(self, val: float):
        self._sp1 = val

    @property
    def setpoint2(self) -> float:
        return self._sp2

    @setpoint2.setter
    def setpoint2(self, val: float):
        self._sp2 = val

    # Stubs for properties called in _start_2200 but not relevant to simulation
    @property
    def active_setpoint(self) -> int:
        return 1

    @active_setpoint.setter
    def active_setpoint(self, val: int):
        pass

    @property
    def end_type(self) -> str:
        return "dwell"

    @end_type.setter
    def end_type(self, val: str):
        pass

    @property
    def dwell_time(self) -> float:
        return 1.0

    @dwell_time.setter
    def dwell_time(self, val: float):
        pass


# ---------------------------------------------------------------------------
# Background polling thread
# ---------------------------------------------------------------------------

class DataWorker(QObject):
    """Polls the furnace on a background thread every interval_s seconds.

    Reading from a serial port blocks for up to 1 second, so running it on
    a background thread keeps the window responsive. Results come back via
    Qt signals because widget updates are not allowed from background threads.

    Emits (temperature_C, power_pct, working_setpoint_C, program_status).
    program_status is used by the GUI to detect when a hardware ramp ends.
    """

    # signal now carries 4 values: temp, power, working_setpoint, and program_status.
    # program_status was added so the GUI can tell when a hardware ramp finishes
    # on its own without needing a separate polling check.
    data_ready     = pyqtSignal(float, float, float, str)
    error_occurred = pyqtSignal(str)

    def __init__(self, driver, interval_s: float = 1.0):
        super().__init__()
        self.driver     = driver
        self.interval_s = interval_s
        self._active    = False

    def start(self):
        self._active = True
        # daemon=True: thread dies automatically when the main window closes
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self._active = False

    def _loop(self):
        while self._active:
            try:
                temp       = self.driver.process_value
                power      = self.driver.output_level
                working_sp = self.driver.working_setpoint
                # program_status tells us when a hardware ramp has ended.
                # Most drivers have this; catch any exception and default to "".
                try:
                    status = self.driver.program_status
                except Exception:
                    status = ""
                self.data_ready.emit(temp, power, working_sp, status)
            except Exception as exc:
                self.error_occurred.emit(str(exc))
            time.sleep(self.interval_s)


# ---------------------------------------------------------------------------
# Ramp manager
# ---------------------------------------------------------------------------

class RampManager(QObject):
    """Configures and starts the Eurotherm's built-in ramp programmer.

    The original approach wrote target_setpoint every second from a Python loop.
    That caused a NoResponseError crash after ~90 seconds when the controller
    briefly stopped answering writes. This version matches what instruments/heater/
    already does: write the ramp parameters once, then send program_status = "run"
    and let the controller handle it internally. No more continuous writes.

    Model-specific setup:
      * 2200 series: setpoint_rate_limit + setpoint2 + program_status = "run"
      * 2400 series: programs/segments API
      * 3216 series: segments API

    ramp_error is emitted if any setup write fails so the GUI can show it.
    """

    ramp_error = pyqtSignal(str)

    def __init__(self, driver, model_name: str):
        super().__init__()
        self.driver     = driver
        self.model_name = model_name

    def start(self, target: float, rate_c_per_min: float, current_temp: float):
        """Configure the built-in ramp program and start it.

        Runs in a daemon thread so the Modbus setup writes do not block the GUI.
        """
        t = threading.Thread(
            target=self._setup,
            args=(target, rate_c_per_min, current_temp),
            daemon=True,
        )
        t.start()

    def stop(self, current_temp: float):
        """Reset the program and hold at the current temperature.

        Also runs in a thread so the GUI does not freeze during the writes.
        """
        t = threading.Thread(
            target=self._teardown,
            args=(current_temp,),
            daemon=True,
        )
        t.start()

    def _setup(self, target: float, rate: float, current_temp: float):
        try:
            if "2400" in self.model_name:
                self._start_2400(target, rate)
            elif "3216" in self.model_name:
                self._start_3216(target, rate, current_temp)
            else:
                # 2200 series (covers 2216, 2204, etc.)
                self._start_2200(target, rate, current_temp)
        except Exception as exc:
            self.ramp_error.emit(str(exc))

    def _teardown(self, current_temp: float):
        try:
            self.driver.program_status = "reset"
        except Exception:
            pass
        # Write current temperature as the new setpoint so the controller
        # holds where it is rather than jumping to SP1 after the reset.
        try:
            self.driver.target_setpoint = current_temp
        except Exception:
            pass

    def _start_2200(self, target: float, rate: float, current_temp: float):
        # matches the sequence in instruments/heater/eurotherm2200.py exactly
        self.driver.program_status    = "reset"
        self.driver.active_setpoint   = 1
        self.driver.end_type          = "dwell"
        # SP1 is set to current temp so the ramp starts from where the furnace
        # actually is, not from whatever SP1 was left at previously
        self.driver.setpoint1         = current_temp
        self.driver.setpoint_rate_limit = rate
        self.driver.setpoint2         = target
        self.driver.dwell_time        = 1   # 1 second minimum dwell
        self.driver.program_status    = "run"

    def _start_2400(self, target: float, rate: float):
        # matches instruments/heater/eurotherm2400.py
        self.driver.program_status = "reset"
        self.driver.current_program = 1
        self.driver.programs[1].refresh()
        self.driver.programs[1].segments[1]["segment type"] = "ramp rate"
        self.driver.programs[1].segments[1]["rate"]             = rate
        self.driver.programs[1].segments[1]["target setpoint"]  = target
        self.driver.programs[1].segments[2]["segment type"] = "dwell"
        self.driver.programs[1].segments[2]["duration"]         = 1
        self.driver.programs[1].segments[3]["segment type"] = "end"
        self.driver.programs[1].segments[3]["end type"]         = "dwell"
        self.driver.program_status = "run"

    def _start_3216(self, target: float, rate: float, current_temp: float):
        # matches instruments/heater/eurotherm3216.py
        self.driver.program_status = "reset"
        self.driver.end_type = "dwell"
        for segment in self.driver.segments:
            segment.clear()
        # 3216 runs all 8 segments sequentially; put the actual ramp in the last one
        # so the others (all cleared to 0) pass through quickly
        self.driver.segments[-1].target_setpoint = target
        self.driver.segments[-1].ramp_rate       = rate
        self.driver.segments[-1].dwell           = 1
        self.driver.program_status = "run"


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class FurnaceGUI(QMainWindow):

    def __init__(self):
        super().__init__()

        self.driver      = None
        self.worker      = None
        self.ramp_mgr    = None
        self._log_file   = None
        self._log_writer = None
        self._t0         = None    # set at connect - used for graph x-axis
        self._log_t0     = None    # set at Start Logging - CSV elapsed resets here
        self._connected  = False
        self._logging    = False
        self._ramp_running = False   # True between Begin Ramp and completion/stop
        self._model_name   = ""      # tracks which model is connected

        # Rolling graph buffer - 600 points at 1 s/reading = 10 min of history.
        # deque auto-drops the oldest entry when maxlen is reached.
        N = 600
        self._times  = deque(maxlen=N)
        self._temps  = deque(maxlen=N)
        self._powers = deque(maxlen=N)

        self._last_temp = 25.0   # updated every poll; used as ramp starting point

        self._build_ui()
        self.setWindowTitle("NUPyLab - Eurotherm Furnace Control")
        self.setMinimumSize(1200, 700)

    # -----------------------------------------------------------------------
    # UI layout
    # -----------------------------------------------------------------------

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        h = QHBoxLayout(root)
        h.setContentsMargins(10, 10, 10, 10)
        h.setSpacing(12)
        h.addWidget(self._sidebar(), 0)
        h.addWidget(self._main_area(), 1)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready - not connected.")

    def _sidebar(self) -> QWidget:
        box = QGroupBox("Connection && Logging")
        box.setFixedWidth(215)
        v = QVBoxLayout(box)
        v.setSpacing(6)

        v.addWidget(QLabel("Eurotherm Model"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(EUROTHERM_MODEL_NAMES)
        v.addWidget(self.model_combo)

        v.addWidget(self._hline())

        v.addWidget(QLabel("Serial Port"))
        self.port_edit = QComboBox()
        self.port_edit.addItems(_SERIAL_PORTS)
        self.port_edit.setEditable(True)
        self.port_edit.setStyleSheet(EDIT_STYLE)
        v.addWidget(self.port_edit)

        v.addWidget(QLabel("Modbus Address"))
        self.addr_edit = QLineEdit("1")
        self.addr_edit.setStyleSheet(EDIT_STYLE)
        v.addWidget(self.addr_edit)

        v.addWidget(self._hline())

        v.addWidget(QLabel("Poll Interval (s)"))
        self.log_interval_edit = QLineEdit("1")
        self.log_interval_edit.setStyleSheet(EDIT_STYLE)
        v.addWidget(self.log_interval_edit)

        v.addWidget(self._hline())

        v.addWidget(QLabel("Log File"))
        self.filepath_edit = QLineEdit()
        self.filepath_edit.setPlaceholderText("(not set)")
        self.filepath_edit.setReadOnly(True)
        v.addWidget(self.filepath_edit)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_file)
        v.addWidget(self.browse_btn)

        # Logging is separate from connecting so the user can monitor
        # temperature before deciding to record anything to disk.
        self.log_btn = QPushButton("Start Logging")
        self.log_btn.setCheckable(True)
        self.log_btn.setEnabled(False)
        self.log_btn.clicked.connect(self._toggle_logging)
        v.addWidget(self.log_btn)

        v.addStretch()
        return box

    def _main_area(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(10)
        v.addWidget(self._controls_row())
        v.addWidget(self._graph_panel(), 1)
        return w

    def _controls_row(self) -> QWidget:
        """
        Layout left to right:
          Current T   (dark, live)
          Current Power (dark, live)
          Current SP  (dark, live) - what the controller is targeting right now
          Set Temp    (light blue, user input)
          Ramp Rate   (light blue, user input)
          Connect / Begin Ramp buttons
        """
        box = QGroupBox("Furnace Parameters")
        g = QGridLayout(box)
        g.setSpacing(12)

        live_style = (
            f"background-color: {LIVE_BG}; color: {LIVE_FG};"
            "font-size: 16px; font-weight: bold; border: 1px solid #003300;"
        )

        g.addWidget(QLabel("Current T (C)"), 0, 0)
        self.temp_display = QLineEdit("---")
        self.temp_display.setReadOnly(True)
        self.temp_display.setStyleSheet(live_style)
        self.temp_display.setMinimumWidth(100)
        g.addWidget(self.temp_display, 1, 0)

        g.addWidget(QLabel("Current Power (%)"), 0, 1)
        self.power_display = QLineEdit("---")
        self.power_display.setReadOnly(True)
        self.power_display.setStyleSheet(live_style)
        self.power_display.setMinimumWidth(100)
        g.addWidget(self.power_display, 1, 1)

        # working_setpoint (register 5) shows the intermediate SP the controller
        # is targeting right now. During a ramp this steps toward Set Temp while
        # Current T shows the actual measured temperature lagging behind it.
        # Added per Danielle's feedback - makes it easier to see how far the ramp
        # has progressed vs just watching the thermometer.
        g.addWidget(QLabel("Current SP (C)"), 0, 2)
        self.sp_display = QLineEdit("---")
        self.sp_display.setReadOnly(True)
        self.sp_display.setStyleSheet(live_style)
        self.sp_display.setMinimumWidth(100)
        g.addWidget(self.sp_display, 1, 2)

        g.addWidget(QLabel("Set Temp (C)"), 0, 3)
        self.set_temp_edit = QLineEdit("25.0")
        self.set_temp_edit.setStyleSheet(EDIT_STYLE)
        g.addWidget(self.set_temp_edit, 1, 3)

        g.addWidget(
            QLabel(f"Ramp Rate (C/min)  [max {MAX_RAMP_RATE_C_PER_MIN:.0f}]"), 0, 4
        )
        self.ramp_rate_edit = QLineEdit("5.0")
        self.ramp_rate_edit.setStyleSheet(EDIT_STYLE)
        g.addWidget(self.ramp_rate_edit, 1, 4)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setCheckable(True)
        self.connect_btn.clicked.connect(self._toggle_connect)
        g.addWidget(self.connect_btn, 0, 5)

        self.ramp_btn = QPushButton("Begin Ramp")
        self.ramp_btn.setCheckable(True)
        self.ramp_btn.setEnabled(False)
        self.ramp_btn.clicked.connect(self._toggle_ramp)
        g.addWidget(self.ramp_btn, 1, 5)

        return box


    def _graph_panel(self) -> QWidget:
        """
        Live plot with two independent Y-axes:
          Left  (red)  - Temperature in C, auto-scales
          Right (blue) - Power output, fixed 0-100 %
          Bottom       - Elapsed time in minutes

        pyqtgraph does not support two Y-axes natively. The fix is to create
        a second ViewBox and layer it on top of the main plot, then link the
        right axis to it. _sync_power_axis keeps both layers aligned on resize.

        X-axis bounds can be adjusted via the min/max fields below the graph.
        Leave both at 0 to let pyqtgraph auto-scale the time axis.
        """
        box = QGroupBox("Live Data - Temperature && Power Output")
        v = QVBoxLayout(box)

        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget()

        self.plot.setLabel("left", "Temperature (C)", color=TEMP_COLOR)
        self.plot.setLabel("bottom", "Elapsed Time (min)")
        self.plot.getAxis("left").setPen(pg.mkPen(color=TEMP_COLOR, width=1))
        self.plot.getAxis("left").setTextPen(pg.mkPen(color=TEMP_COLOR))
        self.plot.showGrid(x=True, y=True, alpha=0.25)

        self.plot.showAxis("right")
        self.plot.setLabel("right", "Power (%)", color=POWER_COLOR)
        self.plot.getAxis("right").setPen(pg.mkPen(color=POWER_COLOR, width=1))
        self.plot.getAxis("right").setTextPen(pg.mkPen(color=POWER_COLOR))

        self._power_vb = pg.ViewBox()
        self.plot.scene().addItem(self._power_vb)
        self.plot.getAxis("right").linkToView(self._power_vb)
        self._power_vb.setXLink(self.plot)
        self._power_vb.setYRange(0, 100, padding=0.05)
        self._power_vb.enableAutoRange(axis="y", enable=False)

        self.plot.getViewBox().sigResized.connect(self._sync_power_axis)
        self._sync_power_axis()

        self.curve_temp = self.plot.plot(pen=pg.mkPen(color=TEMP_COLOR, width=2))
        self.curve_power = pg.PlotCurveItem(pen=pg.mkPen(color=POWER_COLOR, width=2))
        self._power_vb.addItem(self.curve_power)

        legend = self.plot.addLegend(offset=(10, 10))
        legend.addItem(self.curve_temp,  "Temperature (C)")
        legend.addItem(self.curve_power, "Power (%)")

        v.addWidget(self.plot)

        # X-axis range fields added so you can zoom into a specific time window
        # without having to use the mouse scroll on the graph.
        # Both fields default to 0, which just leaves pyqtgraph on auto-scale.
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("X-axis min (min):"))
        self.xmin_edit = QLineEdit("0")
        self.xmin_edit.setFixedWidth(60)
        self.xmin_edit.setStyleSheet(EDIT_STYLE)
        range_row.addWidget(self.xmin_edit)
        range_row.addWidget(QLabel("max (min):"))
        self.xmax_edit = QLineEdit("0")
        self.xmax_edit.setFixedWidth(60)
        self.xmax_edit.setStyleSheet(EDIT_STYLE)
        range_row.addWidget(self.xmax_edit)
        apply_btn = QPushButton("Apply Range")
        apply_btn.clicked.connect(self._apply_x_range)
        range_row.addWidget(apply_btn)
        reset_btn = QPushButton("Auto")
        reset_btn.clicked.connect(self._reset_x_range)
        range_row.addWidget(reset_btn)
        range_row.addStretch()
        v.addLayout(range_row)

        return box

    def _apply_x_range(self):
        try:
            xmin = float(self.xmin_edit.text())
            xmax = float(self.xmax_edit.text())
        except ValueError:
            return
        if xmax > xmin:
            self.plot.setXRange(xmin, xmax, padding=0)

    def _reset_x_range(self):
        self.plot.enableAutoRange(axis="x")

    def _sync_power_axis(self):
        self._power_vb.setGeometry(self.plot.getViewBox().sceneBoundingRect())
        self._power_vb.linkedViewChanged(
            self.plot.getViewBox(), self._power_vb.XAxis
        )

    def _hline(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(_HLINE)
        return f


    # -----------------------------------------------------------------------
    # File picker
    # -----------------------------------------------------------------------

    def _browse_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Choose Log File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self.filepath_edit.setText(path)

    # -----------------------------------------------------------------------
    # Connect / Disconnect
    # -----------------------------------------------------------------------

    def _toggle_connect(self, checked: bool):
        if checked:
            self._connect()
        else:
            self._disconnect()

    def _connect(self):
        model = self.model_combo.currentText()
        port  = self.port_edit.currentText().strip()

        # Convert VISA-style port names (ASRL5::INSTR) to real Windows COM port names.
        # list_resources() returns VISA format; minimalmodbus needs COM5 or /dev/ttyS5.
        # Using the same replace() approach as instruments/heater/eurotherm2200.py.
        if "COM" not in port:
            port = port.replace("ASRL", "COM").replace("::INSTR", "")

        try:
            addr = int(self.addr_edit.text().strip())
        except ValueError:
            QMessageBox.critical(
                self, "Input Error",
                "Modbus address must be an integer (usually 1)."
            )
            self.connect_btn.setChecked(False)
            return

        try:
            if HARDWARE_AVAILABLE:
                driver_class = EUROTHERM_DRIVER_MAP[model]
                self.driver  = driver_class(port, addr)
            else:
                self.driver = SimulatedEurotherm()

            first_temp = self.driver.process_value

        except Exception as exc:
            QMessageBox.critical(
                self, "Connection Failed",
                f"Could not connect to {model} on {port}:\n{exc}"
            )
            self.connect_btn.setChecked(False)
            self.driver = None
            return

        if not (TEMP_SANITY_MIN <= first_temp <= TEMP_SANITY_MAX):
            QMessageBox.warning(
                self, "Unusual Reading",
                f"First temperature reading is {first_temp:.1f} C, which is outside "
                f"the expected range ({TEMP_SANITY_MIN:.0f} to {TEMP_SANITY_MAX:.0f} C).\n\n"
                "Check the Eurotherm decimal place setting in its comms menu."
            )

        self._model_name = model
        interval = self._poll_interval()
        self.worker = DataWorker(self.driver, interval_s=interval)
        self.worker.data_ready.connect(self._on_data)
        self.worker.error_occurred.connect(self._on_worker_error)

        # _t0 must be set before worker.start(). The worker fires its first
        # signal almost immediately, and _on_data uses _t0. If it is still
        # None at that point the elapsed-time calculation crashes.
        self._t0        = time.time()
        self._connected = True
        self.worker.start()

        self.connect_btn.setText("Disconnect")
        self.ramp_btn.setEnabled(True)
        self.log_btn.setEnabled(True)
        self.model_combo.setEnabled(False)
        self.port_edit.setEnabled(False)
        self.addr_edit.setEnabled(False)

        self.statusBar().showMessage(
            f"Connected - {model} on {port}, address {addr}."
        )


    def _disconnect(self):
        # Stop logging before stopping the thread that writes to it.
        # Stop the thread before closing the serial port it reads from.
        if self._logging:
            self._stop_logging()

        if self._ramp_running and self.ramp_mgr:
            # Reset the program cleanly before tearing down
            self.ramp_mgr.stop(self._last_temp)
            time.sleep(0.3)

        if self.worker:
            self.worker.stop()
            self.worker = None
        if self.ramp_mgr:
            self.ramp_mgr = None
        if self.driver:
            if hasattr(self.driver, "serial"):
                try:
                    self.driver.serial.close()
                except Exception:
                    pass
            self.driver = None

        self._connected    = False
        self._ramp_running = False
        self._model_name   = ""

        self.connect_btn.setChecked(False)
        self.connect_btn.setText("Connect")
        self.ramp_btn.setEnabled(False)
        self.ramp_btn.setChecked(False)
        self.ramp_btn.setText("Begin Ramp")
        self.log_btn.setEnabled(False)
        self.log_btn.setChecked(False)
        self.log_btn.setText("Start Logging")
        self.model_combo.setEnabled(True)
        self.port_edit.setEnabled(True)
        self.addr_edit.setEnabled(True)

        self.statusBar().showMessage("Disconnected.")

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    def _toggle_logging(self, checked: bool):
        if checked:
            self._start_logging()
        else:
            self._stop_logging()

    def _start_logging(self):
        path = self.filepath_edit.text().strip()
        if not path:
            QMessageBox.warning(
                self, "No File Set",
                "Use Browse... to choose a log file before starting logging."
            )
            self.log_btn.setChecked(False)
            return
        try:
            self._log_file   = open(path, "w", newline="")
            self._log_writer = csv.writer(self._log_file)
            self._log_writer.writerow(
                ["Timestamp", "Elapsed_min", "Temperature_C", "Power_pct", "Working_SP_C"]
            )
            # _log_t0 is set here, not at connect time. This means elapsed minutes
            # in the CSV always starts at 0 from the moment you hit Start Logging,
            # so the file clock matches when you actually started recording.
            self._log_t0  = time.time()
            self._logging = True
            self.log_btn.setText("Stop Logging")
            self.browse_btn.setEnabled(False)
            self.statusBar().showMessage(f"Logging to: {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Logging Error",
                                f"Could not open log file:\n{exc}")
            self.log_btn.setChecked(False)

    def _stop_logging(self):
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
        self._log_file   = None
        self._log_writer = None
        self._log_t0     = None
        self._logging    = False
        self.log_btn.setText("Start Logging")
        self.log_btn.setChecked(False)
        self.browse_btn.setEnabled(True)
        self.statusBar().showMessage("Logging stopped.")


    # -----------------------------------------------------------------------
    # Ramp
    # -----------------------------------------------------------------------

    def _toggle_ramp(self, checked: bool):
        if checked:
            self._start_ramp()
        else:
            self._stop_ramp()

    def _start_ramp(self):
        try:
            target = float(self.set_temp_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Input Error",
                                 "Set Temperature must be a number.")
            self.ramp_btn.setChecked(False)
            return

        try:
            rate = float(self.ramp_rate_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Input Error",
                                 "Ramp Rate must be a number.")
            self.ramp_btn.setChecked(False)
            return

        if rate <= 0:
            QMessageBox.critical(self, "Input Error",
                                 "Ramp Rate must be greater than 0.")
            self.ramp_btn.setChecked(False)
            return

        if rate > MAX_RAMP_RATE_C_PER_MIN:
            answer = QMessageBox.warning(
                self, "Ramp Rate Too High",
                f"{rate:.1f} C/min exceeds the lab maximum of "
                f"{MAX_RAMP_RATE_C_PER_MIN:.0f} C/min.\n\n"
                f"Clamp to {MAX_RAMP_RATE_C_PER_MIN:.0f} C/min and continue?",
                _MSGBOX_YES | _MSGBOX_CANCEL,
            )
            if answer == _MSGBOX_YES:
                rate = MAX_RAMP_RATE_C_PER_MIN
                self.ramp_rate_edit.setText(f"{rate:.1f}")
            else:
                self.ramp_btn.setChecked(False)
                return

        self.ramp_mgr = RampManager(self.driver, self._model_name)
        self.ramp_mgr.ramp_error.connect(self._on_ramp_error)
        self.ramp_mgr.start(target, rate, self._last_temp)
        self._ramp_running = True

        self.ramp_btn.setText("Stop Ramp")
        self.statusBar().showMessage(
            f"Ramping to {target:.1f} C at {rate:.1f} C/min ..."
        )

    def _stop_ramp(self):
        if self.ramp_mgr:
            # Sends program_status = "reset" to the controller, then writes the
            # current temperature as the new setpoint. Without this second step
            # the Eurotherm keeps its old goal temperature and the heaters stay on
            # trying to reach it even after you've clicked stop.
            self.ramp_mgr.stop(self._last_temp)
        self._ramp_running = False
        self.ramp_btn.setChecked(False)
        self.ramp_btn.setText("Begin Ramp")
        self.statusBar().showMessage("Ramp stopped.")

    def _on_ramp_error(self, msg: str):
        if not self._connected:
            return
        self._ramp_running = False
        self.ramp_btn.setChecked(False)
        self.ramp_btn.setText("Begin Ramp")
        self.statusBar().showMessage(f"Ramp stopped - error: {msg}")


    # -----------------------------------------------------------------------
    # Data slot - fires every poll interval via signal from DataWorker
    # -----------------------------------------------------------------------

    def _on_data(self, temp: float, power: float, working_sp: float, status: str):
        self.temp_display.setText(f"{temp:.1f}")
        self.power_display.setText(f"{power:.1f}")
        self.sp_display.setText(f"{working_sp:.1f}")
        self._last_temp = temp

        # Check if the hardware ramp finished on its own (controller reached target).
        # RAMP_DONE_STATES covers the end-of-program status strings across all three models.
        # This auto-resets the button so it doesn't stay stuck saying "Stop Ramp".
        if self._ramp_running and status in RAMP_DONE_STATES:
            self._ramp_running = False
            self.ramp_btn.setChecked(False)
            self.ramp_btn.setText("Begin Ramp")
            self.statusBar().showMessage("Ramp complete - target temperature reached.")

        # x-axis: time since connect, in minutes
        elapsed_min = (time.time() - self._t0) / 60.0

        self._times.append(elapsed_min)
        self._temps.append(temp)
        self._powers.append(power)

        # pyqtgraph requires numpy arrays, not deques or plain lists
        t_arr = np.array(self._times)
        self.curve_temp.setData(t_arr, np.array(self._temps))
        self.curve_power.setData(t_arr, np.array(self._powers))

        if self._log_writer and self._log_t0 is not None:
            # elapsed from _log_t0, not _t0, so the CSV clock starts at 0
            # from when you clicked Start Logging rather than from connect time
            log_elapsed = (time.time() - self._log_t0) / 60.0
            self._log_writer.writerow([
                datetime.now().isoformat(),
                f"{log_elapsed:.4f}",
                f"{temp:.2f}",
                f"{power:.2f}",
                f"{working_sp:.2f}",
            ])
            # Flush after every row so data is on disk even if the program crashes
            self._log_file.flush()

    def _on_worker_error(self, msg: str):
        self.temp_display.setText("ERR")
        self.power_display.setText("ERR")
        self.sp_display.setText("ERR")
        self.statusBar().showMessage(f"Communication error: {msg}", 6000)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _poll_interval(self) -> float:
        try:
            return max(0.5, float(self.log_interval_edit.text()))
        except ValueError:
            return 1.0

    def closeEvent(self, event):
        self._disconnect()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = FurnaceGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
