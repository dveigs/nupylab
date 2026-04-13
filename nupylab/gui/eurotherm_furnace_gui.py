#!/usr/bin/env python3
"""
eurotherm_furnace_gui.py
NUPyLab - Standalone Eurotherm Furnace Control GUI

Connects to a Eurotherm 2200 / 2400 / 3216 over RS-485, reads live
temperature and heater output, ramps the setpoint at a controlled rate,
and optionally logs to CSV.

Ramping uses the controller's built-in programmer - matches instruments/heater/.

Dependencies: PyQt5 or PyQt6, pyqtgraph, numpy, nupylab drivers
"""

import sys
import csv
import time
import threading
from datetime import datetime
from collections import deque

from nupylab.drivers.eurotherm2200 import Eurotherm2200
from nupylab.drivers.eurotherm2400 import Eurotherm2400
from nupylab.drivers.eurotherm3216 import Eurotherm3216

EUROTHERM_DRIVER_MAP = {
    "Eurotherm 2200": Eurotherm2200,
    "Eurotherm 2400": Eurotherm2400,
    "Eurotherm 3216": Eurotherm3216,
}

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
    _SERIAL_PORTS = ["COM1"]

import numpy as np

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

if _PYQT6:
    _HLINE         = QFrame.Shape.HLine
    _MSGBOX_YES    = QMessageBox.StandardButton.Yes
    _MSGBOX_CANCEL = QMessageBox.StandardButton.Cancel
else:
    _HLINE         = QFrame.HLine
    _MSGBOX_YES    = QMessageBox.Yes
    _MSGBOX_CANCEL = QMessageBox.Cancel

import pyqtgraph as pg


EUROTHERM_MODEL_NAMES   = ["Eurotherm 2200", "Eurotherm 2400", "Eurotherm 3216"]
MAX_RAMP_RATE_C_PER_MIN = 10.0
RAMP_DONE_STATES        = frozenset({"off", "end", "complete"})
TEMP_SANITY_MIN         = -50.0
TEMP_SANITY_MAX         = 1800.0

TEMP_COLOR  = "#ff5555"
POWER_COLOR = "#55aaff"
LIVE_BG     = "#0a0a0a"
LIVE_FG     = "#00e676"
EDIT_STYLE  = "background-color: #dff0f7; color: #111111;"


# ---------------------------------------------------------------------------
# Background polling thread
# ---------------------------------------------------------------------------

class DataWorker(QObject):
    """Polls the furnace on a background thread every interval_s seconds."""

    data_ready     = pyqtSignal(float, float, float, str)
    error_occurred = pyqtSignal(str)

    def __init__(self, driver, interval_s: float = 1.0):
        super().__init__()
        self.driver     = driver
        self.interval_s = interval_s
        self._active    = False

    def start(self):
        self._active = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._active = False

    def _loop(self):
        while self._active:
            try:
                temp       = self.driver.process_value
                power      = self.driver.output_level
                working_sp = self.driver.working_setpoint
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
    """Writes ramp parameters once and starts the controller's built-in program."""

    ramp_error = pyqtSignal(str)

    def __init__(self, driver, model_name: str):
        super().__init__()
        self.driver     = driver
        self.model_name = model_name

    def start(self, target: float, rate: float, current_temp: float):
        threading.Thread(
            target=self._setup, args=(target, rate, current_temp), daemon=True
        ).start()

    def stop(self, current_temp: float):
        threading.Thread(
            target=self._teardown, args=(current_temp,), daemon=True
        ).start()

    def _setup(self, target: float, rate: float, current_temp: float):
        try:
            if "2400" in self.model_name:
                self._start_2400(target, rate)
            elif "3216" in self.model_name:
                self._start_3216(target, rate, current_temp)
            else:
                self._start_2200(target, rate, current_temp)
        except Exception as exc:
            self.ramp_error.emit(str(exc))

    def _teardown(self, current_temp: float):
        try:
            self.driver.program_status = "reset"
        except Exception:
            pass
        try:
            self.driver.target_setpoint = current_temp
        except Exception:
            pass

    def _start_2200(self, target: float, rate: float, current_temp: float):
        self.driver.program_status      = "reset"
        self.driver.active_setpoint     = 1
        self.driver.end_type            = "dwell"
        self.driver.setpoint1           = current_temp
        self.driver.setpoint_rate_limit = rate
        self.driver.setpoint2           = target
        self.driver.dwell_time          = 1
        self.driver.program_status      = "run"

    def _start_2400(self, target: float, rate: float):
        self.driver.program_status = "reset"
        self.driver.current_program = 1
        self.driver.programs[1].refresh()
        self.driver.programs[1].segments[1]["segment type"]    = "ramp rate"
        self.driver.programs[1].segments[1]["rate"]            = rate
        self.driver.programs[1].segments[1]["target setpoint"] = target
        self.driver.programs[1].segments[2]["segment type"]    = "dwell"
        self.driver.programs[1].segments[2]["duration"]        = 1
        self.driver.programs[1].segments[3]["segment type"]    = "end"
        self.driver.programs[1].segments[3]["end type"]        = "dwell"
        self.driver.program_status = "run"

    def _start_3216(self, target: float, rate: float, current_temp: float):
        self.driver.program_status = "reset"
        self.driver.end_type = "dwell"
        for segment in self.driver.segments:
            segment.clear()
        self.driver.segments[-1].target_setpoint = target
        self.driver.segments[-1].ramp_rate       = rate
        self.driver.segments[-1].dwell           = 1
        self.driver.program_status = "run"


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class FurnaceGUI(QMainWindow):

    _reconnect_success = pyqtSignal()
    _reconnect_failed  = pyqtSignal(str)
    _reconnect_status  = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.driver      = None
        self.worker      = None
        self.ramp_mgr    = None
        self._log_file   = None
        self._log_writer = None
        self._t0         = None
        self._log_t0     = None
        self._connected  = False
        self._logging    = False
        self._ramp_running      = False
        self._model_name        = ""
        self._user_disconnected = False
        self._reconnecting      = False

        N = 600
        self._times  = deque(maxlen=N)
        self._temps  = deque(maxlen=N)
        self._powers = deque(maxlen=N)

        self._last_temp = 25.0

        self._build_ui()
        self.setWindowTitle("NUPyLab - Eurotherm Furnace Control")
        self.setMinimumSize(1200, 700)

        # Connect reconnect signals once here so they don't accumulate
        # extra connections if the user disconnects and reconnects repeatedly.
        self._reconnect_success.connect(self._on_reconnect_success)
        self._reconnect_failed.connect(self._on_reconnect_failed)
        self._reconnect_status.connect(lambda msg: self.statusBar().showMessage(msg))

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

        self.curve_temp  = self.plot.plot(pen=pg.mkPen(color=TEMP_COLOR, width=2))
        self.curve_power = pg.PlotCurveItem(pen=pg.mkPen(color=POWER_COLOR, width=2))
        self._power_vb.addItem(self.curve_power)

        legend = self.plot.addLegend(offset=(10, 10))
        legend.addItem(self.curve_temp,  "Temperature (C)")
        legend.addItem(self.curve_power, "Power (%)")

        v.addWidget(self.plot)

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

        if "COM" not in port:
            port = port.replace("ASRL", "COM").replace("::INSTR", "")

        try:
            addr = int(self.addr_edit.text().strip())
        except ValueError:
            QMessageBox.critical(self, "Input Error",
                                 "Modbus address must be an integer (usually 1).")
            self.connect_btn.setChecked(False)
            return

        try:
            driver_class = EUROTHERM_DRIVER_MAP[model]
            self.driver  = driver_class(port, addr)
            first_temp   = self.driver.process_value
        except Exception as exc:
            QMessageBox.critical(self, "Connection Failed",
                                 f"Could not connect to {model} on {port}:\n{exc}")
            self.connect_btn.setChecked(False)
            self.driver = None
            return

        if not (TEMP_SANITY_MIN <= first_temp <= TEMP_SANITY_MAX):
            QMessageBox.warning(
                self, "Unusual Reading",
                f"First reading is {first_temp:.1f} C - check the Eurotherm "
                "decimal place setting in its comms menu."
            )

        self._model_name        = model
        self._t0                = time.time()
        self._connected         = True
        self._user_disconnected = False
        self._reconnecting      = False

        self._times.clear()
        self._temps.clear()
        self._powers.clear()
        self.curve_temp.setData([], [])
        self.curve_power.setData([], [])

        self.worker = DataWorker(self.driver, interval_s=self._poll_interval())
        self.worker.data_ready.connect(self._on_data)
        self.worker.error_occurred.connect(self._on_worker_error)
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
        self._user_disconnected = True
        self._reconnecting      = False

        if self._logging:
            self._stop_logging()

        if self._ramp_running and self.ramp_mgr:
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
            QMessageBox.warning(self, "No File Set",
                                "Use Browse... to choose a log file before starting.")
            self.log_btn.setChecked(False)
            return
        try:
            self._log_file   = open(path, "w", newline="")
            self._log_writer = csv.writer(self._log_file)
            self._log_writer.writerow(
                ["Timestamp", "Elapsed_min", "Temperature_C", "Power_pct", "Working_SP_C"]
            )
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
            QMessageBox.critical(self, "Input Error", "Set Temperature must be a number.")
            self.ramp_btn.setChecked(False)
            return

        try:
            rate = float(self.ramp_rate_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Ramp Rate must be a number.")
            self.ramp_btn.setChecked(False)
            return

        if rate <= 0:
            QMessageBox.critical(self, "Input Error", "Ramp Rate must be greater than 0.")
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
        self.statusBar().showMessage(f"Ramp error: {msg}")

    # -----------------------------------------------------------------------
    # Data slot
    # -----------------------------------------------------------------------

    def _on_data(self, temp: float, power: float, working_sp: float, status: str):
        self.temp_display.setText(f"{temp:.1f}")
        self.power_display.setText(f"{power:.1f}")
        self.sp_display.setText(f"{working_sp:.1f}")
        self._last_temp = temp

        if self._ramp_running and status in RAMP_DONE_STATES:
            self._ramp_running = False
            self.ramp_btn.setChecked(False)
            self.ramp_btn.setText("Begin Ramp")
            self.statusBar().showMessage("Ramp complete.")

        elapsed_min = (time.time() - self._t0) / 60.0
        self._times.append(elapsed_min)
        self._temps.append(temp)
        self._powers.append(power)

        t_arr = np.array(self._times)
        self.curve_temp.setData(t_arr, np.array(self._temps))
        self.curve_power.setData(t_arr, np.array(self._powers))

        if self._log_writer and self._log_t0 is not None:
            log_elapsed = (time.time() - self._log_t0) / 60.0
            self._log_writer.writerow([
                datetime.now().isoformat(),
                f"{log_elapsed:.4f}",
                f"{temp:.2f}",
                f"{power:.2f}",
                f"{working_sp:.2f}",
            ])
            self._log_file.flush()

    # -----------------------------------------------------------------------
    # Error handling and auto-reconnect
    # -----------------------------------------------------------------------

    def _on_worker_error(self, msg: str):
        self.temp_display.setText("ERR")
        self.power_display.setText("ERR")
        self.sp_display.setText("ERR")
        if not self._user_disconnected and not self._reconnecting:
            self._start_reconnect(msg)
        else:
            self.statusBar().showMessage(f"Communication error: {msg}", 6000)

    def _start_reconnect(self, initial_error: str):
        self._reconnecting = True
        if self.worker:
            self.worker.stop()
            self.worker = None
        self._reconnect_status.emit(
            f"Connection lost ({initial_error}) - attempting to reconnect..."
        )
        threading.Thread(target=self._reconnect_loop, daemon=True).start()

    def _reconnect_loop(self):
        model = self._model_name
        port  = self.port_edit.currentText().strip()
        if "COM" not in port:
            port = port.replace("ASRL", "COM").replace("::INSTR", "")
        try:
            addr = int(self.addr_edit.text().strip())
        except ValueError:
            self._reconnect_failed.emit("Invalid Modbus address - cannot reconnect.")
            return

        for attempt in range(1, 6):
            if self._user_disconnected:
                return
            self._reconnect_status.emit(f"Reconnect attempt {attempt}/5...")
            try:
                driver_class = EUROTHERM_DRIVER_MAP[model]
                new_driver   = driver_class(port, addr)
                _            = new_driver.process_value
                self.driver  = new_driver
                self._reconnect_success.emit()
                return
            except Exception:
                pass
            time.sleep(1.0)

        self._reconnect_failed.emit(
            "Lost connection to furnace - could not reconnect after 5 attempts."
        )

    def _on_reconnect_success(self):
        self._reconnecting = False
        self.worker = DataWorker(self.driver, interval_s=self._poll_interval())
        self.worker.data_ready.connect(self._on_data)
        self.worker.error_occurred.connect(self._on_worker_error)
        self.worker.start()
        self.statusBar().showMessage("Reconnected successfully.")

    def _on_reconnect_failed(self, msg: str):
        self._reconnecting      = False
        self._user_disconnected = True
        self.statusBar().showMessage(f"Error: {msg}")
        self._disconnect()

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