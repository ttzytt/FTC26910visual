from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QCheckBox, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from typing import Dict, Type, Set, Union
from enum import Enum
from src.preprocessor import PreprocType


class DebugControlWidget(QWidget):
    options_changed = Signal()

    def __init__(self, detector_debug_type: Type[Enum], parent=None):
        super().__init__(parent)
        self.detector_states: Dict[Enum, bool] = {}
        self.preproc_states: Dict[PreprocType, bool] = {}

        self.detector_checkboxes: Dict[Enum, QCheckBox] = {}
        self.preproc_checkboxes: Dict[PreprocType, QCheckBox] = {}

        self.init_ui(detector_debug_type)
        self.setWindowTitle("Debug Controls")
        self.setMinimumSize(400, 600)

    def init_ui(self, detector_debug_type: Type[Enum]):
        layout = QVBoxLayout()
        scroll = QScrollArea()
        content = QWidget()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)

        main_layout = QVBoxLayout(content)

        # Detector debug group
        detector_group = QGroupBox("Detector Debug Options")
        detector_layout = QVBoxLayout()
        for opt in detector_debug_type:
            checkbox = QCheckBox(opt.value)
            checkbox.stateChanged.connect(self._handle_detector_change)
            self.detector_states[opt] = False
            self.detector_checkboxes[opt] = checkbox  # 记录 checkbox
            detector_layout.addWidget(checkbox)
        detector_group.setLayout(detector_layout)

        # Preprocessor debug group
        preproc_group = QGroupBox("Preprocessor Debug Steps")
        preproc_layout = QVBoxLayout()
        for step in PreprocType:
            checkbox = QCheckBox(step.value)
            checkbox.stateChanged.connect(self._handle_preproc_change)
            self.preproc_states[step] = False
            self.preproc_checkboxes[step] = checkbox  # 记录 checkbox
            preproc_layout.addWidget(checkbox)
        preproc_group.setLayout(preproc_layout)

        main_layout.addWidget(detector_group)
        main_layout.addWidget(preproc_group)
        layout.addWidget(scroll)
        self.setLayout(layout)

    def _handle_detector_change(self, state):
        checkbox = self.sender()
        if checkbox:
            opt = next(
                opt for opt in self.detector_states if opt.value == checkbox.text()
            )
            self.detector_states[opt] = checkbox.isChecked()
            self.options_changed.emit()

    def _handle_preproc_change(self, state):
        checkbox = self.sender()
        if checkbox:
            step = next(
                step for step in self.preproc_states if step.value == checkbox.text()
            )
            self.preproc_states[step] = checkbox.isChecked()
            self.options_changed.emit()

    def set_states(self, detector_states: Dict[Enum, bool], preproc_states: Dict[PreprocType, bool]):
        for opt, enabled in detector_states.items():
            self.detector_states[opt] = enabled
            if opt in self.detector_checkboxes:
                self.detector_checkboxes[opt].setChecked(enabled)

        for step, enabled in preproc_states.items():
            self.preproc_states[step] = enabled
            if step in self.preproc_checkboxes:
                self.preproc_checkboxes[step].setChecked(enabled)

    def get_states(self):
        return self.detector_states.copy(), self.preproc_states.copy()
