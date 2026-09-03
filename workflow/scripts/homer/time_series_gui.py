"""Interactive GUI to plot fNIRS probe and channel time courses.

Initial Contributors:
    - Sung Ahn | ahnsm@bu.edu | 2024
"""

import sys
import time
import warnings
import pickle
import gzip
import os
import shutil
import psutil
import yaml
import re
import subprocess
import copy
from datetime import datetime, timedelta
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtCore, QtWidgets, QtGui
from PySide6.QtGui import QAction
from matplotlib.figure import Figure

import cedalion
import cedalion.dataclasses as cdc

warnings.simplefilter("ignore")


def _resolve_conda_command():
    """Return an executable Conda command, including .bat on Windows."""
    candidates = ['conda.exe', 'conda.bat', 'conda']
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    user_home = os.path.expanduser('~')
    if sys.platform == 'win32':
        candidates = [
            os.path.join(user_home, 'AppData', 'Local', 'miniconda3', 'condabin', 'conda.bat'),
            os.path.join(user_home, 'AppData', 'Local', 'anaconda3', 'condabin', 'conda.bat'),
            os.path.join(user_home, 'miniconda3', 'condabin', 'conda.bat'),
            os.path.join(user_home, 'anaconda3', 'condabin', 'conda.bat'),
            r'C:\ProgramData\miniconda3\condabin\conda.bat',
            r'C:\ProgramData\anaconda3\condabin\conda.bat',
        ]
    else:
        candidates = [
            os.path.join(user_home, 'miniconda3', 'bin', 'conda'),
            os.path.join(user_home, 'anaconda3', 'bin', 'conda'),
            '/opt/miniconda3/bin/conda',
            '/opt/anaconda3/bin/conda',
        ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not locate the Conda executable. Start HomerPy from an "
        "Anaconda/Miniconda prompt or add Conda's condabin directory to PATH."
    )


def _resolve_conda_env_python(conda_env):
    """Return python.exe for a Conda env path/name when it can be resolved."""
    if not conda_env:
        return None

    env_path = os.path.expanduser(conda_env)
    if os.path.isdir(env_path):
        candidates = (
            [os.path.join(env_path, 'python.exe')] if sys.platform == 'win32'
            else [os.path.join(env_path, 'bin', 'python')]
        )
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

    if os.path.isabs(env_path):
        return None

    conda_prefix = os.environ.get('CONDA_PREFIX')
    candidate_roots = []
    if conda_prefix:
        candidate_roots.append(os.path.join(os.path.dirname(conda_prefix), conda_env))

    user_home = os.path.expanduser('~')
    if sys.platform == 'win32':
        candidate_roots.extend([
            os.path.join(user_home, 'AppData', 'Local', 'miniconda3', 'envs', conda_env),
            os.path.join(user_home, 'AppData', 'Local', 'anaconda3', 'envs', conda_env),
        ])
    else:
        candidate_roots.extend([
            os.path.join(user_home, 'miniconda3', 'envs', conda_env),
            os.path.join(user_home, 'anaconda3', 'envs', conda_env),
        ])

    for root in candidate_roots:
        python_path = (
            os.path.join(root, 'python.exe') if sys.platform == 'win32'
            else os.path.join(root, 'bin', 'python')
        )
        if os.path.isfile(python_path):
            return python_path

    return None


def _build_snakemake_command(cmd, conda_env=None):
    """Build a runnable command, preferring env Python over conda run."""
    env_python = _resolve_conda_env_python(conda_env)
    if env_python:
        if cmd and cmd[0] == 'snakemake':
            return [env_python, '-m', 'snakemake', *cmd[1:]]
        return [env_python, '-m', *cmd]

    conda_cmd = _resolve_conda_command()
    return [conda_cmd, 'run', '-n', conda_env, '--no-capture-output', *cmd]


class ConfigEditorDialog(QtWidgets.QDialog):
    """Dialog for editing YAML configuration blocks"""
    
    def __init__(self, config_data, block_name, readonly_keys=None, field_tooltips=None, parent=None, file_map=None, subjects=None):
        super().__init__(parent)
        self.config_data = config_data
        self.block_name = block_name
        self.readonly_keys = readonly_keys or []
        self.field_tooltips = field_tooltips or {}  # Map of field_key -> tooltip text
        self.field_widgets = {}
        self.file_map = file_map  # For dataset matching calculations
        self.subjects = subjects  # Full subject list from GUI
        self.original_derivatives_subfolder = config_data.get('derivatives_subfolder', '')  # Track original for switch detection
        
        self.setWindowTitle(f"Edit {block_name.replace('_', ' ').title()} Configuration")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        
        self.init_ui()
    
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # Initialize matching_info_label early (before building form)
        self.matching_info_label = None
        
        # Add scroll area for the form
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QtWidgets.QWidget()
        self.form_layout = QtWidgets.QFormLayout()
        scroll_widget.setLayout(self.form_layout)
        scroll.setWidget(scroll_widget)
        
        # Build form from config data
        self._build_form(self.config_data, "")
        
        layout.addWidget(scroll)
        
        # Add matching info label for dataset configuration (below form)
        if self.block_name == 'dataset' and self.file_map and self.subjects:
            self.matching_info_label = QtWidgets.QLabel()
            self.matching_info_label.setStyleSheet("font-size: 11pt;")
            self.matching_info_label.setWordWrap(True)
            layout.addWidget(self.matching_info_label)
            # Calculate initial matching info
            self._update_matching_info()
            
            # Add "Update Dataset Info" button for dataset configuration
            update_button = QtWidgets.QPushButton("🔄 Update Dataset Info")
            update_button.setToolTip("Re-scan available subjects, tasks, and runs from BIDS folder")
            update_button.setMaximumWidth(200)
            update_button.clicked.connect(self._update_dataset_info)
            layout.addWidget(update_button)
        
        # Add buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)

    def _is_dark_mode(self):
        return self.palette().color(QtGui.QPalette.Window).lightness() < 128

    def _theme_color(self, role):
        dark = self._is_dark_mode()
        colors = {
            "text": "#F0F0F0" if dark else "#111111",
            "muted": "#A8A8A8" if dark else "#666666",
            "background": "#2B2B2B" if dark else "#F0F0F0",
            "panel": "#333333" if dark else "#FFFFFF",
            "readonly": "#3A3A3A" if dark else "#F0F0F0",
            "border": "#5A5A5A" if dark else "#C8C8C8",
        }
        return colors[role]

    def _unit_label_stylesheet(self):
        return (
            f"color: {self._theme_color('muted')}; "
            f"background-color: {self._theme_color('readonly')}; "
            "padding: 3px 8px; "
            f"border: 1px solid {self._theme_color('border')}; "
            "border-radius: 3px;"
        )

    def _readonly_field_stylesheet(self):
        return (
            f"color: {self._theme_color('muted')}; "
            f"background-color: {self._theme_color('readonly')};"
        )
    
    def _build_form(self, data, prefix=""):
        """Recursively build form from nested dict"""
        if not isinstance(data, dict):
            return
            
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Add a section label for nested dicts
                label = QtWidgets.QLabel(f"<b>{key}:</b>")
                self.form_layout.addRow(label)
                self._build_form(value, full_key)
            else:
                # Create appropriate widget for the value
                is_readonly = full_key in self.readonly_keys
                widget = self._create_widget(value, is_readonly, key)
                self.field_widgets[full_key] = (widget, value)
                
                # Connect signal for dataset matching info updates
                if self.block_name == 'dataset' and self.file_map and self.subjects:
                    if key in ['subjects_to_exclude', 'task']:
                        if isinstance(widget, QtWidgets.QLineEdit):
                            widget.textChanged.connect(self._update_matching_info)
                
                # Create label with tooltip if available
                label = QtWidgets.QLabel(f"{key}:")
                # Try multiple key formats to find tooltip
                tooltip = None
                if key in self.field_tooltips:
                    tooltip = self.field_tooltips[key]
                elif full_key in self.field_tooltips:
                    tooltip = self.field_tooltips[full_key]
                
                # Always set a tooltip, even if empty, so users know hover works
                if tooltip:
                    label.setToolTip(tooltip)
                else:
                    label.setToolTip("(no description available)")
                
                self.form_layout.addRow(label, widget)
    
    def _create_widget(self, value, readonly=False, key=None):
        """Create appropriate widget based on value type"""
        # Special handling for derivatives_subfolder in dataset block
        if key == 'derivatives_subfolder' and self.block_name == 'dataset' and not readonly:
            widget = QtWidgets.QComboBox()
            widget.setEditable(True)  # Allow typing new folder names
            widget.setInsertPolicy(QtWidgets.QComboBox.NoInsert)  # Don't auto-add typed values
            
            # Add "Create New Pipeline..." as first option
            widget.addItem("📁 Create New Pipeline...")
            widget.insertSeparator(1)  # Add separator after the create option
            
            # Populate with available pipeline folders
            available_pipelines = self._get_available_pipelines()
            if available_pipelines:
                widget.addItems(available_pipelines)
            
            # Set current value (skip the "Create New" option)
            current_value = str(value) if value else ""
            if current_value:
                # Set to current value (will add if not in list)
                index = widget.findText(current_value)
                if index >= 0:
                    widget.setCurrentIndex(index)
                else:
                    widget.setEditText(current_value)
            elif len(available_pipelines) > 0:
                # If no current value but pipelines exist, select first pipeline (not "Create New")
                widget.setCurrentIndex(2)  # Index 0 is "Create New", 1 is separator, 2 is first pipeline
            
            # Connect signal to handle "Create New" selection
            # Store original value for reset purposes
            widget.setProperty('original_value', current_value)
            widget.currentTextChanged.connect(
                lambda text, w=widget: self._handle_create_new_pipeline(w, text)
            )
            
            widget.setToolTip("Select existing pipeline, type new name, or choose 'Create New Pipeline...'")
            return widget
        
        if isinstance(value, bool):
            widget = QtWidgets.QCheckBox()
            widget.setChecked(value)
            widget.setEnabled(not readonly)
        elif isinstance(value, list):
            widget = QtWidgets.QLineEdit()
            # Convert list to comma-separated string for editing
            widget.setText(", ".join(str(v) for v in value))
            widget.setReadOnly(readonly)
        elif isinstance(value, (int, float)):
            widget = QtWidgets.QLineEdit()
            widget.setText(str(value))
            widget.setReadOnly(readonly)
        else:  # string or other
            # Check if this is a string with units (e.g., "5 mm", "10 second", "0.5 Hz")
            str_value = str(value)
            unit_pattern = r'^([\d\.\-e]+)\s+(mm|second|seconds|Hz|micromolar\*\*2|micromolar)$'
            match = re.match(unit_pattern, str_value)
            
            if match and not readonly:
                # Create a composite widget with editable number and read-only unit
                widget = QtWidgets.QWidget()
                layout = QtWidgets.QHBoxLayout()
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setSpacing(5)
                
                # Editable number field
                number_field = QtWidgets.QLineEdit()
                number_field.setText(match.group(1))
                number_field.setMaximumWidth(100)
                layout.addWidget(number_field)
                
                # Read-only unit label
                unit_label = QtWidgets.QLabel(match.group(2))
                unit_label.setStyleSheet(self._unit_label_stylesheet())
                layout.addWidget(unit_label)
                
                layout.addStretch()
                widget.setLayout(layout)
                
                # Store references for later retrieval
                widget._number_field = number_field
                widget._unit = match.group(2)
            else:
                # Regular string field
                widget = QtWidgets.QLineEdit()
                widget.setText(str_value)
                widget.setReadOnly(readonly)
        
        if readonly and isinstance(widget, QtWidgets.QLineEdit):
            widget.setStyleSheet(self._readonly_field_stylesheet())
        
        return widget
    
    def _update_matching_info(self):
        """Calculate and display matching subjects/runs based on current form values"""
        if not self.matching_info_label or not self.file_map or not self.subjects:
            return
        
        try:
            # Get current values from widgets
            subjects_to_exclude_text = ""
            task_text = ""
            
            for key, (widget, _) in self.field_widgets.items():
                if 'subjects_to_exclude' in key and isinstance(widget, QtWidgets.QLineEdit):
                    subjects_to_exclude_text = widget.text().strip()
                elif key == 'task' and isinstance(widget, QtWidgets.QLineEdit):
                    task_text = widget.text().strip()
            
            # Parse subjects_to_exclude (comma-separated list with optional quotes/brackets)
            excluded_subjects = set()
            if subjects_to_exclude_text:
                # Remove brackets if present (handles ["01", "02"] format)
                text = subjects_to_exclude_text.strip().strip('[]')
                # Split by comma and clean up each item
                for s in text.split(','):
                    s = s.strip().strip('"\'').strip()
                    if s:
                        # Add "sub-" prefix if not already present
                        if not s.startswith('sub-'):
                            s = f"sub-{s}"
                        excluded_subjects.add(s)
            
            
            # Calculate matching subjects
            all_subjects = set(self.subjects)
            matching_subjects = all_subjects - excluded_subjects
            num_matching_subjects = len(matching_subjects)
            
            # Calculate matching runs based on task pattern
            # Count runs in file_map that match the task for non-excluded subjects
            # Use BIDS naming convention: task-<name>_ to ensure exact task match
            num_matching_runs = 0
            if task_text:  # Only count if task is specified
                for subject in matching_subjects:
                    if subject in self.file_map:
                        for run in self.file_map[subject].keys():
                            # Check if run matches task pattern using BIDS convention
                            # Task name is bounded by 'task-' and '_'
                            if f"task-{task_text}_" in run:
                                num_matching_runs += 1
            
            # Format and display the information
            info_text = (
                f"<b>📊 Pipeline Scope:</b><br>"
                f"• <b>{num_matching_subjects}</b> subjects will be included "
                f"(out of {len(all_subjects)} total)<br>"
                f"• <b>{num_matching_runs}</b> runs match the task pattern"
            )
            
            self.matching_info_label.setText(info_text)
            
        except Exception as e:
            print(f"Error updating matching info: {e}")
            self.matching_info_label.setText("⚠️ Error calculating matches")
    
    def _update_dataset_info(self):
        """Update dataset tooltips by re-scanning available data from parent GUI"""
        if not self.parent():
            QtWidgets.QMessageBox.warning(
                self,
                "Cannot Update",
                "Cannot access parent GUI to update dataset info."
            )
            return
        
        try:
            # Get updated tooltips from parent
            parent_gui = self.parent()
            if hasattr(parent_gui, '_generate_dataset_tooltips'):
                new_tooltips = parent_gui._generate_dataset_tooltips()
                
                # Update field_tooltips
                self.field_tooltips.update(new_tooltips)
                
                # Update tooltips on existing form labels
                for i in range(self.form_layout.rowCount()):
                    label_item = self.form_layout.itemAt(i, QtWidgets.QFormLayout.LabelRole)
                    if label_item:
                        label_widget = label_item.widget()
                        if isinstance(label_widget, QtWidgets.QLabel):
                            # Extract field key from label text
                            label_text = label_widget.text().rstrip(':')
                            if label_text in new_tooltips:
                                label_widget.setToolTip(new_tooltips[label_text])
                
                # Update matching info as well
                self._update_matching_info()
                
                QtWidgets.QMessageBox.information(
                    self,
                    "Dataset Info Updated",
                    "Dataset information has been refreshed from the BIDS folder.\n"
                    "Hover over field labels to see updated available options."
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Cannot Update",
                    "Parent GUI does not support dataset info updates."
                )
        
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Update Error",
                f"Error updating dataset info:\n{str(e)}"
            )
            print(f"Error in _update_dataset_info: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_available_pipelines(self):
        """Get list of available pipeline folders from derivatives/cedalion/"""
        available_pipelines = []
        
        try:
            # Get root_dir and cedalion path from parent GUI
            parent_gui = self.parent()
            if not parent_gui or not hasattr(parent_gui, 'path_to_data'):
                return available_pipelines
            
            # Get the root_dir from config
            root_dir = self.config_data.get('root_dir', '')
            if not root_dir or not os.path.exists(root_dir):
                return available_pipelines
            
            cedalion_path = os.path.join(root_dir, 'derivatives', 'cedalion')
            if not os.path.exists(cedalion_path):
                return available_pipelines
            
            # List all subdirectories in derivatives/cedalion/
            for item in os.listdir(cedalion_path):
                item_path = os.path.join(cedalion_path, item)
                if os.path.isdir(item_path):
                    available_pipelines.append(item)
            
            # Sort for consistent display
            available_pipelines.sort()
            
        except Exception as e:
            print(f"Error getting available pipelines: {e}")
        
        return available_pipelines
    
    def _handle_create_new_pipeline(self, widget, text):
        """Handle when user selects 'Create New Pipeline...'"""
        if text == "📁 Create New Pipeline...":
            # Get the original/previous value
            original_value = widget.property('original_value') or ""
            
            # Prompt user for new pipeline folder name
            new_name, ok = QtWidgets.QInputDialog.getText(
                self,
                "Create New Pipeline",
                "Enter new pipeline folder name:",
                QtWidgets.QLineEdit.Normal,
                ""
            )
            
            if ok and new_name:
                # Validate folder name (no special characters except underscore and dash)
                import re
                if not re.match(r'^[a-zA-Z0-9_-]+$', new_name):
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Invalid Name",
                        "Pipeline folder name can only contain letters, numbers, underscores, and dashes."
                    )
                    # Reset to original value
                    widget.blockSignals(True)
                    index = widget.findText(original_value)
                    if index >= 0:
                        widget.setCurrentIndex(index)
                    else:
                        widget.setEditText(original_value)
                    widget.blockSignals(False)
                    return
                
                # Set the new name in the combobox
                widget.blockSignals(True)
                widget.setEditText(new_name)
                widget.setProperty('original_value', new_name)  # Update stored value
                widget.blockSignals(False)
            else:
                # User cancelled - reset to original value
                widget.blockSignals(True)
                index = widget.findText(original_value)
                if index >= 0:
                    widget.setCurrentIndex(index)
                else:
                    widget.setEditText(original_value)
                widget.blockSignals(False)
    
    def get_updated_data(self):
        """Extract updated values from form widgets"""
        updated = {}
        
        for full_key, (widget, original_value) in self.field_widgets.items():
            keys = full_key.split('.')
            
            # Get the new value from widget
            if isinstance(widget, QtWidgets.QCheckBox):
                new_value = widget.isChecked()
            elif isinstance(widget, QtWidgets.QComboBox):
                # For QComboBox (used for derivatives_subfolder)
                new_value = widget.currentText().strip()
                # Skip the "Create New Pipeline..." option
                if new_value.startswith("📁 Create New Pipeline"):
                    continue
            elif isinstance(widget, QtWidgets.QLineEdit):
                text = widget.text().strip()
                if isinstance(original_value, list):
                    # Parse comma-separated list back to original format
                    new_value = [v.strip().strip('"\'') for v in text.split(',') if v.strip()]
                elif isinstance(original_value, int):
                    try:
                        new_value = int(text)
                    except ValueError:
                        new_value = original_value
                elif isinstance(original_value, float):
                    try:
                        new_value = float(text)
                    except ValueError:
                        new_value = original_value
                else:
                    new_value = text
            elif hasattr(widget, '_number_field') and hasattr(widget, '_unit'):
                # This is a composite widget with units
                number_text = widget._number_field.text().strip()
                unit = widget._unit
                new_value = f"{number_text} {unit}"
            else:
                continue
            
            # Build nested dict structure
            current = updated
            for i, key in enumerate(keys[:-1]):
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = new_value
        
        return updated


class SnakemakeSetupDialog(QtWidgets.QDialog):
    """Dialog for setting up Snakemake pipeline (selecting Snakefile and loading config)"""
    
    def __init__(self, parent=None, current_snakefile=None, current_config=None):
        super().__init__(parent)
        self.snakefile_path = current_snakefile or ""
        self.config_path = current_config or ""
        
        self.setWindowTitle("Setup Snakemake Pipeline")
        self.setMinimumWidth(600)
        
        self.init_ui()
    
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        form_layout = QtWidgets.QFormLayout()
        
        # Snakefile path
        snakefile_layout = QtWidgets.QHBoxLayout()
        self.snakefile_edit = QtWidgets.QLineEdit()
        if self.snakefile_path:
            self.snakefile_edit.setText(self.snakefile_path)
        else:
            self.snakefile_edit.setPlaceholderText("Path to Snakefile")
        snakefile_browse_btn = QtWidgets.QPushButton("Browse...")
        snakefile_browse_btn.clicked.connect(self._browse_snakefile)
        snakefile_layout.addWidget(self.snakefile_edit)
        snakefile_layout.addWidget(snakefile_browse_btn)
        form_layout.addRow("Snakefile:", snakefile_layout)
        
        # Config file path
        config_layout = QtWidgets.QHBoxLayout()
        self.config_edit = QtWidgets.QLineEdit()
        if self.config_path:
            self.config_edit.setText(self.config_path)
        else:
            self.config_edit.setPlaceholderText("Auto-detected from Snakefile location")
        config_browse_btn = QtWidgets.QPushButton("Browse...")
        config_browse_btn.clicked.connect(self._browse_config)
        config_layout.addWidget(self.config_edit)
        config_layout.addWidget(config_browse_btn)
        form_layout.addRow("Config File:", config_layout)
        
        layout.addLayout(form_layout)
        
        # Info label
        info_label = QtWidgets.QLabel(
            "Select a Snakefile to load its configuration.\n"
            "Config file will be auto-detected from: <snakefile_dir>/config/<snakefile>.yaml"
        )
        info_label.setWordWrap(True)
        muted_color = "#A8A8A8" if self.palette().color(QtGui.QPalette.Window).lightness() < 128 else "#666666"
        info_label.setStyleSheet(f"color: {muted_color}; font-size: 10pt;")
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        # Add buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def _browse_snakefile(self):
        # Try to determine a smart starting directory
        start_dir = ""
        
        # Method 1: If we already have a snakefile path, start in that directory
        if self.snakefile_edit.text():
            existing_path = self.snakefile_edit.text()
            if os.path.exists(existing_path):
                start_dir = os.path.dirname(existing_path)
        
        # Method 2: Try to find workflow folder relative to this script
        if not start_dir:
            try:
                # Get the directory of the current file (time_series_gui.py)
                # Should be: cedalion-pipeline/workflow/scripts/homer/
                current_file_dir = os.path.dirname(os.path.abspath(__file__))
                
                # Go up two levels to get to workflow folder
                # ../ -> scripts, ../ -> workflow
                workflow_dir = os.path.normpath(os.path.join(current_file_dir, '..', '..'))
                
                # Verify this is actually a workflow folder by checking for common files
                if os.path.exists(workflow_dir) and os.path.isdir(workflow_dir):
                    # Check if directory name is 'workflow' or contains a Snakefile
                    dir_name = os.path.basename(workflow_dir)
                    has_snakefile = any(
                        os.path.exists(os.path.join(workflow_dir, f)) 
                        for f in ['Snakefile', 'snakefile', 'Snakefile.smk']
                    )
                    
                    if dir_name.lower() == 'workflow' or has_snakefile:
                        start_dir = workflow_dir
                        print(f"Auto-detected workflow directory: {start_dir}")
            except Exception as e:
                print(f"Could not auto-detect workflow directory: {e}")
        
        # Fallback to home directory if no smart start found
        if not start_dir:
            start_dir = os.path.expanduser("~")
        
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Snakefile",
            start_dir,
            "Snakefile (Snakefile);;All Files (*)"
        )
        if file_path:
            self.snakefile_edit.setText(file_path)
            self._auto_detect_config(file_path)
    
    def _browse_config(self):
        # Try to determine a smart starting directory
        start_dir = ""
        
        # Method 1: If we have a snakefile path, look for config folder there
        snakefile_path = self.snakefile_edit.text()
        if snakefile_path and os.path.exists(snakefile_path):
            snakefile_dir = os.path.dirname(snakefile_path)
            config_dir = os.path.join(snakefile_dir, 'config')
            if os.path.exists(config_dir) and os.path.isdir(config_dir):
                start_dir = config_dir
                print(f"Starting config browser in: {start_dir}")
        
        # Method 2: If we already have a config path, start in that directory
        if not start_dir and self.config_edit.text():
            existing_path = self.config_edit.text()
            if os.path.exists(existing_path):
                start_dir = os.path.dirname(existing_path)
        
        # Method 3: Try to find workflow/config folder relative to this script
        if not start_dir:
            try:
                current_file_dir = os.path.dirname(os.path.abspath(__file__))
                workflow_dir = os.path.normpath(os.path.join(current_file_dir, '..', '..'))
                config_dir = os.path.join(workflow_dir, 'config')
                
                if os.path.exists(config_dir) and os.path.isdir(config_dir):
                    start_dir = config_dir
                    print(f"Auto-detected config directory: {start_dir}")
            except Exception as e:
                print(f"Could not auto-detect config directory: {e}")
        
        # Fallback to home directory if no smart start found
        if not start_dir:
            start_dir = os.path.expanduser("~")
        
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Config File",
            start_dir,
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if file_path:
            self.config_edit.setText(file_path)
    
    def _auto_detect_config(self, snakefile_path):
        """Auto-detect config file based on Snakefile location"""
        snakefile_dir = os.path.dirname(snakefile_path)
        snakefile_name = os.path.basename(snakefile_path)
        
        # Look for config/<snakefile_name>.yaml
        config_dir = os.path.join(snakefile_dir, 'config')
        if os.path.exists(config_dir):
            config_file = os.path.join(config_dir, f"{snakefile_name}.yaml")
            if os.path.exists(config_file):
                self.config_edit.setText(config_file)
                return
        
        # Clear if nothing found
        self.config_edit.clear()
    
    def _validate_and_accept(self):
        self.snakefile_path = self.snakefile_edit.text().strip()
        self.config_path = self.config_edit.text().strip()
        
        # Auto-detect config if not specified
        if not self.config_path and self.snakefile_path:
            snakefile_dir = os.path.dirname(self.snakefile_path)
            snakefile_name = os.path.basename(self.snakefile_path)
            config_file = os.path.join(snakefile_dir, 'config', f"{snakefile_name}.yaml")
            if os.path.exists(config_file):
                self.config_path = config_file
        
        if not self.snakefile_path:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing Snakefile",
                "Please specify the path to the Snakefile."
            )
            return
        
        if not os.path.exists(self.snakefile_path):
            QtWidgets.QMessageBox.warning(
                self,
                "File Not Found",
                f"Snakefile not found at:\n{self.snakefile_path}"
            )
            return
        
        if not self.config_path:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing Config",
                "Could not auto-detect config file. Please specify it manually."
            )
            return
        
        if not os.path.exists(self.config_path):
            QtWidgets.QMessageBox.warning(
                self,
                "File Not Found",
                f"Config file not found at:\n{self.config_path}"
            )
            return
        
        self.accept()


class SnakemakeRunDialog(QtWidgets.QDialog):
    """Dialog for running Snakemake pipeline (simplified - assumes setup already done)"""
    
    def __init__(self, parent=None, current_subject=None, current_run=None):
        super().__init__(parent)
        self.current_subject = current_subject
        self.current_run = current_run
        
        self.setWindowTitle("Run Snakemake Pipeline")
        self.setMinimumWidth(500)
        
        self.init_ui()
    
    def _get_snakefile_rules(self, snakefile_path):
        """Extract available rules from Snakefile"""
        if not snakefile_path or not os.path.exists(snakefile_path):
            return []
        
        rules = []
        try:
            with open(snakefile_path, 'r') as f:
                content = f.read()
            
            # Find all rule definitions using regex
            import re
            # Match: rule rule_name:
            rule_matches = re.finditer(r'^rule\s+(\w+)\s*:', content, re.MULTILINE)
            for match in rule_matches:
                rule_name = match.group(1)
                if rule_name not in rules:
                    rules.append(rule_name)
            
            print(f"Found {len(rules)} rules in Snakefile: {rules}")
        except Exception as e:
            print(f"Error parsing Snakefile: {e}")
        
        return rules
    
    def _get_conda_environments(self):
        """Get list of available conda environments."""
        import subprocess
        import shutil
        environments = []
        error_msg = None
        
        try:
            conda_cmd = _resolve_conda_command()
            print(f"DEBUG: Resolved conda command: {conda_cmd}")
        except FileNotFoundError as e:
            error_msg = str(e)
            print(f"ERROR: {error_msg}")
            return ["cedalion_snakemake", "cedalion_snakemake_dev"]
        
        try:
            result = subprocess.run(
                [conda_cmd, 'env', 'list'],
                capture_output=True,
                text=True,
                timeout=10,
                shell=False
            )
            
            if result.returncode == 0:
                # Parse output to extract environment names
                for line in result.stdout.splitlines():
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith('#'):
                        # Environment name is the first word
                        parts = line.split()
                        if parts:
                            env_name = parts[0]
                            # Skip base environment marker (*)
                            if env_name != '*':
                                environments.append(env_name)
            else:
                error_msg = f"conda env list failed with return code {result.returncode}"
                print(f"ERROR: {error_msg}")
                
        except subprocess.TimeoutExpired:
            error_msg = "conda env list command timed out after 10 seconds"
            print(f"ERROR: {error_msg}")
        except FileNotFoundError as e:
            error_msg = f"conda executable not found: {e}"
            print(f"ERROR: {error_msg}")
        except Exception as e:
            error_msg = f"Unexpected error getting conda environments: {e}"
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
        
        # If we got an error or no environments, return fallback
        if not environments:
            print(f"WARNING: Falling back to default environment list")
            if error_msg:
                print(f"Reason: {error_msg}")
            return ["cedalion_snakemake", "cedalion_snakemake_dev"]
        
        return environments
    
    def _get_current_conda_environment(self):
        """Get the currently active conda environment name."""
        # Check CONDA_DEFAULT_ENV environment variable
        current_env = os.environ.get('CONDA_DEFAULT_ENV', None)
        if current_env:
            return current_env
        
        # Fallback: try to detect from CONDA_PREFIX
        conda_prefix = os.environ.get('CONDA_PREFIX', None)
        if conda_prefix:
            # Extract environment name from path (last folder)
            env_name = os.path.basename(conda_prefix)
            return env_name
        
        return None

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        form_layout = QtWidgets.QFormLayout()
        
        # Run on current selection checkbox
        self.current_selection_checkbox = QtWidgets.QCheckBox("Run on current selection only")
        if self.current_subject and self.current_run:
            self.current_selection_checkbox.setEnabled(True)
            self.current_selection_checkbox.setToolTip(f"Run only for {self.current_subject}, {self.current_run}")
        else:
            self.current_selection_checkbox.setEnabled(False)
            self.current_selection_checkbox.setToolTip("No selection available")
        form_layout.addRow("", self.current_selection_checkbox)
        
        # Dry run checkbox
        self.dry_run_checkbox = QtWidgets.QCheckBox("Dry run (-n)")
        self.dry_run_checkbox.setChecked(True)
        self.dry_run_checkbox.setToolTip("Preview what will be run without executing")
        form_layout.addRow("", self.dry_run_checkbox)
        
        # Unlock checkbox
        self.unlock_checkbox = QtWidgets.QCheckBox("Unlock directory (--unlock)")
        self.unlock_checkbox.setChecked(False)
        self.unlock_checkbox.setToolTip("Remove locks from previous interrupted runs")
        form_layout.addRow("", self.unlock_checkbox)
        
        # Rerun incomplete checkbox
        self.rerun_incomplete_checkbox = QtWidgets.QCheckBox("Rerun incomplete files (--rerun-incomplete)")
        self.rerun_incomplete_checkbox.setChecked(False)
        self.rerun_incomplete_checkbox.setToolTip("Regenerate incomplete files from interrupted runs")
        form_layout.addRow("", self.rerun_incomplete_checkbox)
        
        # Summary checkbox
        self.summary_checkbox = QtWidgets.QCheckBox("Show summary (--summary)")
        self.summary_checkbox.setChecked(False)
        self.summary_checkbox.setToolTip("Show summary of all output files with their status (ok, missing, needs update)")
        form_layout.addRow("", self.summary_checkbox)
        
        # Target rule selector
        target_layout = QtWidgets.QHBoxLayout()
        self.target_combo = QtWidgets.QComboBox()
        
        # Get available rules from Snakefile if parent has the info
        available_rules = []
        if self.parent() and hasattr(self.parent(), 'snakefile_path'):
            available_rules = self._get_snakefile_rules(self.parent().snakefile_path)
        
        # Populate with available rules, or use defaults
        if available_rules:
            for rule in available_rules:
                self.target_combo.addItem(rule, rule)
        else:
            # Fallback to default options
            self.target_combo.addItem("all_default", "all_default")
            self.target_combo.addItem("all_groupaverage", "all_groupaverage")
        
        self.target_combo.setEditable(True)
        self.target_combo.setCurrentIndex(0)
        self.target_combo.setToolTip("Select which Snakemake target rule to run")
        target_layout.addWidget(self.target_combo)
        target_layout.addStretch()
        form_layout.addRow("Target rule:", target_layout)
        
        # Number of cores with "all" option - dynamically based on system
        cores_layout = QtWidgets.QHBoxLayout()
        self.cores_combo = QtWidgets.QComboBox()
        self.cores_combo.addItem("all")
        
        # Get available CPU count
        try:
            cpu_count = psutil.cpu_count(logical=True)
            # Generate core options: 1, 2, 4, 8, ... up to available cores
            core_options = [1, 2, 4, 8, 16, 32, 64, 128]
            for i in core_options:
                if i <= cpu_count:
                    self.cores_combo.addItem(str(i))
                else:
                    break
            # If CPU count doesn't match any of the powers of 2, add it
            if cpu_count not in core_options:
                self.cores_combo.addItem(str(cpu_count))
        except:
            # Fallback if psutil fails
            for i in [1, 2, 4, 8, 16]:
                self.cores_combo.addItem(str(i))
        
        self.cores_combo.setEditable(True)
        self.cores_combo.setCurrentText("1")
        cores_layout.addWidget(self.cores_combo)
        cores_layout.addStretch()
        form_layout.addRow("Cores:", cores_layout)
        
        # Conda environment selector
        env_layout = QtWidgets.QHBoxLayout()
        self.env_combo = QtWidgets.QComboBox()
        
        # Get available conda environments
        conda_envs = self._get_conda_environments()
        for env in conda_envs:
            self.env_combo.addItem(env)
        
        self.env_combo.setEditable(True)
        
        # Set default to currently active environment, otherwise cedalion_snakemake, otherwise first in list
        current_env = self._get_current_conda_environment()
        if current_env and current_env in conda_envs:
            self.env_combo.setCurrentText(current_env)
        elif "cedalion_snakemake" in conda_envs:
            self.env_combo.setCurrentText("cedalion_snakemake")
        else:
            self.env_combo.setCurrentIndex(0)
        
        self.env_combo.setToolTip("Select conda environment to use for running Snakemake")
        env_layout.addWidget(self.env_combo)
        env_layout.addStretch()
        form_layout.addRow("Conda Environment:", env_layout)
        
        layout.addLayout(form_layout)
        
        # Add buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)


class SummaryWorker(QtCore.QThread):
    """Background worker thread for running snakemake --summary command"""
    summary_completed = QtCore.Signal(dict)  # Emits file_status_map
    summary_failed = QtCore.Signal(str)  # Emits error message
    
    def __init__(self, snakefile_path, config_path, conda_env=None, target_rule='all_default', workdir=None, output_base_dir=None, config_args=None):
        super().__init__()
        self.snakefile_path = snakefile_path
        self.config_path = config_path
        self.conda_env = conda_env
        self.target_rule = target_rule
        self.workdir = workdir
        self.output_base_dir = output_base_dir
        self.config_args = config_args or []
        self._is_canceled = False
    
    def cancel(self):
        """Cancel the worker thread"""
        self._is_canceled = True
    
    def run(self):
        """Run the summary command in background thread"""
        if self._is_canceled:
            return
            
        try:
            file_status_map = self._run_snakemake_summary_impl()
            if not self._is_canceled:
                self.summary_completed.emit(file_status_map)
        except Exception as e:
            if not self._is_canceled:
                self.summary_failed.emit(str(e))
    
    def _run_snakemake_summary_impl(self):
        """Implementation of snakemake summary command (runs in background thread)"""
        try:
            # Build command with conda activation if environment is set
            if self.conda_env:
                cmd = _build_snakemake_command(
                     ['snakemake', '-s', self.snakefile_path,
                      '--configfile', self.config_path, *self.config_args,
                      '--nolock', '--summary', self.target_rule],
                    self.conda_env
                )
            else:
                cmd = ['snakemake', '-s', self.snakefile_path,
                       '--configfile', self.config_path, *self.config_args,
                       '--nolock', '--summary', self.target_rule]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.workdir or os.path.dirname(self.snakefile_path)
            )
            
            if self._is_canceled:
                return {}
            
            if result.returncode != 0:
                print(f"WARNING: snakemake --summary returned error code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                return {}
            
            # Parse the summary output (tab-delimited table)
            file_status_map = {}
            lines = result.stdout.strip().split('\n')
            
            if len(lines) < 2:
                print("WARNING: Summary output has fewer than 2 lines")
                return {}
            
            # Skip header line
            for line in lines[1:]:
                if self._is_canceled:
                    return {}
                    
                if not line.strip():
                    continue
                
                # Split by tabs to handle multi-word values
                parts = line.split('\t')
                if len(parts) < 6:
                    continue
                
                # Extract: output file, date, rule, log-file(s), status, plan
                file_path = parts[0].strip()
                status = parts[4].strip()
                plan = parts[5].strip()
                
                # Normalize path for consistent comparison
                file_path = os.path.normpath(file_path)
                if self.output_base_dir and not os.path.isabs(file_path):
                    file_path = os.path.normpath(os.path.join(self.output_base_dir, file_path))
                
                file_status_map[file_path] = {
                    'status': status,
                    'plan': plan
                }
            
            return file_status_map
            
        except Exception as e:
            print(f"Error running snakemake summary: {str(e)}")
            return {}


class ImageReconDialog(QtWidgets.QDialog):
    """Dialog for selecting image reconstruction options"""
    def __init__(self, trial_types, is_group_avg=False, time_bounds=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Reconstruction Options")
        self.setModal(False)  # Non-modal so it doesn't block and can stay open
        self.setMinimumWidth(450)
        self.setMinimumHeight(600)
        self.launch_callback = None  # Will be set by parent
        self.widgets_to_protect = []  # Track widgets that need scroll protection
        self.is_group_avg = is_group_avg  # Track if this is group average data
        self.time_bounds = time_bounds if time_bounds else (-100, 100)  # (min, max)
        
        main_layout = QtWidgets.QVBoxLayout()
        
        # Create scroll area for all options
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        scroll_widget.setLayout(layout)
        scroll.setWidget(scroll_widget)
        
        # Trial Type Selection
        trial_type_group = QtWidgets.QGroupBox("Trial Type")
        trial_type_layout = QtWidgets.QVBoxLayout()
        self.trial_type_combo = QtWidgets.QComboBox()
        self.trial_type_combo.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.trial_type_combo.installEventFilter(self)  # Block scroll when not focused
        self.widgets_to_protect.append(self.trial_type_combo)
        self.trial_type_combo.addItems(trial_types)
        trial_type_layout.addWidget(self.trial_type_combo)
        trial_type_group.setLayout(trial_type_layout)
        layout.addWidget(trial_type_group)
        
        # View Type Selection (includes both chromophore and surface type)
        view_type_group = QtWidgets.QGroupBox("View Type")
        view_type_layout = QtWidgets.QVBoxLayout()
        self.view_type_combo = QtWidgets.QComboBox()
        self.view_type_combo.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.view_type_combo.installEventFilter(self)  # Block scroll when not focused
        self.widgets_to_protect.append(self.view_type_combo)
        self.view_type_combo.addItems([
            "hbo_brain", "hbr_brain", 
            "hbo_scalp", "hbr_scalp"
        ])
        view_type_layout.addWidget(self.view_type_combo)
        view_type_group.setLayout(view_type_layout)
        layout.addWidget(view_type_group)
        
        # Metric Selection
        metric_group = QtWidgets.QGroupBox("Metric")
        metric_layout = QtWidgets.QVBoxLayout()
        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.metric_combo.installEventFilter(self)  # Block scroll when not focused
        self.widgets_to_protect.append(self.metric_combo)
        
        # Populate metric options based on group average flag
        if self.is_group_avg:
            self.metric_combo.addItems([
                "mag",
                "std_err",
                "t_stat",
                "std_err_btwn_subjs",
                "std_err_within_subjs"
            ])
        else:
            self.metric_combo.addItems([
                "mag",
                "std_err",
                "t_stat"
            ])
        
        metric_layout.addWidget(self.metric_combo)
        metric_group.setLayout(metric_layout)
        layout.addWidget(metric_group)
        
        # View Mode and Position Selection
        view_mode_group = QtWidgets.QGroupBox("View Mode")
        view_mode_layout = QtWidgets.QVBoxLayout()
        
        self.multi_view = QtWidgets.QCheckBox("Multi-view (6 views)")
        self.multi_view.setChecked(True)
        self.multi_view.stateChanged.connect(self._toggle_view_position)
        view_mode_layout.addWidget(self.multi_view)
        
        view_pos_layout = QtWidgets.QHBoxLayout()
        view_pos_layout.addWidget(QtWidgets.QLabel("Single View Position:"))
        self.view_position_combo = QtWidgets.QComboBox()
        self.view_position_combo.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.view_position_combo.installEventFilter(self)  # Block scroll when not focused
        self.widgets_to_protect.append(self.view_position_combo)
        self.view_position_combo.addItems([
            "superior", "left", "right", 
            "anterior", "posterior", "inferior"
        ])
        self.view_position_combo.setEnabled(False)
        view_pos_layout.addWidget(self.view_position_combo)
        view_mode_layout.addLayout(view_pos_layout)
        
        view_mode_group.setLayout(view_mode_layout)
        layout.addWidget(view_mode_group)
        
        # Colormap Selection
        cmap_group = QtWidgets.QGroupBox("Colormap")
        cmap_layout = QtWidgets.QVBoxLayout()
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.cmap_combo.installEventFilter(self)  # Block scroll when not focused
        self.widgets_to_protect.append(self.cmap_combo)
        self.cmap_combo.addItems(["seismic", "RdBu_r", "viridis", "coolwarm", "jet"])
        cmap_layout.addWidget(self.cmap_combo)
        cmap_group.setLayout(cmap_layout)
        layout.addWidget(cmap_group)
        
        # Color Limits (clim)
        clim_group = QtWidgets.QGroupBox("Color Limits")
        clim_layout = QtWidgets.QGridLayout()
        
        self.auto_clim = QtWidgets.QCheckBox("Auto (99th percentile)")
        self.auto_clim.setChecked(True)
        self.auto_clim.stateChanged.connect(self._toggle_clim_inputs)
        clim_layout.addWidget(self.auto_clim, 0, 0, 1, 2)
        
        clim_layout.addWidget(QtWidgets.QLabel("Min:"), 1, 0)
        self.clim_min = QtWidgets.QDoubleSpinBox()
        self.clim_min.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.clim_min.installEventFilter(self)  # Block scroll when not focused
        self.widgets_to_protect.append(self.clim_min)
        self.clim_min.setRange(-1e6, 1e6)
        self.clim_min.setValue(-1e-5)
        self.clim_min.setDecimals(8)
        self.clim_min.setSingleStep(1e-6)
        self.clim_min.setEnabled(False)
        clim_layout.addWidget(self.clim_min, 1, 1)
        
        clim_layout.addWidget(QtWidgets.QLabel("Max:"), 2, 0)
        self.clim_max = QtWidgets.QDoubleSpinBox()
        self.clim_max.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.clim_max.installEventFilter(self)  # Block scroll when not focused
        self.widgets_to_protect.append(self.clim_max)
        self.clim_max.setRange(-1e6, 1e6)
        self.clim_max.setValue(1e-5)
        self.clim_max.setDecimals(8)
        self.clim_max.setSingleStep(1e-6)
        self.clim_max.setEnabled(False)
        clim_layout.addWidget(self.clim_max, 2, 1)
        
        clim_group.setLayout(clim_layout)
        layout.addWidget(clim_group)
        
        # Time Range Selection
        time_range_group = QtWidgets.QGroupBox("Time Range (seconds)")
        time_range_layout = QtWidgets.QGridLayout()
        
        # Display available time bounds
        min_time, max_time = self.time_bounds
        bounds_label = QtWidgets.QLabel(f"Available: {min_time:.1f} to {max_time:.1f}s")
        muted_color = "#A8A8A8" if self.palette().color(QtGui.QPalette.Window).lightness() < 128 else "#666666"
        bounds_label.setStyleSheet(f"color: {muted_color}; font-style: italic;")
        time_range_layout.addWidget(bounds_label, 0, 0, 1, 2)
        
        time_range_layout.addWidget(QtWidgets.QLabel("Start:"), 1, 0)
        self.time_start = QtWidgets.QDoubleSpinBox()
        self.time_start.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.time_start.installEventFilter(self)  # Block scroll when not focused
        self.widgets_to_protect.append(self.time_start)
        self.time_start.setRange(min_time, max_time)
        self.time_start.setValue(max(min_time, -2))  # Default to -2 or min_time if larger
        self.time_start.setSingleStep(0.5)
        self.time_start.valueChanged.connect(self._validate_time_range)
        time_range_layout.addWidget(self.time_start, 1, 1)
        
        time_range_layout.addWidget(QtWidgets.QLabel("End:"), 2, 0)
        self.time_end = QtWidgets.QDoubleSpinBox()
        self.time_end.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.time_end.installEventFilter(self)  # Block scroll when not focused
        self.widgets_to_protect.append(self.time_end)
        self.time_end.setRange(min_time, max_time)
        self.time_end.setValue(min(max_time, 35))  # Default to 35 or max_time if smaller
        self.time_end.setSingleStep(0.5)
        self.time_end.valueChanged.connect(self._validate_time_range)
        time_range_layout.addWidget(self.time_end, 2, 1)
        
        time_range_layout.addWidget(QtWidgets.QLabel("Step:"), 3, 0)
        self.time_step = QtWidgets.QDoubleSpinBox()
        self.time_step.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.time_step.installEventFilter(self)  # Block scroll when not focused
        self.widgets_to_protect.append(self.time_step)
        self.time_step.setRange(0.1, 10)
        self.time_step.setValue(0.5)
        self.time_step.setSingleStep(0.1)
        time_range_layout.addWidget(self.time_step, 3, 1)
        
        self.mean_over_time = QtWidgets.QCheckBox("Mean over time range")
        self.mean_over_time.setChecked(False)
        self.mean_over_time.stateChanged.connect(self._toggle_mean_time_range)
        time_range_layout.addWidget(self.mean_over_time, 4, 0, 1, 2)
        
        time_range_group.setLayout(time_range_layout)
        layout.addWidget(time_range_group)
        
        # FPS for animation
        fps_group = QtWidgets.QGroupBox("Animation")
        fps_layout = QtWidgets.QHBoxLayout()
        fps_layout.addWidget(QtWidgets.QLabel("FPS:"))
        self.fps_spin = QtWidgets.QSpinBox()
        self.fps_spin.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.fps_spin.installEventFilter(self)  # Block scroll when not focused
        self.widgets_to_protect.append(self.fps_spin)
        self.fps_spin.setRange(1, 30)
        self.fps_spin.setValue(6)
        fps_layout.addWidget(self.fps_spin)
        fps_group.setLayout(fps_layout)
        layout.addWidget(fps_group)
        
        # Window Size
        wdw_group = QtWidgets.QGroupBox("Window Size (pixels)")
        wdw_layout = QtWidgets.QGridLayout()
        
        wdw_layout.addWidget(QtWidgets.QLabel("Width:"), 0, 0)
        self.wdw_width = QtWidgets.QSpinBox()
        self.wdw_width.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.wdw_width.installEventFilter(self)  # Block scroll when not focused
        self.widgets_to_protect.append(self.wdw_width)
        self.wdw_width.setRange(400, 3840)
        self.wdw_width.setValue(1024)
        self.wdw_width.setSingleStep(100)
        wdw_layout.addWidget(self.wdw_width, 0, 1)
        
        wdw_layout.addWidget(QtWidgets.QLabel("Height:"), 1, 0)
        self.wdw_height = QtWidgets.QSpinBox()
        self.wdw_height.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.wdw_height.installEventFilter(self)  # Block scroll when not focused
        self.widgets_to_protect.append(self.wdw_height)
        self.wdw_height.setRange(300, 2160)
        self.wdw_height.setValue(768)
        self.wdw_height.setSingleStep(100)
        wdw_layout.addWidget(self.wdw_height, 1, 1)
        
        wdw_group.setLayout(wdw_layout)
        layout.addWidget(wdw_group)
        
        # Display Options
        display_group = QtWidgets.QGroupBox("Display Options")
        display_layout = QtWidgets.QVBoxLayout()
        
        self.show_geo3d = QtWidgets.QCheckBox("Show labeled points (if available)")
        self.show_geo3d.setChecked(False)
        display_layout.addWidget(self.show_geo3d)
        
        self.custom_title = QtWidgets.QCheckBox("Custom title")
        self.custom_title.setChecked(False)
        self.custom_title.stateChanged.connect(self._toggle_title_input)
        display_layout.addWidget(self.custom_title)
        
        self.title_input = QtWidgets.QLineEdit()
        self.title_input.setPlaceholderText("Custom plot title")
        self.title_input.setEnabled(False)
        display_layout.addWidget(self.title_input)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # Save option
        save_group = QtWidgets.QGroupBox("Output")
        save_layout = QtWidgets.QVBoxLayout()
        self.save_checkbox = QtWidgets.QCheckBox("Save output (PNG/GIF)")
        self.save_checkbox.setChecked(False)
        self.save_checkbox.stateChanged.connect(self._toggle_filename_input)
        save_layout.addWidget(self.save_checkbox)
        
        self.filename_input = QtWidgets.QLineEdit()
        self.filename_input.setPlaceholderText("Filename (without extension)")
        self.filename_input.setEnabled(False)
        save_layout.addWidget(self.filename_input)
        
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll)
        
        # Dialog buttons (outside scroll area)
        button_layout = QtWidgets.QHBoxLayout()
        self.ok_button = QtWidgets.QPushButton("Launch Viewer")
        self.ok_button.clicked.connect(self._on_launch_clicked)
        self.cancel_button = QtWidgets.QPushButton("Close")
        self.cancel_button.clicked.connect(self.close)
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        
        # Initialize control states
        self._toggle_clim_inputs()
        self._toggle_title_input()
        self._toggle_view_position()
        self._toggle_mean_time_range()
    
    def _toggle_filename_input(self):
        """Enable/disable filename input based on save checkbox"""
        self.filename_input.setEnabled(self.save_checkbox.isChecked())
    
    def _toggle_clim_inputs(self):
        """Enable/disable color limit inputs based on auto checkbox"""
        enabled = not self.auto_clim.isChecked()
        self.clim_min.setEnabled(enabled)
        self.clim_max.setEnabled(enabled)
    
    def eventFilter(self, obj, event):
        """Filter wheel events on widgets that don't have focus"""
        if event.type() == QtCore.QEvent.Wheel:
            if obj in self.widgets_to_protect and not obj.hasFocus():
                return True  # Block the wheel event
        return super().eventFilter(obj, event)
    
    def _toggle_title_input(self):
        """Enable/disable title input based on custom title checkbox"""
        self.title_input.setEnabled(self.custom_title.isChecked())
    
    def _toggle_view_position(self):
        """Enable/disable view position based on multi-view checkbox"""
        self.view_position_combo.setEnabled(not self.multi_view.isChecked())
    
    def _toggle_mean_time_range(self):
        """Enable/disable step field based on mean over time checkbox"""
        # Step is disabled when mean over time is checked
        self.time_step.setEnabled(not self.mean_over_time.isChecked())
    
    def _validate_time_range(self):
        """Ensure start time is less than end time"""
        if self.time_start.value() >= self.time_end.value():
            # Auto-adjust end to be greater than start
            self.time_end.blockSignals(True)  # Prevent recursive signal
            self.time_end.setValue(self.time_start.value() + 0.5)
            self.time_end.blockSignals(False)
    
    def _on_launch_clicked(self):
        """Handle Launch Viewer button click - trigger callback without closing dialog"""
        if self.launch_callback:
            options = self.get_options()
            self.launch_callback(options)
    
    def get_options(self):
        """Return selected options as a dictionary"""
        options = {
            'trial_type': self.trial_type_combo.currentText(),
            'view_type': self.view_type_combo.currentText(),
            'metric': self.metric_combo.currentText(),
            'cmap': self.cmap_combo.currentText(),
            'fps': self.fps_spin.value(),
            'save': self.save_checkbox.isChecked(),
            'filename': self.filename_input.text() if self.save_checkbox.isChecked() else None,
            'wdw_size': (self.wdw_width.value(), self.wdw_height.value()),
            'show_geo3d': self.show_geo3d.isChecked(),
            'multi_view': self.multi_view.isChecked(),
            'view_position': self.view_position_combo.currentText()
        }
        
        # Time range - always included
        options['time_range'] = (
            self.time_start.value(),
            self.time_end.value(),
            self.time_step.value()
        )
        options['mean_over_time'] = self.mean_over_time.isChecked()
        
        # Color limits
        if not self.auto_clim.isChecked():
            options['clim'] = (self.clim_min.value(), self.clim_max.value())
        else:
            options['clim'] = None  # Will be auto-calculated
        
        # Custom title
        if self.custom_title.isChecked() and self.title_input.text():
            options['title_str'] = self.title_input.text()
        else:
            options['title_str'] = None  # Will be auto-generated
        
        return options


class _MAIN_GUI(QtWidgets.QMainWindow):
    def __init__(self, gui_data=None):
        # Initialize
        super().__init__()
        
        self.oftype = "pkl" # We'll assume the new format
        self.subjects = gui_data.get("subjects", [])
        self.subject_to_runs_map = gui_data.get("subject_to_runs_map", {})
        self.file_map = gui_data.get("file_map", {})
        self.path_to_data = gui_data.get("path_to_data")  # Store the selected derivatives/cedalion/XXX folder

        # Caching mechanism
        self.MAX_CACHE_SIZE = 10 # Max number of recordings to keep in memory
        self.MEMORY_THRESHOLD = 500 * 1024 * 1024 # 500 MB in bytes
        self.cache = OrderedDict()
        self.selected = [] # Initialize selected here
        self.snirfRec = None # Initialize here before UI setup
        
        # Snakemake pipeline state
        self.snakefile_path = None
        self.snakemake_config_path = None
        self.snakemake_config = None
        self.snakemake_menu = None  # Will be set during UI setup
        self.dynamic_menu_actions = []  # Track dynamically created menu items
        
        # Pipeline monitoring
        self.pipeline_monitor_timer = None
        self.pipeline_status = {}
        self.file_colors = {}  # Track colors for each (subject, run)
        self.snakemake_process = None  # Track the Snakemake process
        self.file_status_map = {}  # Map of file paths to their status from summary
        self.current_scope_files = {}  # Files in current config scope from summary
        self.all_scope_files = {}  # All possible files from full scope summary
        self.expected_pipeline_outputs = set()  # Files expected from summary
        self.completed_pipeline_outputs = set()  # Files that have been completed
        self.summary_worker = None  # Background worker for running summary command
        self.conda_env = None  # Track conda environment for pipeline execution
        
        # Track "Run on current selection only" mode
        self.run_current_only_mode = False  # Whether we're in current selection mode
        self.current_selection_subject = None  # Subject ID (e.g., "15")
        self.current_selection_task = None  # Task name (e.g., "IWHD")
        self.current_selection_run = None  # Run ID (e.g., "01")
        
        # Axis zoom preservation
        self.preserved_xlim = None  # Store x-axis limits
        self.preserved_ylim = None  # Store y-axis limits
        
        # GUI state restoration flag
        self._restoring_state = False  # Set to True while restoring saved state

        self._UI_SETUP()
        
        # Try to auto-load Snakemake configuration from homer.config
        self._auto_load_snakemake_config()
        
        # Restore pipeline state and update colors
        self._restore_pipeline_state()
        
        # Check if we have saved GUI state to restore
        has_saved_state = self._check_for_saved_gui_state()
        
        # Get the initial recording (only if no saved state to restore)
        if not has_saved_state and self.subjects:
            initial_subject = self.subjects[0]
            if initial_subject in self.file_map and self.subject_to_runs_map.get(initial_subject):
                 initial_run = self.subject_to_runs_map[initial_subject][0]
                 self._update_recording_data(initial_subject, initial_run, subject_changed=True)

        if self.snirfRec is None and not has_saved_state:
             print("Warning: Could not load an initial recording.")
        
        # Load saved GUI state (subject, run, selections, etc.)
        if has_saved_state:
            self._load_gui_state()
        
        # Check if this is a relaunch after pipeline switch
        self._check_pipeline_switch_state()
    
    def closeEvent(self, event):
        """Clean up resources when GUI is closed"""
        # Save current GUI state before closing
        self._save_gui_state()
        
        # Stop monitoring
        if self.pipeline_monitor_timer:
            self.pipeline_monitor_timer.stop()
        
        # Cancel and wait for summary worker to finish
        if self.summary_worker and self.summary_worker.isRunning():
            print("Closing GUI: Waiting for background worker to finish...")
            self.summary_worker.cancel()
            self.summary_worker.wait(5000)  # Wait up to 5 seconds
            if self.summary_worker.isRunning():
                print("WARNING: Background worker did not stop, forcing termination")
                self.summary_worker.terminate()
        
        # Accept the close event
        event.accept()

    def _load_events_from_tsv(self, snirf_path):
        """Load stimulus events from a BIDS-style *_events.tsv file.
        
        Args:
            snirf_path: Path to the SNIRF file. The events file is assumed to be
                        in the same directory with _nirs.snirf replaced by _events.tsv
        
        Returns:
            pd.DataFrame: DataFrame containing stimulus information with columns
                          [onset, duration, value, trial_type], or None if file not found
        """
        try:
            # Construct events.tsv path: replace _nirs.snirf with _events.tsv
            events_path = snirf_path.replace("_nirs.snirf", "_events.tsv")
            
            if not os.path.exists(events_path):
                print(f"No events file found at {events_path}")
                return None
            
            # Read TSV file
            df = pd.read_csv(events_path, sep='\t')
            
            # Ensure required columns exist
            if 'onset' not in df.columns or 'trial_type' not in df.columns:
                print(f"Events file {events_path} missing required columns (onset, trial_type)")
                return None
            
            # Add duration column if missing (default to 0)
            if 'duration' not in df.columns:
                df['duration'] = 0.0
            
            # Add value column if missing (default to 1.0)
            if 'value' not in df.columns:
                df['value'] = 1.0
            
            # Ensure column order matches expected format
            df = df[['onset', 'duration', 'value', 'trial_type']]
            
            print(f"Loaded {len(df)} events from {events_path}")
            return df
            
        except Exception as e:
            print(f"Failed to read events file: {e}")
            return None

    def _read_snirf_with_lock_fallback(self, snirf_path):
        """Read SNIRF, retrying through a temp copy if Windows/HDF5 locks block reads."""
        import cedalion.io as io
        import tempfile
        import time

        def _is_lock_error(exc):
            message = str(exc).lower()
            return (
                "unable to lock file" in message
                or "getlasterror() = 33" in message
                or "being used by another process" in message
                or "permission denied" in message
            )

        last_error = None
        for attempt in range(8):
            try:
                return io.read_snirf(snirf_path)
            except OSError as exc:
                if not _is_lock_error(exc):
                    raise
                last_error = exc

            wait_s = min(0.5 * (attempt + 1), 3.0)
            print(
                f"SNIRF read hit file lock; retrying from a temporary copy "
                f"(attempt {attempt + 1}/8): {snirf_path}"
            )
            temp_path = None
            try:
                suffix = os.path.splitext(snirf_path)[1] or ".snirf"
                fd, temp_path = tempfile.mkstemp(suffix=suffix)
                os.close(fd)
                shutil.copyfile(snirf_path, temp_path)
                return io.read_snirf(temp_path)
            except OSError as exc:
                if not _is_lock_error(exc):
                    raise
                last_error = exc
                time.sleep(wait_s)
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError as cleanup_error:
                        print(f"WARNING: Could not remove temporary SNIRF copy {temp_path}: {cleanup_error}")

        raise last_error

    def _prepare_data(self, rec_amp, rec_processed=None):
        """
        Performs all initial calculations on a recording.
        
        Args:
            rec_amp: Recording object with amplitude data from SNIRF
            rec_processed: Optional dict with processed data from PKL (od, conc, etc.)
        """
        prepared = {}
        prepared['snirfRec'] = rec_amp
        prepared['optodes_drawn'] = False

        # Get timeseries - prioritize amp from SNIRF
        if 'amp' in rec_amp.timeseries:
            prepared['snirfData'] = rec_amp.timeseries['amp']
        else:
            prepared['snirfData'] = rec_amp.timeseries[list(rec_amp.timeseries.keys())[0]]

        sPos = rec_amp.geo2d.sel(label=["S" in str(s.values) for s in rec_amp.geo2d.label])
        dPos = rec_amp.geo2d.sel(label=["D" in str(s.values) for s in rec_amp.geo2d.label])
        prepared['sPos'] = sPos
        prepared['dPos'] = dPos
        prepared['sPosVal'] = sPos.values
        prepared['dPosVal'] = dPos.values

        prepared['slabel'] = sPos.label.values
        prepared['dlabel'] = dPos.label.values
        prepared['opt_label'] = np.append(sPos.label.values, dPos.label.values)

        prepared['no_channels'] = len(prepared['snirfData'].channel)
        prepared['no_wvls'] = len(prepared['snirfData'].wavelength)

        prepared['channel_idx'] = np.arange(0, prepared['no_channels'])
        prepared['src_idx'] = [
            np.arange(0, len(sPos))[sPos.label == src][0]
            for src in prepared['snirfData'].source
        ]
        prepared['det_idx'] = [
            np.arange(0, len(dPos))[dPos.label == det][0]
            for det in prepared['snirfData'].detector
        ]

        prepared['sx'] = prepared['sPosVal'][:, 0]
        prepared['sy'] = prepared['sPosVal'][:, 1]
        prepared['dx'] = prepared['dPosVal'][:, 0]
        prepared['dy'] = prepared['dPosVal'][:, 1]

        prepared['sdx'] = np.append(prepared['sx'], prepared['dx'])
        prepared['sdy'] = np.append(prepared['sy'], prepared['dy'])

        prepared['src_label_handles'] = [0] * len(prepared['sx'])
        prepared['det_label_handles'] = [0] * len(prepared['dx'])
        
        # Get timeseries keys - combine from both sources if processed data exists
        
        if rec_processed and hasattr(rec_processed, 'timeseries'):
            # Merge timeseries from processed data (od, conc) with amp from SNIRF
            all_ts_keys = set(rec_amp.timeseries.keys())
            all_ts_keys.update(rec_processed.timeseries.keys())
            all_ts_keys.discard('amp')  # Remove amp from processed, use SNIRF version
            prepared['timeseries_keys'] = ['amp'] + sorted([k for k in all_ts_keys if k != 'amp'])
            
            # Store processed rec for later access
            prepared['processed_rec'] = rec_processed
        else:
            prepared['timeseries_keys'] = list(rec_amp.timeseries.keys())
            prepared['processed_rec'] = None
        
        prepared['aux_ts_keys'] = list(rec_amp.aux_ts.keys())
        prepared['has_stim'] = len(rec_amp.stim) > 0

        return prepared

    @staticmethod
    def _geo2d_is_valid(geo2d):
        """A snirf file's stored 2D probe positions are unusable for plotting
        when the source/detector positions (not landmarks) are missing, all
        zero/collapsed to one point, or contain NaN/inf. Only S/D-labeled
        rows are checked: some files carry a valid landmark layout (Nz, Cz,
        10-5 system points, ...) while every source/detector is [0, 0], which
        would otherwise look "valid" if the whole array were checked."""
        if geo2d is None or len(geo2d) == 0:
            return False
        labels = geo2d.label.values
        is_sd = np.array(["S" in str(label) or "D" in str(label) for label in labels])
        if not is_sd.any():
            return False
        try:
            values = geo2d.pint.dequantify().values
        except Exception:
            values = geo2d.values if hasattr(geo2d, "values") else np.asarray(geo2d)
        values = np.asarray(values, dtype=float)[is_sd]
        if values.size == 0 or not np.isfinite(values).all():
            return False
        if np.allclose(values, 0):
            return False
        unique_points = np.unique(values.round(6), axis=0)
        return len(unique_points) > 1

    def _ensure_valid_geo2d(self, rec):
        """If rec.geo2d is missing/degenerate, synthesize a 2D circular probe
        layout from rec.geo3d (mutates rec.geo2d in place) so the probe plot
        doesn't render garbage. Uses cedalion's own azimuthal cap-style
        projection (the same one cedalion.vis.anatomy.scalp_plot uses
        internally), which requires Nz/LPA/RPA landmarks in geo3d."""
        if self._geo2d_is_valid(getattr(rec, "geo2d", None)):
            return

        geo3d = getattr(rec, "geo3d", None)
        if geo3d is None or len(geo3d) == 0:
            print(
                "WARNING: 2D probe positions are missing/invalid and no 3D "
                "positions are available to generate a fallback layout."
            )
            return

        try:
            import cedalion.geometry.registration as registration
            rec.geo2d = registration.simple_scalp_projection(geo3d)
            print(
                "2D probe positions were missing or degenerate; generated a "
                "circular layout from the 3D optode positions instead."
            )
            self.statbar.showMessage(
                "2D probe positions were invalid - generated a circular layout "
                "from 3D positions",
                5000,
            )
        except Exception as e:
            print(f"WARNING: could not synthesize a 2D probe layout from geo3d: {e}")

    def _get_recording(self, subj_key, run_key):
        """
        Retrieves a prepared recording, using a cache.
        Now loads amp from SNIRF file and processed data from preprocessed SNIRF file.
        """
        cache_key = (subj_key, run_key)

        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            expected_preproc_path = self._get_preprocessing_file_path(subj_key, run_key)
            if (
                expected_preproc_path
                and os.path.exists(expected_preproc_path)
                and (
                    cached_data.get('processed_rec') is None
                    or cached_data.get('pkl_mtime') != os.path.getmtime(expected_preproc_path)
                )
            ):
                print(f"Cache entry for {subj_key} - {run_key} is stale or missing processed data; reloading from disk.")
                del self.cache[cache_key]

        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            print(f"Cache hit for {subj_key} - {run_key}")
            self.statbar.showMessage(f"📂 Loading preloaded data: {subj_key}/{run_key}", 2000)
            QtWidgets.QApplication.processEvents()  # Force GUI update to show message
            return self.cache[cache_key]

        print(f"Cache miss for {subj_key} - {run_key}. Loading from disk.")
        self.statbar.showMessage(f"💾 Loading data from disk: {subj_key}/{run_key}...", 0)  # 0 = permanent until changed
        QtWidgets.QApplication.processEvents()  # Force GUI update to show message
        
        available_memory = psutil.virtual_memory().available
        if available_memory < self.MEMORY_THRESHOLD:
            if self.cache:
                removed_key, _ = self.cache.popitem(last=False)
                print(f"Low memory warning. Removed {removed_key} from cache.")

        if len(self.cache) >= self.MAX_CACHE_SIZE:
            removed_key, _ = self.cache.popitem(last=False)
            print(f"Cache full. Removed {removed_key} from cache.")

        file_info = self.file_map.get(subj_key, {}).get(run_key)
        if not file_info:
            print(f"Error: No file info found for {subj_key} - {run_key}")
            return None
        
        snirf_path = file_info.get('snirf_path')
        pkl_path = file_info.get('pkl_path')

        if not pkl_path or not os.path.exists(pkl_path):
            expected_preproc_path = self._get_preprocessing_file_path(subj_key, run_key)
            if expected_preproc_path and os.path.exists(expected_preproc_path):
                print(f"DEBUG: Recovered processed path from config: {expected_preproc_path}")
                pkl_path = expected_preproc_path
                file_info['pkl_path'] = pkl_path
        
        
        if not snirf_path:
            print(f"Error: No SNIRF path for {subj_key} - {run_key}")
            return None
        
        print(f"Loading from SNIRF: {snirf_path}")
        
        # Load amplitude data from SNIRF file
        try:
            rec_amp = self._read_snirf_with_lock_fallback(snirf_path)[0]  # read_snirf returns a list, take first element
            print(f"Loaded amplitude data from SNIRF")
            self._ensure_valid_geo2d(rec_amp)

            # Try to load events from TSV file and overwrite stim data
            events_df = self._load_events_from_tsv(snirf_path)
            if events_df is not None:
                rec_amp.stim = events_df
                print(f"Overwritten stim data with events from TSV file")
            else:
                print(f"Using stim data from SNIRF file")
        except Exception as e:
            print(f"Error loading SNIRF file: {e}")
            return None
        
        # If preprocessed SNIRF file exists, load processed data
        rec_processed = None
        if pkl_path and os.path.exists(pkl_path):
            print(f"Loading processed data from SNIRF: {pkl_path}")
            try:
                rec_processed_list = self._read_snirf_with_lock_fallback(pkl_path)  # read_snirf returns a list
                rec_processed = rec_processed_list[0] if rec_processed_list else None
                print(f"Loaded processed data from SNIRF")
                
                # Try to load events from TSV file and overwrite stim data
                if rec_processed is not None:
                    events_df = self._load_events_from_tsv(snirf_path)  # Use raw snirf_path, not pkl_path
                    if events_df is not None:
                        rec_processed.stim = events_df
                        print(f"Overwritten processed stim data with events from TSV file")
            except Exception as e:
                print(f"Error loading preprocessed SNIRF file: {e}")
                rec_processed = None
        else:
            print(f"No processed data available for {subj_key} - {run_key}")
        
        # Prepare data using both sources
        prepared_data = self._prepare_data_from_snirf_and_pkl(rec_amp, rec_processed)
        
        # Store file paths for later reference
        prepared_data['snirf_path'] = snirf_path
        prepared_data['pkl_path'] = pkl_path
        prepared_data['pkl_mtime'] = os.path.getmtime(pkl_path) if pkl_path and os.path.exists(pkl_path) else None
        prepared_data['has_processed_data'] = rec_processed is not None
        
        # Try to load corresponding HRF data (only if processed data exists)
        hrf_data = None
        if pkl_path:
            try:
                # Construct HRF file path
                if '_preprocessed.snirf' in pkl_path:
                    # Normalize path separators for cross-platform compatibility
                    hrf_file_path = pkl_path.replace(os.path.join('Outputs', 'preprocessed_data'), os.path.join('Outputs', 'hrf_estimate'))
                    
                    # Remove '_run-<run-name>' pattern using regex
                    import re
                    hrf_file_path = re.sub(r'_run-[^_]+', '', hrf_file_path)
                    
                    hrf_file_path = hrf_file_path.replace('_preprocessed.snirf', '_hrf_estimate_conc.nc')
                    
                    if os.path.exists(hrf_file_path):
                        print(f"Loading HRF data from: {hrf_file_path}")
                        hrf_data = xr.open_dataset(hrf_file_path)
                        print(f"HRF data loaded successfully")
                        if hasattr(hrf_data, 'data_vars'):
                            print(f"HRF data variables: {list(hrf_data.data_vars.keys())}")
                    else:
                        print(f"No HRF file found at: {hrf_file_path}")
            except Exception as e:
                print(f"Error loading HRF data: {e}")
        
        prepared_data['hrf_data'] = hrf_data
        self.cache[cache_key] = prepared_data
        self.statbar.showMessage(f"✓ Data loaded from disk: {subj_key}/{run_key}", 0)  # Keep until changed
        QtWidgets.QApplication.processEvents()  # Force GUI update
        
        return prepared_data
    
    def _prepare_data_from_snirf_and_pkl(self, rec_amp, rec_processed):
        """
        Prepare data from SNIRF (amp) and preprocessed SNIRF (processed data).
        Returns a dict with all necessary data for the GUI.
        """
        # Both rec_amp and rec_processed are Recording objects
        # rec_amp has amplitude data, rec_processed has od/conc data
        # Just pass both to _prepare_data which will merge them
        return self._prepare_data(rec_amp, rec_processed)
    
    def _load_group_average_hrf(self):
        """Load group average HRF data from groupaverage folder"""
        print("=== _load_group_average_hrf called ===")
        try:
            # Get the current file path from the loaded recording data
            current_subj = self.subj.currentText()
            current_run = self.run.currentText()
            print(f"Current subject: {current_subj}, run: {current_run}")
            
            # Get from cache if available
            cache_key = (current_subj, current_run)
            if cache_key in self.cache:
                pkl_path = self.cache[cache_key].get('pkl_path')
                print(f"Found pkl path in cache: {pkl_path}")
            else:
                # Fallback to file_map
                file_info = self.file_map.get(current_subj, {}).get(current_run)
                pkl_path = file_info.get('pkl_path') if file_info else None
                print(f"PKL path from file_map: {pkl_path}")
            
            if not pkl_path:
                print("No valid PKL file path found - returning None")
                return None
            
            print(f"Using PKL file: {pkl_path}")
            
            # Construct group average HRF file path
            # groupaverage folder is at the same level as hrf_estimate or preprocessed_data folder
            import re
            
            # Get the base path (up to and including the working directory)
            # Could be either: .../derivatives/cedalion/new_inclQ_first_walk or similar
            outputs_hrf = os.path.join('Outputs', 'hrf_estimate')
            outputs_prep = os.path.join('Outputs', 'preprocessed_data')
            
            if outputs_hrf in pkl_path:
                # Currently looking at HRF file, extract base path
                base_path = pkl_path.split(outputs_hrf)[0]
            elif outputs_prep in pkl_path:
                # Currently looking at preprocessed file
                base_path = pkl_path.split(outputs_prep)[0]
            else:
                print("Could not determine base path - neither Outputs/hrf_estimate nor Outputs/preprocessed_data found")
                return None
            
            print(f"Base path: {base_path}")
            
            # Get the task name from the current file
            task_match = re.search(r'task-([^_/]+)', pkl_path)
            task_name = task_match.group(1) if task_match else 'unknown'
            print(f"Task name: {task_name}")
            
            # Try different naming patterns
            groupavg_dir = os.path.join(base_path, 'Outputs', 'group_results')
            possible_patterns = [
                f"task-{task_name}_nirs_groupaverage_chanspace_conc.nc",
                f"task-{task_name}_hrf_estimate_conc.nc",
                f"{task_name}_hrf_estimate_conc.nc"
            ]
            
            for pattern in possible_patterns:
                group_avg_path = os.path.join(groupavg_dir, pattern)
                if os.path.exists(group_avg_path):
                    print(f"Loading group average data from: {group_avg_path}")
                    
                    # Load netCDF file using xarray
                    data = xr.open_dataset(group_avg_path)
                    print(f"Loaded netCDF dataset with variables: {list(data.data_vars.keys())}")
                    
                    # Return the full xarray Dataset - it already contains all data variables
                    # (hrf_est/group_average, total_stderr, tstat, mse_t, etc.)
                    return data
            
            # If no file found, list what's available
            print(f"No group average file found. Tried patterns: {possible_patterns}")
            if os.path.exists(groupavg_dir):
                files = os.listdir(groupavg_dir)
                print(f"Available files in {groupavg_dir}:")
                for f in files:
                    print(f"  - {f}")
            else:
                print(f"Group average directory does not exist: {groupavg_dir}")
            
            return None
        except Exception as e:
            print(f"Error loading group average HRF: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def _UI_SETUP(self):
        # Set central widget
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)

        # Initialize layout
        window_layout = QtWidgets.QVBoxLayout(self._main)
        window_layout.setContentsMargins(10, 0, 10, 10)
        window_layout.setSpacing(10)

        # Set Minimum Size
        self.setMinimumSize(1000, 850)

        # Set Window Title
        self.setWindowTitle("Time Series")

        # Create Status Bar
        self.statbar = self.statusBar()
        self.statbar.setStyleSheet(self._status_bar_stylesheet())
        self.statbar.showMessage("Ready to Load SNIRF File!")

        # Filler plot for now
        self.plots = FigureCanvas(Figure(figsize=(30, 8)))
        self.plots.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.plots.setFocus()

        (self._dataTimeSeries_ax, self._optode_ax) = self.plots.figure.subplots(
            1, 2, width_ratios=[2, 1]
        )
        self._auxTimeSeries_ax = self._dataTimeSeries_ax.twinx()
        self._auxTimeSeries_ax.set_visible(False)
        self.plots.figure.tight_layout()
        self._optode_ax.axis('off')
        self._dataTimeSeries_ax.grid("True",axis="y")
        # Get the current position
        pos = self._dataTimeSeries_ax.get_position()
        # Adjust the left position
        new_pos = [pos.x0 + 0.075, pos.y0, pos.width - 0.075, pos.height]
        # Set the new position
        self._dataTimeSeries_ax.set_position(new_pos)
        self._dataTimeSeries_ax.clear()

        window_layout.addWidget(NavigationToolbar(self.plots,self),stretch=1)
        window_layout.addWidget(self.plots, stretch=8)

        # Connect Plots
        self.shift_pressed = False
        self.plots.mpl_connect("key_press_event", self._shift_is_pressed)
        self.plots.mpl_connect("key_release_event", self._shift_is_released)
        self.plots.mpl_connect("pick_event", self._optode_picked)
        
        # Connect axis limit change event for zoom preservation
        self._dataTimeSeries_ax.callbacks.connect('xlim_changed', self._on_xlims_change)
        self._dataTimeSeries_ax.callbacks.connect('ylim_changed', self._on_xlims_change)

        # Create Control Panel
        control_panel = QtWidgets.QGroupBox("Control Panel")
        control_panel_layout = QtWidgets.QHBoxLayout()
        control_panel_layout.setSpacing(20)
        control_panel.setLayout(control_panel_layout)
        window_layout.addWidget(control_panel, stretch=1)
        
        # Create File Control Layout
        file_layout = QtWidgets.QGridLayout()
        file_layout.setAlignment(QtCore.Qt.AlignTop)
        control_panel_layout.addLayout(
            file_layout,
        )
        
        ## Subject Selector
        self.subj = QtWidgets.QComboBox()
        if self.oftype == "rec":
            self.subj.addItems(["None"])
        else:
            self.subj.addItems(self.subjects)
        self.subj.setCurrentIndex(0)
        self.subj.setFixedWidth(200)
        self.subj.currentTextChanged.connect(self._subj_changed)
        file_layout.addWidget(QtWidgets.QLabel("Subject:"), 0, 0)
        file_layout.addWidget(self.subj, 0, 1)
        
        ## Run Selector
        self.run = QtWidgets.QComboBox()
        self._update_run_box() # Populate runs for the initial subject
        self.run.setFixedWidth(200)
        self.run.currentTextChanged.connect(self._run_changed)
        file_layout.addWidget(QtWidgets.QLabel("Run:"), 1, 0)
        file_layout.addWidget(self.run, 1, 1)

        # Create Timeseries Controls Layout
        ts_layout = QtWidgets.QGridLayout()
        ts_layout.setAlignment(QtCore.Qt.AlignTop)
        control_panel_layout.addLayout(
            ts_layout,
        )

        ## Create Timeseries Controls
        self.ts = QtWidgets.QListWidget()
        self.ts.addItems(["None"])
        self.ts.setCurrentRow(0)
        self.ts.setFixedHeight(90)
        self.ts.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.ts.currentTextChanged.connect(self._ts_changed)
        ts_layout.addWidget(QtWidgets.QLabel("Timeseries:"), 0, 0)
        ts_layout.addWidget(self.ts, 0, 1)
        
        ## Create HRF View Checkbox and chromophore selectors
        hrf_widget = QtWidgets.QWidget()
        hrf_main_layout = QtWidgets.QVBoxLayout()  # Changed to vertical layout
        hrf_main_layout.setContentsMargins(0, 0, 0, 0)
        hrf_main_layout.setSpacing(5)
        
        # First row: checkboxes
        hrf_checkbox_layout = QtWidgets.QHBoxLayout()
        hrf_checkbox_layout.setSpacing(10)
        
        self.hrf_view = QtWidgets.QCheckBox("View HRF")
        self.hrf_view.stateChanged.connect(self._toggle_hrf_view)
        self.hrf_view.setEnabled(False)  # Will be enabled when HRF data is available
        hrf_checkbox_layout.addWidget(self.hrf_view)
        
        # Note: HbO/HbR selection uses the wavelength/chromophore list widget below (wv)
        # No separate checkboxes needed for HRF view
        # Color coding is automatic: Preprocessing when HRF view off, HRF Estimate when HRF view on
        
        # Add separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.VLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        hrf_checkbox_layout.addWidget(separator)
        
        # Add Group Average checkbox (enabled all the time)
        self.hrf_group_avg = QtWidgets.QCheckBox("Group Average")
        self.hrf_group_avg.setChecked(False)
        self.hrf_group_avg.stateChanged.connect(self._hrf_group_avg_changed)
        self.hrf_group_avg.setEnabled(True)  # Always enabled
        hrf_checkbox_layout.addWidget(self.hrf_group_avg)
        hrf_checkbox_layout.addStretch()
        
        hrf_main_layout.addLayout(hrf_checkbox_layout)
        
        # Second row: buttons arranged horizontally
        hrf_button_layout = QtWidgets.QHBoxLayout()
        hrf_button_layout.setSpacing(10)
        
        # Add Launch Plot Probe button
        self.launch_plot_probe_btn = QtWidgets.QPushButton("Launch Plot Probe")
        self.launch_plot_probe_btn.clicked.connect(self._launch_plot_probe)
        self.launch_plot_probe_btn.setEnabled(False)  # Will be enabled when HRF view is active
        self.launch_plot_probe_btn.setMinimumWidth(150)  # Make button wider to show full text
        hrf_button_layout.addWidget(self.launch_plot_probe_btn)
        
        # Add Image Recon button (independent of HRF view)
        self.image_recon_btn = QtWidgets.QPushButton("Image Reconstruction")
        self.image_recon_btn.clicked.connect(self._launch_image_recon)
        self.image_recon_btn.setEnabled(False)  # Will be enabled when data is loaded
        self.image_recon_btn.setMinimumWidth(150)  # Make button wider to show full text
        hrf_button_layout.addWidget(self.image_recon_btn)

        # Add Brain Parcel Viewer button (needs an image recon result to map onto parcels)
        self.parcel_viewer_btn = QtWidgets.QPushButton("Brain Parcel Viewer")
        self.parcel_viewer_btn.clicked.connect(self._launch_parcel_viewer)
        self.parcel_viewer_btn.setEnabled(False)  # Will be enabled when data is loaded
        self.parcel_viewer_btn.setMinimumWidth(150)  # Make button wider to show full text
        hrf_button_layout.addWidget(self.parcel_viewer_btn)

        hrf_button_layout.addStretch()
        hrf_main_layout.addLayout(hrf_button_layout)
        
        hrf_widget.setLayout(hrf_main_layout)
        ts_layout.addWidget(hrf_widget, 1, 1)

        # Create Aux/wv selector Layout
        aux_layout = QtWidgets.QGridLayout()
        aux_layout.setAlignment(QtCore.Qt.AlignTop)
        control_panel_layout.addLayout(aux_layout)

        ## Aux Selector
        self.auxs = QtWidgets.QComboBox()
        self.auxs.addItems(["None"])
        self.auxs.setCurrentIndex(0)
        self.auxs.currentTextChanged.connect(self._aux_changed)
        aux_layout.addWidget(QtWidgets.QLabel("Aux:"), 0, 0)
        aux_layout.addWidget(self.auxs, 0, 1)

        ## Create Wavelength / Concentration Controls
        self.wv = QtWidgets.QListWidget()
        self.wv.setFixedHeight(45)
        self.wv.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.wv.itemSelectionChanged.connect(self._wv_changed)
        self.wv_label = QtWidgets.QLabel("Wavelength/Concentration:")
        aux_layout.addWidget(self.wv_label, 1, 0)
        aux_layout.addWidget(self.wv, 1, 1)

        # Create Optional Controls Layout
        opt_layout = QtWidgets.QVBoxLayout()
        opt_layout.setAlignment(QtCore.Qt.AlignTop)
        control_panel_layout.addLayout(
            opt_layout,
        )

        ## Create Opt2Circ Button
        self.opt2circ = QtWidgets.QCheckBox("View optodes as circles")
        self.opt2circ.stateChanged.connect(self._toggle_circles)
        opt_layout.addWidget(self.opt2circ)
        
        ## Create Preserve Axis Zoom Checkbox
        self.preserve_axis_zoom = QtWidgets.QCheckBox("Preserve Axis Zoom")
        self.preserve_axis_zoom.setChecked(False)
        self.preserve_axis_zoom.setToolTip("Keep current axis zoom when switching measurements, subjects, or wavelengths")
        self.preserve_axis_zoom.stateChanged.connect(self._preserve_zoom_changed)
        opt_layout.addWidget(self.preserve_axis_zoom)
        
        ## Create Auto Scale Y-axis Checkbox (sub-option of Preserve Axis Zoom)
        self.auto_scale_y = QtWidgets.QCheckBox("Auto scale Y-axis")
        self.auto_scale_y.setChecked(False)
        self.auto_scale_y.setEnabled(False)  # Disabled by default until preserve zoom is checked
        self.auto_scale_y.setToolTip("Preserve X-axis zoom but auto-scale Y-axis for better amplitude visibility")
        self.auto_scale_y.stateChanged.connect(self._save_gui_state)
        opt_layout.addWidget(self.auto_scale_y)

        ## Create Stimulus Selection Dropdown with checkboxes
        stim_label = QtWidgets.QLabel("Stimuli:")
        opt_layout.addWidget(stim_label)
        
        # Create a button that opens stimulus selection dialog
        self.stim_button = QtWidgets.QPushButton("Select Stimuli")
        self.stim_button.clicked.connect(self._open_stim_dialog)
        opt_layout.addWidget(self.stim_button)
        
        # Track selected stimuli and available stimulus types
        self.selected_stim_types = set()
        self.available_stim_types = []
        self.hrf_available_stim_types = []  # Track which stim types have HRF data

        ## Spacer
        control_panel_layout.addStretch()

        # Create button action for changing dataset
        change_dataset_btn = QAction("Change Dataset...", self)
        change_dataset_btn.setStatusTip("Switch to a different BIDS dataset/pipeline")
        change_dataset_btn.triggered.connect(self._change_dataset_dialog)

        ## Create menu
        # Use self.menuBar() for proper cross-platform menu bar support
        # On macOS this integrates with the native menu bar
        menu = self.menuBar()

        ## Populate menu

        file_menu = menu.addMenu("&File")
        file_menu.addAction(change_dataset_btn)
        
        # Create Snakemake menu with Setup first, then config items, then Run
        self.snakemake_menu = menu.addMenu("&Snakemake")
        
        setup_action = QAction("Setup Pipeline...", self)
        setup_action.setStatusTip("Select Snakefile and load configuration")
        setup_action.setMenuRole(QAction.MenuRole.NoRole)  # Prevent macOS from moving this to app menu
        setup_action.triggered.connect(self._snakemake_setup_pipeline)
        self.snakemake_menu.addAction(setup_action)
        
        self.snakemake_menu.addSeparator()
        
        dataset_action = QAction("Dataset", self)
        dataset_action.setStatusTip("Dataset configuration")
        dataset_action.triggered.connect(self._snakemake_dataset)
        self.snakemake_menu.addAction(dataset_action)
        
        # Dynamic config items will be inserted here after setup
        
        self.snakemake_menu.addSeparator()
        
        run_pipeline_action = QAction("Run Pipeline...", self)
        run_pipeline_action.setStatusTip("Run Snakemake pipeline")
        run_pipeline_action.triggered.connect(self._snakemake_run_pipeline)
        self.snakemake_menu.addAction(run_pipeline_action)

        # In case there is snirfRec
        if self.snirfRec is not None:
            self._init_widgets()

    def _change_dataset_dialog(self):
        """Open dialog to select a completely new BIDS dataset and pipeline"""
        try:
            # Step 1: Select BIDS root directory
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setWindowTitle("Change Dataset")
            msg.setText("Select a new BIDS dataset")
            msg.setInformativeText(
                "You will:\n"
                "1. Select a BIDS root directory (where sub-XXX folders are)\n"
                "2. Select or create a pipeline folder\n\n"
                "The GUI will restart with the new dataset."
            )
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            
            if msg.exec() != QtWidgets.QMessageBox.Ok:
                return
            
            # Get current directory as starting point
            current_dir = os.getcwd()
            
            # Open folder selection for BIDS root
            new_bids_root = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Step 1: Select BIDS Root Directory (where sub-XXX folders are)",
                current_dir,
                QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks
            )
            
            if not new_bids_root:
                return  # User cancelled
            
            print(f"Selected BIDS root: {new_bids_root}")
            
            # Step 2: Check/create derivatives/cedalion
            derivatives_path = os.path.join(new_bids_root, 'derivatives')
            cedalion_path = os.path.join(derivatives_path, 'cedalion')
            
            if not os.path.exists(cedalion_path):
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Create Derivatives Folder?",
                    f"derivatives/cedalion/ does not exist in the selected dataset.\n\n"
                    f"Create it now?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.Yes
                )
                if reply == QtWidgets.QMessageBox.Yes:
                    os.makedirs(cedalion_path, exist_ok=True)
                    print(f"Created: {cedalion_path}")
                else:
                    return  # User cancelled
            
            # Step 3: Select or create pipeline folder
            msg2 = QtWidgets.QMessageBox()
            msg2.setIcon(QtWidgets.QMessageBox.Information)
            msg2.setWindowTitle("Select Pipeline")
            msg2.setText("Step 2: Select or create a pipeline folder")
            msg2.setInformativeText(
                f"Select a folder inside:\n{cedalion_path}\n\n"
                "You can select an existing pipeline or create a new folder."
            )
            msg2.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            
            if msg2.exec() != QtWidgets.QMessageBox.Ok:
                return
            
            new_pipeline_path = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Step 2: Select or Create Pipeline Folder in derivatives/cedalion/",
                cedalion_path,
                QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks
            )
            
            if not new_pipeline_path:
                return  # User cancelled
            
            # Verify the selected folder is within derivatives/cedalion
            normalized_selected = os.path.normpath(new_pipeline_path)
            normalized_cedalion = os.path.normpath(cedalion_path)
            
            if not normalized_selected.startswith(normalized_cedalion):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid Selection",
                    f"Pipeline folder must be inside:\n{cedalion_path}\n\n"
                    f"You selected:\n{normalized_selected}"
                )
                return
            
            print(f"Selected pipeline: {new_pipeline_path}")
            
            # Check if it's the same as current dataset/pipeline
            if os.path.normpath(new_pipeline_path) == os.path.normpath(self.path_to_data):
                QtWidgets.QMessageBox.information(
                    self,
                    "Same Dataset/Pipeline",
                    "You selected the same dataset and pipeline that is currently loaded."
                )
                return
            
            # Check if pipeline folder is new
            is_new = not os.path.exists(os.path.join(new_pipeline_path, 'snakemake_config.yaml'))
            
            # Confirm the change
            pipeline_name = os.path.basename(new_pipeline_path)
            dataset_name = os.path.basename(new_bids_root)
            
            msg = f"Change to:\n"
            msg += f"  Dataset: {dataset_name}\n"
            msg += f"  Pipeline: {pipeline_name}\n\n"
            msg += "The GUI will restart and load data from this dataset.\n"
            if is_new:
                msg += "\n⚠️  This pipeline has no configuration yet.\n"
                msg += "You'll need to set it up after switching."
            msg += "\nContinue?"
            
            reply = QtWidgets.QMessageBox.question(
                self,
                'Confirm Dataset Change',
                msg,
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes
            )
            
            if reply == QtWidgets.QMessageBox.Yes:
                # Save switch state and relaunch
                self._switch_pipeline(new_pipeline_path, is_new)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                f"Error changing dataset:\n{str(e)}"
            )
            print(f"Error in _change_dataset_dialog: {e}")
            import traceback
            traceback.print_exc()
    
    def _open_dialog(self):
        # Grab the appropriate SNIRF file
        self._fname = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open File",
            "${HOME}",
            "SNIRF Files (*.snirf)",
        )[0]
        self.statbar.showMessage("Loading SNIRF File...")
        t0 = time.time()
        cread = cedalion.io.read_snirf(self._fname)
        self.snirfRec = cread[0]
        
        # Try to load events from TSV file and overwrite stim data
        events_df = self._load_events_from_tsv(self._fname)
        if events_df is not None:
            self.snirfRec.stim = events_df
            print(f"Overwritten stim data with events from TSV file")
        else:
            print(f"Using stim data from SNIRF file")
        
        t1 = time.time()
        self.statbar.showMessage(f"File Loaded in {t1 - t0:.2f} seconds!")
        self.auxs.setCurrentIndex(0)
        self.ts.setCurrentRow(0)
        self.aux_window.setText("0")
        self.selected_stim_types = set()  # Clear stimulus selection
        self._init_widgets()

    def _get_data_directory(self):
        """Extract the data directory from file_map paths"""
        print("\n" + "="*60)
        
        # Try to get a file path from file_map
        if not self.file_map:
            return os.getcwd()
        
        for subj, subj_dict in self.file_map.items():
            for run, file_info in subj_dict.items():
                # Extract pkl_path from the dict
                file_path = file_info.get('pkl_path') if isinstance(file_info, dict) else None
                if file_path:
                    # Extract the base path up to and including the main data directory
                    # Path structure: .../data_dir/derivatives/cedalion/...
                    # We want to get to the data_dir level
                    if 'derivatives' in file_path:
                        parts = file_path.split('derivatives')
                        # The first part contains the data directory path
                        data_dir = parts[0].rstrip(os.sep).rstrip('/')
                        print("="*60 + "\n")
                        return data_dir
        # Fallback to current directory if no paths found
        print("="*60 + "\n")
        return os.getcwd()
    
    def _detect_derivatives_subfolder(self):
        """Detect derivatives_subfolder from file_map paths"""
        # Try to extract derivatives subfolder from file paths
        if not self.file_map:
            return ""
        
        for subj, subj_dict in self.file_map.items():
            for run, file_info in subj_dict.items():
                # Extract pkl_path from the dict
                file_path = file_info.get('pkl_path') if isinstance(file_info, dict) else None
                if file_path and 'derivatives' in file_path:
                    # Extract the path between 'derivatives/' and the subject folder
                    parts = file_path.split('derivatives')
                    if len(parts) > 1:
                        # Get everything after derivatives/
                        after_derivatives = parts[1].lstrip(os.sep).lstrip('/')
                        # Split by path separator to get folder structure
                        path_parts = after_derivatives.split(os.sep)
                        # Keep folders until we hit Outputs, preprocessed_data, hrf_estimate, or sub- folders
                        # These indicate we're at the pipeline output level or subject level
                        pipeline_folders = ['Outputs', 'preprocessed_data', 'hrf_estimate', 'image_results', 
                                          'group_results', 'qa_reports']
                        subfolder_parts = []
                        for part in path_parts:
                            # Stop at subject folders or pipeline output folders
                            if part.startswith('sub-') or part in pipeline_folders or not part:
                                break
                            subfolder_parts.append(part)
                        
                        if subfolder_parts:
                            derivatives_subfolder = os.path.join(*subfolder_parts)
                            print(f"Detected derivatives_subfolder: {derivatives_subfolder}")
                            return derivatives_subfolder
        
        print("Could not detect derivatives_subfolder, using empty string")
        return ""
    
    def _snakemake_dataset(self):
        """Handle Dataset menu action"""
        if not self.snakemake_config_path:
            # Show minimal dialog with only root_dir and derivatives_subfolder
            data_dir = self._get_data_directory()
            
            # Try to find config ONLY in the selected path_to_data folder
            config_path = None
            if self.path_to_data and os.path.exists(self.path_to_data):
                # Check directly in the selected folder only
                test_path = os.path.join(self.path_to_data, 'snakemake_config.yaml')
                if os.path.exists(test_path):
                    config_path = test_path
                    print(f"Found config in selected folder: {config_path}")
                else:
                    # Config doesn't exist in selected folder, use path as default location
                    config_path = test_path
                    print(f"Config not found, will create at: {config_path}")
            else:
                # Fallback if path_to_data not set
                config_path = os.path.join(data_dir, 'derivatives', 'snakemake_config.yaml')
            
            # Only show root_dir and derivatives_subfolder if no config loaded
            if not self.snakemake_config:
                self._edit_config_minimal_dataset(config_path)
            else:
                self._edit_config_block(config_path, 'dataset', readonly_keys=['root_dir'])
        else:
            self._edit_config_block(self.snakemake_config_path, 'dataset', readonly_keys=['root_dir'])
        
    def _snakemake_config_item(self, block_name):
        """Handle dynamic config menu action"""
        if self.snakemake_config_path:
            self._edit_config_block(self.snakemake_config_path, block_name)
        else:
            QtWidgets.QMessageBox.information(
                self,
                "No Config Loaded",
                "Please use 'Setup Pipeline...' first to select a Snakefile and load the config."
            )
    
    def _snakemake_setup_pipeline(self):
        """Handle Setup Pipeline menu action - select Snakefile and load config"""
        dialog = SnakemakeSetupDialog(self, self.snakefile_path, self.snakemake_config_path)
        
        if dialog.exec():
            snakefile_path = dialog.snakefile_path
            config_path = dialog.config_path
            
            # Auto-detect root_dir and derivatives_subfolder from current data
            data_dir = self._get_data_directory()
            derivatives_subfolder = self._detect_derivatives_subfolder()
            
            # If detected subfolder is exactly "cedalion", set to empty since Snakefile adds it
            if derivatives_subfolder == 'cedalion':
                print(f"Auto-detected derivatives_subfolder is 'cedalion', setting to empty (Snakefile adds it)")
                derivatives_subfolder = ''
            
            print(f"Auto-detected root_dir: {data_dir}")
            print(f"Auto-detected derivatives_subfolder: {derivatives_subfolder}")
            
            # Load the source config
            with open(config_path, 'r') as f:
                temp_config = yaml.safe_load(f)
            
            # Use path_to_data directly as the target directory (user selected folder)
            if self.path_to_data and os.path.exists(self.path_to_data):
                target_dir = self.path_to_data
                print(f"Using selected path_to_data as target: {target_dir}")
            elif derivatives_subfolder:
                target_dir = os.path.join(data_dir, 'derivatives', derivatives_subfolder)
                print(f"Using detected derivatives_subfolder: {target_dir}")
            else:
                target_dir = os.path.join(data_dir, 'derivatives')
                print(f"Using default derivatives folder: {target_dir}")
            
            print(f"Target directory: {target_dir}")
            os.makedirs(target_dir, exist_ok=True)
            print(f"Created target directory (if it didn't exist)")
            
            # Target config path in target directory
            target_config_path = os.path.join(target_dir, 'snakemake_config.yaml')
            
            print(f"Source config path: {config_path}")
            print(f"Target config path: {target_config_path}")
            print(f"Paths are same: {os.path.abspath(config_path) == os.path.abspath(target_config_path)}")
            
            # Copy config file and update root_dir/derivatives_subfolder while preserving format
            if os.path.abspath(config_path) != os.path.abspath(target_config_path):
                try:
                    # Extract derivatives_subfolder from path_to_data
                    # e.g., path_to_data = "C:/data/derivatives/cedalion/new_inclQ_first_walk"
                    # derivatives_subfolder should be "new_inclQ_first_walk" (NOT "cedalion/new_inclQ_first_walk")
                    # because Snakefile adds "cedalion/" automatically
                    if self.path_to_data and 'derivatives' in self.path_to_data:
                        parts = self.path_to_data.split('derivatives')
                        if len(parts) > 1:
                            derivatives_subfolder = parts[1].lstrip(os.sep).lstrip('/')
                            # Remove 'cedalion' prefix since Snakefile adds it automatically
                            # Handle both "cedalion/..." and exactly "cedalion"
                            if derivatives_subfolder == 'cedalion':
                                derivatives_subfolder = ''  # Working directly in cedalion folder
                            elif derivatives_subfolder.startswith('cedalion/') or derivatives_subfolder.startswith('cedalion\\'):
                                derivatives_subfolder = derivatives_subfolder.split('cedalion', 1)[1].lstrip(os.sep).lstrip('/')
                            print(f"Extracted derivatives_subfolder from path_to_data: {derivatives_subfolder}")
                    
                    # Read original file as text to preserve formatting
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_text = f.read()
                    
                    # Convert paths to use forward slashes (works on Windows and avoids YAML escape issues)
                    data_dir_normalized = data_dir.replace('\\', '/')
                    derivatives_subfolder_normalized = derivatives_subfolder.replace('\\', '/') if derivatives_subfolder else ''
                    
                    # Replace root_dir value using regex (preserves comments and formatting)
                    config_text = re.sub(
                        r'(root_dir:\s*)(["\']?).*?\2(\s*(?:#.*)?$)',
                        rf'\1\2{data_dir_normalized}\2\3',
                        config_text,
                        flags=re.MULTILINE
                    )
                    
                    # Replace derivatives_subfolder value using regex
                    config_text = re.sub(
                        r'(derivatives_subfolder:\s*)(["\']?).*?\2(\s*(?:#.*)?$)',
                        rf'\1\2{derivatives_subfolder_normalized}\2\3',
                        config_text,
                        flags=re.MULTILINE
                    )
                    
                    # Write to target location
                    with open(target_config_path, 'w', encoding='utf-8') as f:
                        f.write(config_text)
                    
                    print(f"Copied config from {config_path} to {target_config_path}")
                    print(f"Updated config with root_dir={data_dir} and derivatives_subfolder={derivatives_subfolder}")
                    
                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Config Copy Failed",
                        f"Could not copy config to derivatives folder:\n{str(e)}\n\nUsing original location."
                    )
                    target_config_path = config_path
            
            # Save homer.config with Snakefile path in the same directory
            self._save_homer_config(snakefile_path, target_config_path, target_dir)
            
            # Load config and update menu
            self._load_snakemake_config(snakefile_path, target_config_path)
    
    def _resolve_target_rule(self, target_rule):
        """Map wildcard worker rules to concrete aggregate rules for Snakemake targets."""
        target_map = {
            'preprocess': 'all_preprocess',
            'hrf_estimation': 'all_hrf_estimation',
            'groupaverage': 'all_groupaverage',
            'imagerecon': 'all_imagerecon',
        }
        return target_map.get(target_rule, target_rule)

    def _get_windows_relative_run_context(self, config_path):
        """Return workdir/config args that keep Snakemake metadata paths short on Windows."""
        if sys.platform != 'win32' or not config_path:
            return None, []

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            dataset = config_data.get('dataset', {})
            dataset_root = dataset.get('root_dir') or config_data.get('root_dir')
            if not dataset_root:
                return None, []

            dataset_root = os.path.normpath(dataset_root)
            if dataset_root in ('.', ''):
                return None, []

            return dataset_root, ['--config', 'root_dir=.']
        except Exception as e:
            print(f"WARNING: Could not configure relative Snakemake run context: {e}")
            return None, []

    def _snakemake_run_pipeline(self):
        """Handle Run Pipeline menu action"""
        # Check if setup has been done
        if not self.snakefile_path or not self.snakemake_config_path:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Setup Required",
                "Pipeline has not been set up yet. Would you like to set it up now?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                self._snakemake_setup_pipeline()
                if not self.snakefile_path:  # User cancelled setup
                    return
            else:
                return
        
        # Get current subject and run selection
        current_subject = self.subj.currentText() if self.subj.currentText() != "None" else None
        current_run = self.run.currentText() if self.run.currentText() and self.run.currentText() != "" else None
        
        dialog = SnakemakeRunDialog(self, current_subject, current_run)
        if dialog.exec():
            dry_run = dialog.dry_run_checkbox.isChecked()
            show_summary = dialog.summary_checkbox.isChecked()
            cores_text = dialog.cores_combo.currentText()
            run_current_only = dialog.current_selection_checkbox.isChecked()
            conda_env = dialog.env_combo.currentText()
            # Store conda environment for future summary commands
            self.conda_env = conda_env
            
            # Use the already-loaded snakefile and config
            snakefile_path = self.snakefile_path
            config_path = self.snakemake_config_path
            original_config_path = self.snakemake_config_path  # Keep original for summary
            
            # Build the snakemake command
            cmd = ['snakemake', '-s', snakefile_path]
            
            # Handle config overrides for current selection if enabled
            if run_current_only and current_subject and current_run:
                # Parse subject and run from the selection
                # Format: "sub-01" or "sub-01_ses-01", "task-IWHD_run_run-01"
                # Use non-greedy matching to handle task names with underscores
                subject_match = re.search(r'sub-(\w+)', current_subject)
                # Match task name up until _run- (non-greedy)
                task_match = re.search(r'task-(.+?)(?:_run-|$)', current_run) if current_run else None
                # Match run number after run-
                run_match = re.search(r'run-(\w+)', current_run) if current_run else None
                
                # Store current selection for color logic
                self.run_current_only_mode = True
                self.current_selection_subject = subject_match.group(1) if subject_match else None
                self.current_selection_task = task_match.group(1) if task_match else None
                self.current_selection_run = run_match.group(1) if run_match else None
                
                print(f"  Selected: sub-{self.current_selection_subject}, task-{self.current_selection_task}, run-{self.current_selection_run}")
                
                # Create a temporary config file with overrides
                # This preserves nested config structure which command-line args cannot handle
                if config_path and os.path.exists(config_path):
                    try:
                        # Load config as dictionary
                        import yaml
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config_data = yaml.safe_load(f)
                        
                        
                        # Override task with single-item list
                        if task_match:
                            task_id = task_match.group(1)
                            config_data['dataset']['task'] = [task_id]
                        
                        # Override run with single-item list
                        if run_match:
                            run_id = run_match.group(1)
                            config_data['dataset']['run'] = [run_id]
                        
                        # Set run_list to match the single run (replaces num_runs approach)
                        if run_match:
                            config_data['dataset']['run_list'] = [run_id]
                        
                        # Set subjects_to_exclude to all subjects except the current one
                        if subject_match:
                            subject_id = subject_match.group(1)
                            root_dir = config_data.get('dataset', {}).get('root_dir', '')
                            
                            # Find all sub-* directories in root_dir
                            if root_dir and os.path.exists(root_dir):
                                all_subjects = []
                                for item in os.listdir(root_dir):
                                    if item.startswith('sub-') and os.path.isdir(os.path.join(root_dir, item)):
                                        # Extract subject ID (without "sub-" prefix)
                                        sub_id = item.replace('sub-', '')
                                        all_subjects.append(sub_id)
                                
                                # Exclude all subjects except current one
                                subjects_to_exclude = [s for s in all_subjects if s != subject_id]
                                config_data['dataset']['subjects_to_exclude'] = subjects_to_exclude
                        
                        # Write temporary config file
                        # Save in same directory as original config
                        config_dir = os.path.dirname(config_path)
                        temp_config_path = os.path.join(config_dir, 'snakemake_config_temp.yaml')
                        with open(temp_config_path, 'w', encoding='utf-8') as f:
                            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                        
                        # Store original config path for monitoring
                        original_config_path = config_path
                        config_path = temp_config_path  # Use temp config for execution
                        
                    except Exception as e:
                        print(f"ERROR: Config override failed: {e}")
                        import traceback
                        traceback.print_exc()
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Config Override Failed",
                            f"Could not create temporary config with overrides:\n{str(e)}\n\nProceeding without current selection filter."
                        )
                        self.run_current_only_mode = False
            else:
                # Not in current selection mode
                self.run_current_only_mode = False
                self.current_selection_subject = None
                self.current_selection_task = None
                self.current_selection_run = None

            snakemake_workdir = None
            output_base_dir = None
            snakemake_config_args = []
            runtime_workdir, snakemake_config_args = self._get_windows_relative_run_context(config_path)
            if runtime_workdir:
                snakemake_workdir = runtime_workdir
                output_base_dir = runtime_workdir
            
            if config_path:
                cmd.extend(['--configfile', config_path])
            cmd.extend(snakemake_config_args)

            if dry_run:
                cmd.append('-n')
            
            if show_summary:
                cmd.append('--summary')
            
            if dialog.unlock_checkbox.isChecked():
                cmd.append('--unlock')
            
            if dialog.rerun_incomplete_checkbox.isChecked():
                cmd.append('--rerun-incomplete')
            
            cmd.extend(['-p', '--cores', cores_text])
            
            # Add target rule (default is all_default, which runs if not specified)
            target_rule = dialog.target_combo.currentData() or dialog.target_combo.currentText()
            target_rule = self._resolve_target_rule(target_rule)
            if target_rule and target_rule != 'all_default':
                cmd.append(target_rule)
            
            print(f"DEBUG: Snakemake command: {subprocess.list2cmdline(cmd)}")

            try:
                # First, unlock the directory in case of stale locks
                print("Unlocking workflow directory...")
                unlock_cmd = _build_snakemake_command(
                    ['snakemake', '-s', snakefile_path, '--configfile', config_path, *snakemake_config_args, '--unlock'],
                    conda_env
                )
                print(f"DEBUG: Unlock command: {subprocess.list2cmdline(unlock_cmd)}")
                unlock_result = subprocess.run(unlock_cmd, capture_output=True, text=True, timeout=30, cwd=snakemake_workdir)
                if unlock_result.returncode == 0:
                    print("Workflow directory unlocked successfully")
                else:
                    print(f"WARNING: Unlock command returned code {unlock_result.returncode}")
                    print(f"STDERR: {unlock_result.stderr[:200]}")

                # Now run summary to get status of all workflow files
                # When running "current selection only", use temp config for summary
                # This ensures summary only detects files for the current selection
                summary_config = config_path  # Use same config as execution (temp if current selection only)
                config_type = "temp (current selection)" if run_current_only else "full"
                print(f"Running summary for {config_type} config scope...")
                file_status_map = self._run_snakemake_summary(
                    snakefile_path,
                    summary_config,
                    target_rule,
                    workdir=snakemake_workdir,
                    output_base_dir=output_base_dir,
                    config_args=snakemake_config_args
                )

                if file_status_map:
                    print(f"Summary found {len(file_status_map)} workflow files in current scope")
                    # Store in current_scope_files for simplified color system
                    self.current_scope_files = file_status_map
                    self.file_status_map = file_status_map  # Backward compatibility
                    self.all_scope_files = {}  # No dual summary system

                    # Extract files that need processing
                    self.expected_pipeline_outputs = {
                        path for path, info in file_status_map.items()
                        if info['plan'] not in ['no', 'update']  # 'update pending' or 'create'
                    }
                    print(f"Files needing processing: {len(self.expected_pipeline_outputs)}")
                else:
                    print("WARNING: No workflow files detected from summary")
                    response = QtWidgets.QMessageBox.warning(
                        self,
                        "Summary Warning",
                        "Summary did not detect any workflow files.\n\n"
                        "This may indicate:\n"
                        "- Pipeline configuration issues\n"
                        "- Summary parsing failure\n\n"
                        "Do you want to continue anyway?",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                    )
                    if response == QtWidgets.QMessageBox.No:
                        return
                    self.current_scope_files = {}
                    self.file_status_map = {}
                    self.all_scope_files = {}
                    self.expected_pipeline_outputs = set()

                # Save pipeline status and update colors only for actual runs (not dry runs or summaries)
                if not dry_run and not show_summary:
                    # Store target_rule for monitoring and final summary
                    self.current_target_rule = target_rule

                    # Check if there are actually files to process
                    if self.expected_pipeline_outputs:
                        # Clear ALL cache ONLY if files need processing
                        # This ensures fresh data loads when files complete, but preserves cache if nothing to do
                        if self.cache:
                            self.cache.clear()
                            self.statbar.showMessage(f"Starting pipeline - {len(self.expected_pipeline_outputs)} files to process")
                    else:
                        self.statbar.showMessage("All files already up-to-date - nothing to process")

                    # Save pipeline status before starting
                    self._save_pipeline_status_on_start()

                    # Immediately update all colors to reflect new pipeline state
                    # Files in config will turn orange, files not in config stay/turn red
                    self._update_all_file_colors()

                # Run in a new console window on Windows
                if sys.platform == 'win32':
                    launch_cmd = _build_snakemake_command(cmd, conda_env)
                    print(f"DEBUG: Launch command: {subprocess.list2cmdline(launch_cmd)}")
                    console_command = subprocess.list2cmdline(launch_cmd)
                    subprocess.Popen(
                        [os.environ.get('COMSPEC', r'C:\Windows\System32\cmd.exe'),
                         '/k', console_command],
                        creationflags=subprocess.CREATE_NEW_CONSOLE,
                        cwd=snakemake_workdir,
                    )

                    # Only monitor actual pipeline runs, not dry runs or summaries
                    if not dry_run and not show_summary:
                        # Store the actual config path used (temp or original) for monitoring
                        self._actual_config_used = config_path
                        self._actual_snakemake_workdir = snakemake_workdir
                        self._actual_output_base_dir = output_base_dir
                        self._actual_snakemake_config_args = snakemake_config_args

                        # Wait for snakemake process to start with retries
                        import time
                        snakemake_pid = None
                        for attempt in range(5):  # Try 5 times
                            time.sleep(2)  # Wait 2 seconds between attempts (total 10 seconds)
                            snakemake_pid = self._find_snakemake_process_pid()
                            if snakemake_pid:
                                break

                        if snakemake_pid:
                            self._store_pipeline_process_info(snakemake_pid)
                        else:
                            print("WARNING: Could not find snakemake PID after 10 seconds")
                            print("WARNING: Pipeline started but monitoring may not work correctly")

                    self.statbar.showMessage("Snakemake pipeline started in a new terminal window.", 5000)
                else:
                    # For Unix-like systems
                    self.snakemake_process = subprocess.Popen(cmd)
                    if self.snakemake_process:
                        self._store_pipeline_process_info(self.snakemake_process.pid)

                    self.statbar.showMessage("Snakemake pipeline started.", 5000)

                # Start monitoring timer only for actual pipeline runs (not dry runs or summaries)
                if not dry_run and not show_summary:
                    self._start_pipeline_monitoring()

            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to run pipeline:\n{str(e)}"
                )
    
    def _save_homer_config(self, snakefile_path, config_path, derivatives_dir, pipeline_status=None):
        """Save homer.config file with Snakefile, config paths, and pipeline status"""
        try:
            homer_config_path = os.path.join(derivatives_dir, 'homer.config')
            
            # Load existing config to preserve other sections (like gui_state)
            homer_config = {}
            if os.path.exists(homer_config_path):
                try:
                    with open(homer_config_path, 'r') as f:
                        homer_config = yaml.safe_load(f) or {}
                except yaml.constructor.ConstructorError:
                    try:
                        with open(homer_config_path, 'r') as f:
                            homer_config = yaml.unsafe_load(f) or {}
                    except:
                        homer_config = {}
            
            # Update snakemake paths
            homer_config['snakefile_path'] = snakefile_path
            homer_config['config_path'] = config_path
            
            if pipeline_status:
                homer_config['pipeline_status'] = pipeline_status
            
            # Clean config to remove numpy objects before saving
            homer_config_clean = self._clean_config_for_yaml(homer_config)
            
            with open(homer_config_path, 'w') as f:
                yaml.dump(homer_config_clean, f, default_flow_style=False, sort_keys=False)
            print(f"Saved homer.config to {homer_config_path}")
        except Exception as e:
            print(f"Warning: Could not save homer.config: {str(e)}")
    
    def _auto_load_snakemake_config(self):
        """Auto-load Snakemake configuration from homer.config if it exists"""
        try:
            # Only check in the selected path_to_data folder
            if not self.path_to_data or not os.path.exists(self.path_to_data):
                print("No path_to_data set, skipping auto-load")
                return
            
            # Check for homer.config ONLY in the selected folder
            homer_config_path = os.path.join(self.path_to_data, 'homer.config')
            
            if not os.path.exists(homer_config_path):
                print(f"No homer.config found in {self.path_to_data}, skipping auto-load")
                return
            
            print(f"Found homer.config at {homer_config_path}")
            
            # Try safe_load first, fall back to unsafe if needed
            homer_config = None
            try:
                with open(homer_config_path, 'r') as f:
                    homer_config = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                # File contains unsafe YAML tags (numpy objects, etc.)
                print("Warning: homer.config contains unsafe YAML tags, using unsafe_load")
                try:
                    with open(homer_config_path, 'r') as f:
                        homer_config = yaml.unsafe_load(f)
                except Exception as e:
                    print(f"Warning: Could not load homer.config even with unsafe_load: {e}")
                    return
            
            if not homer_config:
                return
            
            snakefile_path = homer_config.get('snakefile_path')
            config_path = homer_config.get('config_path')
            
            if snakefile_path and config_path:
                if os.path.exists(snakefile_path) and os.path.exists(config_path):
                    print(f"Auto-loading Snakemake config from homer.config")
                    self._load_snakemake_config(snakefile_path, config_path)
                else:
                    print(f"Warning: Paths in homer.config not found")
                    if not os.path.exists(snakefile_path):
                        print(f"  Snakefile not found: {snakefile_path}")
                    if not os.path.exists(config_path):
                        print(f"  Config not found: {config_path}")
        except Exception as e:
            print(f"Warning: Could not auto-load homer.config: {str(e)}")
    
    def _save_gui_state(self):
        """Save current GUI state (subject, run, selections) to homer.config"""
        try:
            # Don't save if we're in the middle of restoring state
            if hasattr(self, '_restoring_state') and self._restoring_state:
                return
            
            if not self.path_to_data or not os.path.exists(self.path_to_data):
                return
            
            homer_config_path = os.path.join(self.path_to_data, 'homer.config')
            
            # Load existing config or create new one
            # Try safe_load first, fall back to unsafe if needed
            homer_config = {}
            if os.path.exists(homer_config_path):
                try:
                    with open(homer_config_path, 'r') as f:
                        homer_config = yaml.safe_load(f) or {}
                except yaml.constructor.ConstructorError:
                    # File contains unsafe YAML tags (e.g., numpy objects)
                    # Load with unsafe_load but only extract safe parts
                    try:
                        with open(homer_config_path, 'r') as f:
                            homer_config = yaml.unsafe_load(f) or {}
                    except:
                        homer_config = {}
            
            # Collect GUI state
            gui_state = {}
            
            # Subject and run
            if hasattr(self, 'subj') and self.subj.currentText():
                gui_state['subject'] = str(self.subj.currentText())
            if hasattr(self, 'run') and self.run.currentText():
                gui_state['run'] = str(self.run.currentText())
            
            # Timeseries selection
            if hasattr(self, 'ts') and self.ts.currentItem():
                gui_state['timeseries'] = str(self.ts.currentItem().text())
            
            # Wavelength/chromo selection
            if hasattr(self, 'wv'):
                gui_state['wavelength_index'] = int(self.wv.currentRow())
            
            # Selected channels - convert to native Python ints
            if hasattr(self, 'selected_channels'):
                gui_state['selected_channels'] = [int(ch) for ch in self.selected_channels]
            
            # Preserve axis zoom checkbox state
            if hasattr(self, 'preserve_axis_zoom'):
                gui_state['preserve_axis_zoom'] = bool(self.preserve_axis_zoom.isChecked())
            
            # Auto scale Y-axis checkbox state
            if hasattr(self, 'auto_scale_y'):
                gui_state['auto_scale_y'] = bool(self.auto_scale_y.isChecked())
            
            # View optodes as circles checkbox state
            if hasattr(self, 'opt2circ'):
                gui_state['opt2circ'] = bool(self.opt2circ.isChecked())
            
            # Auxiliary data selection
            if hasattr(self, 'aux') and self.aux.currentText():
                gui_state['aux'] = str(self.aux.currentText())
            
            # Selected stimulus types (for both regular and HRF view)
            if hasattr(self, 'selected_stim_types'):
                gui_state['selected_stim_types'] = [str(s) for s in self.selected_stim_types]
            
            # HRF view state
            if hasattr(self, 'hrf_view'):
                gui_state['hrf_view'] = bool(self.hrf_view.isChecked())
                if self.hrf_view.isChecked():
                    # HbO/HbR selection is handled by wavelength_index (wv widget)
                    if hasattr(self, 'hrf_group_avg'):
                        gui_state['hrf_group_avg'] = bool(self.hrf_group_avg.isChecked())
            
            # Update homer.config with gui_state
            homer_config['gui_state'] = gui_state
            
            # Clean homer_config to remove any numpy objects before saving
            homer_config_clean = self._clean_config_for_yaml(homer_config)
            
            # Write back to file
            with open(homer_config_path, 'w') as f:
                yaml.dump(homer_config_clean, f, default_flow_style=False, sort_keys=False)
            
            print(f"Saved GUI state to {homer_config_path}")
            
        except Exception as e:
            print(f"Warning: Could not save GUI state: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _clean_config_for_yaml(self, obj):
        """Recursively clean config object to remove numpy types and make it YAML-safe"""
        if isinstance(obj, dict):
            return {str(k): self._clean_config_for_yaml(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._clean_config_for_yaml(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            # For other types, try to convert to string
            try:
                return str(obj)
            except:
                return None
    
    def _check_for_saved_gui_state(self):
        """Check if there's a saved GUI state in homer.config"""
        try:
            if not self.path_to_data or not os.path.exists(self.path_to_data):
                return False
            
            homer_config_path = os.path.join(self.path_to_data, 'homer.config')
            
            if not os.path.exists(homer_config_path):
                return False
            
            # Try to load and check for gui_state section
            try:
                with open(homer_config_path, 'r') as f:
                    homer_config = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                try:
                    with open(homer_config_path, 'r') as f:
                        homer_config = yaml.unsafe_load(f)
                except:
                    return False
            
            return homer_config is not None and 'gui_state' in homer_config
            
        except Exception:
            return False
    
    def _load_gui_state(self):
        """Load and restore GUI state from homer.config"""
        try:
            if not self.path_to_data or not os.path.exists(self.path_to_data):
                return
            
            homer_config_path = os.path.join(self.path_to_data, 'homer.config')
            
            if not os.path.exists(homer_config_path):
                return
            
            # Try safe_load first, fall back to unsafe if needed
            homer_config = None
            try:
                with open(homer_config_path, 'r') as f:
                    homer_config = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                # File contains unsafe YAML tags - try unsafe_load
                try:
                    with open(homer_config_path, 'r') as f:
                        homer_config = yaml.unsafe_load(f)
                except Exception as e:
                    print(f"Warning: Could not load homer.config even with unsafe_load: {e}")
                    return
            
            if not homer_config or 'gui_state' not in homer_config:
                return
            
            gui_state = homer_config['gui_state']
            print(f"Restoring GUI state from homer.config: {list(gui_state.keys())}")
            
            # Set a flag to prevent saving during restoration
            self._restoring_state = True
            
            try:
                # Restore subject
                if 'subject' in gui_state and hasattr(self, 'subj'):
                    subject = gui_state['subject']
                    index = self.subj.findText(subject)
                    if index >= 0:
                        self.subj.blockSignals(True)
                        self.subj.setCurrentIndex(index)
                        self.subj.blockSignals(False)
                        print(f"  Restored subject: {subject}")
                
                # Restore run
                if 'run' in gui_state and hasattr(self, 'run'):
                    run = gui_state['run']
                    # Update run list for the selected subject first
                    if hasattr(self, 'subj'):
                        subject_key = self.subj.currentText()
                        if subject_key in self.subject_to_runs_map:
                            self.run.blockSignals(True)
                            self.run.clear()
                            self.run.addItems(self.subject_to_runs_map[subject_key])
                            self.run.blockSignals(False)
                    
                    # Now set the run
                    index = self.run.findText(run)
                    if index >= 0:
                        self.run.blockSignals(True)
                        self.run.setCurrentIndex(index)
                        self.run.blockSignals(False)
                        print(f"  Restored run: {run}")
                
                # Load the data for subject/run before restoring other selections
                if 'subject' in gui_state and 'run' in gui_state:
                    self._update_recording_data(gui_state['subject'], gui_state['run'], subject_changed=True)
                
                # Restore timeseries selection
                if 'timeseries' in gui_state and hasattr(self, 'ts') and hasattr(self, 'snirfRec'):
                    ts_name = gui_state['timeseries']
                    for i in range(self.ts.count()):
                        if self.ts.item(i).text() == ts_name:
                            self.ts.blockSignals(True)
                            self.ts.setCurrentRow(i)
                            self.ts.blockSignals(False)
                            # Manually trigger the timeseries change without saving
                            if ts_name != 'amp' and self.processed_rec and hasattr(self.processed_rec, 'timeseries') and ts_name in self.processed_rec.timeseries:
                                self.snirfData = self.processed_rec.timeseries[ts_name]
                            else:
                                self.snirfData = self.snirfRec.timeseries.get(ts_name)
                            self.ts_sel = ts_name
                            print(f"  Restored timeseries: {ts_name}")
                            break
                
                # Update wavelength/concentration dropdown based on timeseries
                if hasattr(self, 'snirfData') and self.snirfData is not None:
                    if "wavelength" in self.snirfData.dims:
                        self.wv_label.setText("Wavelength:")
                        self.wv.clear()
                        for i_w, wvl in enumerate(self.snirfData.wavelength.values):
                            self.wv.insertItem(i_w, str(wvl))
                    elif "chromo" in self.snirfData.dims:
                        self.wv_label.setText("Concentration:")
                        self.wv.clear()
                        for i_w, wvl in enumerate(self.snirfData.chromo.values):
                            self.wv.insertItem(i_w, f"[{str(wvl)}]")
                
                # Restore wavelength/chromo index
                if 'wavelength_index' in gui_state and hasattr(self, 'wv'):
                    wv_index = gui_state['wavelength_index']
                    if 0 <= wv_index < self.wv.count():
                        self.wv.blockSignals(True)
                        self.wv.setCurrentRow(wv_index)
                        self.wv.blockSignals(False)
                        print(f"  Restored wavelength index: {wv_index}")
                
                # Restore selected channels
                if 'selected_channels' in gui_state and hasattr(self, 'snirfData') and self.snirfData is not None:
                    saved_channels = gui_state['selected_channels']
                    # Validate channels are within range
                    max_channel = len(self.snirfData.channel.values) - 1
                    valid_channels = [ch for ch in saved_channels if 0 <= ch <= max_channel]
                    self.selected_channels = valid_channels
                    print(f"  Restored {len(valid_channels)} selected channels")
                
                # Restore preserve axis zoom state
                if 'preserve_axis_zoom' in gui_state and hasattr(self, 'preserve_axis_zoom'):
                    self.preserve_axis_zoom.blockSignals(True)
                    self.preserve_axis_zoom.setChecked(gui_state['preserve_axis_zoom'])
                    self.preserve_axis_zoom.blockSignals(False)
                    # Enable/disable auto_scale_y based on preserve_axis_zoom state
                    if hasattr(self, 'auto_scale_y'):
                        self.auto_scale_y.setEnabled(gui_state['preserve_axis_zoom'])
                    print(f"  Restored preserve_axis_zoom: {gui_state['preserve_axis_zoom']}")
                
                # Restore auto scale Y-axis state
                if 'auto_scale_y' in gui_state and hasattr(self, 'auto_scale_y'):
                    self.auto_scale_y.blockSignals(True)
                    self.auto_scale_y.setChecked(gui_state['auto_scale_y'])
                    self.auto_scale_y.blockSignals(False)
                    print(f"  Restored auto_scale_y: {gui_state['auto_scale_y']}")
                
                # Restore view optodes as circles state
                if 'opt2circ' in gui_state and hasattr(self, 'opt2circ'):
                    self.opt2circ.blockSignals(True)
                    self.opt2circ.setChecked(gui_state['opt2circ'])
                    self.opt2circ.blockSignals(False)
                    # Manually trigger the toggle to update visibility
                    self._toggle_circles()
                    print(f"  Restored opt2circ: {gui_state['opt2circ']}")
                
                # Restore auxiliary selection
                if 'aux' in gui_state and hasattr(self, 'aux'):
                    aux_name = gui_state['aux']
                    index = self.aux.findText(aux_name)
                    if index >= 0:
                        self.aux.blockSignals(True)
                        self.aux.setCurrentIndex(index)
                        self.aux.blockSignals(False)
                        # Manually set aux selection
                        if aux_name != "None" and aux_name != "dark signal":
                            self.aux_sel = self.snirfRec.aux_ts[aux_name]
                            self.aux_type = aux_name
                        else:
                            self.aux_sel = []
                            self.aux_type = None
                        print(f"  Restored aux: {aux_name}")
                
                # Restore selected stimulus types (for both regular and HRF view)
                if 'selected_stim_types' in gui_state:
                    if hasattr(self, 'available_stim_types'):
                        # Only restore stimuli that are available in current dataset
                        saved_stims = set(gui_state['selected_stim_types'])
                        valid_stims = saved_stims.intersection(set(self.available_stim_types))
                        self.selected_stim_types = valid_stims
                        self._update_stim_button_text()
                        print(f"  Restored {len(valid_stims)} selected stimuli")
                    else:
                        # Store for later if available_stim_types not yet set
                        self.selected_stim_types = set(gui_state['selected_stim_types'])
                        print(f"  Stored {len(self.selected_stim_types)} selected stimuli")
                
                # Restore HRF view state
                if 'hrf_view' in gui_state and hasattr(self, 'hrf_view'):
                    self.hrf_view.setChecked(gui_state['hrf_view'])
                    
                    if gui_state['hrf_view']:
                        # HbO/HbR selection is handled by wavelength_index (wv widget)
                        if 'hrf_group_avg' in gui_state and hasattr(self, 'hrf_group_avg'):
                            self.hrf_group_avg.setChecked(gui_state['hrf_group_avg'])
                        print(f"  Restored HRF view state")
                
                # Redraw with restored state
                if hasattr(self, 'snirfData') and self.snirfData is not None:
                    self._draw_timeseries()
                    print("GUI state restored successfully")
                else:
                    print("GUI state partially restored (no data to draw)")
                
            finally:
                # Clear the restoration flag
                self._restoring_state = False
            
        except Exception as e:
            print(f"Warning: Could not load GUI state: {str(e)}")
            import traceback
            traceback.print_exc()
            self._restoring_state = False
    
    def _check_pipeline_switch_state(self):
        """Check if GUI was relaunched after a pipeline switch and show notification"""
        try:
            if not self.path_to_data or not os.path.exists(self.path_to_data):
                return
            
            switch_state_file = os.path.join(self.path_to_data, '.pipeline_switch_state')
            
            if not os.path.exists(switch_state_file):
                return  # No pending switch
            
            # Read the switch state
            with open(switch_state_file, 'r') as f:
                lines = f.readlines()
                switched_path = lines[0].strip() if len(lines) > 0 else None
                was_new = bool(int(lines[1].strip())) if len(lines) > 1 else False
            
            # Delete the state file
            try:
                os.remove(switch_state_file)
                print(f"Removed pipeline switch state file")
            except:
                pass
            
            # Show notification
            pipeline_name = os.path.basename(self.path_to_data)
            
            if was_new:
                # Show reminder for new pipelines
                QtWidgets.QMessageBox.information(
                    self,
                    "Pipeline Switched",
                    f"✓ Successfully switched to new pipeline:\n{pipeline_name}\n\n"
                    "⚠️ This pipeline needs configuration!\n\n"
                    "Next steps:\n"
                    "1. Go to: Snakemake → Setup Pipeline\n"
                    "2. Select Snakefile and create config\n"
                    "3. Configure your pipeline settings"
                )
            else:
                # Just show success message for existing pipelines
                self.statbar.showMessage(f"✓ Switched to pipeline: {pipeline_name}", 5000)
                print(f"\\n{'='*70}")
                print(f"PIPELINE SWITCH SUCCESSFUL: {pipeline_name}")
                print(f"{'='*70}\\n")
                
        except Exception as e:
            print(f"Warning: Error checking pipeline switch state: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_snakemake_config(self, snakefile_path, config_path):
        """Load Snakemake config and update menu dynamically"""
        try:
            if not os.path.exists(config_path):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Config Not Found",
                    f"Config file not found at:\n{config_path}"
                )
                return
            
            # Load the config
            with open(config_path, 'r') as f:
                self.snakemake_config = yaml.safe_load(f)
            
            self.snakefile_path = snakefile_path
            self.snakemake_config_path = config_path
            
            # Load file status maps (current scope + full scope)
            self._load_file_status_maps()
            
            # Rebuild the menu
            self._rebuild_snakemake_menu()
            
            self.statbar.showMessage(f"Loaded config from {os.path.basename(config_path)}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error Loading Config",
                f"Failed to load config:\n{str(e)}"
            )
    
    def _rebuild_snakemake_menu(self):
        """Rebuild Snakemake menu based on loaded config"""
        if not self.snakemake_menu or not self.snakemake_config:
            return
        
        # Remove all dynamic menu items
        for action in self.dynamic_menu_actions:
            self.snakemake_menu.removeAction(action)
        self.dynamic_menu_actions.clear()
        
        # Find the Dataset action to insert dynamic items after it
        dataset_action = None
        second_separator = None
        found_first_separator = False
        
        for action in self.snakemake_menu.actions():
            if action.text() == "Dataset":
                dataset_action = action
            # Find the second separator (the one before Run Pipeline)
            if action.isSeparator():
                if found_first_separator:
                    second_separator = action
                    break
                else:
                    found_first_separator = True
        
        # Add menu items for each config block (except 'dataset')
        # Preserve order from config file
        insert_after = dataset_action
        for block_name in self.snakemake_config.keys():
            if block_name == 'dataset':
                continue  # Already have Dataset menu item
            
            # Create a friendly display name
            display_name = block_name.replace('_', ' ').title()
            
            action = QAction(display_name, self)
            action.setStatusTip(f"{display_name} configuration")
            action.triggered.connect(lambda checked, bn=block_name: self._snakemake_config_item(bn))
            
            # Insert after the previous action (Dataset or last dynamic item)
            if insert_after:
                self.snakemake_menu.insertAction(self._get_next_action(insert_after), action)
            else:
                self.snakemake_menu.addAction(action)
            
            self.dynamic_menu_actions.append(action)
            insert_after = action  # Next item inserts after this one
        
        # Add Refresh Colors action before the second separator (Run Pipeline)
        if second_separator:
            refresh_action = QAction("Refresh File Status", self)
            refresh_action.setStatusTip("Manually refresh file processing status and colors")
            refresh_action.triggered.connect(self._manual_refresh_colors)
            self.snakemake_menu.insertAction(second_separator, refresh_action)
            self.dynamic_menu_actions.append(refresh_action)
            
            # Add Stop Monitoring action
            stop_action = QAction("Stop Monitoring", self)
            stop_action.setStatusTip("Stop automatic pipeline monitoring")
            stop_action.triggered.connect(self._stop_pipeline_monitoring)
            self.snakemake_menu.insertAction(second_separator, stop_action)
            self.dynamic_menu_actions.append(stop_action)
    
    def _manual_refresh_colors(self):
        """Manually trigger a refresh of file colors"""
        self._update_all_file_colors()
        self.statbar.showMessage("File status refreshed")
    
    def _get_next_action(self, action):
        """Get the action that comes after the given action in the menu"""
        actions = self.snakemake_menu.actions()
        try:
            idx = actions.index(action)
            if idx + 1 < len(actions):
                return actions[idx + 1]
        except (ValueError, IndexError):
            pass
        return None
    
    def _edit_config_minimal_dataset(self, config_path):
        """Edit only root_dir and derivatives_subfolder for dataset before config is loaded"""
        try:
            # Check if file exists, if not create a minimal one
            if not os.path.exists(config_path):
                # Create directory if needed
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                
                # Create minimal config with just dataset
                minimal_config = {
                    'dataset': {
                        'root_dir': '',
                        'derivatives_subfolder': ''
                    }
                }
                with open(config_path, 'w') as f:
                    yaml.dump(minimal_config, f, default_flow_style=False)
            
            # Load existing config
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            
            if 'dataset' not in full_config:
                full_config['dataset'] = {}
            
            # Extract only root_dir and derivatives_subfolder
            minimal_data = {
                'root_dir': full_config['dataset'].get('root_dir', ''),
                'derivatives_subfolder': full_config['dataset'].get('derivatives_subfolder', '')
            }
            
            # Open editor dialog with only these two fields
            dialog = ConfigEditorDialog(minimal_data, 'dataset (basic)', readonly_keys=None, field_tooltips=None, parent=self)
            
            if dialog.exec() == QtWidgets.QDialog.Accepted:
                # Get updated data
                updated_data = dialog.get_updated_data()
                
                # Update only these fields in the full config
                full_config['dataset']['root_dir'] = updated_data.get('root_dir', '')
                full_config['dataset']['derivatives_subfolder'] = updated_data.get('derivatives_subfolder', '')
                
                # Save back to file
                with open(config_path, 'w') as f:
                    yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)
                
                self.statbar.showMessage("Dataset configuration saved!")
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                f"Error editing configuration:\n{str(e)}"
            )
    
    def _edit_config_block(self, config_path, block_name, readonly_keys=None):
        """Load, edit, and save a specific block from config file"""
        try:
            # Check if file exists
            if not os.path.exists(config_path):
                QtWidgets.QMessageBox.warning(
                    self, 
                    "Config File Not Found",
                    f"Could not find config file at:\n{config_path}"
                )
                return
            
            # Read the original file to preserve comments
            with open(config_path, 'r') as f:
                original_content = f.read()
                original_lines = original_content.splitlines()
            
            # Load the config data
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            
            if block_name not in full_config:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Block Not Found",
                    f"Block '{block_name}' not found in config file"
                )
                return
            
            # Get the specific block
            block_data = full_config[block_name]
            
            # Extract field tooltips from comments
            field_tooltips = self._extract_field_comments(original_lines, block_name)
            
            # For dataset block, add dynamic tooltips based on available data
            if block_name == 'dataset':
                dataset_tooltips = self._generate_dataset_tooltips()
                field_tooltips.update(dataset_tooltips)

            if block_name == 'preprocess':
                od2conc_cfg = (
                    block_data
                    .setdefault('steps', {})
                    .setdefault('od2conc', {})
                )
                od2conc_cfg.setdefault('dpf', [1, 1])
                field_tooltips.setdefault(
                    'steps.od2conc.dpf',
                    'Differential pathlength factors for each wavelength. '
                    'Current Cedalion behavior treats dpf[0] = 1 as the 1 mm '
                    'pathlength branch; other values use source-detector distance times DPF.'
                )
            
            # Open editor dialog with file_map and subjects if editing dataset block
            if block_name == 'dataset':
                dialog = ConfigEditorDialog(
                    block_data, block_name, readonly_keys, field_tooltips, self,
                    file_map=self.file_map, subjects=self.subjects
                )
            else:
                dialog = ConfigEditorDialog(block_data, block_name, readonly_keys, field_tooltips, self)
            
            if dialog.exec() == QtWidgets.QDialog.Accepted:
                # Get updated data
                updated_block = dialog.get_updated_data()
                
                # Check if derivatives_subfolder changed BEFORE saving
                # We need to handle this specially to avoid corrupting the current pipeline's config
                pipeline_switch_needed = False
                new_derivatives_subfolder = None
                
                if block_name == 'dataset':
                    new_derivatives_subfolder = updated_block.get('derivatives_subfolder', '')
                    if new_derivatives_subfolder != dialog.original_derivatives_subfolder:
                        pipeline_switch_needed = True
                        # Don't save the derivatives_subfolder change to current config
                        # Keep the original value for the current pipeline
                        updated_block['derivatives_subfolder'] = dialog.original_derivatives_subfolder
                
                # Update the full config with the edited block
                full_config[block_name] = updated_block
                
                # Save back to file, preserving comments and formatting where possible
                self._save_yaml_with_comments(config_path, full_config, original_lines, block_name)
                
                self.statbar.showMessage(f"{block_name.replace('_', ' ').title()} configuration saved!")
                
                # Now handle pipeline switch if needed
                if pipeline_switch_needed:
                    # Pass the updated config with the NEW derivatives_subfolder
                    updated_block['derivatives_subfolder'] = new_derivatives_subfolder
                    full_config[block_name] = updated_block
                    
                    self._handle_pipeline_switch(
                        full_config['dataset']['root_dir'],
                        new_derivatives_subfolder,
                        full_config  # Pass the config to save to new location
                    )
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                f"Error editing configuration:\n{str(e)}"
            )
    
    def _generate_dataset_tooltips(self):
        """Generate tooltips for dataset fields based on available data"""
        tooltips = {}
        
        if not self.file_map or not self.subjects:
            return tooltips
        
        try:
            # Extract available subjects
            available_subjects = sorted(self.subjects)
            subjects_str = ', '.join(available_subjects)
            
            # Extract available tasks
            tasks = set()
            runs = set()
            for subject in self.file_map.values():
                for run_key in subject.keys():
                    # Extract task from run_key (e.g., "task-racing_run-01")
                    if 'task-' in run_key:
                        task_part = run_key.split('task-')[1]
                        task = task_part.split('_')[0]
                        tasks.add(task)
                    # Extract run number
                    if 'run-' in run_key:
                        run_part = run_key.split('run-')[1]
                        run = run_part.split('_')[0]  # Get just the number
                        runs.add(run)
            
            tasks_str = ', '.join(sorted(tasks)) if tasks else 'None'
            runs_str = ', '.join(sorted(runs)) if runs else 'None'
            
            # Generate tooltips
            tooltips['subjects_to_exclude'] = (
                f"Available subjects in dataset: {subjects_str}\n\n"
                f"Enter subject IDs to exclude (comma-separated).\n"
                f"Example: 752, 753 or sub-752, sub-753"
            )
            
            tooltips['task'] = (
                f"Available tasks in dataset: {tasks_str}\n\n"
                f"Specify which task to process from the SNIRF files."
            )
            
            tooltips['run_list'] = (
                f"Available runs in dataset: {runs_str}\n\n"
                f"List of run numbers to include (comma-separated).\n"
                f"Example: ['01', '02'] or leave empty for all runs"
            )
            
            tooltips['num_runs'] = (
                f"Available runs in dataset: {runs_str}\n\n"
                f"Number of runs per subject to process."
            )
            
        except Exception as e:
            print(f"Error generating dataset tooltips: {e}")
        
        return tooltips

    def _handle_pipeline_switch(self, root_dir, new_derivatives_subfolder, updated_config=None):
        """Handle switching to a different pipeline folder
        
        Args:
            root_dir: BIDS root directory
            new_derivatives_subfolder: New pipeline folder name
            updated_config: Config dict (not used - kept for compatibility)
        """
        try:
            # Build new path_to_data
            new_path_to_data = os.path.join(root_dir, 'derivatives', 'cedalion', new_derivatives_subfolder)
            
            # Create folder if it doesn't exist
            created_new = False
            if not os.path.exists(new_path_to_data):
                os.makedirs(new_path_to_data, exist_ok=True)
                print(f"Created new pipeline folder: {new_path_to_data}")
                created_new = True
                print(f"Note: New pipeline folder has no config - will need to be configured after switch")
            else:
                print(f"Switching to existing pipeline folder: {new_path_to_data}")
            
            # Confirm switch with user
            msg = f"Switch to pipeline folder:\n{new_derivatives_subfolder}\n\n"
            msg += "This will reload all data from the new pipeline location.\n"
            if created_new:
                msg += "\n⚠️  New pipeline folder created.\nYou'll need to configure it after switching:\nSnakemake → Setup Pipeline → Configure"
            msg += "\nContinue?"
            
            reply = QtWidgets.QMessageBox.question(
                self,
                'Switch Pipeline?',
                msg,
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes
            )
            
            if reply == QtWidgets.QMessageBox.Yes:
                self._switch_pipeline(new_path_to_data, created_new)
            else:
                # User cancelled - clean up if we created a new folder
                if created_new and os.path.exists(new_path_to_data):
                    # Remove the newly created folder
                    import shutil
                    try:
                        shutil.rmtree(new_path_to_data)
                        print(f"Removed cancelled pipeline folder: {new_path_to_data}")
                    except Exception as e:
                        print(f"Warning: Could not remove folder: {e}")
                
                QtWidgets.QMessageBox.information(
                    self,
                    "Switch Cancelled",
                    "Pipeline switch cancelled. Your current pipeline settings remain unchanged."
                )
        
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Pipeline Switch Error",
                f"Error switching pipeline:\\n{str(e)}"
            )
            print(f"Error in _handle_pipeline_switch: {e}")
            import traceback
            traceback.print_exc()
    
    def _switch_pipeline(self, new_path_to_data, created_new=False):
        """Switch to a new pipeline folder by relaunching the GUI"""
        try:
            print(f"\\n{'='*70}")
            print(f"PREPARING TO SWITCH PIPELINE: {new_path_to_data}")
            print(f"{'='*70}\\n")
            
            # Save the new path to a temporary state file for the relaunch
            switch_state_file = os.path.join(new_path_to_data, '.pipeline_switch_state')
            with open(switch_state_file, 'w') as f:
                f.write(f"{new_path_to_data}\n")
                f.write(f"{int(created_new)}\n")  # Store as 1 or 0
            
            print(f"Saved switch state to: {switch_state_file}")
            
            # Show user message
            msg = f"Pipeline switch prepared.\n\n"
            msg += f"The GUI will now restart to load:\n{os.path.basename(new_path_to_data)}\n\n"
            if created_new:
                msg += "⚠️  New pipeline folder created (no config yet).\nYou'll need to configure it after restart."
            else:
                msg += "All data will be loaded from the selected pipeline."
            
            QtWidgets.QMessageBox.information(
                self,
                "Restarting GUI",
                msg
            )
            
            # Relaunch the GUI with pipeline path as argument
            python_exe = sys.executable
            script_path = os.path.abspath(sys.argv[0])
            
            print(f"Relaunching: {python_exe} {script_path} {new_path_to_data}")
            print(f"Working directory: {os.getcwd()}")
            
            # Launch new instance with pipeline path as argument
            # Keep using same terminal, but close stdin to allow terminal to return
            subprocess.Popen(
                [python_exe, script_path, new_path_to_data],
                cwd=os.getcwd(),
                stdin=subprocess.DEVNULL  # Close stdin so parent can exit cleanly
            )
            
            # Close current instance
            print("Closing current GUI instance...")
            print("New GUI instance starting...\n")
            
            # Force exit the entire application
            QtWidgets.QApplication.quit()
            sys.exit(0)  # Force clean exit of parent process
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Switch Failed",
                f"Failed to switch pipeline:\\n{str(e)}"
            )
            print(f"Error in _switch_pipeline: {e}")
            import traceback
            traceback.print_exc()

    def _extract_field_comments(self, original_lines, block_name):
        """Extract comments from YAML file and map them to field keys"""
        field_comments = {}
        
        # Find the block in the file
        in_block = False
        block_indent = 0
        pending_comment = None
        
        for i, line in enumerate(original_lines):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            
            # Check if we're entering the target block
            if stripped.startswith(f"{block_name}:"):
                in_block = True
                block_indent = indent
                # Check for inline comment on the block declaration line
                if '#' in line:
                    comment_part = line.split('#', 1)[1].strip()
                    if comment_part:
                        field_comments['__block__'] = comment_part
                continue
            
            if not in_block:
                continue
            
            # Check if we've exited the block (another top-level key at same or lower indent)
            if stripped and not stripped.startswith('#') and indent <= block_indent:
                break
            
            # Skip empty lines
            if not stripped:
                pending_comment = None
                continue
            
            # Check for standalone comment line (comment for the next field)
            if stripped.startswith('#'):
                comment = stripped[1:].strip()
                pending_comment = comment
                continue
            
            # This is a field line with a key
            if ':' in line:
                # Extract field name
                field_part = line.split('#')[0]  # Remove inline comment first
                field_name = field_part.split(':')[0].strip()
                
                # Check for inline comment (comment on same line as field)
                if '#' in line:
                    inline_comment = line.split('#', 1)[1].strip()
                    if inline_comment:
                        field_comments[field_name] = inline_comment
                        pending_comment = None  # Clear pending since we used inline
                        continue
                
                # Use pending comment if no inline comment
                if pending_comment:
                    field_comments[field_name] = pending_comment
                    pending_comment = None
        
        return field_comments
    
    def _save_yaml_with_comments(self, config_path, full_config, original_lines, modified_block):
        """Save YAML config while preserving comments and quote styles"""
        # Find where the modified block starts and ends in the original file
        block_start = -1
        block_indent = 0
        
        for i, line in enumerate(original_lines):
            stripped = line.lstrip()
            if stripped.startswith(f"{modified_block}:"):
                block_start = i
                block_indent = len(line) - len(stripped)
                break
        
        if block_start == -1:
            # Fallback to standard YAML dump if we can't find the block
            with open(config_path, 'w') as f:
                yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)
            return
        
        # Find block end (next top-level key or end of file)
        block_end = len(original_lines)
        for i in range(block_start + 1, len(original_lines)):
            line = original_lines[i]
            if line.strip() and not line.startswith(' ') and not line.strip().startswith('#'):
                block_end = i
                break
        
        # Extract comments and quote styles from the modified block
        block_comments = {}
        quoted_values = set()  # Track which values were originally quoted
        list_quote_patterns = {}  # Track which list keys had quoted items
        
        for i in range(block_start, block_end):
            line = original_lines[i]
            
            # Check for lists with quoted strings
            if '[' in line and ']' in line:
                # Extract the key for this list
                key_match = re.match(r'\s*([^:]+):\s*\[', line)
                if key_match:
                    list_key = key_match.group(1).strip()
                    # Check if any items in the list are quoted
                    if '"' in line or "'" in line:
                        list_quote_patterns[list_key] = True
            
            # Check for quoted strings in the original
            if '"' in line or "'" in line:
                # Find all quoted strings in this line
                quoted_matches = re.findall(r'["\']([^"\']+)["\']', line)
                quoted_values.update(quoted_matches)
            
            if '#' in line:
                # Extract the key and comment
                before_comment = line.split('#')[0]
                comment = '#' + '#'.join(line.split('#')[1:])
                
                # Try to find the key
                match = re.match(r'\s*([^:]+):', before_comment)
                if match:
                    key = match.group(1).strip()
                    block_comments[key] = comment
        
        # Generate new YAML for the modified block
        block_yaml_lines = []
        block_data = full_config[modified_block]
        
        # Track which keys should have quoted list items
        current_list_key = None
        
        # Use a custom dumper to preserve quote styles
        class QuotePreservingDumper(yaml.SafeDumper):
            pass
        
        def represent_str(dumper, data):
            # If this value was originally quoted, keep quotes
            if data in quoted_values:
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')
            
            # Check if string needs quotes (has special chars, spaces, etc.)
            if any(char in data for char in [' ', ':', '#', '-', '[', ']', '{', '}', '!', '*', '&', '/', '.']):
                # Use double quotes for strings with special characters
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')
            
            # Check if it looks like a boolean or number
            if data.lower() in ['true', 'false', 'yes', 'no', 'on', 'off']:
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')
            
            try:
                float(data)
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')
            except ValueError:
                pass
            
            return dumper.represent_scalar('tag:yaml.org,2002:str', data)
        
        def represent_list(dumper, data):
            # Keep lists inline (flow style) like [item1, item2]
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        
        QuotePreservingDumper.add_representer(str, represent_str)
        QuotePreservingDumper.add_representer(list, represent_list)
        
        # Dump just the block data
        block_yaml = yaml.dump({modified_block: block_data}, 
                               Dumper=QuotePreservingDumper, 
                               default_flow_style=False, 
                               sort_keys=False)
        
        block_yaml_lines = block_yaml.splitlines()
        
        # Post-process lines to add quotes to list items if the original had them
        for i, line in enumerate(block_yaml_lines):
            if '[' in line and ']' in line:
                # This is a list line
                key_match = re.match(r'\s*([^:]+):\s*\[', line)
                if key_match:
                    list_key = key_match.group(1).strip()
                    # If this list originally had quoted items, add quotes to all items
                    if list_key in list_quote_patterns:
                        # Extract the list content
                        list_match = re.search(r'\[(.*?)\]', line)
                        if list_match:
                            list_content = list_match.group(1)
                            # Split by comma and process each item
                            items = [item.strip() for item in list_content.split(',')]
                            # Add quotes to items that don't already have them
                            quoted_items = []
                            for item in items:
                                if item and not (item.startswith('"') or item.startswith("'")):
                                    quoted_items.append(f'"{item}"')
                                else:
                                    quoted_items.append(item)
                            # Reconstruct the line
                            before_list = line[:line.index('[') + 1]
                            after_list = ']' + line[line.rindex(']') + 1:]
                            block_yaml_lines[i] = before_list + ', '.join(quoted_items) + after_list
        
        # Re-attach comments to matching keys
        for i, line in enumerate(block_yaml_lines):
            if ':' in line and not line.strip().startswith('#'):
                match = re.match(r'\s*([^:]+):', line)
                if match:
                    key = match.group(1).strip()
                    if key in block_comments:
                        # Append the comment to the line
                        block_yaml_lines[i] = line.rstrip() + '  ' + block_comments[key]
        
        # Reconstruct the full file
        new_lines = (
            original_lines[:block_start] +
            block_yaml_lines +
            original_lines[block_end:]
        )
        
        # Write back to file
        with open(config_path, 'w') as f:
            f.write('\n'.join(new_lines))
            if new_lines:  # Add final newline if there's content
                f.write('\n')

    def _init_widgets(self, redraw_optodes=True):
        """Initializes widgets based on the current prepared data."""
        if redraw_optodes:
            self.optodes_drawn = False

        # Clear any existing channel highlights first
        if hasattr(self, 'chan_highlight') and self.chan_highlight:
            for line in self.chan_highlight:
                try:
                    line.remove()
                except:
                    pass
        
        # Initialize holders to control each part of the plot first
        self.src_label = [0] * len(self.sx)
        self.det_label = [0] * len(self.dx)
        self.chan_highlight = []
        self.aux_sel = []
        self.auxplot = []
        self.aux_type = None
        self.aux_rect_width = 0
        
        # Track selected channels directly (for line picking)
        self.selected_channels = []
        
        # Update stimulus types if we have stimulus data
        self._update_stim_types()
        
        # Block signals to prevent premature updates
        self.ts.blockSignals(True)
        self.auxs.blockSignals(True)

        # Reset and populate widgets
        self.ts.clear()
        self.ts.addItems(["None"])
        self.ts.addItems(self.timeseries_keys)
        
        self.auxs.clear()
        self.auxs.addItems(["None"])
        self.auxs.addItems(self.aux_ts_keys)

        # Safely unblock signals
        self.ts.blockSignals(False)
        self.auxs.blockSignals(False)

        # Update stimulus selection based on available data
        self._update_stim_types()
        
        # Enable/disable HRF view checkbox based on data availability
        # and determine which stimulus types have HRF data
        # HRF should be enabled if either individual data OR group average is available
        hrf_available = False
        
        if self.hrf_group_avg.isChecked():
            # When group average is checked, keep HRF view enabled
            hrf_available = True
            # Try to load group average to get available stim types
            group_avg = self._load_group_average_hrf()
            if group_avg is not None:
                if isinstance(group_avg, dict):
                    hrf_est = group_avg.get('hrf_est') or group_avg.get('groupaverage') or group_avg.get('data')
                    for key, value in group_avg.items():
                        if hrf_est is None and hasattr(value, 'dims'):
                            hrf_est = value
                            break
                else:
                    # If it's an xarray Dataset, extract the data variable
                    if hasattr(group_avg, 'data_vars'):
                        # It's an xarray Dataset, extract the data variable
                        if 'hrf_est' in group_avg.data_vars:
                            hrf_est = group_avg['hrf_est']
                        elif 'group_average' in group_avg.data_vars:
                            hrf_est = group_avg['group_average']
                        elif 'hrf_estimate' in group_avg.data_vars:
                            hrf_est = group_avg['hrf_estimate']
                        else:
                            # Use the first data variable
                            first_var = list(group_avg.data_vars.keys())[0]
                            hrf_est = group_avg[first_var]
                    else:
                        # It's already a DataArray
                        hrf_est = group_avg
                    
                if hrf_est is not None and hasattr(hrf_est, 'coords') and 'trial_type' in hrf_est.coords:
                    self.hrf_available_stim_types = list(hrf_est.coords['trial_type'].values)
                else:
                    self.hrf_available_stim_types = []
            else:
                self.hrf_available_stim_types = []
        elif hasattr(self, 'hrf_data') and self.hrf_data is not None:
            hrf_available = True
            # Get available stimulus types from HRF data
            hrf_est = self.hrf_data.get('hrf_est')
            if hrf_est is not None and 'trial_type' in hrf_est.coords:
                self.hrf_available_stim_types = list(hrf_est.coords['trial_type'].values)
            else:
                self.hrf_available_stim_types = []
        else:
            self.hrf_available_stim_types = []
        
        if hrf_available:
            self.hrf_view.setEnabled(True)
        else:
            self.hrf_view.setEnabled(False)
            self.hrf_view.setChecked(False)

        # Reset selection and data
        self.selected = []
        self.snirfData = None
        if redraw_optodes:
            self._draw_optodes()

    def _draw_optodes(self):
        if self.optodes_drawn:
            return

        self._optode_ax.clear()

        # First, draw channel lines (bottom layer, zorder=1)
        # Drawing them first ensures they're underneath everything else
        self.channel_lines = []
        for i_ch in range(self.no_channels):
            si = self.src_idx[i_ch]
            di = self.det_idx[i_ch]

            line, = self._optode_ax.plot(
                [self.sx[si], self.dx[di]],
                [self.sy[si], self.dy[di]],
                "-",
                color=[0.8, 0.8, 0.8],
                zorder=1,
                picker=True,  # Enable picking but with very precise detection
                pickradius=1,  # Very small radius for precise line selection
                linewidth=2  # Make lines slightly thicker for easier clicking
            )
            # Store the line with its channel index
            line.channel_index = i_ch
            self.channel_lines.append(line)

        # Second, draw the invisible picker scatter (middle layer, zorder=10)
        # This is drawn on top of lines to prioritize optode clicks
        self.picker = self._optode_ax.scatter(
            self.sdx,
            self.sdy,
            color=[[0, 0, 0, 0]] * (len(self.sx) + len(self.dx)),
            zorder=10,  # Higher zorder to prioritize over channel lines
            picker=6,   # Reduced from 8 for more precise clicking
        )

        # Third, draw visible optode circles (top layer, zorder=15)
        self.optodes = self._optode_ax.scatter(
            self.sdx,
            self.sdy,
            color=["r"] * len(self.sx) + ["b"] * len(self.dx),
            zorder=15,  # Higher than channel lines (1) and picker (10)
            visible=False,
        )

        # Finally, draw optode labels (highest layer, zorder=20)
        for idx, source in enumerate(self.sPos.label):
            self.src_label[idx] = self._optode_ax.text(
                self.sx[idx],
                self.sy[idx],
                f"{source.values}",
                color="r",
                fontsize=8,
                ha="center",
                va="center",
                zorder=20,  # Highest z-order so text is always visible
                clip_on=True,
            )

        for idx, detector in enumerate(self.dPos.label):
            self.det_label[idx] = self._optode_ax.text(
                self.dx[idx],
                self.dy[idx],
                f"{detector.values}",
                color="b",
                fontsize=8,
                ha="center",
                va="center",
                zorder=20,  # Highest z-order so text is always visible
                clip_on=True,
            )

        self.optodes_drawn = True

        self._optode_ax.set_aspect("equal")
        self._optode_ax.axis("off")
        self._optode_ax.figure.canvas.draw()

    def _shift_is_pressed(self, event):
        if (not self.shift_pressed) and event.key == "shift":
            self.shift_pressed = True
        else:
            return

    def _shift_is_released(self, event):
        if self.shift_pressed and event.key == "shift":
            self.shift_pressed = False
        else:
            return

    def _optode_picked(self, event):
        if self.ts.currentItem() is None or self.ts.currentItem().text() == "None":
            return

        # Debug: Print what was clicked
        # print(f"Clicked on: {type(event.artist)}, has channel_index: {hasattr(event.artist, 'channel_index')}")
        
        # Check if an optode was clicked (prioritize optodes over channel lines)
        if event.artist == self.picker:
            self._optode_scatter_picked(event)
            return  # Exit early to prevent line selection
        # Check if a channel line was clicked
        elif hasattr(event.artist, 'channel_index'):
            self._channel_line_picked(event)
            return  # Exit early to prevent optode selection

    def _channel_line_picked(self, event):
        """Handle clicking on channel lines for direct channel selection"""
        channel_idx = event.artist.channel_index
        
        # Check if Shift key is held down
        shift_pressed = event.mouseevent.key == 'shift'
        
        if shift_pressed:
            # Shift+click: Toggle channel selection
            if channel_idx in self.selected_channels:
                self.selected_channels.remove(channel_idx)
            else:
                self.selected_channels.append(channel_idx)
        else:
            # Regular click: Clear previous selection and select only this channel
            self.selected_channels = [channel_idx]
        
        self._draw_timeseries()
        
        # Save GUI state after channel selection
        self._save_gui_state()

    def _optode_scatter_picked(self, event):
        """Handle clicking on optodes - toggles channels from optode in selection"""
        if not hasattr(self, 'snirfData') or self.snirfData is None:
            return
            
        N = len(event.ind)
        if not N:
            return

        # Click location
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        distances = np.hypot(x - self.sdx[event.ind], y - self.sdy[event.ind])
        indmin = distances.argmin()
        dataind = event.ind[indmin]

        # Get channels connected to this optode
        opt_label = self.opt_label[dataind]
        
        if "S" in opt_label:
            # Source optode - find all channels with this source
            channels_from_optode = self.snirfData.source[
                self.snirfData.source == opt_label
            ].channel.values.tolist()
        elif "D" in opt_label:
            # Detector optode - find all channels with this detector
            channels_from_optode = self.snirfData.detector[
                self.snirfData.detector == opt_label
            ].channel.values.tolist()
        else:
            channels_from_optode = []
            
        # Convert channel names to indices
        channel_indices = []
        for chan in channels_from_optode:
            chan_idx = np.where(self.snirfData.channel.values == chan)[0]
            if len(chan_idx) > 0:
                channel_indices.append(chan_idx[0])
        
        # Check if Shift key is held down
        shift_pressed = event.mouseevent.key == 'shift'
        
        if shift_pressed:
            # Shift+click: Toggle channels from this optode
            already_selected = any(idx in self.selected_channels for idx in channel_indices)
            
            if already_selected:
                # Remove all channels from this optode
                for idx in channel_indices:
                    if idx in self.selected_channels:
                        self.selected_channels.remove(idx)
            else:
                # Add all channels from this optode
                for idx in channel_indices:
                    if idx not in self.selected_channels:
                        self.selected_channels.append(idx)
        else:
            # Regular click: Clear previous selection and select only channels from this optode
            self.selected_channels = channel_indices.copy()
        
        # Clear old optode selection method - we now use unified channel selection
        self.selected = []
        
        self._draw_timeseries()
        
        # Save GUI state after optode selection
        self._save_gui_state()

    def _toggle_circles(self):
        if self.opt2circ.isChecked():
            for idx, source in enumerate(self.sPos.label):
                self.src_label[idx].set_visible(False)
            for idx, detector in enumerate(self.dPos.label):
                self.det_label[idx].set_visible(False)

            self.optodes.set_visible(True)
        else:
            for idx, source in enumerate(self.sPos.label):
                self.src_label[idx].set_visible(True)
            for idx, detector in enumerate(self.dPos.label):
                self.det_label[idx].set_visible(True)

            self.optodes.set_visible(False)

        self._optode_ax.figure.canvas.draw()
    
    def _preserve_zoom_changed(self):
        """Handle preserve axis zoom checkbox state change"""
        if self.preserve_axis_zoom.isChecked():
            # Save current axis limits when checkbox is enabled
            self.preserved_xlim = self._dataTimeSeries_ax.get_xlim()
            self.preserved_ylim = self._dataTimeSeries_ax.get_ylim()
            # Enable the auto scale Y-axis option
            self.auto_scale_y.setEnabled(True)
            print(f"Preserve zoom enabled. Saved limits: x={self.preserved_xlim}, y={self.preserved_ylim}")
        else:
            # Clear saved limits when checkbox is disabled
            self.preserved_xlim = None
            self.preserved_ylim = None
            # Disable the auto scale Y-axis option
            self.auto_scale_y.setEnabled(False)
            print("Preserve zoom disabled. Cleared saved limits.")
        
        # Save GUI state after preserve zoom setting change
        self._save_gui_state()
    
    def _on_xlims_change(self, event_ax):
        """Callback when axis limits change (e.g., after zooming)"""
        if self.preserve_axis_zoom.isChecked() and event_ax == self._dataTimeSeries_ax:
            # Update preserved limits when user zooms
            self.preserved_xlim = self._dataTimeSeries_ax.get_xlim()
            self.preserved_ylim = self._dataTimeSeries_ax.get_ylim()
            print(f"Axis limits updated: x={self.preserved_xlim}, y={self.preserved_ylim}")

    def _toggle_hrf_view(self):
        """Toggle between time series and HRF display"""
        if self.hrf_view.isChecked():
            # When enabling HRF view, select all available HRF stimuli
            if hasattr(self, 'hrf_available_stim_types') and self.hrf_available_stim_types:
                self.selected_stim_types = set(self.hrf_available_stim_types)
                self._update_stim_button_text()
            
            # Automatically switch to concentration timeseries to show HbO/HbR
            # Find a concentration-based timeseries (with chromo dimension)
            conc_ts_found = False
            if hasattr(self, 'timeseries_keys'):
                for ts_name in self.timeseries_keys:
                    if 'conc' in ts_name.lower():
                        # Found concentration data, switch to it
                        self.ts.blockSignals(True)
                        for i in range(self.ts.count()):
                            if self.ts.item(i).text() == ts_name:
                                self.ts.setCurrentRow(i)
                                conc_ts_found = True
                                break
                        self.ts.blockSignals(False)
                        # Trigger the change manually
                        if conc_ts_found:
                            self._ts_changed(ts_name)
                        break
            
            if not conc_ts_found:
                self.statbar.showMessage("Warning: No concentration data found. HRF view requires concentration timeseries.")
            
            # Chromophore selection is handled by wv widget (no separate checkboxes)
            self.launch_plot_probe_btn.setEnabled(True)
            # If group average is checked, disable subject/run dropdowns
            if self.hrf_group_avg.isChecked():
                self.subj.setEnabled(False)
                self.run.setEnabled(False)
        else:
            # When disabling HRF view, clear all stimulus selections
            self.selected_stim_types = set()
            self._update_stim_button_text()
            # Chromophore selection is handled by wv widget (no separate checkboxes)
            self.launch_plot_probe_btn.setEnabled(False)
            # Re-enable subject/run dropdowns
            self.subj.setEnabled(True)
            self.run.setEnabled(True)
        
        # Update file colors to reflect the new view mode
        self._update_all_file_colors()
        self._update_combobox_selection_colors()
        
        self._draw_timeseries()
        
        # Save GUI state after HRF view toggle
        self._save_gui_state()
    
    def _hrf_group_avg_changed(self):
        """Handle changes to HRF group average checkbox"""
        print(f"Group average checkbox changed to: {self.hrf_group_avg.isChecked()}")
        if self.hrf_group_avg.isChecked():
            # Disable subject and run selectors when group average is active
            self.subj.setEnabled(False)
            self.run.setEnabled(False)
        else:
            # Re-enable subject and run selectors
            self.subj.setEnabled(True)
            self.run.setEnabled(True)
        
        # Redraw if HRF view is active
        if self.hrf_view.isChecked():
            print("Calling _draw_timeseries from _hrf_group_avg_changed")
            self._draw_timeseries()
            # Save GUI state after HRF group average change
            self._save_gui_state()
    
    def _launch_plot_probe(self):
        """Launch the Plot Probe GUI with current HRF data"""
        print("Launching Plot Probe GUI...")
        
        # Get HRF data based on group average checkbox state
        if self.hrf_group_avg.isChecked():
            print("Loading group average HRF data for Plot Probe")
            hrf_data = self._load_group_average_hrf()
            if hrf_data is None:
                self.statbar.showMessage("No group average HRF data found")
                return
        else:
            print("Using individual HRF data for Plot Probe")
            # Use the already-loaded hrf_data from the current subject/run
            if not hasattr(self, 'hrf_data') or self.hrf_data is None:
                self.statbar.showMessage("No HRF data found for current selection")
                return
            hrf_data = self.hrf_data
            if isinstance(hrf_data, dict):
                pass
            elif hasattr(hrf_data, 'dims'):
                pass
        
        # Extract the actual xarray DataArray from the loaded data
        blockaverage = None
        stderr = None
        
        if hasattr(hrf_data, 'data_vars'):
            # It's an xarray Dataset, extract the data variable
            print(f"HRF data is a Dataset with variables: {list(hrf_data.data_vars.keys())}")
            if 'hrf_est' in hrf_data.data_vars:
                blockaverage = hrf_data['hrf_est']
            elif 'group_average' in hrf_data.data_vars:
                blockaverage = hrf_data['group_average']
            elif 'hrf_estimate' in hrf_data.data_vars:
                blockaverage = hrf_data['hrf_estimate']
            else:
                # Use the first data variable
                first_var = list(hrf_data.data_vars.keys())[0]
                blockaverage = hrf_data[first_var]
                print(f"Using first data variable: {first_var}")
        elif hasattr(hrf_data, 'dims'):
            # It's already an xarray DataArray
            print("HRF data is already a DataArray")
            blockaverage = hrf_data
        elif isinstance(hrf_data, dict):
            print("HRF data is a dict, extracting hrf_est")
            blockaverage = hrf_data.get('hrf_est') 
            if blockaverage is None:
                blockaverage = hrf_data.get('groupaverage')
            if blockaverage is None:
                blockaverage = hrf_data.get('data')
            if blockaverage is None:
                # Look for first xarray object in dict
                for key, value in hrf_data.items():
                    if hasattr(value, 'dims'):
                        blockaverage = value
                        print(f"Found xarray data in key: {key}")
                        break
        
        # Extract standard error from total_stderr (group average) or mse_t (individual HRF)
        # This runs for all hrf_data types (Dataset, DataArray, or dict)
        if hasattr(hrf_data, 'data_vars'):
            pass
        if 'total_stderr' in hrf_data:
            # Group average data has total_stderr already computed
            stderr = hrf_data['total_stderr']
            print(f"Found total_stderr with dims: {stderr.dims}")
        elif 'mse_t' in hrf_data:
            # Individual HRF data has mse_t which we convert to stderr
            import numpy as np
            mse_t = hrf_data['mse_t']
            print(f"Found mse_t with dims: {mse_t.dims}")
            
            # Compute standard error: stderr = sqrt(mse_t)
            stderr = np.sqrt(mse_t)
            
            # Transpose to match blockaverage dimensions if needed
            # blockaverage dims: ('trial_type', 'channel', 'chromo', 'time')
            # mse_t dims: ('trial_type', 'time', 'channel', 'chromo')
            if blockaverage is not None and hasattr(blockaverage, 'dims'):
                target_dims = blockaverage.dims
                if stderr.dims != target_dims:
                    stderr = stderr.transpose(*target_dims)
                    print(f"Transposed stderr to dims: {stderr.dims}")
            
            print(f"Standard error computed and ready to pass to plot_probe")
        
        if blockaverage is None or not hasattr(blockaverage, 'dims'):
            self.statbar.showMessage("Invalid HRF data format")
            return
        
        # Get geometry from snirfRec
        if self.snirfRec is None:
            self.statbar.showMessage("No recording loaded - geometry not available")
            return
        
        if not hasattr(self.snirfRec, 'geo2d') or not hasattr(self.snirfRec, 'geo3d'):
            self.statbar.showMessage("Recording missing geometry information")
            return
        
        geo2d = self.snirfRec.geo2d
        geo3d = self.snirfRec.geo3d
        
        # Import and launch the plot_probe_gui
        try:
            from plot_probe_gui import _MAIN_GUI
            print(f"Launching with HRF data dims: {blockaverage.dims}")
            if stderr is not None:
                print(f"Passing stderr with dims: {stderr.dims}")
            else:
                print(f"WARNING: No stderr data to pass to plot_probe_gui")
            
            # Create the Plot Probe GUI as a child window (don't create new QApplication)
            self.plot_probe_window = _MAIN_GUI(
                snirfData=blockaverage, 
                stderr=stderr, 
                geo2d=geo2d, 
                geo3d=geo3d
            )
            self.plot_probe_window.show()
            
        except Exception as e:
            print(f"Error launching Plot Probe GUI: {e}")
            import traceback
            traceback.print_exc()
            self.statbar.showMessage(f"Error launching Plot Probe: {str(e)}")

    def _find_image_recon_dataset(self):
        """Locate and load the Xs_*.nc image reconstruction result matching the
        currently selected subject/run (or group average), same file this
        recording's data would come from for _launch_image_recon.

        Returns (img_data, hrf_est, task_name) on success, or (None, None, msg)
        with msg an explanation to show in the status bar.
        """
        current_subject = self.subj.currentText() if self.subj.currentText() != "None" else None
        current_run = self.run.currentText() if self.run.currentText() != "None" else None

        task_name = None
        base_path = None
        if current_subject and current_run:
            file_info = self.file_map.get(current_subject, {}).get(current_run, {})
            pkl_path = None
            if file_info:
                pkl_path = (file_info.get('preprocessing_snirfData') or
                           file_info.get('pkl_path') or
                           file_info.get('snirf_path'))
            if pkl_path:
                task_match = re.search(r'task-([^_/\\]+)', pkl_path)
                task_name = task_match.group(1) if task_match else None
                derivatives_match = re.search(r'(.*[/\\]derivatives[/\\]cedalion[/\\][^/\\]+)', pkl_path)
                if derivatives_match:
                    base_path = derivatives_match.group(1).replace('\\', '/')

        if not task_name or not base_path:
            return None, None, "Cannot determine task name or base path. Please load data first."

        image_results_dir = os.path.join(base_path, 'Outputs', 'image_results')
        if not os.path.exists(image_results_dir):
            return None, None, f"Image results directory not found: {image_results_dir}"

        import glob
        if self.hrf_group_avg.isChecked():
            files = glob.glob(os.path.join(image_results_dir, f"Xs_groupavg_{task_name}_*.nc"))
            if not files:
                return None, None, f"No group average image recon found for task {task_name}"
        else:
            if not current_subject:
                return None, None, "No subject selected"
            subj_dir = os.path.join(image_results_dir, current_subject)
            files = glob.glob(os.path.join(subj_dir, f"Xs_{current_subject}_{task_name}_*.nc"))
            if not files:
                return None, None, f"No image recon found for {current_subject}, task {task_name}"

        img_data = xr.open_dataset(files[0])
        if 'Xs' in img_data.data_vars:
            hrf_est = img_data['Xs']
        elif 'hrf_est' in img_data.data_vars:
            hrf_est = img_data['hrf_est']
        elif 'group_average' in img_data.data_vars:
            hrf_est = img_data['group_average']
        else:
            return None, None, (
                f"Invalid image reconstruction data format. Variables found: "
                f"{list(img_data.data_vars.keys())}"
            )

        return img_data, hrf_est, task_name

    def _launch_parcel_viewer(self):
        """Launch the Brain Parcel Viewer, mapping the current image
        reconstruction result (Xs_*.nc, vertex-space HRF) onto Schaefer
        parcels so individual parcels' HRF curves can be inspected."""
        img_data, hrf_est, msg = self._find_image_recon_dataset()
        if img_data is None:
            self.statbar.showMessage(msg)
            return

        if 'vertex' not in hrf_est.dims:
            self.statbar.showMessage(
                "Image reconstruction data is not in vertex space; cannot map to parcels"
            )
            return

        head_model = "icbm152"
        if self.snakemake_config and 'image_recon' in self.snakemake_config:
            head_model = self.snakemake_config['image_recon'].get('head_model', 'ICBM152').lower()

        try:
            from parcel_viewer_gui import ParcelData, ParcelViewerWindow
            data = ParcelData(head_model=head_model)
            # keep a reference so the window isn't garbage-collected once this returns
            self.parcel_viewer_window = ParcelViewerWindow(data, hrf_data=hrf_est)
            self.parcel_viewer_window.show()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.statbar.showMessage(f"Error launching Brain Parcel Viewer: {str(e)}")

    def _launch_image_recon(self):
        """Launch the Image Reconstruction viewer with current task data"""
        print("Launching Image Reconstruction...")
        
        # Get the current task name from the current data file path
        task_name = None
        base_path = None
        
        # Get current subject and run from combo boxes
        current_subject = self.subj.currentText() if self.subj.currentText() != "None" else None
        current_run = self.run.currentText() if self.run.currentText() != "None" else None
        
        
        # Try to extract from the file_map using current subject/run
        if current_subject and current_run:
            file_info = self.file_map.get(current_subject, {}).get(current_run, {})
            # Try different possible keys for the path
            pkl_path = None
            if file_info:
                pkl_path = (file_info.get('preprocessing_snirfData') or 
                           file_info.get('pkl_path') or 
                           file_info.get('snirf_path'))
            
            if pkl_path:
                task_match = re.search(r'task-([^_/\\]+)', pkl_path)
                task_name = task_match.group(1) if task_match else None
                
                # Get base path - extract up to config folder
                # Path structure: .../derivatives/cedalion/{config_name}/...
                # We want everything up to and including {config_name}
                derivatives_match = re.search(r'(.*[/\\]derivatives[/\\]cedalion[/\\][^/\\]+)', pkl_path)
                if derivatives_match:
                    base_path = derivatives_match.group(1)
                    # Normalize to forward slashes for consistency
                    base_path = base_path.replace('\\', '/')
                else:
                    pass
        
        
        if not task_name or not base_path:
            msg = "Cannot determine task name or base path. Please load data first."
            print(f"ERROR: {msg}")
            self.statbar.showMessage(msg)
            return
        
        print(f"Task: {task_name}, Base path: {base_path}")
        
        # Check if image_results directory exists
        image_results_dir = os.path.join(base_path, 'Outputs', 'image_results')
        if not os.path.exists(image_results_dir):
            msg = f"Image results directory not found: {image_results_dir}"
            print(f"ERROR: {msg}")
            self.statbar.showMessage(msg)
            return
        
        
        # Load image reconstruction data
        img_data = None
        if self.hrf_group_avg.isChecked():
            # Load group average image recon
            print("Loading group average image reconstruction")
            # Look for .nc files
            pattern_nc = f"Xs_groupavg_{task_name}_*.nc"
            import glob
            files = glob.glob(os.path.join(image_results_dir, pattern_nc))
            
            if files:
                img_path = files[0]
                print(f"Loading: {img_path}")
                img_data = xr.open_dataset(img_path)
            else:
                msg = f"No group average image recon found for task {task_name}"
                print(f"ERROR: {msg}")
                self.statbar.showMessage(msg)
                return
        else:
            # Load individual subject image recon
            if not current_subject:
                self.statbar.showMessage("No subject selected")
                return
            print(f"Loading image reconstruction for {current_subject}")
            subj_dir = os.path.join(image_results_dir, current_subject)
            if not os.path.exists(subj_dir):
                self.statbar.showMessage(f"No image results for {current_subject}")
                return
            
            pattern = f"Xs_{current_subject}_{task_name}_*.nc"
            import glob
            files = glob.glob(os.path.join(subj_dir, pattern))
            if files:
                img_path = files[0]
                print(f"Loading: {img_path}")
                img_data = xr.open_dataset(img_path)
            else:
                msg = f"No image recon found for {current_subject}, task {task_name}"
                print(f"ERROR: {msg}")
                self.statbar.showMessage(msg)
                return
        
        
        # img_data is now an xarray Dataset (from netCDF)
        if img_data is None:
            msg = "Invalid image reconstruction data format. Data is None."
            print(f"ERROR: {msg}")
            self.statbar.showMessage(msg)
            return
        
        # Check for required data variables in the Dataset
        if not isinstance(img_data, xr.Dataset):
            msg = f"Invalid image reconstruction data format. Expected xarray.Dataset, got {type(img_data)}."
            print(f"ERROR: {msg}")
            self.statbar.showMessage(msg)
            return
        
        has_data = 'Xs' in img_data.data_vars or 'hrf_est' in img_data.data_vars or 'group_average' in img_data.data_vars
        if not has_data:
            msg = f"Invalid image reconstruction data format. Variables found: {list(img_data.data_vars.keys())}, but none of 'Xs', 'hrf_est', or 'group_average' found."
            print(f"ERROR: {msg}")
            self.statbar.showMessage(msg)
            return
        
        # Extract the HRF estimate (try different variable names)
        if 'Xs' in img_data.data_vars:
            hrf_est = img_data['Xs']
        elif 'hrf_est' in img_data.data_vars:
            hrf_est = img_data['hrf_est']
        elif 'group_average' in img_data.data_vars:
            hrf_est = img_data['group_average']
        else:
            error_msg = f"Invalid image reconstruction data format. Variables found: {list(img_data.data_vars.keys())}, but none of 'Xs', 'hrf_est', or 'group_average' found."
            print(f"ERROR: {error_msg}")
            QtWidgets.QMessageBox.critical(self, "Error", error_msg)
            return
            
        print(f"Loaded image recon with dims: {hrf_est.dims}, shape: {hrf_est.shape}")
        
        # Determine if this is group average data
        is_group_avg = 'group_average' in img_data.data_vars
        
        # Extract time bounds if time dimension exists
        time_bounds = None
        if 'time' in hrf_est.dims:
            time_values = hrf_est.time.values
            min_time = float(time_values.min())
            max_time = float(time_values.max())
            time_bounds = (min_time, max_time)
        else:
            pass
        
        # Get available trial types
        if 'trial_type' in hrf_est.dims:
            trial_types = [str(tt) for tt in hrf_est.trial_type.values]
        else:
            trial_types = [task_name]
        
        
        # Create options dialog with group average flag and time bounds
        dialog = ImageReconDialog(trial_types, is_group_avg=is_group_avg, 
                                 time_bounds=time_bounds, parent=self)
        
        # Set up callback for when Launch Viewer is clicked
        def launch_with_options(options):
            print(f"User selected options: {options}")
            self._perform_image_recon_launch(hrf_est, img_data, task_name, options, 
                                            img_path, current_subject, is_group_avg)
        
        dialog.launch_callback = launch_with_options
        dialog.show()  # Non-modal, stays open
    
    def _perform_image_recon_launch(self, hrf_est, img_data, task_name, options, 
                                    img_path, current_subject, is_group_avg):
        """Perform the actual image reconstruction launch with the given options"""
        # Load head model from config or default to icbm152
        try:
            import cedalion.dot as dot
            
            # Try to get head model from config
            head_model = "icbm152"  # Default
            if self.snakemake_config and 'image_recon' in self.snakemake_config:
                config_head_model = self.snakemake_config['image_recon'].get('head_model', 'ICBM152')
                head_model = config_head_model.lower()  # Convert to lowercase
            else:
                pass
            
            head = dot.get_standard_headmodel(head_model)
        except Exception as e:
            msg = f"Error loading head model: {str(e)}"
            print(f"ERROR: {msg}")
            import traceback
            traceback.print_exc()
            self.statbar.showMessage(msg)
            return
        
        # Extract selected metric
        metric = options.get('metric', 'mag')
        
        # Determine if this is group average data
        is_group_avg = 'group_average' in img_data.data_vars
        
        # Prepare data based on selected metric
        if metric == 'mag':
            # Use HRF magnitude (already loaded as hrf_est)
            data_to_viz = hrf_est
            
        elif metric == 'std_err':
            # Calculate standard error: sqrt(mse)
            import numpy as np
            if is_group_avg and 'total_stderr' in img_data.data_vars:
                # Group average: sqrt(total_stderr)
                data_to_viz = np.sqrt(img_data['total_stderr'])
            elif 'mse_t' in img_data.data_vars:
                # Subject level: sqrt(mse_t)
                data_to_viz = np.sqrt(img_data['mse_t'])
            else:
                msg = "Cannot compute std_err: no mse data found"
                print(f"ERROR: {msg}")
                QtWidgets.QMessageBox.critical(self, "Error", msg)
                return
                
        elif metric == 't_stat':
            # T-statistic: mag / std_err
            import numpy as np
            if is_group_avg and 'tstat' in img_data.data_vars:
                # Group average: use stored t-stat
                data_to_viz = img_data['tstat']
            else:
                # Subject level: calculate mag / std_err
                if 'mse_t' in img_data.data_vars:
                    std_err = np.sqrt(img_data['mse_t'])
                    data_to_viz = hrf_est / std_err
                else:
                    msg = "Cannot compute t_stat: no mse data found"
                    print(f"ERROR: {msg}")
                    QtWidgets.QMessageBox.critical(self, "Error", msg)
                    return
                    
        elif metric == 'std_err_btwn_subjs':
            # Between-subjects standard error (group only)
            import numpy as np
            if is_group_avg and 'mse_weighted_btwn_subjs' in img_data.data_vars:
                data_to_viz = np.sqrt(img_data['mse_weighted_btwn_subjs'])
            else:
                msg = "std_err_btwn_subjs is only available for group average data"
                print(f"ERROR: {msg}")
                QtWidgets.QMessageBox.critical(self, "Error", msg)
                return
                
        elif metric == 'std_err_within_subjs':
            # Within-subjects standard error (group only)
            import numpy as np
            if is_group_avg and 'mse_mean_within_subj' in img_data.data_vars:
                data_to_viz = np.sqrt(img_data['mse_mean_within_subj'])
            else:
                msg = "std_err_within_subjs is only available for group average data"
                print(f"ERROR: {msg}")
                QtWidgets.QMessageBox.critical(self, "Error", msg)
                return
        else:
            msg = f"Unknown metric: {metric}"
            print(f"ERROR: {msg}")
            QtWidgets.QMessageBox.critical(self, "Error", msg)
            return
        
        # Select the chosen trial_type
        if 'trial_type' in data_to_viz.dims and len(data_to_viz.trial_type) > 0:
            X_ts = data_to_viz.sel(trial_type=options['trial_type'])
        else:
            X_ts = data_to_viz
        
        # Transpose to expected format: (vertex, chromo, time)
        expected_dims = ('vertex', 'chromo', 'time')
        if X_ts.dims != expected_dims:
            try:
                X_ts = X_ts.transpose(*expected_dims)
            except Exception as e:
                print(f"WARNING: Could not transpose to {expected_dims}: {e}")
        
        # Handle time series data
        SAVE = options['save']
        mean_over_time = options.get('mean_over_time', False)
        
        if 'time' in X_ts.dims and X_ts.sizes['time'] > 1:
            if not SAVE and not mean_over_time:
                # Warn user that save option needs to be selected or mean over time
                QtWidgets.QMessageBox.warning(
                    self,
                    "Time Series Data",
                    "Time series data detected. Please either:\n"
                    "1. Enable 'Save output (PNG/GIF)' to create an animation, OR\n"
                    "2. Enable 'Mean over time range' to display averaged data"
                )
                print("ERROR: Time series data requires either SAVE=True or mean_over_time=True")
                self.statbar.showMessage("Please enable Save or Mean over time range for time series data")
                return
            elif mean_over_time and not SAVE:
                # Compute mean over time range
                import numpy as np
                time_range = options.get('time_range')
                if time_range:
                    start_time, end_time, _ = time_range
                    # Select time slice
                    X_ts = X_ts.sel(time=slice(start_time, end_time))
                else:
                    pass
                
                # Compute mean over time dimension
                X_ts = X_ts.mean(dim='time')
        
        # Extract chromophore from view_type (e.g., "hbo_brain" -> "HbO")
        view_type = options['view_type']
        if 'hbo' in view_type:
            chromo = 'HbO'
        elif 'hbr' in view_type:
            chromo = 'HbR'
        else:
            chromo = 'HbO'  # Default fallback
        
        # Calculate or use custom color limits
        if options['clim'] is None:
            import numpy as np
            
            # Check if metric should use symmetric or non-negative scaling
            symmetric_metrics = ['mag', 't_stat']
            if metric in symmetric_metrics:
                # Symmetric color scale (e.g., -scl to +scl)
                scl = np.percentile(np.abs(X_ts.sel(chromo=chromo)).values, 99)
                clim = (-scl, scl)
            else:
                # Non-negative color scale for std_err types (0 to +scl)
                scl = np.percentile(X_ts.sel(chromo=chromo).values, 99)
                clim = (0, scl)
        else:
            clim = options['clim']
        
        # Prepare time range
        time_range = options['time_range']
        if SAVE:
            # Construct structured save path
            # Extract pkl basename without extension
            pkl_basename = os.path.basename(img_path)
            # Remove .pkl.gz or .pkl extension
            if pkl_basename.endswith('.pkl.gz'):
                pkl_basename = pkl_basename[:-7]  # Remove .pkl.gz
            elif pkl_basename.endswith('.pkl'):
                pkl_basename = pkl_basename[:-4]  # Remove .pkl
            
            # Build directory structure
            plots_dir = os.path.join(self.path_to_data, 'plots', 'image_results')
            
            if is_group_avg:
                # Group: plots/image_results/Xs_groupavg_task_timestamp/
                save_dir = os.path.join(plots_dir, pkl_basename)
            else:
                # Subject: plots/image_results/sub-XX/Xs_sub-XX_task_timestamp/
                save_dir = os.path.join(plots_dir, current_subject, pkl_basename)
            
            # Create directory structure
            os.makedirs(save_dir, exist_ok=True)
            
            # Prepend directory to user's filename
            full_filename = os.path.join(save_dir, options['filename'])
            
            # For animation, convert time range to units
            try:
                import pint
                import pint_xarray
                # Convert to units
                time_range = (time_range[0], time_range[1], time_range[2]) * pint.Unit('second')
            except Exception as e:
                print(f"WARNING: Could not create time range with units: {e}")
                time_range = None
        else:
            # For static display, time_range is not used (either warned earlier or mean already computed)
            full_filename = None
            time_range = None
        
        # Launch the image reconstruction viewer
        try:
            # Determine title string
            if options['title_str'] is not None:
                title_str = options['title_str']
            else:
                # Auto-generate title based on metric
                metric_labels = {
                    'mag': f'{chromo} / µM',
                    'std_err': f'{chromo} std_err / µM',
                    't_stat': f'{chromo} t-stat',
                    'std_err_btwn_subjs': f'{chromo} std_err (between subjs) / µM',
                    'std_err_within_subjs': f'{chromo} std_err (within subjs) / µM'
                }
                metric_label = metric_labels.get(metric, f'{chromo} {metric}')
                title_str = f'{metric_label} - {task_name}'
                if is_group_avg:
                    title_str += ' (Group Average)'
            
            # Check if geo3d data is available
            geo3d_plot = None
            if options['show_geo3d'] and 'geo3d' in img_data.data_vars:
                geo3d_plot = img_data['geo3d']
            else:
                pass
            
            if options['multi_view']:
                # Multi-view mode: show all 6 views
                from cedalion.vis.anatomy import image_recon_multi_view
                
                print(f"  - X_ts shape: {X_ts.shape}, dims: {X_ts.dims}")
                print(f"  - view_type: {options['view_type']}")
                print(f"  - cmap: {options['cmap']}")
                print(f"  - clim: {clim}")
                print(f"  - SAVE: {SAVE}")
                print(f"  - fps: {options['fps']}")
                print(f"  - wdw_size: {options['wdw_size']}")
                print(f"  - geo3d_plot: {'Yes' if geo3d_plot is not None else 'No'}")
                
                # Show saving message
                saving_msg = None
                if SAVE:
                    saving_msg = QtWidgets.QMessageBox(self)
                    saving_msg.setWindowTitle("Saving")
                    saving_msg.setText("Saving visualization, please wait...")
                    saving_msg.setStandardButtons(QtWidgets.QMessageBox.NoButton)
                    saving_msg.setModal(False)
                    saving_msg.show()
                    QtWidgets.QApplication.processEvents()  # Force UI update
                
                try:
                    image_recon_multi_view(
                        X_ts,
                        head,
                        cmap=options['cmap'],
                        clim=clim,
                        view_type=options['view_type'],
                        title_str=title_str,
                        filename=full_filename,
                        SAVE=SAVE,
                        time_range=time_range,
                        fps=options['fps'],
                        geo3d_plot=geo3d_plot,
                        wdw_size=options['wdw_size']
                    )
                    
                    if SAVE:
                        self.statbar.showMessage(f"Saved to: {full_filename}")
                finally:
                    if saving_msg is not None:
                        saving_msg.close()
                        saving_msg.deleteLater()
            else:
                # Single-view mode: show one view at specified position
                from cedalion.vis.anatomy import image_recon
                
                print(f"  - X_ts shape: {X_ts.shape}, dims: {X_ts.dims}")
                print(f"  - view_type: {options['view_type']}")
                print(f"  - view_position: {options['view_position']}")
                print(f"  - cmap: {options['cmap']}")
                print(f"  - clim: {clim}")
                print(f"  - wdw_size: {options['wdw_size']}")
                print(f"  - geo3d_plot: {'Yes' if geo3d_plot is not None else 'No'}")
                
                # Show saving message if saving
                saving_msg = None
                if SAVE:
                    saving_msg = QtWidgets.QMessageBox(self)
                    saving_msg.setWindowTitle("Saving")
                    saving_msg.setText("Saving visualization, please wait...")
                    saving_msg.setStandardButtons(QtWidgets.QMessageBox.NoButton)
                    saving_msg.setModal(False)
                    saving_msg.show()
                    QtWidgets.QApplication.processEvents()  # Force UI update
                
                try:
                    # For single view
                    p0, surf, lab = image_recon(
                        X_ts,
                        head,
                        cmap=options['cmap'],
                        clim=clim,
                        view_type=options['view_type'],
                        view_position=options['view_position'],
                        title_str=title_str,
                        off_screen=SAVE,  # Off-screen if saving
                        show_scalar_bar=True,
                        wdw_size=options['wdw_size']
                    )
                    
                    # Add geo3d if available
                    if geo3d_plot is not None:
                        from cedalion.vis.blocks import plot_labeled_points
                        plot_labeled_points(p0, geo3d_plot)
                    
                    # Save if requested
                    if SAVE:
                        p0.screenshot(full_filename + '.png')
                        self.statbar.showMessage(f"Saved to: {full_filename}.png")
                    else:
                        p0.show()
                    
                finally:
                    if saving_msg is not None:
                        saving_msg.close()
                        saving_msg.deleteLater()
            
            self.statbar.showMessage("Image reconstruction viewer launched")
        except Exception as e:
            print(f"ERROR: Exception launching image reconstruction viewer: {e}")
            import traceback
            traceback.print_exc()
            self.statbar.showMessage(f"Error launching image recon: {str(e)}")

    def _update_stim_types(self):
        """Update available stimulus types from the current recording"""
        if hasattr(self, 'snirfRec') and self.snirfRec is not None and len(self.snirfRec.stim) > 0:
            self.available_stim_types = list(np.unique(self.snirfRec.stim.trial_type))
            self.stim_button.setEnabled(True)
            # Update button text to show selection count
            self._update_stim_button_text()
        else:
            self.available_stim_types = []
            self.stim_button.setEnabled(False)
            self.stim_button.setText("No Stimuli")

    def _update_stim_button_text(self):
        """Update the button text to show current selection"""
        if not self.selected_stim_types:
            self.stim_button.setText("Select Stimuli")
        elif len(self.selected_stim_types) == len(self.available_stim_types):
            self.stim_button.setText("All Stimuli")
        else:
            self.stim_button.setText(f"Stimuli ({len(self.selected_stim_types)})")

    def _open_stim_dialog(self):
        """Open dialog for selecting stimuli"""
        if not self.available_stim_types:
            return
            
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select Stimuli")
        dialog.setModal(True)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Add "All" and "None" buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        all_button = QtWidgets.QPushButton("All")
        none_button = QtWidgets.QPushButton("None")
        
        button_layout.addWidget(all_button)
        button_layout.addWidget(none_button)
        layout.addLayout(button_layout)
        
        # Create checkboxes for each stimulus type
        self.stim_checkboxes = {}
        for stim_type in self.available_stim_types:
            # Check if this stimulus has HRF data
            has_hrf = stim_type in self.hrf_available_stim_types
            
            # Create label with HRF indicator
            if has_hrf:
                label_text = f"{stim_type} (HRF available)"
            else:
                label_text = stim_type
            
            checkbox = QtWidgets.QCheckBox(label_text)
            checkbox.setChecked(stim_type in self.selected_stim_types)
            
            # If HRF view is enabled, only enable checkboxes for stimuli with HRF data
            if self.hrf_view.isChecked() and not has_hrf:
                checkbox.setEnabled(False)
                checkbox.setChecked(False)
            
            self.stim_checkboxes[stim_type] = checkbox
            layout.addWidget(checkbox)
        
        # OK and Cancel buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # Connect signals
        all_button.clicked.connect(lambda: self._select_all_stims(True))
        none_button.clicked.connect(lambda: self._select_all_stims(False))
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Show dialog and update selection if accepted
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            self.selected_stim_types = set()
            for stim_type, checkbox in self.stim_checkboxes.items():
                if checkbox.isChecked():
                    self.selected_stim_types.add(stim_type)
            
            self._update_stim_button_text()
            self._draw_timeseries()
            
            # Save GUI state after stimuli selection change
            self._save_gui_state()

    def _select_all_stims(self, select_all):
        """Select or deselect all stimulus checkboxes"""
        for checkbox in self.stim_checkboxes.values():
            checkbox.setChecked(select_all)
        
    def _subj_changed(self, s):
        if not s or s == "None":
            return

        # 1. Store current run and optode selection
        current_run = self.run.currentText()
        
        # 2. Get new subject and their available runs
        new_subj = self.subj.currentText()
        new_runs = self.subject_to_runs_map.get(new_subj, [])

        # 3. Block signals from run box to prevent premature updates
        self.run.blockSignals(True)
        self.run.clear()

        if new_runs:
            self.run.addItems(new_runs)
            
            # Reapply colors to the new run items
            self._reapply_run_colors(new_subj, new_runs)
            
            # 4. Try to restore the previous run selection
            if current_run in new_runs:
                self.run.setCurrentText(current_run)
            else:
                # Try to find a 'run-1' as a fallback
                run1_fallback = next((r for r in new_runs if r.startswith("run-1")), None)
                if run1_fallback:
                    self.run.setCurrentText(run1_fallback)
                # Otherwise, the first item is selected by default
        else:
            self.run.addItem("None")

        # 5. Unblock signals and manually trigger the update
        self.run.blockSignals(False)
        self._run_changed(self.run.currentText(), subject_changed=True)
        
        # Update current selection colors
        self._update_combobox_selection_colors()

    def _run_changed(self, s, subject_changed=False):
        if not s or s == "None":
            return
        
        subj_key = self.subj.currentText()
        self._update_recording_data(subj_key, s, subject_changed=subject_changed)
        
        # Update current selection colors
        self._update_combobox_selection_colors()
        
        # Save GUI state after run change
        self._save_gui_state()

    def _update_run_box(self):
        """(DEPRECATED) - This logic is now handled in _subj_changed."""
        current_subj = self.subj.currentText()
        self.run.clear()
        
        if current_subj in self.subject_to_runs_map:
            available_runs = self.subject_to_runs_map[current_subj]
            self.run.addItems(available_runs)
        else:
            self.run.addItem("None")

    def _update_recording_data(self, subj_key, run_key, subject_changed=False):
        """Loads the prepared data for a subject/run and updates the GUI."""
        
        new_prepared_data = self._get_recording(subj_key, run_key)

        if new_prepared_data is None:
            print(f"Data is None for Subject: {subj_key}, Run: {run_key}")
            self._dataTimeSeries_ax.clear()
            self._optode_ax.clear()
            self.plots.figure.canvas.draw()
            return

        # Check if data is actually new
        if self.snirfRec is new_prepared_data['snirfRec']:
            return
        
        redraw_optodes = subject_changed

        # --- Save previous selections ---
        prev_ts = None
        prev_wv = []
        prev_selected = []
        prev_selected_channels = []
        prev_selected_stims = set()
        if hasattr(self, 'ts') and self.ts.currentItem() is not None:
            prev_ts = self.ts.currentItem().text()
        if hasattr(self, 'wv'):
            prev_wv = [item.text() for item in self.wv.selectedItems()]
        prev_selected = list(self.selected) if hasattr(self, 'selected') else []
        prev_selected_channels = list(self.selected_channels) if hasattr(self, 'selected_channels') else []
        prev_selected_stims = set(self.selected_stim_types) if hasattr(self, 'selected_stim_types') else set()

        # Unpack the prepared data into the main object
        for key, value in new_prepared_data.items():
            setattr(self, key, value)

        print(f"Loaded data for Subject: {subj_key}, Run: {run_key}")
        
        # Enable image recon button if we have data
        self.image_recon_btn.setEnabled(True)
        self.parcel_viewer_btn.setEnabled(True)
        
        self._init_widgets(redraw_optodes=redraw_optodes)
        
        # Show ready status
        self.statbar.showMessage(f"✓ Ready: {subj_key}/{run_key}", 0)  # Keep until next change
        QtWidgets.QApplication.processEvents()  # Force GUI update

        # --- Restore previous selections ---
        # Timeseries
        if prev_ts and prev_ts in self.timeseries_keys:
            idx = self.ts.findItems(prev_ts, QtCore.Qt.MatchExactly)
            if idx:
                self.ts.setCurrentItem(idx[0])
        elif len(self.timeseries_keys) > 0:
            # Default to first timeseries if no previous selection
            self.ts.setCurrentRow(1)  # Row 1 since row 0 is "None"
        # Wavelength/Concentration
        if prev_wv:
            for i in range(self.wv.count()):
                item = self.wv.item(i)
                if item.text() in prev_wv:
                    item.setSelected(True)
        # Channel selection (now unified approach)
        if prev_selected_channels and max(prev_selected_channels) < self.no_channels:
            self.selected_channels = prev_selected_channels
        elif prev_selected and max(prev_selected) < len(self.opt_label):
            # Convert old optode selection to channel selection for compatibility
            self.selected_channels = []
            for opt_idx in prev_selected:
                opt_label = self.opt_label[opt_idx]
                if "S" in opt_label:
                    channels_to_add = self.snirfData.source[
                        self.snirfData.source == opt_label
                    ].channel.values.tolist()
                elif "D" in opt_label:
                    channels_to_add = self.snirfData.detector[
                        self.snirfData.detector == opt_label
                    ].channel.values.tolist()
                else:
                    channels_to_add = []
                    
                for chan in channels_to_add:
                    chan_idx = np.where(self.snirfData.channel.values == chan)[0]
                    if len(chan_idx) > 0:
                        idx = chan_idx[0]
                        if idx not in self.selected_channels:
                            self.selected_channels.append(idx)
        elif len(self.opt_label) > 0:
            # Default: select channels from first optode
            opt_label = self.opt_label[0]
            self.selected_channels = []
            if "S" in opt_label:
                channels_to_add = self.snirfData.source[
                    self.snirfData.source == opt_label
                ].channel.values.tolist()
            elif "D" in opt_label:
                channels_to_add = self.snirfData.detector[
                    self.snirfData.detector == opt_label
                ].channel.values.tolist()
            else:
                channels_to_add = []
                
            for chan in channels_to_add:
                chan_idx = np.where(self.snirfData.channel.values == chan)[0]
                if len(chan_idx) > 0:
                    self.selected_channels.append(chan_idx[0])
        
        # Clear old optode selection - we now use unified channel selection
        self.selected = []
        
        # Restore stimulus selection (only keep stimuli that exist in new data)
        if prev_selected_stims and hasattr(self, 'available_stim_types'):
            self.selected_stim_types = prev_selected_stims.intersection(set(self.available_stim_types))
            self._update_stim_button_text()

        self._draw_timeseries()

    def _wv_changed(self):
        self._draw_timeseries()
        
        # Save GUI state after wavelength change
        self._save_gui_state()

    def _ts_changed(self, s):
        # Extract data
        if s == "None":
            self.snirfData = None
            self._dataTimeSeries_ax.clear()
            return

        # Check if we have processed data with this timeseries
        if s != 'amp' and self.processed_rec and hasattr(self.processed_rec, 'timeseries') and s in self.processed_rec.timeseries:
            self.snirfData = self.processed_rec.timeseries[s]
        else:
            # Use amplitude data from SNIRF
            self.snirfData = self.snirfRec.timeseries[s]
        
        self.ts_sel = s

        # Determine wavelength/concentration
        if "wavelength" in self.snirfData.dims:
            self.wv_label.setText("Wavelength:")
            self.wv.clear()

            for i_w, wvl in enumerate(self.snirfData.wavelength.values):
                self.wv.insertItem(i_w, str(wvl))
            self.wv.setCurrentRow(0)

        elif "chromo" in self.snirfData.dims:
            self.wv_label.setText("Concentration:")
            self.wv.clear()

            for i_w, wvl in enumerate(self.snirfData.chromo.values):
                self.wv.insertItem(i_w, f"[{str(wvl)}]")
            self.wv.setCurrentRow(0)

        self._draw_timeseries()
        
        # Save GUI state after timeseries change
        self._save_gui_state()

    def _format_unit_label(self, units_value):
        """Format xarray/pint unit metadata for plot labels."""
        if units_value is None:
            return None

        units_text = str(units_value).strip()
        if not units_text or units_text == "None":
            return None

        normalized = units_text.replace(" ", "").replace("**", "^").lower()
        if normalized in ("micromolar", "um", "umol/liter"):
            return r"$\mu$M"
        if normalized in ("molar", "mol/liter"):
            return "M"
        if normalized in (
            "micromolar*millimeter",
            "micromolar*mm",
            "um*mm",
            "um*millimeter",
        ):
            return r"$\mu$M$\cdot$mm"
        if normalized == "dimensionless":
            return "A.U."

        return units_text

    def _get_data_unit_label(self, data_array):
        """Return a display label from a DataArray's pint units or saved attrs."""
        if data_array is None:
            return None

        try:
            unit_label = self._format_unit_label(data_array.pint.units)
            if unit_label:
                return unit_label
        except Exception:
            pass

        attrs = getattr(data_array, 'attrs', {})
        for key in ("units", "unit"):
            unit_label = self._format_unit_label(attrs.get(key))
            if unit_label:
                return unit_label

        return None

    def _get_concentration_unit_label(self, data_array=None):
        """Choose concentration display units from the plotted data when possible."""
        if data_array is None:
            data_array = getattr(self, 'snirfData', None)

        unit_label = self._get_data_unit_label(data_array)
        if unit_label:
            return unit_label

        return r"$\mu$M"

    def _aux_has_data(self):
        """Return True when an auxiliary time series is selected for plotting."""
        try:
            return self.aux_sel is not None and len(self.aux_sel) > 0
        except TypeError:
            return False

    def _set_aux_axis_visible(self, visible):
        """Show the right-side auxiliary axis only while aux data is plotted."""
        if not hasattr(self, '_auxTimeSeries_ax'):
            return

        if not visible:
            self._auxTimeSeries_ax.clear()
            self.auxplot = []

        self._auxTimeSeries_ax.set_visible(visible)
        self._auxTimeSeries_ax.yaxis.set_visible(visible)
        self._auxTimeSeries_ax.spines["right"].set_visible(visible)

    def _draw_aux_timeseries(self):
        """Draw selected auxiliary data and hide the right axis when none is selected."""
        self._set_aux_axis_visible(False)

        if not self._aux_has_data():
            return

        self._set_aux_axis_visible(True)
        self.auxplot = self._auxTimeSeries_ax.plot(
            self.aux_sel.time,
            self.aux_sel,
            zorder=2,
            color="r",
            alpha=0.3,
            linewidth=0.5,
        )
        self._auxTimeSeries_ax.set_ylabel(self.aux_type, rotation=270, ha="right")
        self._auxTimeSeries_ax.yaxis.set_label_position("right")

    def _update_channel_highlights(self):
        """Update the channel line highlighting on the probe display"""
        # Color palette
        chan_col = [
            "#FF0000", "#0000FF", "#00FF00", "#FF00FF", "#FFFF00", "#00FFFF", "#FF8000", "#8000FF",
            "#FF0080", "#00FF80", "#0080FF", "#FF6B6B", "#4ECDC4", "#FFB000", "#117733", "#DC267F",
            "#332288", "#882255", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#45B7D1",
            "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9", "#F8C471"
        ]
        
        # Reset all channel lines to default first
        for line in self.channel_lines:
            line.set_color([0.8, 0.8, 0.8])
            line.set_linewidth(2)
            
        # Highlight selected channels
        for i, chan_idx in enumerate(self.selected_channels):
            if chan_idx < len(self.channel_lines):
                # Use a smart color assignment to avoid conflicts with nearby channels
                # If we have fewer selected channels than colors, use sequential assignment
                if len(self.selected_channels) <= len(chan_col):
                    color_idx = i
                else:
                    # For many channels, use the channel index with better distribution
                    color_idx = chan_idx % len(chan_col)
                
                self.channel_lines[chan_idx].set_color(chan_col[color_idx])
                self.channel_lines[chan_idx].set_linewidth(3)
                    
        self._optode_ax.figure.canvas.draw()

    def _draw_timeseries(self):
        # Save current axis limits before clearing
        xxlim = self._dataTimeSeries_ax.get_xlim()
        yylim = self._dataTimeSeries_ax.get_ylim()
        
        # If preserve zoom is enabled and we have saved limits, use those
        if self.preserve_axis_zoom.isChecked():
            if self.preserved_xlim is not None:
                xxlim = self.preserved_xlim
            # Only preserve Y-axis if auto_scale_y is NOT checked
            if self.preserved_ylim is not None and not self.auto_scale_y.isChecked():
                yylim = self.preserved_ylim

        self._dataTimeSeries_ax.clear()

        if self.snirfData is None:
            return
        if len(self.selected_channels) == 0:
            return

        # Check if HRF view is enabled
        if self.hrf_view.isChecked() and hasattr(self, 'hrf_data') and self.hrf_data is not None:
            # Don't pass xxlim to HRF view since time scales are very different
            self._draw_hrf_timeseries()
            return

        # Extract time information
        self.t = self.snirfData.time.values
        
        # Check if we need to reset xlim (e.g., switching from HRF view or initial load)
        # Skip this check if preserve axis zoom is enabled
        if not self.preserve_axis_zoom.isChecked():
            # If the saved xlim is outside the time series data range, reset it
            if xxlim[1] < self.t[-1] * 0.5 or xxlim[0] < 0 or xxlim[1] > self.t[-1]:
                xxlim = (0, 1)  # This will trigger auto-scaling later

        # Use unified channel selection - convert indices to channel names
        chan_sel = [self.snirfData.channel.values[i] for i in self.selected_channels 
                   if i < len(self.snirfData.channel.values)]
        chan_sel = np.unique(chan_sel)

        ## Grab coordinates
        nempty_chan_sel = []
        x_chan_sel = [[], []]
        y_chan_sel = [[], []]

        for chan in chan_sel:
            if not np.isnan(self.snirfData.sel(channel=chan)[0][0]):
                x_chan_sel[0].append(
                    self.sx[
                        self.slabel == self.snirfData.sel(channel=chan).source.values
                    ][0]
                )
                x_chan_sel[1].append(
                    self.dx[
                        self.dlabel == self.snirfData.sel(channel=chan).detector.values
                    ][0]
                )
                y_chan_sel[0].append(
                    self.sy[
                        self.slabel == self.snirfData.sel(channel=chan).source.values
                    ][0]
                )
                y_chan_sel[1].append(
                    self.dy[
                        self.dlabel == self.snirfData.sel(channel=chan).detector.values
                    ][0]
                )
                nempty_chan_sel.append(chan)

        wvl_idx = self.wv.selectedItems()
        wvl_idx = [foo.text() for foo in wvl_idx]
        
        # Use solid line if only one wavelength/chromophore selected, 
        # otherwise use solid and dotted to distinguish them
        if len(wvl_idx) == 1:
            wvl_ls = ["-"]  # Only solid line
        else:
            wvl_ls = ["-", ":"]  # Solid and dotted

        ## Grab timeseries y-label
        ylabel = self.ts_sel
        if "amp" in ylabel:
            ylabel = "amp (A.U.)"
        elif "od" in ylabel:
            ylabel = r"$\Delta$ OD (A.U.)"
        elif "conc" in ylabel or "chromo" in self.snirfData.dims:
            ylabel = rf"$\Delta$ Concentration ({self._get_concentration_unit_label(self.snirfData)})"

        # Update channel highlighting on probe display
        self._update_channel_highlights()
        
        # Color palette for time series plots - ordered from high contrast to low contrast
        chan_col = [
            "#FF0000", "#0000FF", "#00FF00", "#FF00FF", "#FFFF00", "#00FFFF", "#FF8000", "#8000FF",
            "#FF0080", "#00FF80", "#0080FF", "#FF6B6B", "#4ECDC4", "#FFB000", "#117733", "#DC267F",
            "#332288", "#882255", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#45B7D1",
            "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9", "#F8C471"
        ]

        # Clean up existing channel highlights
        for item in self.chan_highlight:
            if hasattr(item, 'remove'):  # Check if it's actually a line object
                try:
                    item.remove()
                except:
                    pass

        self.chan_highlight = []


        self._draw_aux_timeseries()

        ymin = 100
        ymax = -100

        # Plot timeseries
        if "wavelength" in self.snirfData.dims:
            for i_wv, sel_wv in enumerate(wvl_idx):
                # Use index within selected wavelengths for line style
                ls_idx = i_wv % len(wvl_ls) if len(wvl_ls) > 1 else 0
                
                for i_ch, chan in enumerate(nempty_chan_sel):
                    # Get the actual channel index for consistent coloring
                    chan_idx = np.where(self.snirfData.channel.values == chan)[0][0]
                    
                    # Use smart color assignment to avoid conflicts
                    if len(self.selected_channels) <= len(chan_col):
                        # Find position of this channel in selected channels
                        color_idx = self.selected_channels.index(chan_idx) if chan_idx in self.selected_channels else i_ch
                    else:
                        color_idx = chan_idx % len(chan_col)
                    
                    self.timeSeries = self._dataTimeSeries_ax.plot(
                        self.t,
                        self.snirfData.sel(channel=chan, wavelength=sel_wv).T,
                        ls=wvl_ls[ls_idx],
                        zorder=5,
                        color=chan_col[color_idx],
                    )

                    ymin = min(
                        ymin,
                        min(
                            self.snirfData.sel(
                                channel=chan, wavelength=sel_wv
                            ).values.ravel()
                        ),
                    )
                    ymax = max(
                        ymax,
                        max(
                            self.snirfData.sel(
                                channel=chan, wavelength=sel_wv
                            ).values.ravel()
                        ),
                    )

        elif "chromo" in self.snirfData.dims:
            for i_wv, sel_wv in enumerate(wvl_idx):
                # Use index within selected chromophores for line style
                ls_idx = i_wv % len(wvl_ls) if len(wvl_ls) > 1 else 0
                
                if "[" in sel_wv:
                    sel_wv = sel_wv[1:-1]

                for i_ch, chan in enumerate(nempty_chan_sel):
                    # Get the actual channel index for consistent coloring
                    chan_idx = np.where(self.snirfData.channel.values == chan)[0][0]
                    
                    # Use smart color assignment to avoid conflicts
                    if len(self.selected_channels) <= len(chan_col):
                        # Find position of this channel in selected channels
                        color_idx = self.selected_channels.index(chan_idx) if chan_idx in self.selected_channels else i_ch
                    else:
                        color_idx = chan_idx % len(chan_col)
                    
                    self.timeSeries = self._dataTimeSeries_ax.plot(
                        self.t,
                        self.snirfData.sel(channel=chan, chromo=sel_wv).T,
                        ls=wvl_ls[ls_idx],
                        zorder=5,
                        color=chan_col[color_idx],
                    )

                    ymin = min(
                        ymin,
                        min(
                            self.snirfData.sel(
                                channel=chan, chromo=sel_wv
                            ).values.ravel()
                        ),
                    )
                    ymax = max(
                        ymax,
                        max(
                            self.snirfData.sel(
                                channel=chan, chromo=sel_wv
                            ).values.ravel()
                        ),
                    )

        # Plot stims
        stim_col = ["#648FFF", "#DC267F", "#FFB000", "#785EF0", "#FE6100"]
        if self.selected_stim_types and hasattr(self, 'snirfRec') and len(self.snirfRec.stim) > 0:
            ymax = ymax + (0.05 * (ymax - ymin))
            ymin = ymin - (0.05 * (ymax - ymin))

            # Only plot selected stimulus types
            all_stim_types = np.unique(self.snirfRec.stim.trial_type)
            selected_types = [tt for tt in all_stim_types if tt in self.selected_stim_types]
            
            for i_t, tt in enumerate(selected_types):
                label_on = True
                for i_r, dat in self.snirfRec.stim.loc[
                    self.snirfRec.stim["trial_type"] == tt
                ].iterrows():
                    if label_on:
                        self._dataTimeSeries_ax.axvline(
                            dat.onset,
                            ls="--",
                            lw=1,
                            zorder=1,
                            c=stim_col[i_t % 5],
                            label=tt,
                        )
                        self._dataTimeSeries_ax.fill(
                            [
                                dat.onset,
                                dat.onset,
                                dat.onset + dat.duration,
                                dat.onset + dat.duration,
                            ],
                            [ymin, ymax, ymax, ymin],
                            color=stim_col[i_t % 5] + "22",
                            zorder=1,
                        )
                    else:
                        self._dataTimeSeries_ax.axvline(
                            dat.onset, ls="--", lw=1, zorder=1, c=stim_col[i_t % 5]
                        )
                        self._dataTimeSeries_ax.fill(
                            [
                                dat.onset,
                                dat.onset,
                                dat.onset + dat.duration,
                                dat.onset + dat.duration,
                            ],
                            [ymin, ymax, ymax, ymin],
                            color=stim_col[i_t % 5] + "22",
                            zorder=1,
                        )

                    label_on = False

            self._dataTimeSeries_ax.legend(loc="upper right")

        self._dataTimeSeries_ax.set_ylabel(ylabel)
        self._dataTimeSeries_ax.grid("True", axis="y")

        # Set axis limits if preserve zoom is enabled or if we have non-default xlim
        if self.preserve_axis_zoom.isChecked():
            # If we have explicitly preserved limits, always restore them
            if self.preserved_xlim is not None:
                self._dataTimeSeries_ax.set_xlim(xxlim)
            elif xxlim[0] != 0 or xxlim[1] != 1:
                # Fallback: restore current limits if they're not default
                self._dataTimeSeries_ax.set_xlim(xxlim)
            
            # Only restore Y limits if auto_scale_y is NOT checked
            if not self.auto_scale_y.isChecked():
                if self.preserved_ylim is not None:
                    self._dataTimeSeries_ax.set_ylim(yylim)
                elif yylim[0] != 0 or yylim[1] != 1:
                    # Fallback: restore current limits if they're not default
                    self._dataTimeSeries_ax.set_ylim(yylim)
            # If auto_scale_y is checked, don't set ylim - let matplotlib auto-scale
        else:
            # Original behavior - only preserve xlim if not default
            if xxlim[0] != 0 or xxlim[1] != 1:
                self._dataTimeSeries_ax.set_xlim(xxlim)

        self._dataTimeSeries_ax.figure.canvas.draw()

        self.statbar.showMessage("Timeseries Drawn!")

    def _draw_hrf_timeseries(self):
        """Draw HRF estimates instead of time series"""
        
        print(f"_draw_hrf_timeseries called. Group avg checkbox state: {self.hrf_group_avg.isChecked()}")
        self._set_aux_axis_visible(False)
        
        # Update channel highlighting on probe display
        self._update_channel_highlights()
        
        # Check if group average is selected
        if self.hrf_group_avg.isChecked():
            # Load group average HRF data
            print("Attempting to load group average HRF data...")
            hrf_data = self._load_group_average_hrf()
            print(f"Load result: {hrf_data is not None}")
            if hrf_data is None:
                print("Group average HRF data is None - returning early")
                self.statbar.showMessage("No group average HRF data found")
                self._dataTimeSeries_ax.figure.canvas.draw()
                return
            
            # Extract the actual xarray DataArray from the loaded data
            if isinstance(hrf_data, dict):
                print(f"Loaded data is a dict with keys: {hrf_data.keys()}")
                # Try common keys where the HRF data might be stored
                hrf_est = hrf_data.get('hrf_est') or hrf_data.get('groupaverage') or hrf_data.get('data')
                if hrf_est is None:
                    # If still None, look for the first xarray object in the dict
                    for key, value in hrf_data.items():
                        if hasattr(value, 'dims'):
                            hrf_est = value
                            print(f"Found xarray data in key: {key}")
                            break
            else:
                # If it's an xarray Dataset, try to extract the data variable
                if hasattr(hrf_data, 'data_vars'):
                    # It's an xarray Dataset, extract the data variable
                    print(f"Data is an xarray Dataset with variables: {list(hrf_data.data_vars.keys())}")
                    # Try common variable names
                    if 'hrf_est' in hrf_data.data_vars:
                        hrf_est = hrf_data['hrf_est']
                    elif 'group_average' in hrf_data.data_vars:
                        hrf_est = hrf_data['group_average']
                    elif 'hrf_estimate' in hrf_data.data_vars:
                        hrf_est = hrf_data['hrf_estimate']
                    else:
                        # Use the first data variable
                        first_var = list(hrf_data.data_vars.keys())[0]
                        hrf_est = hrf_data[first_var]
                        print(f"Using first data variable: {first_var}")
                else:
                    # It's already a DataArray
                    hrf_est = hrf_data
            
            if hrf_est is None or not hasattr(hrf_est, 'dims'):
                print(f"Could not find valid HRF data. Type: {type(hrf_data)}")
                self.statbar.showMessage("Invalid group average HRF data format")
                self._dataTimeSeries_ax.figure.canvas.draw()
                return
            
            print(f"Group average HRF loaded. Dimensions: {hrf_est.dims}")
            print(f"Trial types: {hrf_est.coords['trial_type'].values}")
            print(f"Chromophores: {hrf_est.coords['chromo'].values}")
            # For group average, use selected channels if any are selected
            if hasattr(self, 'selected_channels') and len(self.selected_channels) > 0:
                chan_sel = [self.snirfData.channel.values[i] for i in self.selected_channels 
                           if i < len(self.snirfData.channel.values)]
                chan_sel = np.unique(chan_sel)
                print(f"Using {len(chan_sel)} selected channels for group average")
            else:
                chan_sel = None
                print("Using all channels for group average")
        else:
            # Get HRF data from current subject/run
            if not hasattr(self, 'hrf_data') or self.hrf_data is None:
                self.statbar.showMessage("No HRF data loaded for current recording")
                return
            hrf_est = self.hrf_data.get('hrf_est')
            if hrf_est is None:
                self.statbar.showMessage("No HRF estimates found in data")
                return
            
            # For individual data, use selected channels
            chan_sel = [self.snirfData.channel.values[i] for i in self.selected_channels 
                       if i < len(self.snirfData.channel.values)]
            chan_sel = np.unique(chan_sel)
            
            if len(chan_sel) == 0:
                self.statbar.showMessage("No channels selected")
                self._dataTimeSeries_ax.figure.canvas.draw()
                return
        
        # Get time vector from HRF data
        hrf_time = hrf_est.coords['time'].values
        hrf_unit_label = self._get_concentration_unit_label(hrf_est)
        
        # Color palette (same as time series) - ordered from high contrast to low contrast
        chan_col = [
            "#FF0000", "#0000FF", "#00FF00", "#FF00FF", "#FFFF00", "#00FFFF", "#FF8000", "#8000FF",
            "#FF0080", "#00FF80", "#0080FF", "#FF6B6B", "#4ECDC4", "#FFB000", "#117733", "#DC267F",
            "#332288", "#882255", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#45B7D1",
            "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9", "#F8C471"
        ]
        
        # Get trial types from HRF data
        trial_types = hrf_est.coords['trial_type'].values
        
        # Filter trial types based on selected stimulus types
        # If no stimuli are selected, don't display any HRF
        if not self.selected_stim_types:
            self.statbar.showMessage("No stimuli selected for HRF display")
            self._dataTimeSeries_ax.figure.canvas.draw()
            return
            
        trial_types = [tt for tt in trial_types if tt in self.selected_stim_types]
        
        if len(trial_types) == 0:
            self.statbar.showMessage("No HRF data available for selected stimuli")
            self._dataTimeSeries_ax.figure.canvas.draw()
            return
        
        # Get available chromophores from HRF data
        available_chromos = list(hrf_est.coords['chromo'].values)
        
        # Get selected chromophores from wv widget (same as regular time series)
        wvl_idx = self.wv.selectedItems()
        wvl_idx = [foo.text() for foo in wvl_idx]
        
        # Filter chromophores based on wv selection
        selected_chromos = []
        for sel_wv in wvl_idx:
            # Remove brackets if present (e.g., "[HbO]" -> "HbO")
            chromo = sel_wv.strip('[]')
            if chromo in available_chromos:
                selected_chromos.append(chromo)
        
        # If no chromophores selected, show message and return
        if not selected_chromos:
            self.statbar.showMessage("No chromophores selected for HRF display")
            self._dataTimeSeries_ax.figure.canvas.draw()
            return
        
        print(f"Plotting HRF - Trial types: {trial_types}, Chromos: {selected_chromos}, Group avg: {self.hrf_group_avg.isChecked()}")
        
        # Define line styles for chromophores
        # Use solid if only one chromophore selected, otherwise alternate solid/dotted
        if len(selected_chromos) == 1:
            chromo_ls = {selected_chromos[0]: '-'}  # Only solid line
        else:
            chromo_ls = {'HbO': '-', 'HbR': ':'}  # Solid for HbO, dotted for HbR
        
        ymin = 100
        ymax = -100
        
        # Define line styles for different stimuli (trial types)
        stim_line_styles = ['-', '--', '-.', ':']  # solid, dashed, dash-dot, dotted
        
        # Define markers for different stimuli (optional, adds more distinction)
        stim_markers = ['', '', '', '']  # No markers by default, or use: ['o', 's', '^', 'v']
        
        # Plot HRF for each selected channel and trial type
        if 'chromo' in hrf_est.dims:
            if self.hrf_group_avg.isChecked():
                # For group average: plot individual lines for each selected channel
                print(f"Group average plotting mode. Trial types to plot: {trial_types}")
                
                # Determine which channels to plot
                if chan_sel is not None and len(chan_sel) > 0:
                    channels_to_plot = chan_sel
                    print(f"Plotting {len(channels_to_plot)} selected channels")
                else:
                    # If no channels selected, plot all channels
                    channels_to_plot = hrf_est.coords['channel'].values
                    print(f"No channels selected, plotting all {len(channels_to_plot)} channels")
                
                num_plots = 0
                for i_chan, channel in enumerate(channels_to_plot):
                    # Use channel-specific color
                    chan_color = chan_col[i_chan % len(chan_col)]
                    
                    for i_tt, trial_type in enumerate(trial_types):
                        # Get line style for this stimulus type
                        line_style = stim_line_styles[i_tt % len(stim_line_styles)]
                        marker = stim_markers[i_tt % len(stim_markers)]
                        
                        for chromo_name in selected_chromos:
                            try:
                                # Extract HRF data for this channel, trial type, and chromo
                                hrf_values = hrf_est.sel(channel=channel, trial_type=trial_type, chromo=chromo_name).values
                                
                                # Get line style for this chromophore
                                chromo_line_style = chromo_ls.get(chromo_name, '-')
                                
                                # Combine stimulus line style with chromophore style
                                # For multiple stimuli, vary the dash pattern
                                if len(trial_types) == 1:
                                    combined_line_style = chromo_line_style
                                else:
                                    # Use chromophore style as base, add stimulus distinction if needed
                                    combined_line_style = line_style if chromo_line_style == '-' else chromo_line_style
                                
                                # Set consistent styling
                                linewidth = 2.0
                                alpha = 0.85
                                marker_style = marker if marker else ''
                                marker_size = 3 if marker else 0
                                
                                # Plot with label showing channel, trial type and chromo
                                label = f"{chromo_name}-{trial_type}-{channel}"
                                self._dataTimeSeries_ax.plot(
                                    hrf_time,
                                    hrf_values,
                                    ls=combined_line_style,
                                    marker=marker_style,
                                    markersize=marker_size,
                                    markevery=5,  # Only show markers every 5 points to avoid clutter
                                    linewidth=linewidth,
                                    zorder=5,
                                    color=chan_color,
                                    alpha=alpha,
                                    label=label
                                )
                                num_plots += 1
                                
                                ymin = min(ymin, np.nanmin(hrf_values))
                                ymax = max(ymax, np.nanmax(hrf_values))
                            except (KeyError, ValueError) as e:
                                print(f"Error plotting group average for channel {channel}, {trial_type}, {chromo_name}: {e}")
                                continue
                print(f"Total plots created: {num_plots}")
            else:
                # For individual data: plot each channel separately
                for i_ch, chan in enumerate(chan_sel):
                    # Get the actual channel index for consistent coloring with probe plot
                    chan_idx = np.where(self.snirfData.channel.values == chan)[0][0]
                    
                    # Use smart color assignment (same logic as time series plotting)
                    if len(self.selected_channels) <= len(chan_col):
                        # Find position of this channel in selected channels
                        color_idx = self.selected_channels.index(chan_idx) if chan_idx in self.selected_channels else i_ch
                    else:
                        color_idx = chan_idx % len(chan_col)
                    
                    # Get the channel color (same as in probe plot)
                    channel_color = chan_col[color_idx]
                    
                    for i_tt, trial_type in enumerate(trial_types):
                        # Get line style for this stimulus type
                        line_style = stim_line_styles[i_tt % len(stim_line_styles)]
                        marker = stim_markers[i_tt % len(stim_markers)]
                        
                        for chromo_name in selected_chromos:
                            try:
                                # Extract HRF data for this channel, trial type, and chromo
                                hrf_values = hrf_est.sel(trial_type=trial_type, channel=chan, chromo=chromo_name).values
                                
                                # Get line style for this chromophore
                                chromo_line_style = chromo_ls.get(chromo_name, '-')
                                
                                # Combine stimulus line style with chromophore style
                                # For multiple stimuli, vary the dash pattern
                                if len(trial_types) == 1:
                                    combined_line_style = chromo_line_style
                                else:
                                    # Use chromophore style as base, add stimulus distinction if needed
                                    combined_line_style = line_style if chromo_line_style == '-' else chromo_line_style
                                
                                # Set consistent styling
                                linewidth = 2.0
                                alpha = 0.85
                                marker_style = marker if marker else ''
                                marker_size = 3 if marker else 0
                                
                                # Plot with label showing channel, trial type, and chromo
                                label = f"{chan}-{chromo_name}-{trial_type}"
                                self._dataTimeSeries_ax.plot(
                                    hrf_time,
                                    hrf_values,
                                    ls=combined_line_style,
                                    marker=marker_style,
                                    markersize=marker_size,
                                    markevery=5,  # Only show markers every 5 points to avoid clutter
                                    linewidth=linewidth,
                                    zorder=5,
                                    color=channel_color,
                                    alpha=alpha,
                                    label=label
                                )
                                
                                ymin = min(ymin, np.nanmin(hrf_values))
                                ymax = max(ymax, np.nanmax(hrf_values))
                            except (KeyError, ValueError) as e:
                                continue
        
        # Set labels and formatting
        self._dataTimeSeries_ax.set_xlabel("Time (s)")
        self._dataTimeSeries_ax.set_ylabel(
            rf"HRF Amplitude ($\Delta$ Concentration {hrf_unit_label})"
        )
        self._dataTimeSeries_ax.grid(True, axis="y")
        self._dataTimeSeries_ax.legend(loc="upper right")
        
        # Auto-scale to HRF time range (don't use time series xlim)
        self._dataTimeSeries_ax.set_xlim(hrf_time[0], hrf_time[-1])
        
        self._dataTimeSeries_ax.figure.canvas.draw()
        self.statbar.showMessage("HRF Drawn!")

    def _aux_changed(self, s):  # TODO
        self._auxTimeSeries_ax.clear()

        if s == "None" or s == "dark signal":
            self.aux_sel = []
            self.aux_type = None
            # Remove all auxplot lines if any
            if hasattr(self, 'auxplot') and self.auxplot:
                for line in self.auxplot:
                    try:
                        line.remove()
                    except Exception:
                        pass
                self.auxplot = []
        # elif s == 'dark signal':
        #     return
        else:
            self.aux_sel = self.snirfRec.aux_ts[s]
            self.aux_type = s

        self._draw_timeseries()
        
        # Save GUI state after auxiliary selection change
        self._save_gui_state()

    # ============== Pipeline Monitoring Methods ==============
    
    def _find_snakemake_process_pid(self):
        """Find the PID of the snakemake process using our config file"""
        try:
            import psutil
            # Use the actual config path that was used (temp config or original)
            config_to_search = getattr(self, '_actual_config_used', self.snakemake_config_path)
            config_path = os.path.abspath(config_to_search)
            # Normalize to forward slashes for comparison (Windows command lines often mix slashes)
            config_path_normalized = config_path.replace('\\', '/')
            
            
            snakemake_procs = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if not cmdline:
                        continue
                    
                    cmdline_str = ' '.join(cmdline).lower()
                    
                    # Look for snakemake in command line (could be python snakemake or snakemake.exe)
                    if 'snakemake' in cmdline_str:
                        # Check if our config file is in the command
                        cmdline_full = ' '.join(cmdline)
                        snakemake_procs.append((proc.info['pid'], cmdline_full))
                        
                        # Normalize cmdline to forward slashes for comparison (handles mixed slashes)
                        cmdline_normalized = cmdline_full.replace('\\', '/')
                        
                        # Check if our normalized config path is in normalized cmdline
                        if config_path_normalized in cmdline_normalized:
                            return proc.info['pid']
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if snakemake_procs:
                pass
            return None
        except Exception as e:
            print(f"Error finding snakemake process: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _store_pipeline_process_info(self, pid):
        """Store process ID and start time in pipeline_status"""
        try:
            import psutil
            proc = psutil.Process(pid)
            
            # Store PID and process creation time
            self.pipeline_status['process_pid'] = pid
            self.pipeline_status['process_start_time'] = proc.create_time()
            
            
            # Save to homer.config immediately
            derivatives_dir = os.path.dirname(self.snakemake_config_path)
            self._save_homer_config(
                self.snakefile_path,
                self.snakemake_config_path,
                derivatives_dir,
                self.pipeline_status
            )
        except Exception as e:
            print(f"Error storing process info: {str(e)}")
    
    def _extract_all_tasks_and_runs_from_gui(self):
        """Extract all unique tasks and run numbers from subject_to_runs_map"""
        import re
        all_tasks = set()
        all_runs = set()
        
        for subject, runs in self.subject_to_runs_map.items():
            for run_str in runs:
                # Parse "task-STS_run-01" format
                match = re.search(r'task-(\w+)_run-(\d+)', run_str)
                if match:
                    all_tasks.add(match.group(1))  # "STS"
                    all_runs.add(match.group(2))    # "01"
        
        return sorted(all_tasks), sorted(all_runs)
    
    def _create_full_scope_temp_config(self):
        """Create temporary config with all subjects, tasks, runs from GUI data
        
        Returns path to temporary config file, or None if creation fails
        """
        import tempfile
        import copy
        
        try:
            if not self.snakemake_config:
                print("No snakemake_config loaded, cannot create full scope config")
                return None
            
            # Extract all tasks and runs from GUI data
            all_tasks, all_runs = self._extract_all_tasks_and_runs_from_gui()
            
            # Get subjects that have at least one run
            subjects_with_runs = [s for s in self.subjects if self.subject_to_runs_map.get(s)]
            
            if not subjects_with_runs or not all_tasks or not all_runs:
                print(f"Cannot create full scope config: subjects={len(subjects_with_runs)}, tasks={len(all_tasks)}, runs={len(all_runs)}")
                return None
            
            print(f"Creating full scope config: {len(subjects_with_runs)} subjects, {len(all_tasks)} tasks, {len(all_runs)} runs")
            print(f"  Subjects: {subjects_with_runs}")
            print(f"  Tasks: {all_tasks}")
            print(f"  Runs: {all_runs}")
            
            # Create a deep copy of the current config
            full_config = copy.deepcopy(self.snakemake_config)
            
            # Update dataset section with all subjects/tasks/runs
            if 'dataset' not in full_config:
                full_config['dataset'] = {}
            
            full_config['dataset']['subject'] = subjects_with_runs
            full_config['dataset']['task'] = all_tasks
            full_config['dataset']['run'] = all_runs
            
            # Write to temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            yaml.dump(full_config, temp_file, default_flow_style=False)
            temp_file.close()
            
            print(f"Full scope temp config created: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            print(f"Error creating full scope temp config: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_file_status_maps(self):
        """Load both current scope and full scope file status maps from summary
        
        This runs two summary commands:
        1. With current user config (filtered subjects/tasks/runs)
        2. With full scope config (all subjects/tasks/runs from GUI)
        """
        print("\n=== Loading file status maps ===")
        
        if not self.snakefile_path or not self.snakemake_config_path:
            print("No snakefile or config path set, skipping status map load")
            return
        
        try:
            # Run summary using the actual config file (no temp config needed)
            # The config file already has the correct subjects configured
            print("Running summary for current config scope...")
            summary_workdir, summary_config_args = self._get_windows_relative_run_context(self.snakemake_config_path)
            summary_output_base_dir = summary_workdir
            self.current_scope_files = self._run_snakemake_summary(
                self.snakefile_path,
                self.snakemake_config_path,
                target_rule='all_imagerecon',  # Use all_imagerecon to get individual subject files
                workdir=summary_workdir,
                output_base_dir=summary_output_base_dir,
                config_args=summary_config_args
            )
            
            print(f"  ✓ Loaded status for {len(self.current_scope_files)} files in current config")
            
            # No need for full scope summary - files not in current scope will be gray
            self.all_scope_files = {}
            
            # Also update legacy file_status_map for backward compatibility
            self.file_status_map = self.current_scope_files
            
            print(f"=== File status map loaded successfully ===\n")
            
        except Exception as e:
            print(f"Error loading file status maps: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fallback to empty maps
            self.current_scope_files = {}
            self.all_scope_files = {}
            self.file_status_map = {}
    
    def _run_snakemake_summary(self, snakefile_path, config_path, target_rule='all_default', workdir=None, output_base_dir=None, config_args=None):
        """Run snakemake with --summary to get status of all workflow files"""
        try:
            config_args = config_args or []
            # Build command with conda activation if environment is set
            if self.conda_env:
                cmd = _build_snakemake_command(
                    ['snakemake', '-s', snakefile_path, '--configfile', config_path, *config_args, '--nolock', '--summary', target_rule],
                    self.conda_env
                )
            else:
                cmd = ['snakemake', '-s', snakefile_path, '--configfile', config_path, *config_args, '--nolock', '--summary', target_rule]
            

            
            # Run the summary command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=workdir
            )
            
            
            if result.returncode != 0:
                print(f"ERROR: Summary command failed")
                print(f"STDERR: {result.stderr[:500]}")
                return {}
            
            # Parse summary table output
            # Format: output_file\tdate\trule\tlog-file(s)\tstatus\tplan
            # Columns are tab-separated
            file_status_map = {}
            lines = result.stdout.split('\n')
            
            for line in lines:
                line = line.strip()
                # Skip empty lines and header
                if not line or line.startswith('output_file') or line.startswith('Building'):
                    continue
                
                # Split by tabs (the table is tab-delimited)
                parts = line.split('\t')
                if len(parts) >= 6:
                    # Extract columns: file, date, rule, log, status, plan
                    file_path = parts[0].strip()
                    date_str = parts[1].strip()
                    rule = parts[2].strip()
                    log_file = parts[3].strip()
                    status = parts[4].strip()
                    plan = parts[5].strip()  # Can be "no update", "update pending", etc.
                    
                    # Normalize path to handle Windows paths
                    file_path = os.path.normpath(file_path)
                    if output_base_dir and not os.path.isabs(file_path):
                        file_path = os.path.normpath(os.path.join(output_base_dir, file_path))
                    
                    file_status_map[file_path] = {
                        'date': date_str,
                        'rule': rule,
                        'status': status,
                        'plan': plan
                    }
            
            if file_status_map:
                for f, info in list(file_status_map.items())[:3]:
                    print(f"  {os.path.basename(f)}: status={info['status']}, plan={info['plan']}")
            
            return file_status_map
            
        except subprocess.TimeoutExpired:
            print("Dry run timed out after 30 seconds")
            return set()
        except Exception as e:
            print(f"Error running dry run: {str(e)}")
            import traceback
            traceback.print_exc()
            return set()
    
    def _save_pipeline_status_on_start(self):
        """Save pipeline status when starting a run"""
        if not self.snakemake_config_path:
            return
            
        try:
            derivatives_dir = os.path.dirname(self.snakemake_config_path)
            
            # Get current timestamp
            current_time = datetime.now().isoformat()
            
            # Build status dict
            pipeline_status = {
                'last_run_time': current_time,
                'status': 'running',
                'target_rule': getattr(self, 'current_target_rule', 'all_default'),
                'expected_outputs': list(self.expected_pipeline_outputs),
                'completed_outputs': []
            }
            
            
            # Save to homer.config
            self._save_homer_config(
                self.snakefile_path,
                self.snakemake_config_path,
                derivatives_dir,
                pipeline_status
            )
            
            self.pipeline_status = pipeline_status
            self.completed_pipeline_outputs = set()  # Reset completed outputs
            
        except Exception as e:
            print(f"Warning: Could not save pipeline status: {str(e)}")
    
    def _restore_pipeline_state(self):
        """Restore pipeline state from homer.config and update file colors"""
        if not self.snakemake_config_path:
            return
            
        try:
            derivatives_dir = os.path.dirname(self.snakemake_config_path)
            homer_config_path = os.path.join(derivatives_dir, 'homer.config')
            
            if os.path.exists(homer_config_path):
                # Try safe_load first, fall back to unsafe if needed
                homer_config = None
                try:
                    with open(homer_config_path, 'r') as f:
                        homer_config = yaml.safe_load(f) or {}
                except yaml.constructor.ConstructorError:
                    print("Warning: homer.config contains unsafe YAML tags, using unsafe_load for pipeline state")
                    try:
                        with open(homer_config_path, 'r') as f:
                            homer_config = yaml.unsafe_load(f) or {}
                    except Exception as e:
                        print(f"Warning: Could not load homer.config even with unsafe_load: {e}")
                        return
                
                self.pipeline_status = homer_config.get('pipeline_status', {})
                
                # Restore expected and completed outputs
                self.expected_pipeline_outputs = set(self.pipeline_status.get('expected_outputs', []))
                self.completed_pipeline_outputs = set(self.pipeline_status.get('completed_outputs', []))
                
                # Get current file status from summary
                if self.snakefile_path and self.snakemake_config_path:
                    # Use the same target rule that was used to start the pipeline
                    target_rule = self._resolve_target_rule(self.pipeline_status.get('target_rule', 'all_default'))
                    self.file_status_map = self._run_snakemake_summary(
                        self.snakefile_path,
                        self.snakemake_config_path,
                        target_rule
                    )
                
                # Update colors for all current files
                self._update_all_file_colors()
                
                # Update the currently selected item colors in the comboboxes
                self._update_combobox_selection_colors()
                
                # If pipeline was running, check if it's STILL running
                if self.pipeline_status.get('status') == 'running':
                    # Validate the stored process is still running
                    if self._is_stored_pipeline_running():
                        self._start_pipeline_monitoring()
                    else:
                        # Pipeline finished while GUI was closed
                        self.pipeline_status['status'] = 'completed'
                        self.pipeline_status['completed_time'] = datetime.now().isoformat()
                        derivatives_dir = os.path.dirname(self.snakemake_config_path)
                        self._save_homer_config(
                            self.snakefile_path,
                            self.snakemake_config_path,
                            derivatives_dir,
                            self.pipeline_status
                        )
                        self.statbar.showMessage("Pipeline completed while GUI was closed")
                    
        except Exception as e:
            print(f"Warning: Could not restore pipeline state: {str(e)}")
    
    def _is_stored_pipeline_running(self):
        """Check if the stored pipeline process (PID + start time) is still running"""
        stored_pid = self.pipeline_status.get('process_pid')
        stored_start_time = self.pipeline_status.get('process_start_time')
        
        if not stored_pid:
            return False
        
        
        try:
            import psutil
            
            # Check if process with stored PID exists
            if psutil.pid_exists(stored_pid):
                proc = psutil.Process(stored_pid)
                proc_name = proc.name()
                
                # Verify it's the same process by comparing start time
                # (PIDs can be reused after process dies)
                if stored_start_time:
                    actual_start_time = proc.create_time()
                    time_diff = abs(actual_start_time - stored_start_time)
                    if time_diff < 1.0:  # Within 1 second
                        # Verify it's actually a snakemake process
                        cmdline = ' '.join(proc.cmdline()).lower()
                        has_snakemake = 'snakemake' in cmdline
                        if has_snakemake:
                            return True
                        else:
                            pass
                    else:
                        pass
                else:
                    # No start time stored, just check if it's snakemake
                    cmdline = ' '.join(proc.cmdline()).lower()
                    if 'snakemake' in cmdline:
                        return True
            else:
                pass
                
            return False
            
        except psutil.NoSuchProcess:
            return False
        except Exception as e:
            print(f"ERROR _is_stored_pipeline_running: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _start_pipeline_monitoring(self):
        """Start the QTimer to monitor file updates every 5 seconds"""
        if self.pipeline_monitor_timer is None:
            self.pipeline_monitor_timer = QtCore.QTimer(self)
            self.pipeline_monitor_timer.timeout.connect(self._check_file_updates)
        
        # Check immediately, then every 5 seconds
        self._check_file_updates()
        self.pipeline_monitor_timer.start(5000)  # 5 seconds
        
        self.statbar.showMessage("Pipeline monitoring active...")
    
    def _stop_pipeline_monitoring(self):
        """Stop the monitoring timer and cancel any running worker"""
        if self.pipeline_monitor_timer:
            self.pipeline_monitor_timer.stop()
        
        # Cancel and clean up any running summary worker
        if self.summary_worker and self.summary_worker.isRunning():
            self.summary_worker.cancel()
            self.summary_worker.wait(3000)  # Wait up to 3 seconds for thread to finish
            if self.summary_worker.isRunning():
                print("WARNING: Summary worker did not stop in time")
            self.summary_worker = None
        
        self.statbar.showMessage("Pipeline monitoring stopped")
    
    def _check_file_updates(self):
        """Start background summary check to monitor pipeline progress"""
        
        if not self.snakemake_config_path:
            return
        
        # Skip updates if pipeline is not running
        if self.pipeline_status.get('status') != 'running':
            self._stop_pipeline_monitoring()
            return
        
        # Don't start new worker if one is already running
        if self.summary_worker and self.summary_worker.isRunning():
            return
        
        try:
            # Determine which config to use for monitoring
            # In "run current only" mode, use the temp config
            monitor_config = getattr(self, '_actual_config_used', self.snakemake_config_path)
            monitor_workdir = getattr(self, '_actual_snakemake_workdir', None)
            monitor_output_base_dir = getattr(self, '_actual_output_base_dir', None)
            monitor_config_args = getattr(self, '_actual_snakemake_config_args', [])
            
            if self.run_current_only_mode:
                pass
            else:
                pass
            
            # Reload config file to pick up any changes
            with open(monitor_config, 'r') as f:
                self.snakemake_config = yaml.safe_load(f)
            
            # Start background worker to get current file status from summary
            if self.snakefile_path and monitor_config:
                # Use the same target rule that was used to start the pipeline
                target_rule = self._resolve_target_rule(self.pipeline_status.get('target_rule', 'all_default'))
                self.summary_worker = SummaryWorker(
                    self.snakefile_path,
                    monitor_config,
                    self.conda_env,
                    target_rule,
                    workdir=monitor_workdir,
                    output_base_dir=monitor_output_base_dir,
                    config_args=monitor_config_args
                )
                self.summary_worker.summary_completed.connect(self._on_summary_completed)
                self.summary_worker.summary_failed.connect(self._on_summary_failed)
                self.summary_worker.start()
                self.statbar.showMessage("Updating pipeline status...")
                
        except Exception as e:
            print(f"Error checking file updates: {str(e)}")
    
    def _on_summary_completed(self, file_status_map):
        """Handle summary results from background worker (runs in main GUI thread)"""
        try:
            
            # Update the file status maps
            # In "run current only" mode, UPDATE the existing scope (don't replace)
            # This preserves the initial full-config summary while updating the selected file
            if self.run_current_only_mode:
                # Update only the files in the new summary, keep others unchanged
                for file_path, status_info in file_status_map.items():
                    self.current_scope_files[file_path] = status_info
            else:
                # Normal mode: replace entire scope
                self.current_scope_files = file_status_map
            
            self.file_status_map = self.current_scope_files  # Backward compatibility
            self.all_scope_files = {}  # No dual summary
            
            # Update colors for all files based on current state
            files_changed_to_black = self._update_all_file_colors()
            
            # Force comboboxes to repaint to show updated colors
            if hasattr(self, 'subj'):
                self.subj.repaint()
                # Also repaint the dropdown view
                if self.subj.view():
                    self.subj.view().repaint()
            if hasattr(self, 'run'):
                self.run.repaint()
                if self.run.view():
                    self.run.view().repaint()
            
            # Update currently selected item colors
            self._update_combobox_selection_colors()
            
            # Check for newly completed files and update homer.config
            self._update_completed_outputs()
            self._refresh_current_hrf_availability()
            
            # Clear cache for files that changed to black so they reload fresh
            if files_changed_to_black:
                for subj, run in files_changed_to_black:
                    print(f"  - {subj} / {run}")
                self._mark_updated_files_for_reload(files_changed_to_black)
                
                # Auto-reload if currently displayed file just completed
                # NOTE: This only triggers ONCE when file status changes to black, not every 5 seconds
                # files_changed_to_black only contains files that changed THIS update cycle
                current_subj = self.subj.currentText() if hasattr(self, 'subj') else None
                current_run = self.run.currentText() if hasattr(self, 'run') else None
                
                
                if current_subj and current_subj != "None" and current_run and current_run != "None":
                    if (current_subj, current_run) in files_changed_to_black:
                        self.statbar.showMessage(f"Auto-reloading {current_subj} {current_run} with fresh processed data...")
                        # Trigger reload by calling the plot function
                        QtCore.QTimer.singleShot(500, lambda: self._auto_reload_current_file())
                    else:
                        # Check if ANY file type for this subject/run changed to black
                        # (files_changed_to_black only tracks the "Color by" file type)
                        # So let's check all file types manually
                        any_file_changed = self._check_any_file_type_changed_to_black(current_subj, current_run)
                        if any_file_changed:
                            self.statbar.showMessage(f"Auto-reloading {current_subj} {current_run} with fresh processed data...")
                            QtCore.QTimer.singleShot(500, lambda: self._auto_reload_current_file())
                else:
                    pass
            
            # Only check pipeline completion and update monitoring status if still running
            if self.pipeline_status.get('status') == 'running':
                # Check if pipeline process has completed
                self._check_pipeline_completion()
                
                # Update status bar with monitoring info (only if still running after check)
                if self.pipeline_status.get('status') == 'running':
                    completed_count = sum(1 for info in file_status_map.values() 
                                         if info.get('status') == 'ok' and 'no update' in info.get('plan', '').lower())
                    total_count = len(file_status_map)
                    self.statbar.showMessage(f"Pipeline monitoring: {completed_count}/{total_count} files complete")
            else:
                pass
            
        except Exception as e:
            print(f"Error handling summary results: {str(e)}")
    
    def _on_summary_failed(self, error_msg):
        """Handle summary failure from background worker"""
        print(f"WARNING: Background summary check failed: {error_msg}")
    
    def _update_completed_outputs(self):
        """Check expected outputs and mark completed ones, save to homer.config"""
        if not self.expected_pipeline_outputs or not self.pipeline_status.get('last_run_time'):
            return
        
        try:
            last_run_time = datetime.fromisoformat(self.pipeline_status['last_run_time'])
            newly_completed = []
            
            # Check each expected output
            for file_path in self.expected_pipeline_outputs:
                # Skip if already marked as completed
                if file_path in self.completed_pipeline_outputs:
                    continue
                
                # Check if file exists and is newer than pipeline start
                if os.path.exists(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime >= (last_run_time - timedelta(seconds=5)):
                        self.completed_pipeline_outputs.add(file_path)
                        newly_completed.append(file_path)
            
            # If we found newly completed files, update homer.config
            if newly_completed:
                self.pipeline_status['completed_outputs'] = list(self.completed_pipeline_outputs)
                
                derivatives_dir = os.path.dirname(self.snakemake_config_path)
                self._save_homer_config(
                    self.snakefile_path,
                    self.snakemake_config_path,
                    derivatives_dir,
                    self.pipeline_status
                )
                
        except Exception as e:
            print(f"Error updating completed outputs: {str(e)}")
    
    def _check_pipeline_completion(self):
        """Check if the Snakemake pipeline is still running using stored process info"""
        
        if self.pipeline_status.get('status') != 'running':
            return
        
        # If we don't have a PID, we can't check completion via process monitoring
        # This can happen if PID detection failed at launch
        if not self.pipeline_status.get('process_pid'):
            return
            
        try:
            # Use stored PID validation instead of scanning all processes
            snakemake_running = self._is_stored_pipeline_running()
            
            # If stored process not running, pipeline has completed
            if not snakemake_running:
                
                # Do final summary check to get latest file states
                if self.snakefile_path and self.snakemake_config_path:
                    # Use temp config for final summary in "run current only" mode
                    # This checks the file we actually processed, not all files in original config
                    final_summary_config = getattr(self, '_actual_config_used', self.snakemake_config_path)
                    final_summary_workdir = getattr(self, '_actual_snakemake_workdir', None)
                    final_summary_output_base_dir = getattr(self, '_actual_output_base_dir', None)
                    final_summary_config_args = getattr(self, '_actual_snakemake_config_args', [])
                    
                    # Use the same target rule that was used to start the pipeline
                    target_rule = self._resolve_target_rule(self.pipeline_status.get('target_rule', 'all_default'))
                    
                    file_status_map = self._run_snakemake_summary(
                        self.snakefile_path,
                        final_summary_config,
                        target_rule,
                        workdir=final_summary_workdir,
                        output_base_dir=final_summary_output_base_dir,
                        config_args=final_summary_config_args
                    )
                    
                    # In "run current only" mode, MERGE results (don't replace entire scope)
                    # This preserves the initial status of non-selected files
                    if self.run_current_only_mode:
                        for file_path, status_info in file_status_map.items():
                            self.current_scope_files[file_path] = status_info
                        self.file_status_map = self.current_scope_files
                    else:
                        # Normal mode: replace entire scope
                        self.current_scope_files = file_status_map
                        self.file_status_map = file_status_map
                        
                    self.all_scope_files = {}  # No dual summary
                
                derivatives_dir = os.path.dirname(self.snakemake_config_path)
                
                # Update status - check latest snakemake log for success/failure
                status = 'completed'
                snakemake_log_dir = os.path.join(
                    os.path.dirname(os.path.dirname(derivatives_dir)),
                    '.snakemake', 'log'
                )
                
                # Check if there were errors in the latest log
                if os.path.exists(snakemake_log_dir):
                    try:
                        logs = sorted(os.listdir(snakemake_log_dir), reverse=True)
                        if logs:
                            latest_log = os.path.join(snakemake_log_dir, logs[0])
                            with open(latest_log, 'r') as f:
                                log_content = f.read()
                                if 'error' in log_content.lower() or 'failed' in log_content.lower():
                                    status = 'failed'
                    except:
                        pass
                
                # Update status in homer.config
                self.pipeline_status['status'] = status
                self.pipeline_status['completed_time'] = datetime.now().isoformat()
                self._save_homer_config(
                    self.snakefile_path,
                    self.snakemake_config_path,
                    derivatives_dir,
                    self.pipeline_status
                )
                
                # Stop monitoring
                self._stop_pipeline_monitoring()
                self.snakemake_process = None
                
                # Clear "run on current selection only" mode
                self.run_current_only_mode = False
                self.current_selection_subject = None
                self.current_selection_task = None
                self.current_selection_run = None
                
                # Update file colors with final summary (orange files should turn red/black)
                self._update_all_file_colors()
                self._refresh_current_hrf_availability()
                
                # Update status bar
                self.statbar.showMessage(f"Pipeline {status}")
                            
        except Exception as e:
            print(f"Error checking pipeline completion: {str(e)}")
    
    def _update_all_file_colors(self):
        """Update colors for all subject/run combinations and return list of files that changed to black"""
        if not self.snakemake_config:
            return []
        
        files_changed_to_black = []  # List of (subject, run) tuples that changed to black
        dataset = self.snakemake_config.get('dataset', {})
        subjects = dataset.get('subject', [])
        runs = dataset.get('run', [])
        
        
        # First pass: collect all run colors per subject
        subject_run_colors = {}  # {subject: [(run, color), ...]}
        
        for subject in self.subjects:
            if subject == "None":
                continue
            available_runs = self.subject_to_runs_map.get(subject, [])
            subject_run_colors[subject] = []
            
            for run in available_runs:
                if run == "None":
                    continue
                    
                old_color = self.file_colors.get((subject, run))
                new_color = self._get_file_color(subject, run, subjects, runs)
                
                
                if old_color != new_color:
                    self.file_colors[(subject, run)] = new_color
                    # Track files that changed to black (newly completed)
                    if new_color == 'black':
                        files_changed_to_black.append((subject, run))
                
                subject_run_colors[subject].append((run, new_color))
        
        # Second pass: apply colors and determine subject aggregate color
        for subject, run_colors in subject_run_colors.items():
            # Determine subject color priority: black > orange > red > gray
            # Priority explanation:
            # - black: at least one file in scope is complete (best)
            # - orange: at least one file in scope needs work, pipeline running
            # - red: at least one file in scope needs work, pipeline stopped
            # - gray: all files out of scope (or no files found)
            subject_color = 'gray'  # Default
            for run, color in run_colors:
                if color == 'black':
                    subject_color = 'black'
                    break  # Highest priority
                elif color == 'orange' and subject_color not in ['black']:
                    subject_color = 'orange'
                elif color == 'red' and subject_color not in ['black', 'orange']:
                    subject_color = 'red'
            
            # Apply run colors
            for run, color in run_colors:
                self._apply_run_color_to_combobox(subject, run, color)
            
            # Apply subject color once
            self._apply_subject_color_to_combobox(subject, subject_color)
        
        # Update image reconstruction button color based on current selection
        self._update_image_recon_button_color()
        
        return files_changed_to_black
    
    def _update_image_recon_button_color(self):
        """Update the Image Reconstruction button text color based on file status"""
        if not hasattr(self, 'image_recon_btn'):
            return
        
        # Get current selection
        current_subject = self.subj.currentText() if hasattr(self, 'subj') else None
        current_run = self.run.currentText() if hasattr(self, 'run') else None
        
        if not current_subject or not current_run or current_subject == "None" or current_run == "None":
            # No selection, reset to default
            self.image_recon_btn.setStyleSheet("")
            return
        
        # Get the image recon file path for current selection
        file_path = self._get_image_recon_file_path(current_subject, current_run)
        
        if not file_path or file_path not in self.current_scope_files:
            # File not in scope, gray (use default styling)
            self.image_recon_btn.setStyleSheet(f"color: {self._status_color_map()['gray']};")
            return
        
        # Get status info
        status_info = self.current_scope_files[file_path]
        status = status_info.get('status', '')
        plan = status_info.get('plan', '')
        is_up_to_date = (status == 'ok' and 'no update' in plan.lower())
        
        # Determine color
        if is_up_to_date:
            color = 'black'  # File complete
        elif self._is_pipeline_running():
            color = 'orange'  # Pipeline working on it
        else:
            color = 'red'  # Needs work, pipeline not running
        
        # Apply color to button text (bold only for non-black)
        color_value = self._status_color_map()[color]
        if color == 'black':
            self.image_recon_btn.setStyleSheet(f"color: {color_value};")
        else:
            self.image_recon_btn.setStyleSheet(f"color: {color_value}; font-weight: bold;")
    
    def _is_pipeline_running(self):
        """Check if the Snakemake pipeline is currently running"""
        # Check pipeline status from config
        status = self.pipeline_status.get('status')
        
        if status == 'running':
            # Verify process is actually still alive
            is_running = self._is_stored_pipeline_running()
            return is_running
        
        return False
    
    def _get_file_color(self, subject, run, config_subjects, config_runs):
        """
        Determine color for a subject/run combination.
        
        The file checked depends on the "Color by" selector:
        - Preprocessing: checks preprocessing file status
        - HRF Estimate: checks HRF estimate file status
        - Image Recon: checks image reconstruction file status
        - All (any complete): checks all three, shows black if any is complete
        
        Normal mode (4 colors):
        - Black: In current scope, up-to-date
        - Gray: Not in current scope (any status)
        - Orange: In current scope, needs update, pipeline running
        - Red: In current scope, needs update, pipeline stopped
        
        "Run on current selection only" mode (3-tier):
        - Selected file: Orange (needs update) or Black (up-to-date)
        - Non-selected file, up-to-date: Black (already done)
        - Non-selected file, needs update: Red (needs work but skipping)
        - Gray: File not in GUI config at all
        """
        # Automatically determine color coding mode based on HRF view state
        # When HRF view is active, check HRF estimate files; otherwise check preprocessing files
        color_by = "HRF Estimate" if (hasattr(self, 'hrf_view') and self.hrf_view.isChecked()) else "Preprocessing"
        
        file_paths_to_check = []
        
        if color_by == "HRF Estimate":
            file_path = self._get_expected_file_path(subject, run)
            file_type = "HRF estimate"
            if file_path:
                file_paths_to_check.append(file_path)
        elif color_by == "Image Recon":
            file_path = self._get_image_recon_file_path(subject, run)
            file_type = "image reconstruction"
            if file_path:
                file_paths_to_check.append(file_path)
        else:  # Default: "Preprocessing"
            file_path = self._get_preprocessing_file_path(subject, run)
            file_type = "preprocessing"
            if file_path:
                file_paths_to_check.append(file_path)
        
        
        if not file_paths_to_check:
            return 'gray'
        
        # Single file mode - check the specific file
        file_path = file_paths_to_check[0]
        
        # First check if file exists on disk
        import os
        file_exists = os.path.exists(file_path)
        
        # If file exists on disk, it's up-to-date (black)
        # We don't need Snakemake to tell us a file that exists is complete
        if file_exists:
            return 'black'
        
        # File doesn't exist yet - check if it's in the pipeline scope
        in_current_scope = file_path in self.current_scope_files
        
        if not in_current_scope:
            # Not in current config and doesn't exist → gray (out of scope)
            return 'gray'
        
        # Get status from current scope map
        status_info = self.current_scope_files[file_path]
        status = status_info.get('status', '')
        plan = status_info.get('plan', '')
        
        # Check if file is up to date
        # File is up-to-date if it exists (ok) AND Snakemake says no update needed
        is_up_to_date = (status == 'ok' and 'no update' in plan.lower())
        
        # Special handling for "Run on current selection only" mode
        if self.run_current_only_mode:
            
            # Parse subject/task/run from parameters
            # subject format: "sub-15" → "15"
            # run format: "task-IWHD_run-01" → task="IWHD", run_id="01"
            subject_match = re.search(r'sub-(\w+)', subject) if subject else None
            task_match = re.search(r'task-(.+?)(?:_run-|$)', run) if run else None
            run_match = re.search(r'run-(\w+)', run) if run else None
            
            file_subject = subject_match.group(1) if subject_match else None
            file_task = task_match.group(1) if task_match else None
            file_run = run_match.group(1) if run_match else None
            
            # Check if this file matches the current selection
            is_selected = (
                file_subject == self.current_selection_subject and
                file_task == self.current_selection_task and
                file_run == self.current_selection_run
            )
            
            
            if is_selected:
                # Tier 1: File matches current selection (will be processed)
                if is_up_to_date:
                    return 'black'
                else:
                    # Needs update and will be processed
                    color = 'orange' if self._is_pipeline_running() else 'red'
                    return color
            else:
                # Tier 2: File in GUI config but not selected (won't be processed this run)
                if is_up_to_date:
                    return 'black'
                else:
                    # Needs update but won't be processed in this run
                    return 'red'
        
        # Normal mode (not in "run current only")
        if is_up_to_date:
            # In scope and up to date
            return 'black'
        else:
            # In scope and needs update
            # Orange if pipeline running, red if stopped
            color = 'orange' if self._is_pipeline_running() else 'red'
            return color
    
    def _check_file_exists(self, subject, run):
        """Check if the expected output file exists"""
        try:
            file_path = self._get_expected_file_path(subject, run)
            if file_path and os.path.exists(file_path):
                return True
            return False
        except Exception as e:
            print(f"Error checking file existence: {str(e)}")
            return False
    
    def _is_file_processed(self, subject, run):
        """Check if a file has been processed based on mtime vs last_run_time"""
        if not self.pipeline_status.get('last_run_time'):
            # No pipeline run yet, file is not processed
            return False
        
        try:
            # Get the expected file path
            file_path = self._get_expected_file_path(subject, run)
            if not file_path or not os.path.exists(file_path):
                return False
            
            # Get file modification time
            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            last_run_time = datetime.fromisoformat(self.pipeline_status['last_run_time'])
            
            
            # File is processed if modified after (or close to) pipeline start
            # Allow 5 second tolerance for clock sync issues
            return file_mtime >= (last_run_time - timedelta(seconds=5))
            
        except Exception as e:
            print(f"Error checking file processing status: {str(e)}")
            return False
    
    def _get_preprocessing_file_path(self, subject, run):
        """Get the preprocessing file path (unique per run) for checking if run is in workflow"""
        if not self.snakemake_config_path:
            return None
        
        try:
            config_dir = os.path.dirname(self.snakemake_config_path)
            
            # Extract task and run number from combined run string
            # e.g., "task-STS_run-01" -> task="STS", run_num="01"
            task = run.split('task-')[1].split('_')[0] if 'task-' in run else 'unknown'
            run_num = run.split('run-')[1] if 'run-' in run else '01'
            
            # Construct preprocessing file path matching Snakemake output format:
            # derivatives/cedalion/XXX/Outputs/preprocessed_data/sub-10/sub-10_task-STS_run-01_nirs_preprocessed.snirf
            preproc_path = os.path.join(
                config_dir,
                'Outputs',
                'preprocessed_data',
                subject,
                f"{subject}_task-{task}_run-{run_num}_nirs_preprocessed.snirf"
            )
            return os.path.normpath(preproc_path)
        except Exception as e:
            return None
    
    def _has_preprocessing(self, subject, run):
        """Check if preprocessing file exists for this subject/run"""
        preproc_path = self._get_preprocessing_file_path(subject, run)
        return preproc_path and os.path.exists(preproc_path)
    
    def _get_expected_file_path(self, subject, run):
        """Get the expected HRF estimate file path for pipeline output based on Snakemake structure"""
        if not self.snakemake_config_path:
            return None
        
        try:
            # Get the config directory and dataset info
            config_dir = os.path.dirname(self.snakemake_config_path)
            dataset = self.snakemake_config.get('dataset', {})
            
            # Extract task name from run (e.g., "task-STS_run-01" -> "STS")
            task = run.split('task-')[1].split('_')[0] if 'task-' in run else 'unknown'
            
            # Get rec_str from hrf_estimation config (e.g., "conc", "od")
            hrf_config = self.snakemake_config.get('hrf_estimation', {})
            rec_str = hrf_config.get('rec_str', 'conc')
            
            # Construct expected output path matching Snakemake output format:
            # derivatives/cedalion/XXX/Outputs/hrf_estimate/sub-10/sub-10_task-STS_nirs_hrf_estimate_<rec_str>.nc
            file_path = os.path.join(
                config_dir,
                'Outputs',
                'hrf_estimate',
                subject,
                f"{subject}_{run.split('_run-')[0]}_nirs_hrf_estimate_{rec_str}.nc"
            )
            
            # Normalize to match summary output format
            file_path = os.path.normpath(file_path)
            
            return file_path
            
        except Exception as e:
            return None
    
    def _get_image_recon_file_path(self, subject, run):
        """Get the expected image reconstruction file path for pipeline output"""
        if not self.snakemake_config_path:
            return None
        
        try:
            import yaml
            config_dir = os.path.dirname(self.snakemake_config_path)
            
            # Load config to get image_recon parameters
            with open(self.snakemake_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract task name from run (e.g., "task-STS_run-01" -> "STS")
            task = run.split('task-')[1].split('_')[0] if 'task-' in run else 'unknown'
            
            # Extract subject number (e.g., "sub-752" -> "752")
            subj_num = subject.split('-')[1] if '-' in subject else subject
            
            # Build filename matching Snakefile's get_imagerecon_output() function
            img_cfg = config.get('image_recon', {})
            
            # Get alpha values and convert to string (handle both string and numeric types from YAML)
            alpha_spatial = img_cfg.get('alpha_spatial', '1e-3')
            alpha_meas = img_cfg.get('alpha_meas', '1e4')
            # Convert to string as-is if string, otherwise use default string conversion
            alpha_spatial_str = str(alpha_spatial)
            alpha_meas_str = str(alpha_meas)
            
            filename = (
                f"Xs_sub-{subj_num}_{task}"
                f"_cov_alpha_spatial_{alpha_spatial_str}"
                f"_alpha_meas_{alpha_meas_str}"
                f"_recon_mode_{img_cfg.get('recon_mode', 'mua2conc')}"
                f"{'_Cmeas' if img_cfg.get('Cmeas', {}).get('enable', False) else '_noCmeas'}"
                f"{'_SB' if img_cfg.get('spatial_basis', {}).get('enable', False) else '_noSB'}"
                f"{'_mag' if img_cfg.get('mag', {}).get('enable', False) else '_ts'}"
            )
            
            # Add time window if mag is enabled
            if img_cfg.get('mag', {}).get('enable', False):
                t_win = img_cfg.get('mag', {}).get('t_win', [5, 8])
                filename += f"_{t_win[0]}_{t_win[1]}"
            
            filename += ".nc"
            
            # Image reconstruction file path (individual subject):
            file_path = os.path.join(
                config_dir,
                'Outputs',
                'image_results',
                subject,
                filename
            )
            
            # Normalize to match summary output format
            file_path = os.path.normpath(file_path)
            
            if hasattr(self, 'current_scope_files') and len(self.current_scope_files) > 0:
                # Show ALL paths in the summary for debugging
                all_paths = list(self.current_scope_files.keys())
                # Show matching paths in the summary
                matching = [p for p in self.current_scope_files.keys() if 'image_results' in p and subject in p]
                if matching:
                    pass
                else:
                    pass
            
            return file_path
            
        except Exception as e:
            return None
    
    def _mark_updated_files_for_reload(self, files_changed_to_black):
        """Clear cache for files that changed to black so they reload fresh on next selection"""
        cleared_count = 0
        for subject, run in files_changed_to_black:
            cache_key = (subject, run)
            if cache_key in self.cache:
                del self.cache[cache_key]
                cleared_count += 1
                print(f"Cleared cache for {subject} {run} - will reload fresh processed data on next selection")
            
            # Also update file_map to point to newly processed files
            self._update_file_map_for_processed_data(subject, run)
        
        if cleared_count > 0:
            self.statbar.showMessage(f"Cleared cache for {cleared_count} newly completed file(s) - fresh data will load on selection")
    
    def _auto_reload_current_file(self):
        """Auto-reload the currently displayed file after it completes processing"""
        try:
            current_subj = self.subj.currentText()
            current_run = self.run.currentText()
            
            
            if current_subj and current_subj != "None" and current_run and current_run != "None":
                # First, update the file_map to point to the newly processed files
                self._update_file_map_for_processed_data(current_subj, current_run)
                
                # Force a replot by calling the selection changed handler with the current run text
                self._run_changed(current_run, subject_changed=False)
                self.statbar.showMessage(f"Reloaded {current_subj} {current_run} with fresh data")
            else:
                pass
        except Exception as e:
            print(f"ERROR auto-reloading file: {str(e)}")
            import traceback
            traceback.print_exc()

    def _refresh_current_hrf_availability(self):
        """Reload current run when a newly created HRF file should enable HRF view."""
        try:
            if not hasattr(self, 'subj') or not hasattr(self, 'run'):
                return False

            current_subj = self.subj.currentText()
            current_run = self.run.currentText()
            if not current_subj or current_subj == "None" or not current_run or current_run == "None":
                return False

            hrf_path = self._get_expected_file_path(current_subj, current_run)
            if not hrf_path or not os.path.exists(hrf_path):
                return False

            cache_key = (current_subj, current_run)
            cached_hrf = self.cache.get(cache_key, {}).get('hrf_data') if cache_key in self.cache else None
            if cached_hrf is not None and self.hrf_view.isEnabled():
                return False

            if cache_key in self.cache:
                del self.cache[cache_key]

            self._update_file_map_for_processed_data(current_subj, current_run)
            self._run_changed(current_run, subject_changed=False)
            self.statbar.showMessage(f"HRF available for {current_subj} {current_run}", 5000)
            return True
        except Exception as e:
            print(f"ERROR refreshing HRF availability: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _update_file_map_for_processed_data(self, subject, run):
        """Update file_map to point to newly processed files after pipeline completion"""
        try:
            
            # Construct the path to the newly processed file
            if not self.snakemake_config_path:
                return
            
            config_dir = os.path.dirname(self.snakemake_config_path)
            
            # Extract task and run number from combined run string
            task = run.split('task-')[1].split('_')[0] if 'task-' in run else 'unknown'
            run_num = run.split('run-')[1] if 'run-' in run else '01'
            
            # Construct preprocessed file path
            preproc_path = os.path.join(
                config_dir,
                'Outputs',
                'preprocessed_data',
                subject,
                f"{subject}_task-{task}_run-{run_num}_nirs_preprocessed.snirf"
            )
            preproc_path = os.path.normpath(preproc_path)
            
            # Check if the file exists
            if os.path.exists(preproc_path):
                
                # Update file_map
                if subject not in self.file_map:
                    self.file_map[subject] = {}
                if run not in self.file_map[subject]:
                    self.file_map[subject][run] = {}
                
                # Update the pkl_path to point to the processed file
                old_path = self.file_map[subject][run].get('pkl_path')
                self.file_map[subject][run]['pkl_path'] = preproc_path
            else:
                pass
                
        except Exception as e:
            print(f"ERROR updating file_map: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _check_any_file_type_changed_to_black(self, subject, run):
        """Check if any file type (preprocessing, HRF, or image recon) just completed for this subject/run
        
        This checks all file types to see if any are now up-to-date in the current scope,
        which would indicate they were just processed and should trigger an auto-reload.
        """
        try:
            
            # Define the three file types to check
            file_types = [
                ("preprocessing", self._get_preprocessing_file_path(subject, run)),
                ("HRF", self._get_expected_file_path(subject, run)),
                ("image_recon", self._get_image_recon_file_path(subject, run))
            ]
            
            for file_type_name, file_path in file_types:
                if not file_path:
                    continue
                    
                if file_path not in self.current_scope_files:
                    continue
                
                status_info = self.current_scope_files[file_path]
                status = status_info.get('status', '')
                plan = status_info.get('plan', '')
                is_up_to_date = (status == 'ok' and 'no update' in plan.lower())
                
                
                if is_up_to_date:
                    # File is up-to-date in the current scope
                    # This likely means it was just processed (since cache was cleared)
                    # Or if running "current selection only", this is the file being processed
                    return True
            
            return False
            
        except Exception as e:
            print(f"ERROR _check_any_file_type_changed_to_black: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _reapply_run_colors(self, subject, runs):
        """Reapply colors to run items after run combobox is repopulated"""
        if not self.snakemake_config:
            return
        
        dataset = self.snakemake_config.get('dataset', {})
        config_subjects = dataset.get('subject', [])
        config_runs = dataset.get('run', [])
        
        for run in runs:
            if run == "None":
                continue
            color = self._get_file_color(subject, run, config_subjects, config_runs)
            self.file_colors[(subject, run)] = color
            self._apply_run_color_to_combobox(subject, run, color)
        
        # Update the stylesheet for the currently selected run
        self._update_combobox_selection_colors()

    def _is_dark_mode(self):
        """Detect whether the active Qt palette is dark."""
        return self.palette().color(QtGui.QPalette.Window).lightness() < 128

    def _theme_color(self, role):
        """Return theme-aware colors for manually styled widgets."""
        dark = self._is_dark_mode()
        colors = {
            "text": "#F0F0F0" if dark else "#111111",
            "muted": "#A8A8A8" if dark else "#666666",
            "background": "#2B2B2B" if dark else "#F0F0F0",
            "panel": "#333333" if dark else "#FFFFFF",
            "border": "#5A5A5A" if dark else "#C8C8C8",
        }
        return colors[role]

    def _status_color_map(self):
        """Semantic pipeline status colors with enough contrast in light/dark mode."""
        if self._is_dark_mode():
            return {
                'red': '#FF6B6B',
                'orange': '#FFB84D',
                'gray': '#A8A8A8',
                'black': '#F0F0F0',
            }

        return {
            'red': '#C62828',
            'orange': '#D06B00',
            'gray': '#777777',
            'black': '#111111',
        }

    def _combobox_status_stylesheet(self, text_color):
        """Preserve native combobox theme while overriding only readable text colors."""
        return (
            f"QComboBox {{ color: {text_color}; }}"
            f"QComboBox QAbstractItemView {{ color: {self._theme_color('text')}; "
            f"background-color: {self._theme_color('panel')}; "
            f"selection-color: {self._theme_color('text')}; }}"
        )

    def _status_bar_stylesheet(self):
        return (
            "QStatusBar { "
            f"color: {self._theme_color('text')}; "
            f"background-color: {self._theme_color('background')}; "
            "padding: 5px; font-size: 11pt; "
            "}"
        )
    
    def _update_combobox_selection_colors(self):
        """Update the color of the currently selected text in comboboxes"""
        color_map = self._status_color_map()
        
        # Update subject combobox current selection color
        if hasattr(self, 'subj'):
            current_subj = self.subj.currentText()
            if current_subj != "None":
                # Aggregate colors for this subject using priority: black > orange > red > gray
                subject_color = 'gray'
                for (subj, run), color in self.file_colors.items():
                    if subj == current_subj:
                        if color == 'black':
                            subject_color = 'black'
                            break
                        elif color == 'orange' and subject_color not in ['black']:
                            subject_color = 'orange'
                        elif color == 'red' and subject_color not in ['black', 'orange']:
                            subject_color = 'red'
                
                if subject_color:
                    self.subj.setStyleSheet(self._combobox_status_stylesheet(color_map[subject_color]))
        
        # Update run combobox current selection color
        if hasattr(self, 'run'):
            current_run = self.run.currentText()
            current_subj = self.subj.currentText()
            if current_run != "None" and current_subj != "None":
                run_color = self.file_colors.get((current_subj, current_run))
                if run_color:
                    self.run.setStyleSheet(self._combobox_status_stylesheet(color_map[run_color]))
        
        # Update image reconstruction button color for current selection
        self._update_image_recon_button_color()
    
    def _apply_color_to_combobox(self, subject, run, color):
        """Apply color styling to combobox items for subject/run"""
        self._apply_run_color_to_combobox(subject, run, color)
        self._apply_subject_color_to_combobox(subject, color)
    
    def _apply_subject_color_to_combobox(self, subject, color):
        """Apply color to subject in subject dropdown"""
        color_map = self._status_color_map()
        
        try:
            if hasattr(self, 'subj'):
                idx = self.subj.findText(subject)
                if idx >= 0:
                    self.subj.setItemData(idx, QtGui.QColor(color_map[color]), QtCore.Qt.ForegroundRole)
        except Exception as e:
            print(f"Error applying color to subject combobox: {str(e)}")
    
    def _apply_run_color_to_combobox(self, subject, run, color):
        """Apply color to run in run dropdown"""
        # Map colors to Qt stylesheet colors
        color_map = self._status_color_map()
        
        try:
            # Update run combobox only if the current subject matches
            if hasattr(self, 'run') and hasattr(self, 'subj'):
                current_subject = self.subj.currentText()
                if current_subject == subject:
                    idx = self.run.findText(run)
                    if idx >= 0:
                        self.run.setItemData(idx, QtGui.QColor(color_map[color]), QtCore.Qt.ForegroundRole)
        except Exception as e:
            print(f"Error applying color to run combobox: {str(e)}")


def run_vis(gui_data: dict):
    """Opens the visualization GUI.

    Args:
        gui_data: A dictionary containing the data for visualization.
    """

    app = QtWidgets.QApplication(sys.argv)
    main_gui = _MAIN_GUI(gui_data=gui_data)
    main_gui.show()
    sys.exit(app.exec())
