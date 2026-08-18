"""Interactive GUI to display Schaefer-17-network brain parcels on the cortical surface.

Parcel naming convention (Schaefer2018, 600 parcels / 17 networks), e.g.
``ContA_PFCl_1_LH``:
    - network:   ContA                     (one of 17 top-level networks)
    - subregion: PFCl                      (named region within the network)
    - instance:  1                         (numbered patch of that subregion)
    - hemi:      LH / RH                   (hemisphere, always the trailing token)

Some networks (e.g. ``TempPar``, ``SomMotA``) have no named subregion at all,
just numbered patches directly under the network (e.g. ``TempPar_3_RH``) - for
those there is nothing meaningful to expose in the sub-parcel selector.

Patch/instance numbers are an artifact of how the atlas was generated and are
not shown anywhere in the UI - selection and the legend are always expressed
in terms of network, sub-parcel (named subregion), and hemisphere only.
"""

import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
# qtpy (not a hardcoded binding) so this stays compatible when launched as a
# child window from time_series_gui.py, which runs on PySide6.
from qtpy import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor

import cedalion.data

BACKGROUND_COLOR = np.array([200, 200, 200], dtype=np.uint8)
BACKGROUND_PARCEL = "Background+FreeSurfer_Defined_Medial_Wall"

# Camera direction (viewed FROM) and view-up vector for each named view.
# Mesh axes are Right(+x)/Anterior(+y)/Superior(+z) - see _set_camera_view.
VIEW_DIRECTIONS = {
    "Anterior": ((0, 1, 0), (0, 0, 1)),
    "Posterior": ((0, -1, 0), (0, 0, 1)),
    "Left": ((-1, 0, 0), (0, 0, 1)),
    "Right": ((1, 0, 0), (0, 0, 1)),
    "Superior": ((0, 0, 1), (0, 1, 0)),
    "Inferior": ((0, 0, -1), (0, 1, 0)),
}

# Qualitative palette for distinguishing sub-parcels from one another when
# "combine" is off. Concatenating several tab20 variants gives 60 distinct,
# well-separated colors before any repeats.
SUBPARCEL_PALETTE = [
    tuple(int(c * 255) for c in color)
    for cmap_name in ("tab20", "tab20b", "tab20c")
    for color in plt.get_cmap(cmap_name).colors
]


def palette_color(index):
    return np.array(SUBPARCEL_PALETTE[index % len(SUBPARCEL_PALETTE)], dtype=np.uint8)

NAME_RE = re.compile(r"^(?P<network>[A-Za-z]+)_(?P<rest>.+)_(?P<hemi>LH|RH)$")


def parse_parcel_name(name):
    """Split a parcel name into (network, subregion, hemi).

    ``subregion`` is ``None`` when the network has no named subregion and the
    parcel is just a numbered patch directly under the network
    (e.g. ``TempPar_3_RH``).
    """
    m = NAME_RE.match(name)
    if not m:
        return None
    network = m.group("network")
    hemi = m.group("hemi")
    rest = m.group("rest")
    if rest.isdigit():
        return network, None, hemi
    parts = rest.rsplit("_", 1)
    subregion = parts[0] if len(parts) == 2 and parts[1].isdigit() else rest
    return network, subregion, hemi


class ParcelData:
    """Loads and indexes a head model's parcellation for lookup by network/subparcel/hemi."""

    def __init__(self, head_model="colin27"):
        if head_model.lower() == "icbm152":
            hmfiles = cedalion.data.get_icbm152_headmodel_files()
        else:
            hmfiles = cedalion.data.get_colin27_headmodel_files()
        self.parcel_colors = hmfiles.load_parcel_colors()
        bvc = hmfiles.load_brain_vertex_coordinates()

        self.vertex_parcel = bvc.sort_values("vertex")["parcel"].to_numpy()

        mesh_path = hmfiles.basedir / hmfiles.brain_surface_obj
        self.mesh = pv.read(str(mesh_path))
        if self.mesh.n_points != len(self.vertex_parcel):
            raise RuntimeError(
                f"Mesh has {self.mesh.n_points} vertices but "
                f"brain_vertex_coordinates has {len(self.vertex_parcel)} rows"
            )

        # parcel name -> boolean vertex mask, for fast color assignment
        self.parcel_vertex_mask = {
            name: self.vertex_parcel == name for name in np.unique(self.vertex_parcel)
        }

        # network -> hemi -> [full parcel names]  (every instance, any subregion)
        self.network_hemi_names = defaultdict(lambda: defaultdict(list))
        # (network, subregion) -> hemi -> [full parcel names], subregion may be None
        self.subregion_hemi_names = defaultdict(lambda: defaultdict(list))
        network_colors = defaultdict(list)

        for name in self.parcel_colors:
            parsed = parse_parcel_name(name)
            if parsed is None:
                continue
            network, subregion, hemi = parsed
            self.network_hemi_names[network][hemi].append(name)
            self.subregion_hemi_names[(network, subregion)][hemi].append(name)
            network_colors[network].append(self.parcel_colors[name])

        self.network_names = sorted(self.network_hemi_names.keys())
        self.network_color = {
            net: np.mean(colors, axis=0).astype(np.uint8)
            for net, colors in network_colors.items()
        }

        # network -> sorted list of named subregions (excludes bare-patch networks)
        self.network_subregions = defaultdict(list)
        for (network, subregion) in self.subregion_hemi_names:
            if subregion is not None:
                self.network_subregions[network].append(subregion)
        for network in self.network_subregions:
            self.network_subregions[network].sort()

    def subparcel_keys_for(self, networks, hemis):
        """Sub-parcel selector entries ("Network_Subregion") available for the
        given networks, restricted to ones present in at least one selected hemi."""
        keys = []
        for network in networks:
            for subregion in sorted(self.network_subregions.get(network, [])):
                hemi_map = self.subregion_hemi_names[(network, subregion)]
                if any(hemi_map.get(hemi) for hemi in hemis):
                    keys.append(f"{network}_{subregion}")
        return keys

    def names_for_networks(self, networks, hemis):
        """All parcel instances (every subregion) under the given networks/hemis."""
        names = []
        for network in networks:
            for hemi in hemis:
                names.extend(self.network_hemi_names[network].get(hemi, []))
        return names

    def names_for_subparcel_keys(self, keys, hemis):
        """All parcel instances matching the selected "Network_Subregion" keys."""
        names = []
        for key in keys:
            network, subregion = key.split("_", 1)
            hemi_map = self.subregion_hemi_names.get((network, subregion), {})
            for hemi in hemis:
                names.extend(hemi_map.get(hemi, []))
        return names

    @staticmethod
    def group_key(name, combine):
        """The color-grouping key for a parcel: its network (combine=True) or
        its "Network_Subregion" sub-parcel (combine=False)."""
        network, subregion, _hemi = parse_parcel_name(name)
        if combine:
            return network
        return f"{network}_{subregion}" if subregion else network

    def group_colors(self, selected_parcels, combine):
        """Map each distinct group among selected_parcels to a display color."""
        if combine:
            networks = sorted({parse_parcel_name(n)[0] for n in selected_parcels})
            return {net: self.network_color[net] for net in networks}
        keys = sorted({self.group_key(n, False) for n in selected_parcels})
        return {key: palette_color(i) for i, key in enumerate(keys)}

    def color_array(self, selected_parcels, combine):
        """Returns (per-vertex RGB array, {group_key: color} used for it)."""
        colors = np.tile(BACKGROUND_COLOR, (len(self.vertex_parcel), 1))
        group_colors = self.group_colors(selected_parcels, combine)
        for name in selected_parcels:
            mask = self.parcel_vertex_mask.get(name)
            if mask is None:
                continue
            colors[mask] = group_colors[self.group_key(name, combine)]
        return colors, group_colors

    def group_vertex_masks(self, selected_parcels, combine):
        """{group_key: boolean vertex mask}, unioning all member parcels' vertices.

        Mirrors the grouping in color_array/group_colors so a curve extracted
        for a group corresponds exactly to what's painted on the brain for it.
        """
        n_vertices = len(self.vertex_parcel)
        masks = defaultdict(lambda: np.zeros(n_vertices, dtype=bool))
        for name in selected_parcels:
            mask = self.parcel_vertex_mask.get(name)
            if mask is None:
                continue
            masks[self.group_key(name, combine)] |= mask
        return dict(masks)


class MultiSelectButton(QtWidgets.QToolButton):
    """A dropdown button with a checkbox per item, for multi-selection.

    Uses QWidgetAction-wrapped QCheckBoxes so clicking an item toggles it
    without closing the popup menu.
    """

    selectionChanged = QtCore.Signal()

    def __init__(self, label, parent=None):
        super().__init__(parent)
        self._label = label
        self._checkboxes = {}
        self.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self._menu = QtWidgets.QMenu(self)
        self.setMenu(self._menu)
        self._update_text()

    def set_items(self, names, checked_names=None):
        checked_names = set(checked_names or [])
        self._menu.clear()
        self._checkboxes = {}
        for name in names:
            checkbox = QtWidgets.QCheckBox(name, self._menu)
            checkbox.setChecked(name in checked_names)
            checkbox.stateChanged.connect(self._on_item_toggled)
            action = QtWidgets.QWidgetAction(self._menu)
            action.setDefaultWidget(checkbox)
            self._menu.addAction(action)
            self._checkboxes[name] = checkbox
        self._update_text()

    def _on_item_toggled(self, _state):
        self._update_text()
        self.selectionChanged.emit()

    def checked_items(self):
        return [name for name, cb in self._checkboxes.items() if cb.isChecked()]

    def _update_text(self):
        checked = self.checked_items()
        total = len(self._checkboxes)
        if total == 0:
            self.setText(f"{self._label}: (none available)")
        elif not checked:
            self.setText(f"{self._label}: (none selected)")
        elif len(checked) == total:
            self.setText(f"{self._label}: all ({', '.join(sorted(checked))})")
        else:
            self.setText(f"{self._label}: {', '.join(sorted(checked))}")


class ParcelViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, data: ParcelData, hrf_data=None):
        """
        Args:
            data: ParcelData for the head model the parcellation is drawn on.
            hrf_data: optional reconstructed image-space DataArray (from an
                image_recon.py ``Xs_*.nc`` output) with a ``vertex`` dimension
                on the SAME head model as ``data``, and typically ``chromo``
                and ``trial_type``/``time`` dimensions. When provided, a
                time-series panel is added below the 3D view that plots the
                selected parcel(s)' averaged HRF curve.
        """
        super().__init__()
        self.data = data
        self.hrf_data = self._to_brain_only(hrf_data)
        hrf_data = self.hrf_data
        self.setWindowTitle("Cedalion Brain Parcel Viewer")
        self.resize(1300, 1000 if hrf_data is not None else 800)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        central_layout = QtWidgets.QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)

        viewer_widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(viewer_widget)

        controls = QtWidgets.QWidget()
        controls.setFixedWidth(340)
        controls_layout = QtWidgets.QVBoxLayout(controls)

        hemi_box = QtWidgets.QGroupBox("Hemisphere")
        hemi_layout = QtWidgets.QHBoxLayout(hemi_box)
        self.lh_checkbox = QtWidgets.QCheckBox("LH")
        self.rh_checkbox = QtWidgets.QCheckBox("RH")
        self.lh_checkbox.setChecked(True)
        self.rh_checkbox.setChecked(True)
        hemi_layout.addWidget(self.lh_checkbox)
        hemi_layout.addWidget(self.rh_checkbox)
        controls_layout.addWidget(hemi_box)

        controls_layout.addWidget(QtWidgets.QLabel("Networks (17):"))
        self.network_button = MultiSelectButton("Networks")
        self.network_button.set_items(self.data.network_names)
        controls_layout.addWidget(self.network_button)

        controls_layout.addWidget(QtWidgets.QLabel("Sub-parcels:"))
        self.subparcel_button = MultiSelectButton("Sub-parcels")
        controls_layout.addWidget(self.subparcel_button)

        self.combine_checkbox = QtWidgets.QCheckBox(
            "Combine sub-parcels (one color per network)"
        )
        controls_layout.addWidget(self.combine_checkbox)

        view_box = QtWidgets.QGroupBox("Camera View")
        view_layout = QtWidgets.QGridLayout(view_box)
        view_buttons = [
            ("Anterior", 0, 0), ("Superior", 0, 1), ("Posterior", 0, 2),
            ("Left", 1, 0), ("Inferior", 1, 1), ("Right", 1, 2),
        ]
        for label, row, col in view_buttons:
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(
                lambda _checked, name=label: self._set_camera_view(name)
            )
            view_layout.addWidget(button, row, col)
        reset_zoom_button = QtWidgets.QPushButton("Reset Zoom")
        reset_zoom_button.clicked.connect(self._reset_zoom)
        view_layout.addWidget(reset_zoom_button, 2, 0, 1, 3)
        controls_layout.addWidget(view_box)

        controls_layout.addWidget(QtWidgets.QLabel("Legend:"))
        self.legend_list = QtWidgets.QListWidget()
        controls_layout.addWidget(self.legend_list, stretch=1)

        controls_layout.addStretch()
        layout.addWidget(controls)

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor, stretch=1)

        self.mesh = self.data.mesh.copy()
        self.mesh.point_data["colors"] = np.tile(
            BACKGROUND_COLOR, (self.mesh.n_points, 1)
        )
        self.plotter.add_mesh(
            self.mesh, scalars="colors", rgb=True, smooth_shading=True
        )
        self.plotter.set_background("white")

        self._last_networks = set()

        if self.hrf_data is not None:
            splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            splitter.addWidget(viewer_widget)
            splitter.addWidget(self._build_timeseries_panel())
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 2)
            central_layout.addWidget(splitter)
        else:
            central_layout.addWidget(viewer_widget)

        self.lh_checkbox.toggled.connect(self._on_hemi_toggled)
        self.rh_checkbox.toggled.connect(self._on_hemi_toggled)
        self.network_button.selectionChanged.connect(self._on_network_changed)
        self.subparcel_button.selectionChanged.connect(self._on_selection_changed)
        self.combine_checkbox.toggled.connect(self._on_selection_changed)

        self._on_network_changed()

    def _to_brain_only(self, hrf_data):
        """Reduce hrf_data's vertex dim to brain-only vertices matching self.data.

        image_recon.py's Xs can include scalp vertices too (when BRAIN_ONLY is
        disabled), concatenated after the brain ones, with an ``is_brain``
        boolean coordinate marking which is which. Parcel lookups here are
        brain-only and positionally aligned to that first block, so strip the
        scalp vertices (if any) before anything indexes into hrf_data by
        parcel vertex mask.
        """
        if hrf_data is None:
            return None

        if "vertex" not in hrf_data.dims:
            return hrf_data

        if "is_brain" in hrf_data.coords:
            hrf_data = hrf_data.isel(vertex=hrf_data["is_brain"].values)

        n_expected = len(self.data.vertex_parcel)
        if hrf_data.sizes["vertex"] != n_expected:
            raise ValueError(
                f"hrf_data has {hrf_data.sizes['vertex']} brain vertices but the "
                f"selected head model has {n_expected}. They must be the same "
                f"head model as the one used for image reconstruction."
            )
        return hrf_data

    def _build_timeseries_panel(self):
        panel = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(panel)

        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(QtWidgets.QLabel("Trial type:"))
        self.trial_type_combo = QtWidgets.QComboBox()
        trial_types = (
            [str(t) for t in self.hrf_data.trial_type.values]
            if "trial_type" in self.hrf_data.dims
            else []
        )
        self.trial_type_combo.addItems(trial_types)
        self.trial_type_combo.setEnabled(bool(trial_types))
        self.trial_type_combo.currentTextChanged.connect(self._on_selection_changed)
        top_row.addWidget(self.trial_type_combo)
        top_row.addStretch()
        panel_layout.addLayout(top_row)

        self.timeseries_canvas = FigureCanvas(Figure(figsize=(8, 3)))
        self.timeseries_ax = self.timeseries_canvas.figure.subplots()
        panel_layout.addWidget(self.timeseries_canvas)

        return panel

    def _selected_hemis(self):
        hemis = []
        if self.lh_checkbox.isChecked():
            hemis.append("LH")
        if self.rh_checkbox.isChecked():
            hemis.append("RH")
        return hemis

    def _on_hemi_toggled(self):
        # enforce at least one hemisphere checked
        if not self.lh_checkbox.isChecked() and not self.rh_checkbox.isChecked():
            sender = self.sender()
            sender.blockSignals(True)
            sender.setChecked(True)
            sender.blockSignals(False)
            return
        self._on_network_changed()

    def _on_network_changed(self):
        networks = self.network_button.checked_items()
        hemis = self._selected_hemis()

        # auto-check every sub-parcel of a network the moment it gets selected
        # (the user can then narrow down by unchecking individual ones);
        # networks that were already selected keep whatever check state the
        # user left them in.
        previously_checked = set(self.subparcel_button.checked_items())
        newly_selected_networks = set(networks) - self._last_networks
        self._last_networks = set(networks)

        available = self.data.subparcel_keys_for(networks, hemis)
        checked = {k for k in previously_checked if k in available}
        checked |= {
            k for k in available if k.split("_", 1)[0] in newly_selected_networks
        }

        self.subparcel_button.set_items(available, checked_names=checked)
        self._on_selection_changed()

    def _on_selection_changed(self):
        networks = self.network_button.checked_items()
        hemis = self._selected_hemis()
        subparcel_keys = self.subparcel_button.checked_items()

        selected = self.data.names_for_subparcel_keys(subparcel_keys, hemis)
        # networks with no named subregion have nothing to show up in the
        # sub-parcel selector, so they're included in full whenever selected
        bare_networks = [n for n in networks if not self.data.network_subregions.get(n)]
        selected += self.data.names_for_networks(bare_networks, hemis)

        combine = self.combine_checkbox.isChecked()
        colors, group_colors = self.data.color_array(selected, combine)
        self.mesh.point_data["colors"] = colors
        self.plotter.render()
        self._update_legend(group_colors)

        if self.hrf_data is not None:
            group_masks = self.data.group_vertex_masks(selected, combine)
            self._update_timeseries_plot(group_masks, group_colors)

    def _update_legend(self, group_colors):
        self.legend_list.clear()
        for key, color in sorted(group_colors.items()):
            self._add_legend_row(key, color)

    def _update_timeseries_plot(self, group_masks, group_colors):
        ax = self.timeseries_ax
        ax.clear()

        data = self.hrf_data
        if "trial_type" in data.dims:
            trial_type = self.trial_type_combo.currentText()
            if not trial_type:
                self.timeseries_canvas.draw()
                return
            data = data.sel(trial_type=trial_type)

        keys = sorted(k for k in group_masks if group_masks[k].any())
        if not keys:
            self.timeseries_canvas.draw()
            return

        if "time" in data.dims:
            self._plot_group_curves(ax, data, keys, group_masks, group_colors)
        else:
            self._plot_group_magnitudes(ax, data, keys, group_masks, group_colors)

        ax.grid(True, axis="y")
        self.timeseries_canvas.figure.tight_layout()
        self.timeseries_canvas.draw()

    def _plot_group_curves(self, ax, data, keys, group_masks, group_colors):
        """Full HRF time-course per selected parcel group (HbO solid, HbR dashed)."""
        has_chromo = "chromo" in data.dims
        chromo_styles = {"HbO": "-", "HbR": "--"}

        for key in keys:
            group_data = data.isel(vertex=group_masks[key]).mean(dim="vertex")
            color = np.array(group_colors[key]) / 255.0

            if has_chromo:
                for chromo, linestyle in chromo_styles.items():
                    if chromo not in group_data.chromo.values:
                        continue
                    curve = group_data.sel(chromo=chromo)
                    ax.plot(
                        curve.time.values, curve.values, linestyle,
                        color=color, label=f"{key} ({chromo})",
                    )
            else:
                ax.plot(
                    group_data.time.values, group_data.values, "-",
                    color=color, label=key,
                )

        ax.set_xlabel("time (s)")
        ax.set_ylabel("concentration")

    def _plot_group_magnitudes(self, ax, data, keys, group_masks, group_colors):
        """Bar chart of the reconstructed magnitude (e.g. mean over the HRF
        t_win) per selected parcel group - used when there is no time
        dimension to plot as a curve (mag.enable in image_recon config)."""
        has_chromo = "chromo" in data.dims
        x = np.arange(len(keys))
        width = 0.35
        chromo_styles = {"HbO": (-width / 2, 1.0), "HbR": (width / 2, 0.5)}
        chromos = list(chromo_styles) if has_chromo else [None]

        for chromo in chromos:
            offset, alpha = chromo_styles[chromo] if has_chromo else (0.0, 1.0)
            values, colors = [], []
            for key in keys:
                group_data = data.isel(vertex=group_masks[key])
                if has_chromo:
                    if chromo not in group_data.chromo.values:
                        values.append(np.nan)
                        colors.append([0, 0, 0])
                        continue
                    group_data = group_data.sel(chromo=chromo)
                values.append(float(group_data.mean(dim="vertex").values))
                colors.append(np.array(group_colors[key]) / 255.0)
            ax.bar(
                x + offset, values, width, color=colors, alpha=alpha,
                edgecolor="black", linewidth=0.5, label=chromo or "magnitude",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("concentration (magnitude)")

    def _add_legend_row(self, label, rgb):
        item = QtWidgets.QListWidgetItem(label)
        pixmap = QtGui.QPixmap(12, 12)
        pixmap.fill(QtGui.QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
        item.setIcon(QtGui.QIcon(pixmap))
        self.legend_list.addItem(item)

    def _set_camera_view(self, name):
        # mesh axes are Right(+x)/Anterior(+y)/Superior(+z), verified against
        # known-anatomical parcel locations (e.g. visual cortex is posterior).
        direction, view_up = VIEW_DIRECTIONS[name]
        bounds = np.array(self.mesh.bounds).reshape(3, 2)
        center = bounds.mean(axis=1)
        radius = np.linalg.norm(bounds[:, 1] - center)
        camera_pos = center + np.array(direction) * radius * 4
        self.plotter.camera_position = [
            tuple(camera_pos), tuple(center), tuple(view_up)
        ]
        self.plotter.render()

    def _reset_zoom(self):
        self.plotter.reset_camera()
        self.plotter.render()


def main():
    app = QtWidgets.QApplication(sys.argv)
    data = ParcelData()
    window = ParcelViewerWindow(data)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
