import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import csv
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

if getattr(sys, "frozen", False):
    os.chdir(Path(sys.executable).resolve().parent)

from PIL import ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors
import CarbonScoreProcess


class _NullGuiStream:
    def write(self, _text):
        return 0

    def flush(self):
        return None

    def isatty(self):
        return False


if sys.stdout is None:
    sys.stdout = _NullGuiStream()
if sys.stderr is None:
    sys.stderr = _NullGuiStream()


def _close_carbon_thread_connection():
    conn = getattr(CarbonScoreProcess.thread_local, "conn", None)
    if conn is None:
        return
    try:
        conn.close()
    finally:
        try:
            delattr(CarbonScoreProcess.thread_local, "conn")
        except AttributeError:
            pass


from PROCESSSDFFILES import SDFFileSelector
from CarbonScorerResult import *
from HSQCScorerResult import *
from CombineScorerResult import *
import json
import re

from recqc_cluster import (
    DEFAULT_REGION_WINDOWS,
    format_c_values,
    format_hsqc_points,
    parse_c_values,
    parse_hsqc_points,
    parse_region_windows,
    run_c_common_mask_backend_clustering,
    run_c_presence_mask_clustering,
    run_c_v5_clustering,
    run_c_single_spectrum_clustering,
    run_hsqc_common_mask_backend_clustering,
    run_hsqc_cross_clustering,
    run_hsqc_v5_full_clustering,
    model_display_label,
    model_key_from_display,
)
from services.nmr_prediction_bridge import (
    DEFAULT_NMR_PREDICTOR_ROOT,
    annotate_prediction_sdf,
    build_nmr_prediction_launch,
    default_output_dir,
    describe_nmr_predictor_root,
    detect_input_type,
)
from services.nmr_assignment import (
    HydrogenAssignment,
    build_assignment,
    export_assignment_text,
    list_sdf_molecule_ids,
    prepare_display_molecule,
    write_assigned_sdf,
)

APP_NAME = "CORD-NMR"
APP_VERSION = "1.0.0"
APP_DISPLAY_NAME = f"{APP_NAME} {APP_VERSION}"

C_FRONTEND_LABELS = [
    "global strict",
    "common-mask",
]
DEFAULT_C_FRONTEND_LABEL = "global strict"
GLOBAL_STRICT_DEFAULTS = {
    "span": "1.2",
    "residual_gate": "1.0",
    "max_tracks_3": "32",
    "max_tracks_4": "28",
    "max_tracks_5": "26",
    "frac_3": "0.55",
    "frac_4": "0.47",
    "frac_5": "0.41",
    "setpacking_high_mask_bonus": 0.10,
    "qg_max_rise": 1,
}
COMMON_MASK_DEFAULTS = {
    "span": "1.0",
    "residual_gate": "1.0",
    "max_tracks_3": "28",
    "max_tracks_4": "24",
    "max_tracks_5": "22",
    "frac_3": "0.50",
    "frac_4": "0.43",
    "frac_5": "0.36",
    "setpacking_high_mask_bonus": 0.0,
    "qg_max_rise": 4,
}

CROSS_MODEL_LABELS = [
    model_display_label("v5_enum", include_full_name=False),
    model_display_label("v5_pmtc", include_full_name=False),
]
DEFAULT_CROSS_MODEL_LABEL = model_display_label("v5_enum", include_full_name=False)

class ChemTheme:

    COLORS = {
        'background': '#2E5266',     # original deep blue background
        'foreground': '#E2E8E4',     # original off-white working surface
        'accent1': '#6B8F71',        # original leaf green
        'accent2': '#A3C4BC',        # original turquoise
        'highlight': '#FFD166',      # original amber
        'surface': '#E2E8E4',
        'panel': '#A3C4BC',
        'muted': '#2E5266',
        'border': '#A3C4BC',
        'nav': '#2E5266'
    }

    FONTS = {
        'title': ('Helvetica', 24, 'bold'),
        'subtitle': ('Helvetica', 12),
        'label': ('Arial', 10),
        'input': ('Courier New', 10)
    }


    @classmethod
    def configure_style(cls):
        style = ttk.Style()
        style.theme_create('chem', parent='clam', settings={
            'TFrame': {'configure': {'background': cls.COLORS['background']}},
            'TLabel': {
                'configure': {
                    'foreground': cls.COLORS['foreground'],
                    'background': cls.COLORS['background'],
                    'font': cls.FONTS['label']
                }
            },
            'TButton': {
                'configure': {
                    'foreground': cls.COLORS['background'],
                    'background': cls.COLORS['accent1'],
                    'bordercolor': cls.COLORS['accent1'],
                    'focusthickness': 0,
                    'padding': (10, 5),
                    'font': cls.FONTS['label']
                },
                'map': {
                    'background': [('active', cls.COLORS['highlight'])]
                }
            },
            'TEntry': {
                'configure': {
                    'fieldbackground': cls.COLORS['foreground'],
                    'foreground': cls.COLORS['background'],
                    'bordercolor': cls.COLORS['border'],
                    'lightcolor': cls.COLORS['border'],
                    'darkcolor': cls.COLORS['border'],
                    'font': cls.FONTS['input']
                }
            },
            'TCombobox': {
                'configure': {
                    'fieldbackground': cls.COLORS['foreground'],
                    'foreground': cls.COLORS['background'],
                    'font': cls.FONTS['input']
                }
            },
            'TRadiobutton': {
                'configure': {
                    'foreground': cls.COLORS['background'],
                    'background': cls.COLORS['surface'],
                    'font': cls.FONTS['label']
                }
            },
            'TCheckbutton': {
                'configure': {
                    'foreground': cls.COLORS['background'],
                    'background': cls.COLORS['surface'],
                    'font': cls.FONTS['label']
                }
            },
            'TLabelframe': {
                'configure': {
                    'background': cls.COLORS['surface'],
                    'bordercolor': cls.COLORS['border'],
                    'relief': 'solid'
                }
            },
            'TLabelframe.Label': {
                'configure': {
                    'foreground': cls.COLORS['accent1'],
                    'background': cls.COLORS['background'],
                    'font': ('Helvetica', 11, 'bold')
                }
            },
        })
        style.theme_use('chem')
        style.configure('Header.TFrame', background=cls.COLORS['background'])
        style.configure('Header.TLabel', background=cls.COLORS['background'], foreground=cls.COLORS['highlight'])
        style.configure('Subheader.TLabel', background=cls.COLORS['background'], foreground=cls.COLORS['foreground'])
        style.configure('Nav.TFrame', background=cls.COLORS['nav'])
        style.configure('Card.TFrame', background=cls.COLORS['surface'])
        style.configure('Card.TLabel', background=cls.COLORS['surface'], foreground=cls.COLORS['background'])
        style.configure('Muted.TLabel', background=cls.COLORS['surface'], foreground=cls.COLORS['muted'])
        style.configure('Section.TLabel', background=cls.COLORS['surface'], foreground=cls.COLORS['background'], font=('Helvetica', 11, 'bold'))
        style.configure('Secondary.TButton', foreground=cls.COLORS['background'], background=cls.COLORS['accent2'], bordercolor=cls.COLORS['border'])
        style.map('Secondary.TButton', background=[('active', cls.COLORS['highlight'])])

class BaseModeFrame(ttk.Frame):
    """Schema base class framework"""
    def __init__(self, master, mode_name):
        super().__init__(master)
        self.mode_name = mode_name
        self.file_selector = SDFFileSelector()
        self.create_widgets()
        self.db_path = 'chem_data.db'

    def create_widgets(self):
        file_frame = ttk.Frame(self)
        ttk.Button(file_frame, text="Select the SDF file", command=self.select_files).pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT)
        file_frame.pack(pady=10, fill=tk.X)

        self.create_parameters()

        self.btn_frame = ttk.Frame(self)
        ttk.Button(self.btn_frame, text="Start the analysis", command=self.start_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.btn_frame, text="Reset parameter", command=self.reset_parameters).pack(side=tk.LEFT)
        self.btn_frame.pack(pady=10)

    def select_files(self):
        files = self.file_selector.select_files_via_gui()
        self.file_label.config(text="\n".join([Path(f).name for f in files]))

    def create_parameters(self):
        raise NotImplementedError("The parameter creation method must be implemented")

    def validate_input(self):
        raise NotImplementedError("An input validation method must be implemented")

    def start_analysis(self):
        if self.validate_input():
            self.run_analysis()

    def run_analysis(self):
        raise NotImplementedError("Analytical methods must be implemented")

    def reset_parameters(self):
        raise NotImplementedError("The reset method must be implemented")

    def show_results(self, results):
        pass


class CombinedModeFrame(BaseModeFrame):
    """Joint matching mode interface"""
    def __init__(self, master):
        super().__init__(master, "Joint Matching")

    def create_parameters(self):

        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        canvas = tk.Canvas(main_container, height=300, highlightthickness=0, bd=0)
        vsb = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)


        vsb.pack(side="right", fill="y", padx=0)
        canvas.pack(side="left", fill="both", expand=True, padx=0)

        scroll_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        def _configure_canvas(e):
            canvas.itemconfig("all", width=e.width)
            canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.bind("<Configure>", _configure_canvas)
        carbon_frame = ttk.LabelFrame(scroll_frame, text="C parameters")
        self.create_carbon_params(carbon_frame)
        carbon_frame.pack(fill=tk.X, padx=5, pady=2, anchor=tk.NW)

        score_frame = ttk.LabelFrame(scroll_frame, text="Scoring parameters")
        self.create_score_params(score_frame)
        score_frame.pack(fill=tk.X, padx=5, pady=2, anchor=tk.NW)

        hmqc_frame = ttk.LabelFrame(scroll_frame, text="HSQC parameters")
        self.create_hmqc_params(hmqc_frame)
        hmqc_frame.pack(fill=tk.X, padx=5, pady=2, anchor=tk.NW)

        advanced_frame = ttk.LabelFrame(scroll_frame, text="C-HSQC weight")
        self.create_advanced_params(advanced_frame)
        advanced_frame.pack(fill=tk.X, padx=5, pady=2, anchor=tk.NW)


    def create_carbon_params(self, parent):
        mode_frame = ttk.Frame(parent)
        self.c_mode = tk.StringVar(value='typed')
        self.c_merge = tk.StringVar(value='N')
        ttk.Radiobutton(mode_frame, text="C_typed", variable=self.c_mode,
                        value='typed', command=self.toggle_c_params).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="C_untyped", variable=self.c_mode,
                        value='untyped', command=self.toggle_c_params).pack(side=tk.LEFT)
        mode_frame.pack(anchor=tk.W)
        ttk.Label(mode_frame, text=" C_merge:").pack(side=tk.LEFT, padx=(10,0))
        ttk.Radiobutton(mode_frame, text="Y", variable=self.c_merge,
                        value='Y').pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="N", variable=self.c_merge,
                        value='N').pack(side=tk.LEFT)

        self.param_container = ttk.Frame(parent)

        self.typed_params = {}
        typed_frame = ttk.Frame(self.param_container)
        labels = ['    All_C   ', '+Dept_135',' -Dept_135', '  Dept_90 ']
        for label in labels:
            row = ttk.Frame(typed_frame)
            ttk.Label(row, text=f"{label}:").pack(side=tk.LEFT)
            entry = tk.Text(row, height=2, width=60)
            entry.pack(side=tk.LEFT, padx=5)
            self.typed_params[label] = entry
            row.pack(anchor=tk.W)
        self.typed_frame = typed_frame

        untyped_frame = ttk.Frame(self.param_container)
        self.global_entry = tk.Text(untyped_frame, height=3, width=60)
        ttk.Label(untyped_frame, text="All_C:").pack(side=tk.LEFT)
        self.global_entry.pack(side=tk.LEFT, padx=5)
        self.untyped_frame = untyped_frame

        self.param_container.pack()
        self.toggle_c_params()

        range_frame = ttk.Frame(parent)
        ttk.Label(range_frame, text="C_num range:").pack(side=tk.LEFT)
        self.c_min = ttk.Entry(range_frame, width=5)
        self.c_min.insert(0, "5")
        self.c_min.pack(side=tk.LEFT)
        ttk.Label(range_frame, text="-").pack(side=tk.LEFT)
        self.c_max = ttk.Entry(range_frame, width=5)
        self.c_max.insert(0, "30")
        self.c_max.pack(side=tk.LEFT)

        ttk.Label(range_frame, text=" | ").pack(side=tk.LEFT, padx=5)

        ttk.Label(range_frame, text="MW range:").pack(side=tk.LEFT)
        self.m_min = ttk.Entry(range_frame, width=5)
        self.m_min.insert(0, "100")
        self.m_min.pack(side=tk.LEFT)
        ttk.Label(range_frame, text="-").pack(side=tk.LEFT)
        self.m_max = ttk.Entry(range_frame, width=5)
        self.m_max.insert(0, "500")
        self.m_max.pack(side=tk.LEFT)

        range_frame.pack(pady=5)

    def create_score_params(self, parent):
        env_frame = ttk.Frame(parent)
        ttk.Label(env_frame, text="Env atom level:").pack(side=tk.LEFT)
        self.env_level = ttk.Combobox(env_frame, values=[1, 2, 3], width=3)
        self.env_level.set(1)
        self.env_level.pack(side=tk.LEFT)

        # 权重设置
        ttk.Label(env_frame, text="Self weight:").pack(side=tk.LEFT, padx=(10,0))
        self.self_weight = ttk.Entry(env_frame, width=5)
        self.self_weight.insert(0, "0.7")
        self.self_weight.pack(side=tk.LEFT)

        ttk.Label(env_frame, text="Env weight:").pack(side=tk.LEFT, padx=(10,0))
        self.env_weight = ttk.Entry(env_frame, width=5, state='readonly')
        self.env_weight.config(state='normal')
        self.env_weight.insert(0, "0.3")
        self.env_weight.config(state='readonly')
        self.env_weight.pack(side=tk.LEFT)

        self.self_weight.bind("<KeyRelease>", self.update_env_weight)
        env_frame.pack(anchor=tk.W)

        # 评分模式
        self.score_mode = tk.StringVar(value='global')
        ttk.Radiobutton(parent, text="Global mode", variable=self.score_mode,
                        value='global', command=self.toggle_score_mode).pack(anchor=tk.W)
        ttk.Radiobutton(parent, text="Fine model", variable=self.score_mode,
                        value='fine', command=self.toggle_score_mode).pack(anchor=tk.W)

        self.global_frame = ttk.Frame(parent)
        ttk.Label(self.global_frame, text="Global threshold:").pack(side=tk.LEFT)
        self.green_thresh = ttk.Entry(self.global_frame, width=8)
        self.green_thresh.insert(0, "0.5")
        self.green_thresh.pack(side=tk.LEFT)
        ttk.Label(self.global_frame, text="-").pack(side=tk.LEFT)
        self.yellow_thresh = ttk.Entry(self.global_frame, width=8)
        self.yellow_thresh.insert(0, "2.0")
        self.yellow_thresh.pack(side=tk.LEFT)
        self.global_frame.pack(anchor=tk.W, fill=tk.X)
        self.fine_container = ttk.Frame(parent)
        self.canvas = tk.Canvas(self.fine_container, height=60)
        scrollbar = ttk.Scrollbar(self.fine_container, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = ttk.Frame(self.canvas)

        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.range_rows = []
        self.add_range_row((0, 220), 1, 3)

        btn_frame = ttk.Frame(self.fine_container)
        ttk.Button(btn_frame, text="+ Add range",
                   command=lambda: self.add_range_row((0, 0), 0, 0)).pack(pady=5)
        btn_frame.pack(fill=tk.X)

        self.fine_container.pack_forget()

    def update_env_weight(self, event=None):
        try:
            self_val = float(self.self_weight.get() or 0)  # Handle empty input
            env_val = max(0.0, min(1.0, 1 - self_val))  # Clamp between 0-1
            self.env_weight.config(state='normal')
            self.env_weight.delete(0, tk.END)
            self.env_weight.insert(0, f"{env_val:.2f}")
            self.env_weight.config(state='readonly')
        except ValueError:
            self.env_weight.config(state='normal')
            self.env_weight.delete(0, tk.END)
            self.env_weight.insert(0, "1.00")  # Default value
            self.env_weight.config(state='readonly')

    def toggle_score_mode(self):
        if self.score_mode.get() == 'global':
            self.global_frame.pack(anchor=tk.W, fill=tk.X)
            self.fine_container.pack_forget()
        else:
            self.global_frame.pack_forget()
            self.fine_container.pack(fill=tk.X)

    def add_range_row(self, carbon_range, green_val, yellow_val):
        if len(self.range_rows) >= 10:
            return

        frame = ttk.Frame(self.scroll_frame)
        ttk.Label(frame, text="C_range:").pack(side=tk.LEFT)
        start_entry = ttk.Entry(frame, width=6)
        start_entry.insert(0, str(carbon_range[0]))
        start_entry.pack(side=tk.LEFT)
        ttk.Label(frame, text="-").pack(side=tk.LEFT)
        end_entry = ttk.Entry(frame, width=6)
        end_entry.insert(0, str(carbon_range[1]))
        end_entry.pack(side=tk.LEFT)
        # 阈值
        ttk.Label(frame, text="Green threshold:").pack(side=tk.LEFT, padx=(10,0))
        green_entry = ttk.Entry(frame, width=6)
        green_entry.insert(0, str(green_val))
        green_entry.pack(side=tk.LEFT)
        ttk.Label(frame, text="Yellow threshold:").pack(side=tk.LEFT, padx=(10,0))
        yellow_entry = ttk.Entry(frame, width=6)
        yellow_entry.insert(0, str(yellow_val))
        yellow_entry.pack(side=tk.LEFT)
        # 删除按钮
        ttk.Button(frame, text="×", width=2,
                   command=lambda: self.remove_range_row(frame)).pack(side=tk.RIGHT, padx=5)
        self.range_rows.append({
            'frame': frame,
            'start': start_entry,
            'end': end_entry,
            'green': green_entry,
            'yellow': yellow_entry
        })
        frame.pack(fill=tk.X, pady=2)

    def remove_range_row(self, frame):
        for row in self.range_rows:
            if row['frame'] == frame:
                frame.destroy()
                self.range_rows.remove(row)
                break

    def toggle_c_params(self):
        """切换参数输入界面"""
        for widget in self.param_container.winfo_children():
            widget.pack_forget()

        if self.c_mode.get() == 'typed':
            self.typed_frame.pack(anchor=tk.W)
        else:
            self.untyped_frame.pack(anchor=tk.W)

    def create_hmqc_params(self, parent):
        mode_frame = ttk.Frame(parent)
        ttk.Label(mode_frame, text="Matching pattern:").pack(side=tk.LEFT)
        self.hmqc_mode = ttk.Combobox(mode_frame, values=[1,2,3], state='readonly')
        self.hmqc_mode.current(1)
        self.hmqc_mode.pack(side=tk.LEFT)
        self.hmqc_mode.bind("<<ComboboxSelected>>", lambda e: self.create_hmqc_input_groups())
        self.ch_merge = tk.StringVar(value='N')  # 默认值设为N
        ttk.Label(mode_frame, text=" | CH_merge:").pack(side=tk.LEFT, padx=(10,0))
        ttk.Radiobutton(mode_frame, text="Y", variable=self.ch_merge,
                        value='Y').pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="N", variable=self.ch_merge,
                        value='N').pack(side=tk.LEFT)
        mode_frame.pack(anchor=tk.W)

        self.hmqc_input_container = ttk.Frame(parent)
        self.hmqc_input_container.pack(fill=tk.X, padx=5, pady=2)
        self.point_groups = {}
        self.create_hmqc_input_groups()

        tol_frame = ttk.Frame(parent)
        ttk.Label(tol_frame, text="C tolerance:").pack(side=tk.LEFT)
        self.c_tol = ttk.Entry(tol_frame, width=8)
        self.c_tol.insert(0, "1.0")
        self.c_tol.pack(side=tk.LEFT)
        ttk.Label(tol_frame, text="H tolerance:").pack(side=tk.LEFT)
        self.h_tol = ttk.Entry(tol_frame, width=8)
        self.h_tol.insert(0, "0.2")
        self.h_tol.pack(side=tk.LEFT)
        tol_frame.pack(anchor=tk.W)

    def create_hmqc_input_groups(self):
        for widget in self.hmqc_input_container.winfo_children():
            widget.destroy()
        self.point_groups.clear()

        current_mode = int(self.hmqc_mode.get())
        types_config = {
            1: ['All_type'],
            2: ['type13', 'type2'],
            3: ['type1', 'type2', 'type3']
        }

        for t in types_config[current_mode]:
            group = ttk.LabelFrame(self.hmqc_input_container, text=t)
            entry = tk.Text(group, height=3, width=80)
            scroll = ttk.Scrollbar(group, command=entry.yview)
            entry.config(yscrollcommand=scroll.set)
            entry.pack(side=tk.LEFT)
            scroll.pack(side=tk.LEFT, fill=tk.Y)
            self.point_groups[t] = entry
            group.pack(fill=tk.X, padx=5, pady=2, side=tk.TOP)

    def create_advanced_params(self, parent):
        weight_frame = ttk.Frame(parent)
        ttk.Label(weight_frame, text="C weight:").pack(side=tk.LEFT)
        self.c_weight = ttk.Entry(weight_frame, width=8)
        self.c_weight.insert(0, "0.7")
        self.c_weight.pack(side=tk.LEFT)
        ttk.Label(weight_frame, text="CH weight:").pack(side=tk.LEFT)
        self.ch_weight = ttk.Entry(weight_frame, width=8)
        self.ch_weight.insert(0, "0.3")
        self.ch_weight.pack(side=tk.LEFT)
        weight_frame.pack(anchor=tk.W)

    def validate_input(self):
        try:
            if not self.file_selector.selected_files:
                raise ValueError("Please select the SDF file first")

            if self.c_mode.get() == 'typed':
                for label in ['    All_C   ', '+Dept_135',' -Dept_135', '  Dept_90 ']:
                    entry = self.typed_params[label]
                    text = entry.get("1.0", tk.END).strip() if isinstance(entry, tk.Text) else entry.get().strip()
                    if not text:
                        raise ValueError(f"{label.strip()} cannot be empty")
            else:
                text = self.global_entry.get("1.0", tk.END).strip().replace('，', ',')
                if not text:
                    raise ValueError("Global mode requires a list of displacements")

            if not (self.c_min.get().isdigit() and self.c_max.get().isdigit()):
                raise ValueError("The carbon number range must be an integer")

            current_mode = int(self.hmqc_mode.get())
            required_fields = {
                1: ['All_type'],
                2: ['type13', 'type2'],
                3: ['type1', 'type2', 'type3']
            }[current_mode]

            pattern = r'^\(\s*(?:\d+\.?\d*|\.\d+)\s*,\s*(?:\d+\.?\d*|\.\d+)\s*\)(?:[,\s]*\(\s*(?:\d+\.?\d*|\.\d+)\s*,\s*(?:\d+\.?\d*|\.\d+)\s*\))*$'

            for field in required_fields:
                text = self.point_groups[field].get("1.0", tk.END).strip()
                # 统一符号格式
                text = text.replace('，', ',').replace('（', '(').replace('）', ')')
                if not text:
                    raise ValueError(f"{field} cannot be empty")
                if not re.match(pattern, text):
                    raise ValueError(f"{field} format error, should be (numbers, numbers) list, support decimals, do not support negative numbers")
            return True
        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return False

    def reset_parameters(self):

        self.c_mode.set('typed')
        self.toggle_c_params()

        for entry in self.typed_params.values():
            entry.delete("1.0", tk.END)

        self.global_entry.delete("1.0", tk.END)

        self.c_min.delete(0, tk.END)
        self.c_min.insert(0, "10")
        self.c_max.delete(0, tk.END)
        self.c_max.insert(0, "30")
        self.m_min.delete(0, tk.END)
        self.m_min.insert(0, "100")
        self.m_max.delete(0, tk.END)
        self.m_max.insert(0, "500")
        self.env_level.set(1)
        self.self_weight.delete(0, tk.END)
        self.self_weight.insert(0, "0.7")
        self.update_env_weight()
        self.score_mode.set('global')
        self.toggle_score_mode()
        self.green_thresh.delete(0, tk.END)
        self.green_thresh.insert(0, "0.5")
        self.yellow_thresh.delete(0, tk.END)
        self.yellow_thresh.insert(0, "2.0")

        for row in self.range_rows.copy():
            self.remove_range_row(row['frame'])
        self.add_range_row((0, 220), 1, 3)  # 添加默认范围

        self.hmqc_mode.current(1)
        self.create_hmqc_input_groups()
        self.c_tol.delete(0, tk.END)
        self.c_tol.insert(0, "1.0")
        self.h_tol.delete(0, tk.END)
        self.h_tol.insert(0, "0.2")

        self.c_weight.delete(0, tk.END)
        self.c_weight.insert(0, '0.7')
        self.ch_weight.delete(0, tk.END)
        self.ch_weight.insert(0, '0.3')

    def get_c_data(self):
        if self.c_mode.get() == 'typed':
            data = {
                label.strip(): [
                    float(x) for x in
                    (entry.get("1.0", tk.END).strip() if isinstance(entry, tk.Text) else entry.get().strip()
                     ).replace('，', ',')  # Replace Chinese commas with English commas
                    .split(',')]
                for label, entry in self.typed_params.items()
            }

            all_c = data['All_C']
            plus_dept = data['+Dept_135']
            minus_dept = data['-Dept_135']
            dept_90 = data['Dept_90']
            dept135 = plus_dept + minus_dept

            # Process Type 0 (All_C excluding matches with +Dept_135)
            exclude_all_c = set()
            for val in dept135:
                lower, upper = val - 0.1, val + 0.1
                for idx, c_val in enumerate(all_c):
                    if lower <= c_val <= upper and idx not in exclude_all_c:
                       exclude_all_c.add(idx)
                       break

            # Process Type 3 (+Dept_135 excluding matches with Dept_90)
            exclude_plus_dept = set()
            for val in dept_90:
                lower, upper = val - 0.1, val + 0.1
                for idx, p_val in enumerate(plus_dept):
                    if lower <= p_val <= upper and idx not in exclude_plus_dept:
                        exclude_plus_dept.add(idx)
                        break

            return {
                0: [v for i, v in enumerate(all_c) if i not in exclude_all_c],
                1: dept_90,
                2: minus_dept,
                3: [v for i, v in enumerate(plus_dept) if i not in exclude_plus_dept]
            }
        else:
            text = self.global_entry.get("1.0", tk.END).strip().replace('，', ',')  # 处理中文逗号
            values = [x.strip() for x in text.split(',') if x.strip()]  # 分割并过滤空值
            return [float(x) for x in values]

    def process_carbon_scoring_config(self):
        scoring_config = {}

        scoring_mode = str(self.score_mode.get())
        scoring_config['carbon_score_mode'] = scoring_mode

        if scoring_mode == 'global':
            scoring_config['carbon_global_thresholds'] = (
                float(self.green_thresh.get()),
                float(self.yellow_thresh.get())
            )
        elif scoring_mode == 'fine':
            fine_ranges = []
            for row in self.range_rows:
                try:
                    c_start = float(row['start'].get())
                    c_end = float(row['end'].get())
                    green_val = float(row['green'].get())
                    yellow_val = float(row['yellow'].get())
                    fine_ranges.append((
                        (c_start, c_end),
                        green_val,
                        yellow_val
                    ))
                except ValueError:
                    continue
            scoring_config['carbon_fine_ranges'] = fine_ranges

        return scoring_config

    def run_analysis(self):
        env_params = (
            int(self.env_level.get()),
            float(self.self_weight.get()),
            float(self.env_weight.get())
        )

        config = {
            'sdf_files': self.file_selector.selected_files,
            'c_mode': str(self.c_mode.get()),
             'C_merge':str(self.c_merge.get()),
            'c_data': self.get_c_data(),
            'c_range': tuple(map(int, (self.c_min.get(), self.c_max.get()))),
            'fw_range': tuple(map(float, (self.m_min.get(), self.m_max.get()))),
            'env_params': env_params,
            'ch_mode': int(self.hmqc_mode.get()),
            'CH_merge':str(self.ch_merge.get()),
            'ch_data': self.get_ch_data(),
            'ch_tolerances': tuple(map(float, (self.c_tol.get(), self.h_tol.get()))),
            'final_weights': tuple(map(float, (self.c_weight.get(), self.ch_weight.get())))
        }

        scoring_config = self.process_carbon_scoring_config()
        config.update(scoring_config)

        # try:
        pipeline = CombinedScorerGUI(**config)
        results = pipeline.execute()
        pipeline._generate_plots(output_dir="combined_results")

        viewer = CombinedResultViewer(
            db_path="chem_data.db",
            results=results,
            output_dir="combined_results"
        )
        viewer.create_result_window(self.master)
        #
        # except Exception as e:
        #     messagebox.showerror("Analysis error", f"An error occurred while performing the analysis:\n{str(e)}")
    def get_ch_data(self):

        ch_data = {}
        current_mode = int(self.hmqc_mode.get())
        types_config = {
            1: ['All_type'],
            2: ['type13', 'type2'],
            3: ['type1', 'type2', 'type3']
        }[current_mode]

        for t in types_config:
            text = self.point_groups[t].get("1.0", tk.END).strip()

            text = (text.replace('（', '(')
                    .replace('）', ')')
                    .replace('，', ','))

            matches = re.findall(r'\(([\d.]+),\s*([\d.]+)\)', text)
            ch_data[t] = [[float(m[0]), float(m[1])] for m in matches]

        return ch_data



from pathlib import Path
import sqlite3

class CarbonOnlyModeFrame(BaseModeFrame):

    def __init__(self, master):
        super().__init__(master, "C matching mode")

    def create_parameters(self):
        carbon_frame = ttk.LabelFrame(self, text="C parameters")
        mode_frame = ttk.Frame(carbon_frame)
        self.c_mode = tk.StringVar(value='typed')
        self.c_merge = tk.StringVar(value='N')
        ttk.Radiobutton(mode_frame, text="C_typed", variable=self.c_mode,
                        value='typed', command=self.toggle_c_params).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="C_untype", variable=self.c_mode,
                        value='untyped', command=self.toggle_c_params).pack(side=tk.LEFT)
        ttk.Label(mode_frame, text=" C_merge:").pack(side=tk.LEFT, padx=(10,0))
        ttk.Radiobutton(mode_frame, text="Y", variable=self.c_merge,
                        value='Y').pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="N", variable=self.c_merge,
                        value='N').pack(side=tk.LEFT)
        mode_frame.pack(anchor=tk.W)
        self.param_container = ttk.Frame(carbon_frame)
        self.typed_params = {}
        typed_frame = ttk.Frame(self.param_container)
        labels = ['   All_C    ', '+Dept_135','-Dept_135', '  Dept_90 ']
        for label in labels:
            row = ttk.Frame(typed_frame)
            ttk.Label(row, text=f"{label}:").pack(side=tk.LEFT)
            entry = tk.Text(row, height=2, width=60)
            entry.pack(side=tk.LEFT, padx=5)
            self.typed_params[label] = entry
            row.pack(anchor=tk.W)
        self.typed_frame = typed_frame
        untyped_frame = ttk.Frame(self.param_container)
        self.global_entry = tk.Text(untyped_frame, height=3, width=60)
        ttk.Label(untyped_frame, text="All_C:").pack(side=tk.LEFT)
        self.global_entry.pack(side=tk.LEFT, padx=5)
        self.untyped_frame = untyped_frame

        self.param_container.pack()
        self.toggle_c_params()
        range_frame = ttk.Frame(carbon_frame)
        ttk.Label(range_frame, text="C_num range:").pack(side=tk.LEFT)
        self.c_min = ttk.Entry(range_frame, width=5)
        self.c_min.insert(0, "5")
        self.c_min.pack(side=tk.LEFT)
        ttk.Label(range_frame, text="-").pack(side=tk.LEFT)
        self.c_max = ttk.Entry(range_frame, width=5)
        self.c_max.insert(0, "30")
        self.c_max.pack(side=tk.LEFT)

        ttk.Label(range_frame, text=" | ").pack(side=tk.LEFT, padx=5)
        ttk.Label(range_frame, text=":MW range").pack(side=tk.LEFT)
        self.m_min = ttk.Entry(range_frame, width=5)
        self.m_min.insert(0, "100")
        self.m_min.pack(side=tk.LEFT)
        ttk.Label(range_frame, text="-").pack(side=tk.LEFT)
        self.m_max = ttk.Entry(range_frame, width=5)
        self.m_max.insert(0, "500")
        self.m_max.pack(side=tk.LEFT)

        range_frame.pack(pady=5)

        carbon_frame.pack(fill=tk.X, padx=10, pady=5)


        score_frame = ttk.LabelFrame(self, text="Scoring parameters")


        env_frame = ttk.Frame(score_frame)
        ttk.Label(env_frame, text="Env atom level:").pack(side=tk.LEFT)
        self.env_level = ttk.Combobox(env_frame, values=[1, 2, 3], width=3)
        self.env_level.set(1)
        self.env_level.pack(side=tk.LEFT)

        # 权重设置
        ttk.Label(env_frame, text="Self weight:").pack(side=tk.LEFT, padx=(10,0))
        self.self_weight = ttk.Entry(env_frame, width=5)
        self.self_weight.insert(0, "0.7")
        self.self_weight.pack(side=tk.LEFT)

        ttk.Label(env_frame, text="Env weight:").pack(side=tk.LEFT, padx=(10,0))
        self.env_weight = ttk.Entry(env_frame, width=5, state='readonly')
        self.env_weight.config(state='normal')
        self.env_weight.insert(0, "0.3")
        self.env_weight.config(state='readonly')
        self.env_weight.pack(side=tk.LEFT)

        self.self_weight.bind("<KeyRelease>", self.update_env_weight)
        env_frame.pack(anchor=tk.W)
        self.score_mode = tk.StringVar(value='global')
        ttk.Radiobutton(score_frame, text="Global mode", variable=self.score_mode,
                        value='global', command=self.toggle_score_mode).pack(anchor=tk.W)
        ttk.Radiobutton(score_frame, text="Fine model", variable=self.score_mode,
                        value='fine', command=self.toggle_score_mode).pack(anchor=tk.W)


        self.global_frame = ttk.Frame(score_frame)
        ttk.Label(self.global_frame, text="Global threshold:").pack(side=tk.LEFT)
        self.green_thresh = ttk.Entry(self.global_frame, width=8)
        self.green_thresh.insert(0, "0.5")
        self.green_thresh.pack(side=tk.LEFT)
        ttk.Label(self.global_frame, text="-").pack(side=tk.LEFT)
        self.yellow_thresh = ttk.Entry(self.global_frame, width=8)
        self.yellow_thresh.insert(0, "2.0")
        self.yellow_thresh.pack(side=tk.LEFT)
        self.global_frame.pack(anchor=tk.W, fill=tk.X)

        self.fine_container = ttk.Frame(score_frame)

        self.canvas = tk.Canvas(self.fine_container, height=60)
        scrollbar = ttk.Scrollbar(self.fine_container, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = ttk.Frame(self.canvas)

        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.range_rows = []
        self.add_range_row((0, 220), 1, 3)  # 初始行

        btn_frame = ttk.Frame(self.fine_container)
        add_btn = ttk.Button(btn_frame, text="+ Add range",
                             command=lambda: self.add_range_row((0, 0), 0, 0))
        add_btn.pack(pady=5)
        btn_frame.pack(fill=tk.X)

        self.fine_container.pack_forget()
        score_frame.pack(fill=tk.X, padx=10, pady=5)

    def update_env_weight(self, event=None):
        try:
            self_val = float(self.self_weight.get() or 0)  # Handle empty input
            env_val = max(0.0, min(1.0, 1 - self_val))  # Clamp between 0-1
            self.env_weight.config(state='normal')
            self.env_weight.delete(0, tk.END)
            self.env_weight.insert(0, f"{env_val:.2f}")
            self.env_weight.config(state='readonly')
        except ValueError:
            self.env_weight.config(state='normal')
            self.env_weight.delete(0, tk.END)
            self.env_weight.insert(0, "1.00")  # Default value
            self.env_weight.config(state='readonly')


    def toggle_score_mode(self):
        if self.score_mode.get() == 'global':
            self.global_frame.pack(anchor=tk.W, fill=tk.X)
            self.fine_container.pack_forget()
        else:
            self.global_frame.pack_forget()
            self.fine_container.pack(fill=tk.X)

    def add_range_row(self, carbon_range, green_val, yellow_val):
        if len(self.range_rows) >= 10:
            return

        frame = ttk.Frame(self.scroll_frame)

        ttk.Label(frame, text="C_range:").pack(side=tk.LEFT)
        start_entry = ttk.Entry(frame, width=6)
        start_entry.insert(0, str(carbon_range[0]))
        start_entry.pack(side=tk.LEFT)
        ttk.Label(frame, text="-").pack(side=tk.LEFT)
        end_entry = ttk.Entry(frame, width=6)
        end_entry.insert(0, str(carbon_range[1]))
        end_entry.pack(side=tk.LEFT)

        # 阈值
        ttk.Label(frame, text="Green threshold:").pack(side=tk.LEFT, padx=(10,0))
        green_entry = ttk.Entry(frame, width=6)
        green_entry.insert(0, str(green_val))
        green_entry.pack(side=tk.LEFT)

        ttk.Label(frame, text="Yellow threshold:").pack(side=tk.LEFT, padx=(10,0))
        yellow_entry = ttk.Entry(frame, width=6)
        yellow_entry.insert(0, str(yellow_val))
        yellow_entry.pack(side=tk.LEFT)

        del_btn = ttk.Button(frame, text="×", width=2,
                             command=lambda: self.remove_range_row(frame))
        del_btn.pack(side=tk.RIGHT, padx=5)

        self.range_rows.append({
            'frame': frame,
            'start': start_entry,
            'end': end_entry,
            'green': green_entry,
            'yellow': yellow_entry
        })
        frame.pack(fill=tk.X, pady=2)

    def remove_range_row(self, frame):
        for row in self.range_rows:
            if row['frame'] == frame:
                frame.destroy()
                self.range_rows.remove(row)
                break
    def toggle_c_params(self):
        for widget in self.param_container.winfo_children():
            widget.pack_forget()

        if self.c_mode.get() == 'typed':
            self.typed_frame.pack(anchor=tk.W)
        else:
            self.untyped_frame.pack(anchor=tk.W)


    def validate_input(self):
        try:
            if not self.file_selector.selected_files:
                raise ValueError("Please select the SDF file first")

            if self.c_mode.get() == 'typed':
                for label in ['   All_C    ', '+Dept_135','-Dept_135', '  Dept_90 ']:
                    text = self.typed_params[label].get("1.0", tk.END).strip()
                    if not text:
                        raise ValueError(f"{label.strip()} cannot be empty")
            else:
                text = self.global_entry.get("1.0", tk.END).strip().replace('，', ',')
                if not text:
                    raise ValueError("Global mode requires a list of displacements")
            if not (self.c_min.get().isdigit() and self.c_max.get().isdigit()):
                raise ValueError("The carbon number range must be an integer")
            return True
        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return False

    def process_carbon_scoring_config(self):
        scoring_config = {}

        scoring_mode = str(self.score_mode.get())
        scoring_config['score_mode'] = scoring_mode

        if scoring_mode == 'global':
            scoring_config['global_thresholds'] = (
                float(self.green_thresh.get()),
                float(self.yellow_thresh.get())
            )
        elif scoring_mode == 'fine':
            fine_ranges = []
            for row in self.range_rows:
                try:
                    c_start = float(row['start'].get())
                    c_end = float(row['end'].get())
                    green_val = float(row['green'].get())
                    yellow_val = float(row['yellow'].get())
                    fine_ranges.append((
                        (c_start, c_end),
                        green_val,
                        yellow_val
                    ))
                except ValueError:
                    continue
            scoring_config['fine_ranges'] = fine_ranges

        return scoring_config

    def run_analysis(self):

        config = {
            'sdf_files': self.file_selector.selected_files,
            'c_mode': str(self.c_mode.get()),
            'C_merge': str(self.c_merge.get()),
            'c_data': self.get_c_data(),
            'c_range': tuple(map(int, (self.c_min.get(), self.c_max.get()))),
            'fw_range': tuple(map(float, (self.m_min.get(), self.m_max.get()))),
            'env_level':int(self.env_level.get()),
            'self_weight':float(self.self_weight.get()),
            'env_weight': float(self.env_weight.get())
        }

        scoring_config = self.process_carbon_scoring_config()
        config.update(scoring_config)

        try:
            pipeline = CarbonOnlyScorerGUI(**config)
            try:
                results = pipeline.execute()
            finally:
                _close_carbon_thread_connection()
            visualizer = CarbonResultVisualizer(
                db_path="chem_data.db",
                top_results=results
            )
            visualizer.generate_plots(output_dir="molecule_plots")

            # 显示结果窗口
            ResultViewer("chem_data.db", results).create_result_window(self.master)

        except Exception as e:
            messagebox.showerror("Analysis error", str(e))

    def get_c_data(self):
        if self.c_mode.get() == 'typed':
            data = {
                label.strip(): [
                    float(x) for x in
                    (entry.get("1.0", tk.END).strip() if isinstance(entry, tk.Text) else entry.get().strip()
                     ).replace('，', ',')  # Replace Chinese commas with English commas
                    .split(',')]
                for label, entry in self.typed_params.items()
            }

            all_c = data['All_C']
            plus_dept = data['+Dept_135']
            minus_dept = data['-Dept_135']
            dept_90 = data['Dept_90']
            dept135 = plus_dept + minus_dept

            # Process Type 0 (All_C excluding matches with +Dept_135)
            exclude_all_c = set()
            for val in dept135:
                lower, upper = val - 0.1, val + 0.1
                for idx, c_val in enumerate(all_c):
                    if lower <= c_val <= upper and idx not in exclude_all_c:
                        exclude_all_c.add(idx)
                        break

            # Process Type 3 (+Dept_135 excluding matches with Dept_90)
            exclude_plus_dept = set()
            for val in dept_90:
                lower, upper = val - 0.1, val + 0.1
                for idx, p_val in enumerate(plus_dept):
                    if lower <= p_val <= upper and idx not in exclude_plus_dept:
                        exclude_plus_dept.add(idx)
                        break

            return {
                0: [v for i, v in enumerate(all_c) if i not in exclude_all_c],
                1: dept_90,
                2: minus_dept,
                3: [v for i, v in enumerate(plus_dept) if i not in exclude_plus_dept]
            }
        else:
            text = self.global_entry.get("1.0", tk.END).strip().replace('，', ',')  # 处理中文逗号
            values = [x.strip() for x in text.split(',') if x.strip()]  # 分割并过滤空值
            return [float(x) for x in values]
class CHMatchModeFrame(BaseModeFrame):

    def __init__(self, master):
        super().__init__(master, "CH matching mode")

    def create_parameters(self):
        hmqc_frame = ttk.LabelFrame(self, text="HSQC parameters")

        mode_frame = ttk.Frame(hmqc_frame)
        ttk.Label(mode_frame, text="Matching pattern:").pack(side=tk.LEFT)
        self.hmqc_mode = ttk.Combobox(mode_frame, values=[1,2,3], state='readonly')
        self.hmqc_mode.current(1)
        self.hmqc_mode.pack(side=tk.LEFT)
        self.hmqc_mode.bind("<<ComboboxSelected>>", lambda e: self.create_input_groups())
        self.ch_merge = tk.StringVar(value='N')  # 默认值设为N
        ttk.Label(mode_frame, text=" | CH_merge:").pack(side=tk.LEFT, padx=(10,0))
        ttk.Radiobutton(mode_frame, text="Y", variable=self.ch_merge,
                        value='Y').pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="N", variable=self.ch_merge,
                        value='N').pack(side=tk.LEFT)
        mode_frame.pack(anchor=tk.W)

        self.input_container = ttk.Frame(hmqc_frame)
        self.input_container.pack(fill=tk.X, padx=5, pady=2)

        self.point_groups = {}
        self.create_input_groups()

        tol_frame = ttk.Frame(hmqc_frame)
        ttk.Label(tol_frame, text="C tolerance:").pack(side=tk.LEFT)
        self.c_tol = ttk.Entry(tol_frame, width=8)
        self.c_tol.insert(0, "1.0")
        self.c_tol.pack(side=tk.LEFT)
        ttk.Label(tol_frame, text="H tolerance:").pack(side=tk.LEFT)
        self.h_tol = ttk.Entry(tol_frame, width=8)
        self.h_tol.insert(0, "0.2")
        self.h_tol.pack(side=tk.LEFT)
        tol_frame.pack(anchor=tk.W)

        hmqc_frame.pack(fill=tk.X, padx=10, pady=5)

    def create_input_groups(self):

        for widget in self.input_container.winfo_children():
            widget.destroy()
        self.point_groups.clear()

        current_mode = int(self.hmqc_mode.get())
        types_config = {
            1: ['All_type'],
            2: ['type13', 'type2'],
            3: ['type1', 'type2', 'type3']
        }

        for t in types_config[current_mode]:
            group = ttk.LabelFrame(self.input_container, text=t)
            entry = tk.Text(group, height=3, width=40)
            scroll = ttk.Scrollbar(group, command=entry.yview)
            entry.config(yscrollcommand=scroll.set)
            entry.pack(side=tk.LEFT)
            scroll.pack(side=tk.LEFT, fill=tk.Y)
            self.point_groups[t] = entry
            group.pack(fill=tk.X, padx=5, pady=2, side=tk.TOP)

    def validate_input(self):
        try:
            if not self.file_selector.selected_files:
                raise ValueError("Please select the SDF file first")

            current_mode = int(self.hmqc_mode.get())
            required_fields = {
                1: ['All_type'],
                2: ['type13', 'type2'],
                3: ['type1', 'type2', 'type3']
            }[current_mode]

            pattern = r'^\(\s*(?:\d+\.?\d*|\.\d+)\s*,\s*(?:\d+\.?\d*|\.\d+)\s*\)(?:[,\s]*\(\s*(?:\d+\.?\d*|\.\d+)\s*,\s*(?:\d+\.?\d*|\.\d+)\s*\))*$'

            for field in required_fields:
                text = self.point_groups[field].get("1.0", tk.END).strip()

                text = text.replace('，', ',').replace('（', '(').replace('）', ')')
                if not text:
                    raise ValueError(f"{field} cannot be empty")
                if not re.match(pattern, text):
                    raise ValueError(f"{field} format error, should be (numbers, numbers) list, support decimals, do not support negative numbers")
            return True
        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return False

    def run_analysis(self):
            config = {
            'sdf_files': self.file_selector.selected_files,
            'ch_mode': int(self.hmqc_mode.get()),
                'CH_merge': str(self.ch_merge.get()),
                'ch_data': self.get_ch_data(),
            'tolerances': tuple(map(float, (self.c_tol.get(), self.h_tol.get()))),
        }

            try:

                pipeline = CHOnlyScorerGUI(**config)
                results = pipeline.execute()
                visualizer = CHMatchVisualizer(
                db_path="chem_data.db",
                top_results=results,
                mode=config['ch_mode']
                )
                results = pipeline.execute()
                viewer = CHResultViewer(db_path="chem_data.db", results=results)
                viewer.create_result_window(self.master)
                visualizer.generate_plots()
            except Exception as e:
                messagebox.showerror("Analysis error", str(e))

    def get_ch_data(self):
        ch_data = {}
        current_mode = int(self.hmqc_mode.get())
        types_config = {
            1: ['All_type'],
            2: ['type13', 'type2'],
            3: ['type1', 'type2', 'type3']
        }[current_mode]

        for t in types_config:
            text = self.point_groups[t].get("1.0", tk.END).strip()
            text = (text.replace('（', '(')
                    .replace('）', ')')
                    .replace('，', ','))
            matches = re.findall(r'\(([\d.]+),\s*([\d.]+)\)', text)
            ch_data[t] = [[float(m[0]), float(m[1])] for m in matches]

        return ch_data


class ClusterResultWindow:
    """Editable popup window for CORD-NMR clustering results."""

    def __init__(self, app, owner_frame, clusters, kind, output_dir=None):
        self.app = app
        self.owner_frame = owner_frame
        self.all_clusters = self._sort_clusters(list(clusters))
        self.clusters = list(self.all_clusters)
        self.kind = kind
        self.output_dir = output_dir
        self.filter_small_clusters = None
        self.min_cluster_tracks = None
        self.filter_status_label = None
        self.result_body = None

    def create_result_window(self):
        win = tk.Toplevel(self.app)
        win.title(f"{APP_DISPLAY_NAME} {self.kind} clustering results")
        win.geometry("900x650")
        win.configure(bg=ChemTheme.COLORS['background'])

        header = ttk.Frame(win)
        ttk.Label(
            header,
            text=f"{self.kind} clustering results (editable)",
            font=('Helvetica', 15, 'bold')
        ).pack(side=tk.LEFT, padx=10, pady=8)
        ttk.Button(header, text="Export edited clusters", command=lambda: self.export_clusters(win)).pack(side=tk.RIGHT, padx=10)
        header.pack(fill=tk.X)
        if self.output_dir:
            ttk.Label(win, text=f"Full run outputs: {self.output_dir}").pack(fill=tk.X, padx=12, pady=(0, 6))

        tip = (
            "C clusters use the same format as CORD-NMR C_untyped input. "
            "HSQC clusters use the same point-list format as CORD-NMR HSQC All_type input."
        )
        ttk.Label(win, text=tip).pack(fill=tk.X, padx=12, pady=(0, 8))

        filter_bar = ttk.Frame(win)
        self.filter_small_clusters = tk.BooleanVar(value=True)
        self.min_cluster_tracks = tk.StringVar(value="5")
        ttk.Checkbutton(
            filter_bar,
            text="Hide clusters with n <",
            variable=self.filter_small_clusters,
            command=self.refresh_cluster_blocks,
        ).pack(side=tk.LEFT, padx=(12, 4), pady=(0, 6))
        ttk.Entry(filter_bar, textvariable=self.min_cluster_tracks, width=5).pack(side=tk.LEFT, padx=(0, 4), pady=(0, 6))
        ttk.Button(filter_bar, text="Apply", command=self.refresh_cluster_blocks).pack(side=tk.LEFT, padx=(0, 10), pady=(0, 6))
        ttk.Label(filter_bar, text="Sorted by common-mask desc, then n desc").pack(side=tk.LEFT, padx=(4, 8), pady=(0, 6))
        self.filter_status_label = ttk.Label(filter_bar, text="")
        self.filter_status_label.pack(side=tk.RIGHT, padx=(8, 12), pady=(0, 6))
        filter_bar.pack(fill=tk.X)

        self.text_widgets = []
        self.selection_widgets = []
        if self.kind == "C":
            merge_bar = ttk.LabelFrame(win, text="Merged C-cluster actions")
            ttk.Button(merge_bar, text="Select all", command=lambda: self.set_all_selected(True)).pack(side=tk.LEFT, padx=5, pady=6)
            ttk.Button(merge_bar, text="Clear", command=lambda: self.set_all_selected(False)).pack(side=tk.LEFT, padx=5, pady=6)
            ttk.Label(merge_bar, text=" C_merge:").pack(side=tk.LEFT, padx=(12, 2), pady=6)
            ttk.Radiobutton(merge_bar, text="Y", variable=self.owner_frame.c_merge, value="Y").pack(side=tk.LEFT, padx=2, pady=6)
            ttk.Radiobutton(merge_bar, text="N", variable=self.owner_frame.c_merge, value="N").pack(side=tk.LEFT, padx=(0, 8), pady=6)
            ttk.Button(merge_bar, text="Fill selected C_untyped page", command=self.fill_selected_c_clusters).pack(side=tk.LEFT, padx=12, pady=6)
            ttk.Button(merge_bar, text="Analyze selected C_untyped now", command=self.analyze_selected_c_clusters).pack(side=tk.LEFT, padx=5, pady=6)
            ttk.Button(merge_bar, text="Copy selected", command=self.copy_selected_c_clusters).pack(side=tk.LEFT, padx=5, pady=6)
            merge_bar.pack(fill=tk.X, padx=12, pady=(0, 8))

        container = ttk.Frame(win)
        canvas = tk.Canvas(container, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas)
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        body_id = canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(body_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.result_body = body

        self.refresh_cluster_blocks()

    @staticmethod
    def _mask_rank(mask):
        text = str(mask or "")
        if text and set(text) <= {"0", "1"}:
            return int(text, 2)
        return -1

    @classmethod
    def _sort_clusters(cls, clusters):
        return sorted(
            clusters,
            key=lambda cluster: (
                -cls._mask_rank(getattr(cluster, "presence_mask", "")),
                -int(getattr(cluster, "n_tracks", 0) or 0),
                str(getattr(cluster, "cluster_id", "")),
            ),
        )

    def _visible_clusters(self):
        clusters = self._sort_clusters(self.all_clusters)
        if not self.filter_small_clusters or not self.filter_small_clusters.get():
            return clusters
        try:
            min_tracks = max(0, int(float(self.min_cluster_tracks.get())))
        except Exception:
            min_tracks = 5
            self.min_cluster_tracks.set("5")
        return [cluster for cluster in clusters if int(getattr(cluster, "n_tracks", 0) or 0) >= min_tracks]

    def refresh_cluster_blocks(self):
        if self.result_body is None:
            return
        for child in self.result_body.winfo_children():
            child.destroy()
        self.text_widgets = []
        self.selection_widgets = []
        self.clusters = self._visible_clusters()
        if self.filter_status_label is not None:
            self.filter_status_label.config(text=f"Showing {len(self.clusters)}/{len(self.all_clusters)}")
        for idx, cluster in enumerate(self.clusters, 1):
            self._add_cluster_block(self.result_body, idx, cluster)

    def _format_cluster(self, cluster):
        if self.kind == "C":
            return format_c_values(cluster.values)
        return format_hsqc_points(cluster.values)

    def _add_cluster_block(self, parent, idx, cluster):
        block = ttk.LabelFrame(
            parent,
            text=f"{cluster.cluster_id} | n={cluster.n_tracks} | common-mask={cluster.presence_mask or '-'}"
        )
        block.pack(fill=tk.X, expand=True, padx=8, pady=6)

        selected_var = None
        if self.kind == "C":
            selected_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                block,
                text="Include this cluster in merged C_untyped analysis",
                variable=selected_var,
            ).pack(anchor=tk.W, padx=6, pady=(5, 0))

        ttk.Label(block, text=cluster.details).pack(anchor=tk.W, padx=6, pady=(4, 0))
        formatted = self._format_cluster(cluster)
        item_count = formatted.count(",") + 1 if formatted.strip() else 1
        height = max(2, min(7, item_count // 8 + 2))

        txt = tk.Text(block, height=height, width=96, wrap=tk.WORD, font=ChemTheme.FONTS['input'])
        txt.insert("1.0", formatted)
        txt.pack(fill=tk.X, padx=6, pady=5)
        self.text_widgets.append((cluster.cluster_id, txt))
        if selected_var is not None:
            self.selection_widgets.append((cluster.cluster_id, selected_var, txt))

        btns = ttk.Frame(block)
        if self.kind == "C":
            ttk.Button(
                btns,
                text="Fill C_untyped page",
                command=lambda t=txt: self.app.fill_c_untyped_from_cluster(t.get("1.0", tk.END).strip())
            ).pack(side=tk.LEFT, padx=4)
            ttk.Button(
                btns,
                text="Analyze C_untyped now",
                command=lambda t=txt: self.owner_frame.analyze_c_text(t.get("1.0", tk.END).strip())
            ).pack(side=tk.LEFT, padx=4)
        else:
            ttk.Button(
                btns,
                text="Fill HSQC All_type page",
                command=lambda t=txt: self.app.fill_hsqc_alltype_from_cluster(t.get("1.0", tk.END).strip())
            ).pack(side=tk.LEFT, padx=4)
            ttk.Button(
                btns,
                text="Analyze HSQC now",
                command=lambda t=txt: self.owner_frame.analyze_hsqc_text(t.get("1.0", tk.END).strip())
            ).pack(side=tk.LEFT, padx=4)

        ttk.Button(btns, text="Copy", command=lambda t=txt: self.copy_text(t)).pack(side=tk.LEFT, padx=4)
        btns.pack(anchor=tk.W, padx=6, pady=(0, 6))

    def copy_text(self, text_widget):
        data = text_widget.get("1.0", tk.END).strip()
        self.app.clipboard_clear()
        self.app.clipboard_append(data)
        messagebox.showinfo("Copied", "Cluster data copied to clipboard.")

    def set_all_selected(self, selected):
        for _, var, _ in self.selection_widgets:
            var.set(bool(selected))

    def _selected_c_cluster_text(self):
        values = []
        selected_ids = []
        for cid, var, txt in self.selection_widgets:
            if not var.get():
                continue
            selected_ids.append(cid)
            values.extend(parse_c_values(txt.get("1.0", tk.END).strip()))
        if not selected_ids:
            raise ValueError("Please select at least one C cluster to merge.")
        if not values:
            raise ValueError("The selected C clusters do not contain any usable ppm values.")
        return format_c_values(values), selected_ids

    def fill_selected_c_clusters(self):
        try:
            text, _ = self._selected_c_cluster_text()
            self.app.fill_c_untyped_from_cluster(text)
        except Exception as e:
            messagebox.showerror("Merged C clusters", str(e))

    def analyze_selected_c_clusters(self):
        try:
            text, selected_ids = self._selected_c_cluster_text()
            c_merge = self.owner_frame._direct_c_merge_mode()
            proceed = messagebox.askyesno(
                "Analyze merged C clusters",
                f"Merge {len(selected_ids)} selected C clusters and run C_untyped analysis now?\nC_merge={c_merge}"
            )
            if proceed:
                self.owner_frame.analyze_c_text(text)
        except Exception as e:
            messagebox.showerror("Merged C clusters", str(e))

    def copy_selected_c_clusters(self):
        try:
            text, selected_ids = self._selected_c_cluster_text()
            self.app.clipboard_clear()
            self.app.clipboard_append(text)
            messagebox.showinfo("Copied", f"Copied merged data from {len(selected_ids)} selected C clusters.")
        except Exception as e:
            messagebox.showerror("Merged C clusters", str(e))

    def export_clusters(self, win):
        path = filedialog.asksaveasfilename(
            parent=win,
            title="Export edited cluster results",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Text", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write("cluster_id,kind,data\n")
            for cid, txt in self.text_widgets:
                data = txt.get("1.0", tk.END).strip().replace('"', '""')
                f.write(f'"{cid}","{self.kind}","{data}"\n')
        messagebox.showinfo("Exported", f"Cluster results exported:\n{path}")


class ClusterModeFrame(ttk.Frame):
    """CORD-NMR cross-spectrum clustering workflow."""

    def __init__(self, master):
        super().__init__(master)
        self.sample_files = []
        self.sdf_files = []
        self._last_sample_dir = ""
        self._last_sdf_dir = ""
        self.c_merge = tk.StringVar(value="N")
        self.create_widgets_modern()

    def _labeled_entry(self, parent, label, default, row, col, width=8, padx=4):
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky=tk.E, padx=(padx, 2), pady=3)
        ent = ttk.Entry(parent, width=width)
        ent.insert(0, str(default))
        ent.grid(row=row, column=col + 1, sticky=tk.W, padx=(0, padx), pady=3)
        return ent

    def _set_entry(self, entry, value):
        entry.config(state="normal")
        entry.delete(0, tk.END)
        entry.insert(0, str(value))

    def _region_default_text(self):
        return "\n".join(f"{lo:g}-{hi:g}: {tol:g}" for lo, hi, tol in DEFAULT_REGION_WINDOWS)

    def _section_title(self, parent, text):
        ttk.Label(parent, text=text, style='Section.TLabel').pack(anchor=tk.W, padx=14, pady=(12, 6))

    def _param_entry(self, parent, label, default, row, col, width=9):
        ttk.Label(parent, text=label, style='Card.TLabel').grid(row=row, column=col, sticky=tk.W, padx=(12, 6), pady=5)
        ent = ttk.Entry(parent, width=width)
        ent.insert(0, str(default))
        ent.grid(row=row, column=col + 1, sticky=tk.W, padx=(0, 14), pady=5)
        return ent

    def _hidden_entry(self, default):
        ent = ttk.Entry(self)
        ent.insert(0, str(default))
        return ent

    def _file_row(self, parent, title, button_text, command, label_attr):
        row = ttk.Frame(parent, style='Card.TFrame')
        row.pack(fill=tk.X, padx=14, pady=6)
        left = ttk.Frame(row, style='Card.TFrame')
        left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(left, text=title, style='Card.TLabel', font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)
        label = ttk.Label(left, text="No files selected", style='Muted.TLabel', wraplength=660)
        label.pack(anchor=tk.W, pady=(2, 0))
        setattr(self, label_attr, label)
        ttk.Button(row, text=button_text, command=command).pack(side=tk.RIGHT, padx=(14, 0))

    def _make_card(self, parent, title):
        card = ttk.LabelFrame(parent, text=title)
        card.pack(fill=tk.X, padx=4, pady=7)
        return card

    def _dialog_parent(self):
        parent = self.winfo_toplevel()
        try:
            parent.update_idletasks()
            parent.lift()
        except tk.TclError:
            pass
        return parent

    def _initial_dir(self, last_dir_attr, candidates):
        last_dir = getattr(self, last_dir_attr, "")
        if last_dir and Path(last_dir).exists():
            return str(Path(last_dir))
        for candidate in candidates:
            path = Path(candidate)
            if path.exists():
                return str(path)
        return str(Path.home())

    def _sample_initial_dir(self):
        project_root = Path(__file__).resolve().parent
        return self._initial_dir(
            "_last_sample_dir",
            [
                project_root / "data" / "bench45_mask_balanced_data" / "blind_peaklists_and_manifests",
                Path.home() / "Downloads",
                project_root,
            ],
        )

    def _sdf_initial_dir(self):
        project_root = Path(__file__).resolve().parent
        return self._initial_dir(
            "_last_sdf_dir",
            [
                project_root,
                Path.home() / "Downloads",
                Path.home(),
            ],
        )

    def _choose_files_in_app(self, title, initialdir, extensions):
        """Small Tk file picker used to avoid Windows native dialog hangs."""
        result = []
        current_dir = Path(initialdir)
        if not current_dir.exists():
            current_dir = Path.home()
        extensions = {ext.lower() for ext in extensions}

        win = tk.Toplevel(self._dialog_parent())
        win.title(title)
        win.geometry("820x520")
        win.minsize(680, 420)
        win.transient(self.winfo_toplevel())

        current_var = tk.StringVar(value=str(current_dir))
        dir_paths = []
        file_paths = []

        top = ttk.Frame(win)
        top.pack(fill=tk.X, padx=10, pady=(10, 4))
        ttk.Label(top, text="Folder").pack(side=tk.LEFT)
        path_entry = ttk.Entry(top, textvariable=current_var)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        lists = ttk.Frame(win)
        lists.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        dir_frame = ttk.LabelFrame(lists, text="Folders")
        dir_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        file_frame = ttk.LabelFrame(lists, text="Files")
        file_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        dir_list = tk.Listbox(dir_frame, activestyle="dotbox")
        dir_scroll = ttk.Scrollbar(dir_frame, orient=tk.VERTICAL, command=dir_list.yview)
        dir_list.configure(yscrollcommand=dir_scroll.set)
        dir_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        dir_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        file_list = tk.Listbox(file_frame, selectmode=tk.EXTENDED, activestyle="dotbox")
        file_scroll = ttk.Scrollbar(file_frame, orient=tk.VERTICAL, command=file_list.yview)
        file_list.configure(yscrollcommand=file_scroll.set)
        file_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        file_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        status_var = tk.StringVar(value="")
        ttk.Label(win, textvariable=status_var).pack(fill=tk.X, padx=10, pady=(0, 4))

        def natural_key(path):
            stem = path.stem if path.is_file() else path.name
            chunks = re.split(r"(\d+)", stem.lower())
            return [int(c) if c.isdigit() else c for c in chunks]

        def refresh():
            nonlocal current_dir, dir_paths, file_paths
            current_var.set(str(current_dir))
            dir_list.delete(0, tk.END)
            file_list.delete(0, tk.END)
            dir_paths = []
            file_paths = []
            try:
                entries = list(current_dir.iterdir())
            except Exception as exc:
                status_var.set(f"Cannot open folder: {exc}")
                return

            if current_dir.parent != current_dir:
                dir_paths.append(current_dir.parent)
                dir_list.insert(tk.END, "..")

            dirs = sorted([p for p in entries if p.is_dir()], key=lambda p: p.name.lower())
            files = sorted(
                [p for p in entries if p.is_file() and (not extensions or p.suffix.lower() in extensions)],
                key=natural_key,
            )
            for path in dirs:
                dir_paths.append(path)
                dir_list.insert(tk.END, path.name)
            for path in files:
                file_paths.append(path)
                file_list.insert(tk.END, path.name)
            status_var.set(f"{len(dirs)} folders, {len(files)} matching files")

        def go_to_path(_event=None):
            nonlocal current_dir
            path = Path(current_var.get()).expanduser()
            if path.exists() and path.is_dir():
                current_dir = path
                refresh()
            else:
                status_var.set("Folder does not exist.")

        def open_selected_dir(_event=None):
            nonlocal current_dir
            selection = dir_list.curselection()
            if not selection:
                return
            current_dir = dir_paths[int(selection[0])]
            refresh()

        def select_all_files():
            if file_paths:
                file_list.selection_set(0, tk.END)
                file_list.focus_set()

        def accept(_event=None):
            nonlocal result
            selection = file_list.curselection()
            if not selection:
                messagebox.showwarning(title, "Select one or more files.", parent=win)
                return
            result = [str(file_paths[int(i)]) for i in selection]
            try:
                win.grab_release()
            except tk.TclError:
                pass
            win.destroy()

        def cancel():
            try:
                win.grab_release()
            except tk.TclError:
                pass
            win.destroy()

        path_entry.bind("<Return>", go_to_path)
        dir_list.bind("<Double-Button-1>", open_selected_dir)
        file_list.bind("<Double-Button-1>", accept)

        buttons = ttk.Frame(win)
        buttons.pack(fill=tk.X, padx=10, pady=(4, 10))
        ttk.Button(buttons, text="Go", command=go_to_path).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Open folder", command=open_selected_dir).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Select all files", command=select_all_files).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Cancel", command=cancel).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Use selected files", command=accept).pack(side=tk.RIGHT)

        refresh()
        win.protocol("WM_DELETE_WINDOW", cancel)
        win.grab_set()
        win.focus_force()
        win.wait_window()
        return result

    def create_widgets_modern(self):
        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(outer, highlightthickness=0, bd=0, bg=ChemTheme.COLORS['background'])
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        main = ttk.Frame(canvas)
        main_id = canvas.create_window((0, 0), window=main, anchor="nw")
        main.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(main_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        inner = ttk.Frame(main)
        inner.pack(fill=tk.BOTH, expand=True, padx=18, pady=12)

        files_card = self._make_card(inner, "Input")
        self._file_row(files_card, "Cluster sample peak-lists", "Import peak-lists", self.select_sample_files, "sample_file_label")
        self._file_row(files_card, "SDF database for direct analysis", "Import SDF", self.select_sdf_files, "sdf_file_label")

        mode_card = self._make_card(inner, "Mode")
        mode_content = ttk.Frame(mode_card, style='Card.TFrame')
        mode_content.pack(fill=tk.X, padx=14, pady=10)
        self.cluster_kind = tk.StringVar(value="C_MULTI")
        ttk.Radiobutton(mode_content, text="13C cross-sample clustering", variable=self.cluster_kind,
                        value="C_MULTI", command=self.toggle_cluster_kind).grid(row=0, column=0, sticky=tk.W, padx=(0, 18), pady=3)
        ttk.Radiobutton(mode_content, text="13C single-spectrum", variable=self.cluster_kind,
                        value="C_SINGLE", command=self.toggle_cluster_kind).grid(row=0, column=1, sticky=tk.W, padx=(0, 18), pady=3)
        ttk.Radiobutton(mode_content, text="HSQC full 2D clustering", variable=self.cluster_kind,
                        value="HSQC", command=self.toggle_cluster_kind).grid(row=0, column=2, sticky=tk.W, padx=(0, 18), pady=3)
        ttk.Label(mode_content, text="Frontend", style='Card.TLabel').grid(row=0, column=3, sticky=tk.E, padx=(12, 6), pady=3)
        self.c_frontend = ttk.Combobox(mode_content, values=C_FRONTEND_LABELS, width=20, state="readonly")
        self.c_frontend.set(DEFAULT_C_FRONTEND_LABEL)
        self.c_frontend.grid(row=0, column=4, sticky=tk.W, pady=3)
        self.c_frontend.bind("<<ComboboxSelected>>", self.on_frontend_changed)
        ttk.Label(mode_content, text="Backend", style='Card.TLabel').grid(row=0, column=5, sticky=tk.E, padx=(12, 6), pady=3)
        self.c_model = ttk.Combobox(mode_content, values=CROSS_MODEL_LABELS, width=14, state="readonly")
        self.c_model.set(DEFAULT_CROSS_MODEL_LABEL)
        self.c_model.grid(row=0, column=6, sticky=tk.W, pady=3)
        mode_content.columnconfigure(7, weight=1)

        param_card = self._make_card(inner, "Clustering Parameters")
        param_grid = ttk.Frame(param_card, style='Card.TFrame')
        param_grid.pack(fill=tk.X, padx=14, pady=(6, 12))
        param_grid.columnconfigure(0, weight=1)
        param_grid.columnconfigure(1, weight=1)

        left = ttk.Frame(param_grid, style='Card.TFrame')
        right = ttk.Frame(param_grid, style='Card.TFrame')
        left.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 14))
        right.grid(row=0, column=1, sticky=tk.NSEW, padx=(14, 0))

        self._section_title(left, "13C Regional Windows")
        ttk.Label(left, text="lo-hi: tolerance", style='Muted.TLabel').pack(anchor=tk.W, padx=14)
        self.region_text = tk.Text(left, height=5, width=36, font=ChemTheme.FONTS['input'], bd=1, relief=tk.SOLID)
        self.region_text.insert("1.0", self._region_default_text())
        self.region_text.pack(anchor=tk.W, fill=tk.X, padx=14, pady=(4, 10))

        c_params = ttk.Frame(left, style='Card.TFrame')
        c_params.pack(fill=tk.X)
        self._section_title(c_params, "13C Cross-Sample")
        c_entry_grid = ttk.Frame(c_params, style='Card.TFrame')
        c_entry_grid.pack(fill=tk.X)
        self.c_span_tol = self._param_entry(c_entry_grid, "Track span <=", GLOBAL_STRICT_DEFAULTS["span"], 0, 0)
        self.residual_gate = self._param_entry(c_entry_grid, "Corrected shift gate", GLOBAL_STRICT_DEFAULTS["residual_gate"], 0, 2)
        self.top_k_per_seed = self._hidden_entry("6")
        self.max_per_sample = self._hidden_entry("3")
        self.c_presence_only = tk.BooleanVar(value=False)
        self.c_presence_only_check = ttk.Checkbutton(
            c_entry_grid,
            text="Common-mask output only",
            variable=self.c_presence_only,
        )
        self.c_presence_only_check.grid(row=0, column=4, columnspan=2, sticky=tk.W, padx=4, pady=3)

        h_params = ttk.Frame(right, style='Card.TFrame')
        h_params.pack(fill=tk.X)
        self._section_title(h_params, "HSQC Full 2D")
        h_entry_grid = ttk.Frame(h_params, style='Card.TFrame')
        h_entry_grid.pack(fill=tk.X)
        self.hsqc_c_tol = self._param_entry(h_entry_grid, "C window", "1.0", 0, 0)
        self.hsqc_h_tol = self._param_entry(h_entry_grid, "H window", "0.1", 0, 2)
        self.hsqc_c_span_tol = self._param_entry(h_entry_grid, "C span <=", "1.0", 1, 0)
        self.hsqc_h_span_tol = self._param_entry(h_entry_grid, "H span <=", "0.1", 1, 2)

        single_params = ttk.Frame(right, style='Card.TFrame')
        single_params.pack(fill=tk.X, pady=(8, 0))
        self._section_title(single_params, "13C Single-Spectrum")
        single_entry_grid = ttk.Frame(single_params, style='Card.TFrame')
        single_entry_grid.pack(fill=tk.X)
        self.single_gmm_min = self._param_entry(single_entry_grid, "GMM min k", "1", 0, 0)
        self.single_gmm_max = self._param_entry(single_entry_grid, "GMM max k", "4", 0, 2)
        self.single_min_cluster = self._param_entry(single_entry_grid, "Min cluster", "1", 1, 0)
        self.single_gmm_use_log = tk.BooleanVar(value=True)
        ttk.Checkbutton(single_entry_grid, text="Log intensity", variable=self.single_gmm_use_log).grid(row=1, column=2, columnspan=2, sticky=tk.W, padx=4, pady=3)

        correction_card = self._make_card(inner, "Optional Intensity Correction")
        correction_params = ttk.Frame(correction_card, style='Card.TFrame')
        correction_params.pack(fill=tk.X, padx=14, pady=10)
        self.correction_enabled = tk.BooleanVar(value=False)
        self.correction_use_area = tk.BooleanVar(value=False)
        self.correction_use_type = tk.BooleanVar(value=False)
        ttk.Checkbutton(correction_params, text="Apply before clustering", variable=self.correction_enabled).grid(row=0, column=0, sticky=tk.W, padx=(0, 12), pady=3)
        ttk.Checkbutton(correction_params, text="Use area target", variable=self.correction_use_area).grid(row=0, column=1, sticky=tk.W, padx=(0, 12), pady=3)
        ttk.Checkbutton(correction_params, text="Use DEPT/type", variable=self.correction_use_type).grid(row=0, column=2, sticky=tk.W, padx=(0, 12), pady=3)
        ttk.Label(correction_params, text="Mode", style='Card.TLabel').grid(row=1, column=0, sticky=tk.W, padx=(0, 6), pady=3)
        self.correction_mode = ttk.Combobox(correction_params, values=["height_stabilize", "area_match", "auto"], width=15, state="readonly")
        self.correction_mode.set("height_stabilize")
        self.correction_mode.grid(row=1, column=1, sticky=tk.W, padx=(0, 12), pady=3)
        self.correction_shrink = self._param_entry(correction_params, "Shrink", "1.235", 1, 2)
        self.correction_delta_clip = self._param_entry(correction_params, "Delta clip", "0.63", 2, 0)
        self.correction_ppm_bin = self._param_entry(correction_params, "ppm bin", "9.75", 2, 2)
        self.correction_width_bin = self._param_entry(correction_params, "width bin", "0.35", 2, 4)

        search_card = self._make_card(inner, "PMTC/QG-PMTC Cluster Controls")
        search = ttk.Frame(search_card, style='Card.TFrame')
        search.pack(fill=tk.X, padx=14, pady=10)
        self.pmtc_min_cluster_size = self._param_entry(search, "PMTC min cluster", "3", 0, 0)
        self.max_tracks_3 = self._param_entry(search, "Max tracks n=3", GLOBAL_STRICT_DEFAULTS["max_tracks_3"], 1, 0)
        self.max_tracks_4 = self._param_entry(search, "n=4", GLOBAL_STRICT_DEFAULTS["max_tracks_4"], 1, 2)
        self.max_tracks_5 = self._param_entry(search, "n=5", GLOBAL_STRICT_DEFAULTS["max_tracks_5"], 1, 4)
        self.frac_3 = self._param_entry(search, "Frac n=3", GLOBAL_STRICT_DEFAULTS["frac_3"], 2, 0)
        self.frac_4 = self._param_entry(search, "n=4", GLOBAL_STRICT_DEFAULTS["frac_4"], 2, 2)
        self.frac_5 = self._param_entry(search, "n=5", GLOBAL_STRICT_DEFAULTS["frac_5"], 2, 4)

        self.min_track_size = self._hidden_entry("2")
        self.candidate_min_score = self._hidden_entry("0.40")
        self.pair_score_weight = self._hidden_entry("1.20")
        self.reciprocal_best_bonus = self._hidden_entry("0.18")
        self.exact_limit = self._hidden_entry("12")
        self.beam_width = self._hidden_entry("120")
        self.node_limit = self._hidden_entry("100000")

        direct_card = self._make_card(inner, "Direct Database Analysis")
        direct = ttk.Frame(direct_card, style='Card.TFrame')
        direct.pack(fill=tk.X, padx=14, pady=10)
        self.c_min = self._param_entry(direct, "C min", "5", 0, 0, width=6)
        self.c_max = self._param_entry(direct, "C max", "30", 0, 2, width=6)
        self.m_min = self._param_entry(direct, "MW min", "100", 0, 4, width=7)
        self.m_max = self._param_entry(direct, "MW max", "500", 0, 6, width=7)
        self.c_green = self._param_entry(direct, "Green threshold", "0.5", 1, 0, width=7)
        self.c_yellow = self._param_entry(direct, "Yellow threshold", "2.0", 1, 2, width=7)
        ttk.Label(direct, text="C_merge", style='Card.TLabel').grid(row=1, column=4, sticky=tk.W, padx=(12, 6), pady=5)
        ttk.Radiobutton(direct, text="Y", variable=self.c_merge, value="Y").grid(row=1, column=5, sticky=tk.W, padx=(0, 6), pady=5)
        ttk.Radiobutton(direct, text="N", variable=self.c_merge, value="N").grid(row=1, column=6, sticky=tk.W, padx=(0, 14), pady=5)

        hint = (
            "C files: CPPM,intensity[,area] or ppm,intensity[,area]. "
            "HSQC files: CPPM,HPPM,intensity[,area]. No-header numeric text is accepted."
        )
        ttk.Label(inner, text=hint, style='Muted.TLabel', wraplength=980).pack(fill=tk.X, padx=8, pady=(0, 12))

        action_bar = ttk.Frame(self)
        action_bar.pack(fill=tk.X, padx=18, pady=(0, 12))
        ttk.Button(action_bar, text="Execute Clustering", command=self.start_clustering).pack(side=tk.RIGHT, padx=5, ipadx=14, ipady=4)
        ttk.Button(action_bar, text="Reset", style='Secondary.TButton', command=self.reset_parameters).pack(side=tk.RIGHT, padx=5)

        self._apply_frontend_parameter_defaults()
        self.toggle_cluster_kind()

    def create_widgets(self):
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        source_frame = ttk.LabelFrame(main, text="1. Cluster input files")
        ttk.Button(source_frame, text="Import clustering sample files", command=self.select_sample_files).pack(side=tk.LEFT, padx=5, pady=8)
        self.sample_file_label = ttk.Label(source_frame, text="No sample files selected")
        self.sample_file_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        source_frame.pack(fill=tk.X, pady=5)

        db_frame = ttk.LabelFrame(main, text="2. SDF database files for optional direct analysis")
        ttk.Button(db_frame, text="Import SDF database", command=self.select_sdf_files).pack(side=tk.LEFT, padx=5, pady=8)
        self.sdf_file_label = ttk.Label(db_frame, text="No SDF files selected")
        self.sdf_file_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        db_frame.pack(fill=tk.X, pady=5)

        mode_frame = ttk.LabelFrame(main, text="3. Clustering mode")
        self.cluster_kind = tk.StringVar(value="C_MULTI")
        ttk.Radiobutton(mode_frame, text="13C cross-sample clustering", variable=self.cluster_kind,
                        value="C_MULTI", command=self.toggle_cluster_kind).pack(side=tk.LEFT, padx=6, pady=8)
        ttk.Radiobutton(mode_frame, text="13C single-spectrum clustering", variable=self.cluster_kind,
                        value="C_SINGLE", command=self.toggle_cluster_kind).pack(side=tk.LEFT, padx=6, pady=8)
        ttk.Radiobutton(mode_frame, text="HSQC full 2D clustering", variable=self.cluster_kind,
                        value="HSQC", command=self.toggle_cluster_kind).pack(side=tk.LEFT, padx=6, pady=8)
        ttk.Label(mode_frame, text=" Frontend:").pack(side=tk.LEFT, padx=(14, 2))
        self.c_frontend = ttk.Combobox(mode_frame, values=C_FRONTEND_LABELS, width=20, state="readonly")
        self.c_frontend.set(DEFAULT_C_FRONTEND_LABEL)
        self.c_frontend.bind("<<ComboboxSelected>>", self.on_frontend_changed)
        self.c_frontend.pack(side=tk.LEFT, padx=2)
        ttk.Label(mode_frame, text=" Backend:").pack(side=tk.LEFT, padx=(14, 2))
        self.c_model = ttk.Combobox(mode_frame, values=CROSS_MODEL_LABELS, width=12, state="readonly")
        self.c_model.set(DEFAULT_CROSS_MODEL_LABEL)
        self.c_model.pack(side=tk.LEFT, padx=2)
        mode_frame.pack(fill=tk.X, pady=5)

        param_outer = ttk.LabelFrame(main, text="4. Clustering parameters")
        param_outer.pack(fill=tk.X, pady=5)

        region_frame = ttk.Frame(param_outer)
        ttk.Label(region_frame, text="13C ppm regional windows\nlo-hi: tolerance").pack(side=tk.LEFT, padx=(6, 4), pady=3)
        self.region_text = tk.Text(region_frame, height=4, width=34, font=ChemTheme.FONTS['input'])
        self.region_text.insert("1.0", self._region_default_text())
        self.region_text.pack(side=tk.LEFT, padx=4, pady=4)
        region_frame.pack(fill=tk.X, pady=(2, 0))

        align_frame = ttk.Frame(param_outer)
        self.c_span_tol = self._labeled_entry(align_frame, "C track span ≤", "1.0", 0, 0)
        self.residual_gate = self._labeled_entry(align_frame, "C corrected gate", "1.0", 0, 2)
        self.top_k_per_seed = self._labeled_entry(align_frame, "top_k/seed", "6", 0, 4)
        self.max_per_sample = self._labeled_entry(align_frame, "max/sample", "3", 0, 6)

        self.hsqc_c_tol = self._labeled_entry(align_frame, "HSQC C window", "1.0", 1, 0)
        self.hsqc_h_tol = self._labeled_entry(align_frame, "HSQC H window", "0.1", 1, 2)
        self.hsqc_c_span_tol = self._labeled_entry(align_frame, "HSQC C span ≤", "1.0", 1, 4)
        self.hsqc_h_span_tol = self._labeled_entry(align_frame, "HSQC H span ≤", "0.1", 1, 6)
        align_frame.pack(fill=tk.X, pady=2)

        score_frame = ttk.Frame(param_outer)
        self.min_track_size = self._labeled_entry(score_frame, "min track size", "2", 0, 0)
        self.candidate_min_score = self._labeled_entry(score_frame, "candidate min score", "0.40", 0, 2)
        self.pair_score_weight = self._labeled_entry(score_frame, "pair score weight", "1.20", 0, 4)
        self.reciprocal_best_bonus = self._labeled_entry(score_frame, "reciprocal bonus", "0.18", 0, 6)
        self.exact_limit = self._labeled_entry(score_frame, "exact limit", "12", 1, 0)
        self.beam_width = self._labeled_entry(score_frame, "beam width", "120", 1, 2)
        self.node_limit = self._labeled_entry(score_frame, "node limit", "100000", 1, 4, width=10)
        score_frame.pack(fill=tk.X, pady=2)

        pmtc_frame = ttk.Frame(param_outer)
        self.c_presence_only = tk.BooleanVar(value=False)
        self.c_presence_only_check = ttk.Checkbutton(pmtc_frame, text="Common-mask output only", variable=self.c_presence_only)
        self.c_presence_only_check.grid(row=2, column=0, columnspan=4, sticky=tk.W, padx=4, pady=3)
        self.pmtc_min_cluster_size = self._labeled_entry(pmtc_frame, "PMTC min cluster", "3", 0, 0)
        self.max_tracks_3 = self._labeled_entry(pmtc_frame, "max tracks n=3", GLOBAL_STRICT_DEFAULTS["max_tracks_3"], 0, 2)
        self.max_tracks_4 = self._labeled_entry(pmtc_frame, "n=4", GLOBAL_STRICT_DEFAULTS["max_tracks_4"], 0, 4)
        self.max_tracks_5 = self._labeled_entry(pmtc_frame, "n=5", GLOBAL_STRICT_DEFAULTS["max_tracks_5"], 0, 6)
        self.frac_3 = self._labeled_entry(pmtc_frame, "frac n=3", GLOBAL_STRICT_DEFAULTS["frac_3"], 1, 0)
        self.frac_4 = self._labeled_entry(pmtc_frame, "n=4", GLOBAL_STRICT_DEFAULTS["frac_4"], 1, 2)
        self.frac_5 = self._labeled_entry(pmtc_frame, "n=5", GLOBAL_STRICT_DEFAULTS["frac_5"], 1, 4)
        pmtc_frame.pack(fill=tk.X, pady=(2, 5))

        single_frame = ttk.Frame(param_outer)
        self.single_gap_factor = self._labeled_entry(single_frame, "single gap factor", "1.8", 0, 0)
        self.single_max_peaks = self._labeled_entry(single_frame, "single max peaks", "30", 0, 2)
        self.single_min_cluster = self._labeled_entry(single_frame, "single min cluster", "1", 0, 4)
        self.single_intensity_weight = self._labeled_entry(single_frame, "single intensity split", "0.10", 0, 6)
        single_frame.pack(fill=tk.X, pady=(2, 5))

        db_param_frame = ttk.LabelFrame(main, text="5. Parameters used when a cluster is directly sent to database analysis")
        ttk.Label(db_param_frame, text="C_num:").pack(side=tk.LEFT, padx=(6, 2))
        self.c_min = ttk.Entry(db_param_frame, width=5)
        self.c_min.insert(0, "5")
        self.c_min.pack(side=tk.LEFT)
        ttk.Label(db_param_frame, text="-").pack(side=tk.LEFT)
        self.c_max = ttk.Entry(db_param_frame, width=5)
        self.c_max.insert(0, "30")
        self.c_max.pack(side=tk.LEFT)
        ttk.Label(db_param_frame, text=" MW:").pack(side=tk.LEFT, padx=(10, 2))
        self.m_min = ttk.Entry(db_param_frame, width=6)
        self.m_min.insert(0, "100")
        self.m_min.pack(side=tk.LEFT)
        ttk.Label(db_param_frame, text="-").pack(side=tk.LEFT)
        self.m_max = ttk.Entry(db_param_frame, width=6)
        self.m_max.insert(0, "500")
        self.m_max.pack(side=tk.LEFT)
        ttk.Label(db_param_frame, text=" C threshold:").pack(side=tk.LEFT, padx=(10, 2))
        self.c_green = ttk.Entry(db_param_frame, width=6)
        self.c_green.insert(0, "0.5")
        self.c_green.pack(side=tk.LEFT)
        ttk.Label(db_param_frame, text="-").pack(side=tk.LEFT)
        self.c_yellow = ttk.Entry(db_param_frame, width=6)
        self.c_yellow.insert(0, "2.0")
        self.c_yellow.pack(side=tk.LEFT)
        ttk.Label(db_param_frame, text=" C_merge:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Radiobutton(db_param_frame, text="Y", variable=self.c_merge, value="Y").pack(side=tk.LEFT)
        ttk.Radiobutton(db_param_frame, text="N", variable=self.c_merge, value="N").pack(side=tk.LEFT)
        db_param_frame.pack(fill=tk.X, pady=5)

        hint = (
            "Input format: C cross-sample and C single-spectrum files use CPPM,intensity[,area] or ppm,intensity[,area]. "
            "HSQC full 2D files are the same clustering table with one extra H column after CPPM: "
            "CPPM,HPPM,intensity[,area]. No-header numeric text is also accepted."
        )
        ttk.Label(main, text=hint, wraplength=880).pack(fill=tk.X, pady=5)

        btn_frame = ttk.Frame(main)
        ttk.Button(btn_frame, text="Execute clustering analysis", command=self.start_clustering).pack(side=tk.LEFT, padx=5, ipadx=12, ipady=4)
        ttk.Button(btn_frame, text="Reset", command=self.reset_parameters).pack(side=tk.LEFT, padx=5)
        btn_frame.pack(pady=10)

        self._apply_frontend_parameter_defaults()
        self.toggle_cluster_kind()

    def select_sample_files(self):
        files = self._choose_files_in_app(
            "Select sample peak-list files",
            self._sample_initial_dir(),
            (".csv", ".tsv", ".txt", ".list", ".peak", ".peaks"),
        )
        if files:
            self.sample_files = list(files)
            self._last_sample_dir = str(Path(self.sample_files[0]).parent)
            self.sample_file_label.config(text="\n".join([Path(f).name for f in self.sample_files]))

    def select_sdf_files(self):
        files = self._choose_files_in_app(
            "Select SDF file(s) for database analysis",
            self._sdf_initial_dir(),
            (".sdf", ".sd"),
        )
        if files:
            self.sdf_files = [f for f in files if str(f).lower().endswith((".sdf", ".sd"))]
            self._last_sdf_dir = str(Path(self.sdf_files[0]).parent) if self.sdf_files else self._last_sdf_dir
            self.sdf_file_label.config(text="\n".join([Path(f).name for f in self.sdf_files]))

    def toggle_cluster_kind(self):
        kind = self.cluster_kind.get()
        self.c_model.config(state="readonly" if kind in ("C_MULTI", "HSQC") else "disabled")
        self.c_frontend.config(state="readonly" if kind in ("C_MULTI", "HSQC") else "disabled")

        # Cross-sample C and full 2D-HSQC both use SPTC-style track enumeration / set-packing controls.
        cross_state = "normal" if kind in ("C_MULTI", "HSQC") else "disabled"
        c_cross_state = "normal" if kind == "C_MULTI" else "disabled"
        h_state = "normal" if kind == "HSQC" else "disabled"
        single_state = "normal" if kind == "C_SINGLE" else "disabled"
        correction_state = "normal" if kind in ("C_MULTI", "C_SINGLE", "HSQC") else "disabled"
        frontend_key = self._selected_frontend_key() if kind in ("C_MULTI", "HSQC") else ""
        mask_only_state = "normal" if kind in ("C_MULTI", "HSQC") and frontend_key == "common_mask" else "disabled"
        if mask_only_state == "disabled":
            self.c_presence_only.set(False)

        for ent in [self.c_span_tol, self.residual_gate]:
            ent.config(state=c_cross_state)
        self.c_presence_only_check.config(state=mask_only_state)
        for ent in [self.top_k_per_seed, self.max_per_sample, self.exact_limit, self.beam_width, self.node_limit,
                    self.min_track_size, self.candidate_min_score, self.pair_score_weight, self.reciprocal_best_bonus]:
            ent.config(state=cross_state)
        for ent in [self.hsqc_c_tol, self.hsqc_h_tol, self.hsqc_c_span_tol, self.hsqc_h_span_tol]:
            ent.config(state=h_state)
        for ent in [self.single_gmm_min, self.single_gmm_max, self.single_min_cluster]:
            ent.config(state=single_state)
        for widget in [
            self.correction_mode,
            self.correction_shrink,
            self.correction_delta_clip,
            self.correction_ppm_bin,
            self.correction_width_bin,
        ]:
            widget.config(state=correction_state if widget is not self.correction_mode else ("readonly" if correction_state == "normal" else "disabled"))
        for ent in [self.pmtc_min_cluster_size, self.max_tracks_3, self.max_tracks_4, self.max_tracks_5,
                    self.frac_3, self.frac_4, self.frac_5]:
            ent.config(state=cross_state)

    def reset_parameters(self):
        self.sample_files = []
        self.sdf_files = []
        self.sample_file_label.config(text="No sample files selected")
        self.sdf_file_label.config(text="No SDF files selected")
        self.cluster_kind.set("C_MULTI")
        self.c_frontend.set(DEFAULT_C_FRONTEND_LABEL)
        self.c_model.set(DEFAULT_CROSS_MODEL_LABEL)
        self.c_merge.set("N")
        self.correction_enabled.set(False)
        self.correction_use_area.set(False)
        self.correction_use_type.set(False)
        self.correction_mode.set("height_stabilize")
        self.c_presence_only.set(False)
        self.single_gmm_use_log.set(True)
        self.region_text.delete("1.0", tk.END)
        self.region_text.insert("1.0", self._region_default_text())
        for entry, val in [
            (self.c_span_tol, GLOBAL_STRICT_DEFAULTS["span"]), (self.residual_gate, GLOBAL_STRICT_DEFAULTS["residual_gate"]), (self.top_k_per_seed, "6"),
            (self.max_per_sample, "3"), (self.hsqc_c_tol, "1.0"), (self.hsqc_h_tol, "0.1"),
            (self.hsqc_c_span_tol, "1.0"), (self.hsqc_h_span_tol, "0.1"),
            (self.min_track_size, "2"), (self.candidate_min_score, "0.40"),
            (self.pair_score_weight, "1.20"), (self.reciprocal_best_bonus, "0.18"),
            (self.exact_limit, "12"), (self.beam_width, "120"), (self.node_limit, "100000"),
            (self.pmtc_min_cluster_size, "3"), (self.max_tracks_3, GLOBAL_STRICT_DEFAULTS["max_tracks_3"]), (self.max_tracks_4, GLOBAL_STRICT_DEFAULTS["max_tracks_4"]),
            (self.max_tracks_5, GLOBAL_STRICT_DEFAULTS["max_tracks_5"]), (self.frac_3, GLOBAL_STRICT_DEFAULTS["frac_3"]), (self.frac_4, GLOBAL_STRICT_DEFAULTS["frac_4"]), (self.frac_5, GLOBAL_STRICT_DEFAULTS["frac_5"]),
            (self.single_gmm_min, "1"), (self.single_gmm_max, "4"), (self.single_min_cluster, "1"),
            (self.correction_shrink, "1.235"), (self.correction_delta_clip, "0.63"),
            (self.correction_ppm_bin, "9.75"), (self.correction_width_bin, "0.35"),
            (self.c_min, "5"), (self.c_max, "30"), (self.m_min, "100"), (self.m_max, "500"),
            (self.c_green, "0.5"), (self.c_yellow, "2.0")
        ]:
            self._set_entry(entry, val)
        self.toggle_cluster_kind()

    def _selected_model_key(self):
        return model_key_from_display(self.c_model.get())

    def _selected_frontend_key(self):
        label = self.c_frontend.get() if hasattr(self, "c_frontend") else DEFAULT_C_FRONTEND_LABEL
        return "common_mask" if str(label).strip().lower() == "common-mask" else "global_strict"

    def on_frontend_changed(self, _event=None):
        self._apply_frontend_parameter_defaults()
        self.toggle_cluster_kind()

    def _frontend_parameter_defaults(self):
        return COMMON_MASK_DEFAULTS if self._selected_frontend_key() == "common_mask" else GLOBAL_STRICT_DEFAULTS

    def _apply_frontend_parameter_defaults(self):
        if not all(hasattr(self, name) for name in ("c_span_tol", "residual_gate", "max_tracks_3", "max_tracks_4", "max_tracks_5", "frac_3", "frac_4", "frac_5")):
            return
        defaults = self._frontend_parameter_defaults()
        self._set_entry(self.c_span_tol, defaults["span"])
        self._set_entry(self.residual_gate, defaults["residual_gate"])
        self._set_entry(self.max_tracks_3, defaults["max_tracks_3"])
        self._set_entry(self.max_tracks_4, defaults["max_tracks_4"])
        self._set_entry(self.max_tracks_5, defaults["max_tracks_5"])
        self._set_entry(self.frac_3, defaults["frac_3"])
        self._set_entry(self.frac_4, defaults["frac_4"])
        self._set_entry(self.frac_5, defaults["frac_5"])

    def _setpacking_high_mask_bonus(self):
        return float(self._frontend_parameter_defaults()["setpacking_high_mask_bonus"])

    def _guarded_quality_max_cluster_rise(self):
        return int(self._frontend_parameter_defaults()["qg_max_rise"])

    def _common_cluster_params(self):
        region_windows = parse_region_windows(self.region_text.get("1.0", tk.END))
        pmtc_max_tracks = {
            3: int(self.max_tracks_3.get()),
            4: int(self.max_tracks_4.get()),
            5: int(self.max_tracks_5.get()),
        }
        pmtc_frac = {
            3: float(self.frac_3.get()),
            4: float(self.frac_4.get()),
            5: float(self.frac_5.get()),
        }
        return region_windows, pmtc_max_tracks, pmtc_frac

    def _correction_options(self):
        clip_text = self.correction_delta_clip.get().strip()
        delta_clip = None if clip_text.lower() in {"", "none", "null", "off"} else float(clip_text)
        return {
            "enabled": bool(self.correction_enabled.get()),
            "use_type": bool(self.correction_use_type.get()),
            "use_area": bool(self.correction_use_area.get()),
            "calibration_mode": self.correction_mode.get(),
            "shrink_strength": float(self.correction_shrink.get()),
            "delta_clip": delta_clip,
            "ppm_bin_width": float(self.correction_ppm_bin.get()),
            "width_bin_width": float(self.correction_width_bin.get()),
        }

    def _new_cluster_output_dir(self, kind):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("results") / "gui_cluster_runs" / f"{stamp}_{kind.lower()}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def start_clustering(self):
        try:
            if not self.sample_files:
                raise ValueError("Please select clustering peak-list file(s).")
            kind = self.cluster_kind.get()
            region_windows, pmtc_max_tracks, pmtc_frac = self._common_cluster_params()
            run_dir = self._new_cluster_output_dir(kind)
            correction_options = self._correction_options()

            if kind == "C_MULTI":
                if len(self.sample_files) < 2:
                    raise ValueError("13C cross-sample clustering needs at least two sample peak-list files.")
                model_key = self._selected_model_key()
                frontend_key = self._selected_frontend_key()
                if bool(self.c_presence_only.get()):
                    if frontend_key != "common_mask":
                        raise ValueError("Mask-only output is available only for the common-mask front end.")
                    clusters = run_c_presence_mask_clustering(
                        self.sample_files,
                        model=model_key,
                        region_windows=region_windows,
                        span_tol=float(self.c_span_tol.get()),
                        residual_gate=float(self.residual_gate.get()),
                        top_k_per_seed=int(self.top_k_per_seed.get()),
                        max_per_sample=int(self.max_per_sample.get()),
                        exact_limit=int(self.exact_limit.get()),
                        beam_width=int(self.beam_width.get()),
                        node_limit=int(self.node_limit.get()),
                        setpacking_high_mask_bonus=self._setpacking_high_mask_bonus(),
                        correction_options=correction_options,
                        output_dir=run_dir,
                    )
                elif frontend_key == "common_mask":
                    clusters = run_c_common_mask_backend_clustering(
                        self.sample_files,
                        model=model_key,
                        region_windows=region_windows,
                        span_tol=float(self.c_span_tol.get()),
                        residual_gate=float(self.residual_gate.get()),
                        top_k_per_seed=int(self.top_k_per_seed.get()),
                        max_per_sample=int(self.max_per_sample.get()),
                        exact_limit=int(self.exact_limit.get()),
                        beam_width=int(self.beam_width.get()),
                        node_limit=int(self.node_limit.get()),
                        pmtc_max_tracks_by_n_samples=pmtc_max_tracks,
                        pmtc_frac_limit_by_n_samples=pmtc_frac,
                        pmtc_min_cluster_size=int(self.pmtc_min_cluster_size.get()),
                        setpacking_high_mask_bonus=self._setpacking_high_mask_bonus(),
                        guarded_quality_max_cluster_rise=self._guarded_quality_max_cluster_rise(),
                        correction_options=correction_options,
                        output_dir=run_dir,
                    )
                else:
                    clusters = run_c_v5_clustering(
                        self.sample_files,
                        model=model_key,
                        region_windows=region_windows,
                        span_tol=float(self.c_span_tol.get()),
                        residual_gate=float(self.residual_gate.get()),
                        top_k_per_seed=int(self.top_k_per_seed.get()),
                        max_per_sample=int(self.max_per_sample.get()),
                        exact_limit=int(self.exact_limit.get()),
                        beam_width=int(self.beam_width.get()),
                        node_limit=int(self.node_limit.get()),
                        pmtc_max_tracks_by_n_samples=pmtc_max_tracks,
                        pmtc_frac_limit_by_n_samples=pmtc_frac,
                        pmtc_min_cluster_size=int(self.pmtc_min_cluster_size.get()),
                        setpacking_high_mask_bonus=self._setpacking_high_mask_bonus(),
                        guarded_quality_max_cluster_rise=self._guarded_quality_max_cluster_rise(),
                        correction_options=correction_options,
                        output_dir=run_dir,
                    )
                result_kind = "C"
            elif kind == "C_SINGLE":
                clusters = run_c_single_spectrum_clustering(
                    self.sample_files,
                    region_windows=region_windows,
                    min_cluster_size=int(self.single_min_cluster.get()),
                    correction_enabled=bool(correction_options["enabled"]),
                    use_type=bool(correction_options["use_type"]),
                    use_area=bool(correction_options["use_area"]),
                    calibration_mode=str(correction_options["calibration_mode"]),
                    shrink_strength=float(correction_options["shrink_strength"]),
                    delta_clip=correction_options["delta_clip"],
                    ppm_bin_width=float(correction_options["ppm_bin_width"]),
                    width_bin_width=float(correction_options["width_bin_width"]),
                    gmm_min_components=int(self.single_gmm_min.get()),
                    gmm_max_components=int(self.single_gmm_max.get()),
                    gmm_use_log=bool(self.single_gmm_use_log.get()),
                    output_dir=run_dir,
                )
                result_kind = "C"
            else:
                if len(self.sample_files) < 2:
                    raise ValueError("HSQC cross-spectrum clustering needs at least two sample peak-list files.")
                model_key = self._selected_model_key()
                frontend_key = self._selected_frontend_key()
                hsqc_kwargs = dict(
                    model=model_key,
                    c_tol=float(self.hsqc_c_tol.get()),
                    h_tol=float(self.hsqc_h_tol.get()),
                    c_span_tol=float(self.hsqc_c_span_tol.get()),
                    h_span_tol=float(self.hsqc_h_span_tol.get()),
                    min_track_size=int(self.min_track_size.get()),
                    candidate_min_score=float(self.candidate_min_score.get()),
                    pair_score_weight=float(self.pair_score_weight.get()),
                    reciprocal_best_bonus=float(self.reciprocal_best_bonus.get()),
                    top_k_per_seed=int(self.top_k_per_seed.get()),
                    max_per_sample=int(self.max_per_sample.get()),
                    exact_limit=int(self.exact_limit.get()),
                    beam_width=int(self.beam_width.get()),
                    node_limit=int(self.node_limit.get()),
                    pmtc_max_tracks_by_n_samples=pmtc_max_tracks,
                    pmtc_frac_limit_by_n_samples=pmtc_frac,
                    pmtc_min_cluster_size=int(self.pmtc_min_cluster_size.get()),
                    guarded_quality_max_cluster_rise=self._guarded_quality_max_cluster_rise(),
                    correction_options=correction_options,
                    output_dir=run_dir,
                )
                if bool(self.c_presence_only.get()):
                    if frontend_key != "common_mask":
                        raise ValueError("Mask-only output is available only for the common-mask front end.")
                    clusters = run_hsqc_common_mask_backend_clustering(
                        self.sample_files,
                        presence_only=True,
                        **hsqc_kwargs,
                    )
                elif frontend_key == "common_mask":
                    clusters = run_hsqc_common_mask_backend_clustering(
                        self.sample_files,
                        presence_only=False,
                        **hsqc_kwargs,
                    )
                else:
                    clusters = run_hsqc_cross_clustering(
                        self.sample_files,
                        presence_only=False,
                        **hsqc_kwargs,
                    )
                result_kind = "HSQC"

            if not clusters:
                messagebox.showinfo("No results", "No clusters were produced.")
                return
            ClusterResultWindow(self.master, self, clusters, result_kind, output_dir=str(run_dir)).create_result_window()
        except Exception as e:
            messagebox.showerror("Clustering error", str(e))

    def _ensure_sdf_files(self):
        if self.sdf_files:
            return self.sdf_files
        files = self._choose_files_in_app(
            "Select SDF file(s) for database analysis",
            self._sdf_initial_dir(),
            (".sdf", ".sd"),
        )
        self.sdf_files = [f for f in files if str(f).lower().endswith((".sdf", ".sd"))]
        if self.sdf_files:
            self._last_sdf_dir = str(Path(self.sdf_files[0]).parent)
            self.sdf_file_label.config(text="\n".join([Path(f).name for f in self.sdf_files]))
        return self.sdf_files

    def _direct_c_merge_mode(self):
        value = str(self.c_merge.get()).strip().upper()
        if value not in {"Y", "N"}:
            value = "N"
            self.c_merge.set(value)
        return value

    def analyze_c_text(self, text):
        try:
            c_values = parse_c_values(text)
            if not c_values:
                raise ValueError("The edited C cluster is empty.")
            sdf_files = self._ensure_sdf_files()
            if not sdf_files:
                return
            config = {
                'sdf_files': sdf_files,
                'c_mode': 'untyped',
                'C_merge': self._direct_c_merge_mode(),
                'c_data': c_values,
                'c_range': tuple(map(int, (self.c_min.get(), self.c_max.get()))),
                'fw_range': tuple(map(float, (self.m_min.get(), self.m_max.get()))),
                'score_mode': 'global',
                'global_thresholds': tuple(map(float, (self.c_green.get(), self.c_yellow.get()))),
                'env_level': 1,
                'self_weight': 0.7,
                'env_weight': 0.3,
            }
            pipeline = CarbonOnlyScorerGUI(**config)
            try:
                results = pipeline.execute()
            finally:
                _close_carbon_thread_connection()
            visualizer = CarbonResultVisualizer(db_path="chem_data.db", top_results=results)
            visualizer.generate_plots(output_dir="molecule_plots")
            ResultViewer("chem_data.db", results).create_result_window(self.master)
        except Exception as e:
            messagebox.showerror("C analysis error", str(e))

    def analyze_hsqc_text(self, text):
        try:
            points = parse_hsqc_points(text)
            if not points:
                raise ValueError("The edited HSQC cluster is empty.")
            sdf_files = self._ensure_sdf_files()
            if not sdf_files:
                return
            c_tol = float(self.hsqc_c_tol.get() or 1.0)
            h_tol = float(self.hsqc_h_tol.get() or 0.1)
            config = {
                'sdf_files': sdf_files,
                'ch_mode': 1,
                'CH_merge': 'N',
                'ch_data': {'All_type': points},
                'tolerances': (c_tol, h_tol),
            }
            pipeline = CHOnlyScorerGUI(**config)
            results = pipeline.execute()
            visualizer = CHMatchVisualizer(db_path="chem_data.db", top_results=results, mode=1)
            viewer = CHResultViewer(db_path="chem_data.db", results=results)
            viewer.create_result_window(self.master)
            visualizer.generate_plots()
        except Exception as e:
            messagebox.showerror("HSQC analysis error", str(e))



class NMRPredictionFrame(ttk.Frame):
    """CORD-NMR wrapper around the portable NMR prediction engine."""

    def __init__(self, master):
        super().__init__(master)
        self.portable_root = tk.StringVar(value=str(DEFAULT_NMR_PREDICTOR_ROOT))
        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.mode = tk.StringVar(value="CH")
        self.input_type = tk.StringVar(value="sdf")
        self.c_engine = tk.StringVar(value="nmrnet")
        self.max_conformers = tk.StringVar(value="9")
        self.max_iters = tk.StringVar(value="300")
        self.forcefield = tk.StringVar(value="auto")
        self.time_limit_seconds = tk.StringVar(value="20")
        self.coord_route = tk.StringVar(value="standard")
        self.route_initial_confs = tk.StringVar(value="27")
        self.route_prune_rms_thresh = tk.StringVar(value="0.5")
        self.route_coarse_steps = tk.StringVar(value="10")
        self.route_keep_top_k = tk.StringVar(value="9")
        self.route_fine_steps = tk.StringVar(value="300")
        self.smiles_column = tk.StringVar(value="smiles")
        self.id_column = tk.StringVar(value="id")
        self.optimize_existing = tk.BooleanVar(value=True)
        self.allow_2d_if_h_nonzero = tk.BooleanVar(value=True)
        self._running = False
        self.create_widgets()
        self.refresh_portable_status()

    def create_widgets(self):
        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True, padx=18, pady=12)

        source = ttk.LabelFrame(outer, text="NMR Predictor Engine")
        source.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(source, text="Portable root").grid(row=0, column=0, sticky=tk.W, padx=8, pady=8)
        ttk.Entry(source, textvariable=self.portable_root, width=92).grid(row=0, column=1, sticky=tk.EW, padx=6, pady=8)
        ttk.Button(source, text="Browse", command=self.select_portable_root).grid(row=0, column=2, padx=4, pady=8)
        ttk.Button(source, text="Default", command=self.use_default_root).grid(row=0, column=3, padx=4, pady=8)
        self.status_label = ttk.Label(source, text="", style='Card.TLabel')
        self.status_label.grid(row=1, column=1, columnspan=3, sticky=tk.W, padx=6, pady=(0, 8))
        source.columnconfigure(1, weight=1)

        files = ttk.LabelFrame(outer, text="Input and Output")
        files.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(files, text="Input file").grid(row=0, column=0, sticky=tk.W, padx=8, pady=8)
        ttk.Entry(files, textvariable=self.input_path, width=92).grid(row=0, column=1, sticky=tk.EW, padx=6, pady=8)
        ttk.Button(files, text="Select", command=self.select_input_file).grid(row=0, column=2, padx=4, pady=8)
        ttk.Label(files, text="Output folder").grid(row=1, column=0, sticky=tk.W, padx=8, pady=8)
        ttk.Entry(files, textvariable=self.output_dir, width=92).grid(row=1, column=1, sticky=tk.EW, padx=6, pady=8)
        ttk.Button(files, text="Select", command=self.select_output_dir).grid(row=1, column=2, padx=4, pady=8)
        files.columnconfigure(1, weight=1)

        opts = ttk.LabelFrame(outer, text="Prediction Options")
        opts.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(opts, text="Mode").grid(row=0, column=0, sticky=tk.W, padx=8, pady=6)
        ttk.Combobox(opts, textvariable=self.mode, values=["C", "H", "CH"], width=10, state="readonly").grid(row=0, column=1, sticky=tk.W, padx=6, pady=6)
        ttk.Label(opts, text="Input type").grid(row=0, column=2, sticky=tk.W, padx=8, pady=6)
        ttk.Combobox(opts, textvariable=self.input_type, values=["sdf", "csv"], width=10, state="readonly").grid(row=0, column=3, sticky=tk.W, padx=6, pady=6)
        ttk.Label(opts, text="C engine").grid(row=0, column=4, sticky=tk.W, padx=8, pady=6)
        ttk.Combobox(opts, textvariable=self.c_engine, values=["nmrnet", "cascade2"], width=12, state="readonly").grid(row=0, column=5, sticky=tk.W, padx=6, pady=6)
        ttk.Label(opts, text="Forcefield").grid(row=0, column=6, sticky=tk.W, padx=8, pady=6)
        ttk.Combobox(opts, textvariable=self.forcefield, values=["auto", "mmff", "uff"], width=10, state="readonly").grid(row=0, column=7, sticky=tk.W, padx=6, pady=6)
        ttk.Label(opts, text="Conformers").grid(row=1, column=0, sticky=tk.W, padx=8, pady=6)
        ttk.Entry(opts, textvariable=self.max_conformers, width=12).grid(row=1, column=1, sticky=tk.W, padx=6, pady=6)
        ttk.Label(opts, text="Optimization steps").grid(row=1, column=2, sticky=tk.W, padx=8, pady=6)
        ttk.Entry(opts, textvariable=self.max_iters, width=12).grid(row=1, column=3, sticky=tk.W, padx=6, pady=6)
        ttk.Label(opts, text="Timeout / molecule").grid(row=1, column=4, sticky=tk.W, padx=8, pady=6)
        ttk.Entry(opts, textvariable=self.time_limit_seconds, width=12).grid(row=1, column=5, sticky=tk.W, padx=6, pady=6)
        ttk.Label(opts, text="3D route").grid(row=1, column=6, sticky=tk.W, padx=8, pady=6)
        ttk.Combobox(opts, textvariable=self.coord_route, values=["standard", "staged27"], width=10, state="readonly").grid(row=1, column=7, sticky=tk.W, padx=6, pady=6)
        ttk.Checkbutton(opts, text="Optimize existing 3D coordinates", variable=self.optimize_existing).grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=8, pady=6)
        ttk.Checkbutton(opts, text="Allow 2D fallback with nonzero H coordinates", variable=self.allow_2d_if_h_nonzero).grid(row=2, column=3, columnspan=5, sticky=tk.W, padx=8, pady=6)

        csv_opts = ttk.LabelFrame(outer, text="CSV Columns")
        csv_opts.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(csv_opts, text="SMILES column").grid(row=0, column=0, sticky=tk.W, padx=8, pady=6)
        ttk.Entry(csv_opts, textvariable=self.smiles_column, width=16).grid(row=0, column=1, sticky=tk.W, padx=6, pady=6)
        ttk.Label(csv_opts, text="ID column").grid(row=0, column=2, sticky=tk.W, padx=8, pady=6)
        ttk.Entry(csv_opts, textvariable=self.id_column, width=16).grid(row=0, column=3, sticky=tk.W, padx=6, pady=6)

        adv = ttk.LabelFrame(outer, text="Staged 3D Parameters")
        adv.pack(fill=tk.X, pady=(0, 10))
        labels = [
            ("Initial conformers", self.route_initial_confs),
            ("RMS prune", self.route_prune_rms_thresh),
            ("Coarse steps", self.route_coarse_steps),
            ("Keep top", self.route_keep_top_k),
            ("Fine steps", self.route_fine_steps),
        ]
        for col, (text, var) in enumerate(labels):
            ttk.Label(adv, text=text).grid(row=0, column=col * 2, sticky=tk.W, padx=8, pady=6)
            ttk.Entry(adv, textvariable=var, width=12).grid(row=0, column=col * 2 + 1, sticky=tk.W, padx=6, pady=6)

        actions = ttk.Frame(outer)
        actions.pack(fill=tk.X, pady=(0, 10))
        self.run_button = ttk.Button(actions, text="Run Prediction", command=self.run_prediction)
        self.run_button.pack(side=tk.LEFT, padx=(0, 8), ipadx=12, ipady=4)
        ttk.Button(actions, text="Open Output", style='Secondary.TButton', command=self.open_output_dir).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Check Engine", style='Secondary.TButton', command=self.check_engine).pack(side=tk.LEFT, padx=4)

        log_box = ttk.LabelFrame(outer, text="Run Log")
        log_box.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_box, height=18, wrap=tk.WORD, font=ChemTheme.FONTS['input'])
        scroll = ttk.Scrollbar(log_box, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=6)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 6), pady=6)

    def use_default_root(self):
        self.portable_root.set(str(DEFAULT_NMR_PREDICTOR_ROOT))
        self.refresh_portable_status()

    def select_portable_root(self):
        initial = str(Path(self.portable_root.get()).parent)
        path = filedialog.askdirectory(title="Select NMR-Predictor-Portable root", initialdir=initial)
        if path:
            self.portable_root.set(path)
            self.refresh_portable_status()

    def refresh_portable_status(self):
        status = describe_nmr_predictor_root(self.portable_root.get())
        if status.ready_for_cascade2:
            text = "Ready: NMRNet and CASCADE-2.0 runtimes found."
        elif status.ready_for_nmrnet:
            text = "Ready for NMRNet. CASCADE-2.0 runtime is missing."
        else:
            text = "Not ready. Missing NMRNet runtime or migrated app script."
        self.status_label.config(text=text)
        return status

    def check_engine(self):
        status = self.refresh_portable_status()
        lines = [
            f"Portable root: {status.root}",
            f"Unified script: {status.script}",
            f"NMRNet Python: {status.nmrnet_python}",
            f"CASCADE-2.0 Python: {status.cascade2_python}",
        ]
        if status.missing_required:
            lines.append("")
            lines.append("Missing required files:")
            lines.extend(f"- {item}" for item in status.missing_required)
        if status.missing_optional:
            lines.append("")
            lines.append("Missing optional files:")
            lines.extend(f"- {item}" for item in status.missing_optional)

        if hasattr(self, "log_text"):
            self.append_log("Engine check:")
            for line in lines:
                self.append_log(line)

        message = "\n".join(lines)
        if status.ready_for_nmrnet:
            title = "Engine check"
            messagebox.showinfo(title, message)
        else:
            title = "Engine incomplete"
            messagebox.showwarning(title, message)

    def select_input_file(self):
        path = filedialog.askopenfilename(
            title="Select SDF or CSV input",
            filetypes=[("SDF/CSV", "*.sdf *.sd *.csv"), ("SDF", "*.sdf *.sd"), ("CSV", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.input_path.set(path)
            try:
                self.input_type.set(detect_input_type(path))
            except Exception:
                pass
            if not self.output_dir.get():
                self.output_dir.set(str(default_output_dir(path)))

    def select_output_dir(self):
        path = filedialog.askdirectory(title="Select prediction output folder")
        if path:
            self.output_dir.set(path)

    def append_log(self, text):
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def _set_running(self, running):
        self._running = running
        self.run_button.config(state="disabled" if running else "normal")

    def _build_launch(self):
        if not self.input_path.get().strip():
            raise ValueError("Select an SDF or CSV input file.")
        if not self.output_dir.get().strip():
            self.output_dir.set(str(default_output_dir(self.input_path.get())))
        return build_nmr_prediction_launch(
            root=self.portable_root.get(),
            input_path=self.input_path.get(),
            output_dir=self.output_dir.get(),
            mode=self.mode.get(),
            input_type=self.input_type.get(),
            c_engine=self.c_engine.get(),
            max_conformers=int(self.max_conformers.get()),
            max_iters=int(self.max_iters.get()),
            forcefield=self.forcefield.get(),
            time_limit_seconds=float(self.time_limit_seconds.get()),
            coord_route=self.coord_route.get(),
            route_initial_confs=int(self.route_initial_confs.get()),
            route_prune_rms_thresh=float(self.route_prune_rms_thresh.get()),
            route_coarse_steps=int(self.route_coarse_steps.get()),
            route_keep_top_k=int(self.route_keep_top_k.get()),
            route_fine_steps=int(self.route_fine_steps.get()),
            optimize_existing=bool(self.optimize_existing.get()),
            allow_2d_if_h_nonzero=bool(self.allow_2d_if_h_nonzero.get()),
            smiles_column=self.smiles_column.get(),
            id_column=self.id_column.get(),
        )

    def run_prediction(self):
        if self._running:
            return
        try:
            launch = self._build_launch()
        except Exception as exc:
            messagebox.showerror("Prediction setup error", str(exc))
            self.refresh_portable_status()
            return

        self.log_text.delete("1.0", tk.END)
        self.append_log("Starting NMR prediction...")
        self.append_log(" ".join(f'"{part}"' if " " in part else part for part in launch.command))
        self._set_running(True)

        def worker():
            code = -1
            annotation_error = None
            tail_lines = []
            try:
                proc = subprocess.Popen(
                    launch.command,
                    cwd=str(launch.cwd),
                    env=launch.env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    clean_line = line.rstrip()
                    tail_lines.append(clean_line)
                    tail_lines[:] = tail_lines[-12:]
                    self.after(0, self.append_log, clean_line)
                code = proc.wait()
                if code == 0:
                    try:
                        result = annotate_prediction_sdf(launch.expected_final_sdf)
                        self.after(
                            0,
                            self.append_log,
                            f"Annotated final SDF with ID and FW for {result.molecule_count} molecule(s).",
                        )
                    except Exception as exc:
                        annotation_error = exc
                        self.after(0, self.append_log, f"ERROR: final SDF annotation failed: {exc}")
            except Exception as exc:
                self.after(0, self.append_log, f"ERROR: {exc}")
                self.after(0, messagebox.showerror, "Prediction error", str(exc))
            finally:
                def finish():
                    self._set_running(False)
                    if code == 0 and annotation_error is None:
                        self.append_log(f"Done. Final SDF: {launch.expected_final_sdf}")
                        messagebox.showinfo("Prediction complete", f"Output folder:\n{launch.output_dir}")
                    elif annotation_error is not None:
                        messagebox.showerror("Prediction post-processing failed", str(annotation_error))
                    elif code != -1:
                        self.append_log(f"Prediction failed with exit code {code}.")
                        detail = f"Exit code: {code}"
                        if tail_lines:
                            detail += "\n\nLast log lines:\n" + "\n".join(tail_lines[-8:])
                        messagebox.showerror("Prediction failed", detail)
                self.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def open_output_dir(self):
        path = self.output_dir.get().strip()
        if not path:
            messagebox.showwarning("Output folder", "No output folder selected.")
            return
        out = Path(path)
        if not out.exists():
            messagebox.showwarning("Output folder", "The output folder does not exist yet.")
            return
        os.startfile(str(out))


class NMRAssignmentFrame(ttk.Frame):
    """CORD-NMR predicted-structure-based NMR assignment workflow."""

    EXAMPLE_SMILES = r"CC([C@@H]1OC([C@@H]2[C@H](C2(C)C)/C=C(C(OC)=O)\C)=O)=C(C(C1)=O)C/C=C\C=C"

    STATUS_TAGS = {
        "ok": {"background": "#F4FFF6"},
        "yellow": {"background": "#FFF2B8"},
        "red": {"background": "#FFD4D4"},
        "missing": {"background": "#EEF2F4"},
    }

    def __init__(self, master):
        super().__init__(master)
        self.portable_root = tk.StringVar(value=str(self._detect_predictor_root()))
        self.assignment_mode = tk.StringVar(value="Single")
        self.smiles = tk.StringVar()
        self.sdf_path = tk.StringVar()
        self.batch_sdf_path = tk.StringVar()
        self.batch_c_dir = tk.StringVar()
        self.batch_hsqc_dir = tk.StringVar()
        self.batch_output_dir = tk.StringVar()
        self.molecule_index = tk.StringVar(value="1")
        self.prediction_source = tk.StringVar(value="NMRNet")
        self.shift_input_mode = tk.StringVar(value="C")
        self.use_hsqc = tk.BooleanVar(value=True)
        self.solvent = tk.StringVar(value="CD3OD")
        self.carbon_mhz = tk.StringVar(value="101")
        self.proton_mhz = tk.StringVar(value="401")
        self.carbon_prefix = tk.StringVar(value="C-")
        self.proton_prefix = tk.StringVar(value="H-")
        self.c_tolerance = tk.StringVar(value="1.5")
        self.h_tolerance = tk.StringVar(value="0.15")
        self.c_yellow = tk.StringVar(value="1.5")
        self.c_red = tk.StringVar(value="3.0")
        self.h_yellow = tk.StringVar(value="0.15")
        self.h_red = tk.StringVar(value="0.30")
        self.ambiguity_window = tk.StringVar(value="1.5")
        self.ambiguity_mean_error = tk.StringVar(value="1.0")
        self.local_window = tk.StringVar(value="5.0")
        self.equivalence_tolerance = tk.StringVar(value="0.03")
        self.display_mode = tk.StringVar(value="Label")
        self.result = None
        self.current_source_sdf_path = None
        self._auto_prediction_sdf_path = ""
        self.batch_items = []
        self.batch_index = 0
        self._active_shift_input_mode = "C"
        self._shift_text_cache = {"C": "", "H": ""}
        self.result_window = None
        self.smiles_text = None
        self.single_data_frame = None
        self.single_guide_frame = None
        self.batch_frame = None
        self.workspace_title_label = None
        self.batch_nav_label = None
        self.prev_button = None
        self.next_button = None
        self.structure_canvas = None
        self.table = None
        self._structure_photo = None
        self._hover_atom = None
        self._atom_positions = {}
        self._item_to_atom = {}
        self._item_to_hgroup = {}
        self._atom_to_item = {}
        self._drag_exp_item = None
        self._assignment_running = False
        self.create_widgets()

    def _detect_predictor_root(self):
        candidates = [DEFAULT_NMR_PREDICTOR_ROOT]
        for candidate in candidates:
            status = describe_nmr_predictor_root(candidate)
            if status.ready_for_nmrnet:
                return candidate
        return DEFAULT_NMR_PREDICTOR_ROOT

    def create_widgets(self):
        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

        intro = ttk.LabelFrame(outer, text="NMR Assignment")
        intro.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(
            intro,
            text=(
                "Assign experimental 13C/1H shifts to structure atom indices using predicted CH shifts. "
                "Rows are color-coded only: yellow = ambiguous/caution, red = large final error."
            ),
            style="Card.TLabel",
            wraplength=980,
        ).pack(anchor=tk.W, padx=10, pady=8)

        setup = ttk.LabelFrame(outer, text="Input and Parameters")
        setup.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(setup, text="Assignment mode").grid(row=0, column=0, sticky=tk.W, padx=8, pady=5)
        mode = ttk.Combobox(setup, textvariable=self.assignment_mode, values=["Single", "Batch"], width=12, state="readonly")
        mode.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        mode.bind("<<ComboboxSelected>>", lambda _e: self.refresh_assignment_mode())
        ttk.Label(setup, text="Single uses SMILES or precomputed SDF; Batch uses total SDF + folders.", style="Muted.TLabel").grid(row=0, column=2, columnspan=7, sticky=tk.W, padx=5, pady=5)

        ttk.Label(setup, text="SMILES").grid(row=1, column=0, sticky=tk.W, padx=8, pady=5)
        self.smiles_text = tk.Text(setup, height=2, width=96, wrap=tk.NONE, font=ChemTheme.FONTS["input"])
        self.smiles_text.grid(row=1, column=1, columnspan=6, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(setup, text="Validate", style="Secondary.TButton", command=self.validate_smiles).grid(row=1, column=7, sticky=tk.W, padx=5, pady=5)
        ttk.Button(setup, text="Example", style="Secondary.TButton", command=self.load_example_smiles).grid(row=1, column=8, sticky=tk.W, padx=5, pady=5)

        ttk.Label(setup, text="13C source").grid(row=2, column=0, sticky=tk.W, padx=8, pady=5)
        ttk.Combobox(setup, textvariable=self.prediction_source, values=["NMRNet", "CASCADE-2.0"], width=14, state="readonly").grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(setup, text="1H source: NMRNet only", style="Muted.TLabel").grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Label(setup, text="Solvent").grid(row=2, column=3, sticky=tk.E, padx=5, pady=5)
        ttk.Entry(setup, textvariable=self.solvent, width=12).grid(row=2, column=4, sticky=tk.W, padx=5, pady=5)
        ttk.Label(setup, text="13C MHz").grid(row=2, column=5, sticky=tk.E, padx=5, pady=5)
        ttk.Entry(setup, textvariable=self.carbon_mhz, width=8).grid(row=2, column=6, sticky=tk.W, padx=5, pady=5)
        ttk.Label(setup, text="1H MHz").grid(row=2, column=7, sticky=tk.E, padx=5, pady=5)
        ttk.Entry(setup, textvariable=self.proton_mhz, width=8).grid(row=2, column=8, sticky=tk.W, padx=5, pady=5)

        ttk.Label(setup, text="C prefix").grid(row=3, column=0, sticky=tk.W, padx=8, pady=5)
        ttk.Entry(setup, textvariable=self.carbon_prefix, width=8).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(setup, text="H prefix").grid(row=3, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(setup, textvariable=self.proton_prefix, width=8).grid(row=3, column=3, sticky=tk.W, padx=5, pady=5)
        ttk.Checkbutton(setup, text="Use HSQC", variable=self.use_hsqc).grid(row=3, column=4, columnspan=2, sticky=tk.W, padx=5, pady=5)
        ttk.Label(setup, text="Precomputed SDF").grid(row=4, column=0, sticky=tk.W, padx=8, pady=5)
        ttk.Entry(setup, textvariable=self.sdf_path, width=72).grid(row=4, column=1, columnspan=5, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(setup, text="Browse", command=self.select_sdf).grid(row=4, column=6, sticky=tk.W, padx=5, pady=5)
        ttk.Label(setup, text="Molecule").grid(row=4, column=7, sticky=tk.E, padx=5, pady=5)
        ttk.Entry(setup, textvariable=self.molecule_index, width=6).grid(row=4, column=8, sticky=tk.W, padx=5, pady=5)
        ttk.Label(setup, text="Predictor root").grid(row=5, column=0, sticky=tk.W, padx=8, pady=5)
        ttk.Entry(setup, textvariable=self.portable_root, width=72).grid(row=5, column=1, columnspan=5, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(setup, text="Browse", command=self.select_predictor_root).grid(row=5, column=6, sticky=tk.W, padx=5, pady=5)
        ttk.Button(setup, text="Check", style="Secondary.TButton", command=self.check_predictor_root).grid(row=5, column=7, sticky=tk.W, padx=5, pady=5)
        setup.columnconfigure(1, weight=1)

        self.batch_frame = ttk.LabelFrame(outer, text="Batch Inputs")
        ttk.Label(self.batch_frame, text="Total predicted SDF").grid(row=0, column=0, sticky=tk.W, padx=8, pady=5)
        ttk.Entry(self.batch_frame, textvariable=self.batch_sdf_path, width=80).grid(row=0, column=1, columnspan=4, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(self.batch_frame, text="Browse", command=self.select_batch_sdf).grid(row=0, column=5, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.batch_frame, text="C data folder").grid(row=1, column=0, sticky=tk.W, padx=8, pady=5)
        ttk.Entry(self.batch_frame, textvariable=self.batch_c_dir, width=80).grid(row=1, column=1, columnspan=4, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(self.batch_frame, text="Browse", command=lambda: self.select_folder(self.batch_c_dir, "Select C data folder")).grid(row=1, column=5, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.batch_frame, text="HSQC folder").grid(row=2, column=0, sticky=tk.W, padx=8, pady=5)
        ttk.Entry(self.batch_frame, textvariable=self.batch_hsqc_dir, width=80).grid(row=2, column=1, columnspan=4, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(self.batch_frame, text="Browse", command=lambda: self.select_folder(self.batch_hsqc_dir, "Select HSQC folder")).grid(row=2, column=5, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.batch_frame, text="Output folder").grid(row=3, column=0, sticky=tk.W, padx=8, pady=5)
        ttk.Entry(self.batch_frame, textvariable=self.batch_output_dir, width=80).grid(row=3, column=1, columnspan=4, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(self.batch_frame, text="Browse", command=lambda: self.select_folder(self.batch_output_dir, "Select batch output folder")).grid(row=3, column=5, sticky=tk.W, padx=5, pady=5)
        ttk.Label(
            self.batch_frame,
            text="Files are matched by SDF ID and filename stem. If exact filename matching fails, sorted file order is used.",
            style="Muted.TLabel",
        ).grid(row=4, column=1, columnspan=5, sticky=tk.W, padx=5, pady=(0, 6))
        self.batch_frame.columnconfigure(1, weight=1)

        thresholds = ttk.LabelFrame(outer, text="Color Thresholds")
        thresholds.pack(fill=tk.X, pady=(0, 8))
        items = [
            ("C tolerance", self.c_tolerance),
            ("H tolerance", self.h_tolerance),
            ("C yellow", self.c_yellow),
            ("C red", self.c_red),
            ("H yellow", self.h_yellow),
            ("H red", self.h_red),
            ("Ambiguity window", self.ambiguity_window),
            ("Ambiguity mean", self.ambiguity_mean_error),
            ("Local window", self.local_window),
            ("H equivalence", self.equivalence_tolerance),
        ]
        for idx, (label, var) in enumerate(items):
            row = idx // 5
            col = (idx % 5) * 2
            ttk.Label(thresholds, text=label).grid(row=row, column=col, sticky=tk.W, padx=(8, 2), pady=4)
            ttk.Entry(thresholds, textvariable=var, width=8).grid(row=row, column=col + 1, sticky=tk.W, padx=(0, 8), pady=4)

        self.single_data_frame = ttk.LabelFrame(outer, text="Experimental Data")
        self.single_data_frame.pack(fill=tk.X, pady=(0, 8))
        one_d = ttk.Frame(self.single_data_frame)
        one_d.grid(row=0, column=0, sticky=tk.NSEW, padx=6, pady=6)
        header = ttk.Frame(one_d)
        header.pack(fill=tk.X)
        ttk.Label(header, text="1D shift table", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text="Input type").pack(side=tk.LEFT, padx=(12, 4))
        mode_box = ttk.Combobox(header, textvariable=self.shift_input_mode, values=["C", "H"], width=5, state="readonly")
        mode_box.pack(side=tk.LEFT)
        mode_box.bind("<<ComboboxSelected>>", self.on_shift_input_mode_change)
        ttk.Label(header, text="C: carbon shifts; H: proton shifts", style="Muted.TLabel").pack(side=tk.LEFT, padx=(10, 0))
        self.shift_text = tk.Text(one_d, height=6, width=50, wrap=tk.WORD, font=ChemTheme.FONTS["input"])
        self.shift_text.pack(fill=tk.BOTH, expand=True)
        self.hsqc_text = self._text_panel(self.single_data_frame, "HSQC table: C/H shifts with optional intensity; negative intensity matches CH2 only", 1)
        self.single_data_frame.columnconfigure(0, weight=1)
        self.single_data_frame.columnconfigure(1, weight=1)

        action_bar = ttk.Frame(outer)
        action_bar.pack(fill=tk.X, pady=(0, 8))
        self.assign_button = ttk.Button(action_bar, text="Assign", command=self.run_assignment)
        self.assign_button.pack(side=tk.LEFT, padx=(0, 6), ipadx=14)
        self.status_label = ttk.Label(action_bar, text="", style="Card.TLabel")
        self.status_label.pack(side=tk.RIGHT, padx=8)

        self.single_guide_frame = ttk.LabelFrame(outer, text="Assignment Workspace")
        self.single_guide_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            self.single_guide_frame,
            text=(
                "Click Assign to open a separate workspace window. "
                "The workspace keeps the structure and table spacious, and label edits update the drawing and export text."
            ),
            style="Card.TLabel",
            wraplength=980,
        ).pack(anchor=tk.W, padx=10, pady=10)
        self.refresh_assignment_mode()

    def _text_panel(self, parent, title, col):
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=col, sticky=tk.NSEW, padx=6, pady=6)
        ttk.Label(frame, text=title, style="Card.TLabel").pack(anchor=tk.W)
        txt = tk.Text(frame, height=6, width=36, wrap=tk.WORD, font=ChemTheme.FONTS["input"])
        txt.pack(fill=tk.BOTH, expand=True)
        parent.columnconfigure(col, weight=1)
        return txt

    def open_workspace_window(self):
        if self.result_window is not None and self.result_window.winfo_exists():
            self.result_window.destroy()
        win = tk.Toplevel(self)
        self.result_window = win
        win.title("NMR Assignment Workspace")
        win.geometry("1120x700")
        win.minsize(960, 560)
        win.configure(bg=ChemTheme.COLORS["background"])
        win.protocol("WM_DELETE_WINDOW", self._close_workspace_window)

        top = ttk.Frame(win)
        top.pack(fill=tk.X, padx=10, pady=(10, 6))
        self.workspace_title_label = ttk.Label(top, text="NMR Assignment Workspace", font=("Helvetica", 14, "bold"))
        self.workspace_title_label.pack(side=tk.LEFT)
        if self.batch_items:
            self.prev_button = ttk.Button(top, text="Previous", style="Secondary.TButton", command=lambda: self.show_batch_item(self.batch_index - 1))
            self.prev_button.pack(side=tk.LEFT, padx=(14, 4))
            self.next_button = ttk.Button(top, text="Next", style="Secondary.TButton", command=lambda: self.show_batch_item(self.batch_index + 1))
            self.next_button.pack(side=tk.LEFT, padx=4)
            self.batch_nav_label = ttk.Label(top, text="", style="Card.TLabel")
            self.batch_nav_label.pack(side=tk.LEFT, padx=8)
            ttk.Button(top, text="Save Batch TXT", style="Secondary.TButton", command=self.save_batch_txt_dialog).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="Export SDF", style="Secondary.TButton", command=self.export_sdf_dialog).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="Export Text", style="Secondary.TButton", command=self.show_export_text).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top, text="Reset Labels", style="Secondary.TButton", command=self.reset_labels).pack(side=tk.RIGHT, padx=4)
        ttk.Label(top, text="Structure label display").pack(side=tk.RIGHT, padx=(14, 4))
        mode_box = ttk.Combobox(top, textvariable=self.display_mode, values=["Label", "Index", "Both", "None"], width=10, state="readonly")
        mode_box.pack(side=tk.RIGHT, padx=4)
        mode_box.bind("<<ComboboxSelected>>", lambda _e: self.render_structure())

        pane = ttk.Panedwindow(win, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        structure_frame = ttk.LabelFrame(pane, text="Structure")
        table_frame = ttk.LabelFrame(pane, text="Assignment Table")
        pane.add(structure_frame, weight=2)
        pane.add(table_frame, weight=3)

        self.structure_canvas = tk.Canvas(
            structure_frame,
            width=520,
            height=560,
            bg="white",
            highlightthickness=1,
            highlightbackground=ChemTheme.COLORS["border"],
        )
        self.structure_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.structure_canvas.bind("<Motion>", self.on_structure_motion)
        self.structure_canvas.bind("<Leave>", lambda _e: self.set_hover_atom(None))

        columns = ("atom", "label", "type", "pred", "exp", "delta")
        self.table = ttk.Treeview(table_frame, columns=columns, show="tree headings", height=22)
        self.table.heading("#0", text="")
        self.table.column("#0", width=24, stretch=False)
        headings = {
            "atom": "Atom",
            "label": "Label",
            "type": "Type",
            "pred": "Shift pred",
            "exp": "Shift exp",
            "delta": "Delta",
        }
        widths = {"atom": 95, "label": 80, "type": 70, "pred": 100, "exp": 100, "delta": 85}
        for col in columns:
            self.table.heading(col, text=headings[col])
            self.table.column(col, width=widths[col], anchor=tk.CENTER)
        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        xscroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.table.grid(row=0, column=0, sticky=tk.NSEW, padx=(8, 0), pady=(8, 0))
        yscroll.grid(row=0, column=1, sticky=tk.NS, padx=(0, 8), pady=(8, 0))
        xscroll.grid(row=1, column=0, sticky=tk.EW, padx=(8, 0), pady=(0, 8))
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        for tag, opts in self.STATUS_TAGS.items():
            self.table.tag_configure(tag, **opts)
        self.table.tag_configure("manual", foreground="#C2185B")
        self.table.bind("<Motion>", self.on_table_motion)
        self.table.bind("<Leave>", lambda _e: self.set_hover_atom(None))
        self.table.bind("<Double-1>", self.edit_table_cell)
        self.table.bind("<ButtonPress-1>", self.start_experimental_drag)
        self.table.bind("<ButtonRelease-1>", self.finish_experimental_drag)
        self.update_workspace_header()

    def _close_workspace_window(self):
        if self.result_window is not None and self.result_window.winfo_exists():
            self.result_window.destroy()
        self.result_window = None
        self.workspace_title_label = None
        self.batch_nav_label = None
        self.prev_button = None
        self.next_button = None
        self.structure_canvas = None
        self.table = None
        self._structure_photo = None
        self._hover_atom = None
        self._atom_positions = {}
        self._item_to_atom = {}
        self._item_to_hgroup = {}
        self._atom_to_item = {}
        self._drag_exp_item = None

    def _store_shift_text(self):
        if hasattr(self, "shift_text"):
            self._shift_text_cache[self._active_shift_input_mode] = self.shift_text.get("1.0", tk.END).strip()

    def on_shift_input_mode_change(self, _event=None):
        self._store_shift_text()
        new_mode = self.shift_input_mode.get() or "C"
        self._active_shift_input_mode = new_mode
        self.shift_text.delete("1.0", tk.END)
        cached = self._shift_text_cache.get(new_mode, "")
        if cached:
            self.shift_text.insert("1.0", cached)

    def select_sdf(self):
        path = filedialog.askopenfilename(
            title="Select predicted SDF",
            filetypes=[("SDF files", "*.sdf *.sd"), ("All files", "*.*")]
        )
        if path:
            self.sdf_path.set(path)
            self._auto_prediction_sdf_path = ""

    def refresh_assignment_mode(self):
        is_batch = self.assignment_mode.get() == "Batch"
        if self.batch_frame is not None:
            if is_batch:
                self.batch_frame.pack(fill=tk.X, pady=(0, 8))
            else:
                self.batch_frame.pack_forget()
        if self.single_data_frame is not None:
            if is_batch:
                self.single_data_frame.pack_forget()
            else:
                self.single_data_frame.pack(fill=tk.X, pady=(0, 8))
        if self.single_guide_frame is not None:
            if is_batch:
                self.single_guide_frame.pack_forget()
            else:
                self.single_guide_frame.pack(fill=tk.BOTH, expand=True)
        if hasattr(self, "assign_button"):
            self.assign_button.config(text="Run Batch" if is_batch else "Assign")

    def select_batch_sdf(self):
        path = filedialog.askopenfilename(
            title="Select total predicted SDF",
            filetypes=[("SDF files", "*.sdf *.sd"), ("All files", "*.*")]
        )
        if path:
            self.batch_sdf_path.set(path)

    def select_folder(self, var, title):
        path = filedialog.askdirectory(title=title)
        if path:
            var.set(path)

    def select_predictor_root(self):
        initial = str(Path(self.portable_root.get()).parent) if self.portable_root.get().strip() else str(Path.home())
        path = filedialog.askdirectory(title="Select NMR-Predictor-Portable root", initialdir=initial)
        if path:
            self.portable_root.set(path)
            self.check_predictor_root(show_success=True)

    def check_predictor_root(self, show_success=True):
        status = describe_nmr_predictor_root(self.portable_root.get())
        if status.ready_for_cascade2:
            text = "Predictor ready: NMRNet and CASCADE-2.0."
            self.status_label.config(text=text)
            if show_success:
                messagebox.showinfo("Predictor ready", text)
            return True
        if status.ready_for_nmrnet:
            text = "Predictor ready for NMRNet. CASCADE-2.0 runtime is missing."
            self.status_label.config(text=text)
            if show_success:
                messagebox.showwarning("Predictor partial", text)
            return True
        missing = "\n".join(status.missing_required + status.missing_optional)
        text = "Predictor root is incomplete."
        self.status_label.config(text=text)
        if show_success:
            messagebox.showerror("Predictor incomplete", f"{text}\n\nMissing:\n{missing}")
        return False

    def _get_smiles(self):
        if self.smiles_text is not None:
            raw = self.smiles_text.get("1.0", tk.END)
        else:
            raw = self.smiles.get()
        smiles = re.sub(r"\s+", "", raw or "")
        self.smiles.set(smiles)
        return smiles

    def load_example_smiles(self):
        if self.smiles_text is not None:
            self.smiles_text.delete("1.0", tk.END)
            self.smiles_text.insert("1.0", self.EXAMPLE_SMILES)
        self.smiles.set(self.EXAMPLE_SMILES)
        self.validate_smiles(show_success=True)

    def _smiles_error_message(self, smiles):
        return (
            "The SMILES string could not be parsed.\n\n"
            "For the pyrethrin-like test structure, paste exactly:\n"
            f"{self.EXAMPLE_SMILES}\n\n"
            "Common issue: the stereocenter must stay bracketed as [C@H](C2(C)C); "
            "C@H(C) without brackets/ring closure is not valid SMILES."
        )

    def validate_smiles(self, show_success=True):
        smiles = self._get_smiles()
        if not smiles:
            messagebox.showwarning("Missing SMILES", "Enter a SMILES string first.")
            return False
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            messagebox.showerror("Invalid SMILES", self._smiles_error_message(smiles))
            return False
        c_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        formula = ""
        try:
            formula = rdMolDescriptors.CalcMolFormula(Chem.AddHs(mol))
        except Exception:
            pass
        detail = f"Valid SMILES. C atoms: {c_count}" + (f"; formula: {formula}" if formula else "")
        self.status_label.config(text=detail)
        if show_success:
            messagebox.showinfo("Valid SMILES", detail)
        return True

    def _float(self, var, default):
        try:
            return float(var.get())
        except Exception:
            return float(default)

    def _int(self, var, default):
        try:
            return int(float(var.get()))
        except Exception:
            return int(default)

    def _is_auto_prediction_sdf(self, sdf_path):
        if not sdf_path or not self._auto_prediction_sdf_path:
            return False
        try:
            return Path(sdf_path).resolve() == Path(self._auto_prediction_sdf_path).resolve()
        except Exception:
            return str(sdf_path).strip() == str(self._auto_prediction_sdf_path).strip()

    def run_assignment(self):
        if self._assignment_running:
            return
        if self.assignment_mode.get() == "Batch":
            self.run_batch_assignment()
            return
        self._close_workspace_window()
        self.result = None
        self.batch_items = []
        self.batch_index = 0
        self._store_shift_text()
        carbon_text = self._shift_text_cache.get("C", "")
        proton_text = self._shift_text_cache.get("H", "")
        hsqc_text = self.hsqc_text.get("1.0", tk.END)
        sdf_path = self.sdf_path.get().strip()
        smiles = self._get_smiles()
        use_sdf = bool(sdf_path) and not (bool(smiles) and self._is_auto_prediction_sdf(sdf_path))
        if use_sdf:
            self.status_label.config(text="Assigning from selected SDF...")
            self._assign_from_sdf(sdf_path, self._int(self.molecule_index, 1), carbon_text, proton_text, hsqc_text)
            return
        if not smiles:
            messagebox.showwarning("Missing SMILES", "Enter a SMILES string first.")
            return
        if not self.validate_smiles(show_success=False):
            return
        self._predict_from_smiles_then_assign(smiles, carbon_text, proton_text, hsqc_text)

    def _assign_from_sdf(self, sdf_path, molecule_index, carbon_text, proton_text, hsqc_text):
        try:
            self.result = build_assignment(
                sdf_path=sdf_path,
                molecule_index=molecule_index,
                carbon_text=carbon_text,
                proton_text=proton_text,
                hsqc_text=hsqc_text,
                use_hsqc=bool(self.use_hsqc.get()),
                c_tolerance=self._float(self.c_tolerance, 1.5),
                h_tolerance=self._float(self.h_tolerance, 0.15),
                c_yellow_threshold=self._float(self.c_yellow, 1.5),
                c_red_threshold=self._float(self.c_red, 3.0),
                h_yellow_threshold=self._float(self.h_yellow, 0.15),
                h_red_threshold=self._float(self.h_red, 0.30),
                ambiguity_window=self._float(self.ambiguity_window, 1.5),
                ambiguity_mean_error=self._float(self.ambiguity_mean_error, 1.0),
                local_window=self._float(self.local_window, 5.0),
                equivalence_tolerance=self._float(self.equivalence_tolerance, 0.03),
            )
        except Exception as exc:
            messagebox.showerror("Assignment error", str(exc))
            return
        self.open_workspace_window()
        self.refresh_table()
        self.render_structure()
        self.current_source_sdf_path = sdf_path
        caution = sum(1 for c in self.result.carbons if c.status == "yellow")
        red = sum(1 for c in self.result.carbons if c.status == "red")
        self.status_label.config(text=f"{self.result.molecule_name}: {len(self.result.carbons)} C atoms, {caution} yellow, {red} red")
        if self.result.warnings:
            messagebox.showwarning("Assignment warnings", "\n".join(self.result.warnings))

    def _assignment_build_kwargs(self):
        return {
            "use_hsqc": bool(self.use_hsqc.get()),
            "c_tolerance": self._float(self.c_tolerance, 1.5),
            "h_tolerance": self._float(self.h_tolerance, 0.15),
            "c_yellow_threshold": self._float(self.c_yellow, 1.5),
            "c_red_threshold": self._float(self.c_red, 3.0),
            "h_yellow_threshold": self._float(self.h_yellow, 0.15),
            "h_red_threshold": self._float(self.h_red, 0.30),
            "ambiguity_window": self._float(self.ambiguity_window, 1.5),
            "ambiguity_mean_error": self._float(self.ambiguity_mean_error, 1.0),
            "local_window": self._float(self.local_window, 5.0),
            "equivalence_tolerance": self._float(self.equivalence_tolerance, 0.03),
        }

    def _safe_name(self, text):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
        return safe or "molecule"

    def _default_batch_output_dir(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(__file__).resolve().parent / "results" / "nmr_assignment_batch" / stamp

    def _collect_data_files(self, folder):
        root = Path(folder).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Folder does not exist: {root}")
        exts = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}
        return sorted([path for path in root.iterdir() if path.is_file() and path.suffix.lower() in exts])

    def _read_data_file(self, path):
        path = Path(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            try:
                import pandas as pd
            except Exception as exc:
                raise RuntimeError("Reading Excel files requires pandas/openpyxl.") from exc
            frame = pd.read_excel(path, header=None)
            lines = []
            for row in frame.itertuples(index=False):
                vals = ["" if value != value else str(value) for value in row]
                lines.append("\t".join(vals))
            return "\n".join(lines)
        return path.read_text(encoding="utf-8", errors="ignore")

    def _match_files_to_ids(self, ids, files):
        by_stem = {path.stem: path for path in files}
        out = {}
        missing = []
        for mol_id in ids:
            if mol_id in by_stem:
                out[mol_id] = by_stem[mol_id]
            else:
                missing.append(mol_id)
        if not missing:
            return out, []
        warnings = []
        if len(files) >= len(ids):
            warnings.append("Some filenames did not match SDF IDs exactly; sorted file order was used for missing IDs.")
            used = set(out.values())
            remaining_files = [path for path in files if path not in used]
            for mol_id, path in zip(missing, remaining_files):
                out[mol_id] = path
        else:
            warnings.append("Some SDF IDs have no corresponding data file and were skipped: " + ", ".join(missing))
        return out, warnings

    def run_batch_assignment(self):
        sdf_path = self.batch_sdf_path.get().strip()
        c_dir = self.batch_c_dir.get().strip()
        if not sdf_path or not c_dir:
            messagebox.showwarning("Batch input missing", "Select a total SDF and a C data folder.")
            return
        out_dir = Path(self.batch_output_dir.get().strip()).expanduser() if self.batch_output_dir.get().strip() else self._default_batch_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        self.batch_output_dir.set(str(out_dir))
        try:
            ids = list_sdf_molecule_ids(sdf_path)
            c_files = self._collect_data_files(c_dir)
            c_map, warnings = self._match_files_to_ids(ids, c_files)
            hsqc_map = {}
            if self.batch_hsqc_dir.get().strip():
                hsqc_files = self._collect_data_files(self.batch_hsqc_dir.get().strip())
                hsqc_map, hsqc_warnings = self._match_files_to_ids(ids, hsqc_files)
                warnings.extend(hsqc_warnings)
        except Exception as exc:
            messagebox.showerror("Batch setup error", str(exc))
            return

        items = []
        params = self._assignment_build_kwargs()
        errors = []
        for index, mol_id in enumerate(ids, start=1):
            c_file = c_map.get(mol_id)
            if c_file is None:
                continue
            try:
                c_text = self._read_data_file(c_file)
                hsqc_file = hsqc_map.get(mol_id)
                hsqc_text = self._read_data_file(hsqc_file) if hsqc_file else ""
                result = build_assignment(
                    sdf_path=sdf_path,
                    molecule_index=index,
                    molecule_id=mol_id,
                    carbon_text=c_text,
                    proton_text="",
                    hsqc_text=hsqc_text,
                    **{**params, "use_hsqc": bool(params["use_hsqc"] and hsqc_text.strip())},
                )
                item = {
                    "id": mol_id,
                    "index": index,
                    "result": result,
                    "c_file": str(c_file),
                    "hsqc_file": str(hsqc_file) if hsqc_file else "",
                    "output_dir": out_dir,
                }
                items.append(item)
                self._autosave_batch_item(item)
            except Exception as exc:
                errors.append(f"{mol_id}: {exc}")

        if not items:
            messagebox.showerror("Batch assignment failed", "No molecule was assigned.\n" + "\n".join(errors[:8]))
            return

        self.batch_items = items
        self.batch_index = 0
        self.result = items[0]["result"]
        self.current_source_sdf_path = sdf_path
        self.open_workspace_window()
        self.refresh_table()
        self.render_structure()
        self.update_workspace_header()
        self.write_batch_txt(out_dir / "batch_assignment_autosave.txt")
        self.status_label.config(text=f"Batch assigned {len(items)}/{len(ids)} molecule(s). Output: {out_dir}")
        notes = warnings + errors
        if notes:
            messagebox.showwarning("Batch notes", "\n".join(notes[:12]))

    def _set_assignment_running(self, running, text=None):
        self._assignment_running = bool(running)
        if hasattr(self, "assign_button"):
            self.assign_button.config(state="disabled" if running else "normal")
        if text is not None:
            self.status_label.config(text=text)

    def _prediction_output_root(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(__file__).resolve().parent / "results" / "nmr_assignment_predictions" / f"{stamp}_smiles_assignment"

    def _predict_from_smiles_then_assign(self, smiles, carbon_text, proton_text, hsqc_text):
        output_dir = self._prediction_output_root()
        output_dir.mkdir(parents=True, exist_ok=True)
        input_csv = output_dir / "assignment_input.csv"
        try:
            with open(input_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "smiles"])
                writer.writeheader()
                writer.writerow({"id": "assignment_target", "smiles": smiles})
            c_engine = "cascade2" if self.prediction_source.get() == "CASCADE-2.0" else "nmrnet"
            launch = build_nmr_prediction_launch(
                root=self.portable_root.get().strip() or DEFAULT_NMR_PREDICTOR_ROOT,
                input_path=input_csv,
                output_dir=output_dir,
                mode="CH",
                input_type="csv",
                c_engine=c_engine,
                max_conformers=9,
                max_iters=300,
                forcefield="auto",
                time_limit_seconds=20.0,
                coord_route="standard",
                route_initial_confs=27,
                route_prune_rms_thresh=0.5,
                route_coarse_steps=10,
                route_keep_top_k=9,
                route_fine_steps=300,
                smiles_column="smiles",
                id_column="id",
            )
        except Exception as exc:
            messagebox.showerror("Prediction setup error", str(exc))
            return

        self._set_assignment_running(True, "Predicting C/H shifts from SMILES...")

        def worker():
            code = -1
            tail_lines = []
            error = None
            try:
                proc = subprocess.Popen(
                    launch.command,
                    cwd=str(launch.cwd),
                    env=launch.env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    clean = line.rstrip()
                    if clean:
                        tail_lines.append(clean)
                        tail_lines[:] = tail_lines[-10:]
                code = proc.wait()
                if code == 0:
                    annotate_prediction_sdf(launch.expected_final_sdf)
            except Exception as exc:
                error = exc

            def finish():
                self._set_assignment_running(False)
                if error is not None:
                    messagebox.showerror("Prediction error", str(error))
                    self.status_label.config(text="Prediction failed.")
                    return
                if code != 0:
                    detail = f"Prediction failed with exit code {code}."
                    if tail_lines:
                        detail += "\n\nLast log lines:\n" + "\n".join(tail_lines)
                    messagebox.showerror("Prediction failed", detail)
                    self.status_label.config(text="Prediction failed.")
                    return
                self._auto_prediction_sdf_path = str(launch.expected_final_sdf)
                self.sdf_path.set(str(launch.expected_final_sdf))
                self.molecule_index.set("1")
                self.status_label.config(text="Prediction complete. Opening assignment workspace...")
                self._assign_from_sdf(str(launch.expected_final_sdf), 1, carbon_text, proton_text, hsqc_text)

            self.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def _fmt(self, value, ndigits=2):
        if value is None:
            return ""
        return f"{float(value):.{ndigits}f}"

    def _parse_shift_entry_values(self, text):
        values = []
        for match in re.finditer(r"[-+]?\d+(?:\.\d+)?", text or ""):
            try:
                values.append(float(match.group(0)))
            except ValueError:
                continue
        return values

    def _hydrogen_group_initial_text(self, group):
        if group is None or group.experimental_shift is None:
            return ""
        return self._fmt(group.experimental_shift, 3)

    def refresh_table(self):
        if self.table is None:
            return
        for item in self.table.get_children():
            self.table.delete(item)
        self._item_to_atom = {}
        self._item_to_hgroup = {}
        self._atom_to_item = {}
        if self.result is None:
            return
        for carbon in self.result.carbons:
            c_item = f"C:{carbon.atom_index}"
            tags = [carbon.status]
            if carbon.manually_edited:
                tags.append("manual")
            self.table.insert(
                "",
                tk.END,
                iid=c_item,
                text="",
                values=(
                    f"C{carbon.atom_index + 1}",
                    carbon.label,
                    carbon.carbon_type,
                    self._fmt(carbon.predicted_shift),
                    self._fmt(carbon.experimental_shift),
                    self._fmt(carbon.error),
                ),
                tags=tuple(tags),
            )
            self._item_to_atom[c_item] = carbon.atom_index
            self._atom_to_item[carbon.atom_index] = c_item
            for gi, group in enumerate(carbon.hydrogens):
                h_item = f"H:{carbon.atom_index}:{gi}"
                atom_text = "/".join(f"H{idx + 1}" for idx in group.atom_indices)
                label = f"{carbon.label}{group.suffix}"
                self.table.insert(
                    c_item,
                    tk.END,
                    iid=h_item,
                    text="",
                    values=(
                        atom_text,
                        label,
                        "H eq" if group.equivalent else "H",
                        self._fmt(group.predicted_shift),
                        self._fmt(group.experimental_shift),
                        self._fmt(group.error),
                    ),
                    tags=(group.status, "manual") if carbon.manually_edited else (group.status,),
                )
                self._item_to_atom[h_item] = carbon.atom_index
                self._item_to_hgroup[h_item] = (carbon.atom_index, gi)
            self.table.item(c_item, open=True)

    def _display_labels(self):
        if self.result is None or self.display_mode.get() == "None":
            return {}
        labels = {}
        for carbon in self.result.carbons:
            original = carbon.original_label or str(carbon.atom_index + 1)
            current = carbon.label or original
            if self.display_mode.get() == "Index":
                labels[carbon.atom_index] = original
            elif self.display_mode.get() == "Both":
                labels[carbon.atom_index] = f"{original}/{current}" if original != current else current
            else:
                labels[carbon.atom_index] = current
        return labels

    def render_structure(self):
        if self.structure_canvas is None:
            return
        self.structure_canvas.delete("all")
        self._atom_positions = {}
        width = max(520, self.structure_canvas.winfo_width())
        height = max(380, self.structure_canvas.winfo_height())
        if self.result is None:
            self.structure_canvas.create_text(width // 2, height // 2, text="Run assignment to draw structure.", fill=ChemTheme.COLORS["muted"])
            return
        try:
            draw_mol = prepare_display_molecule(self.result.mol, self._display_labels())
            try:
                display_mol = Chem.RemoveHs(draw_mol, sanitize=False)
            except TypeError:
                display_mol = Chem.RemoveHs(draw_mol)
            highlight_atoms = []
            highlight_colors = {}
            for carbon in self.result.carbons:
                if carbon.manually_edited:
                    highlight_atoms.append(carbon.atom_index)
                    highlight_colors[carbon.atom_index] = (0.95, 0.25, 0.62)
            if self._hover_atom is not None:
                highlight_atoms.append(self._hover_atom)
                highlight_colors[self._hover_atom] = (0.12, 0.38, 0.95)
            image = Draw.MolToImage(
                display_mol,
                size=(width, height),
                wedgeBonds=False,
                highlightAtoms=sorted(set(highlight_atoms)),
                highlightAtomColors=highlight_colors,
            )
            self._structure_photo = ImageTk.PhotoImage(image)
            self.structure_canvas.create_image(0, 0, anchor=tk.NW, image=self._structure_photo)
            self._estimate_atom_positions(display_mol, width, height)
        except Exception as exc:
            self.structure_canvas.create_text(width // 2, height // 2, text=f"Structure drawing failed: {exc}", fill="#A00000")

    def _estimate_atom_positions(self, mol, width, height):
        try:
            conf = mol.GetConformer()
        except Exception:
            return
        carbons = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
        if not carbons:
            return
        coords = []
        for idx in carbons:
            pos = conf.GetAtomPosition(idx)
            coords.append((idx, float(pos.x), float(pos.y)))
        xs = [x for _idx, x, _y in coords]
        ys = [y for _idx, _x, y in coords]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        xspan = max(xmax - xmin, 1e-6)
        yspan = max(ymax - ymin, 1e-6)
        pad = 34
        scale = min((width - 2 * pad) / xspan, (height - 2 * pad) / yspan)
        xoff = (width - xspan * scale) / 2
        yoff = (height - yspan * scale) / 2
        self._atom_positions = {
            idx: (xoff + (x - xmin) * scale, yoff + (ymax - y) * scale)
            for idx, x, y in coords
        }

    def set_hover_atom(self, atom_idx):
        if atom_idx == self._hover_atom:
            return
        self._hover_atom = atom_idx
        if self.table is not None and atom_idx is not None and atom_idx in self._atom_to_item:
            item = self._atom_to_item[atom_idx]
            self.table.selection_set(item)
            self.table.see(item)
        self.render_structure()

    def on_table_motion(self, event):
        if self.table is None:
            return
        item = self.table.identify_row(event.y)
        self.set_hover_atom(self._item_to_atom.get(item))

    def on_structure_motion(self, event):
        if not self._atom_positions:
            return
        nearest = None
        best = 9999.0
        for atom_idx, (x, y) in self._atom_positions.items():
            dist = ((event.x - x) ** 2 + (event.y - y) ** 2) ** 0.5
            if dist < best:
                best = dist
                nearest = atom_idx
        self.set_hover_atom(nearest if best <= 28 else None)

    def _carbon_by_item(self, item):
        if self.result is None or not item.startswith("C:"):
            return None
        atom_idx = int(item.split(":", 1)[1])
        for carbon in self.result.carbons:
            if carbon.atom_index == atom_idx:
                return carbon
        return None

    def _hydrogen_group_by_item(self, item):
        if self.result is None:
            return None, None
        key = self._item_to_hgroup.get(item)
        if key is None:
            return None, None
        atom_idx, group_idx = key
        for carbon in self.result.carbons:
            if carbon.atom_index == atom_idx and 0 <= group_idx < len(carbon.hydrogens):
                return carbon, carbon.hydrogens[group_idx]
        return None, None

    def _status_for_manual_error(self, error, yellow, red):
        if error is None:
            return "missing"
        if error > red:
            return "red"
        if error > yellow:
            return "yellow"
        return "ok"

    def _carbon_status_after_manual_edit(self, carbon):
        predicted = carbon.predicted_shift
        experimental = carbon.experimental_shift
        error = carbon.error
        if predicted is None or experimental is None or error is None:
            return "missing"
        c_yellow = self._float(self.c_yellow, 1.5)
        c_red = self._float(self.c_red, 3.0)
        if error > c_red:
            return "red"

        ambiguity_window = self._float(self.ambiguity_window, 1.5)
        ambiguity_mean_error = self._float(self.ambiguity_mean_error, 1.0)
        local_window = self._float(self.local_window, 5.0)
        all_predicted = [c.predicted_shift for c in self.result.carbons] if self.result is not None else []
        all_experimental = [c.experimental_shift for c in self.result.carbons if c.experimental_shift is not None] if self.result is not None else []

        exp_near = sorted(abs(float(v) - float(predicted)) for v in all_experimental if abs(float(v) - float(predicted)) <= ambiguity_window)
        if len(exp_near) >= 2 and (sum(exp_near[:2]) / 2.0) > ambiguity_mean_error:
            return "yellow"

        pred_in_local = [
            v for v in all_predicted
            if v is not None and abs(float(v) - float(experimental)) <= local_window
        ]
        exp_in_local = [v for v in all_experimental if abs(float(v) - float(predicted)) <= local_window]
        if len(pred_in_local) <= 1 and len(exp_in_local) <= 1 and error > c_yellow:
            return "yellow"
        return self._status_for_manual_error(error, c_yellow, c_red)

    def _refresh_all_carbon_statuses(self):
        if self.result is None:
            return
        for carbon in self.result.carbons:
            carbon.status = self._carbon_status_after_manual_edit(carbon)

    def _apply_experimental_shift(self, item, shift_value):
        carbon = self._carbon_by_item(item)
        if carbon is not None:
            carbon.experimental_shift = shift_value
            carbon.error = abs(float(carbon.predicted_shift) - shift_value) if carbon.predicted_shift is not None and shift_value is not None else None
            self._refresh_all_carbon_statuses()
            return True

        _carbon, group = self._hydrogen_group_by_item(item)
        if group is not None:
            group.experimental_shift = shift_value
            group.error = abs(float(group.predicted_shift) - shift_value) if group.predicted_shift is not None and shift_value is not None else None
            group.status = self._status_for_manual_error(
                group.error,
                self._float(self.h_yellow, 0.15),
                self._float(self.h_red, 0.30),
            )
            return True
        return False

    def _split_hydrogen_group_with_shifts(self, item, shift_values):
        carbon, group = self._hydrogen_group_by_item(item)
        if carbon is None or group is None:
            return False
        key = self._item_to_hgroup.get(item)
        if key is None:
            return False
        _atom_idx, group_idx = key
        atom_indices = list(group.atom_indices)
        if len(shift_values) > len(atom_indices):
            messagebox.showwarning(
                "Too many shifts",
                f"{self.table.set(item, 'atom')} contains {len(atom_indices)} H atom(s); enter no more than {len(atom_indices)} shift value(s).",
            )
            return False
        if len(atom_indices) <= 1:
            messagebox.showwarning("Cannot split", "This row contains only one H atom. Edit it with a single shift value.")
            return False

        new_groups = []
        for idx, atom_index in enumerate(atom_indices):
            exp_shift = shift_values[idx] if idx < len(shift_values) else None
            error = (
                abs(float(group.predicted_shift) - float(exp_shift))
                if group.predicted_shift is not None and exp_shift is not None
                else None
            )
            suffix = chr(ord("a") + idx)
            new_groups.append(
                HydrogenAssignment(
                    atom_indices=[atom_index],
                    predicted_shift=group.predicted_shift,
                    experimental_shift=exp_shift,
                    error=error,
                    equivalent=False,
                    suffix=suffix,
                    status=self._status_for_manual_error(
                        error,
                        self._float(self.h_yellow, 0.15),
                        self._float(self.h_red, 0.30),
                    ),
                )
            )
        carbon.hydrogens[group_idx:group_idx + 1] = new_groups
        return True

    def _merge_carbon_hydrogens_with_shift(self, item, shift_value):
        carbon, group = self._hydrogen_group_by_item(item)
        if carbon is None or group is None:
            return False
        if len(carbon.hydrogens) <= 1:
            return self._apply_experimental_shift(item, shift_value)

        atom_indices = []
        predicted_values = []
        for h_group in carbon.hydrogens:
            atom_indices.extend(h_group.atom_indices)
            if h_group.predicted_shift is not None:
                predicted_values.append(float(h_group.predicted_shift))
        if len(atom_indices) <= 1:
            return self._apply_experimental_shift(item, shift_value)

        predicted = sum(predicted_values) / len(predicted_values) if predicted_values else None
        error = (
            abs(float(predicted) - float(shift_value))
            if predicted is not None and shift_value is not None
            else None
        )
        carbon.hydrogens = [
            HydrogenAssignment(
                atom_indices=atom_indices,
                predicted_shift=predicted,
                experimental_shift=shift_value,
                error=error,
                equivalent=True,
                suffix="",
                status=self._status_for_manual_error(
                    error,
                    self._float(self.h_yellow, 0.15),
                    self._float(self.h_red, 0.30),
                ),
            )
        ]
        return True

    def _edit_label_for_carbon(self, carbon):
        value = simpledialog.askstring(
            "Edit C label",
            "Enter numeric label for this carbon:",
            initialvalue=carbon.label,
            parent=self.result_window if self.result_window is not None else self,
        )
        if value is None:
            return
        value = value.strip()
        if not re.fullmatch(r"\d+", value):
            messagebox.showwarning("Invalid label", "Use numbers only, for example 1, 2, 22.")
            return
        carbon.label = value
        carbon.manually_edited = value != carbon.original_label
        self.refresh_table()
        self.render_structure()
        self._autosave_current_batch_item()

    def _edit_experimental_shift(self, item):
        carbon = self._carbon_by_item(item)
        group = None
        if carbon is None:
            carbon, group = self._hydrogen_group_by_item(item)
        if carbon is None and group is None:
            return

        current = group.experimental_shift if group is not None else carbon.experimental_shift
        atom_label = self.table.set(item, "atom") if self.table is not None else ""
        value = simpledialog.askstring(
            "Edit experimental shift",
            f"Enter experimental shift for {atom_label}. Leave blank to clear:",
            initialvalue=self._hydrogen_group_initial_text(group) if group is not None else ("" if current is None else self._fmt(current, 3)),
            parent=self.result_window if self.result_window is not None else self,
        )
        if value is None:
            return
        value = value.strip()
        if not value:
            shift_value = None
            if not self._apply_experimental_shift(item, shift_value):
                return
        else:
            shift_values = self._parse_shift_entry_values(value)
            if not shift_values:
                messagebox.showwarning("Invalid shift", "Enter numeric chemical shift values, or leave it blank to clear.")
                return
            if group is not None and len(shift_values) > 1:
                if not self._split_hydrogen_group_with_shifts(item, shift_values):
                    return
            elif group is not None and len(shift_values) == 1:
                if not self._merge_carbon_hydrogens_with_shift(item, shift_values[0]):
                    return
            else:
                if len(shift_values) > 1:
                    messagebox.showwarning("Invalid shift", "Carbon rows accept only one experimental shift value.")
                    return
                if not self._apply_experimental_shift(item, shift_values[0]):
                    return
        self.refresh_table()
        self.render_structure()
        self._autosave_current_batch_item()

    def _carbon_item_for_table_item(self, item):
        if self.table is None or not item:
            return ""
        if item.startswith("C:"):
            return item
        parent = self.table.parent(item)
        return parent if parent and parent.startswith("C:") else ""

    def _capture_experimental_packet(self, carbon):
        return {
            "carbon_shift": carbon.experimental_shift,
            "hydrogen_shifts": [group.experimental_shift for group in carbon.hydrogens],
        }

    def _install_experimental_packet(self, carbon, packet):
        carbon.experimental_shift = packet.get("carbon_shift")
        carbon.error = (
            abs(float(carbon.predicted_shift) - float(carbon.experimental_shift))
            if carbon.predicted_shift is not None and carbon.experimental_shift is not None
            else None
        )
        incoming_h = packet.get("hydrogen_shifts") or []
        for idx, group in enumerate(carbon.hydrogens):
            group.experimental_shift = incoming_h[idx] if idx < len(incoming_h) else None
            group.error = (
                abs(float(group.predicted_shift) - float(group.experimental_shift))
                if group.predicted_shift is not None and group.experimental_shift is not None
                else None
            )
            group.status = self._status_for_manual_error(
                group.error,
                self._float(self.h_yellow, 0.15),
                self._float(self.h_red, 0.30),
            )

    def _swap_experimental_packets(self, source_item, target_item):
        source_carbon = self._carbon_by_item(source_item)
        target_carbon = self._carbon_by_item(target_item)
        if source_carbon is None or target_carbon is None or source_carbon is target_carbon:
            return False
        source_packet = self._capture_experimental_packet(source_carbon)
        target_packet = self._capture_experimental_packet(target_carbon)
        self._install_experimental_packet(source_carbon, target_packet)
        self._install_experimental_packet(target_carbon, source_packet)
        self._refresh_all_carbon_statuses()
        self.refresh_table()
        self.render_structure()
        self._autosave_current_batch_item()
        return True

    def start_experimental_drag(self, event):
        self._drag_exp_item = None
        if self.table is None:
            return
        item = self.table.identify_row(event.y)
        column = self.table.identify_column(event.x)
        if column != "#5":
            return
        c_item = self._carbon_item_for_table_item(item)
        if c_item and self._carbon_by_item(c_item) is not None:
            self._drag_exp_item = c_item

    def finish_experimental_drag(self, event):
        if self.table is None or not self._drag_exp_item:
            return
        source_item = self._drag_exp_item
        self._drag_exp_item = None
        target_item = self._carbon_item_for_table_item(self.table.identify_row(event.y))
        if not target_item or target_item == source_item:
            return
        if self._swap_experimental_packets(source_item, target_item) and self.status_label is not None:
            source_label = self.table.set(source_item, "atom") if self.table.exists(source_item) else source_item
            target_label = self.table.set(target_item, "atom") if self.table.exists(target_item) else target_item
            self.status_label.config(text=f"Swapped experimental C/H assignments: {source_label} <-> {target_label}")

    def edit_table_cell(self, event):
        if self.table is None:
            return
        item = self.table.identify_row(event.y)
        column = self.table.identify_column(event.x)
        if not item:
            return
        if column == "#2":
            carbon = self._carbon_by_item(item)
            if carbon is not None:
                self._edit_label_for_carbon(carbon)
            return
        if column == "#5":
            self._edit_experimental_shift(item)

    def reset_labels(self):
        if self.result is None:
            return
        for carbon in self.result.carbons:
            carbon.label = carbon.original_label
            carbon.manually_edited = False
        self.refresh_table()
        self.render_structure()
        self._autosave_current_batch_item()

    def update_workspace_header(self):
        if self.result is None:
            return
        if self.workspace_title_label is not None:
            title = f"NMR Assignment Workspace - {self.result.molecule_name}"
            self.workspace_title_label.config(text=title)
        if self.batch_items and self.batch_nav_label is not None:
            self.batch_nav_label.config(text=f"{self.batch_index + 1}/{len(self.batch_items)} | ID {self.batch_items[self.batch_index]['id']}")
        if self.prev_button is not None:
            self.prev_button.config(state="normal" if self.batch_index > 0 else "disabled")
        if self.next_button is not None:
            self.next_button.config(state="normal" if self.batch_index < len(self.batch_items) - 1 else "disabled")

    def show_batch_item(self, index):
        if not self.batch_items:
            return
        if index < 0 or index >= len(self.batch_items):
            return
        self._autosave_current_batch_item()
        self.batch_index = index
        self.result = self.batch_items[index]["result"]
        self.refresh_table()
        self.render_structure()
        self.update_workspace_header()

    def _assignment_text_for_result(self, result):
        return export_assignment_text(
            result,
            carbon_mhz=self.carbon_mhz.get().strip() or "101",
            proton_mhz=self.proton_mhz.get().strip() or "401",
            solvent=self.solvent.get().strip() or "CD3OD",
            carbon_prefix=self.carbon_prefix.get().strip() or "C-",
            proton_prefix=self.proton_prefix.get().strip() or "H-",
        )

    def _autosave_batch_item(self, item):
        out_dir = Path(item["output_dir"]) / "per_molecule"
        out_dir.mkdir(parents=True, exist_ok=True)
        mol_id = item["id"]
        path = out_dir / f"{self._safe_name(mol_id)}.txt"
        text = self._assignment_text_for_result(item["result"])
        path.write_text(f"{mol_id}\n{text}\n", encoding="utf-8")
        item["text_path"] = str(path)

    def _autosave_current_batch_item(self):
        if not self.batch_items:
            return
        self._autosave_batch_item(self.batch_items[self.batch_index])

    def write_batch_txt(self, path):
        blocks = []
        for item in self.batch_items:
            blocks.append(f"{item['id']}\n{self._assignment_text_for_result(item['result'])}")
        Path(path).write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
        return Path(path)

    def save_batch_txt_dialog(self):
        if not self.batch_items:
            return
        self._autosave_current_batch_item()
        default_dir = Path(self.batch_output_dir.get().strip()).expanduser() if self.batch_output_dir.get().strip() else self._default_batch_output_dir()
        default_dir.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Save batch assignment text",
            initialdir=str(default_dir),
            initialfile="batch_assignment.txt",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        saved = self.write_batch_txt(path)
        messagebox.showinfo("Saved", f"Batch assignment text saved:\n{saved}")

    def export_sdf_dialog(self):
        if self.result is None:
            messagebox.showwarning("Export unavailable", "Run assignment first.")
            return
        source_sdf = self.current_source_sdf_path or self.sdf_path.get().strip() or self.batch_sdf_path.get().strip()
        if not source_sdf:
            messagebox.showwarning("Export unavailable", "No source SDF is available.")
            return
        results = [item["result"] for item in self.batch_items] if self.batch_items else [self.result]
        initial_dir = Path(self.batch_output_dir.get().strip()).expanduser() if self.batch_items and self.batch_output_dir.get().strip() else Path(source_sdf).parent
        initial_dir.mkdir(parents=True, exist_ok=True)
        initialfile = "batch_assigned_experimental.sdf" if self.batch_items else "assigned_experimental.sdf"
        path = filedialog.asksaveasfilename(
            title="Export assigned SDF",
            initialdir=str(initial_dir),
            initialfile=initialfile,
            defaultextension=".sdf",
            filetypes=[("SDF", "*.sdf"), ("All files", "*.*")]
        )
        if not path:
            return
        self._autosave_current_batch_item()
        saved = write_assigned_sdf(source_sdf_path=source_sdf, results=results, output_sdf_path=path)
        messagebox.showinfo("Saved", f"Assigned SDF saved:\n{saved}")

    def _assignment_text(self):
        if self.result is None:
            raise ValueError("Run assignment before exporting.")
        return self._assignment_text_for_result(self.result)

    def show_export_text(self):
        try:
            text = self._assignment_text()
        except Exception as exc:
            messagebox.showwarning("Export unavailable", str(exc))
            return
        win = tk.Toplevel(self)
        win.title("Exported NMR Assignment Text")
        win.geometry("860x360")
        win.configure(bg=ChemTheme.COLORS["background"])
        txt = tk.Text(win, wrap=tk.WORD, font=ChemTheme.FONTS["input"])
        txt.insert("1.0", text)
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 6))
        bar = ttk.Frame(win)
        bar.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(bar, text="Copy", command=lambda: self.copy_export_text(txt.get("1.0", tk.END).strip())).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Save", command=lambda: self.save_export_text(txt.get("1.0", tk.END).strip())).pack(side=tk.LEFT, padx=4)

    def copy_export_text(self, text):
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo("Copied", "Assignment text copied to clipboard.")

    def save_export_text(self, text):
        path = filedialog.asksaveasfilename(
            title="Save assignment text",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.strip() + "\n")
        messagebox.showinfo("Saved", f"Assignment text saved:\n{path}")


class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title(f"{APP_DISPLAY_NAME} - Natural product analysis with C/HSQC clustering and NMR prediction")
        self.geometry("1120x780")
        self.minsize(980, 680)
        self.configure(bg=ChemTheme.COLORS['background'])
        ChemTheme.configure_style()
        self.create_ui()

    def create_ui(self):

        header = ttk.Frame(self, style='Header.TFrame')
        ttk.Label(header, text=APP_DISPLAY_NAME, style='Header.TLabel', font=ChemTheme.FONTS['title']).pack(pady=(12, 2))
        ttk.Label(header, text="Analysis of natural product mixtures based on C spectrum and HSQC spectrum",
                  style='Subheader.TLabel', font=ChemTheme.FONTS['subtitle']).pack(pady=(0, 12))
        header.pack(fill=tk.X)

        self.mode_frames = {}
        nav_frame = ttk.Frame(self, style='Nav.TFrame')
        modes = [
            ('Joint matching mode', CombinedModeFrame),
            ('C matching mode', CarbonOnlyModeFrame),
            ('CH matching mode', CHMatchModeFrame),
            ('Clustering mode', ClusterModeFrame),
            ('NMR prediction mode', NMRPredictionFrame),
            ('NMR assignment mode', NMRAssignmentFrame),
        ]

        for text, frame_class in modes:
            btn = ttk.Button(nav_frame, text=text,
                             command=lambda fc=frame_class: self.show_mode(fc))
            btn.pack(side=tk.LEFT, padx=6, pady=10, ipadx=12, ipady=4)
            self.mode_frames[text] = frame_class(self)

        nav_frame.pack(fill=tk.X)
        self.show_mode(ClusterModeFrame)


    def fill_c_untyped_from_cluster(self, cluster_text):
        """Open C matching mode and fill an editable cluster as C_untyped data."""
        self.show_mode(CarbonOnlyModeFrame)
        frame = self.current_frame
        frame.c_mode.set('untyped')
        frame.toggle_c_params()
        frame.global_entry.delete("1.0", tk.END)
        frame.global_entry.insert("1.0", cluster_text.strip())

    def fill_hsqc_alltype_from_cluster(self, cluster_text):
        """Open HSQC matching mode and fill an editable cluster as All_type points."""
        self.show_mode(CHMatchModeFrame)
        frame = self.current_frame
        frame.hmqc_mode.current(0)  # pattern 1 = All_type
        frame.create_input_groups()
        frame.point_groups['All_type'].delete("1.0", tk.END)
        frame.point_groups['All_type'].insert("1.0", cluster_text.strip())
        cluster_frame = self.mode_frames.get('Clustering mode')
        c_tol = "1.0"
        h_tol = "0.1"
        if cluster_frame is not None:
            try:
                c_tol = cluster_frame.hsqc_c_tol.get() or c_tol
                h_tol = cluster_frame.hsqc_h_tol.get() or h_tol
            except Exception:
                pass
        frame.c_tol.delete(0, tk.END)
        frame.c_tol.insert(0, c_tol)
        frame.h_tol.delete(0, tk.END)
        frame.h_tol.insert(0, h_tol)

    def show_mode(self, frame_class):
        if hasattr(self, 'current_frame'):
            self.current_frame.pack_forget()

        for name, frame in self.mode_frames.items():
            if isinstance(frame, frame_class):
                self.current_frame = frame
                self.current_frame.pack(fill=tk.BOTH, expand=True)
                break

if __name__ == "__main__":
    app = App()
    app.mainloop()
