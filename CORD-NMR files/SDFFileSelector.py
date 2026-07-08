import os
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from typing import List, Union
import re


class SDFFileSelector:
    """
    处理SDF文件选择的类，支持：
    - GUI文件选择
    - 目录扫描
    - 手动路径设置
    """

    def __init__(self):
        self.selected_files = []

    def select_files_via_gui(self) -> List[str]:
        """通过GUI选择多个文件"""
        root = tk.Tk()
        root.withdraw()
        files = filedialog.askopenfilenames(
            title="选择SDF文件",
            filetypes=[("SDF文件", "*.sdf"), ("所有文件", "*.*")]
        )
        self.selected_files = list(files)
        return self._validate_files()

    def select_directory_via_gui(self) -> List[str]:
        """通过GUI选择目录并扫描SDF文件"""
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="选择包含SDF文件的目录")
        self.selected_files = self._find_sdf_in_dir(folder) if folder else []
        return self._validate_files()

    def set_files_manually(self, paths: Union[str, List[str]]) -> List[str]:
        """手动设置文件路径"""
        self.selected_files = [str(Path(p).resolve()) for p in ([paths] if isinstance(paths, str) else paths)]
        return self._validate_files()

    def _find_sdf_in_dir(self, directory: str) -> List[str]:
        """扫描目录中的SDF文件"""
        return [os.path.join(directory, f) for f in os.listdir(directory)
                if f.lower().endswith(('.sdf', '.sd'))]

    def _validate_files(self) -> List[str]:
        """验证文件有效性"""
        valid_files = []
        for f in self.selected_files:
            if not os.path.isfile(f):
                print(f"无效文件: {f}")
                continue
            if not f.lower().endswith(('.sdf', '.sd')):
                print(f"忽略非SDF文件: {f}")
                continue
            valid_files.append(f)
        return valid_files
