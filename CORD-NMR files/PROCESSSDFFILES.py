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
            title="Select the SDF file",
            filetypes=[("SDF file", "*.sdf"), ("All files", "*.*")]
        )
        self.selected_files = list(files)
        return self._validate_files()

    def select_directory_via_gui(self) -> List[str]:
        """通过GUI选择目录并扫描SDF文件"""
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="Select the directory that contains the SDF file")
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
                print(f"Invalid file: {f}")
                continue
            if not f.lower().endswith(('.sdf', '.sd')):
                print(f"Ignore non-SDF files: {f}")
                continue
            valid_files.append(f)
        return valid_files

import sqlite3
import json
from rdkit import Chem, Geometry
from tqdm import tqdm
from typing import List

class SDFBatchProcessor:
    """
    处理SDF文件的核心类，功能包括：
    - 解析HydrogenShifts和CarbonShifts
    - 数据批量入库
    - 错误隔离处理
    """

    def __init__(self, output_db: str = "chem_data.db"):
        self.output_db = output_db
        self._init_database()

    def _init_database(self):
        """初始化数据库结构"""
        with sqlite3.connect(self.output_db) as conn:
            conn.execute("DROP TABLE IF EXISTS molecules")
            conn.execute("DROP TABLE IF EXISTS carbons")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS molecules (
                    mol_id INTEGER PRIMARY KEY,
                    source_file TEXT,
                    name TEXT,
                    num_atoms INTEGER,
                    num_bonds INTEGER,
                    carbon_count INTEGER, 
                    fw REAL,
                    mol_block TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS carbons (
                    carbon_id INTEGER PRIMARY KEY,
                    mol_id INTEGER,
                    c_index INTEGER,
                    h_count INTEGER,
                    h_indices TEXT,
                    h_shifts TEXT,
                    c_shift REAL,
                    connected_carbon TEXT,
                    FOREIGN KEY(mol_id) REFERENCES molecules(mol_id)
                )
            """)
            conn.commit()

    def process_files(self, file_paths: List[str]):
        """处理文件列表"""
        with sqlite3.connect(self.output_db) as conn:
            conn.execute("PRAGMA synchronous = NORMAL")
            total_mols = 0

            with tqdm(total=len(file_paths), desc="Processing progress") as pbar:
                for sdf_path in file_paths:
                    try:
                        suppl = Chem.SDMolSupplier(sdf_path)
                        for mol in suppl:
                            if not mol:
                                continue
                            total_mols += 1
                            self._process_molecule(mol, conn, sdf_path)
                            pbar.update(1)
                            pbar.set_postfix_str(os.path.basename(sdf_path))
                    except Exception as e:
                        print(f"File {sdf_path} processing failed: {str(e)}")

            print(f"Processing is complete! Processed {total_mols} molecules")

    def _process_molecule(self, mol, conn, source_file):
        """处理单个分子"""
        # 检查分子是否已包含氢原子
        has_hydrogen = any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())

        # 只在没有氢原子时加氢
        if not has_hydrogen:
            mol = Chem.AddHs(mol)
            pass


        carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)

        # 获取分子量
        try:
            fw = float(mol.GetProp('FW'))
        except KeyError:
            print(f"Molecule lacks FW property: {mol. GetProp('_Name')}")
            return
            # 解析位移数据
        h_shifts = self._parse_shift_data(mol, "HydrogenShifts")
        c_shifts = self._parse_shift_data(mol, "Predicted 13C shifts")
        mol_block = Chem.MolToMolBlock(mol)
        # 插入分子记录
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO molecules 
            (source_file, name, num_atoms, num_bonds, carbon_count, fw, mol_block)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            source_file,
            mol.GetProp("ID") if mol.HasProp("ID") else "",
            mol.GetNumAtoms(),
            mol.GetNumBonds(),
            carbon_count,
            fw,
            mol_block
        ))
        mol_id = cursor.lastrowid

        # 处理碳原子
        carbons = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                h_info = self._get_ch_info(atom, h_shifts)
                carbons.append((
                    mol_id,
                    atom.GetIdx(),
                    h_info['count'],
                    json.dumps(h_info['indices']),
                    json.dumps(h_info['shifts']),
                    c_shifts.get(atom.GetIdx()),
                    json.dumps(h_info['connected_c_idx'])
                ))

        # 批量插入碳原子数据
        cursor.executemany("""
            INSERT INTO carbons
            (mol_id, c_index, h_count, h_indices, h_shifts, c_shift,connected_carbon)
            VALUES (?, ?, ?, ?, ?, ?,?)
        """, carbons)
        conn.commit()

    def _parse_shift_data(self, mol, tag: str) -> dict:
        """统一解析含中括号的位移数据"""
        shift_map = {}
        if mol.HasProp(tag):
            for line in mol.GetProp(tag).split('\n'):
                line = line.strip()
                if not line:
                    continue

                # 匹配格式：任意数字[实际索引+1] 位移值
                match = re.match(r'(\d+)([a-zA-Z]+)?\[(\d+)\]\s+([-+]?\d*\.?\d+)', line)
                if match:
                    try:
                        atom_idx = int(match.group(3)) - 1  # 转换为实际索引
                        shift = float(match.group(4))
                        shift_map[atom_idx] = shift
                    except (ValueError, IndexError) as e:
                        pass
                        print(f"Invalid line: {line} ({str(e)})")
                else:
                    print(f"Format error: {line}")
        return shift_map

    def _get_ch_info(self, carbon_atom, h_shift_map: dict) -> dict:
        """获取连接的氢原子信息"""
        h_indices = []
        h_shifts = []
        c_indices = []
        for neighbor in carbon_atom.GetNeighbors():
            if neighbor.GetSymbol() == 'H':
                h_idx = neighbor.GetIdx()
                h_indices.append(h_idx)
                h_shifts.append(h_shift_map.get(h_idx))
            if neighbor.GetSymbol() == 'C':
                idx = neighbor.GetIdx()
                c_indices.append(idx)
        return {
            'count': len(h_indices),
            'indices': h_indices,
            'shifts': h_shifts,
            'connected_c_idx': c_indices
        }


