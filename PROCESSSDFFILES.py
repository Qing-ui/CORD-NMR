import atexit
import os
import re
import tkinter as tk
import uuid
from pathlib import Path
from tkinter import filedialog
from typing import List, Union


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
        try:
            files = filedialog.askopenfilenames(
                title="Select the SDF file",
                filetypes=[("SDF file", "*.sdf"), ("All files", "*.*")],
            )
        finally:
            root.destroy()
        self.selected_files = list(files)
        return self._validate_files()

    def select_directory_via_gui(self) -> List[str]:
        """通过GUI选择目录并扫描SDF文件"""
        root = tk.Tk()
        root.withdraw()
        try:
            folder = filedialog.askdirectory(
                title="Select the directory that contains the SDF file"
            )
        finally:
            root.destroy()
        self.selected_files = self._find_sdf_in_dir(folder) if folder else []
        return self._validate_files()

    def set_files_manually(self, paths: Union[str, List[str]]) -> List[str]:
        """手动设置文件路径"""
        self.selected_files = [
            str(Path(p).resolve()) for p in ([paths] if isinstance(paths, str) else paths)
        ]
        return self._validate_files()

    def _find_sdf_in_dir(self, directory: str) -> List[str]:
        """扫描目录中的SDF文件"""
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith((".sdf", ".sd"))
        ]

    def _validate_files(self) -> List[str]:
        """验证文件有效性"""
        valid_files = []
        for f in self.selected_files:
            if not os.path.isfile(f):
                print(f"Invalid file: {f}")
                continue
            if not f.lower().endswith((".sdf", ".sd")):
                print(f"Ignore non-SDF files: {f}")
                continue
            valid_files.append(f)
        self.selected_files = valid_files
        return valid_files


import json
import shutil
import sqlite3
import tempfile
from contextlib import closing
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
from threading import Lock
from typing import List

from rdkit import Chem, Geometry
from tqdm import tqdm


@lru_cache(maxsize=4096)
def _indices_json(indices):
    return json.dumps(indices)


class SDFBatchProcessor:
    """
    处理SDF文件的核心类，功能包括：
    - 解析HydrogenShifts和CarbonShifts
    - 数据批量入库
    - 错误隔离处理
    """

    _prepared_lock = Lock()
    _prepared_dir = None
    _prepared_source = None
    _prepared = {}
    _shift_line = re.compile(r"(\d+)([a-zA-Z]+)?\[(\d+)\]\s+([-+]?\d*\.?\d+)")

    @staticmethod
    def _digest(path):
        digest = sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _source_key(cls, file_paths):
        # The spelling is written to source_file, and selection order determines
        # atom and molecule row order. Both are part of the key.
        def stamp(path):
            stat = path.stat()
            return (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

        key = []
        try:
            for name in file_paths:
                path = Path(name).resolve(strict=True)
                if not path.is_file():
                    return None
                before = stamp(path)
                digest = cls._digest(path)
                if before != stamp(path):
                    return None
                key.append((os.fspath(name), str(path), digest))
        except (OSError, RuntimeError):
            return None
        return tuple(key)

    def __init__(self, output_db: str = "chem_data.db"):
        self.output_db = output_db
        self._init_database()

    @classmethod
    def _restore_prepared(cls, key, prepare_hsqc, staging):
        cached = cls._prepared.get(prepare_hsqc) if key == cls._prepared_source else None
        if cached is None:
            return None
        path, checksum, report = cached
        try:
            shutil.copyfile(path, staging)
            if cls._digest(staging) == checksum:
                print(
                    f"Reused SDF library. Accepted: {report['accepted']}, "
                    f"rejected: {report['rejected']}"
                )
                return deepcopy(report)
        except OSError:
            pass
        staging.unlink(missing_ok=True)
        cls._prepared.pop(prepare_hsqc, None)
        return None

    @classmethod
    def _remember_prepared(cls, key, prepare_hsqc, staging, report):
        if key != cls._prepared_source:
            cls._prepared.clear()
            cls._prepared_source = key
        try:
            if cls._prepared_dir is None:
                cls._prepared_dir = tempfile.TemporaryDirectory(prefix="recqc-library-")
                atexit.register(cls._prepared_dir.cleanup)
            name = "library-hsqc.sqlite" if prepare_hsqc else "library.sqlite"
            cached = Path(cls._prepared_dir.name) / name
            shutil.copyfile(staging, cached)
            cls._prepared[prepare_hsqc] = (cached, cls._digest(cached), deepcopy(report))
        except OSError:
            cls._prepared.pop(prepare_hsqc, None)

    @classmethod
    def process_files_atomically(cls, file_paths: List[str], output_db: str, *, prepare_hsqc=False):
        """Import into a staging database and replace the live DB only on success."""
        destination = Path(output_db).resolve()
        staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.importing")
        with cls._prepared_lock:
            try:
                key = cls._source_key(file_paths)
                report = cls._restore_prepared(key, prepare_hsqc, staging)
                if report is None:
                    report = cls._restore_prepared(key, False, staging) if prepare_hsqc else None
                    reusable = report is not None
                    if report is None:
                        report = cls(str(staging)).process_files(file_paths)
                        if report["accepted"] == 0:
                            raise ValueError(
                                "No valid molecules were imported from the selected SDF files"
                            )
                        # File-level errors may be transient. Record-level rejections
                        # remain part of the report when the inputs are identical.
                        complete = all(error["record"] is not None for error in report["errors"])
                        reusable = (
                            key is not None and complete and key == cls._source_key(file_paths)
                        )
                        if reusable:
                            cls._remember_prepared(key, False, staging, report)
                    if prepare_hsqc:
                        from HSQCScoreProcess import CarbonCoordinateExtractor

                        CarbonCoordinateExtractor(str(staging)).process_all()
                        if reusable:
                            cls._remember_prepared(key, True, staging, report)
                os.replace(staging, destination)
                return report
            finally:
                staging.unlink(missing_ok=True)

    def _init_database(self):
        """初始化数据库结构"""
        with closing(sqlite3.connect(self.output_db)) as conn:
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

    @staticmethod
    def _sdf_records(path):
        # CORD-NMR predictor tags reference the original explicit atom indices.
        try:
            supplier = Chem.SDMolSupplier(path, removeHs=False)
        except OSError:
            if os.fspath(path).isascii():
                raise
            # Python can open paths that the RDKit file constructor rejects on Windows.
            with open(path, "rb") as source:
                if not source.peek(1):
                    raise
                yield from Chem.ForwardSDMolSupplier(source, removeHs=False)
        else:
            yield from supplier

    def process_files(self, file_paths: List[str]):
        """处理文件列表"""
        with closing(sqlite3.connect(self.output_db)) as conn:
            conn.execute("PRAGMA synchronous = NORMAL")
            # Keep molecule savepoints inside a transaction so releasing a
            # savepoint does not commit and sync the database for each record.
            conn.execute("BEGIN")
            report = {"total": 0, "accepted": 0, "rejected": 0, "errors": []}

            with tqdm(total=len(file_paths), desc="Processing progress") as pbar:
                for sdf_path in file_paths:
                    try:
                        for record_number, mol in enumerate(self._sdf_records(sdf_path), start=1):
                            report["total"] += 1
                            if not mol:
                                self._record_rejection(
                                    report, sdf_path, record_number, "RDKit parsing failed"
                                )
                                continue
                            conn.execute("SAVEPOINT one_molecule")
                            try:
                                self._process_molecule(mol, conn, sdf_path)
                            except Exception as exc:
                                conn.execute("ROLLBACK TO one_molecule")
                                conn.execute("RELEASE one_molecule")
                                self._record_rejection(report, sdf_path, record_number, str(exc))
                                continue
                            conn.execute("RELEASE one_molecule")
                            report["accepted"] += 1
                    except Exception as e:
                        report["errors"].append(
                            {"source_file": sdf_path, "record": None, "reason": str(e)}
                        )
                        print(f"File {sdf_path} processing failed: {str(e)}")
                    finally:
                        pbar.update(1)
                        pbar.set_postfix_str(os.path.basename(sdf_path))

            # Store the complete import in one transaction. Committing each
            # molecule separately changes no value but forces a disk sync for
            # every molecule.
            conn.commit()

            self.last_report = report
            print(
                "Processing is complete! "
                f"Total: {report['total']}, accepted: {report['accepted']}, "
                f"rejected: {report['rejected']}"
            )
            return report

    @staticmethod
    def _record_rejection(report, source_file, record_number, reason):
        report["rejected"] += 1
        report["errors"].append(
            {"source_file": source_file, "record": record_number, "reason": reason}
        )
        print(f"Skip {source_file} record {record_number}: {reason}")

    def _process_molecule(self, mol, conn, source_file):
        """处理单个分子"""
        # 检查分子是否已包含氢原子
        has_hydrogen = any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())

        # 只在没有氢原子时加氢
        if not has_hydrogen:
            mol = Chem.AddHs(mol)
            pass

        carbon_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
        carbon_count = len(carbon_atoms)

        # 获取分子量
        try:
            fw = float(mol.GetProp("FW"))
        except (KeyError, ValueError) as exc:
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            raise ValueError(f"Molecule {name!r} has no valid FW property") from exc

        # 解析位移数据
        h_shifts = self._parse_shift_data(mol, "HydrogenShifts")
        c_shifts = self._parse_shift_data(mol, "Predicted 13C shifts")
        carbon_indices = [atom.GetIdx() for atom in carbon_atoms]
        missing_c_shifts = [idx for idx in carbon_indices if idx not in c_shifts]
        if missing_c_shifts:
            raise ValueError(
                "Incomplete Predicted 13C shifts for atom indices: "
                + ", ".join(map(str, missing_c_shifts))
            )
        mol_block = Chem.MolToMolBlock(mol)
        # 插入分子记录
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO molecules
            (source_file, name, num_atoms, num_bonds, carbon_count, fw, mol_block)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                source_file,
                mol.GetProp("ID") if mol.HasProp("ID") else "",
                mol.GetNumAtoms(),
                mol.GetNumBonds(),
                carbon_count,
                fw,
                mol_block,
            ),
        )
        mol_id = cursor.lastrowid

        # 处理碳原子
        carbons = []
        for atom, atom_index in zip(carbon_atoms, carbon_indices):
            h_info = self._get_ch_info(atom, h_shifts)
            carbons.append(
                (
                    mol_id,
                    atom_index,
                    h_info["count"],
                    _indices_json(tuple(h_info["indices"])),
                    json.dumps(h_info["shifts"]),
                    c_shifts.get(atom_index),
                    _indices_json(tuple(h_info["connected_c_idx"])),
                )
            )

        # 批量插入碳原子数据
        cursor.executemany(
            """
            INSERT INTO carbons
            (mol_id, c_index, h_count, h_indices, h_shifts, c_shift,connected_carbon)
            VALUES (?, ?, ?, ?, ?, ?,?)
        """,
            carbons,
        )

    def _parse_shift_data(self, mol, tag: str) -> dict:
        """统一解析含中括号的位移数据"""
        shift_map = {}
        if mol.HasProp(tag):
            for line in mol.GetProp(tag).split("\n"):
                line = line.strip()
                if not line:
                    continue

                # 匹配格式：任意数字[实际索引+1] 位移值
                match = self._shift_line.match(line)
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
            atomic_number = neighbor.GetAtomicNum()
            if atomic_number == 1:
                h_idx = neighbor.GetIdx()
                h_indices.append(h_idx)
                h_shifts.append(h_shift_map.get(h_idx))
            elif atomic_number == 6:
                idx = neighbor.GetIdx()
                c_indices.append(idx)
        return {
            "count": len(h_indices),
            "indices": h_indices,
            "shifts": h_shifts,
            "connected_c_idx": c_indices,
        }
