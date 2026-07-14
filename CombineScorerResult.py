from PROCESSSDFFILES import SDFBatchProcessor
from HSQCScoreProcess import *
from CarbonScoreProcess import *
from typing import Union, List


class CombinedScorer:
    """结合碳原子评分和CH坐标点评分的联合评分类"""

    def __init__(self, db_path: str, c_weight: float, ch_weight: float):
        """
        :param db_path: 数据库路径
        :param c_weight: 碳原子评分权重 (0-1)
        :param ch_weight: CH坐标评分权重 (0-1)
        """
        if not (0 <= c_weight <= 1) or not (0 <= ch_weight <= 1):
            raise ValueError("The weight parameter must be between 0 and 1")
        if abs(c_weight + ch_weight - 1.0) > 1e-6:
            raise ValueError("The sum of the weights must be equal to 1")

        self.db_path = db_path
        self.c_weight = c_weight
        self.ch_weight = ch_weight
        self._create_combined_table()

    def _create_combined_table(self):
        """创建联合评分结果表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS combined_scores")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS combined_scores (
                    mol_id INTEGER PRIMARY KEY,
                    c_score REAL,
                    ch_score REAL,
                    combined_score REAL,
                    c_weight REAL,
                    ch_weight REAL,
                    FOREIGN KEY(mol_id) REFERENCES molecules(mol_id)
                )
            """)
            conn.commit()

    def _get_combined_scores(self) -> List[tuple]:
        """获取并计算联合评分数据"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT 
                    ms.mol_id,
                    ms.final_score AS c_score,
                    mcs.score AS ch_score
                FROM molecule_scores ms
                INNER JOIN molecule_ch_scores mcs 
                ON ms.mol_id = mcs.mol_id
            """)

            combined = []
            for row in cursor:
                c_score = row['c_score']
                ch_score = row['ch_score']
                combined_score = (c_score * self.c_weight) + (ch_score * self.ch_weight)
                combined.append((
                    row['mol_id'],
                    c_score,
                    ch_score,
                    round(combined_score, 4),
                    self.c_weight,
                    self.ch_weight
                ))
            return combined

    def process(self):
        """执行联合评分流程"""
        scores = self._get_combined_scores()

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT INTO combined_scores 
                (mol_id, c_score, ch_score, combined_score, c_weight, ch_weight)
                VALUES (?, ?, ?, ?, ?, ?)
            """, scores)

        print(f"Joint Grading Complete! A total of {len(scores)} molecules were processed")

import sqlite3
import io
from typing import List, Tuple, Dict
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem  # 新增导入
from matplotlib.patches import Ellipse
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import sqlite3
from PIL import Image, ImageTk
import os

class CombinedScorerGUI:
    """完整评分流程集成类，支持所有可配置参数"""

    def __init__(
            self,
            sdf_files: List[str],
            c_range: Tuple[int, int] = (10, 30),
            fw_range: Tuple[float, float] = (100.0, 300.0),
            c_mode: str = 'typed',
            C_merge: str='N',
            c_data: Union[dict, list] = None,
            carbon_score_mode: str = 'global',
            carbon_global_thresholds: Tuple[float, float] = (0.5, 2.0),
            carbon_fine_ranges: List[Tuple[Tuple[float, float], float, float]] = None,
            env_params: Tuple[int, float, float] = (2, 0.6, 0.4),
            ch_mode: int = 2,
            CH_merge: str='N',
            ch_data: Union[dict, list] = None,
            ch_tolerances: Tuple[float, float] = (1.0, 0.2),
            final_weights: Tuple[float, float] = (0.5, 0.5),
            db_path: str = "chem_data.db"
    ):
        """
        :param sdf_files: SDF文件路径列表
        :param c_mode: 碳匹配模式 ('typed'/'untyped')
        :param c_data: 碳匹配输入数据
        :param c_thresholds: 碳评分阈值 (green, yellow)
        :param env_params: 环境参数 (level, self_weight, env_weight)
        :param ch_mode: CH匹配模式 (1-3)
        :param ch_data: CH输入数据
        :param ch_tolerances: CH误差容限 (c_tol, h_tol)
        :param final_weights: 最终权重 (c_weight, ch_weight)
        :param db_path: 数据库路径
        """
        # 初始化数据库
        self.db_path = db_path
        processor = SDFBatchProcessor(db_path)
        processor.process_files(sdf_files)
        extractor = CarbonCoordinateExtractor(db_path)
        extractor.process_all()  # 生成carbon_h_coordinates表

        # 参数存储
        self.c_range = c_range
        self.fw_range = fw_range
        self.c_mode = c_mode
        self.c_data = c_data or ({0: [], 1: [], 2: [], 3: []} if c_mode == 'typed' else [])
        self.carbon_score_mode = carbon_score_mode
        self.carbon_global_thresholds = carbon_global_thresholds
        self.carbon_fine_ranges = carbon_fine_ranges or [
            ((0, 50), 1.0, 2.0),
            ((50, 150), 2.0, 3.0),
            ((150, 300), 3.0, 4.0)
        ]
        self.env_level, self.self_weight, self.env_weight = env_params
        self.ch_mode = ch_mode
        self.CH_merge = CH_merge
        self.C_merge = C_merge
        self.ch_data = ch_data or self._default_ch_data()
        self.c_tol, self.h_tol = ch_tolerances
        self.c_weight, self.ch_weight = final_weights

        self._validate_parameters()

    def _default_ch_data(self):
        """生成默认CH数据结构（与单独 HSQC 模式保持一致）"""
        return {
            1: {'All_type': []},
            2: {'type13': [], 'type2': []},
            3: {'type1': [], 'type2': [], 'type3': []}
        }[self.ch_mode]

    def _validate_parameters(self):
        """综合参数校验"""
        if self.c_range[0] > self.c_range[1]:
            raise ValueError("The carbon number range is invalid, and the starting value needs to be less than the end value")
        if self.fw_range[0] > self.fw_range[1]:
            raise ValueError("The molecular weight range is invalid, and the start value needs to be less than the end value")
        # 碳数据
        if self.c_mode == 'typed' and not isinstance(self.c_data, dict):
            raise ValueError("The type pattern requires dictionary-formatted data")
        if self.c_mode == 'untyped' and not isinstance(self.c_data, list):
            raise ValueError("Global mode requires data in list format")

        if self.carbon_score_mode not in ('global', 'fine'):
            raise ValueError("The carbon scoring model must be Global or Fine")

        if self.carbon_score_mode == 'global':
            if len(self.carbon_global_thresholds) != 2:
                raise ValueError("The global mode requires two thresholds")
            if self.carbon_global_thresholds[0] >= self.carbon_global_thresholds[1]:
                raise ValueError("The green threshold must be less than the yellow threshold")

        if self.carbon_score_mode == 'fine':
            for r in self.carbon_fine_ranges:
                (low, high), g, y = r
                if low >= high or g >= y:
                    raise ValueError(f"Invalid granular range: {r}")

        # 环境参数
        if self.env_level not in {1, 2, 3}:
            raise ValueError("The environment level must be 1-3")
        if abs(self.self_weight + self.env_weight - 1.0) > 1e-6:
            raise ValueError("The sum of environmental weights must be 1")

        # CH数据
        if self.ch_mode == 1 and not {'All_type'}.issubset(self.ch_data):
            raise ValueError("Mode 1 requires All_type key")
        if self.ch_mode == 2 and not {'type13', 'type2'}.issubset(self.ch_data):
            raise ValueError("Mode 2 requires the type13 and type2 keys")
        if self.ch_mode == 3 and not {'type1', 'type2', 'type3'}.issubset(self.ch_data):
            raise ValueError("Mode 3 requires all classification keys")

        # 最终权重
        if abs((self.c_weight + self.ch_weight) - 1.0) > 1e-6:
            raise ValueError("The final sum of weights must be 1")

    def execute(self) -> List[Tuple]:
        """执行完整工作流"""
        # 阶段1：碳原子匹配与评分
        self._process_carbon_matching()

        self._process_ch_matching()

        # 阶段3：综合评分
        return self._get_combined_results()

    def _process_carbon_matching(self):
        """处理碳相关流程"""
        # 碳匹配
        c_matcher = GreedyMatcher(
            db_path=self.db_path,
            user_data=self.c_data,
            c_range=self.c_range,
            fw_range=self.fw_range,
            C_merge=self.C_merge,
            mode=self.c_mode
        )
        # 将 GUI 中用户输入的“第一容差/绿色阈值”真正传给匹配器，
        # 让联合模式下的 C 匹配与 CarbonOnly 模式保持一致。
        if self.carbon_score_mode == 'global':
            c_matcher.match_green_threshold = float(self.carbon_global_thresholds[0])
            c_matcher.match_yellow_threshold = float(self.carbon_global_thresholds[1])
        elif self.carbon_score_mode == 'fine':
            c_matcher.match_fine_ranges = self.carbon_fine_ranges

        c_matcher.run()

        # 碳原子评分
        c_scorer = CarbonScorer(self.db_path)
        c_scorer.create_results_table()
        c_scorer.process_scoring(
            mode=self.carbon_score_mode,
            global_thresholds=self.carbon_global_thresholds,
            fine_ranges=self.carbon_fine_ranges
        )
        # 分子碳评分
        molecule_c_scorer = RobustMoleculeScorer(self.db_path)
        molecule_c_scorer.create_result_table()
        molecule_c_scorer.process_molecules(
            self_weight=self.self_weight,
            env_weight=self.env_weight,
            env_level=self.env_level
        )

    def _process_ch_matching(self):
        """处理CH相关流程（与单独 HSQC 模式保持一致）"""
        # 兼容两种 UserPointMatcher：
        # 1) 新版：__init__ 支持 c_tolerance / h_tolerance，匹配阶段会用到容差
        # 2) 旧版：__init__ 不支持，评分阶段仍可正常工作
        try:
            ch_matcher = UserPointMatcher(
                db_path=self.db_path,
                user_points=self.ch_data,
                mode=self.ch_mode,
                c_tolerance=self.c_tol,
                h_tolerance=self.h_tol,
                CH_merge=self.CH_merge
            )
        except TypeError:
            ch_matcher = UserPointMatcher(
                db_path=self.db_path,
                user_points=self.ch_data,
                mode=self.ch_mode,
                CH_merge=self.CH_merge
            )
            # 旧版匹配器兜底挂属性，尽量与单独 HSQC 模式行为一致
            setattr(ch_matcher, 'c_tolerance', self.c_tol)
            setattr(ch_matcher, 'h_tolerance', self.h_tol)
            setattr(ch_matcher, 'c_tol', self.c_tol)
            setattr(ch_matcher, 'h_tol', self.h_tol)

        ch_matcher.process()

        # CH坐标评分（保持原有矩形外打分逻辑）
        ch_scorer = MoleculeScoreCalculator(
            db_path=self.db_path,
            c_tolerance=self.c_tol,
            h_tolerance=self.h_tol
        )
        ch_scorer.process_all()

    def _get_combined_results(self, top_n: int = 20) -> List[Tuple]:
        """获取综合评分结果"""
        # 执行联合评分
        CombinedScorer(
            db_path=self.db_path,
            c_weight=self.c_weight,
            ch_weight=self.ch_weight
        ).process()

        # 检索结果
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("""
                SELECT m.mol_id, m.name, m.source_file,
                       cs.combined_score, cs.c_score, cs.ch_score
                FROM combined_scores cs
                JOIN molecules m ON cs.mol_id = m.mol_id
                ORDER BY 
                    cs.combined_score DESC,  -- 主排序：综合评分降序
                    m.carbon_count DESC      -- 次排序：碳原子数降序（评分相同时C数多的靠前）
                LIMIT ?
            """, (top_n,)).fetchall()
    def _get_molecule(self, conn: sqlite3.Connection, mol_id: int) -> Chem.Mol:
        """从数据库获取分子对象"""
        mol_block = conn.execute(
            "SELECT mol_block FROM molecules WHERE mol_id = ?", (mol_id,)
        ).fetchone()[0]
        return Chem.MolFromMolBlock(mol_block)

    def _get_carbon_data(self, conn: sqlite3.Connection, mol_id: int) -> List[Dict]:
        """获取碳原子评分数据"""
        return [
            {
                'index': row[0],
                'color': row[1],
                'db_shift': row[2],
                'user_shift': row[3],
                'h_count': row[4]
            }
            for row in conn.execute("""
                SELECT cs.c_index, cs.color, cs.db_shift, cs.user_shift, c.h_count 
                FROM carbon_scores cs
                JOIN carbons c ON cs.mol_id = c.mol_id AND cs.c_index = c.c_index
                WHERE cs.mol_id = ?
            """, (mol_id,))
        ]

    def _get_ch_matches(self, conn: sqlite3.Connection, mol_id: int) -> List[Tuple]:
        """获取CH匹配数据"""
        return conn.execute("""
            SELECT c_index, h_type FROM user_matches
            WHERE mol_id = ? AND distance < 1e9
        """, (mol_id,)).fetchall()

    def _color_to_rgb(self, color_str: str) -> tuple:
        """HEX颜色转RGB元组"""
        if color_str.startswith('#'):
            return (
                int(color_str[1:3], 16)/255.0,
                int(color_str[3:5], 16)/255.0,
                int(color_str[5:7], 16)/255.0
            )
        return (0.8, 0.8, 0.8)

    # def generate_plots(self, output_dir: str = "combined_results"):
    #     """生成联合评分可视化图表"""
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #
    #     results = self._get_combined_results()
    #
    #     with sqlite3.connect(self.db_path) as conn:
    #         for row in results:
    #             mol_id, name = row[0], row[1]
    #             mol = self._get_molecule(conn, mol_id)
    #             if not mol:
    #                 continue
    #
    #             # 生成C评分相关图表
    #             self._generate_c_score_plots(conn, mol, mol_id, name, output_dir)
    #
    #             # 生成CH匹配相关图表（修正调用）
    #             self._generate_ch_match_plots(conn, mol_id, name, output_dir)

    def _generate_plots(self, output_dir: str = "combined_results"):
        """生成联合评分可视化图表（包含C评分和CH匹配两种图表）"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        results = self._get_combined_results()

        with sqlite3.connect(self.db_path) as conn:
            for row in results:
                mol_id, name = row[0], row[1]
                mol = self._get_molecule(conn, mol_id)
                if not mol:
                    continue

                # 生成C评分相关图表（与CarbonOnly模式一致）
                self._generate_c_score_plots(conn, mol, mol_id, name, output_dir)

                # 生成CH匹配相关图表（与CHOnly模式一致）
                self._generate_ch_match_plots(conn, mol_id, name, output_dir)

    def _generate_c_score_plots(self, conn, mol, mol_id, name, output_dir):
        """生成碳评分相关图表"""
        # 获取碳原子数据
        carbons = []
        for row in conn.execute("""
            SELECT cs.c_index, cs.color, cs.db_shift, cs.user_shift, c.h_count 
            FROM carbon_scores cs
            JOIN carbons c ON cs.mol_id = c.mol_id AND cs.c_index = c.c_index
            WHERE cs.mol_id = ?
        """, (mol_id,)):
            carbons.append({
                'index': row[0],
                'color': row[1],
                'db_shift': row[2],
                'user_shift': row[3],
                'h_count': row[4]
            })

        # 生成三种视角的结构图
        self._draw_labeled_molecule(
            mol=mol,
            highlight_atoms=[c['index'] for c in carbons],
            highlight_colors=[self._color_to_rgb(c['color']) for c in carbons],
            labels={c['index']: str(c['index']) for c in carbons},
            filename=f"{output_dir}/{name}_c_indices.png"
        )
        self._draw_labeled_molecule(
            mol=mol,
            highlight_atoms=[c['index'] for c in carbons],
            highlight_colors=[self._color_to_rgb(c['color']) for c in carbons],
            labels={c['index']: f"{c['db_shift']:.1f}" for c in carbons if c.get('db_shift') is not None},
            filename=f"{output_dir}/{name}_c_db_shifts.png"
        )
        self._draw_labeled_molecule(
            mol=mol,
            highlight_atoms=[c['index'] for c in carbons],
            highlight_colors=[self._color_to_rgb(c['color']) for c in carbons],
            labels={c['index']: f"{c['user_shift']:.1f}" for c in carbons if c.get('user_shift') is not None},
            filename=f"{output_dir}/{name}_c_user_shifts.png"
        )

    def _generate_ch_match_plots(self, conn, mol_id, name, output_dir):
        """CH匹配绘图核心方法"""
        # 获取库匹配点
        lib_points = []
        lib_query = conn.execute("""
            SELECT um.db_point, c.h_count 
            FROM user_matches um
            JOIN carbons c ON um.mol_id = c.mol_id AND um.c_index = c.c_index
            WHERE um.mol_id = ? AND um.distance < 1e9
        """, (mol_id,))
        for row in lib_query:
            try:
                c, h = json.loads(row[0])
                lib_points.append( (c, h, row[1]) )
            except:
                continue

        # 获取用户匹配点
        user_points = []
        user_query = conn.execute("""
            SELECT user_point, h_type 
            FROM user_matches 
            WHERE mol_id = ? AND distance < 1e9
        """, (mol_id,))
        for row in user_query:
            try:
                c, h = json.loads(row[0])
                user_points.append( (c, h, row[1]) )
            except:
                continue

        # 绘图执行
        if lib_points:
            self._plot_ch_coordinates(
                points=lib_points,
                title=f"Library CH Matches: {name} (Total: {len(lib_points)})",
                filename=f"{output_dir}/{name}_ch_library.png",
                is_user=False
            )

        if user_points:
            self._plot_ch_coordinates(
                points=user_points,
                title=f"User CH Matches: {name} (Total: {len(user_points)})",
                filename=f"{output_dir}/{name}_ch_user.png",
                is_user=True
            )

    def _plot_ch_coordinates(self, points, title, filename, is_user=False):
        """标准化CH坐标绘图（关键修正）"""
        plt.figure(figsize=(10,6))
        ax = plt.gca()
        ax.invert_xaxis()

        color_map = {
            0: '#1f77b4', 1: '#ff7f0e',
            2: '#2ca02c', 3: '#d62728'
        }
        marker = 'o' if is_user else 's'
        handled_types = set()

        # 带Jitter的散点绘制
        for idx, (c, h, h_type) in enumerate(points):
            color = color_map.get(h_type, '#7f7f7f')
            jitter = 0.02 * idx  # 防重叠偏移
            plt.scatter(
                h + jitter, c + jitter,
                c=color, marker=marker, s=100,
                edgecolors='black', alpha=0.7,
                label=f'Type {h_type}' if h_type not in handled_types else ""
            )
            handled_types.add(h_type)

        # 图例处理
        if handled_types:
            handles = [
                plt.Line2D([0], [0],
                           marker=marker,
                           color='w',
                           markerfacecolor=color_map[t],
                           markersize=10,
                           label=f'Type {t}')
                for t in sorted(handled_types)
            ]
            plt.legend(handles=handles)

        plt.xlabel("1H Shift (ppm)")
        plt.ylabel("13C Shift (ppm)")
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_carbon_visuals(self, mol: Chem.Mol, carbons: List[Dict],
                             name: str, output_dir: str):
        """Generate three perspective visualizations for carbon matches"""
        # 1. Atomic Index View
        self._draw_labeled_molecule(
            mol=mol,
            highlight_atoms=[c['index'] for c in carbons],
            highlight_colors=[self._color_to_rgb(c['color']) for c in carbons],
            labels={c['index']: str(c['index']) for c in carbons},
            filename=f"{output_dir}/{name}_carbon_indices.png"
        )

        # 2. Database Shift View
        self._draw_labeled_molecule(
            mol=mol,
            highlight_atoms=[c['index'] for c in carbons],
            highlight_colors=[self._color_to_rgb(c['color']) for c in carbons],
            labels={c['index']: f"{c['db_shift']:.1f}" for c in carbons},
            filename=f"{output_dir}/{name}_carbon_db_shifts.png"
        )

        # 3. User Shift View
        self._draw_labeled_molecule(
            mol=mol,
            highlight_atoms=[c['index'] for c in carbons],
            highlight_colors=[self._color_to_rgb(c['color']) for c in carbons],
            labels={c['index']: f"{c['user_shift']:.1f}" for c in carbons if c.get('user_shift') is not None},
            filename=f"{output_dir}/{name}_carbon_user_shifts.png"
        )

    def _plot_coordinate_scatter(self, points: List[Tuple], label: str, color: str):
        """Universal scatter plot renderer"""
        if points:
            c_vals = [p[0] for p in points]
            h_vals = [p[1] for p in points]
            plt.scatter(h_vals, c_vals, c=color, label=label, alpha=0.6)
            plt.gca().invert_xaxis()
            plt.xlabel("1H Shift (ppm)")
            plt.ylabel("13C Shift (ppm)")
            plt.legend()

    def _add_tolerance_ellipses(self):
        """Add tolerance range visualization"""
        for (x, y) in [(h,c) for c,h in self.ch_data]:
            ellipse = Ellipse(
                xy=(x, y),
                width=2*self.h_tol,
                height=2*self.c_tol,
                edgecolor='gray',
                facecolor='none',
                linestyle='--'
            )
            plt.gca().add_patch(ellipse)

    def _get_ch_library_points(self, conn: sqlite3.Connection, mol_id: int):
        """Retrieve matched library coordinates"""
        return [
            json.loads(row['db_point'])
            for row in conn.execute("""
                    SELECT db_point FROM user_matches
                    WHERE mol_id=? AND distance < 1e9
                """, (mol_id,))
        ]

    def _get_ch_user_points(self, conn: sqlite3.Connection, mol_id: int):
        """Retrieve user input coordinates"""
        return [
            json.loads(row['user_point'])
            for row in conn.execute("""
                    SELECT user_point FROM user_matches
                    WHERE mol_id=? AND distance < 1e9 
                """, (mol_id,))
        ]

    def _draw_labeled_molecule(self, mol: Chem.Mol,
                               highlight_atoms: List[int],
                               highlight_colors: List[tuple],
                               labels: Dict[int, str],
                               filename: str):
        """优化后的分子结构绘制方法（使用正确坐标类型）"""
        from rdkit import Geometry  # 确保Geometry导入

        # 复制分子并设置原子标签
        mol = Chem.Mol(mol)
        AllChem.Compute2DCoords(mol)
        for idx, label in labels.items():
            mol.GetAtomWithIdx(idx).SetProp("atomNote", label)

        # 配置高级绘图参数
        drawer = Draw.MolDraw2DCairo(500, 500)
        opts = drawer.drawOptions()

        # 设置绘图样式参数
        opts.setBackgroundColour((173/255, 216/255, 230/255))
        opts.bondLineWidth = 2.5
        try:
            opts.dotsPerAngstrom = 1000
        except Exception:
            pass
        opts.fixedBondLength = 70
        opts.baseFontSize = 0.6
        opts.annotationFontScale = 0.8
        opts.highlightRadius = 0.3
        opts.addAtomIndices = False

        # 配置高亮样式
        highlight_dict = {
            atom_idx: color
            for atom_idx, color in zip(highlight_atoms, highlight_colors)
        }

        # 绘制分子
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_dict
        )

        # 添加图例框（使用正确坐标类型）
        try:
            opts.lineWidth = 2
        except Exception:
            pass
        drawer.SetColour((0, 0, 0))
        # 使用Point2D坐标并指定使用原始坐标系
        drawer.DrawRect(
            Geometry.Point2D(0, 0),
            Geometry.Point2D(1, 1),
            True  # rawCoords=True表示使用画布坐标系
        )

        drawer.FinishDrawing()

        # 保存为PNG
        img = Image.open(io.BytesIO(drawer.GetDrawingText()))
        img = img.resize((800, 800), Image.LANCZOS)
        img.save(filename, dpi=(300, 300))

class CombinedResultViewer:
    def __init__(self, db_path, results, output_dir="combined_results"):
        self.db_path = db_path
        self.results = results
        self.output_dir = output_dir
        self.current_page = 0
        os.makedirs(self.output_dir, exist_ok=True)

    def create_result_window(self, parent):
        """创建联合评分结果展示窗口"""
        self.top = tk.Toplevel(parent)
        self.top.title("Joint scoring results")
        self.top.geometry("800x600")  # 统一窗口尺寸

        # 分页控制栏（保持与Carbon一致）
        control_frame = ttk.Frame(self.top)
        ttk.Button(control_frame, text="◀ Previous",
                   command=lambda: self.show_page(-1)).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Next ▶",
                   command=lambda: self.show_page(1)).pack(side=tk.LEFT)
        self.page_label = ttk.Label(control_frame, text="", font=('Arial', 10))
        self.page_label.pack(side=tk.LEFT, padx=10)
        control_frame.pack(pady=5, fill=tk.X)  # 调整间距与Carbon一致

        # 结果展示网格（统一为2行x3列）
        self.grid_frame = ttk.Frame(self.top)
        self.grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 配置网格行列权重（与Carbon一致）
        for r in range(2):
            self.grid_frame.rowconfigure(r, weight=1)
        for c in range(3):
            self.grid_frame.columnconfigure(c, weight=1)

        self.show_page(0)  # 初始化显示方式与Carbon一致

    def show_page(self, delta):
        """分页显示处理（保持与Carbon一致）"""
        new_page = self.current_page + delta
        max_page = (len(self.results) - 1) // 6
        self.current_page = max(0, min(new_page, max_page))
        self.page_label.config(
            text=f"Page {self.current_page+1}/{(len(self.results)+5)//6} "
        )
        self.render_items()

    def render_items(self):
        """渲染当前页的6个结果项（统一为2行3列布局）"""
        # 清空当前内容
        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        # 获取当前页数据
        start = self.current_page * 6
        page_results = self.results[start:start+6]

        # 在2x3网格中排列结果项（与Carbon一致）
        for idx, result in enumerate(page_results):
            row = idx // 3  # 0-1行
            col = idx % 3   # 0-2列
            item = self.create_result_item(result, start + idx)
            item.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

    def create_result_item(self, result, rank):
        """创建结果项（保持与Carbon一致的样式）"""
        mol_id, name, source, combined_score, c_score, ch_score = result
        item_frame = ttk.Frame(self.grid_frame, relief="groove", borderwidth=1)

        # 图片展示（保持200x200尺寸）
        img_path = f"{self.output_dir}/{name}_c_indices.png"
        if os.path.exists(img_path):
            img = Image.open(img_path).resize((200, 200), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label = ttk.Label(item_frame, image=photo)
            label.image = photo
            label.pack(pady=2)

        # 信息展示（统一字体样式）
        info_frame = ttk.Frame(item_frame)
        ttk.Label(info_frame, text=f"Rank：{rank+1}",
                  font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        ttk.Label(info_frame,
                  text=f"synthesis: {float(combined_score):.2f}",
                  font=('Arial', 9)).pack(anchor=tk.W)
        ttk.Label(info_frame,
                  text=f"C: {c_score:.2f} | CH: {ch_score:.2f}",
                  font=('Arial', 9)).pack(anchor=tk.W)

        # 修正后的详情按钮（添加mol_name参数）
        ttk.Button(info_frame, text="MORE DETAIL",
                   command=lambda mid=mol_id, n=name: self.show_detail(mid, n),
                   style='TButton').pack(pady=2)
        info_frame.pack(fill=tk.X)

        return item_frame
    def show_detail(self, mol_id, mol_name):
        """联合评分详情窗口"""
        detail_win = tk.Toplevel(self.top)
        detail_win.title(f"Molecule details - {mol_name}")
        detail_win.geometry("1400x900")
        detail_win.configure(bg='#2C3E50')

        # 顶部信息栏
        header_frame = tk.Frame(detail_win, bg='#2C3E50')
        header_frame.pack(fill=tk.X, padx=10, pady=10)

        with sqlite3.connect(self.db_path) as conn:
            score_data = conn.execute("""
                SELECT combined_score, c_score, ch_score 
                FROM combined_scores 
                WHERE mol_id=?""", (mol_id,)).fetchone()

        info_text = (f"ID: {mol_id} | name: {mol_name} | "
                     f"Score: {score_data[0]:.2f} | "
                     f"C_Score: {score_data[1]:.2f} | "
                     f"CH_Score: {score_data[2]:.2f}")
        tk.Label(header_frame, text=info_text, font=('Arial', 12),
                 bg='#2C3E50', fg='white').pack(side=tk.LEFT)

        # 主内容区域
        main_frame = tk.Frame(detail_win, bg='#2C3E50')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧碳图表区域
        left_panel = tk.Frame(main_frame, bg='#2C3E50')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 碳图表切换控制
        self.carbon_img_type = tk.StringVar(value="indices")
        radio_frame = tk.Frame(left_panel, bg='#2C3E50')
        radio_frame.pack(pady=5)
        for text, img_type in [('C_INDEX', 'indices'),
                               ('DB_SHIFT', 'db_shifts'),
                               ('USER_SHIFT', 'user_shifts')]:
            tk.Radiobutton(radio_frame, text=text, variable=self.carbon_img_type,
                           value=img_type, command=lambda: self.update_carbon_img(mol_name),
                           bg='#2C3E50', fg='white').pack(side=tk.LEFT)

        # 碳图表显示
        self.carbon_img_label = tk.Label(left_panel, bg='#2C3E50')
        self.carbon_img_label.pack(expand=True)
        self.update_carbon_img(mol_name)

        # 右侧CH图表区域
        right_panel = tk.Frame(main_frame, bg='#2C3E50')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # CH库匹配图
        ch_lib_path = f"{self.output_dir}/{mol_name}_ch_library.png"
        if os.path.exists(ch_lib_path):
            img = Image.open(ch_lib_path).resize((600, 300), Image.LANCZOS)
            self.ch_lib_photo = ImageTk.PhotoImage(img)
            tk.Label(right_panel, image=self.ch_lib_photo, bg='#2C3E50').pack(pady=5)

        # CH用户匹配图
        ch_user_path = f"{self.output_dir}/{mol_name}_ch_user.png"
        if os.path.exists(ch_user_path):
            img = Image.open(ch_user_path).resize((600, 300), Image.LANCZOS)
            self.ch_user_photo = ImageTk.PhotoImage(img)
            tk.Label(right_panel, image=self.ch_user_photo, bg='#2C3E50').pack(pady=5)

        # 下部表格区域
        table_frame = tk.LabelFrame(detail_win, text="Carbon atom shift matching details",
                                    bg='#2C3E50', fg='white')
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 创建表格
        columns = ("C_INDEX", "DB_SHIFT", "USER_SHIFT", "ABS", "Match score")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # 加载数据
        self.load_carbon_data(mol_id)

    def update_carbon_img(self, mol_name):
        """更新碳图表显示"""
        img_type = self.carbon_img_type.get()
        img_path = f"{self.output_dir}/{mol_name}_c_{img_type}.png"

        if os.path.exists(img_path):
            img = Image.open(img_path).resize((500, 500), Image.LANCZOS)
            self.carbon_photo = ImageTk.PhotoImage(img)
            self.carbon_img_label.config(image=self.carbon_photo)
            self.carbon_img_label.image = self.carbon_photo

    def load_carbon_data(self, mol_id):
        """加载碳原子数据"""
        with sqlite3.connect(self.db_path) as conn:
            carbons = conn.execute("""
                SELECT c_index, db_shift, user_shift, score 
                FROM carbon_scores 
                WHERE mol_id=?""", (mol_id,)).fetchall()

            for carbon in carbons:
                db_shift = carbon[1]
                user_shift = carbon[2]
                delta = abs(db_shift - user_shift) if (db_shift is not None and user_shift is not None) else None
                self.tree.insert("", tk.END, values=(
                    carbon[0],
                    f"{db_shift:.2f}" if db_shift is not None else "N/A",
                    f"{user_shift:.2f}" if user_shift is not None else "N/A",
                    f"{delta:.2f}" if delta is not None else "N/A",
                    f"{carbon[3]:.2f}" if carbon[3] is not None else "0"
                ))

# if __name__ == "__main__":
#
#     config = {
#         'sdf_files': ['Penicillium_data.sdf'],
#         'c_mode': 'typed',
#         'c_data': {
#             0: [205.0, 199.9, 205.0, 199.9, 205.0, 199.9],
#             1: [128.5, 155.0, 128.5, 155.0, 128.5, 155.0, 128.5, 155.0],
#             2: [95.8, 124.7, 95.8, 124.7, 95.8, 124.7, 95.8, 124.7],
#             3: [21.3, 21.0, 21.3, 21.0, 21.3, 21.0, 21.3, 21.0,21.3, 21.0, 21.3, 21.0, 21.3, 21.0, 21.3, 21.0]
#         },
#         # 'c_mode': 'untyped',
#         # 'c_data': [21.3, 21.0, 21.3, 21.0, 21.3, 21.0, 21.3, 21.0,21.3, 21.0, 21.3, 21.0, 21.3, 21.0, 21.3, 21.0],
#         'c_range': (1, 40),
#         'fw_range': (1, 1000),
#         'carbon_score_mode': 'global',
#         'carbon_global_thresholds': (1, 3),
#         'env_params': (1, 0.7, 0.3),
#         # 'carbon_score_mode': 'fine',
#         # 'carbon_fine_ranges': [
#         #     ((0, 50), 0.8, 1.5),
#         #     ((50, 150), 1.5, 2.5),
#         #     ((150, 300), 2.5, 3.5)
#         # ],
#         'ch_mode': 2,
#         'ch_data': {
#             'type13': [[125.0, 1.2], [130.5, 2.1]],
#             'type2': [[120.0, 1.5], [118.5, 1.6]]
#         },
#         # 'ch_mode': 3,
#         # 'ch_data': {
#         #     'type1': [[125.0, 1.2], [130.5, 2.1]],
#         #     'type2': [[120.0, 1.5], [118.5, 1.6]]
#         #     'type1': [[125.0, 1.2], [130.5, 2.1]],
#         # },
#         'ch_tolerances': (0.8, 0.15),
#         'final_weights': (0.65, 0.35)
#     }
#
#     pipeline = CombinedScorerGUI(**config)
#     results = pipeline.execute()
#     pipeline.generate_plots(output_dir="combined_results")
#
#     print("Top :")
#     for idx, row in enumerate(results, 1):
#         print(f"{idx}. [{row[3]:.4f}] ID:{row[1]}\n"
#               f"   综合评分: {row[3]:.4f} (C: {row[4]:.4f}, CH: {row[5]:.4f})")
