import sqlite3
import io
from typing import List, Tuple, Union
from HSQCScoreProcess import *
from PROCESSSDFFILES import SDFBatchProcessor
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem  # 新增导入

class CHOnlyScorerGUI:
    """仅处理CH坐标点评分的GUI集成类"""

    def __init__(
            self,
            sdf_files: List[str],
            ch_mode: int = 2,
            ch_data: Union[dict, list] = None,
            tolerances: Tuple[float, float] = (1.0, 0.2),
            db_path: str = "chem_data.db",
            CH_merge: str = "N",
    ):
        """
        :param sdf_files: SDF文件路径列表
        :param ch_mode: CH匹配模式 (1-3)
        :param ch_data: CH输入数据
        :param tolerances: 误差容限 (c_tol, h_tol)
        """
        # 初始化数据库
        self.db_path = db_path
        processor = SDFBatchProcessor(db_path)
        processor.process_files(sdf_files)

        # 生成坐标数据
        extractor = CarbonCoordinateExtractor(db_path)
        extractor.process_all()

        # 参数存储
        self.ch_mode = ch_mode
        self.CH_merge = CH_merge
        self.ch_data = ch_data or self._default_ch_data()
        self.c_tol, self.h_tol = tolerances

        self._validate_parameters()

    def _default_ch_data(self):
        """生成默认CH数据结构"""
        return {
            1: {'All_type': []},
            2: {'type13': [], 'type2': []},
            3: {'type1': [], 'type2': [], 'type3': []}
        }[self.ch_mode]

    def _validate_parameters(self):
        """参数验证"""
        if self.ch_mode not in {1, 2, 3}:
            raise ValueError("The CH mode must be 1-3.")

        if self.ch_mode == 1 and not {'All_type'}.issubset(self.ch_data):
            raise ValueError("Mode 1 requires All_type")
        if self.ch_mode == 2 and not {'type13', 'type2'}.issubset(self.ch_data):
            raise ValueError("Mode 2 requires type13 and type2 key")
        if self.ch_mode == 3 and not {'type1', 'type2', 'type3'}.issubset(self.ch_data):
            raise ValueError("Mode 3 requires All_type key")

        if any(t <= 0 for t in (self.c_tol, self.h_tol)):
            raise ValueError("误差容限必须大于0")

    def _save_run_params(self):
        """保存本次运行的显示参数，供结果窗口读取"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS hsqc_run_params")
            conn.execute("CREATE TABLE IF NOT EXISTS hsqc_run_params (id INTEGER PRIMARY KEY CHECK (id = 1), c_tol REAL, h_tol REAL)")
            conn.execute("INSERT OR REPLACE INTO hsqc_run_params (id, c_tol, h_tol) VALUES (1, ?, ?)", (self.c_tol, self.h_tol))
            conn.commit()

    def execute(self) -> List[Tuple]:
        """执行CH评分流程"""
        # CH坐标匹配
        matcher = UserPointMatcher(
            db_path=self.db_path,
            user_points=self.ch_data,
            mode=self.ch_mode,
            c_tolerance=self.c_tol,
            h_tolerance=self.h_tol,
            CH_merge=self.CH_merge,

        )
        matcher.process()

        # CH坐标评分
        scorer = MoleculeScoreCalculator(
            db_path=self.db_path,
            c_tolerance=self.c_tol,
            h_tolerance=self.h_tol
        )
        scorer.process_all()
        self._save_run_params()

        return self._get_results()

    def _get_results(self, top_n: int = 20) -> List[Tuple]:
        """获取评分结果"""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("""
                SELECT m.mol_id, m.name, m.source_file,
                       chs.score AS ch_score,
                       chs.matched_points,
                       chs.total_points
                FROM molecule_ch_scores chs
                JOIN molecules m ON chs.mol_id = m.mol_id
                ORDER BY 
                    chs.score DESC,          
                    m.carbon_count DESC      
                LIMIT ?
            """, (top_n,)).fetchall()

import sqlite3
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm
from typing import List, Tuple, Dict, Union
from matplotlib.patches import Ellipse

class CHMatchVisualizer:
    """基于匹配结果的CH坐标可视化类"""

    def __init__(self, db_path: str,
                 top_results: List[Tuple],
                 mode: int = 2):
        """
        :param db_path: 数据库路径
        :param top_results: 前N结果列表 [(mol_id, mol_name, ...), ...]
        :param mode: 匹配模式 'typed'/'untyped'
        """
        self.db_path = db_path
        self.top_results = top_results
        self.mode = mode
        self._validate_mode()

    def _plot_user_matches_for_molecule(self, output_dir: str, mol_id: int, mol_name: str):
        """为单个分子生成用户匹配坐标图"""
        with sqlite3.connect(self.db_path) as conn:
            points = []
            cursor = conn.execute("""
                SELECT um.user_point, um.h_type 
                FROM user_matches um
                WHERE um.mol_id = ? AND um.distance < 1e9
            """, (mol_id,))

            for row in cursor:
                c, h = json.loads(row['user_point'])
                points.append((c, h, row['h_type']))

            if points:
                self._plot_single(
                    points=points,
                    title=f"User Matches: ID：{mol_name} ",
                    filename=os.path.join(output_dir, f"user_{mol_id}_matches.png")
                )
    def _validate_mode(self):
        """验证模式参数有效性"""
        if self.mode not in (1,2,3):
            raise ValueError("The mode must be 1-3")

    def generate_plots(self, output_dir: str = "ch_match_plots"):
        """生成所有可视化图表"""
        os.makedirs(output_dir, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            for mol_id, mol_name, *_ in tqdm(self.top_results, desc="Generate a chart"):
                # 生成库匹配图
                mol_block = conn.execute(
                    "SELECT mol_block FROM molecules WHERE mol_id = ?",
                    (mol_id,)
                ).fetchone()['mol_block']

                if mol_block:
                    try:
                        self._draw_molecule_structure(
                            mol_block=mol_block,
                            filename=os.path.join(output_dir, f"struct_{mol_id}.png"),
                            title=f"Structure: {mol_name} (ID:{mol_id})"
                        )
                    except Exception as e:
                        print(f"Unable to generate structure diagram {mol_id}: {str(e)}")

                lib_points = self._get_library_points(conn, mol_id)
                if lib_points:
                    self._plot_single(
                        points=lib_points,
                        title=f"Library Matches: {mol_name} (ID:{mol_id})",
                        filename=os.path.join(output_dir, f"lib_{mol_id}_matches.png")
                    )

                # 生成用户匹配图
                user_points = self._get_user_points(conn, mol_id)
                if user_points:
                    self._plot_single(
                        points=user_points,
                        title=f"User Matches: {mol_name} (ID:{mol_id})",
                        filename=os.path.join(output_dir, f"user_{mol_id}_matches.png")
                    )
    def _draw_molecule_structure(self, mol_block: str, filename: str, title: str = ""):
        """从mol_block绘制分子结构图"""
        from rdkit import Chem
        from rdkit.Chem import Draw

        # 转换mol block为分子对象
        mol = Chem.MolFromMolBlock(mol_block)
        if mol is None:
            raise ValueError("The mol block data is invalid")

        # 创建绘图
        img = Draw.MolToImage(mol, size=(600, 400), wedgeBonds=True)

        # 添加标题并保存
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis('off')
        if title:
            plt.title(title, fontsize=12, pad=10)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_library_matches(self, output_dir: str):
        """生成库分子匹配坐标图"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            for mol_id, mol_name, *_ in tqdm(self.top_results, desc="Library molecule plotting"):
                # 获取匹配的数据库坐标点
                points = []
                cursor = conn.execute("""
                    SELECT um.db_point, c.h_count 
                    FROM user_matches um
                    JOIN carbons c ON um.mol_id = c.mol_id AND um.c_index = c.c_index
                    WHERE um.mol_id = ? AND um.distance < 1e9
                """, (mol_id,))

                for row in cursor:
                    c_shift, h_shift = json.loads(row['db_point'])
                    points.append( (c_shift, h_shift, row['h_count']) )

                if points:
                    self._plot_single(
                        points=points,
                        title=f"Library Matches: {mol_name} (ID:{mol_id})",
                        filename=os.path.join(output_dir, f"lib_{mol_id}_matches.png")
                    )
    def _get_library_points(self, conn: sqlite3.Connection, mol_id: int):
        """获取库分子匹配点"""
        points = []
        cursor = conn.execute("""
            SELECT um.db_point, c.h_count 
            FROM user_matches um
            JOIN carbons c ON um.mol_id = c.mol_id AND um.c_index = c.c_index
            WHERE um.mol_id = ? AND um.distance < 1e9
        """, (mol_id,))

        for row in cursor:
            c_shift, h_shift = json.loads(row['db_point'])
            points.append((c_shift, h_shift, row['h_count']))
        return points


    def _get_user_points(self, conn: sqlite3.Connection, mol_id: int):
        """获取用户匹配点"""
        points = []
        cursor = conn.execute("""
            SELECT um.user_point, um.h_type 
            FROM user_matches um
            WHERE um.mol_id = ? AND um.distance < 1e9
        """, (mol_id,))

        for row in cursor:
            c, h = json.loads(row['user_point'])
            points.append((c, h, row['h_type']))
        return points
    def _plot_single(self, points: List[Tuple], title: str, filename: str):
        """绘制单个坐标图（使用椭圆表示固定ppm范围）"""
        plt.figure(figsize=(12, 8))
        ax = plt.gca()

        # 设置坐标轴（保持原有逻辑）
        ax.invert_yaxis()
        ax.set_xlabel("H Shift (ppm)", fontsize=12)
        ax.set_ylabel("C Shift (ppm)", fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        ax.grid(True, linestyle='--', alpha=0.5)

        # 定义同心椭圆参数（单位：ppm）
        ellipse_sizes = [
            (0.3, 0.5),   # 最大椭圆：H宽0.3ppm, C高0.5ppm
            (0.2, 0.33),  # 中等椭圆：H宽0.2ppm, C高0.33ppm
            (0.1, 0.17)   # 最小椭圆：H宽0.1ppm, C高0.17ppm
        ]

        # 绘制每个点
        for c, h, h_type in points:
            color = '#1f77b4' if (self.mode == 2 or self.mode == 3 and h_type == 2) else '#ff0000'

            for i, (h_radius, c_radius) in enumerate(ellipse_sizes):
                alpha = 0.9 - i*0.3
                ellipse = Ellipse(
                    xy=(h, c),
                    width=h_radius,  # H轴方向总宽度
                    height=c_radius,  # C轴方向总高度
                    angle=0,
                    edgecolor=color,
                    linewidth=1.2 + i*0.5,
                    linestyle='--' if i==2 else '-',
                    alpha=alpha,
                    fill=False
                )
                ax.add_patch(ellipse)

        # 自动调整坐标范围（新增逻辑）
        if points:
            all_h = [h for _, h, _ in points]
            all_c = [c for c, _, _ in points]

            # 计算边界（扩展最大椭圆尺寸）
            h_padding = max(ellipse_sizes[0][0]/2, 1.0)  # 至少1ppm
            c_padding = max(ellipse_sizes[0][1]/2, 1.0)

            ax.set_xlim(max(all_h) + h_padding, min(all_h) - h_padding)
            ax.set_ylim(max(all_c) + c_padding, min(all_c) - c_padding)

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_user_matches(self, output_dir: str):
        """修正用户坐标解析方式"""
        with sqlite3.connect(self.db_path) as conn:
            points = []
            cursor = conn.execute("""
                SELECT user_point, h_type 
                FROM user_matches 
                WHERE distance < 1e9
            """)

            for row in cursor:
                # 使用索引访问替代列名（因可能返回元组）
                c, h = json.loads(row[0])  # user_point在第一个位置
                points.append( (c, h, row[1]) )  # h_type在第二个位置

            if points:
                self._plot_single(
                    points=points,
                    title="User Matched Coordinates",
                    filename=os.path.join(output_dir, "user_matches.png")
                )

import tkinter as tk
from tkinter import ttk
from pathlib import Path
import sqlite3
from PIL import Image, ImageTk
import os

class CHResultViewer:
    def __init__(self, db_path, results, output_dir="ch_match_plots"):
        self.db_path = db_path
        self.results = results
        self.output_dir = output_dir
        self.current_page = 0
        self.c_tol, self.h_tol = self._load_tolerances()
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_tolerances(self):
        """读取最近一次 HSQC 运行的容差参数"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute("SELECT c_tol, h_tol FROM hsqc_run_params WHERE id = 1").fetchone()
                if row:
                    return float(row[0]), float(row[1])
        except Exception:
            pass
        return None, None

    def _is_in_tolerance(self, db_point, user_point):
        if self.c_tol is None or self.h_tol is None:
            return False
        if user_point == (0, 0):
            return False
        db_c, db_h = db_point
        user_c, user_h = user_point
        return abs(user_c - db_c) <= self.c_tol and abs(user_h - db_h) <= self.h_tol

    def _point_score(self, db_point, user_point):
        """按当前 HSQC 原始评分逻辑计算单点分数"""
        if self.c_tol is None or self.h_tol is None:
            return None
        if user_point == (0, 0):
            return 0.0

        db_c, db_h = db_point
        user_c, user_h = user_point

        in_c_range = (db_c - self.c_tol <= user_c <= db_c + self.c_tol)
        in_h_range = (db_h - self.h_tol <= user_h <= db_h + self.h_tol)
        if in_c_range and in_h_range:
            return 1.0

        delta_c = user_c - db_c
        delta_h = user_h - db_h

        if delta_c == 0:
            region = 'top' if delta_h > 0 else 'bottom'
        else:
            slope_ratio = abs(delta_h / delta_c)
            main_slope = self.h_tol / self.c_tol
            if slope_ratio > main_slope:
                region = 'top' if delta_h > 0 else 'bottom'
            else:
                region = 'right' if delta_c > 0 else 'left'

        if region in ('left', 'right'):
            denominator = abs(delta_c)
            score = self.c_tol / denominator if denominator != 0 else 0.0
        else:
            denominator = abs(delta_h)
            score = self.h_tol / denominator if denominator != 0 else 0.0

        return max(0.0, min(score, 1.0))

    def _count_in_tolerance(self, mol_id):
        if self.c_tol is None or self.h_tol is None:
            return None
        count = 0
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT db_point, user_point FROM user_matches WHERE mol_id = ? AND distance < 1e9", (mol_id,))
            for row in cursor:
                db_point = tuple(json.loads(row['db_point']))
                user_point = tuple(json.loads(row['user_point']))
                if self._is_in_tolerance(db_point, user_point):
                    count += 1
        return count

    def create_result_window(self, parent):
        """创建结果展示窗口"""
        self.top = tk.Toplevel(parent)
        self.top.title("Top CH matching result")
        self.top.geometry("1000x700")

        # 分页控制
        control_frame = ttk.Frame(self.top)
        ttk.Button(control_frame, text="◀ Previous page",
                   command=lambda: self.show_page(-1)).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Next page ▶",
                   command=lambda: self.show_page(1)).pack(side=tk.LEFT)
        self.page_label = ttk.Label(control_frame, text="", font=('Arial', 10))
        self.page_label.pack(side=tk.LEFT, padx=10)
        control_frame.pack(pady=10, fill=tk.X)

        # 结果网格布局
        self.grid_frame = ttk.Frame(self.top)
        self.grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 配置2x3网格
        for r in range(2):
            self.grid_frame.rowconfigure(r, weight=1)
        for c in range(3):
            self.grid_frame.columnconfigure(c, weight=1)

        self.show_page(0)

    def show_page(self, delta):
        """分页显示处理"""
        new_page = self.current_page + delta
        max_page = (len(self.results) - 1) // 6
        self.current_page = max(0, min(new_page, max_page))
        self.page_label.config(text=f"Page  {self.current_page+1}/{(len(self.results)+5)//6}")
        self.render_items()

    def render_items(self):
        """渲染当前页的6个结果项"""
        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        start = self.current_page * 6
        page_results = self.results[start:start+6]

        for idx, result in enumerate(page_results):
            row = idx // 3
            col = idx % 3
            item = self.create_result_item(result, start + idx)
            item.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

    def create_result_item(self, result, rank):
        """创建单个结果项"""
        mol_id, name, source, score, matched, total = result
        item_frame = ttk.Frame(self.grid_frame, relief="groove", borderwidth=1)

        # 显示结构图
        struct_path = f"{self.output_dir}/struct_{mol_id}.png"
        if os.path.exists(struct_path):
            img = Image.open(struct_path).resize((200, 150), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label = ttk.Label(item_frame, image=photo)
            label.image = photo
            label.pack(pady=2)

        # 显示信息
        info_frame = ttk.Frame(item_frame)
        ttk.Label(info_frame, text=f"Rank：{rank+1}", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"Match score: {float(score):.2f}").pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"Assigned points: {matched}/{total}").pack(anchor=tk.W)
        in_tol = self._count_in_tolerance(mol_id)
        if in_tol is not None:
            ttk.Label(info_frame, text=f"In-tolerance points: {in_tol}/{total}").pack(anchor=tk.W)

        # 详情按钮
        ttk.Button(info_frame, text="View details",
                   command=lambda mid=mol_id: self.show_detail(mid)).pack(pady=2)
        info_frame.pack(fill=tk.X)

        return item_frame

    def show_detail(self, mol_id):
        """显示详情窗口"""
        detail_win = tk.Toplevel(self.top)
        detail_win.title("CH matching details")
        detail_win.geometry("1200x800")
        detail_win.configure(bg='#2C3E50')

        # 获取分子名称
        with sqlite3.connect(self.db_path) as conn:
            mol_name = conn.execute(
                "SELECT name FROM molecules WHERE mol_id=?", (mol_id,)
            ).fetchone()[0]
            score_data = conn.execute(
                "SELECT score, matched_points, total_points FROM molecule_ch_scores WHERE mol_id=?",
                (mol_id,)
            ).fetchone()

        # 顶部信息栏
        header_frame = tk.Frame(detail_win, bg='#2C3E50')
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        in_tol = self._count_in_tolerance(mol_id)
        in_tol_text = f" | In-tolerance points: {in_tol}/{score_data[2]}" if in_tol is not None else ""
        tk.Label(header_frame,
                 text=f"ID: {mol_id} | Name: {mol_name} | Score: {float(score_data[0]):.2f} | Assigned points: {score_data[1]}/{score_data[2]}{in_tol_text}",
                 bg='#2C3E50', fg='white', font=('Arial', 12)).pack(side=tk.LEFT)

        # 主内容区域
        main_frame = tk.Frame(detail_win, bg='#2C3E50')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 上部可视化区域
        viz_frame = tk.Frame(main_frame, bg='#2C3E50', height=350)
        viz_frame.pack(fill=tk.BOTH, pady=5)
        viz_frame.pack_propagate(False)

        # 左侧结构图
        struct_frame = tk.LabelFrame(viz_frame, text="Molecular structure", bg='#2C3E50', fg='white')
        struct_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        struct_path = f"{self.output_dir}/struct_{mol_id}.png"
        if os.path.exists(struct_path):
            img = Image.open(struct_path).resize((400, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label = tk.Label(struct_frame, image=photo, bg='#2C3E50')
            label.image = photo
            label.pack(expand=True)

        # 右侧坐标图
        coord_frame = tk.LabelFrame(viz_frame, text="Coordinate matching", bg='#2C3E50', fg='white')
        coord_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        self.coord_img_type = tk.StringVar(value="lib")
        ttk.Radiobutton(coord_frame, text="Library matching", variable=self.coord_img_type,
                        value="lib", command=lambda: self.update_coord_img(mol_id)).pack(side=tk.LEFT)
        ttk.Radiobutton(coord_frame, text="User matching", variable=self.coord_img_type,
                        value="user", command=lambda: self.update_coord_img(mol_id)).pack(side=tk.LEFT)

        self.coord_label = tk.Label(coord_frame, bg='#2C3E50')
        self.coord_label.pack(expand=True)
        self.update_coord_img(mol_id)

        # 下部表格区域
        table_frame = tk.LabelFrame(main_frame, text="Matching point details", bg='#2C3E50', fg='white', height=350)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        table_frame.pack_propagate(False)

        # 创建表格
        columns = ("Type", "C Displacement (Library)", "H Displacement (Library)", "C Displacement (User)", "H Displacement (User)", "Distance", "Point score", "In tolerance")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # 加载表格数据
        self.load_match_data(mol_id)

    def update_coord_img(self, mol_id):
        """更新坐标图显示"""
        img_type = self.coord_img_type.get()
        img_path = f"{self.output_dir}/{img_type}_{mol_id}_matches.png"

        if os.path.exists(img_path):
            img = Image.open(img_path).resize((400, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.coord_label.config(image=photo)
            self.coord_label.image = photo

    def load_match_data(self, mol_id):
        """加载匹配点数据"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            matches = conn.execute("""
                SELECT um.h_type, um.db_point, um.user_point, um.distance
                FROM user_matches um
                WHERE um.mol_id = ? AND um.distance < 1e9
                ORDER BY um.distance
            """, (mol_id,))

            for row in matches:
                db_c, db_h = json.loads(row['db_point'])
                user_c, user_h = json.loads(row['user_point'])
                db_point = (db_c, db_h)
                user_point = (user_c, user_h)
                point_score = self._point_score(db_point, user_point)
                point_score_text = f"{point_score:.2f}" if point_score is not None else "N/A"
                in_tol_text = "Yes" if self._is_in_tolerance(db_point, user_point) else "No"

                self.tree.insert("", tk.END, values=(
                    self.get_type_name(row['h_type']),
                    f"{db_c:.2f}",
                    f"{db_h:.2f}",
                    f"{user_c:.2f}",
                    f"{user_h:.2f}",
                    f"{row['distance']:.3f}",
                    point_score_text,
                    in_tol_text
                ))

    def get_type_name(self, h_type):
        """获取H类型名称"""
        return {
            1: "CH",
            2: "CH2",
            3: "CH3"
        }.get(h_type, "Unknown")
