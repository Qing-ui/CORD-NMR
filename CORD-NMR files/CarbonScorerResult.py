import sqlite3
import io
from typing import List, Tuple, Dict,Union

from PIL.ImagePath import Path
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem  # 新增导入
from PROCESSSDFFILES import SDFBatchProcessor
from CarbonScoreProcess import *

class CarbonOnlyScorerGUI:
    """仅处理碳评分的GUI集成类"""

    def __init__(
            self,
            sdf_files: List[str],
            # 碳匹配参数
            c_mode: str = 'typed',
            C_merge: str = 'N',
            c_data: Union[dict, list] = None,
            c_range: Tuple[int, int] = (10, 30),
            fw_range: Tuple[float, float] = (100.0, 300.0),
            # 评分参数
            score_mode: str = 'global',
            global_thresholds: Tuple[float, float] = (0.5, 2.0),
            fine_ranges: List[Tuple[Tuple[float, float], float, float]] = None,
            # 环境参数
            env_level: int = 2,
            self_weight: float = 0.6,
            env_weight: float = 0.4,
            db_path: str = "chem_data.db"
    ):
        """
        :param sdf_files: SDF文件路径列表
        :param c_mode: 碳匹配模式 ('typed'/'untyped')
        :param c_data: 碳匹配输入数据
        :param c_range: 碳数过滤范围
        :param fw_range: 分子量过滤范围
        :param score_mode: 评分模式 ('global'/'fine')
        :param global_thresholds: 全局阈值 (green, yellow)
        :param fine_ranges: 精细模式范围
        :param env_level: 环境层级 (1-3)
        :param self_weight: 自身评分权重
        :param env_weight: 环境评分权重
        """
        # 初始化数据库
        self.db_path = db_path
        processor = SDFBatchProcessor(db_path)
        processor.process_files(sdf_files)

        # 参数存储
        self.c_mode = c_mode
        self.C_merge = C_merge
        self.c_data = c_data or self._default_c_data()
        self.c_range = c_range
        self.fw_range = fw_range
        self.score_mode = score_mode
        self.global_thresholds = global_thresholds
        self.fine_ranges = fine_ranges or [
            ((0, 50), 1.0, 2.0),
            ((50, 150), 2.0, 3.0),
            ((150, 300), 3.0, 4.0)
        ]
        self.env_level = env_level
        self.self_weight = self_weight
        self.env_weight = env_weight

        self._validate_parameters()

    def _default_c_data(self):
        """生成默认C数据结构"""
        return {0: [], 1: [], 2: [], 3: []} if self.c_mode == 'typed' else []

    def _validate_parameters(self):
        """综合参数校验"""
        # 范围校验
        if self.c_range[0] > self.c_range[1]:
            raise ValueError("The carbon number range is invalid")
        if self.fw_range[0] > self.fw_range[1]:
            raise ValueError("The molecular weight range is invalid")

        # 评分模式校验
        if self.score_mode not in ('global', 'fine'):
            raise ValueError("The scoring mode must be global/fine")
        if self.score_mode == 'global' and len(self.global_thresholds) != 2:
            raise ValueError("The global mode requires two thresholds")
        if self.score_mode == 'fine' and not self.fine_ranges:
            raise ValueError("Granular mode requires a list of ranges")

        # 环境参数
        if self.env_level not in {1, 2, 3}:
            raise ValueError("The environment level must be 1-3")
        if abs(self.self_weight + self.env_weight - 1.0) > 1e-6:
            raise ValueError("The sum of environmental weights must be 1")

    def execute(self) -> List[Tuple]:
        """执行碳评分流程"""
        # 碳原子匹配
        matcher = GreedyMatcher(
            db_path=self.db_path,
            user_data=self.c_data,
            c_range=self.c_range,
            fw_range=self.fw_range,
            mode=self.c_mode,
            C_merge=self.C_merge
        )

        # 将 GUI 中用户输入的“第一容差/绿色阈值”真正传给匹配器，
        # 让匹配阶段按满分优先策略工作，而不是只在评分阶段才使用。
        if self.score_mode == 'global':
            matcher.match_green_threshold = float(self.global_thresholds[0])
            matcher.match_yellow_threshold = float(self.global_thresholds[1])
        elif self.score_mode == 'fine':
            matcher.match_fine_ranges = self.fine_ranges

        matcher.run()

        # 碳原子评分
        scorer = CarbonScorer(self.db_path)
        scorer.create_results_table()
        scorer.process_scoring(
            mode=self.score_mode,
            global_thresholds=self.global_thresholds,
            fine_ranges=self.fine_ranges
        )

        # 环境评分
        env_scorer = RobustMoleculeScorer(self.db_path)
        env_scorer.create_result_table()
        env_scorer.process_molecules(
            self_weight=self.self_weight,
            env_weight=self.env_weight,
            env_level=self.env_level
        )

        return self._get_results()

    def _get_results(self, top_n: int = 20) -> List[Tuple]:
        """获取评分结果（修改后逻辑：评分相同时C数多的靠前）"""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("""
                SELECT m.mol_id, m.name, m.source_file,
                       ms.final_score AS env_score,
                       AVG(cs.score) AS carbon_score
                FROM molecule_scores ms
                JOIN molecules m ON ms.mol_id = m.mol_id
                JOIN carbon_scores cs ON ms.mol_id = cs.mol_id
                GROUP BY ms.mol_id
                ORDER BY 
                    ms.final_score DESC,  -- 主排序：环境评分降序
                    m.carbon_count DESC   -- 次排序：碳原子数降序（评分相同时C数多的靠前）
                LIMIT ?
            """, (top_n,)).fetchall()


class CarbonResultVisualizer:
    """Carbon scoring result visualization handler (修正数据库连接问题版本)"""

    def __init__(self, db_path: str, top_results: List[Tuple]):
        self.db_path = db_path
        self.top_results = top_results
        self.h_count_intensity = {3: 10000, 2: 8000, 1: 6000, 0: 4000}

    def generate_plots(self, output_dir: str = "plots"):
        import os
        os.makedirs(output_dir, exist_ok=True)

        for result in self.top_results:
            mol_id = result[0]
            mol_name = result[1]


            with sqlite3.connect(self.db_path) as conn:
                carbons = self._get_carbon_data(conn, mol_id)
                mol_block = self._get_mol_block(conn, mol_id)

            if mol_block and carbons:
                mol = Chem.MolFromMolBlock(mol_block)
                if mol:
                    self._plot_structure(mol, carbons, mol_id, mol_name, output_dir)
                    self._plot_shift_intensity(carbons, mol_id, mol_name, output_dir)

    def _get_mol_block(self, conn: sqlite3.Connection, mol_id: int) -> str:
        """从数据库获取分子结构"""
        try:
            return conn.execute(
                "SELECT mol_block FROM molecules WHERE mol_id=?",
                (mol_id,)
            ).fetchone()[0]
        except Exception as e:
            print(f"Error loading molecule {mol_id}: {str(e)}")
            return None
    def _get_carbon_data(self, conn: sqlite3.Connection, mol_id: int) -> List[Dict]:
        """获取带颜色信息的碳原子数据（修正查询语句）"""
        try:
            return [
                {
                    'index': row[0],
                    'color': row[1],
                    'db_shift': row[2] if row[2] is not None else 0,
                    'user_shift': row[3] if row[3] is not None else 0,
                    'h_count': row[4]
                }
                for row in conn.execute("""
                    SELECT cs.c_index, cs.color, cs.db_shift, cs.user_shift, c.h_count 
                    FROM carbon_scores cs
                    JOIN carbons c ON cs.mol_id = c.mol_id AND cs.c_index = c.c_index
                    WHERE cs.mol_id = ?
                """, (mol_id,))
            ]
        except sqlite3.Error as e:
            print(f"The database query failed: {str(e)}")
            return []

    def _plot_structure(self, mol: Chem.Mol, carbons: List[Dict],
                        mol_id: int, name: str, output_dir: str):

        highlight_atoms = [c['index'] for c in carbons]
        highlight_colors = [self._color_to_rgb(c['color']) for c in carbons]

        # 1. 原子索引图
        self._draw_labeled_molecule(
            mol=mol,
            highlight_atoms=highlight_atoms,
            highlight_colors=highlight_colors,
            labels={c['index']: str(c['index']) for c in carbons},
            filename=f"{output_dir}/{name}_indices.png"
        )

        # 2. 数据库位移图
        self._draw_labeled_molecule(
            mol=mol,
            highlight_atoms=highlight_atoms,
            highlight_colors=highlight_colors,
            labels={c['index']: f"{c['db_shift']:.1f}" for c in carbons},
            filename=f"{output_dir}/{name}_db_shifts.png"
        )

        # 3. 用户位移图
        self._draw_labeled_molecule(
            mol=mol,
            highlight_atoms=highlight_atoms,
            highlight_colors=highlight_colors,
            labels={c['index']: f"{c['user_shift']:.1f}"for c in carbons},
            filename=f"{output_dir}/{name}_user_shifts.png"
        )

    def _draw_labeled_molecule(self, mol: Chem.Mol,
                               highlight_atoms: List[int],
                               highlight_colors: List[tuple],
                               labels: Dict[int, str],
                               filename: str):
        """优化后的分子结构绘制方法（使用正确坐标类型）"""
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


        from rdkit import Geometry
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
    def _plot_shift_intensity(self, carbons: List[Dict],
                              mol_id: int, name: str, output_dir: str):
        """Generate shift-intensity comparison plot"""
        plt.figure(figsize=(12, 6))
        for c in carbons:
            intensity = self.h_count_intensity.get(c['h_count'], 0)
            plt.vlines(c['db_shift'], 0, intensity, colors='red', linewidth=2, alpha=0.7)
            plt.vlines(c['user_shift'], 0, intensity, colors='green', linewidth=2, alpha=0.7)

        plt.xlabel('Chemical Shift (ppm)')
        plt.ylabel('Intensity')
        plt.title(f"{name}\nShift-Intensity Comparison")
        plt.gca().invert_xaxis()
        plt.grid(True)
        plt.savefig(f"{output_dir}/{name}_shift_intensity.png")
        plt.close()

    def _color_to_rgb(self, color_str: str) -> tuple:
        """将HEX颜色转换为归一化RGB元组"""
        if color_str.startswith('#'):
            return (
                int(color_str[1:3], 16)/255.0,
                int(color_str[3:5], 16)/255.0,
                int(color_str[5:7], 16)/255.0
            )
        return (0.8, 0.8, 0.8)  # 默认灰色

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os

from PIL import ImageTk
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import sqlite3
from PIL import Image, ImageTk
import os

class ResultViewer:
    def __init__(self, db_path, results, output_dir="results"):
        self.db_path = db_path
        self.results = results
        self.output_dir = output_dir
        self.current_page = 0
        self.current_mol_name = ""
        self.struct_img_type = tk.StringVar(value="indices")  # 初始化图片类型变量
        os.makedirs(self.output_dir, exist_ok=True)


    def create_result_window(self, parent):
        """创建结果展示窗口"""
        self.top = tk.Toplevel(parent)
        self.top.title("Top match result")
        self.top.geometry("800x650")

        # 分页控制栏
        control_frame = ttk.Frame(self.top)
        ttk.Button(control_frame, text="◀ Previous",
                   command=lambda: self.show_page(-1)).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Next ▶",
                   command=lambda: self.show_page(1)).pack(side=tk.LEFT)
        self.page_label = ttk.Label(control_frame, text="", font=('Arial', 10))
        self.page_label.pack(side=tk.LEFT, padx=10)
        control_frame.pack(pady=5, fill=tk.X)

        # 结果展示网格（2行x3列）
        self.grid_frame = ttk.Frame(self.top)
        self.grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 配置网格行列权重
        for r in range(2):
            self.grid_frame.rowconfigure(r, weight=1)
        for c in range(3):
            self.grid_frame.columnconfigure(c, weight=1)

        self.show_page(0)  # 初始显示第一页

    def show_page(self, delta):
        """分页显示处理"""
        new_page = self.current_page + delta
        max_page = (len(self.results) - 1) // 6
        self.current_page = max(0, min(new_page, max_page))
        self.page_label.config(
            text=f"Page {self.current_page+1}/{(len(self.results)+5)//6} "
        )
        self.render_items()

    def render_items(self):
        """渲染当前页的6个结果项"""
        # 清空当前内容
        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        # 获取当前页数据
        start = self.current_page * 6
        page_results = self.results[start:start+6]

        # 在2x3网格中排列结果项
        for idx, result in enumerate(page_results):
            row = idx // 3  # 0-1行
            col = idx % 3   # 0-2列
            item = self.create_result_item(result, start + idx)
            item.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

    def create_result_item(self, result, rank):
        """创建单个结果项组件"""
        mol_id, name, source, env_score, carbon_score = result
        item_frame = ttk.Frame(self.grid_frame, relief="groove", borderwidth=1)

        # 图片展示（200x200）
        img_path = f"molecule_plots/{name}_indices.png"
        if os.path.exists(img_path):
            img = Image.open(img_path).resize((200, 200), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label = ttk.Label(item_frame, image=photo)
            label.image = photo
            label.pack(pady=2)

        # 信息展示
        info_frame = ttk.Frame(item_frame)
        ttk.Label(info_frame, text=f"Rank：{rank+1}",
                  font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        ttk.Label(info_frame,
                  text=f"synthesis: {float(env_score):.2f}",
                  font=('Arial', 9)).pack(anchor=tk.W)

        # 详情按钮
        ttk.Button(info_frame, text="Details",
                   command=lambda mid=mol_id: self.show_detail(mid),
                   style='TButton').pack(pady=2)
        info_frame.pack(fill=tk.X)

        return item_frame


    def show_detail(self, mol_id):
        """修复struct_img_type未定义问题"""
        detail_win = tk.Toplevel(self.top)
        detail_win.title("Molecule details")
        detail_win.geometry("1200x800")
        detail_win.configure(bg='#2C3E50')

        # 初始化结构图类型选择控件
        self.struct_img_type = tk.StringVar(value="indices")

        # 顶部信息栏
        # 顶部信息栏
        header_frame = tk.Frame(detail_win, bg='#2C3E50')
        header_frame.pack(fill=tk.X, padx=10, pady=10)

        with sqlite3.connect(self.db_path) as conn:
            # 获取分子数据
            mol_data = conn.execute("""
                SELECT m.name, m.mol_block, ms.final_score, 
                       (SELECT AVG(score) FROM carbon_scores WHERE mol_id=m.mol_id)
                FROM molecules m
                JOIN molecule_scores ms ON m.mol_id=ms.mol_id
                WHERE m.mol_id=?""", (mol_id,)).fetchone()

            # 获取碳原子评分数据
            carbons = conn.execute("""
                SELECT c_index, db_shift, user_shift, score 
                FROM carbon_scores 
                WHERE mol_id=? ORDER BY c_index""", (mol_id,)).fetchall()

        # 显示分子信息
        info_text = (
            f"MATCH ID: {mol_id} | "
            f"Molecule ID: {mol_data[0]} | "
            f"Overall Score: {float(mol_data[2]):.4f} | "
            f"Carbon equalization: {float(mol_data[3]):.4f}"
        )
        tk.Label(header_frame, text=info_text, font=('Arial', 12),
                 bg='#2C3E50', fg='white').pack(side=tk.LEFT)

        # 主内容区域
        main_frame = tk.Frame(detail_win, bg='#2C3E50')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 上部可视化区域
        viz_frame = tk.Frame(main_frame, bg='#2C3E50', height=450)
        viz_frame.pack(fill=tk.X, pady=5)
        viz_frame.pack_propagate(False)

        # 左侧结构图区域
        struct_frame = tk.LabelFrame(viz_frame, text="结构视图", bg='#2C3E50', fg='white')
        struct_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # 结构图类型选择单选框
        radio_frame = tk.Frame(struct_frame, bg='#2C3E50')
        radio_frame.pack(fill=tk.X, pady=5)

        for text, img_type in [('Atomic indexes', 'indices'),
                               ('DB SHIFT', 'db_shifts'),
                               ('USER SHIFT', 'user_shifts')]:
            tk.Radiobutton(radio_frame, text=text, variable=self.struct_img_type,
                           value=img_type, command=lambda: self.update_struct_img(),
                           bg='#2C3E50', fg='white', selectcolor='#34495E').pack(side=tk.LEFT)

        # 结构图显示
        self.struct_label = tk.Label(struct_frame, bg='#2C3E50')
        self.struct_label.pack(expand=True)

        # 右侧位移对比区域
        shift_frame = tk.LabelFrame(viz_frame, text="SHIFT comparison", bg='#2C3E50', fg='white')
        shift_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        self.shift_label = tk.Label(shift_frame, bg='#2C3E50')
        self.shift_label.pack(expand=True)

        # 下部表格区域
        table_frame = tk.LabelFrame(main_frame, text="Carbon Atom Shift Matching List",
                                    bg='#2C3E50', fg='white', height=350)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        table_frame.pack_propagate(False)

        # 创建表格
        columns = ("C_INDEX", "DB_SHIFT", "USER_SHIFT", "ABS", "Match score")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=12)

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # 加载数据
        self.load_molecule_data(mol_id)

    def load_molecule_data(self, mol_id):
        """加载分子数据"""
        with sqlite3.connect(self.db_path) as conn:
            # 获取分子名称
            self.current_mol_name = conn.execute(
                "SELECT name FROM molecules WHERE mol_id=?", (mol_id,)
            ).fetchone()[0]

            # 更新结构图
            self.update_struct_img()

            # 加载位移对比图
            shift_path = f"molecule_plots/{self.current_mol_name}_shift_intensity.png"
            if os.path.exists(shift_path):
                img = Image.open(shift_path).resize((450,350), Image.LANCZOS)
                self.shift_photo = ImageTk.PhotoImage(img)
                self.shift_label.config(image=self.shift_photo)
                self.shift_label.image = self.shift_photo

            # 加载表格数据
            carbons = conn.execute("""
                SELECT c_index, db_shift, user_shift, score 
                FROM carbon_scores 
                WHERE mol_id=? ORDER BY c_index""", (mol_id,)).fetchall()

            for carbon in carbons:
                db_shift = carbon[1] if carbon[1] is not None else 0
                user_shift = carbon[2] if carbon[2] is not None else 0
                delta = abs(db_shift - user_shift)
                self.tree.insert("", tk.END, values=(
                    carbon[0],
                    f"{carbon[1]:.2f}" if carbon[1] is not None else "N/A",
                    f"{carbon[2]:.2f}" if carbon[2] is not None else "N/A",
                    f"{delta:.2f}",
                    f"{carbon[3]:.2f}" if carbon[3] is not None else "0"
                ))

    def update_struct_img(self):
        """更新结构图显示"""
        if not hasattr(self, 'current_mol_name'):
            return

        img_type = self.struct_img_type.get()
        img_path = f"molecule_plots/{self.current_mol_name}_{img_type}.png"

        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).resize((350,350), Image.LANCZOS)
                self.struct_photo = ImageTk.PhotoImage(img)
                self.struct_label.config(image=self.struct_photo)
                self.struct_label.image = self.struct_photo
            except Exception as e:
                print(f"Error loading image: {str(e)}")

    # After getting results from CarbonOnlyScorerGUI
# if __name__ == "__main__":
#     config = {
#         'sdf_files': [r'E:\Lufeixuecheng\pythonProject\combNAP\NAP23741.sdf'],
#         'c_mode': 'typed',
#         'c_data': {
#             0: [205.0, 199.9, 205.0, 199.9, 205.0, 199.9],
#             1: [128.5, 155.0, 128.5, 155.0, 128.5, 155.0, 128.5, 155.0],
#             2: [95.8, 124.7, 95.8, 124.7, 95.8, 124.7, 95.8, 124.7],
#             3: [21.3, 21.0, 21.3, 21.0, 21.3, 21.0, 21.3, 21.0,21.3, 21.0, 21.3, 21.0, 21.3, 21.0, 21.3, 21.0]
#         },
#         'c_range': (1, 40),
#         'fw_range': (10.0, 3500.0),
#         'score_mode': 'global',
#         'global_thresholds': (3, 5),
#         'env_level': 2,
#         'self_weight': 0.7,
#         'env_weight': 0.3
#     }
#
#     pipeline = CarbonOnlyScorerGUI(**config)
#     results = pipeline.execute()
#     print("Top 碳评分结果:")
#     for idx, row in enumerate(results, 1):
#         print(f"{idx}. {row[1]} ({row[2]})")
#         print(f"   综合评分: {row[3]:.2f} | 碳原子均分: {row[4]:.2f}\n")
#
#     top_results = results
#     # Generate visualizations
#     visualizer = CarbonResultVisualizer(
#         db_path="chem_data.db",
#         top_results=top_results
#     )
#     visualizer.generate_plots(output_dir="molecule_plots")
#
# print("Visualizations generated in 'molecule_plots' directory")
