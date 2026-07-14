from typing import Union

class CarbonCoordinateExtractor:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        """创建坐标存储表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS carbon_h_coordinates")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS carbon_h_coordinates (
                    coord_id INTEGER PRIMARY KEY,
                    mol_id INTEGER,
                    c_index INTEGER,
                    h_count INTEGER,
                    point_count INTEGER,
                    coordinates TEXT,
                    FOREIGN KEY(mol_id) REFERENCES molecules(mol_id)
                )
            """)
            conn.commit()

    def _process_carbon(self, c_data: dict) -> Union[None, dict]:
        """处理单个碳原子数据"""
        h_count = c_data['h_count']
        c_shift = c_data['c_shift']
        h_shifts = json.loads(c_data['h_shifts'])

        # 基础校验
        if h_count == 0 or c_shift is None:
            return None
        if len(h_shifts) != h_count:
            print(f"Carbon atom {c_data['c_index']} hydrogen displacement data mismatch, expected {h_count}, actual {len(h_shifts)}")
            return None

        points = []
        try:
            if h_count == 1:
                if h_shifts[0] is not None:
                    points.append((c_shift, h_shifts[0]))

            elif h_count == 2:
                for hs in h_shifts:
                    if hs is not None:
                        points.append((c_shift, hs))

            elif h_count == 3:
                valid_shifts = [round(hs, 2) for hs in h_shifts if hs is not None]
                if valid_shifts:
                    avg = round(sum(valid_shifts) / len(valid_shifts), 2)  # 修正点
                    points.append((round(c_shift, 2), avg))

            else:  # 处理异常情况
                print(f"Unconventional hydrogen number {h_count} @ mol:{c_data['mol_id']}-c:{c_data['c_index']}")
                return None

        except TypeError as e:
            print(f"Data Processing Errors: {str(e)}")
            return None

        return {
            'mol_id': c_data['mol_id'],
            'c_index': c_data['c_index'],
            'h_count': h_count,
            'points': points,
            'count': len(points)
        }

    def process_all(self):
        """批量处理所有分子（修正进度显示）"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # 获取总碳原子数（h_count>0且c_shift存在）
            total_query = """
                SELECT COUNT(*) 
                FROM carbons 
                WHERE h_count > 0 
                  AND c_shift IS NOT NULL
            """
            total = conn.execute(total_query).fetchone()[0]

            # 获取数据游标
            cursor = conn.execute("""
                SELECT c.mol_id, c.c_index, c.h_count, c.c_shift, c.h_shifts
                FROM carbons c
                WHERE c.h_count > 0
                  AND c.c_shift IS NOT NULL
                ORDER BY c.mol_id, c.c_index
            """)

            batch = []
            with tqdm(total=total, desc="Dealing with carbon atoms", unit="carbon") as pbar:
                for row in cursor:
                    # 处理每个碳原子（无论是否生成坐标点）
                    result = self._process_carbon(dict(row))
                    if result and result['count'] > 0:
                        batch.append((
                            result['mol_id'],
                            result['c_index'],
                            result['h_count'],
                            result['count'],
                            json.dumps(result['points'])
                        ))

                    # 更新进度（每个碳原子计数+1）
                    pbar.update(1)

                    # 批量插入
                    if len(batch) >= 500:
                        conn.executemany("""
                            INSERT INTO carbon_h_coordinates
                            (mol_id, c_index, h_count, point_count, coordinates)
                            VALUES (?, ?, ?, ?, ?)
                        """, batch)
                        batch = []

                # 插入剩余数据
                if batch:
                    conn.executemany("""
                        INSERT INTO carbon_h_coordinates
                        (mol_id, c_index, h_count, point_count, coordinates)
                        VALUES (?, ?, ?, ?, ?)
                    """, batch)
                    conn.commit()

            # 获取实际插入的记录数
            inserted = conn.execute("SELECT COUNT(*) FROM carbon_h_coordinates").fetchone()[0]
            print(f"Processing is complete! Scans {total} carbon atoms to generate {inserted} valid coordinate records")

# 使用示例
# if __name__ == "__main__":
#     extractor = CarbonCoordinateExtractor("chem_data.db")
#     extractor.process_all()
# import sqlite3
#
# conn = sqlite3.connect('chem_data.db')
# cursor = conn.cursor()

# # 查询所有数据
# cursor.execute("SELECT * FROM carbon_h_coordinates")
# rows = cursor.fetchall()
# for row in rows:
#     print(row)

import sqlite3
import json
from tqdm import tqdm

class UserPointMatcher:
    def __init__(self, db_path: str, user_points: dict, mode: int = 1,
                 c_tolerance: float = None, h_tolerance: float = None, CH_merge='N'):
        self.db_path = db_path
        self.user_points = user_points
        self.mode = mode
        self.c_tol = c_tolerance
        self.h_tol = h_tolerance
        self.CH_merge = CH_merge
        self._validate_input()
        self._create_result_table()

    def _validate_input(self):
        if self.mode == 1:
            if 'All_type' not in self.user_points:
                raise ValueError("Mode 1 requires All_type key")
        elif self.mode == 2:
            if not ('type13' in self.user_points and 'type2' in self.user_points):
                raise ValueError("Mode 2 requires the type13 and type2 keys")
        elif self.mode == 3:
            if not all(k in self.user_points for k in ['type1', 'type2', 'type3']):
                raise ValueError("Mode 3 requires the type1/type2/type3 keys")
        else:
            raise ValueError("Invalid Mode Selection (1-3)")

    def _create_result_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS user_matches")
            conn.execute("""
                CREATE TABLE user_matches (
                    match_id INTEGER PRIMARY KEY,
                    mol_id INTEGER,
                    c_index INTEGER,
                    db_point TEXT,
                    user_point TEXT,
                    distance REAL,
                    h_type INTEGER,
                    FOREIGN KEY(mol_id) REFERENCES molecules(mol_id))
            """)
            conn.execute("CREATE INDEX idx_match_h ON user_matches(h_type)")
            conn.commit()

    def _point_distance(self, db_point, user_point):
        db_c, db_h = db_point
        user_c, user_h = user_point
        delta_h = (db_h - user_h) * 10
        delta_c = db_c - user_c
        return float((delta_c ** 2 + delta_h ** 2) ** 0.5)

    def _point_score_for_assignment(self, db_point, user_point):
        if self.c_tol is None or self.h_tol is None:
            return None

        db_c, db_h = db_point
        user_c, user_h = user_point
        delta_c = user_c - db_c
        delta_h = user_h - db_h

        if abs(delta_c) <= self.c_tol and abs(delta_h) <= self.h_tol:
            return 1.0

        c_ratio = abs(delta_c) / self.c_tol if self.c_tol else float('inf')
        h_ratio = abs(delta_h) / self.h_tol if self.h_tol else float('inf')
        worst_ratio = max(c_ratio, h_ratio)
        if worst_ratio <= 0:
            return 1.0
        return max(0.0, min(1.0, 1.0 / worst_ratio))

    def _in_tolerance(self, db_point, user_point):
        if self.c_tol is None or self.h_tol is None:
            return False
        db_c, db_h = db_point
        user_c, user_h = user_point
        return abs(db_c - user_c) <= self.c_tol and abs(db_h - user_h) <= self.h_tol

    def _build_cost_matrix(self, db_coords, user_points):
        import numpy as np
        n_rows = len(db_coords)
        n_cols = len(user_points)
        if n_rows == 0:
            return np.zeros((0, 0), dtype=np.float64)

        SCORE_SCALE = 10000.0
        DUMMY = SCORE_SCALE
        total_cols = n_cols + n_rows
        cost = np.full((n_rows, total_cols), DUMMY, dtype=np.float64)

        for i, db_point in enumerate(db_coords):
            for j, user_point in enumerate(user_points):
                dist = self._point_distance(db_point, user_point)
                if self.c_tol is not None and self.h_tol is not None:
                    point_score = self._point_score_for_assignment(db_point, user_point)
                    cost[i, j] = (1.0 - point_score) * SCORE_SCALE + dist * 1e-3
                else:
                    cost[i, j] = dist
        return cost

    def _solve_assignment(self, db_coords, user_points):
        n_rows = len(db_coords)
        n_cols = len(user_points)
        if n_rows == 0:
            return {}
        if n_cols == 0:
            return {i: None for i in range(n_rows)}

        cost = self._build_cost_matrix(db_coords, user_points)

        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)
            assigned = {}
            for r, c in zip(row_ind, col_ind):
                assigned[r] = c if c < n_cols else None
            for i in range(n_rows):
                assigned.setdefault(i, None)
            return assigned
        except Exception:
            pairs = []
            for i, db_point in enumerate(db_coords):
                for j, user_point in enumerate(user_points):
                    dist = self._point_distance(db_point, user_point)
                    score = self._point_score_for_assignment(db_point, user_point)
                    if score is None:
                        pairs.append((0.0, dist, i, j))
                    else:
                        pairs.append((-score, dist, i, j))
            pairs.sort(key=lambda x: (x[0], x[1]))
            assigned = {}
            used_rows = set()
            used_cols = set()
            for _, _, i, j in pairs:
                if i in used_rows or j in used_cols:
                    continue
                assigned[i] = j
                used_rows.add(i)
                used_cols.add(j)
            for i in range(n_rows):
                assigned.setdefault(i, None)
            return assigned

    def _process_single_molecule(self, mol_id, db_points, user_points, h_type=None):
        if self.CH_merge == 'N':
            matches = []
            db_coords = [p[2] for p in db_points]
            assigned = self._solve_assignment(db_coords, user_points)
            used_db = set()

            for db_idx, user_idx in assigned.items():
                db_entry = db_points[db_idx]
                if user_idx is None:
                    continue
                dist = round(self._point_distance(db_entry[2], user_points[user_idx]), 2)
                matches.append((
                    mol_id,
                    db_entry[0],
                    json.dumps(db_entry[2]),
                    json.dumps(user_points[user_idx]),
                    dist,
                    h_type if h_type else db_entry[1]
                ))
                used_db.add(db_idx)

            for idx, db_entry in enumerate(db_points):
                if idx not in used_db:
                    matches.append((
                        mol_id,
                        db_entry[0],
                        json.dumps(db_entry[2]),
                        json.dumps((0, 0)),
                        float('inf'),
                        h_type if h_type else db_entry[1]
                    ))
            return matches

        if self.CH_merge == 'Y':
            coord_groups = {}
            for idx, entry in enumerate(db_points):
                coord_key = tuple(entry[2])
                coord_groups.setdefault(coord_key, []).append((idx, entry))

            group_coords = sorted(coord_groups.keys())
            assigned = self._solve_assignment(group_coords, user_points)

            matches = []
            matched_coords = set()
            for group_idx, user_idx in assigned.items():
                coord = group_coords[group_idx]
                if user_idx is None:
                    continue
                dist = round(self._point_distance(coord, user_points[user_idx]), 2)
                for idx, entry in coord_groups[coord]:
                    c_index, h_type_val, _ = entry
                    matches.append((
                        mol_id,
                        c_index,
                        json.dumps(coord),
                        json.dumps(user_points[user_idx]),
                        dist,
                        h_type if h_type else h_type_val
                    ))
                matched_coords.add(coord)

            for coord, group in coord_groups.items():
                if coord in matched_coords:
                    continue
                for idx, entry in group:
                    c_index, h_type_val, _ = entry
                    matches.append((
                        mol_id,
                        c_index,
                        json.dumps(coord),
                        json.dumps((0, 0)),
                        float('inf'),
                        h_type if h_type else h_type_val
                    ))
            return matches

    def _match_mode1(self):
        with sqlite3.connect(self.db_path) as conn:
            molecules = {}
            for row in conn.execute("SELECT mol_id, c_index, h_count, coordinates FROM carbon_h_coordinates"):
                points = json.loads(row[3])
                if row[0] not in molecules:
                    molecules[row[0]] = []
                for p in points:
                    molecules[row[0]].append((row[1], row[2], p))

        all_matches = []
        user_pts = [tuple(p) for p in self.user_points['All_type']]
        for mol_id in sorted(molecules.keys()):
            db_points = molecules[mol_id]
            matches = self._process_single_molecule(mol_id=mol_id, db_points=db_points, user_points=user_pts)
            all_matches += matches
        return all_matches

    def _match_mode2(self):
        with sqlite3.connect(self.db_path) as conn:
            molecules = {}
            for row in conn.execute("""
                SELECT mol_id, c_index, h_count, coordinates 
                FROM carbon_h_coordinates 
                WHERE h_count IN (1, 3)"""):
                points = json.loads(row[3])
                mol_id = row[0]
                if mol_id not in molecules:
                    molecules[mol_id] = []
                for p in points:
                    molecules[mol_id].append((row[1], row[2], p))

        all_matches = []
        user_pts = [tuple(p) for p in self.user_points['type13']]
        for mol_id in sorted(molecules.keys()):
            db_points = molecules[mol_id]
            matches = self._process_single_molecule(mol_id=mol_id, db_points=db_points, user_points=user_pts)
            all_matches += matches

        all_matches += self._match_by_type(h_type=2, user_points=self.user_points['type2'])
        return all_matches

    def _match_by_type(self, h_type, user_points):
        with sqlite3.connect(self.db_path) as conn:
            molecules = {}
            for row in conn.execute("""
                SELECT mol_id, c_index, coordinates 
                FROM carbon_h_coordinates 
                WHERE h_count = ?""", (h_type,)):
                points = json.loads(row[2])
                if row[0] not in molecules:
                    molecules[row[0]] = []
                for p in points:
                    molecules[row[0]].append((row[1], p))

        all_matches = []
        user_pts = [tuple(p) for p in user_points]
        for mol_id in sorted(molecules.keys()):
            db_points = [(p[0], h_type, p[1]) for p in molecules[mol_id]]
            matches = self._process_single_molecule(mol_id=mol_id, db_points=db_points, user_points=user_pts, h_type=h_type)
            all_matches += matches
        return all_matches

    def process(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM user_matches")
            matches = []

            if self.mode == 1:
                matches = self._match_mode1()
            elif self.mode == 2:
                matches = self._match_mode2()
            elif self.mode == 3:
                matches += self._match_by_type(h_type=1, user_points=self.user_points['type1'])
                matches += self._match_by_type(h_type=2, user_points=self.user_points['type2'])
                matches += self._match_by_type(h_type=3, user_points=self.user_points['type3'])

            batch_size = 500
            with tqdm(total=len(matches), desc="Save the matching results") as pbar:
                for i in range(0, len(matches), batch_size):
                    batch = matches[i:i + batch_size]
                    conn.executemany("""
                        INSERT INTO user_matches 
                        (mol_id, c_index, db_point, user_point, distance, h_type)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, batch)
                    pbar.update(len(batch))

            matched = conn.execute("SELECT COUNT(*) FROM user_matches WHERE distance < 1e9").fetchone()[0]
            print(f"Match complete! Valid matches {matched} and placeholder {len(matches) - matched} records")

class MoleculeScoreCalculator:
    """分子评分计算器，处理CH坐标匹配结果的评分"""

    def __init__(self, db_path: str, c_tolerance: float, h_tolerance: float):
        """
        :param db_path: 数据库路径
        :param c_tolerance: 碳位移允许误差（ppm）
        :param h_tolerance: 氢位移允许误差（ppm，未放大前）
        """
        self.db_path = db_path
        self.c_tol = c_tolerance
        self.h_tol = h_tolerance
        self._create_score_table()

    def _create_score_table(self):
        """创建评分存储表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS molecule_ch_scores")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS molecule_ch_scores (
                    mol_id INTEGER PRIMARY KEY,
                    score REAL,
                    matched_points INTEGER,
                    total_points INTEGER,
                    FOREIGN KEY(mol_id) REFERENCES molecules(mol_id)
                )
            """)
            conn.commit()

    def _calculate_point_score(self, db_point: tuple, user_point: tuple) -> float:
        """基于原始位移值的评分计算"""
        # 处理占位点
        if user_point == (0, 0):
            return 0.0

        db_c, db_h = db_point
        user_c, user_h = user_point

        # 检查是否在矩形区域内（核心逻辑）
        in_c_range = (db_c - self.c_tol <= user_c <= db_c + self.c_tol)
        in_h_range = (db_h - self.h_tol <= user_h <= db_h + self.h_tol)
        if in_c_range and in_h_range:
            return 1.0

        # 计算原始差异
        delta_c = user_c - db_c
        delta_h = user_h - db_h

        # 区域判断逻辑
        if delta_c == 0:
            region = 'top' if delta_h > 0 else 'bottom'
        else:
            slope_ratio = abs(delta_h / delta_c)
            main_slope = self.h_tol / self.c_tol  # 使用原始误差比值

            if slope_ratio > main_slope:
                region = 'top' if delta_h > 0 else 'bottom'
            else:
                region = 'right' if delta_c > 0 else 'left'

        # 评分计算
        if region in ('left', 'right'):
            denominator = abs(delta_c)
            score = self.c_tol / denominator if denominator != 0 else 0.0
        else:
            denominator = abs(delta_h)
            score = self.h_tol / denominator if denominator != 0 else 0.0

        return max(0.0, min(score, 1.0))

    def process_all(self):
        """批量处理所有分子"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # 获取所有匹配数据
            cursor = conn.execute("""
                SELECT m.mol_id, 
                       json_extract(u.db_point, '$[0]') as db_c,
                       json_extract(u.db_point, '$[1]') as db_h,
                       json_extract(u.user_point, '$[0]') as user_c,
                       json_extract(u.user_point, '$[1]') as user_h
                FROM user_matches u
                JOIN molecules m ON u.mol_id = m.mol_id
            """)

            # 按分子分组计算
            from collections import defaultdict
            mol_data = defaultdict(list)
            for row in cursor:
                db_point = (row['db_c'], row['db_h'])
                user_point = (row['user_c'], row['user_h'])
                mol_data[row['mol_id']].append( (db_point, user_point) )

            # 计算评分
            scores = []
            for mol_id, points in mol_data.items():
                total_score = 0.0
                valid_points = 0

                for db_p, user_p in points:
                    score = self._calculate_point_score(db_p, user_p)
                    total_score += score
                    valid_points += 1 if user_p != (0,0) else 0

                avg_score = total_score / len(points) if points else 0.0
                scores.append( (mol_id, avg_score, valid_points, len(points)) )

            # 批量存储
            conn.executemany("""
                INSERT OR REPLACE INTO molecule_ch_scores 
                (mol_id, score, matched_points, total_points)
                VALUES (?, ?, ?, ?)
            """, scores)
            conn.commit()
#
# if __name__ == "__main__":
#     scorer = MoleculeScoreCalculator("chem_data.db", 1.0, 0.2)
#     scorer.process_all()
#     print("分子评分计算完成！")

# # 模式3示例
# user_data = {
#     'type1': [[125, 1.2], [130, 2.1]],
#     'type2': [[120, 1.5], [118, 1.6]],
#     'type3': [[115, 0.8]]
# }
# matcher = UserPointMatcher("chem_data.db", user_data, mode=3)
# matcher.process()
