from typing import Union


class CarbonCoordinateExtractor:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        """创建坐标存储表"""
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
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
            conn.execute(
                "CREATE INDEX idx_coordinates_mol_type ON carbon_h_coordinates(mol_id, h_count)"
            )
            conn.commit()

    def _process_carbon(self, c_data: dict) -> Union[None, dict]:
        """处理单个碳原子数据"""
        h_count = c_data["h_count"]
        c_shift = c_data["c_shift"]
        h_shifts = json.loads(c_data["h_shifts"])

        # 基础校验
        if h_count == 0 or c_shift is None:
            return None
        if len(h_shifts) != h_count:
            print(
                f"Carbon atom {c_data['c_index']} hydrogen displacement data mismatch, expected {h_count}, actual {len(h_shifts)}"
            )
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
                print(
                    f"Unconventional hydrogen number {h_count} @ mol:{c_data['mol_id']}-c:{c_data['c_index']}"
                )
                return None

        except TypeError as e:
            print(f"Data Processing Errors: {str(e)}")
            return None

        return {
            "mol_id": c_data["mol_id"],
            "c_index": c_data["c_index"],
            "h_count": h_count,
            "points": points,
            "count": len(points),
        }

    def process_all(self):
        """批量处理所有分子（修正进度显示）"""
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
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
                    if result and result["count"] > 0:
                        batch.append(
                            (
                                result["mol_id"],
                                result["c_index"],
                                result["h_count"],
                                result["count"],
                                json.dumps(result["points"]),
                            )
                        )

                    # 更新进度（每个碳原子计数+1）
                    pbar.update(1)

                    # 批量插入
                    if len(batch) >= 500:
                        conn.executemany(
                            """
                            INSERT INTO carbon_h_coordinates
                            (mol_id, c_index, h_count, point_count, coordinates)
                            VALUES (?, ?, ?, ?, ?)
                        """,
                            batch,
                        )
                        batch = []

                # 插入剩余数据
                if batch:
                    conn.executemany(
                        """
                        INSERT INTO carbon_h_coordinates
                        (mol_id, c_index, h_count, point_count, coordinates)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        batch,
                    )
                    conn.commit()

            # 获取实际插入的记录数
            inserted = conn.execute("SELECT COUNT(*) FROM carbon_h_coordinates").fetchone()[0]
            print(
                f"Processing is complete! Scans {total} carbon atoms to generate {inserted} valid coordinate records"
            )


import json
import sqlite3
from contextlib import closing

import numpy as np
from tqdm import tqdm


class UserPointMatcher:
    _unmatched_point = json.dumps((0, 0))

    def __init__(self, db_path: str, user_points: dict, mode: int = 1, CH_merge="N"):
        self.db_path = db_path
        self.user_points = user_points
        self.mode = mode
        self.CH_merge = CH_merge
        self._validate_input()
        self._create_result_table()

    def _validate_input(self):
        if self.CH_merge not in {"Y", "N"}:
            raise ValueError("CH_merge must be Y or N")
        if self.mode == 1:
            if "All_type" not in self.user_points:
                raise ValueError("Mode 1 requires All_type key")
        elif self.mode == 2:
            if not ("type13" in self.user_points and "type2" in self.user_points):
                raise ValueError("Mode 2 requires the type13 and type2 keys")
        elif self.mode == 3:
            if not all(k in self.user_points for k in ["type1", "type2", "type3"]):
                raise ValueError("Mode 3 requires the type1/type2/type3 keys")
        else:
            raise ValueError("Invalid Mode Selection (1-3)")

    def _create_result_table(self):
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
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
            conn.execute("CREATE INDEX idx_match_mol ON user_matches(mol_id)")
            conn.commit()

    @staticmethod
    def _rounded_distances(db_points, user_points):
        db = np.asarray(db_points, dtype=float)
        user = np.asarray(user_points, dtype=float)
        delta_c = db[:, 0, np.newaxis] - user[np.newaxis, :, 0]
        delta_h = (db[:, 1, np.newaxis] - user[np.newaxis, :, 1]) * 10.0
        return np.round(np.sqrt(delta_c * delta_c + delta_h * delta_h), 2)

    def _calculate_distances(self, db_points, user_points):
        if not db_points or not user_points:
            return []
        rounded = self._rounded_distances(db_points, user_points).ravel()

        # Row-major flattening has the same i-then-j order as the original
        # nested loops. A stable sort therefore preserves identical tie order.
        order = np.argsort(rounded, kind="stable")
        user_count = len(user_points)
        return [
            (int(flat // user_count), int(flat % user_count), float(rounded[flat]))
            for flat in order
        ]

    @staticmethod
    def _greedy_pairs(rounded):
        rows, columns = rounded.shape
        pair_count = min(rows, columns)
        if not pair_count:
            return

        if pair_count <= 64 and np.isfinite(rounded).all():
            # Row-major argmin is the first still-available edge in the
            # original stable order. Retire both endpoints after choosing it.
            for _ in range(pair_count):
                row, column = divmod(int(np.argmin(rounded)), columns)
                distance = float(rounded[row, column])
                rounded[row, :] = np.inf
                rounded[:, column] = np.inf
                yield row, column, distance
            return

        # Sort flat indices for larger or nonfinite inputs and materialize
        # only the chosen edges in the original stable order.
        flat = rounded.ravel()
        order = np.argsort(flat, kind="stable")
        used_rows, used_columns = set(), set()
        for index in order:
            row, column = divmod(int(index), columns)
            if row in used_rows or column in used_columns:
                continue
            yield row, column, float(flat[index])
            used_rows.add(row)
            used_columns.add(column)
            if len(used_rows) == pair_count:
                break

    def _process_single_molecule(
        self, mol_id, db_points, user_points, h_type=None, *, user_json=None
    ):
        if user_json is None:
            user_json = {}

        def encode_user(index):
            if index not in user_json:
                user_json[index] = json.dumps(user_points[index])
            return user_json[index]

        if self.CH_merge == "N":
            """处理单个分子，每个分子使用独立的used_user集合"""
            matches = []
            pairs = (
                self._greedy_pairs(self._rounded_distances([p[2] for p in db_points], user_points))
                if db_points and user_points
                else ()
            )
            used_db = set()
            for db_idx, user_idx, dist in pairs:
                db_entry = db_points[db_idx]
                matches.append(
                    (
                        mol_id,
                        db_entry[0],
                        json.dumps(db_entry[2]),
                        encode_user(user_idx),
                        dist,
                        h_type if h_type else db_entry[1],
                    )
                )
                used_db.add(db_idx)

            # 填充未匹配的数据库点
            for idx, db_entry in enumerate(db_points):
                if idx not in used_db:
                    matches.append(
                        (
                            mol_id,
                            db_entry[0],
                            json.dumps(db_entry[2]),
                            self._unmatched_point,
                            float("inf"),
                            h_type if h_type else db_entry[1],
                        )
                    )
            return matches
        if self.CH_merge == "Y":
            """处理单个分子，相同坐标的数据库点匹配同一个用户点"""
            # 按坐标分组数据库点
            coord_groups = {}
            for idx, entry in enumerate(db_points):
                coord_key = tuple(entry[2])  # 使用坐标作为分组键
                if coord_key not in coord_groups:
                    coord_groups[coord_key] = []
                coord_groups[coord_key].append((idx, entry))

            # 计算每组坐标到用户点的距离
            distances = []
            for coord, group in coord_groups.items():
                for j, user_point in enumerate(user_points):
                    db_c, db_h = coord
                    user_c, user_h = user_point
                    delta_h = (db_h - user_h) * 10
                    delta_c = db_c - user_c
                    dist = (delta_c**2 + delta_h**2) ** 0.5
                    distances.append((coord, j, round(dist, 2)))

            # 按距离排序
            distances = sorted(distances, key=lambda x: x[2])

            matches = []
            used_user = set()

            # 为每组坐标匹配最佳用户点
            for d in distances:
                coord, user_idx, dist = d
                if user_idx in used_user or coord not in coord_groups:
                    continue

                # 整组匹配同一个用户点
                ref_json = json.dumps(coord)
                query_json = encode_user(user_idx)
                for idx, entry in coord_groups[coord]:
                    c_index, h_type_val, _ = entry
                    matches.append(
                        (
                            mol_id,
                            c_index,
                            ref_json,
                            query_json,
                            dist,
                            h_type if h_type else h_type_val,
                        )
                    )

                used_user.add(user_idx)
                # 移除已匹配的组
                del coord_groups[coord]

            # 处理未匹配的数据库点
            for coord, group in coord_groups.items():
                ref_json = json.dumps(coord)
                for idx, entry in group:
                    c_index, h_type_val, _ = entry
                    matches.append(
                        (
                            mol_id,
                            c_index,
                            ref_json,
                            self._unmatched_point,
                            float("inf"),
                            h_type if h_type else h_type_val,
                        )
                    )

            return matches

    def _match_mode1(self):
        """模式1：每个分子独立匹配所有用户点"""
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            molecules = {}
            for row in conn.execute(
                "SELECT mol_id, c_index, h_count, coordinates FROM carbon_h_coordinates"
            ):
                points = json.loads(row[3])
                if row[0] not in molecules:
                    molecules[row[0]] = []
                for p in points:
                    molecules[row[0]].append((row[1], row[2], p))

        all_matches = []
        user_pts = [tuple(p) for p in self.user_points["All_type"]]
        user_json = {}
        for mol_id in sorted(molecules.keys()):
            db_points = molecules[mol_id]
            matches = self._process_single_molecule(
                mol_id=mol_id, db_points=db_points, user_points=user_pts, user_json=user_json
            )
            all_matches += matches
        return all_matches

    def _match_mode2(self):
        """新增：模式2专用匹配方法（合并处理type1/3）"""
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            # 获取所有type1和type3的原子
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
                    molecules[mol_id].append((row[1], row[2], p))  # (c_index, h_type, coordinates)

        all_matches = []
        user_pts = [tuple(p) for p in self.user_points["type13"]]
        user_json = {}

        for mol_id in sorted(molecules.keys()):
            db_points = molecules[mol_id]

            # 处理type1/3组合
            matches = self._process_single_molecule(
                mol_id=mol_id, db_points=db_points, user_points=user_pts, user_json=user_json
            )
            all_matches += matches

        # 单独处理type2
        all_matches += self._match_by_type(h_type=2, user_points=self.user_points["type2"])
        return all_matches

    def _match_by_type(self, h_type, user_points):
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            molecules = {}
            for row in conn.execute(
                """
                SELECT mol_id, c_index, coordinates
                FROM carbon_h_coordinates
                WHERE h_count = ?""",
                (h_type,),
            ):
                points = json.loads(row[2])
                if row[0] not in molecules:
                    molecules[row[0]] = []
                for p in points:
                    molecules[row[0]].append((row[1], p))

        all_matches = []
        user_pts = [tuple(p) for p in user_points]
        user_json = {}
        for mol_id in sorted(molecules.keys()):
            # 每个分子独立处理
            db_points = [(p[0], h_type, p[1]) for p in molecules[mol_id]]
            matches = self._process_single_molecule(
                mol_id=mol_id,
                db_points=db_points,
                user_points=user_pts,
                h_type=h_type,
                user_json=user_json,
            )
            all_matches += matches
        return all_matches

    def process(self):
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("DELETE FROM user_matches")
            matches = []

            if self.mode == 1:
                matches = self._match_mode1()

            elif self.mode == 2:
                matches = self._match_mode2()
            elif self.mode == 3:
                # 分别处理三种类型
                matches += self._match_by_type(h_type=1, user_points=self.user_points["type1"])
                matches += self._match_by_type(h_type=2, user_points=self.user_points["type2"])
                matches += self._match_by_type(h_type=3, user_points=self.user_points["type3"])

            # 批量插入数据
            batch_size = 500
            with tqdm(total=len(matches), desc="Save the matching results") as pbar:
                for i in range(0, len(matches), batch_size):
                    batch = matches[i : i + batch_size]
                    conn.executemany(
                        """
                        INSERT INTO user_matches
                        (mol_id, c_index, db_point, user_point, distance, h_type)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        batch,
                    )
                    pbar.update(len(batch))

            # 统计结果
            matched = conn.execute(
                "SELECT COUNT(*) FROM user_matches WHERE distance < 1e9"
            ).fetchone()[0]
            print(
                f"Match complete! Valid matches {matched} and placeholder {len(matches) - matched} records"
            )


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
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
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
        in_c_range = db_c - self.c_tol <= user_c <= db_c + self.c_tol
        in_h_range = db_h - self.h_tol <= user_h <= db_h + self.h_tol
        if in_c_range and in_h_range:
            return 1.0

        # 计算原始差异
        delta_c = user_c - db_c
        delta_h = user_h - db_h

        # 区域判断逻辑
        if delta_c == 0:
            region = "top" if delta_h > 0 else "bottom"
        else:
            slope_ratio = abs(delta_h / delta_c)
            main_slope = self.h_tol / self.c_tol  # 使用原始误差比值

            if slope_ratio > main_slope:
                region = "top" if delta_h > 0 else "bottom"
            else:
                region = "right" if delta_c > 0 else "left"

        # 评分计算
        if region in ("left", "right"):
            denominator = abs(delta_c)
            score = self.c_tol / denominator if denominator != 0 else 0.0
        else:
            denominator = abs(delta_h)
            score = self.h_tol / denominator if denominator != 0 else 0.0

        return max(0.0, min(score, 1.0))

    def process_all(self):
        """批量处理所有分子"""
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
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

            # Accumulate each molecule in cursor order without keeping every
            # pair in memory for a second pass.
            totals = {}
            for row in cursor:
                db_point = (row["db_c"], row["db_h"])
                user_point = (row["user_c"], row["user_h"])
                mol_id = row["mol_id"]
                total_score, matched, count = totals.get(mol_id, (0.0, 0, 0))
                totals[mol_id] = (
                    total_score + self._calculate_point_score(db_point, user_point),
                    matched + (user_point != (0, 0)),
                    count + 1,
                )
            scores = [
                (mol_id, score / count, matched, count)
                for mol_id, (score, matched, count) in totals.items()
            ]

            # 批量存储
            conn.executemany(
                """
                INSERT OR REPLACE INTO molecule_ch_scores
                (mol_id, score, matched_points, total_points)
                VALUES (?, ?, ?, ?)
            """,
                scores,
            )
            conn.commit()
