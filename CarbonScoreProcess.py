import os
import json
import logging
import math
import sqlite3
import threading
from collections import Counter, deque
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# 线程本地存储用于数据库连接
thread_local = threading.local()


class GreedyMatcher:
    def __init__(self, db_path, user_data, c_range=(10, 30), fw_range=(100.0, 300.0), mode='typed', C_merge='N'):
        """
        :param mode: 匹配模式 ('typed'按类型匹配/'untyped'全局匹配)
        :param C_merge: 是否启用等效模式 ('Y'/'N')
        """
        self.db_path = db_path
        self.user_data = user_data
        self.c_range = c_range
        self.fw_range = fw_range
        self.mode = mode
        self.C_merge = C_merge
        self._validate_inputs()
        self._init_results_table()
        self._create_indexes()
        self.carbon_cache = {}
        # 可选：用于“满分优先”匹配的阈值配置，不改变外部调用方式
        self.match_green_threshold = None
        self.match_yellow_threshold = None
        self.match_fine_ranges = None

    def _validate_inputs(self):
        """参数校验（支持两种模式）"""
        if self.mode == 'typed':
            if not isinstance(self.user_data, dict) or any(not isinstance(v, list) for v in self.user_data.values()):
                raise ValueError("The type pattern requires {0:[],1:[],...} format")
            self.user_total = sum(len(v) for v in self.user_data.values())
        elif self.mode == 'untyped':
            if not isinstance(self.user_data, list):
                raise ValueError("The global mode requires a list of displacements")
            self.user_total = len(self.user_data)
        else:
            raise ValueError("Invalid mode: typed/untyped")

        if self.c_range[0] > self.c_range[1] or self.fw_range[0] > self.fw_range[1]:
            raise ValueError("The filter range is invalid")
        if self.user_total <= 0:
            raise ValueError("User data cannot be empty")

    @staticmethod
    def _round_half_up(value):
        """Round halves upward for carbon-count tolerance."""
        return int(math.floor(float(value) + 0.5))

    def _allowed_missing_carbons(self):
        """Allow database molecules up to 10% larger than the user carbon count."""
        return self._round_half_up(int(self.user_total) * 0.10)

    def _is_db_carbon_count_allowed(self, db_carbon_count):
        db_count = int(db_carbon_count)
        return db_count <= int(self.user_total) + self._allowed_missing_carbons()

    def _max_allowed_db_carbon_count(self):
        return min(int(self.c_range[1]), int(self.user_total) + self._allowed_missing_carbons())

    @contextmanager
    def _get_conn(self):
        """线程安全的数据库连接管理"""
        if not hasattr(thread_local, "conn"):
            thread_local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30,
                isolation_level=None
            )
            thread_local.conn.execute("PRAGMA journal_mode=WAL")
            thread_local.conn.execute("PRAGMA synchronous=NORMAL")
            thread_local.conn.execute("PRAGMA cache_size=-20000")  # 20MB缓存
        try:
            yield thread_local.conn
        except sqlite3.DatabaseError:
            thread_local.conn.rollback()
            raise

    def _create_indexes(self):
        """创建必要索引"""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_molecules_filter
                ON molecules(carbon_count, fw)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_carbons_mol
                ON carbons(mol_id)
            """)

    def _init_results_table(self):
        """初始化结果表"""
        with self._get_conn() as conn:
            conn.execute("DROP TABLE IF EXISTS greedy_matches")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS greedy_matches (
                    match_id INTEGER PRIMARY KEY,
                    mol_id INTEGER,
                    mol_name TEXT,
                    user_shift REAL,
                    db_c_index INTEGER,
                    db_shift REAL,
                    type_matched BOOLEAN,
                    connected_h_shifts TEXT,
                    connected_carbons TEXT,
                    FOREIGN KEY(mol_id) REFERENCES molecules(mol_id)
                )
            """)

    def _precache_molecules(self):
        """预加载分子数据到内存（不提前过滤 None，保留到评分阶段做 0 分占位）"""
        max_allowed_db_carbons = self._max_allowed_db_carbon_count()
        with self._get_conn() as conn:
            query = """
                SELECT m.mol_id, m.name, c.c_index, c.h_count, c.c_shift,
                       c.h_shifts, c.connected_carbon
                FROM molecules m
                JOIN carbons c ON m.mol_id = c.mol_id
                WHERE m.carbon_count BETWEEN ? AND ?
                  AND m.fw BETWEEN ? AND ?
                  AND m.carbon_count <= ?
            """
            params = (
                self.c_range[0],
                self.c_range[1],
                self.fw_range[0],
                self.fw_range[1],
                max_allowed_db_carbons
            )
            rows = conn.execute(query, params).fetchall()

            for row in rows:
                mol_id, name, c_idx, hc, c_shift, h_shifts, conn_c = row
                if mol_id not in self.carbon_cache:
                    self.carbon_cache[mol_id] = {
                        'name': name,
                        'carbons': {}
                    }
                self.carbon_cache[mol_id]['carbons'][c_idx] = {
                    'c_index': c_idx,
                    'h_count': hc,
                    'c_shift': c_shift,
                    'h_shifts': h_shifts,
                    'connected_c': conn_c
                }


    def _remaining_values(self, values, used_counter):
        """
        按“出现次数”而不是按“值是否存在”过滤剩余用户位移。
        这样重复值会被视为独立输入实例，各自只能使用一次。
        """
        local_counter = Counter()
        result = []

        for v in values:
            local_counter[v] += 1
            if local_counter[v] > used_counter[v]:
                result.append(v)

        return result

    def _to_float(self, value):
        """安全转换为 float，失败返回 None"""
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _shift_key(self, value, ndigits=6):
        """用于 exact/等效分组的稳定键"""
        fv = self._to_float(value)
        if fv is None:
            return None
        return round(fv, ndigits)

    def _get_green_threshold_for_shift(self, db_shift):
        green, _yellow = self._get_match_thresholds_for_shift(db_shift)
        return green

    def _get_match_thresholds_for_shift(self, db_shift):
        """
        根据当前分配给 matcher 的配置，返回某个数据库位移对应的匹配阈值。
        不改外部调用方式：
        1) matcher.match_fine_ranges = [((low, high), green, yellow), ...]
        2) matcher.match_green_threshold / match_yellow_threshold = green / yellow
        都没设置时返回 None，退化为 exact + 最近邻。
        """
        dbv = self._to_float(db_shift)
        if dbv is None:
            return None, None

        fine_ranges = getattr(self, "match_fine_ranges", None)
        if fine_ranges:
            for item in fine_ranges:
                try:
                    (low, high), green, yellow = item
                    if low is None or high is None:
                        continue
                    if low <= dbv <= high:
                        return float(green), float(yellow)
                except Exception:
                    continue

        green = getattr(self, "match_green_threshold", None)
        yellow = getattr(self, "match_yellow_threshold", None)
        green_value = None
        if green is not None:
            try:
                green_value = float(green)
            except (TypeError, ValueError):
                green_value = None

        yellow_value = None
        if yellow is not None:
            try:
                yellow_value = float(yellow)
            except (TypeError, ValueError):
                yellow_value = None

        return green_value, yellow_value

    def _linear_sum_assignment(self, cost_matrix):
        """
        纯 Python 版最小权匹配（Hungarian / Kuhn-Munkres 的势能实现）。
        输入要求：行数 <= 列数。
        返回：
            row_ind, col_ind
        """
        cost = np.array(cost_matrix, dtype=np.float64)
        n, m = cost.shape
        if n == 0:
            return [], []
        if n > m:
            raise ValueError("The number of rows must be less than or equal to the number of columns")

        u = np.zeros(n + 1, dtype=np.float64)
        v = np.zeros(m + 1, dtype=np.float64)
        p = np.zeros(m + 1, dtype=np.int32)
        way = np.zeros(m + 1, dtype=np.int32)

        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = np.full(m + 1, np.inf, dtype=np.float64)
            used = np.zeros(m + 1, dtype=bool)

            while True:
                used[j0] = True
                i0 = p[j0]
                delta = np.inf
                j1 = 0

                for j in range(1, m + 1):
                    if not used[j]:
                        cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j

                for j in range(m + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta

                j0 = j1
                if p[j0] == 0:
                    break

            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        row_ind = []
        col_ind = []
        for j in range(1, m + 1):
            if p[j] != 0:
                row_ind.append(p[j] - 1)
                col_ind.append(j - 1)

        return row_ind, col_ind

    def _solve_fullscore_assignment(self, row_values, col_values, row_thresholds):
        """
        行 = 数据库目标（单碳或等效组）
        列 = 用户峰实例
        目标：
        1. 有 green/yellow 双阈值时，最大化后续 carbon scorer 的实际得分。
        2. yellow 外的红区匹配优先留空，不强行占用用户峰。
        3. 没有 yellow 配置时保留旧的 green 优先/最近邻行为。
        """
        n_rows = len(row_values)
        n_cols = len(col_values)

        if n_rows == 0:
            return {}
        if n_cols == 0:
            return {i: None for i in range(n_rows)}

        total_cols = n_cols + n_rows

        SCORE_SCALE = 10000.0
        DUMMY_COST = SCORE_SCALE
        RED_COST = SCORE_SCALE * 1000.0
        LEGACY_BAD_BASE = 1e4
        LEGACY_UNMATCHED_BASE = 1e6

        cost = np.full((n_rows, total_cols), LEGACY_UNMATCHED_BASE, dtype=np.float64)

        for i in range(n_rows):
            dbv = row_values[i]
            threshold_item = row_thresholds[i] if row_thresholds else (None, None)
            if isinstance(threshold_item, (tuple, list)):
                green = threshold_item[0] if len(threshold_item) > 0 else None
                yellow = threshold_item[1] if len(threshold_item) > 1 else None
            else:
                green = threshold_item
                yellow = None

            for j in range(n_cols):
                delta = abs(dbv - col_values[j])

                if green is not None and yellow is not None and yellow > green:
                    if delta <= green:
                        pair_score = 1.0
                    elif delta < yellow:
                        pair_score = 1.0 - (delta - green) / (yellow - green)
                    else:
                        pair_score = 0.0

                    if pair_score > 0.0:
                        cost[i, j] = (1.0 - pair_score) * SCORE_SCALE + delta * 1e-3
                    else:
                        cost[i, j] = RED_COST + delta
                elif delta == 0:
                    cost[i, j] = 0.0
                elif green is not None:
                    if delta <= green:
                        cost[i, j] = delta
                    else:
                        cost[i, j] = LEGACY_BAD_BASE + delta
                else:
                    cost[i, j] = LEGACY_BAD_BASE + delta

            if green is not None and yellow is not None and yellow > green:
                cost[i, n_cols + i] = DUMMY_COST
            else:
                cost[i, n_cols + i] = LEGACY_UNMATCHED_BASE

        row_ind, col_ind = self._linear_sum_assignment(cost)

        assigned = {}
        for r, c in zip(row_ind, col_ind):
            if c < n_cols:
                assigned[r] = c
            else:
                assigned[r] = None

        for i in range(n_rows):
            assigned.setdefault(i, None)

        return assigned


    def _vectorized_match(self, user_shifts, db_candidates):
        """
        满分优先的一对一匹配，保持输入输出不变：
        输入:
            user_shifts: List[float]
            db_candidates: List[dict]
        输出:
            matches: [(user_shift, c_index, db_shift), ...]
            unmatched: [carbon_dict, ...]

        规则：
        - typed / untyped 两种模式的上层调用方式不变
        - C_merge == 'N'：严格一对一，用户峰实例和数据库碳都只使用一次
        - C_merge == 'Y'：相同 c_shift 的数据库碳视为一组，整组共享一个用户峰
        - 优先目标：green（第一误差）内的匹配数量最大
        - 次优目标：在上述前提下总误差最小
        - c_shift 为 None / 非法值的数据库碳不参与匹配，但作为 unmatched 返回
        """
        if not user_shifts or not db_candidates:
            return [], db_candidates

        try:
            # 过滤无效 user，并保留“实例身份”
            valid_user_items = []
            for user_idx, shift in enumerate(user_shifts):
                shift_float = self._to_float(shift)
                if shift_float is None:
                    continue
                valid_user_items.append((user_idx, shift, shift_float))

            # 数据库碳分成：可匹配的 / 非法或 None 的
            valid_db_items = []
            invalid_db_candidates = []
            for carbon in db_candidates:
                c_shift_float = self._to_float(carbon.get('c_shift'))
                if c_shift_float is None:
                    invalid_db_candidates.append(carbon)
                else:
                    valid_db_items.append((carbon, c_shift_float))

            if not valid_db_items:
                return [], db_candidates

            # 没有任何合法用户峰：全部 unmatched，确保后续还能进评分表
            if not valid_user_items:
                return [], [carbon for carbon, _ in valid_db_items] + invalid_db_candidates

            if self.C_merge == 'N':
                # 普通模式：每个数据库碳独立匹配
                db_items = sorted(valid_db_items, key=lambda x: (x[1], x[0]['c_index']))
                user_items = sorted(valid_user_items, key=lambda x: (x[2], x[0]))

                row_values = [db_shift_float for _, db_shift_float in db_items]
                row_thresholds = [self._get_match_thresholds_for_shift(carbon['c_shift']) for carbon, _ in db_items]
                col_values = [user_shift_float for _, _, user_shift_float in user_items]

                assigned = self._solve_fullscore_assignment(row_values, col_values, row_thresholds)

                matches = []
                matched_c_idx = set()
                for row_idx, col_idx in assigned.items():
                    carbon, _ = db_items[row_idx]
                    if col_idx is None:
                        continue

                    _, original_user_shift, _ = user_items[col_idx]
                    matches.append((
                        original_user_shift,
                        carbon['c_index'],
                        carbon['c_shift']
                    ))
                    matched_c_idx.add(carbon['c_index'])

                unmatched = [carbon for carbon, _ in db_items if carbon['c_index'] not in matched_c_idx]
                unmatched.extend(invalid_db_candidates)
                return matches, unmatched

            if self.C_merge == 'Y':
                # 等效模式：相同 c_shift 的数据库碳分组，整组共享一个用户峰
                shift_groups = {}
                for carbon, c_shift_float in valid_db_items:
                    key = self._shift_key(c_shift_float)
                    shift_groups.setdefault(key, []).append((carbon, c_shift_float))

                group_items = []
                for key, carbons in shift_groups.items():
                    carbons = sorted(carbons, key=lambda item: item[0]['c_index'])
                    group_shift_float = carbons[0][1]
                    group_items.append((key, group_shift_float, carbons))

                group_items.sort(key=lambda x: (x[1], x[2][0][0]['c_index']))
                user_items = sorted(valid_user_items, key=lambda x: (x[2], x[0]))

                row_values = [group_shift for _, group_shift, _ in group_items]
                row_thresholds = [self._get_match_thresholds_for_shift(group_shift) for _, group_shift, _ in group_items]
                col_values = [user_shift_float for _, _, user_shift_float in user_items]

                assigned = self._solve_fullscore_assignment(row_values, col_values, row_thresholds)

                matches = []
                matched_c_idx = set()
                for row_idx, col_idx in assigned.items():
                    _group_key, _group_shift, carbons = group_items[row_idx]
                    if col_idx is None:
                        continue

                    _, original_user_shift, _ = user_items[col_idx]
                    for carbon, _ in carbons:
                        matches.append((
                            original_user_shift,
                            carbon['c_index'],
                            carbon['c_shift']
                        ))
                        matched_c_idx.add(carbon['c_index'])

                unmatched = []
                for _, _, carbons in group_items:
                    for carbon, _ in carbons:
                        if carbon['c_index'] not in matched_c_idx:
                            unmatched.append(carbon)

                unmatched.extend(invalid_db_candidates)
                return matches, unmatched

            return [], db_candidates

        except Exception as e:
            print(f"匹配错误: {str(e)}")
            return [], db_candidates

    def _process_molecule(self, mol_id):
        """整合两种匹配模式，并保证所有 eligible 分子都写入 greedy_matches"""
        try:
            if mol_id not in self.carbon_cache:
                return

            mol_data = self.carbon_cache[mol_id]
            all_matches = []

            if self.mode == 'typed':
                # 按“出现次数”记录已使用的用户位移，避免重复值被 set() 误伤
                used_user_counter = Counter()

                # 阶段1：按类型匹配
                for h_type in [3, 2, 1, 0]:
                    user_shifts = self.user_data.get(h_type, [])
                    available_user = self._remaining_values(user_shifts, used_user_counter)
                    candidates = [c for c in mol_data['carbons'].values() if c['h_count'] == h_type]

                    matches, _ = self._vectorized_match(available_user, candidates)
                    for u_shift, c_idx, c_shift in matches:
                        all_matches.append((h_type, u_shift, c_idx, c_shift, True))
                        used_user_counter[u_shift] += 1

                # 阶段2：剩余原子全局匹配（使用所有未使用的用户位移）
                matched_c_idx = {m[2] for m in all_matches}
                remaining_candidates = [
                    c for c in mol_data['carbons'].values()
                    if c['c_index'] not in matched_c_idx
                ]

                all_user = []
                for h_type in [0, 1, 2, 3]:
                    all_user.extend(self.user_data.get(h_type, []))

                remaining_user = self._remaining_values(all_user, used_user_counter)

                if remaining_candidates and remaining_user:
                    matches, unmatched = self._vectorized_match(remaining_user, remaining_candidates)
                    for u_shift, c_idx, c_shift in matches:
                        all_matches.append((None, u_shift, c_idx, c_shift, False))
                        used_user_counter[u_shift] += 1

                    # 未匹配上的也写 0 分占位，保证分子进入后续评分
                    for carbon in unmatched:
                        all_matches.append((
                            None,
                            None,
                            carbon['c_index'],
                            carbon.get('c_shift'),
                            False
                        ))
                else:
                    # 第二阶段无法匹配时，剩余碳全部写占位
                    for carbon in remaining_candidates:
                        all_matches.append((
                            None,
                            None,
                            carbon['c_index'],
                            carbon.get('c_shift'),
                            False
                        ))

            elif self.mode == 'untyped':
                candidates = list(mol_data['carbons'].values())
                matches, unmatched = self._vectorized_match(self.user_data, candidates)

                for u_shift, c_idx, c_shift in matches:
                    all_matches.append((None, u_shift, c_idx, c_shift, True))

                # 未匹配的也写占位
                for carbon in unmatched:
                    all_matches.append((
                        None,
                        None,
                        carbon['c_index'],
                        carbon.get('c_shift'),
                        False
                    ))

            # 保险：如果一个分子完全没匹配成功，也要把所有碳写进去
            if not all_matches:
                for carbon in mol_data['carbons'].values():
                    all_matches.append((
                        None,
                        None,
                        carbon['c_index'],
                        carbon.get('c_shift'),
                        False
                    ))

            self._bulk_save_matches(mol_id, mol_data['name'], all_matches)

        except Exception as e:
            print(f"Processing molecule {mol_id} failed: {str(e)}")

    def _bulk_save_matches(self, mol_id, mol_name, matches):
        """保存匹配结果到数据库"""
        data = []
        for match in matches:
            h_type, u_shift, c_idx, db_shift, type_matched = match
            carbon = self.carbon_cache[mol_id]['carbons'].get(c_idx, {})
            data.append((
                mol_id,
                mol_name,
                u_shift,
                c_idx,
                db_shift,
                type_matched,
                carbon.get('h_shifts', '[]'),
                carbon.get('connected_c', '[]')
            ))

        if data:
            try:
                with self._get_conn() as conn:
                    conn.executemany("""
                        INSERT INTO greedy_matches
                        (mol_id, mol_name, user_shift,
                         db_c_index, db_shift, type_matched, connected_h_shifts, connected_carbons)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, data)
                    conn.commit()
            except sqlite3.Error as e:
                print(f"Save molecule {mol_id} result failed: {str(e)}")

    def run(self, max_workers=None):
        """执行匹配流程"""
        max_workers = max_workers or min(os.cpu_count() * 2, 32)

        print("Pre-loaded molecular data...")
        self._precache_molecules()
        qualified = list(self.carbon_cache.keys())
        print(f"Find {len(qualified)} molecules that are eligible")

        batch_size = 500
        batches = [qualified[i:i + batch_size] for i in range(0, len(qualified), batch_size)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=len(qualified), desc="Match progress") as pbar:
                for batch in batches:
                    futures = {
                        executor.submit(self._process_molecule, mol_id): mol_id
                        for mol_id in batch
                    }
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            mid = futures[future]
                            print(f"Molecule {mid} error: {str(e)}")
                        finally:
                            pbar.update(1)

        self._validate_results()

    def _validate_results(self):
        max_allowed_db_carbons = self._max_allowed_db_carbon_count()
        """结果验证"""
        with self._get_conn() as conn:
            invalid = conn.execute("""
                SELECT COUNT(*)
                FROM greedy_matches gm
                JOIN molecules m ON gm.mol_id = m.mol_id
                WHERE m.carbon_count > ?
            """, (max_allowed_db_carbons,)).fetchone()[0]

            if invalid > 0:
                raise ValueError(f"{invalid} violation matches found！")

            total_matches = conn.execute("SELECT COUNT(*) FROM greedy_matches").fetchone()[0]
            print(f"Verified! A total of {total_matches} valid matching records were generated")


class CarbonScorer:
    def __init__(self, db_path: str):
        """
        :param db_path: 数据库路径
        """
        self.db_path = db_path
        self._validate_db_structure()

    def _validate_db_structure(self):
        """验证数据库表结构"""
        required_tables = ['greedy_matches']
        with sqlite3.connect(self.db_path) as conn:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            existing_tables = {t[0] for t in tables}
            missing = set(required_tables) - existing_tables
            if missing:
                raise ValueError(f"Required forms are missing：{missing}")

    def create_results_table(self):
        """创建评分结果表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS carbon_scores")
            conn.execute("""
                CREATE TABLE carbon_scores (
                    score_id INTEGER PRIMARY KEY,
                    mol_id INTEGER,
                    mol_name TEXT,
                    c_index INTEGER,
                    db_shift REAL,
                    user_shift REAL,
                    score REAL,
                    color TEXT,
                    connected_carbons TEXT,
                    FOREIGN KEY(mol_id) REFERENCES molecules(mol_id)
                )
            """)
            conn.commit()

    def process_scoring(
            self,
            mode: str,
            global_thresholds: Tuple[float, float] = None,
            fine_ranges: List[Tuple[Tuple[float, float], float, float]] = None
    ):
        """
        执行评分流程

        :param mode: 模式选择 ('global' 或 'fine')
        :param global_thresholds: 全局阈值 (green_threshold, yellow_threshold)
        :param fine_ranges: 精细模式范围列表 [((low, high), green, yellow), ...]
        """
        if mode == 'global':
            if not global_thresholds or len(global_thresholds) != 2:
                raise ValueError("The global mode requires two thresholds")
            green, yellow = global_thresholds
            if green >= yellow:
                raise ValueError("The green threshold must be less than the yellow threshold")
        elif mode == 'fine':
            if not fine_ranges:
                raise ValueError("Granular mode requires a list of ranges")
            for r in fine_ranges:
                (low, high), g, y = r
                if low >= high or g >= y:
                    raise ValueError("Invalid scope or threshold")
        else:
            raise ValueError("Invalid mode selection")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT gm.mol_id, gm.mol_name, gm.db_c_index as c_index,
                       gm.db_shift, gm.user_shift, gm.type_matched,
                       gm.connected_carbons
                FROM greedy_matches gm
            """)

            batch = []
            for row in cursor:
                db_shift = row['db_shift']
                user_shift = row['user_shift']
                type_matched = row['type_matched']

                if db_shift is not None and user_shift is not None:
                    delta = abs(db_shift - user_shift)
                else:
                    delta = 1000

                if mode == 'global':
                    score, color = self._global_score(delta, type_matched, global_thresholds)
                else:
                    score, color = self._fine_score(db_shift, delta, type_matched, fine_ranges)

                batch.append((
                    row['mol_id'],
                    row['mol_name'],
                    row['c_index'],
                    db_shift,
                    user_shift,
                    score,
                    color,
                    row['connected_carbons']
                ))

                if len(batch) >= 1000:
                    self._insert_batch(conn, batch)
                    batch = []

            if batch:
                self._insert_batch(conn, batch)

    def _global_score(
            self,
            delta: float,
            type_matched: bool,
            thresholds: Tuple[float, float]
    ) -> Tuple[float, str]:
        """全局模式评分"""
        if not type_matched or delta is None:
            return 0.0, '#FF0000'

        green, yellow = thresholds
        if delta < green:
            return 1.0, '#00FF00'
        elif delta > yellow:
            return 0.0, '#FF0000'
        else:
            score = 1 - (delta - green) / (yellow - green)
            return round(score, 2), '#FFFF00'

    def _fine_score(
            self,
            db_shift: float,
            delta: float,
            type_matched: bool,
            ranges: List[Tuple[Tuple[float, float], float, float]]
    ) -> Tuple[float, str]:
        """精细模式评分"""
        if db_shift is None or not type_matched or delta is None:
            return 0.0, 'red'

        matched_range = None
        for r in ranges:
            (low, high), g, y = r
            if low is None or high is None:
                continue
            if low <= db_shift <= high:
                matched_range = (g, y)
                break

        if not matched_range:
            return 0.0, 'red'

        green, yellow = matched_range
        if delta < green:
            return 1.0, 'green'
        elif delta > yellow:
            return 0.0, 'red'
        else:
            score = 1 - (delta - green) / (yellow - green)
            return round(score, 2), 'yellow'

    def _insert_batch(self, conn, batch):
        """批量插入数据"""
        conn.executemany("""
            INSERT INTO carbon_scores
            (mol_id, mol_name, c_index, db_shift, user_shift, score, color, connected_carbons)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, batch)
        conn.commit()


# 配置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class RobustMoleculeScorer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._validate_structure()

    def _validate_structure(self):
        """验证数据库表结构完整性"""
        required_tables = {'carbon_scores', 'molecules', 'greedy_matches'}
        with sqlite3.connect(self.db_path) as conn:
            existing_tables = {row[0] for row in
                               conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            missing = required_tables - existing_tables
            if missing:
                raise ValueError(f"缺失关键数据表: {missing}")

    def create_result_table(self):
        """创建分子评分结果表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS molecule_scores")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS molecule_scores (
                    mol_id INTEGER PRIMARY KEY,
                    mol_name TEXT,
                    final_score REAL,
                    score_details TEXT,
                    FOREIGN KEY(mol_id) REFERENCES molecules(mol_id)
                )
            """)
            conn.commit()

    def process_molecules(
            self,
            self_weight: float,
            env_weight: float,
            env_level: int = 1
    ):
        """
        增强鲁棒性的评分流程

        :param self_weight: 自身评分权重 (0-1)
        :param env_weight: 环境评分权重 (0-1)
        :param env_level: 环境层级 (1-3)
        """
        if not (0 <= self_weight <= 1) or not (0 <= env_weight <= 1):
            raise ValueError("The weight parameter must be between 0 and 1")
        if abs(self_weight + env_weight - 1.0) > 1e-6:
            raise ValueError("The sum of the weights must be equal to 1")
        if env_level not in {1, 2, 3}:
            raise ValueError("The environment level must be 1, 2, or 3")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            mol_ids = [row[0] for row in
                       conn.execute("SELECT DISTINCT mol_id FROM greedy_matches")]

            total_mols = len(mol_ids)
            with tqdm(total=total_mols, desc="Molecular scoring progress", unit="mol") as pbar:
                for mol_id in mol_ids:
                    try:
                        mol_data = self._load_molecule_data(conn, mol_id)
                        connection_graph = self._build_connection_graph(mol_data)
                        self._score_molecule(conn, mol_id, mol_data,
                                             connection_graph, self_weight,
                                             env_weight, env_level)
                    except Exception as e:
                        logging.error(f"Processing Molecule {mol_id} Failed: {str(e)}")
                        raise
                    finally:
                        pbar.update(1)
                        pbar.set_postfix_str(f"Current molecule: {mol_id}")

        print(f"\nGrading done! Processed {total_mols} molecules")

    def _load_molecule_data(self, conn, mol_id: int) -> Dict[int, dict]:
        """加载分子数据并进行完整性校验"""
        cursor = conn.execute("""
            SELECT c_index, score, connected_carbons
            FROM carbon_scores
            WHERE mol_id = ?
        """, (mol_id,))

        molecule_data = {}
        for row in cursor:
            if not (0 <= row['score'] <= 1):
                raise ValueError(
                    f"The carbon atom {row['c_index']} score of the molecule {mol_id} is out of range: {row['score']}"
                )

            try:
                connections = json.loads(row['connected_carbons'])
                if not isinstance(connections, list):
                    raise TypeError("The connection relationship must be a list")
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(f"Carbon atom {row['c_index']} of molecule {mol_id} is malformed") from e

            molecule_data[row['c_index']] = {
                'score': row['score'],
                'connections': connections
            }
        return molecule_data

    def _build_connection_graph(self, mol_data: dict) -> Dict[int, List[int]]:
        """构建带有效性检查的连接关系图"""
        valid_indices = set(mol_data.keys())
        graph = {}

        for c_index, data in mol_data.items():
            valid_connections = []
            for neighbor in data['connections']:
                if neighbor in valid_indices:
                    valid_connections.append(neighbor)
            graph[c_index] = valid_connections
        return graph

    def _get_environment_atoms(
            self,
            root: int,
            graph: Dict[int, List[int]],
            max_level: int
    ) -> List[int]:
        """获取多级环境原子（带连接有效性检查）"""
        visited = {root}
        queue = deque([(root, 0)])
        environment = []

        while queue:
            current_atom, current_level = queue.popleft()
            if current_level >= max_level:
                continue

            for neighbor in graph.get(current_atom, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    environment.append(neighbor)
                    queue.append((neighbor, current_level + 1))
        return environment

    def _calculate_env_score(
            self,
            env_atoms: List[int],
            mol_data: dict
    ) -> float:
        """计算环境评分（带数据完整性检查）"""
        if len(env_atoms) == 0:
            return 1
        else:
            valid_scores = []
            for atom in env_atoms:
                if atom not in mol_data:
                    logging.warning(f" Ignore environmental atoms that do not exist in the molecular data:{atom}")
                    continue
                valid_scores.append(mol_data[atom]['score'])

            if not valid_scores:
                return 0.0
            return sum(valid_scores) / len(valid_scores)

    def _score_molecule(
            self,
            conn,
            mol_id: int,
            mol_data: dict,
            graph: dict,
            self_weight: float,
            env_weight: float,
            env_level: int
    ):
        """执行分子评分并存储结果"""
        total_score = 0.0
        score_details = []

        for c_index, data in mol_data.items():
            env_atoms = self._get_environment_atoms(c_index, graph, env_level)
            try:
                env_score = self._calculate_env_score(env_atoms, mol_data)
            except ZeroDivisionError:
                env_score = 0.0

            if data['score'] == 0:
                composite = 0
            else:
                composite = (self_weight * data['score'] +
                             env_weight * env_score)
            total_score += composite

            score_details.append({
                'c_index': c_index,
                'self_score': data['score'],
                'env_score': round(env_score, 4),
                'env_atoms': env_atoms,
                'composite': round(composite, 4)
            })

        final_score = total_score / len(mol_data) if mol_data else 0
        mol_name = conn.execute(
            "SELECT name FROM molecules WHERE mol_id = ?", (mol_id,)
        ).fetchone()[0]

        conn.execute("""
            INSERT OR REPLACE INTO molecule_scores
            (mol_id, mol_name, final_score, score_details)
            VALUES (?, ?, ?, ?)
        """, (
            mol_id,
            mol_name,
            round(final_score, 4),
            json.dumps(score_details)
        ))
        conn.commit()
