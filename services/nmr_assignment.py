from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


@dataclass
class HydrogenAssignment:
    atom_indices: List[int]
    predicted_shift: Optional[float]
    experimental_shift: Optional[float] = None
    error: Optional[float] = None
    equivalent: bool = False
    suffix: str = ""
    status: str = "missing"


@dataclass
class CarbonAssignment:
    atom_index: int
    predicted_shift: Optional[float]
    experimental_shift: Optional[float]
    error: Optional[float]
    carbon_type: str
    h_count: int
    hydrogens: List[HydrogenAssignment] = field(default_factory=list)
    status: str = "missing"
    label: str = ""
    original_label: str = ""
    manually_edited: bool = False


@dataclass
class AssignmentResult:
    mol: Chem.Mol
    molecule_name: str
    formula: str
    exact_mass: float
    carbons: List[CarbonAssignment]
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class HSQCPoint:
    c_shift: float
    h_shift: float
    intensity: Optional[float] = None


def _numeric_values_from_line(line: str) -> List[float]:
    scrubbed = re.sub(r"\b[CH]\s*-\s*\d+[a-z]?", " ", line or "", flags=re.IGNORECASE)
    scrubbed = re.sub(r"\b\d+\s*H\b", " ", scrubbed, flags=re.IGNORECASE)
    vals: List[float] = []
    for match in re.finditer(r"[-+]?\d+(?:\.\d+)?", scrubbed):
        try:
            vals.append(float(match.group(0)))
        except ValueError:
            continue
    return vals


def parse_float_values(text: str) -> List[float]:
    vals: List[float] = []
    for raw_line in (text or "").replace(";", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        nums = _numeric_values_from_line(line)
        if not nums:
            continue
        if len(nums) == 1:
            vals.append(nums[0])
            continue
        # Assignment tables may be pasted as: label/index, shift.
        if (
            len(nums) == 2
            and abs(nums[0] - round(nums[0])) < 1e-6
            and 0.0 <= nums[1] <= 240.0
        ):
            vals.append(nums[1])
            continue
        # Peak-list rows are commonly: index, intensity, ppm, area, width.
        if (
            len(nums) >= 5
            and abs(nums[0] - round(nums[0])) < 1e-6
            and nums[1] > 1000
            and 0.0 <= nums[2] <= 240.0
        ):
            vals.append(nums[2])
            continue
        vals.extend(nums)
    return vals


def _looks_like_hsqc_shift_pair(a: float, b: float) -> bool:
    if a < 0 or b < 0:
        return False
    return ((10.0 <= a <= 240.0 and b <= 15.0) or (10.0 <= b <= 240.0 and a <= 15.0))


def _coerce_hsqc_pair(a: float, b: float) -> Tuple[float, float]:
    if a < b:
        return float(b), float(a)
    return float(a), float(b)


def _is_row_index(value: float) -> bool:
    return value >= 0 and abs(value - round(value)) < 1e-6


def parse_hsqc_points(text: str) -> List[HSQCPoint]:
    points: List[HSQCPoint] = []
    for line in (text or "").splitlines():
        nums = _numeric_values_from_line(line)
        if len(nums) >= 2:
            intensity: Optional[float] = None
            c_val: Optional[float] = None
            h_val: Optional[float] = None
            if len(nums) >= 4 and _is_row_index(nums[0]) and _looks_like_hsqc_shift_pair(nums[1], nums[2]):
                c_val, h_val = _coerce_hsqc_pair(nums[1], nums[2])
                intensity = float(nums[3])
            elif len(nums) >= 3 and _looks_like_hsqc_shift_pair(nums[0], nums[1]):
                c_val, h_val = _coerce_hsqc_pair(nums[0], nums[1])
                intensity = float(nums[2])
            elif len(nums) >= 3 and _is_row_index(nums[0]) and _looks_like_hsqc_shift_pair(nums[1], nums[2]):
                c_val, h_val = _coerce_hsqc_pair(nums[1], nums[2])
            elif _looks_like_hsqc_shift_pair(nums[-2], nums[-1]):
                c_val, h_val = _coerce_hsqc_pair(nums[-2], nums[-1])
            if c_val is not None and h_val is not None:
                points.append(HSQCPoint(c_shift=float(c_val), h_shift=float(h_val), intensity=intensity))
    return points


def _parse_shift_property(mol: Chem.Mol, prop_name: str) -> Dict[int, float]:
    shifts: Dict[int, float] = {}
    if not mol.HasProp(prop_name):
        return shifts
    for line in mol.GetProp(prop_name).splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.search(r"\[(\d+)\]\s+([-+]?\d+(?:\.\d+)?)", line)
        if not match:
            continue
        atom_index = int(match.group(1)) - 1
        shifts[atom_index] = float(match.group(2))
    return shifts


def _linear_assignment(cost: List[List[float]]) -> List[Tuple[int, int]]:
    if not cost or not cost[0]:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost)
        return [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]
    except Exception:
        pairs = []
        used_r = set()
        used_c = set()
        flat = sorted((value, r, c) for r, row in enumerate(cost) for c, value in enumerate(row))
        for _value, r, c in flat:
            if r in used_r or c in used_c:
                continue
            used_r.add(r)
            used_c.add(c)
            pairs.append((r, c))
            if len(used_r) == len(cost) or len(used_c) == len(cost[0]):
                break
        return pairs


def _nearest_assignment(predicted: Sequence[Optional[float]], experimental: Sequence[float]) -> Dict[int, int]:
    pred_items = [(i, float(v)) for i, v in enumerate(predicted) if v is not None]
    if not pred_items or not experimental:
        return {}
    cost = [[abs(pred - float(exp)) for exp in experimental] for _i, pred in pred_items]
    out: Dict[int, int] = {}
    for row, col in _linear_assignment(cost):
        original_i = pred_items[row][0]
        out[original_i] = col
    return out


def _normalized_hsqc_assignment(
    predicted_points: Sequence[Tuple[int, int, Optional[float], Optional[float], int]],
    experimental_points: Sequence[HSQCPoint],
    c_tolerance: float,
    h_tolerance: float,
) -> Dict[Tuple[int, int], int]:
    valid_pred = [
        (ci, gi, float(pc), float(ph), int(h_count))
        for ci, gi, pc, ph, h_count in predicted_points
        if pc is not None and ph is not None
    ]
    if not valid_pred or not experimental_points:
        return {}
    c_scale = max(float(c_tolerance), 1e-9)
    h_scale = max(float(h_tolerance), 1e-9)
    c_weight = 0.7
    h_weight = 0.3
    impossible_cost = 1e9

    def compatible_by_intensity(point: HSQCPoint, h_count: int) -> bool:
        return not (point.intensity is not None and point.intensity < 0 and h_count != 2)

    cost = []
    for _ci, _gi, pc, ph, h_count in valid_pred:
        row = []
        for point in experimental_points:
            if not compatible_by_intensity(point, h_count):
                row.append(impossible_cost)
                continue
            c_distance = abs(pc - point.c_shift) / c_scale
            h_distance = abs(ph - point.h_shift) / h_scale
            row.append(c_weight * c_distance + h_weight * h_distance)
        cost.append(row)
    out: Dict[Tuple[int, int], int] = {}
    for row, col in _linear_assignment(cost):
        if cost[row][col] >= impossible_cost / 2:
            continue
        ci, gi, _pc, _ph, _h_count = valid_pred[row]
        out[(ci, gi)] = col
    return out


def _carbon_type(h_count: int) -> str:
    return {0: "s", 1: "d", 2: "t", 3: "q"}.get(int(h_count), f"H{h_count}")


def _status_for_error(error: Optional[float], yellow: float, red: float) -> str:
    if error is None:
        return "missing"
    if error > red:
        return "red"
    if error > yellow:
        return "yellow"
    return "ok"


def _carbon_status(
    predicted: Optional[float],
    experimental: Optional[float],
    error: Optional[float],
    all_predicted: Sequence[Optional[float]],
    all_experimental: Sequence[float],
    yellow_threshold: float,
    red_threshold: float,
    ambiguity_window: float,
    ambiguity_mean_error: float,
    local_window: float,
) -> str:
    if predicted is None or experimental is None or error is None:
        return "missing"
    if error > red_threshold:
        return "red"

    exp_near = sorted(abs(float(v) - predicted) for v in all_experimental if abs(float(v) - predicted) <= ambiguity_window)
    if len(exp_near) >= 2 and (sum(exp_near[:2]) / 2.0) > ambiguity_mean_error:
        return "yellow"

    pred_in_local = [
        v for v in all_predicted
        if v is not None and abs(float(v) - float(experimental)) <= local_window
    ]
    exp_in_local = [v for v in all_experimental if abs(float(v) - float(predicted)) <= local_window]
    if len(pred_in_local) <= 1 and len(exp_in_local) <= 1 and error > yellow_threshold:
        return "yellow"

    return _status_for_error(error, yellow_threshold, red_threshold)


def _hydrogen_groups(attached_h: List[int], h_shifts: Dict[int, float], eq_tolerance: float) -> List[HydrogenAssignment]:
    if len(attached_h) == 3:
        shift_values = [h_shifts[h] for h in attached_h if h in h_shifts]
        pred = sum(float(s) for s in shift_values) / len(shift_values) if shift_values else None
        return [
            HydrogenAssignment(
                atom_indices=list(attached_h),
                predicted_shift=pred,
                equivalent=True,
                suffix="",
            )
        ]

    with_shifts = [(h, h_shifts.get(h)) for h in attached_h]
    groups: List[List[Tuple[int, Optional[float]]]] = []
    for h_idx, shift in sorted(with_shifts, key=lambda item: (9999.0 if item[1] is None else float(item[1]), item[0])):
        placed = False
        if shift is not None:
            for group in groups:
                group_shifts = [s for _h, s in group if s is not None]
                if group_shifts and abs(float(shift) - sum(group_shifts) / len(group_shifts)) <= eq_tolerance:
                    group.append((h_idx, shift))
                    placed = True
                    break
        if not placed:
            groups.append([(h_idx, shift)])

    out: List[HydrogenAssignment] = []
    non_equiv_count = len(groups)
    for idx, group in enumerate(groups):
        atom_indices = [h for h, _s in group]
        shift_values = [s for _h, s in group if s is not None]
        pred = sum(float(s) for s in shift_values) / len(shift_values) if shift_values else None
        equivalent = len(group) > 1
        suffix = "" if equivalent or non_equiv_count == 1 else chr(ord("a") + idx)
        out.append(
            HydrogenAssignment(
                atom_indices=atom_indices,
                predicted_shift=pred,
                equivalent=equivalent,
                suffix=suffix,
            )
        )
    return out


def _prepare_loaded_molecule(mol: Chem.Mol) -> Chem.Mol:
    out = Chem.Mol(mol)
    try:
        out.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    return out


def load_predicted_molecule(
    sdf_path: str | Path,
    molecule_index: int = 1,
    molecule_id: str | None = None,
) -> Chem.Mol:
    path = Path(sdf_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"SDF file does not exist: {path}")
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    target_id = str(molecule_id).strip() if molecule_id is not None and str(molecule_id).strip() else None
    for idx, mol in enumerate(supplier, start=1):
        if mol is None:
            continue
        if target_id is not None:
            mol_id = mol.GetProp("ID").strip() if mol.HasProp("ID") else ""
            if mol_id == target_id:
                return _prepare_loaded_molecule(mol)
            continue
        if idx == int(molecule_index):
            return _prepare_loaded_molecule(mol)
    if target_id is not None:
        raise ValueError(f"SDF does not contain a molecule with ID={target_id}.")
    raise ValueError(f"SDF contains fewer than {molecule_index} molecule(s).")


def list_sdf_molecule_ids(sdf_path: str | Path) -> List[str]:
    path = Path(sdf_path).expanduser()
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    ids: List[str] = []
    for idx, mol in enumerate(supplier, start=1):
        if mol is None:
            continue
        mol_id = mol.GetProp("ID").strip() if mol.HasProp("ID") else str(idx)
        ids.append(mol_id or str(idx))
    return ids


def build_assignment(
    *,
    sdf_path: str | Path,
    molecule_index: int,
    molecule_id: str | None = None,
    carbon_text: str,
    proton_text: str,
    hsqc_text: str,
    use_hsqc: bool,
    c_tolerance: float,
    h_tolerance: float,
    c_yellow_threshold: float,
    c_red_threshold: float,
    h_yellow_threshold: float,
    h_red_threshold: float,
    ambiguity_window: float,
    ambiguity_mean_error: float,
    local_window: float,
    equivalence_tolerance: float,
) -> AssignmentResult:
    mol = load_predicted_molecule(sdf_path, molecule_index=molecule_index, molecule_id=molecule_id)
    c_shifts = _parse_shift_property(mol, "Predicted 13C shifts")
    h_shifts = _parse_shift_property(mol, "HydrogenShifts")
    experimental_c = parse_float_values(carbon_text)
    experimental_h = parse_float_values(proton_text)
    experimental_hsqc = parse_hsqc_points(hsqc_text) if use_hsqc else []

    warnings: List[str] = []
    if not c_shifts:
        warnings.append("No predicted 13C shifts were found in the selected SDF molecule.")
    if use_hsqc and not experimental_hsqc:
        warnings.append("HSQC matching is enabled, but no valid HSQC points were parsed.")

    carbons: List[CarbonAssignment] = []
    carbon_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
    predicted_c_values = [c_shifts.get(atom.GetIdx()) for atom in carbon_atoms]
    c_assign = _nearest_assignment(predicted_c_values, experimental_c)

    for ci, atom in enumerate(carbon_atoms):
        atom_idx = atom.GetIdx()
        pred_c = c_shifts.get(atom_idx)
        exp_c = experimental_c[c_assign[ci]] if ci in c_assign else None
        c_error = abs(float(pred_c) - float(exp_c)) if pred_c is not None and exp_c is not None else None
        attached_h = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 1]
        h_groups = _hydrogen_groups(attached_h, h_shifts, equivalence_tolerance)
        label = str(atom_idx + 1)
        carbons.append(
            CarbonAssignment(
                atom_index=atom_idx,
                predicted_shift=pred_c,
                experimental_shift=exp_c,
                error=c_error,
                carbon_type=_carbon_type(len(attached_h)),
                h_count=len(attached_h),
                hydrogens=h_groups,
                status="missing",
                label=label,
                original_label=label,
            )
        )

    hsqc_predictions: List[Tuple[int, int, Optional[float], Optional[float], int]] = []
    for ci, carbon in enumerate(carbons):
        for gi, group in enumerate(carbon.hydrogens):
            if carbon.h_count != 2 and gi > 0:
                continue
            hsqc_predictions.append((ci, gi, carbon.predicted_shift, group.predicted_shift, carbon.h_count))
    hsqc_assign = _normalized_hsqc_assignment(hsqc_predictions, experimental_hsqc, c_tolerance, h_tolerance)

    if use_hsqc and experimental_hsqc:
        best_hsqc_carbon: Dict[int, Tuple[float, float]] = {}
        for ci, gi, _pc, _ph, _h_count in hsqc_predictions:
            if (ci, gi) not in hsqc_assign:
                continue
            point = experimental_hsqc[hsqc_assign[(ci, gi)]]
            exp_c, exp_h = point.c_shift, point.h_shift
            carbon = carbons[ci]
            group = carbon.hydrogens[gi]
            group.experimental_shift = exp_h
            group.error = abs(float(group.predicted_shift) - exp_h) if group.predicted_shift is not None else None
            group.status = _status_for_error(group.error, h_yellow_threshold, h_red_threshold)
            if carbon.predicted_shift is not None:
                c_error = abs(float(carbon.predicted_shift) - exp_c)
                if ci not in best_hsqc_carbon or c_error < best_hsqc_carbon[ci][0]:
                    best_hsqc_carbon[ci] = (c_error, exp_c)
        for ci, (c_error, exp_c) in best_hsqc_carbon.items():
            carbons[ci].experimental_shift = exp_c
            carbons[ci].error = c_error
    else:
        all_h_groups: List[Tuple[int, int, Optional[float]]] = []
        for ci, carbon in enumerate(carbons):
            for gi, group in enumerate(carbon.hydrogens):
                all_h_groups.append((ci, gi, group.predicted_shift))
        h_assign = _nearest_assignment([item[2] for item in all_h_groups], experimental_h)
        for row_idx, exp_idx in h_assign.items():
            ci, gi, _pred = all_h_groups[row_idx]
            group = carbons[ci].hydrogens[gi]
            group.experimental_shift = experimental_h[exp_idx]
            group.error = abs(float(group.predicted_shift) - group.experimental_shift) if group.predicted_shift is not None else None
            group.status = _status_for_error(group.error, h_yellow_threshold, h_red_threshold)

    for carbon in carbons:
        carbon.status = _carbon_status(
            carbon.predicted_shift,
            carbon.experimental_shift,
            carbon.error,
            [c.predicted_shift for c in carbons],
            experimental_c,
            c_yellow_threshold,
            c_red_threshold,
            ambiguity_window,
            ambiguity_mean_error,
            local_window,
        )
        for group in carbon.hydrogens:
            if group.status == "missing":
                group.status = _status_for_error(group.error, h_yellow_threshold, h_red_threshold)

    try:
        formula = rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        formula = ""
    try:
        exact_mass = Descriptors.ExactMolWt(mol)
    except Exception:
        exact_mass = 0.0
    name = mol.GetProp("ID") if mol.HasProp("ID") else mol.GetProp("_Name") if mol.HasProp("_Name") else Path(sdf_path).stem

    return AssignmentResult(
        mol=mol,
        molecule_name=name,
        formula=formula,
        exact_mass=float(exact_mass or 0.0),
        carbons=carbons,
        warnings=warnings,
    )


def prepare_display_molecule(mol: Chem.Mol, carbon_labels: Dict[int, str]) -> Chem.Mol:
    draw_mol = Chem.Mol(mol)
    try:
        draw_mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    try:
        AllChem.Compute2DCoords(draw_mol)
    except Exception:
        pass
    for atom in draw_mol.GetAtoms():
        atom.ClearProp("atomNote") if atom.HasProp("atomNote") else None
        if atom.GetAtomicNum() == 6 and atom.GetIdx() in carbon_labels:
            atom.SetProp("atomNote", str(carbon_labels[atom.GetIdx()]))
    return draw_mol


def export_assignment_text(
    result: AssignmentResult,
    *,
    carbon_mhz: str,
    proton_mhz: str,
    solvent: str,
    carbon_prefix: str,
    proton_prefix: str,
) -> str:
    carbon_parts = []
    proton_parts = []
    sorted_carbons = sorted(result.carbons, key=lambda c: (float("inf") if c.experimental_shift is None else c.experimental_shift))
    for carbon in sorted_carbons:
        label = str(carbon.label or carbon.original_label)
        c_export_label = f"{carbon_prefix}{label}"
        if carbon.experimental_shift is not None:
            carbon_parts.append(f"{carbon.experimental_shift:.1f} ({c_export_label}, {carbon.carbon_type})")
        for group in carbon.hydrogens:
            if group.experimental_shift is None:
                continue
            h_label = f"{proton_prefix}{label}{group.suffix}"
            n_h = len(group.atom_indices)
            proton_parts.append((group.experimental_shift, f"{group.experimental_shift:.2f} ({n_h}H, {h_label})"))

    proton_parts_text = ", ".join(text for _shift, text in sorted(proton_parts, key=lambda item: item[0]))
    carbon_parts_text = ", ".join(carbon_parts)
    formula = result.formula or "Formula not available"
    mz = f"{result.exact_mass:.4f}" if result.exact_mass else "not available"
    h_header = f"1H-NMR({proton_mhz}MHz, {solvent}) δH"
    c_header = f"13C-NMR({carbon_mhz}MHz, {solvent}) δC"
    nmr_sections = []
    if proton_parts_text:
        nmr_sections.append(f"{h_header}: {proton_parts_text}")
    if carbon_parts_text:
        nmr_sections.append(f"{c_header}: {carbon_parts_text}")
    section_text = "; ".join(nmr_sections) if nmr_sections else "No assigned NMR shifts."
    return f"{formula}, calculated m/z: {mz}, {section_text}."


def _replacement_maps(result: AssignmentResult) -> Tuple[Dict[int, float], Dict[int, float]]:
    carbon_values: Dict[int, float] = {}
    hydrogen_values: Dict[int, float] = {}
    for carbon in result.carbons:
        if carbon.experimental_shift is not None:
            carbon_values[int(carbon.atom_index)] = float(carbon.experimental_shift)
        for group in carbon.hydrogens:
            if group.experimental_shift is None:
                continue
            for atom_idx in group.atom_indices:
                hydrogen_values[int(atom_idx)] = float(group.experimental_shift)
    return carbon_values, hydrogen_values


def _replace_shift_property(mol: Chem.Mol, prop_name: str, replacements: Dict[int, float]) -> None:
    if not mol.HasProp(prop_name) or not replacements:
        return
    new_lines: List[str] = []
    for line in mol.GetProp(prop_name).splitlines():
        match = re.search(r"\[(\d+)\]\s+([-+]?\d+(?:\.\d+)?)", line)
        if not match:
            new_lines.append(line)
            continue
        atom_idx = int(match.group(1)) - 1
        if atom_idx not in replacements:
            new_lines.append(line)
            continue
        start, end = match.span(2)
        new_lines.append(f"{line[:start]}{replacements[atom_idx]:.2f}{line[end:]}")
    mol.SetProp(prop_name, "\n".join(new_lines))


def _replace_typed_carbon_property(mol: Chem.Mol, prop_name: str, replacements: Dict[int, float]) -> None:
    if not mol.HasProp(prop_name) or not replacements:
        return
    new_lines: List[str] = []
    for line in mol.GetProp(prop_name).splitlines():
        match = re.search(r"^\s*(\d+)\s+([-+]?\d+(?:\.\d+)?)", line)
        if not match:
            new_lines.append(line)
            continue
        atom_idx = int(match.group(1)) - 1
        if atom_idx not in replacements:
            new_lines.append(line)
            continue
        start, end = match.span(2)
        new_lines.append(f"{line[:start]}{replacements[atom_idx]:.2f}{line[end:]}")
    mol.SetProp(prop_name, "\n".join(new_lines))


def assigned_molecule_from_result(result: AssignmentResult) -> Chem.Mol:
    mol = Chem.Mol(result.mol)
    carbon_values, hydrogen_values = _replacement_maps(result)
    _replace_shift_property(mol, "Predicted 13C shifts", carbon_values)
    for prop_name in ("Quaternaries", "Tertiaries", "Secondaries", "Primaries"):
        _replace_typed_carbon_property(mol, prop_name, carbon_values)
    _replace_shift_property(mol, "HydrogenShifts", hydrogen_values)
    return mol


def write_assigned_sdf(
    *,
    source_sdf_path: str | Path,
    results: Iterable[AssignmentResult],
    output_sdf_path: str | Path,
) -> Path:
    source = Path(source_sdf_path).expanduser()
    output = Path(output_sdf_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    by_id = {str(result.molecule_name): result for result in results}
    supplier = Chem.SDMolSupplier(str(source), removeHs=False, sanitize=False)
    writer = Chem.SDWriter(str(output))
    try:
        for idx, mol in enumerate(supplier, start=1):
            if mol is None:
                continue
            mol_id = mol.GetProp("ID").strip() if mol.HasProp("ID") else str(idx)
            if mol_id in by_id:
                out_mol = assigned_molecule_from_result(by_id[mol_id])
            else:
                out_mol = Chem.Mol(mol)
            writer.write(out_mol)
    finally:
        writer.close()
    return output
