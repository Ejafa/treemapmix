from bisect import bisect_left
from typing import List, Sequence
import torch


@torch.jit.script
def _worst_score(row_sum: float, row_min: float, row_max: float, side: float) -> float:
    """
    Aspect-ratio quality metric (lower is better). Equivalent to the classic
    'worst' metric used by squarify:
        max( (side^2 * max_a) / sum^2,  sum^2 / (side^2 * min_a) )
    """
    if row_sum <= 0.0 or row_min <= 0.0 or row_max <= 0.0 or side <= 0.0:
        return 1e18
    s2 = row_sum * row_sum
    w2 = side * side
    r1 = (w2 * row_max) / s2
    r2 = s2 / (w2 * row_min)
    return r1 if r1 > r2 else r2


@torch.jit.script
def _safe_div(num: float, den: float, eps: float) -> float:
    """Safe division for float control-flow (avoids Python max() in TorchScript)."""
    return num / den if den > eps else 0.0


@torch.jit.script
def _squarify_core_cpu(
    sorted_weights_cpu: torch.Tensor, canvas_size: float
) -> torch.Tensor:
    """
    TorchScript squarified treemap on CPU.

    Args:
        sorted_weights_cpu: (K,) CPU tensor of nonnegative weights (float32 recommended).
        canvas_size: square canvas side length.

    Returns:
        (K,4) float32 CPU tensor of rectangles (x, y, w, h) in [0, canvas_size].
    """
    w_in = sorted_weights_cpu.to(dtype=torch.float32)

    K = int(w_in.numel())
    out = torch.zeros((K, 4), dtype=torch.float32, device=w_in.device)

    if K == 0 or canvas_size <= 0.0:
        return out

    w_in = torch.clamp(w_in, min=0.0)

    total_area = canvas_size * canvas_size
    weight_sum = float(torch.sum(w_in).item())
    if weight_sum > 0.0:
        areas = w_in * (total_area / weight_sum)
    else:
        areas = torch.zeros_like(w_in)

    x = 0.0
    y = 0.0
    W = canvas_size
    H = canvas_size

    eps = 1e-12
    start = 0

    while start < K and W > 0.0 and H > 0.0:
        vertical_stack = W < H  # If tall, place a horizontal strip (stack vertically)
        side = W if vertical_stack else H  # Fixed side length for scoring the strip

        if side <= eps:
            break

        a0 = float(areas[start].item())
        row_sum = a0
        row_min = a0
        row_max = a0
        best = _worst_score(row_sum, row_min, row_max, side)

        row_len = 1
        i = start + 1
        while i < K:
            ai = float(areas[i].item())
            new_sum = row_sum + ai
            new_min = row_min if row_min < ai else ai
            new_max = row_max if row_max > ai else ai
            score = _worst_score(new_sum, new_min, new_max, side)

            if score <= best:
                best = score
                row_sum = new_sum
                row_min = new_min
                row_max = new_max
                row_len += 1
                i += 1
            else:
                break

        end = start + row_len

        perm = (
            torch.randperm(row_len, device=areas.device)
            if row_len > 1
            else torch.zeros((1,), dtype=torch.int64, device=areas.device)
        )

        reverse = float(torch.rand((1,), device=areas.device).item()) < 0.5
        from_end = float(torch.rand((1,), device=areas.device).item()) < 0.5

        if vertical_stack:
            row_h = _safe_div(row_sum, W, eps)
            if row_h > H:
                row_h = H  # clamp in case of tiny numerical drift

            row_y = (y + H - row_h) if from_end else y

            if reverse:
                curr_x = x + W
                for k in range(row_len):
                    idx = int(perm[k].item())
                    area_k = float(areas[start + idx].item())

                    rect_w = (
                        (curr_x - x)
                        if (k == row_len - 1)
                        else _safe_div(area_k, row_h, eps)
                    )
                    if rect_w < 0.0:
                        rect_w = 0.0

                    curr_x -= rect_w
                    out[start + idx, 0] = curr_x
                    out[start + idx, 1] = row_y
                    out[start + idx, 2] = rect_w
                    out[start + idx, 3] = row_h
            else:
                curr_x = x
                for k in range(row_len):
                    idx = int(perm[k].item())
                    area_k = float(areas[start + idx].item())

                    rect_w = (
                        ((x + W) - curr_x)
                        if (k == row_len - 1)
                        else _safe_div(area_k, row_h, eps)
                    )
                    if rect_w < 0.0:
                        rect_w = 0.0

                    out[start + idx, 0] = curr_x
                    out[start + idx, 1] = row_y
                    out[start + idx, 2] = rect_w
                    out[start + idx, 3] = row_h
                    curr_x += rect_w

            H -= row_h
            if not from_end:
                y += row_h

        else:
            row_w = _safe_div(row_sum, H, eps)
            if row_w > W:
                row_w = W  # clamp in case of tiny numerical drift

            row_x = (x + W - row_w) if from_end else x

            if reverse:
                curr_y = y + H
                for k in range(row_len):
                    idx = int(perm[k].item())
                    area_k = float(areas[start + idx].item())

                    rect_h = (
                        (curr_y - y)
                        if (k == row_len - 1)
                        else _safe_div(area_k, row_w, eps)
                    )
                    if rect_h < 0.0:
                        rect_h = 0.0

                    curr_y -= rect_h
                    out[start + idx, 0] = row_x
                    out[start + idx, 1] = curr_y
                    out[start + idx, 2] = row_w
                    out[start + idx, 3] = rect_h
            else:
                curr_y = y
                for k in range(row_len):
                    idx = int(perm[k].item())
                    area_k = float(areas[start + idx].item())

                    rect_h = (
                        ((y + H) - curr_y)
                        if (k == row_len - 1)
                        else _safe_div(area_k, row_w, eps)
                    )
                    if rect_h < 0.0:
                        rect_h = 0.0

                    out[start + idx, 0] = row_x
                    out[start + idx, 1] = curr_y
                    out[start + idx, 2] = row_w
                    out[start + idx, 3] = rect_h
                    curr_y += rect_h

            W -= row_w
            if not from_end:
                x += row_w

        start = end

    return out


def squarify_core(sorted_weights: torch.Tensor, canvas_size: float) -> torch.Tensor:
    """
    Device-aware wrapper around the TorchScript CPU core.

    - If `sorted_weights` is CUDA, runs the sequential control-flow on CPU and
      returns rectangles on the original CUDA device.
    - If `sorted_weights` is CPU, runs entirely on CPU.

    Args:
        sorted_weights: (K,) tensor, typically sorted descending.
        canvas_size: size of square canvas.

    Returns:
        (K,4) float32 tensor of rectangles on the same device as `sorted_weights`.
    """
    if sorted_weights.is_cuda:
        base_cpu = _squarify_core_cpu(sorted_weights.detach().cpu(), float(canvas_size))
        return base_cpu.to(device=sorted_weights.device)
    return _squarify_core_cpu(sorted_weights, float(canvas_size))


def _cluster_sorted(values: List[float], eps: float) -> List[float]:
    """Cluster sorted floats so near-equal boundaries become identical."""
    if not values:
        return []
    values = sorted(values)
    groups: List[List[float]] = [[values[0]]]
    for v in values[1:]:
        if abs(v - groups[-1][-1]) <= eps:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(sum(g) / len(g)) for g in groups]


def _enforce_strictly_increasing(coords: List[int], size_px: int) -> List[int]:
    """
    Enforce:
      coords[0] == 0, coords[-1] == size_px
      and strictly increasing interior coordinates.

    NOTE: This implies you should not use K larger than the canvas resolution in pixels
    in a way that forces too many unique boundaries.
    """
    if not coords:
        return coords
    coords[0] = 0
    coords[-1] = size_px

    for i in range(1, len(coords) - 1):
        coords[i] = max(coords[i], coords[i - 1] + 1)

    for i in range(len(coords) - 2, 0, -1):
        coords[i] = min(coords[i], coords[i + 1] - 1)

    coords[0] = 0
    coords[-1] = size_px
    for i in range(1, len(coords) - 1):
        coords[i] = max(0, min(size_px, coords[i]))
    return coords


def _nearest_index(sorted_vals: Sequence[float], v: float) -> int:
    """Nearest index in a sorted list."""
    i = bisect_left(sorted_vals, v)
    if i <= 0:
        return 0
    if i >= len(sorted_vals):
        return len(sorted_vals) - 1
    return i if abs(sorted_vals[i] - v) < abs(v - sorted_vals[i - 1]) else (i - 1)


def _layout_to_pixel_boxes(
    layout_xywh: torch.Tensor,
    H: int,
    W: int,
    canvas_size: float = 1.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Convert (K,4) float layout (x,y,w,h) into integer pixel boxes (K,4) [x0,y0,x1,y1],
    using globally-consistent boundary snapping to reduce gaps/overlaps.
    """
    K = int(layout_xywh.shape[0])
    S = float(canvas_size)

    xywh = layout_xywh.detach().cpu().tolist()

    x_edges = [0.0, S]
    y_edges = [0.0, S]
    for i in range(K):
        xi, yi, wi, hi = xywh[i]
        x_edges.extend([float(xi), float(xi + wi)])
        y_edges.extend([float(yi), float(yi + hi)])

    x_edges = [min(S, max(0.0, v)) for v in x_edges]
    y_edges = [min(S, max(0.0, v)) for v in y_edges]

    x_reps = _cluster_sorted(x_edges, eps=eps)
    y_reps = _cluster_sorted(y_edges, eps=eps)

    x_reps[0] = 0.0
    x_reps[-1] = S
    y_reps[0] = 0.0
    y_reps[-1] = S

    x_coords = [int(round((v / S) * W)) for v in x_reps]
    y_coords = [int(round((v / S) * H)) for v in y_reps]
    x_coords = _enforce_strictly_increasing(x_coords, size_px=W)
    y_coords = _enforce_strictly_increasing(y_coords, size_px=H)

    boxes = torch.zeros((K, 4), dtype=torch.int64)
    for i in range(K):
        xi, yi, wi, hi = xywh[i]
        x0f = min(S, max(0.0, float(xi)))
        x1f = min(S, max(0.0, float(xi + wi)))
        y0f = min(S, max(0.0, float(yi)))
        y1f = min(S, max(0.0, float(yi + hi)))

        ix0 = _nearest_index(x_reps, x0f)
        ix1 = _nearest_index(x_reps, x1f)
        iy0 = _nearest_index(y_reps, y0f)
        iy1 = _nearest_index(y_reps, y1f)

        boxes[i, 0] = x_coords[min(ix0, ix1)]
        boxes[i, 1] = y_coords[min(iy0, iy1)]
        boxes[i, 2] = x_coords[max(ix0, ix1)]
        boxes[i, 3] = y_coords[max(iy0, iy1)]

    return boxes


def _fallback_stripes_boxes(weights: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Fallback layout: vertical stripes across full height using weights (K,).
    Produces guaranteed tiling boxes (K,4) [x0,y0,x1,y1].
    """
    K = int(weights.numel())
    w = weights.detach().float().cpu()
    s = float(w.sum())
    if s <= 0:
        w = torch.full((K,), 1.0 / K)
    else:
        w = w / s

    cum = torch.cumsum(w, dim=0)
    xs = [0] + [int(round(float(c) * W)) for c in cum[:-1]] + [W]
    xs = _enforce_strictly_increasing(xs, size_px=W)

    boxes = torch.zeros((K, 4), dtype=torch.int64)
    for i in range(K):
        boxes[i, 0] = xs[i]
        boxes[i, 1] = 0
        boxes[i, 2] = xs[i + 1]
        boxes[i, 3] = H
    return boxes


def _boxes_are_valid_and_tile(boxes: torch.Tensor, H: int, W: int) -> bool:
    if boxes.numel() == 0:
        return False
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    if torch.any(x1 <= x0) or torch.any(y1 <= y0):
        return False
    areas = (x1 - x0) * (y1 - y0)
    return int(areas.sum().item()) == int(H * W)


def _apply_box_symmetry(boxes: torch.Tensor, sym: int, S: int) -> torch.Tensor:
    """
    Apply D4 symmetry to integer pixel boxes on a square canvas size S (S == H == W).
    boxes: (K,4) int64 [x0,y0,x1,y1]
    Returns: (K,4) int64 transformed boxes.
    """
    x0 = boxes[:, 0]
    y0 = boxes[:, 1]
    x1 = boxes[:, 2]
    y1 = boxes[:, 3]

    if sym == 0:  # Identity
        nx0, ny0, nx1, ny1 = x0, y0, x1, y1
    elif sym == 1:  # Rot90 CCW: (x,y)->(y, S-x)
        nx0, ny0 = y0, S - x1
        nx1, ny1 = y1, S - x0
    elif sym == 2:  # Rot180
        nx0, ny0 = S - x1, S - y1
        nx1, ny1 = S - x0, S - y0
    elif sym == 3:  # Rot270 CCW
        nx0, ny0 = S - y1, x0
        nx1, ny1 = S - y0, x1
    elif sym == 4:  # FlipH
        nx0, ny0 = S - x1, y0
        nx1, ny1 = S - x0, y1
    elif sym == 5:  # FlipV
        nx0, ny0 = x0, S - y1
        nx1, ny1 = x1, S - y0
    elif sym == 6:  # Transpose
        nx0, ny0 = y0, x0
        nx1, ny1 = y1, x1
    else:  # 7 AntiTranspose
        nx0, ny0 = S - y1, S - x1
        nx1, ny1 = S - y0, S - x0

    return torch.stack([nx0, ny0, nx1, ny1], dim=1).to(dtype=torch.int64)


def _apply_box_symmetry_rect(
    boxes: torch.Tensor, sym: int, H: int, W: int
) -> torch.Tensor:
    """
    Rect-safe symmetries for non-square inputs. Supports only:
      0 identity, 2 rot180, 4 flipH, 5 flipV
    """
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    if sym == 0:
        nx0, ny0, nx1, ny1 = x0, y0, x1, y1
    elif sym == 2:
        nx0, ny0 = W - x1, H - y1
        nx1, ny1 = W - x0, H - y0
    elif sym == 4:
        nx0, ny0 = W - x1, y0
        nx1, ny1 = W - x0, y1
    elif sym == 5:
        nx0, ny0 = x0, H - y1
        nx1, ny1 = x1, H - y0
    else:
        nx0, ny0, nx1, ny1 = x0, y0, x1, y1

    return torch.stack([nx0, ny0, nx1, ny1], dim=1).to(dtype=torch.int64)
