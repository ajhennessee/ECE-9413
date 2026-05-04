"""
Assignment 2 student implementation reference skeleton.

This file documents the frozen student-facing API.
Only 32-bit kernels are compulsory in the base track.
64-bit and 128-bit kernels are intentionally left unimplemented here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)


# -----------------------------------------------------------------------------
# 32-bit primitives (compulsory)
# -----------------------------------------------------------------------------
#
# Strategy: promote operands to uint64 so that a + b, a + q - b, and a * b
# cannot overflow when q < 2**32. Reduce mod q, then cast back to uint32.
# This is JIT-friendly and branchless.
# -----------------------------------------------------------------------------

def _to_u64(x):
    return jnp.asarray(x, dtype=jnp.uint64)


def mod_add_32(a, b, q):
    """Return (a + b) mod q for the 32-bit track."""
    a64, b64, q64 = _to_u64(a), _to_u64(b), _to_u64(q)
    return ((a64 + b64) % q64).astype(jnp.uint32)


def mod_sub_32(a, b, q):
    """Return (a - b) mod q for the 32-bit track.

    Operands are unsigned, so we add q before subtracting to guarantee a
    non-negative intermediate. (a + q - b) is < 2*q < 2**33, safe in uint64.
    """
    a64, b64, q64 = _to_u64(a), _to_u64(b), _to_u64(q)
    return ((a64 + q64 - b64) % q64).astype(jnp.uint32)


def mod_mul_32(a, b, q):
    """Return (a * b) mod q for the 32-bit track.

    Two uint32 values multiplied stay within uint64, so a single % suffices.
    """
    a64, b64, q64 = _to_u64(a), _to_u64(b), _to_u64(q)
    return ((a64 * b64) % q64).astype(jnp.uint32)


# -----------------------------------------------------------------------------
# 64-bit primitives (optional, left for future implementation)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 64-bit support: q can approach 2**64, so we cannot promote to a wider native
# integer to absorb a multiplication. Strategy:
#   * add/sub: stay in uint64 and use overflow-aware comparisons.
#   * mul: schoolbook on 32-bit halves to get the full 128-bit product as
#     (hi, lo), then a shift-and-subtract reduction mod q over 128 bits.
# -----------------------------------------------------------------------------

_U64_MASK_LO32 = jnp.uint64(0xFFFFFFFF)
_U64_SHIFT32 = jnp.uint64(32)


def _u64(x):
    return jnp.asarray(x, dtype=jnp.uint64)


def mod_add_64(a, b, q):
    """Return (a + b) mod q where a, b, q are uint64 with a, b < q < 2**64.

    Compute s = a + b mod 2**64. If s overflowed (s < a) OR s >= q, subtract q.
    Both conditions are detected without going wider than uint64.
    """
    a, b, q = _u64(a), _u64(b), _u64(q)
    s = a + b
    overflowed = s < a            # wrapped past 2**64
    needs_sub = overflowed | (s >= q)
    return jnp.where(needs_sub, s - q, s).astype(jnp.uint64)


def mod_sub_64(a, b, q):
    """Return (a - b) mod q. If b > a, the wrap-around result is exactly
    a - b + 2**64, which differs from the desired a - b + q by a constant;
    so we just add q in that case (one branch via where).
    """
    a, b, q = _u64(a), _u64(b), _u64(q)
    return jnp.where(a >= b, a - b, a + (q - b)).astype(jnp.uint64)


def _mul_64_to_128(a, b):
    """Full 64x64 -> 128 multiply, returned as (hi, lo) uint64 pair.

    Split a = a_hi*2**32 + a_lo, similarly b. Then
        a*b = a_hi*b_hi * 2**64
            + (a_hi*b_lo + a_lo*b_hi) * 2**32
            + a_lo*b_lo
    Each partial product fits in uint64 (32x32 -> at most 64 bits).
    Combine with explicit carry tracking.
    """
    a_lo = a & _U64_MASK_LO32
    a_hi = a >> _U64_SHIFT32
    b_lo = b & _U64_MASK_LO32
    b_hi = b >> _U64_SHIFT32

    ll = a_lo * b_lo                 # bits [0, 64)
    lh = a_lo * b_hi                 # bits [32, 96)
    hl = a_hi * b_lo                 # bits [32, 96)
    hh = a_hi * b_hi                 # bits [64, 128)

    # Sum the two middle terms; track overflow into bit 64.
    mid = lh + hl
    mid_carry = (mid < lh).astype(jnp.uint64) << _U64_SHIFT32  # 1<<64 if wrapped, else 0

    # Add mid<<32 to ll, propagating any carry into hi.
    mid_lo = (mid & _U64_MASK_LO32) << _U64_SHIFT32
    mid_hi = mid >> _U64_SHIFT32

    lo = ll + mid_lo
    lo_carry = (lo < ll).astype(jnp.uint64)

    hi = hh + mid_hi + mid_carry + lo_carry
    return hi, lo


def _reduce_128_mod_q_64(hi, lo, q):
    """Reduce a 128-bit value (hi, lo) modulo q < 2**64 using shift-and-subtract.

    Process 128 bits MSB-first, maintaining a 64-bit residue r in [0, q).
    Per bit: r = (r << 1) | next_bit; if r overflowed (carry out of the shift)
    or r >= q, subtract q. We split the work into the 64 high bits then the
    64 low bits; this is O(128) ops per multiply but stays in uint64 the
    whole time and is fully JIT-friendly.
    """
    q = _u64(q)
    r = jnp.uint64(0)

    def step(r, src_word, bit_idx):
        # bit_idx in [0, 63], MSB-first within src_word.
        bit = (src_word >> jnp.uint64(bit_idx)) & jnp.uint64(1)
        top_bit_of_r = (r >> jnp.uint64(63)) & jnp.uint64(1)   # will shift out
        r_shifted = (r << jnp.uint64(1)) | bit
        # After shift, the "true" value is top_bit_of_r * 2**64 + r_shifted.
        # Subtract q if (top_bit_of_r != 0) or (r_shifted >= q).
        needs_sub = (top_bit_of_r != jnp.uint64(0)) | (r_shifted >= q)
        return jnp.where(needs_sub, r_shifted - q, r_shifted)

    # High word first, MSB to LSB.
    for i in range(63, -1, -1):
        r = step(r, hi, i)
    # Low word, MSB to LSB.
    for i in range(63, -1, -1):
        r = step(r, lo, i)
    return r.astype(jnp.uint64)


def mod_mul_64(a, b, q):
    """Return (a * b) mod q for uint64 operands with q < 2**64."""
    a, b, q = _u64(a), _u64(b), _u64(q)
    hi, lo = _mul_64_to_128(a, b)
    return _reduce_128_mod_q_64(hi, lo, q)


# -----------------------------------------------------------------------------
# 128-bit primitives (optional, left for future implementation)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 128-bit support: values represented as a pair (hi, lo) of uint64, with the
# logical value being hi * 2**64 + lo and 0 <= value < q < 2**128.
# Arithmetic propagates carries/borrows between limbs; multiplication produces
# a 256-bit intermediate that we reduce by shift-and-subtract over 256 bits.
#
# Convention: each "value" passed in is a 2-tuple/list of jax uint64 arrays
# (or scalars), interpreted as (hi, lo). q is similarly a (q_hi, q_lo) pair.
# The dispatch wrappers higher up in the file will route shape conventions
# to/from this representation.
# -----------------------------------------------------------------------------


def _split_pair(x):
    """Accept a (hi, lo) tuple/list or a uint64 array of shape (..., 2).
    Returns (hi, lo) as uint64 jax arrays.
    """
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return _u64(x[0]), _u64(x[1])
    arr = jnp.asarray(x, dtype=jnp.uint64)
    # Shape (..., 2): [..., 0] = hi, [..., 1] = lo.
    return arr[..., 0], arr[..., 1]


def _ge_128(a_hi, a_lo, b_hi, b_lo):
    """Compare two 128-bit values: True iff (a_hi, a_lo) >= (b_hi, b_lo)."""
    return (a_hi > b_hi) | ((a_hi == b_hi) & (a_lo >= b_lo))


def _add_128(a_hi, a_lo, b_hi, b_lo):
    """128-bit add mod 2**128, returning (hi, lo) and the carry out as uint64."""
    lo = a_lo + b_lo
    carry_lo = (lo < a_lo).astype(jnp.uint64)
    hi = a_hi + b_hi + carry_lo
    # Carry out of bit 128: either b_hi+carry_lo overflowed, or that sum + a_hi did.
    inner = b_hi + carry_lo
    inner_carry = (inner < b_hi).astype(jnp.uint64)
    outer_carry = (hi < a_hi).astype(jnp.uint64)
    carry_out = inner_carry | outer_carry
    return hi, lo, carry_out


def _sub_128(a_hi, a_lo, b_hi, b_lo):
    """128-bit subtract: returns (hi, lo) of (a - b) mod 2**128."""
    lo = a_lo - b_lo
    borrow = (a_lo < b_lo).astype(jnp.uint64)
    hi = a_hi - b_hi - borrow
    return hi, lo


def mod_add_128(a, b, q):
    """Return (a + b) mod q for 128-bit operands with a, b < q < 2**128."""
    a_hi, a_lo = _split_pair(a)
    b_hi, b_lo = _split_pair(b)
    q_hi, q_lo = _split_pair(q)

    s_hi, s_lo, carry = _add_128(a_hi, a_lo, b_hi, b_lo)
    overflowed = carry != jnp.uint64(0)
    ge_q = _ge_128(s_hi, s_lo, q_hi, q_lo)
    needs_sub = overflowed | ge_q

    sub_hi, sub_lo = _sub_128(s_hi, s_lo, q_hi, q_lo)
    out_hi = jnp.where(needs_sub, sub_hi, s_hi)
    out_lo = jnp.where(needs_sub, sub_lo, s_lo)
    return out_hi, out_lo


def mod_sub_128(a, b, q):
    """Return (a - b) mod q for 128-bit operands."""
    a_hi, a_lo = _split_pair(a)
    b_hi, b_lo = _split_pair(b)
    q_hi, q_lo = _split_pair(q)

    a_ge_b = _ge_128(a_hi, a_lo, b_hi, b_lo)
    direct_hi, direct_lo = _sub_128(a_hi, a_lo, b_hi, b_lo)
    # Else compute a + (q - b). q > b is guaranteed since b < q.
    qmb_hi, qmb_lo = _sub_128(q_hi, q_lo, b_hi, b_lo)
    alt_hi, alt_lo, _carry = _add_128(a_hi, a_lo, qmb_hi, qmb_lo)

    out_hi = jnp.where(a_ge_b, direct_hi, alt_hi)
    out_lo = jnp.where(a_ge_b, direct_lo, alt_lo)
    return out_hi, out_lo


def _mul_128_to_256(a_hi, a_lo, b_hi, b_lo):
    """Full 128x128 -> 256 multiply.

    Returns four uint64 limbs (w3, w2, w1, w0) where the value is
        w3 * 2**192 + w2 * 2**128 + w1 * 2**64 + w0.

    Built from four 64x64 -> 128 partial products (using _mul_64_to_128),
    summed at the right limb offsets with carry propagation.
    """
    # Partials: pX_Y = (a's X part) * (b's Y part), result is 128 bits.
    p_ll_hi, p_ll_lo = _mul_64_to_128(a_lo, b_lo)  # contributes to limbs (1, 0)
    p_lh_hi, p_lh_lo = _mul_64_to_128(a_lo, b_hi)  # contributes to limbs (2, 1)
    p_hl_hi, p_hl_lo = _mul_64_to_128(a_hi, b_lo)  # contributes to limbs (2, 1)
    p_hh_hi, p_hh_lo = _mul_64_to_128(a_hi, b_hi)  # contributes to limbs (3, 2)

    # Limb 0 is just p_ll_lo.
    w0 = p_ll_lo

    # Limb 1: p_ll_hi + p_lh_lo + p_hl_lo, carry into limb 2.
    s = p_ll_hi + p_lh_lo
    c1 = (s < p_ll_hi).astype(jnp.uint64)
    s2 = s + p_hl_lo
    c1 += (s2 < s).astype(jnp.uint64)
    w1 = s2

    # Limb 2: p_lh_hi + p_hl_hi + p_hh_lo + c1, carry into limb 3.
    t = p_lh_hi + p_hl_hi
    c2 = (t < p_lh_hi).astype(jnp.uint64)
    t2 = t + p_hh_lo
    c2 += (t2 < t).astype(jnp.uint64)
    t3 = t2 + c1
    c2 += (t3 < t2).astype(jnp.uint64)
    w2 = t3

    # Limb 3: p_hh_hi + c2. Cannot overflow because the true product is < 2**256.
    w3 = p_hh_hi + c2

    return w3, w2, w1, w0


def _reduce_256_mod_q_128(w3, w2, w1, w0, q_hi, q_lo):
    """Reduce a 256-bit value (w3, w2, w1, w0) modulo q < 2**128.

    Maintain a 128-bit residue r = (r_hi, r_lo). Process 256 bits MSB-first;
    per bit: shift r left by 1 (tracking the bit shifted out of bit 127),
    OR in the next source bit, then conditionally subtract q if the value
    is >= q (counting the shifted-out bit as a 129th high bit).
    """
    r_hi = jnp.uint64(0)
    r_lo = jnp.uint64(0)

    def step(r_hi, r_lo, src_word, bit_idx):
        bit = (src_word >> jnp.uint64(bit_idx)) & jnp.uint64(1)
        top_bit = (r_hi >> jnp.uint64(63)) & jnp.uint64(1)  # will shift out
        # Shift r left by 1 within 128 bits.
        new_hi = (r_hi << jnp.uint64(1)) | (r_lo >> jnp.uint64(63))
        new_lo = (r_lo << jnp.uint64(1)) | bit
        # Subtract q if top_bit is set, or if (new_hi, new_lo) >= q.
        ge_q = _ge_128(new_hi, new_lo, q_hi, q_lo)
        needs_sub = (top_bit != jnp.uint64(0)) | ge_q
        sub_hi, sub_lo = _sub_128(new_hi, new_lo, q_hi, q_lo)
        out_hi = jnp.where(needs_sub, sub_hi, new_hi)
        out_lo = jnp.where(needs_sub, sub_lo, new_lo)
        return out_hi, out_lo

    for word in (w3, w2, w1, w0):
        for i in range(63, -1, -1):
            r_hi, r_lo = step(r_hi, r_lo, word, i)
    return r_hi, r_lo


def mod_mul_128(a, b, q):
    """Return (a * b) mod q for 128-bit operands with q < 2**128."""
    a_hi, a_lo = _split_pair(a)
    b_hi, b_lo = _split_pair(b)
    q_hi, q_lo = _split_pair(q)
    w3, w2, w1, w0 = _mul_128_to_256(a_hi, a_lo, b_hi, b_lo)
    return _reduce_256_mod_q_128(w3, w2, w1, w0, q_hi, q_lo)


# -----------------------------------------------------------------------------
# Frozen dispatch API
# -----------------------------------------------------------------------------

def mod_add(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_add_32(a, b, q)
    if int(bit_width) == 64:
        return mod_add_64(a, b, q)
    if int(bit_width) == 128:
        return mod_add_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def mod_sub(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_sub_32(a, b, q)
    if int(bit_width) == 64:
        return mod_sub_64(a, b, q)
    if int(bit_width) == 128:
        return mod_sub_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def mod_mul(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_mul_32(a, b, q)
    if int(bit_width) == 64:
        return mod_mul_64(a, b, q)
    if int(bit_width) == 128:
        return mod_mul_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def mle_update_32(zero_eval, one_eval, target_eval, *, q):
    """Compulsory 32-bit MLE update.

    Standard linear-extension fold:
        f(r) = (1 - r) * f(0) + r * f(1)
             = f(0) + r * (f(1) - f(0))

    The second form requires one mul and one add (vs two muls in the first
    form) and is the canonical sumcheck binding step. Operates elementwise
    when the inputs are arrays of equal shape.
    """
    diff = mod_sub_32(one_eval, zero_eval, q)
    scaled = mod_mul_32(diff, target_eval, q)
    return mod_add_32(zero_eval, scaled, q)


def mle_update_64(zero_eval, one_eval, target_eval, *, q):
    """64-bit MLE update: f(r) = f(0) + r * (f(1) - f(0)) mod q."""
    diff = mod_sub_64(one_eval, zero_eval, q)
    scaled = mod_mul_64(diff, target_eval, q)
    return mod_add_64(zero_eval, scaled, q)


def mle_update_128(zero_eval, one_eval, target_eval, *, q):
    """Optional 128-bit MLE update."""
    # TODO(student): implement when enabling 128-bit track.
    raise NotImplementedError


def mle_update(zero_eval, one_eval, target_eval, *, q, bit_width=32):
    if int(bit_width) == 32:
        return mle_update_32(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 64:
        return mle_update_64(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 128:
        return mle_update_128(zero_eval, one_eval, target_eval, q=q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


# -----------------------------------------------------------------------------
# Sumcheck (32-bit)
# -----------------------------------------------------------------------------
#
# Input formats:
#   eval_tables : dict[str, jax.Array]
#       Maps variable name -> its MLE table on the boolean hypercube.
#       Each array has shape (2**num_rounds,) and dtype uint32.
#   expression  : list[list[str]]
#       Sum-of-products. Outer list is additive terms; each inner list is the
#       product of the named variables (each name keys eval_tables).
#       Example: [["a", "b"], ["c"]] represents a*b + c.
#
# Per round:
#   1. Split each table T into T0 = T[:half] (x_i = 0) and T1 = T[half:] (x_i = 1).
#   2. The round polynomial g_i(X) is the multivariate expression with the
#      round variable lifted to the formal indeterminate X. Its degree in X
#      equals the max term length in `expression` (each variable appears
#      linearly in its own table, so a k-fold product is degree k in X).
#   3. Evaluate g_i at X = 0, 1, ..., degree by computing each table's linear
#      extension T(j) = T0 + j*(T1 - T0), pushing through the expression
#      pointwise, and summing over the surviving hypercube axis.
#   4. Fold every table with the verifier's challenge r_i via mle_update_32.
# -----------------------------------------------------------------------------

def _table_at_point(zero_half, one_half, j, q):
    """Evaluate the linear extension T(j) = T0 + j*(T1 - T0) mod q.

    `j` is a Python int; specialize 0 and 1 to keep the trace tight.
    """
    if j == 0:
        return zero_half
    if j == 1:
        return one_half
    j_arr = jnp.asarray(j, dtype=jnp.uint32)
    diff = mod_sub_32(one_half, zero_half, q)
    scaled = mod_mul_32(diff, j_arr, q)
    return mod_add_32(zero_half, scaled, q)


def _sum_mod_q(values, q):
    """Reduce a 1-D uint32 array mod q.

    Cast to uint64 first so the running sum cannot overflow for any tractable
    hypercube size (uint64 holds sums of up to ~2**32 uint32 values comfortably).
    """
    acc = values.astype(jnp.uint64).sum()
    return (acc % jnp.uint64(q)).astype(jnp.uint32)


def _evaluate_sum_of_products(expression, values_by_name, q):
    """Evaluate `expression` pointwise on the per-variable arrays.

    `expression` is sum-of-products in list[list[str]] form.
    `values_by_name` is dict[str, array]; arrays must broadcast together.
    """
    if not expression:
        raise ValueError("expression must contain at least one term")

    term_values = []
    for term in expression:
        if not term:
            raise ValueError("expression terms must contain at least one factor")
        prod = values_by_name[term[0]]
        for var in term[1:]:
            prod = mod_mul_32(prod, values_by_name[var], q)
        term_values.append(prod)

    total = term_values[0]
    for tv in term_values[1:]:
        total = mod_add_32(total, tv, q)
    return total

@partial(jax.jit, static_argnames=["q", "expression", "num_rounds"])
def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    """Compulsory 32-bit sumcheck path.

    Returns
    -------
    (claim0, round_evals) : tuple
        claim0 : uint32 scalar
            The initial sumcheck claim: sum of `expression` over the full
            boolean hypercube {0, 1}**num_rounds, reduced mod q.
        round_evals : list[jax.Array]
            One entry per round. Entry i is a uint32 array of shape
            (degree + 1,) holding (g_i(0), g_i(1), ..., g_i(degree)) where
            g_i is the round-i univariate polynomial.
    """
    tables = {
        name: jnp.asarray(arr, dtype=jnp.uint32)
        for name, arr in eval_tables.items()
    }

    degree = max(len(term) for term in expression)
    num_eval_points = degree + 1
    j_vals = jnp.arange(num_eval_points, dtype=jnp.uint32)

    initial_combined = _evaluate_sum_of_products(expression, tables, q)
    claim0 = _sum_mod_q(initial_combined, q)

    round_evals = []

    for round_idx in range(num_rounds):
        zero_halves = {name: t[::2] for name, t in tables.items()}
        one_halves  = {name: t[1::2] for name, t in tables.items()}

        # Compute diff once per table per round; shared across all j evaluations.
        diffs = {
            name: mod_sub_32(one_halves[name], zero_halves[name], q)
            for name in tables
        }

        def eval_at_j(j):
            # T(j) = T0 + j*(T1 - T0), correct for all j including 0 and 1.
            tables_at_j = {
                name: mod_add_32(zero_halves[name], mod_mul_32(diffs[name], j, q), q)
                for name in tables
            }
            combined = _evaluate_sum_of_products(expression, tables_at_j, q)
            return _sum_mod_q(combined, q)

        # Vectorise over all j values in a single fused kernel.
        round_evals.append(jax.vmap(eval_at_j)(j_vals))

        r = challenges[round_idx]
        tables = {
            name: mle_update_32(zero_halves[name], one_halves[name], r, q=q)
            for name in tables
        }

    if round_evals:
        round_evals_array = jnp.stack(round_evals)
    else:
        round_evals_array = jnp.zeros((0, num_eval_points), dtype=jnp.uint32)

    return claim0, round_evals_array


def _table_at_point_64(zero_half, one_half, j, q):
    """64-bit version of _table_at_point: T(j) = T0 + j*(T1 - T0) mod q."""
    if j == 0:
        return zero_half
    if j == 1:
        return one_half
    j_arr = jnp.asarray(j, dtype=jnp.uint64)
    diff = mod_sub_64(one_half, zero_half, q)
    scaled = mod_mul_64(diff, j_arr, q)
    return mod_add_64(zero_half, scaled, q)


def _sum_mod_q_64(values, q):
    """Reduce a 1-D uint64 array mod q.

    Unlike the 32-bit case we cannot widen to a native integer type, so
    fold with mod_add_64 inside a JAX reduction. jnp.add.reduce with a
    custom op isn't directly available; instead we use a scan-style fold
    via lax.fori_loop semantics expressed as a Python-side reduction.

    For typical hypercube sizes this is dispatched once per round.
    """
    q64 = _u64(q)
    flat = values.astype(jnp.uint64)

    def body(i, acc):
        x = flat[i]
        s = acc + x
        # If overflow OR s >= q, subtract q.
        overflowed = s < acc
        needs_sub = overflowed | (s >= q64)
        return jnp.where(needs_sub, s - q64, s)

    return jax.lax.fori_loop(0, flat.shape[0], body, jnp.uint64(0))


def _evaluate_sum_of_products_64(expression, values_by_name, q):
    """Evaluate sum-of-products expression with 64-bit kernels."""
    if not expression:
        raise ValueError("expression must contain at least one term")

    term_values = []
    for term in expression:
        if not term:
            raise ValueError("expression terms must contain at least one factor")
        prod = values_by_name[term[0]]
        for var in term[1:]:
            prod = mod_mul_64(prod, values_by_name[var], q)
        term_values.append(prod)

    total = term_values[0]
    for tv in term_values[1:]:
        total = mod_add_64(total, tv, q)
    return total


def sumcheck_64(eval_tables, *, q, expression, challenges, num_rounds):
    """64-bit sumcheck path. Mirrors sumcheck_32 but routes through 64-bit kernels."""
    tables = {
        name: jnp.asarray(arr, dtype=jnp.uint64)
        for name, arr in eval_tables.items()
    }
    challenges = jnp.asarray(challenges, dtype=jnp.uint64)

    if not tables:
        raise ValueError("eval_tables must be non-empty")
    if not expression:
        raise ValueError("expression must be non-empty")

    degree = max(len(term) for term in expression)
    num_eval_points = degree + 1

    initial_combined = _evaluate_sum_of_products_64(expression, tables, q)
    claim0 = _sum_mod_q_64(initial_combined, q)

    round_evals = []

    for round_idx in range(num_rounds):
        zero_halves = {name: t[::2] for name, t in tables.items()}
        one_halves = {name: t[1::2] for name, t in tables.items()}

        evals_at_j = []
        for j in range(num_eval_points):
            tables_at_j = {
                name: _table_at_point_64(zero_halves[name], one_halves[name], j, q)
                for name in tables
            }
            combined = _evaluate_sum_of_products_64(expression, tables_at_j, q)
            evals_at_j.append(_sum_mod_q_64(combined, q))

        round_evals.append(jnp.stack(evals_at_j))

        r = challenges[round_idx]
        tables = {
            name: mle_update_64(zero_halves[name], one_halves[name], r, q=q)
            for name in tables
        }

    if round_evals:
        round_evals_array = jnp.stack(round_evals)
    else:
        round_evals_array = jnp.zeros((0, num_eval_points), dtype=jnp.uint64)

    return (claim0, round_evals_array)


def sumcheck_128(eval_tables, *, q, expression, challenges, num_rounds):
    """Optional 128-bit sumcheck path."""
    # TODO(student): implement when enabling 128-bit track.
    raise NotImplementedError


def sumcheck(eval_tables, *, q, expression, challenges, num_rounds, bit_width=32):
    """Frozen dispatcher entrypoint used by the harness."""
    expression = tuple(tuple(term) for term in expression)
    q = int(q)
    
    if int(bit_width) == 32:
        return sumcheck_32(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    if int(bit_width) == 64:
        return sumcheck_64(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    if int(bit_width) == 128:
        return sumcheck_128(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    raise ValueError(f"Unsupported bit_width={bit_width}")