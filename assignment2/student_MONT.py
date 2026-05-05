%%writefile student.py

"""
Assignment 2 student implementation reference skeleton.

This file documents the frozen student-facing API.
Only 32-bit kernels are compulsory in the base track.
64-bit and 128-bit kernels are intentionally left unimplemented here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

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
    """Return (a - b) mod q for the 32-bit track."""
    a64, b64, q64 = _to_u64(a), _to_u64(b), _to_u64(q)
    return ((a64 + q64 - b64) % q64).astype(jnp.uint32)


def mod_mul_32(a, b, q):
    """Return (a * b) mod q for the 32-bit track."""
    a64, b64, q64 = _to_u64(a), _to_u64(b), _to_u64(q)
    return ((a64 * b64) % q64).astype(jnp.uint32)


# --- 32-bit Montgomery Helpers ---
def get_q_inv_32(q):
    """Compute q_inv = -q^{-1} mod 2^32 using Newton iteration (JAX friendly)."""
    q32 = jnp.asarray(q, dtype=jnp.uint32)
    x = q32
    for _ in range(5):
        x = x * (jnp.uint32(2) - q32 * x)
    return jnp.uint32(0) - x

def mont_mul_32(a, b, q, q_inv):
    """Montgomery multiplication: returns (a * b * R^-1) mod q."""
    a64, b64, q64 = _to_u64(a), _to_u64(b), _to_u64(q)
    q_inv64 = _to_u64(q_inv)
    T = a64 * b64
    m = (T.astype(jnp.uint32) * q_inv64.astype(jnp.uint32)).astype(jnp.uint64)

    # Split T and m * q64 into 32-bit halves to prevent 64-bit overflow
    # since T + m * q64 can exceed uint64 limits when q is close to 2^32.
    T_lo = T & 0xFFFFFFFF
    T_hi = T >> 32

    mq = m * q64
    mq_lo = mq & 0xFFFFFFFF
    mq_hi = mq >> 32

    carry = (T_lo + mq_lo) >> 32
    t = T_hi + mq_hi + carry

    return jnp.where(t >= q64, t - q64, t).astype(jnp.uint32)

def to_montgomery_32(x, q, R_mod_q):
    """Convert standard integer to Montgomery form: x * R mod q."""
    return mod_mul_32(x, R_mod_q, q)

def from_montgomery_32(x, q, q_inv):
    """Convert Montgomery form back to standard: x_bar * R^-1 mod q."""
    return mont_mul_32(x, jnp.uint32(1), q, q_inv)

def mle_update_32_mont(zero_eval, one_eval, target_eval_mont, q, q_inv):
    """MLE update using Montgomery multiplication."""
    diff = mod_sub_32(one_eval, zero_eval, q)
    scaled = mont_mul_32(diff, target_eval_mont, q, q_inv)
    return mod_add_32(zero_eval, scaled, q)


# -----------------------------------------------------------------------------
# 64-bit primitives (optional, left for future implementation)
# -----------------------------------------------------------------------------

_U64_MASK_LO32 = jnp.uint64(0xFFFFFFFF)
_U64_SHIFT32 = jnp.uint64(32)


def _u64(x):
    return jnp.asarray(x, dtype=jnp.uint64)


def mod_add_64(a, b, q):
    """Return (a + b) mod q where a, b, q are uint64 with a, b < q < 2**64."""
    a, b, q = _u64(a), _u64(b), _u64(q)
    s = a + b
    overflowed = s < a
    needs_sub = overflowed | (s >= q)
    return jnp.where(needs_sub, s - q, s).astype(jnp.uint64)


def mod_sub_64(a, b, q):
    """Return (a - b) mod q."""
    a, b, q = _u64(a), _u64(b), _u64(q)
    return jnp.where(a >= b, a - b, a + (q - b)).astype(jnp.uint64)


def _mul_64_to_128(a, b):
    """Full 64x64 -> 128 multiply, returned as (hi, lo) uint64 pair."""
    a_lo = a & _U64_MASK_LO32
    a_hi = a >> _U64_SHIFT32
    b_lo = b & _U64_MASK_LO32
    b_hi = b >> _U64_SHIFT32

    ll = a_lo * b_lo
    lh = a_lo * b_hi
    hl = a_hi * b_lo
    hh = a_hi * b_hi

    mid = lh + hl
    mid_carry = (mid < lh).astype(jnp.uint64) << _U64_SHIFT32

    mid_lo = (mid & _U64_MASK_LO32) << _U64_SHIFT32
    mid_hi = mid >> _U64_SHIFT32

    lo = ll + mid_lo
    lo_carry = (lo < ll).astype(jnp.uint64)

    hi = hh + mid_hi + mid_carry + lo_carry
    return hi, lo


def _reduce_128_mod_q_64(hi, lo, q):
    """Reduce a 128-bit value (hi, lo) modulo q < 2**64 using shift-and-subtract."""
    q = _u64(q)
    r = jnp.uint64(0)

    def step(r, src_word, bit_idx):
        bit = (src_word >> jnp.uint64(bit_idx)) & jnp.uint64(1)
        top_bit_of_r = (r >> jnp.uint64(63)) & jnp.uint64(1)
        r_shifted = (r << jnp.uint64(1)) | bit
        needs_sub = (top_bit_of_r != jnp.uint64(0)) | (r_shifted >= q)
        return jnp.where(needs_sub, r_shifted - q, r_shifted)

    for i in range(63, -1, -1):
        r = step(r, hi, i)
    for i in range(63, -1, -1):
        r = step(r, lo, i)
    return r.astype(jnp.uint64)


def mod_mul_64(a, b, q):
    """Return (a * b) mod q for uint64 operands with q < 2**64."""
    a, b, q = _u64(a), _u64(b), _u64(q)
    hi, lo = _mul_64_to_128(a, b)
    return _reduce_128_mod_q_64(hi, lo, q)


# --- 64-bit Montgomery Helpers ---
def get_q_inv_64(q):
    """Compute q_inv = -q^{-1} mod 2^64 using Newton iteration (JAX friendly)."""
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    x = q64
    for _ in range(6):
        x = x * (jnp.uint64(2) - q64 * x)
    return jnp.uint64(0) - x

def mont_mul_64(a, b, q, q_inv):
    """Montgomery multiplication for 64-bit."""
    a, b, q, q_inv = _u64(a), _u64(b), _u64(q), _u64(q_inv)
    T_hi, T_lo = _mul_64_to_128(a, b)
    m = T_lo * q_inv
    mq_hi, mq_lo = _mul_64_to_128(m, q)

    lo_sum = T_lo + mq_lo
    carry = (lo_sum < T_lo).astype(jnp.uint64)

    sum_hi = T_hi + mq_hi
    c1 = (sum_hi < T_hi).astype(jnp.uint64)
    t = sum_hi + carry
    c2 = (t < sum_hi).astype(jnp.uint64)

    overflowed = (c1 | c2) != jnp.uint64(0)
    needs_sub = overflowed | (t >= q)
    return jnp.where(needs_sub, t - q, t)

def to_montgomery_64(x, q, R_mod_q):
    return mod_mul_64(x, R_mod_q, q)

def from_montgomery_64(x, q, q_inv):
    return mont_mul_64(x, jnp.uint64(1), q, q_inv)

def mle_update_64_mont(zero_eval, one_eval, target_eval_mont, q, q_inv):
    diff = mod_sub_64(one_eval, zero_eval, q)
    scaled = mont_mul_64(diff, target_eval_mont, q, q_inv)
    return mod_add_64(zero_eval, scaled, q)


# -----------------------------------------------------------------------------
# 128-bit primitives (optional, left for future implementation)
# -----------------------------------------------------------------------------

def _split_pair(x):
    """Accept a (hi, lo) tuple/list or a uint64 array of shape (..., 2)."""
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return _u64(x[0]), _u64(x[1])
    arr = jnp.asarray(x, dtype=jnp.uint64)
    return arr[..., 0], arr[..., 1]


def _ge_128(a_hi, a_lo, b_hi, b_lo):
    """Compare two 128-bit values: True iff (a_hi, a_lo) >= (b_hi, b_lo)."""
    return (a_hi > b_hi) | ((a_hi == b_hi) & (a_lo >= b_lo))


def _add_128(a_hi, a_lo, b_hi, b_lo):
    """128-bit add mod 2**128, returning (hi, lo) and the carry out as uint64."""
    lo = a_lo + b_lo
    carry_lo = (lo < a_lo).astype(jnp.uint64)
    hi = a_hi + b_hi + carry_lo
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
    qmb_hi, qmb_lo = _sub_128(q_hi, q_lo, b_hi, b_lo)
    alt_hi, alt_lo, _carry = _add_128(a_hi, a_lo, qmb_hi, qmb_lo)

    out_hi = jnp.where(a_ge_b, direct_hi, alt_hi)
    out_lo = jnp.where(a_ge_b, direct_lo, alt_lo)
    return out_hi, out_lo


def _mul_128_to_256(a_hi, a_lo, b_hi, b_lo):
    """Full 128x128 -> 256 multiply."""
    p_ll_hi, p_ll_lo = _mul_64_to_128(a_lo, b_lo)
    p_lh_hi, p_lh_lo = _mul_64_to_128(a_lo, b_hi)
    p_hl_hi, p_hl_lo = _mul_64_to_128(a_hi, b_lo)
    p_hh_hi, p_hh_lo = _mul_64_to_128(a_hi, b_hi)

    w0 = p_ll_lo

    s = p_ll_hi + p_lh_lo
    c1 = (s < p_ll_hi).astype(jnp.uint64)
    s2 = s + p_hl_lo
    c1 += (s2 < s).astype(jnp.uint64)
    w1 = s2

    t = p_lh_hi + p_hl_hi
    c2 = (t < p_lh_hi).astype(jnp.uint64)
    t2 = t + p_hh_lo
    c2 += (t2 < t).astype(jnp.uint64)
    t3 = t2 + c1
    c2 += (t3 < t2).astype(jnp.uint64)
    w2 = t3

    w3 = p_hh_hi + c2

    return w3, w2, w1, w0


def _reduce_256_mod_q_128(w3, w2, w1, w0, q_hi, q_lo):
    """Reduce a 256-bit value (w3, w2, w1, w0) modulo q < 2**128."""
    r_hi = jnp.uint64(0)
    r_lo = jnp.uint64(0)

    def step(r_hi, r_lo, src_word, bit_idx):
        bit = (src_word >> jnp.uint64(bit_idx)) & jnp.uint64(1)
        top_bit = (r_hi >> jnp.uint64(63)) & jnp.uint64(1)
        new_hi = (r_hi << jnp.uint64(1)) | (r_lo >> jnp.uint64(63))
        new_lo = (r_lo << jnp.uint64(1)) | bit
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


# --- 128-bit Montgomery Helpers ---
def _mul_128_lo(a_hi, a_lo, b_hi, b_lo):
    """Returns the lower 128 bits of a 128x128 multiply."""
    p_hi, p_lo = _mul_64_to_128(a_lo, b_lo)
    cross1 = a_hi * b_lo
    cross2 = a_lo * b_hi
    m_hi = p_hi + cross1 + cross2
    return m_hi, p_lo

def get_q_inv_128(q_hi, q_lo):
    """Compute q_inv = -q^{-1} mod 2^128 using Newton iteration."""
    x_hi, x_lo = q_hi, q_lo
    for _ in range(7):
        xq_hi, xq_lo = _mul_128_lo(x_hi, x_lo, q_hi, q_lo)
        diff_hi, diff_lo = _sub_128(jnp.uint64(0), jnp.uint64(2), xq_hi, xq_lo)
        x_hi, x_lo = _mul_128_lo(x_hi, x_lo, diff_hi, diff_lo)
    return _sub_128(jnp.uint64(0), jnp.uint64(0), x_hi, x_lo)

def mont_mul_128(a, b, q, q_inv):
    """Montgomery multiplication for 128-bit."""
    a_hi, a_lo = _split_pair(a)
    b_hi, b_lo = _split_pair(b)
    q_hi, q_lo = _split_pair(q)
    q_inv_hi, q_inv_lo = _split_pair(q_inv)

    T3, T2, T1, T0 = _mul_128_to_256(a_hi, a_lo, b_hi, b_lo)
    m_hi, m_lo = _mul_128_lo(T1, T0, q_inv_hi, q_inv_lo)
    mq3, mq2, mq1, mq0 = _mul_128_to_256(m_hi, m_lo, q_hi, q_lo)

    _, _, carry_lo = _add_128(T1, T0, mq1, mq0)

    sum_hi, sum_lo, c1 = _add_128(T3, T2, mq3, mq2)
    t_hi, t_lo, c2 = _add_128(sum_hi, sum_lo, jnp.uint64(0), carry_lo)

    overflowed = (c1 | c2) != jnp.uint64(0)
    ge_q = _ge_128(t_hi, t_lo, q_hi, q_lo)
    needs_sub = overflowed | ge_q

    sub_hi, sub_lo = _sub_128(t_hi, t_lo, q_hi, q_lo)
    out_hi = jnp.where(needs_sub, sub_hi, t_hi)
    out_lo = jnp.where(needs_sub, sub_lo, t_lo)
    return out_hi, out_lo

def to_montgomery_128(x_hi, x_lo, q_hi, q_lo, R_mod_q_hi, R_mod_q_lo):
    return mod_mul_128((x_hi, x_lo), (R_mod_q_hi, R_mod_q_lo), (q_hi, q_lo))

def from_montgomery_128(x_hi, x_lo, q_hi, q_lo, q_inv_hi, q_inv_lo):
    return mont_mul_128((x_hi, x_lo), (jnp.uint64(0), jnp.uint64(1)), (q_hi, q_lo), (q_inv_hi, q_inv_lo))

def mle_update_128_mont(zero_hi, zero_lo, one_hi, one_lo, target_hi, target_lo, q_hi, q_lo, q_inv_hi, q_inv_lo):
    diff_hi, diff_lo = mod_sub_128((one_hi, one_lo), (zero_hi, zero_lo), (q_hi, q_lo))
    scaled_hi, scaled_lo = mont_mul_128((diff_hi, diff_lo), (target_hi, target_lo), (q_hi, q_lo), (q_inv_hi, q_inv_lo))
    return mod_add_128((zero_hi, zero_lo), (scaled_hi, scaled_lo), (q_hi, q_lo))


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
    """Compulsory 32-bit MLE update."""
    diff = mod_sub_32(one_eval, zero_eval, q)
    scaled = mod_mul_32(diff, target_eval, q)
    return mod_add_32(zero_eval, scaled, q)


def mle_update_64(zero_eval, one_eval, target_eval, *, q):
    """64-bit MLE update: f(r) = f(0) + r * (f(1) - f(0)) mod q."""
    diff = mod_sub_64(one_eval, zero_eval, q)
    scaled = mod_mul_64(diff, target_eval, q)
    return mod_add_64(zero_eval, scaled, q)


def mle_update_128(zero_eval, one_eval, target_eval, *, q):
    """128-bit MLE update."""
    q_hi, q_lo = _split_pair(q)
    z_hi, z_lo = _split_pair(zero_eval)
    o_hi, o_lo = _split_pair(one_eval)
    t_hi, t_lo = _split_pair(target_eval)

    diff_hi, diff_lo = mod_sub_128((o_hi, o_lo), (z_hi, z_lo), (q_hi, q_lo))
    scaled_hi, scaled_lo = mod_mul_128((diff_hi, diff_lo), (t_hi, t_lo), (q_hi, q_lo))
    h, l = mod_add_128((z_hi, z_lo), (scaled_hi, scaled_lo), (q_hi, q_lo))
    return jnp.stack([h, l], axis=-1)


def mle_update(zero_eval, one_eval, target_eval, *, q, bit_width=32):
    if int(bit_width) == 32:
        return mle_update_32(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 64:
        return mle_update_64(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 128:
        return mle_update_128(zero_eval, one_eval, target_eval, q=q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


# -----------------------------------------------------------------------------
# Sumcheck Core
# -----------------------------------------------------------------------------

def _table_at_point_mont_32(zero_half, one_half, j, j_mont, q, q_inv):
    """Evaluate the linear extension T(j) = T0 + j*(T1 - T0) mod q in Montgomery Form."""
    if j == 0:
        return zero_half
    diff = mod_sub_32(one_half, zero_half, q)
    scaled = mont_mul_32(diff, j_mont, q, q_inv)
    return mod_add_32(zero_half, scaled, q)

def _sum_mod_q(values, q):
    """Reduce a 1-D uint32 array mod q."""
    acc = values.astype(jnp.uint64).sum()
    return (acc % jnp.uint64(q)).astype(jnp.uint32)

def _evaluate_sum_of_products_mont_32(expression, values_by_name, q, q_inv):
    """Evaluate `expression` pointwise on the per-variable arrays using Montgomery multiplication."""
    if not expression:
        raise ValueError("expression must contain at least one term")

    term_values = []
    for term in expression:
        if not term:
            raise ValueError("expression terms must contain at least one factor")
        prod = values_by_name[term[0]]
        for var in term[1:]:
            prod = mont_mul_32(prod, values_by_name[var], q, q_inv)
        term_values.append(prod)

    total = term_values[0]
    for tv in term_values[1:]:
        total = mod_add_32(total, tv, q)
    return total

def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    """Compulsory 32-bit sumcheck path (Montgomery Optimized)."""
    tables = {
        name: jnp.asarray(arr, dtype=jnp.uint32)
        for name, arr in eval_tables.items()
    }
    challenges = jnp.asarray(challenges, dtype=jnp.uint32)

    if not tables:
        raise ValueError("eval_tables must be non-empty")
    if not expression:
        raise ValueError("expression must be non-empty")

    q_inv = get_q_inv_32(q)
    R_mod_q = jnp.uint32((jnp.uint64(1) << 32) % _to_u64(q))

    # Convert inputs to Montgomery form
    tables = {name: to_montgomery_32(t, q, R_mod_q) for name, t in tables.items()}
    challenges_mont = to_montgomery_32(challenges, q, R_mod_q)

    degree = max(len(term) for term in expression)
    num_eval_points = degree + 1

    initial_combined = _evaluate_sum_of_products_mont_32(expression, tables, q, q_inv)
    claim0_mont = _sum_mod_q(initial_combined, q)

    round_evals = []

    for round_idx in range(num_rounds):
        zero_halves = {name: t[::2] for name, t in tables.items()}
        one_halves = {name: t[1::2] for name, t in tables.items()}

        evals_at_j = []
        for j in range(num_eval_points):
            j_mont = to_montgomery_32(jnp.uint32(j), q, R_mod_q)
            tables_at_j = {
                name: _table_at_point_mont_32(zero_halves[name], one_halves[name], j, j_mont, q, q_inv)
                for name in tables
            }
            combined = _evaluate_sum_of_products_mont_32(expression, tables_at_j, q, q_inv)
            evals_at_j.append(_sum_mod_q(combined, q))

        round_evals.append(jnp.stack(evals_at_j))

        r_mont = challenges_mont[round_idx]
        tables = {
            name: mle_update_32_mont(zero_halves[name], one_halves[name], r_mont, q, q_inv)
            for name in tables
        }

    if round_evals:
        round_evals_array = jnp.stack(round_evals)
    else:
        round_evals_array = jnp.zeros((0, num_eval_points), dtype=jnp.uint32)

    # Convert back to standard form
    claim0 = from_montgomery_32(claim0_mont, q, q_inv)
    round_evals_array = from_montgomery_32(round_evals_array, q, q_inv)

    return (claim0, round_evals_array)


def _table_at_point_mont_64(zero_half, one_half, j, j_mont, q, q_inv):
    if j == 0:
        return zero_half
    diff = mod_sub_64(one_half, zero_half, q)
    scaled = mont_mul_64(diff, j_mont, q, q_inv)
    return mod_add_64(zero_half, scaled, q)

def _sum_mod_q_64(values, q):
    q64 = _u64(q)
    flat = values.astype(jnp.uint64)

    def body(i, acc):
        x = flat[i]
        s = acc + x
        overflowed = s < acc
        needs_sub = overflowed | (s >= q64)
        return jnp.where(needs_sub, s - q64, s)

    return jax.lax.fori_loop(0, flat.shape[0], body, jnp.uint64(0))

def _evaluate_sum_of_products_mont_64(expression, values_by_name, q, q_inv):
    if not expression:
        raise ValueError("expression must contain at least one term")

    term_values = []
    for term in expression:
        if not term:
            raise ValueError("expression terms must contain at least one factor")
        prod = values_by_name[term[0]]
        for var in term[1:]:
            prod = mont_mul_64(prod, values_by_name[var], q, q_inv)
        term_values.append(prod)

    total = term_values[0]
    for tv in term_values[1:]:
        total = mod_add_64(total, tv, q)
    return total

def sumcheck_64(eval_tables, *, q, expression, challenges, num_rounds):
    """64-bit sumcheck path (Montgomery Optimized)."""
    tables = {
        name: jnp.asarray(arr, dtype=jnp.uint64)
        for name, arr in eval_tables.items()
    }
    challenges = jnp.asarray(challenges, dtype=jnp.uint64)

    if not tables:
        raise ValueError("eval_tables must be non-empty")
    if not expression:
        raise ValueError("expression must be non-empty")

    q_inv = get_q_inv_64(q)
    q64 = _u64(q)
    R_mod_q = (jnp.uint64(0) - q64) % q64

    tables = {name: to_montgomery_64(t, q, R_mod_q) for name, t in tables.items()}
    challenges_mont = to_montgomery_64(challenges, q, R_mod_q)

    degree = max(len(term) for term in expression)
    num_eval_points = degree + 1

    initial_combined = _evaluate_sum_of_products_mont_64(expression, tables, q, q_inv)
    claim0_mont = _sum_mod_q_64(initial_combined, q)

    round_evals = []

    for round_idx in range(num_rounds):
        zero_halves = {name: t[::2] for name, t in tables.items()}
        one_halves = {name: t[1::2] for name, t in tables.items()}

        evals_at_j = []
        for j in range(num_eval_points):
            j_mont = to_montgomery_64(jnp.uint64(j), q, R_mod_q)
            tables_at_j = {
                name: _table_at_point_mont_64(zero_halves[name], one_halves[name], j, j_mont, q, q_inv)
                for name in tables
            }
            combined = _evaluate_sum_of_products_mont_64(expression, tables_at_j, q, q_inv)
            evals_at_j.append(_sum_mod_q_64(combined, q))

        round_evals.append(jnp.stack(evals_at_j))

        r_mont = challenges_mont[round_idx]
        tables = {
            name: mle_update_64_mont(zero_halves[name], one_halves[name], r_mont, q, q_inv)
            for name in tables
        }

    if round_evals:
        round_evals_array = jnp.stack(round_evals)
    else:
        round_evals_array = jnp.zeros((0, num_eval_points), dtype=jnp.uint64)

    claim0 = from_montgomery_64(claim0_mont, q, q_inv)
    round_evals_array = from_montgomery_64(round_evals_array, q, q_inv)

    return (claim0, round_evals_array)


def _table_at_point_mont_128(z_hi, z_lo, o_hi, o_lo, j, j_mont_hi, j_mont_lo, q_hi, q_lo, q_inv_hi, q_inv_lo):
    if j == 0:
        return z_hi, z_lo
    diff_hi, diff_lo = mod_sub_128((o_hi, o_lo), (z_hi, z_lo), (q_hi, q_lo))
    scaled_hi, scaled_lo = mont_mul_128((diff_hi, diff_lo), (j_mont_hi, j_mont_lo), (q_hi, q_lo), (q_inv_hi, q_inv_lo))
    return mod_add_128((z_hi, z_lo), (scaled_hi, scaled_lo), (q_hi, q_lo))

def _sum_mod_q_128(values_hi, values_lo, q_hi, q_lo):
    def body(i, acc):
        acc_hi, acc_lo = acc
        x_hi = values_hi[i]
        x_lo = values_lo[i]
        s_hi, s_lo, carry = _add_128(acc_hi, acc_lo, x_hi, x_lo)
        overflowed = carry != jnp.uint64(0)
        ge_q = _ge_128(s_hi, s_lo, q_hi, q_lo)
        needs_sub = overflowed | ge_q
        sub_hi, sub_lo = _sub_128(s_hi, s_lo, q_hi, q_lo)
        return jnp.where(needs_sub, sub_hi, s_hi), jnp.where(needs_sub, sub_lo, s_lo)

    return jax.lax.fori_loop(0, values_hi.shape[0], body, (jnp.uint64(0), jnp.uint64(0)))

def _evaluate_sum_of_products_mont_128(expression, values_hi, values_lo, q_hi, q_lo, q_inv_hi, q_inv_lo):
    if not expression:
        raise ValueError("expression must contain at least one term")

    term_hi, term_lo = [], []
    for term in expression:
        if not term:
            raise ValueError("expression terms must contain at least one factor")
        prod_hi = values_hi[term[0]]
        prod_lo = values_lo[term[0]]
        for var in term[1:]:
            prod_hi, prod_lo = mont_mul_128(
                (prod_hi, prod_lo), (values_hi[var], values_lo[var]),
                (q_hi, q_lo), (q_inv_hi, q_inv_lo)
            )
        term_hi.append(prod_hi)
        term_lo.append(prod_lo)

    total_hi = term_hi[0]
    total_lo = term_lo[0]
    for i in range(1, len(term_hi)):
        total_hi, total_lo = mod_add_128((total_hi, total_lo), (term_hi[i], term_lo[i]), (q_hi, q_lo))
    return total_hi, total_lo

def sumcheck_128(eval_tables, *, q, expression, challenges, num_rounds):
    """128-bit sumcheck path (Montgomery Optimized)."""
    if not eval_tables:
        raise ValueError("eval_tables must be non-empty")
    if not expression:
        raise ValueError("expression must be non-empty")

    q_hi, q_lo = _split_pair(q)
    q_inv_hi, q_inv_lo = get_q_inv_128(q_hi, q_lo)

    R_mod_q_hi, R_mod_q_lo = _reduce_256_mod_q_128(
        jnp.uint64(0), jnp.uint64(0), jnp.uint64(1), jnp.uint64(0), q_hi, q_lo
    )

    tables_hi = {}
    tables_lo = {}
    for name, arr in eval_tables.items():
        hi, lo = _split_pair(arr)
        hi, lo = to_montgomery_128(hi, lo, q_hi, q_lo, R_mod_q_hi, R_mod_q_lo)
        tables_hi[name] = hi
        tables_lo[name] = lo

    chal_hi, chal_lo = _split_pair(challenges)
    chal_hi, chal_lo = to_montgomery_128(chal_hi, chal_lo, q_hi, q_lo, R_mod_q_hi, R_mod_q_lo)

    degree = max(len(term) for term in expression)
    num_eval_points = degree + 1

    init_hi, init_lo = _evaluate_sum_of_products_mont_128(
        expression, tables_hi, tables_lo, q_hi, q_lo, q_inv_hi, q_inv_lo
    )
    claim0_hi_mont, claim0_lo_mont = _sum_mod_q_128(init_hi, init_lo, q_hi, q_lo)

    round_evals_hi = []
    round_evals_lo = []

    for round_idx in range(num_rounds):
        z_hi = {name: t[::2] for name, t in tables_hi.items()}
        z_lo = {name: t[::2] for name, t in tables_lo.items()}
        o_hi = {name: t[1::2] for name, t in tables_hi.items()}
        o_lo = {name: t[1::2] for name, t in tables_lo.items()}

        evals_j_hi = []
        evals_j_lo = []
        for j in range(num_eval_points):
            j_mont_hi, j_mont_lo = to_montgomery_128(
                jnp.uint64(0), jnp.uint64(j), q_hi, q_lo, R_mod_q_hi, R_mod_q_lo
            )

            t_j_hi = {}
            t_j_lo = {}
            for name in tables_hi:
                h, l = _table_at_point_mont_128(
                    z_hi[name], z_lo[name], o_hi[name], o_lo[name], j,
                    j_mont_hi, j_mont_lo, q_hi, q_lo, q_inv_hi, q_inv_lo
                )
                t_j_hi[name] = h
                t_j_lo[name] = l

            c_hi, c_lo = _evaluate_sum_of_products_mont_128(
                expression, t_j_hi, t_j_lo, q_hi, q_lo, q_inv_hi, q_inv_lo
            )
            sm_hi, sm_lo = _sum_mod_q_128(c_hi, c_lo, q_hi, q_lo)
            evals_j_hi.append(sm_hi)
            evals_j_lo.append(sm_lo)

        round_evals_hi.append(jnp.stack(evals_j_hi))
        round_evals_lo.append(jnp.stack(evals_j_lo))

        r_hi = chal_hi[round_idx]
        r_lo = chal_lo[round_idx]

        for name in tables_hi:
            h, l = mle_update_128_mont(
                z_hi[name], z_lo[name], o_hi[name], o_lo[name],
                r_hi, r_lo, q_hi, q_lo, q_inv_hi, q_inv_lo
            )
            tables_hi[name] = h
            tables_lo[name] = l

    if round_evals_hi:
        r_evals_arr_hi = jnp.stack(round_evals_hi)
        r_evals_arr_lo = jnp.stack(round_evals_lo)
    else:
        r_evals_arr_hi = jnp.zeros((0, num_eval_points), dtype=jnp.uint64)
        r_evals_arr_lo = jnp.zeros((0, num_eval_points), dtype=jnp.uint64)

    claim0_hi, claim0_lo = from_montgomery_128(
        claim0_hi_mont, claim0_lo_mont, q_hi, q_lo, q_inv_hi, q_inv_lo
    )
    r_evals_arr_hi, r_evals_arr_lo = from_montgomery_128(
        r_evals_arr_hi, r_evals_arr_lo, q_hi, q_lo, q_inv_hi, q_inv_lo
    )

    claim0_arr = jnp.stack([claim0_hi, claim0_lo], axis=-1)
    round_evals_arr = jnp.stack([r_evals_arr_hi, r_evals_arr_lo], axis=-1)
    return (claim0_arr, round_evals_arr)


def sumcheck(eval_tables, *, q, expression, challenges, num_rounds, bit_width=32):
    """Frozen dispatcher entrypoint used by the harness."""
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