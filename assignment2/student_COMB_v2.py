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


def _get_montgomery_params_32(q):
    """Compute constants for 32-bit Montgomery arithmetic natively in Python."""
    q_int = int(q)
    R = 1 << 32
    R_mod_q = R % q_int
    R2_mod_q = (R * R) % q_int
    # q_inv = -q^-1 mod R. pow(base, -1, mod) is available in Python 3.8+
    q_inv = (-pow(q_int, -1, R)) % R
    return jnp.uint32(R_mod_q), jnp.uint32(R2_mod_q), jnp.uint32(q_inv)


def montgomery_mul_32(a, b, q, q_inv):
    """Return (a * b * R^-1) mod q for the 32-bit track."""
    a64, b64 = _to_u64(a), _to_u64(b)
    t = a64 * b64

    # Native 32-bit multiplication automatically truncates mod 2^32.
    m32 = t.astype(jnp.uint32) * q_inv

    # Multiply by q in 64-bit space
    mq = _to_u64(m32) * _to_u64(q)

    # Since m = t_lo * q_inv mod 2^32, (t_lo + mq_lo) is a multiple of 2^32
    u = (t + mq) >> jnp.uint64(32)

    q64 = _to_u64(q)
    u_reduced = jnp.where(u >= q64, u - q64, u)
    return u_reduced.astype(jnp.uint32)


# -----------------------------------------------------------------------------
# 64-bit primitives (optional, left for future implementation)
# -----------------------------------------------------------------------------

_U64_MASK_LO32 = jnp.uint64(0xFFFFFFFF)
_U64_SHIFT32 = jnp.uint64(32)


def _u64(x):
    return jnp.asarray(x, dtype=jnp.uint64)


def mod_add_64(a, b, q):
    a, b, q = _u64(a), _u64(b), _u64(q)
    s = a + b
    overflowed = s < a
    needs_sub = overflowed | (s >= q)
    return jnp.where(needs_sub, s - q, s).astype(jnp.uint64)


def mod_sub_64(a, b, q):
    a, b, q = _u64(a), _u64(b), _u64(q)
    return jnp.where(a >= b, a - b, a + (q - b)).astype(jnp.uint64)


def _mul_64_to_128(a, b):
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
    a, b, q = _u64(a), _u64(b), _u64(q)
    hi, lo = _mul_64_to_128(a, b)
    return _reduce_128_mod_q_64(hi, lo, q)


def _get_montgomery_params_64(q):
    """Compute constants for 64-bit Montgomery arithmetic natively in Python."""
    q_int = int(q)
    R = 1 << 64
    R_mod_q = R % q_int
    R2_mod_q = (R * R) % q_int
    q_inv = (-pow(q_int, -1, R)) % R
    return jnp.uint64(R_mod_q), jnp.uint64(R2_mod_q), jnp.uint64(q_inv)


def montgomery_mul_64(a, b, q, q_inv):
    """Return (a * b * R^-1) mod q for the 64-bit track."""
    a, b = _u64(a), _u64(b)

    # 1. Full 128-bit multiplication for a * b
    hi, lo = _mul_64_to_128(a, b)

    # 2. Compute m. Standard 64-bit multiply naturally wraps mod 2^64!
    m = lo * _u64(q_inv)

    # 3. Compute ONLY the high half of m * q!
    # The low half is mathematically guaranteed to cancel out `lo`.
    mq_hi, _ = _mul_64_to_128(m, _u64(q))

    # 4. The carry out of the bottom 64 bits is 1, unless lo == 0.
    carry = (lo != jnp.uint64(0)).astype(jnp.uint64)

    # 5. Final sum and reduction
    u = hi + mq_hi + carry
    overflowed = u < hi
    needs_sub = overflowed | (u >= _u64(q))

    return jnp.where(needs_sub, u - _u64(q), u)


# -----------------------------------------------------------------------------
# 128-bit primitives (optional, left for future implementation)
# -----------------------------------------------------------------------------

def _split_pair(x):
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return _u64(x[0]), _u64(x[1])
    arr = jnp.asarray(x, dtype=jnp.uint64)
    return arr[..., 0], arr[..., 1]


def _ge_128(a_hi, a_lo, b_hi, b_lo):
    return (a_hi > b_hi) | ((a_hi == b_hi) & (a_lo >= b_lo))


def _add_128(a_hi, a_lo, b_hi, b_lo):
    lo = a_lo + b_lo
    carry_lo = (lo < a_lo).astype(jnp.uint64)
    hi = a_hi + b_hi + carry_lo
    inner = b_hi + carry_lo
    inner_carry = (inner < b_hi).astype(jnp.uint64)
    outer_carry = (hi < a_hi).astype(jnp.uint64)
    carry_out = inner_carry | outer_carry
    return hi, lo, carry_out


def _sub_128(a_hi, a_lo, b_hi, b_lo):
    lo = a_lo - b_lo
    borrow = (a_lo < b_lo).astype(jnp.uint64)
    hi = a_hi - b_hi - borrow
    return hi, lo


def mod_add_128(a, b, q):
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
    diff = mod_sub_32(one_eval, zero_eval, q)
    scaled = mod_mul_32(diff, target_eval, q)
    return mod_add_32(zero_eval, scaled, q)


def mle_update_64(zero_eval, one_eval, target_eval, *, q):
    diff = mod_sub_64(one_eval, zero_eval, q)
    scaled = mod_mul_64(diff, target_eval, q)
    return mod_add_64(zero_eval, scaled, q)


def mle_update_128(zero_eval, one_eval, target_eval, *, q):
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
# Sumcheck (32-bit Vectorized + Montgomery)
# -----------------------------------------------------------------------------

def _evaluate_and_sum_stacked_32(expression, t_stacks, key_to_idx, q, q_inv):
    term_tensors = []
    for term in expression:
        prod = t_stacks[key_to_idx[term[0]]]
        for var in term[1:]:
            # Core evaluate inner product uses Montgomery Mod Mul directly
            prod = montgomery_mul_32(prod, t_stacks[key_to_idx[var]], q, q_inv)
        term_tensors.append(prod)

    total = term_tensors[0].astype(jnp.uint64)
    for tv in term_tensors[1:]:
        total = total + tv.astype(jnp.uint64)

    # Note: sum is standard addition and retains the Montgomery scaling factors appropriately!
    return (total.sum(axis=-1) % jnp.uint64(q)).astype(jnp.uint32)


@partial(jax.jit, static_argnames=["q", "expression", "num_rounds"])
def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    keys       = tuple(eval_tables.keys())
    key_to_idx = {k: i for i, k in enumerate(keys)}

    # Fetch Python-level trace constants based on modulus `q`
    _, R2_mod_q, q_inv = _get_montgomery_params_32(q)

    table_stack = jnp.stack(
        [jnp.asarray(eval_tables[k], dtype=jnp.uint32) for k in keys]
    )  # (num_vars, N)

    challenges = jnp.asarray(challenges, dtype=jnp.uint32)

    # Shift operands dynamically to Montgomery Form
    table_stack = montgomery_mul_32(table_stack, R2_mod_q, q, q_inv)
    challenges = montgomery_mul_32(challenges, R2_mod_q, q, q_inv)

    degree = max(len(term) for term in expression)
    j_vals = jnp.arange(degree + 1, dtype=jnp.uint32)  # (degree+1,)
    j_vals = montgomery_mul_32(j_vals, R2_mod_q, q, q_inv)

    all_round_evals = []

    for round_idx in range(num_rounds):
        z = table_stack[:, ::2]
        o = table_stack[:, 1::2]

        # Additions & Subtractions seamlessly broadcast over Montgomery Forms.
        diffs = mod_sub_32(o, z, q)

        t_stacks = mod_add_32(
            z[:, None, :],
            montgomery_mul_32(diffs[:, None, :], j_vals[None, :, None], q, q_inv),
            q,
        )

        all_round_evals.append(
            _evaluate_and_sum_stacked_32(expression, t_stacks, key_to_idx, q, q_inv)
        )

        r = challenges[round_idx]
        table_stack = mod_add_32(z, montgomery_mul_32(diffs, r, q, q_inv), q)

    all_round_evals = jnp.stack(all_round_evals)  # (num_rounds, degree+1)

    claim0 = mod_add_32(all_round_evals[0, 0], all_round_evals[0, 1], q)

    # Revert evaluations back to standard space before returning. (multiply by 1 in Montgomery space limits reduction).
    one_mont = jnp.uint32(1)
    claim0 = montgomery_mul_32(claim0, one_mont, q, q_inv)
    all_round_evals = montgomery_mul_32(all_round_evals, one_mont, q, q_inv)

    return claim0, all_round_evals


# -----------------------------------------------------------------------------
# Sumcheck (64-bit Vectorized + Montgomery)
# -----------------------------------------------------------------------------

def _evaluate_and_sum_stacked_64(expression, t_stacks, key_to_idx, q, q_inv):
    term_tensors = []
    for term in expression:
        prod = t_stacks[key_to_idx[term[0]]]
        for var in term[1:]:
            prod = montgomery_mul_64(prod, t_stacks[key_to_idx[var]], q, q_inv)
        term_tensors.append(prod)

    total = term_tensors[0]
    for tv in term_tensors[1:]:
        total = mod_add_64(total, tv, q)

    def add_mod(a, b):
        q64 = _u64(q)
        s = a + b
        overflowed = s < a
        return jnp.where(overflowed | (s >= q64), s - q64, s)

    return jax.lax.reduce(total, jnp.uint64(0), add_mod, dimensions=[1])


@partial(jax.jit, static_argnames=["q", "expression", "num_rounds"])
def sumcheck_64(eval_tables, *, q, expression, challenges, num_rounds):
    keys       = tuple(eval_tables.keys())
    key_to_idx = {k: i for i, k in enumerate(keys)}

    _, R2_mod_q, q_inv = _get_montgomery_params_64(q)

    table_stack = jnp.stack(
        [jnp.asarray(eval_tables[k], dtype=jnp.uint64) for k in keys]
    )  # (num_vars, N)

    challenges = jnp.asarray(challenges, dtype=jnp.uint64)

    # Shift operands dynamically to Montgomery Form
    table_stack = montgomery_mul_64(table_stack, R2_mod_q, q, q_inv)
    challenges = montgomery_mul_64(challenges, R2_mod_q, q, q_inv)

    degree = max(len(term) for term in expression)
    j_vals = jnp.arange(degree + 1, dtype=jnp.uint64)
    j_vals = montgomery_mul_64(j_vals, R2_mod_q, q, q_inv)

    all_round_evals = []

    for round_idx in range(num_rounds):
        z     = table_stack[:, ::2]
        o     = table_stack[:, 1::2]
        diffs = mod_sub_64(o, z, q)

        t_stacks = mod_add_64(
            z[:, None, :],
            montgomery_mul_64(diffs[:, None, :], j_vals[None, :, None], q, q_inv),
            q,
        )

        all_round_evals.append(
            _evaluate_and_sum_stacked_64(expression, t_stacks, key_to_idx, q, q_inv)
        )

        r = challenges[round_idx]
        table_stack = mod_add_64(z, montgomery_mul_64(diffs, r, q, q_inv), q)

    all_round_evals = jnp.stack(all_round_evals)  # (num_rounds, degree+1)

    claim0 = mod_add_64(all_round_evals[0, 0], all_round_evals[0, 1], q)

    # Revert evaluations back to standard space before returning
    one = jnp.uint64(1)
    claim0 = montgomery_mul_64(claim0, one, q, q_inv)
    all_round_evals = montgomery_mul_64(all_round_evals, one, q, q_inv)

    return claim0, all_round_evals


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