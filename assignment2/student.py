import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)

# -----------------------------------------------------------------------------
# Montgomery Helpers
# -----------------------------------------------------------------------------

def get_q_inv(q):
    """Compute q_inv = -q^{-1} mod 2^32."""
    q = int(q)
    # Modular inverse using Newton's method for 2^k
    # t = 0
    # new_t = 1
    r = 1 << 32
    # Standard modular inverse
    q_inv = pow(q, -1, r)
    return jnp.uint32((r - q_inv) % r)

# def mont_mul_32(a, b, q, q_inv):
#     """Montgomery multiplication: returns (a * b * R^-1) mod q."""
#     # a and b are already in Montgomery form (xR mod q)
#     # This returns (aR * bR * R^-1) = (ab)R mod q
#     T = a.astype(jnp.uint64) * b.astype(jnp.uint64)
#     m = (T.astype(jnp.uint32) * q_inv) # T * q_inv mod 2^32

#     # (T + m*q) / 2^32
#     t = (T + m.astype(jnp.uint64) * q.astype(jnp.uint64)) >> 32

#     # Final conditional subtraction
#     return jnp.where(t >= q, (t - q).astype(jnp.uint32), t.astype(jnp.uint32))

def mont_mul_32(a, b, q, q_inv):
    T = a.astype(jnp.uint64) * b.astype(jnp.uint64)
    m = (T.astype(jnp.uint32) * q_inv).astype(jnp.uint32)
    
    # Split to avoid uint64 overflow: T + m*q can exceed 2^64
    # Instead compute (T >> 32) + m*(q >> 32) + carry terms
    # Simpler: use the fact that we only need (T + m*q) >> 32
    # which equals (T >> 32) + (m*q) >> 32 + carry from lower 32 bits
    T_lo = T & jnp.uint64(0xFFFFFFFF)
    T_hi = T >> 32
    mq = m.astype(jnp.uint64) * q.astype(jnp.uint64)
    mq_lo = mq & jnp.uint64(0xFFFFFFFF)
    mq_hi = mq >> 32
    
    carry = (T_lo + mq_lo) >> 32
    t = T_hi + mq_hi + carry
    
    return jnp.where(t >= q.astype(jnp.uint64),
                        (t - q.astype(jnp.uint64)).astype(jnp.uint32),
                        t.astype(jnp.uint32))

def to_montgomery(x, q, R_mod_q):
    """Convert standard integer to Montgomery form: x * R mod q."""
    # We use standard mod here as it's a one-time setup cost
    return ((x.astype(jnp.uint64) * R_mod_q.astype(jnp.uint64)) % q.astype(jnp.uint64)).astype(jnp.uint32)

def from_montgomery(x_bar, q, q_inv):
    """Convert Montgomery form back to standard: x_bar * R^-1 mod q."""
    # This is just a Montgomery reduction against 1
    T = x_bar.astype(jnp.uint64)
    m = (T.astype(jnp.uint32) * q_inv)
    t = (T + m.astype(jnp.uint64) * q.astype(jnp.uint64)) >> 32
    return jnp.where(t >= q, (t - q).astype(jnp.uint32), t.astype(jnp.uint32))

# -----------------------------------------------------------------------------
# Modular Arithmetic (Montgomery Compatible)
# -----------------------------------------------------------------------------

def mod_add_32(a, b, q):
    """(a + b) mod q. Works same in Montgomery and Standard domain."""
    a64 = a.astype(jnp.uint64)
    b64 = b.astype(jnp.uint64)
    res = a64 + b64
    return jnp.where(res >= q, res - q, res).astype(jnp.uint32)

def mod_sub_32(a, b, q):
    """(a - b) mod q. Works same in Montgomery and Standard domain."""
    a64 = a.astype(jnp.uint64)
    b64 = b.astype(jnp.uint64)
    res = jnp.where(a64 >= b64, a64 - b64, a64 + q - b64)
    return res.astype(jnp.uint32)

def mod_mul_32(a, b, q):
    """Standard modular multiply — required by harness."""
    a64 = a.astype(jnp.uint64)
    b64 = b.astype(jnp.uint64)
    return ((a64 * b64) % q).astype(jnp.uint32)

def mle_update_32_mont(zero_eval, one_eval, target_eval_mont, q, q_inv):
    """MLE update using Montgomery multiplication."""
    diff = mod_sub_32(one_eval, zero_eval, q)
    # target_eval_mont must be in Montgomery form
    prod = mont_mul_32(diff, target_eval_mont, q, q_inv)
    return mod_add_32(zero_eval, prod, q)

# -----------------------------------------------------------------------------
# Sumcheck Core
# -----------------------------------------------------------------------------

def compute_composition_mont(expression, t_stack, key_to_idx, q, q_inv, R_mod_q):
    """Composition using Montgomery multiplication."""
    # Montgomery 0 is 0
    acc = jnp.zeros(t_stack.shape[1], dtype=jnp.uint32)

    for term in expression:
        # Montgomery 1 is R mod q
        term_val = R_mod_q
        for var in term:
            x = t_stack[key_to_idx[var]]
            term_val = mont_mul_32(term_val, x, q, q_inv)
        acc = mod_add_32(acc, term_val, q)

    return acc

@partial(jax.jit, static_argnames=["q", "expression", "num_rounds"])
def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    q_u32 = jnp.uint32(q)
    q_inv = get_q_inv(q)
    R_mod_q = jnp.uint32((1 << 32) % q)

    keys = tuple(eval_tables.keys())
    key_to_idx = {k: i for i, k in enumerate(keys)}

    # 1. Convert initial tables to Montgomery form
    table_stack = jnp.stack([eval_tables[k] for k in keys], axis=0)
    table_stack = to_montgomery(table_stack, q_u32, R_mod_q)

    degree = max(len(term) for term in expression)
    # 2. Prepare t_vals in Montgomery form
    t_vals_raw = jnp.arange(degree + 1, dtype=jnp.uint32)
    t_vals_mont = to_montgomery(t_vals_raw, q_u32, R_mod_q)

    # 3. Convert challenges to Montgomery form
    challenges_mont = to_montgomery(challenges.astype(jnp.uint32), q_u32, R_mod_q)

    all_round_evals_mont = []

    for round_idx in range(num_rounds):
        z = table_stack[:, 0::2]
        o = table_stack[:, 1::2]

        def eval_at_t(t_mont):
            t_stack = jax.vmap(
                lambda z_row, o_row: mle_update_32_mont(z_row, o_row, t_mont, q_u32, q_inv)
            )(z, o)

            vals = compute_composition_mont(expression, t_stack, key_to_idx, q_u32, q_inv, R_mod_q)
            # Standard sum is fine, but must mod q at the end of sum
            # Note: We use uint64 sum to avoid overflow before the mod
            res = (jnp.sum(vals.astype(jnp.uint64)) % q_u32.astype(jnp.uint64)).astype(jnp.uint32)
            return res

        round_evals = jax.vmap(eval_at_t)(t_vals_mont)
        all_round_evals_mont.append(round_evals)

        r_mont = challenges_mont[round_idx]
        table_stack = jax.vmap(
            lambda z_row, o_row: mle_update_32_mont(z_row, o_row, r_mont, q_u32, q_inv)
        )(z, o)

    all_round_evals_mont = jnp.stack(all_round_evals_mont)

    # 4. Final Claimed Sum (still in Montgomery domain)
    claimed_sum_mont = mod_add_32(
        all_round_evals_mont[0, 0],
        all_round_evals_mont[0, 1],
        q_u32,
    )

    # 5. Convert everything back to standard domain
    final_claimed_sum = from_montgomery(claimed_sum_mont, q_u32, q_inv)
    final_round_evals = from_montgomery(all_round_evals_mont, q_u32, q_inv)

    return final_claimed_sum, final_round_evals

def sumcheck(eval_tables, *, q, expression, challenges, num_rounds, bit_width=32):
    expression = tuple(tuple(term) for term in expression)
    q = int(q)
    if int(bit_width) == 32:
        return sumcheck_32(eval_tables, q=q, expression=expression, challenges=challenges, num_rounds=num_rounds)
    raise ValueError(f"Montgomery only implemented for 32-bit. Found bit_width={bit_width}")

# if __name__ == "__main__":
#     q = 3603169181
#     q_u32 = jnp.uint32(q)
#     R_mod_q = jnp.uint32((1 << 32) % q)
#     q_inv = get_q_inv(q)

#     a = to_montgomery(jnp.array([jnp.uint32(3)]), q_u32, R_mod_q)
#     b = to_montgomery(jnp.array([jnp.uint32(4)]), q_u32, R_mod_q)
    
#     prod_mont = mont_mul_32(a, b, q_u32, q_inv)
#     prod = from_montgomery(prod_mont, q_u32, q_inv)
#     print(f"3 * 4 = {prod}")  # expect 12
    
#     # Also test a larger product that would overflow
#     x = to_montgomery(jnp.array([jnp.uint32(999999999)]), q_u32, R_mod_q)
#     y = to_montgomery(jnp.array([jnp.uint32(999999999)]), q_u32, R_mod_q)
#     prod2_mont = mont_mul_32(x, y, q_u32, q_inv)
#     prod2 = from_montgomery(prod2_mont, q_u32, q_inv)
#     expected = (999999999 * 999999999) % q
#     print(f"999999999^2 mod q = {prod2}, expected = {expected}")