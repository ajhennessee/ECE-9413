"""
Assignment 2 student implementation reference skeleton.

This file documents the frozen student-facing API.
Only 32-bit kernels are compulsory in the base track.
64-bit and 128-bit kernels are intentionally left unimplemented here.
"""

from __future__ import annotations

from pprint import pprint
import jax
import jax.numpy as jnp
from functools import partial
jax.config.update("jax_enable_x64", True)


# -----------------------------------------------------------------------------
# 32-bit primitives (compulsory)
# -----------------------------------------------------------------------------

def mod_add_32(a, b, q):
    """Return (a + b) mod q for the 32-bit track."""
    a64 = a.astype(jnp.uint64)
    b64 = b.astype(jnp.uint64)
    return ((a64 + b64) % q).astype(jnp.uint32)


def mod_sub_32(a, b, q):
    """Return (a - b) mod q for the 32-bit track."""
    a64 = a.astype(jnp.int64)
    b64 = b.astype(jnp.int64)
    return ((a64 - b64) % q).astype(jnp.uint32)


def mod_mul_32(a, b, q):
    """Return (a * b) mod q for the 32-bit track."""
    a64 = a.astype(jnp.uint64)
    b64 = b.astype(jnp.uint64)
    return ((a64 * b64) % q).astype(jnp.uint32)


# # -----------------------------------------------------------------------------
# # 64-bit primitives (optional, left for future implementation)
# # -----------------------------------------------------------------------------

# def mod_add_64(a, b, q):
#     """Optional 64-bit modular add kernel."""
#     # TODO(student): implement when enabling 64-bit track.
#     raise NotImplementedError


# def mod_sub_64(a, b, q):
#     """Optional 64-bit modular subtract kernel."""
#     # TODO(student): implement when enabling 64-bit track.
#     raise NotImplementedError


# def mod_mul_64(a, b, q):
#     """Optional 64-bit modular multiply kernel."""
#     # TODO(student): implement when enabling 64-bit track.
#     raise NotImplementedError


# # -----------------------------------------------------------------------------
# # 128-bit primitives (optional, left for future implementation)
# # -----------------------------------------------------------------------------

# def mod_add_128(a, b, q):
#     """Optional 128-bit modular add kernel."""
#     # TODO(student): implement when enabling 128-bit track.
#     raise NotImplementedError


# def mod_sub_128(a, b, q):
#     """Optional 128-bit modular subtract kernel."""
#     # TODO(student): implement when enabling 128-bit track.
#     raise NotImplementedError


# def mod_mul_128(a, b, q):
#     """Optional 128-bit modular multiply kernel."""
#     # TODO(student): implement when enabling 128-bit track.
#     raise NotImplementedError


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
    prod = mod_mul_32(diff, target_eval, q)
    return mod_add_32(zero_eval, prod, q)


# def mle_update_64(zero_eval, one_eval, target_eval, *, q):
#     """Optional 64-bit MLE update."""
#     # TODO(student): implement when enabling 64-bit track.
#     raise NotImplementedError


# def mle_update_128(zero_eval, one_eval, target_eval, *, q):
#     """Optional 128-bit MLE update."""
#     # TODO(student): implement when enabling 128-bit track.
#     raise NotImplementedError


def mle_update(zero_eval, one_eval, target_eval, *, q, bit_width=32):
    if int(bit_width) == 32:
        return mle_update_32(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 64:
        return mle_update_64(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 128:
        return mle_update_128(zero_eval, one_eval, target_eval, q=q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def eval_expression(expression, t_stack, keys, q):
    """
    expression: tuple of tuples e.g. (('a','b'), ('c',))
    t_stack: shape (num_tables, half) — stacked arrays
    keys: list mapping index to variable name
    """
    key_to_idx = {k: i for i, k in enumerate(keys)}
    
    result = None
    for term in expression:
        term_val = None
        for var in term:
            arr = t_stack[key_to_idx[var]].astype(jnp.uint64)
            if term_val is None:
                term_val = arr
            else:
                term_val = (term_val * arr) % q
        result = term_val if result is None else (result + term_val) % q
    
    return result


@partial(jax.jit, static_argnames=['expression', 'q', 'num_rounds'])
def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    keys = list(eval_tables.keys())
    
    # stack tables, for vectorized access
    table_stack = jnp.stack([eval_tables[k] for k in keys], axis=0)  # shape (num_tables, 2^n) - all tables, all entries
    
    degree = max(len(term) for term in expression)
    t_vals = jnp.arange(degree + 1, dtype=jnp.uint32)
    
    all_round_evals = []
    claimed_sum = None
    
    for round_idx in range(num_rounds):
        z = table_stack[:, 0::2]  # shape (num_tables, 2^(n-1)) - all tables, even entries
        o = table_stack[:, 1::2]  # shape (num_tables, 2^(n-1)) - all tables, odd entries
        
        def eval_at_t(t):
            # vmap mle_update across all tables simultaneously
            t_stack = jax.vmap(
                lambda z_row, o_row: mle_update_32(z_row, o_row, t, q=q)
            )(z, o)
            
            return jnp.sum(eval_expression(expression, t_stack, keys, q)) % q
        
        round_evals = jax.vmap(eval_at_t)(t_vals)
        
        if round_idx == 0:
            claimed_sum = (round_evals[0] + round_evals[1]) % q
        
        all_round_evals.append(round_evals)
        
        # update table_stack for next round
        r = jnp.uint32(challenges[round_idx])
        table_stack = jax.vmap(
            lambda z_row, o_row: mle_update_32(z_row, o_row, r, q=q)
        )(z, o)
    
    return claimed_sum, jnp.stack(all_round_evals)


# def sumcheck_64(eval_tables, *, q, expression, challenges, num_rounds):
#     """Optional 64-bit sumcheck path."""
#     # TODO(student): implement when enabling 64-bit track.
#     raise NotImplementedError


# def sumcheck_128(eval_tables, *, q, expression, challenges, num_rounds):
#     """Optional 128-bit sumcheck path."""
#     # TODO(student): implement when enabling 128-bit track.
#     raise NotImplementedError


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


# if __name__ == "__main__":
#     q = 17

#     print("=== Case 1: f = a ===")
#     table_a = jnp.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=jnp.uint32)
#     challenges = [jnp.uint32(8), jnp.uint32(5), jnp.uint32(11)]

#     claimed_sum, round_evals = sumcheck_32(
#         {'a': table_a},
#         q=q,
#         expression=(('a',),),
#         challenges=challenges,
#         num_rounds=3,
#     )
#     print(f"claimed_sum: {claimed_sum}")   # expect 11
#     print(f"round_evals:\n{round_evals}")  # expect [[12,16],[3,7],[1,5]]

#     print("\n=== Case 2: f = a*b ===")
#     # 2 variables, x1=LSB
#     # a = x1, b = x2
#     # idx | (x2,x1) | a | b | a*b
#     #  0  |  (0,0)  | 0 | 0 |  0
#     #  1  |  (0,1)  | 1 | 0 |  0
#     #  2  |  (1,0)  | 0 | 1 |  0
#     #  3  |  (1,1)  | 1 | 1 |  1
#     # sum = 1, claimed_sum = 1 mod 17
#     table_a2 = jnp.array([0, 1, 0, 1], dtype=jnp.uint32)
#     table_b2 = jnp.array([0, 0, 1, 1], dtype=jnp.uint32)
#     challenges2 = [jnp.uint32(3), jnp.uint32(7)]  # r1=3, r2=7 (verifier's)

#     claimed_sum2, round_evals2 = sumcheck_32(
#         {'a': table_a2, 'b': table_b2},
#         q=q,
#         expression=(('a', 'b'),),
#         challenges=challenges2,
#         num_rounds=2,
#     )
#     print(f"claimed_sum: {claimed_sum2}")   # expect 1
#     print(f"round_evals:\n{round_evals2}")  # each row has 3 values: g(0), g(1), g(2)