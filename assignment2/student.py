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


def eval_expression(expression, var_map, q):
    """
    expression is a list of lists e.g. [['a','b'], ['c']] meaning a*b + c
    var_map is a dict mapping variable name to array
    """
    result = None
    for term in expression:
        # multiply all variables in this term together
        term_val = None
        for var in term:
            arr = var_map[var]
            if term_val is None:
                term_val = arr
            else:
                term_val = mod_mul_32(term_val, arr, q)
        # add this term to the running total
        if result is None:
            result = term_val
        else:
            result = mod_add_32(result, term_val, q)
    return result


@partial(jax.jit, static_argnames=['expression', 'q', 'num_rounds'])
def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    keys = list(eval_tables.keys())
    tables = {k: eval_tables[k].copy() for k in keys}
    
    # determine degree from expression: max term length
    degree = max(len(term) for term in expression)
    
    all_round_evals = []
    claimed_sum = None
    
    for round_idx in range(num_rounds):
        z_map = {k: tables[k][0::2] for k in keys}
        o_map = {k: tables[k][1::2] for k in keys}
        
        # compute g(0) and g(1)
        g0 = jnp.sum(eval_expression(expression, z_map, q).astype(jnp.uint64)) % q
        g1 = jnp.sum(eval_expression(expression, o_map, q).astype(jnp.uint64)) % q
        
        round_evals = [g0, g1]
        
        # for degree > 1, compute g(2), g(3), ... g(degree)
        for t in range(2, degree + 1):
            t_val = jnp.uint32(t)
            t_map = {k: mle_update_32(z_map[k], o_map[k], t_val, q=q) for k in keys}
            gt = jnp.sum(eval_expression(expression, t_map, q).astype(jnp.uint64)) % q
            round_evals.append(gt)
        
        if round_idx == 0:
            claimed_sum = (g0 + g1) % q
        
        all_round_evals.append(jnp.array(round_evals, dtype=jnp.uint64))
        
        r = jnp.uint32(challenges[round_idx])
        for k in keys:
            tables[k] = mle_update_32(z_map[k], o_map[k], r, q=q)
    
    round_evals = jnp.stack(all_round_evals)
    return claimed_sum, round_evals


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
#     challenges = [jnp.uint32(8), jnp.uint32(5), jnp.uint32(11)]  # all 3, r3 is verifier's

#     claimed_sum, round_evals = sumcheck_32(
#         {'a': table_a},
#         q=q,
#         expression=[['a']],
#         challenges=challenges,
#         num_rounds=3,
#     )
#     print(f"claimed_sum: {claimed_sum}")   # expect 11
#     print(f"round_evals:\n{round_evals}")  # expect [[12,16],[3,7],[1,5]]