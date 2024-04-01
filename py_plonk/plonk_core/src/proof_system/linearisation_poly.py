from dataclasses import dataclass
from typing import List, Tuple
from ....domain import Radix2EvaluationDomain
from ....bls12_381 import fr
from ....plonk_core.src.proof_system.prover_key import Prover_Key
from ....plonk_core.src.proof_system.widget.mod import WitnessValues
from ....plonk_core.src.proof_system.mod import CustomEvaluations
from ....plonk_core.src.proof_system.widget.range import RangeGate,RangeValues
from ....plonk_core.src.proof_system.widget.logic import LogicGate,LogicValues
from ....plonk_core.src.proof_system.widget.fixed_base_scalar_mul import FBSMGate,FBSMValues
from ....plonk_core.src.proof_system.widget.curve_addition import CAGate,CAValues
from ....arithmetic import poly_mul_const,poly_add_poly,evaluate,compute_first_lagrange_evaluation,from_gmpy_list,from_list_gmpy,from_list_tensor,from_tensor_list,from_gmpy_list_1,calculate_execution_time,neg
import torch
import torch.nn.functional as F
@dataclass
class WireEvaluations:
    # Evaluation of the witness polynomial for the left wire at `z`.
    a_eval: fr.Fr

    # Evaluation of the witness polynomial for the right wire at `z`.
    b_eval: fr.Fr

    # Evaluation of the witness polynomial for the output wire at `z`.
    c_eval: fr.Fr

    # Evaluation of the witness polynomial for the fourth wire at `z`.
    d_eval: fr.Fr

@dataclass
class PermutationEvaluations:
    # Evaluation of the left sigma polynomial at `z`.
    left_sigma_eval: fr.Fr

    # Evaluation of the right sigma polynomial at `z`.
    right_sigma_eval: fr.Fr

    # Evaluation of the out sigma polynomial at `z`.
    out_sigma_eval: fr.Fr

    # Evaluation of the permutation polynomial at `z * omega` where `omega`
    # is a root of unity.
    permutation_eval: fr.Fr

@dataclass 
class LookupEvaluations:
    q_lookup_eval: fr.Fr

    # (Shifted) Evaluation of the lookup permutation polynomial at `z * root of unity`
    z2_next_eval: fr.Fr

    # Evaluations of the first half of sorted plonkup poly at `z`
    h1_eval: fr.Fr

    # (Shifted) Evaluations of the even indexed half of sorted plonkup poly
    # at `z root of unity
    h1_next_eval: fr.Fr

    # Evaluations of the odd indexed half of sorted plonkup poly at `z
    # root of unity
    h2_eval: fr.Fr

    # Evaluations of the query polynomial at `z`
    f_eval: fr.Fr

    # Evaluations of the table polynomial at `z`
    table_eval: fr.Fr

    # Evaluations of the table polynomial at `z * root of unity`
    table_next_eval: fr.Fr

@dataclass 
class ProofEvaluations:
    # Wire evaluations
    wire_evals: WireEvaluations

    # Permutation and sigma polynomials evaluations
    perm_evals: PermutationEvaluations

    # Lookup evaluations
    lookup_evals: LookupEvaluations

    # Evaluations needed for custom gates. This includes selector polynomials
    # and evaluations of wire polynomials at an offset
    custom_evals: CustomEvaluations

@calculate_execution_time
def compute(
    domain: Radix2EvaluationDomain,
    prover_key: Prover_Key,
    alpha: fr.Fr,
    beta: fr.Fr,
    gamma: fr.Fr,
    delta: fr.Fr,
    epsilon: fr.Fr,
    zeta: fr.Fr,
    range_separation_challenge: fr.Fr,
    logic_separation_challenge: fr.Fr,
    fixed_base_separation_challenge: fr.Fr,
    var_base_separation_challenge: fr.Fr,
    lookup_separation_challenge: fr.Fr,
    z_challenge: fr.Fr,
    w_l_poly: torch.tensor,
    w_r_poly: torch.tensor,
    w_o_poly: torch.tensor,
    w_4_poly: torch.tensor,
    t_1_poly: torch.tensor,
    t_2_poly: torch.tensor,
    t_3_poly: torch.tensor,
    t_4_poly: torch.tensor,
    t_5_poly: torch.tensor,
    t_6_poly: torch.tensor,
    t_7_poly: torch.tensor,
    t_8_poly: torch.tensor,
    z_poly: torch.tensor,
    z2_poly: torch.tensor,
    f_poly: torch.tensor,
    h1_poly: torch.tensor,
    h2_poly: torch.tensor,
    table_poly: torch.tensor
    ):
    n = domain.size
    omega = domain.group_gen
    domain_permutation = Radix2EvaluationDomain.new(n,z_challenge)
    z_challenge =torch.tensor(from_gmpy_list_1(z_challenge),dtype=torch.BLS12_381_Fr_G1_Mont)
    omega =torch.tensor(from_gmpy_list_1(omega),dtype=torch.BLS12_381_Fr_G1_Mont)
    shifted_z_challenge = F.mul_mod(z_challenge, omega)

    # Wire evaluations
    a_eval = evaluate(w_l_poly, z_challenge)
    b_eval = evaluate(w_r_poly, z_challenge)
    c_eval = evaluate(w_o_poly, z_challenge)
    d_eval = evaluate(w_4_poly, z_challenge)

    wire_evals = WireEvaluations(a_eval,b_eval,c_eval,d_eval)

    # Permutation evaluations
    left_sigma_eval = evaluate(prover_key.permutation.left_sigma[0], z_challenge)
    right_sigma_eval = evaluate(prover_key.permutation.right_sigma[0], z_challenge)
    out_sigma_eval = evaluate(prover_key.permutation.out_sigma[0], z_challenge)
    permutation_eval = evaluate(z_poly, shifted_z_challenge)

    perm_evals = PermutationEvaluations(
        left_sigma_eval,
        right_sigma_eval,
        out_sigma_eval,
        permutation_eval
    )

    #TODO 不能创建空的tensor
    # from_gmpy_list(prover_key.lookup.q_lookup[0])
    # from_gmpy_list(prover_key.arithmetic.q_c[0])
    # from_gmpy_list(prover_key.arithmetic.q_l[0])
    # from_gmpy_list(prover_key.arithmetic.q_r[0])
    # from_gmpy_list(prover_key.arithmetic.q_arith[0])
    # from_gmpy_list(prover_key.arithmetic.q_hl[0])
    # from_gmpy_list(prover_key.arithmetic.q_hr[0])
    # from_gmpy_list(prover_key.arithmetic.q_h4[0])

    # # prover_key_lookup_q_lookup0=from_list_tensor(prover_key.lookup.q_lookup[0])
    # prover_key_arithmetic_q_c0=from_list_tensor(prover_key.arithmetic.q_c[0])
    # prover_key_arithmetic_q_l0=from_list_tensor(prover_key.arithmetic.q_l[0])
    # prover_key_arithmetic_q_r0=from_list_tensor(prover_key.arithmetic.q_r[0])
    # prover_key_arithmetic_q_arith0=from_list_tensor(prover_key.arithmetic.q_arith[0])
    # prover_key_arithmetic_q_hl0 = from_list_tensor(prover_key.arithmetic.q_hl[0])
    # prover_key_arithmetic_q_hr0 = from_list_tensor(prover_key.arithmetic.q_hr[0])
    # prover_key_arithmetic_q_h40 = from_list_tensor(prover_key.arithmetic.q_h4[0])
    # Arith selector evaluation
    q_arith_eval = evaluate(prover_key.arithmetic.q_arith[0], z_challenge)

    # Lookup selector evaluation
    q_lookup_eval = evaluate(prover_key.lookup.q_lookup[0], z_challenge)

    # Custom gate evaluations
    q_c_eval = evaluate(prover_key.arithmetic.q_c[0], z_challenge)
    q_l_eval = evaluate(prover_key.arithmetic.q_l[0], z_challenge)
    q_r_eval = evaluate(prover_key.arithmetic.q_r[0], z_challenge)
    a_next_eval = evaluate(w_l_poly, shifted_z_challenge)
    b_next_eval = evaluate(w_r_poly, shifted_z_challenge)
    d_next_eval = evaluate(w_4_poly, shifted_z_challenge)

    # High degree selector evaluations
    q_hl_eval = evaluate(prover_key.arithmetic.q_hl[0], z_challenge)
    q_hr_eval = evaluate(prover_key.arithmetic.q_hr[0], z_challenge)
    q_h4_eval = evaluate(prover_key.arithmetic.q_h4[0], z_challenge)

    custom_evals = CustomEvaluations(
        [("q_arith_eval", q_arith_eval),
         ("q_c_eval", q_c_eval),
         ("q_l_eval", q_l_eval),
         ("q_r_eval", q_r_eval),
         ("q_hl_eval", q_hl_eval),
         ("q_hr_eval", q_hr_eval),
         ("q_h4_eval", q_h4_eval),
         ("a_next_eval", a_next_eval),
         ("b_next_eval", b_next_eval),
         ("d_next_eval", d_next_eval)]
    )


    z2_next_eval = evaluate(z2_poly, shifted_z_challenge)
    h1_eval = evaluate(h1_poly, z_challenge)
    h1_next_eval = evaluate(h1_poly, shifted_z_challenge)
    h2_eval = evaluate(h2_poly, z_challenge)
    f_eval = evaluate(f_poly, z_challenge)
    table_eval = evaluate(table_poly, z_challenge)
    table_next_eval = evaluate(table_poly, shifted_z_challenge)

    # Compute the last term in the linearisation polynomial (negative_quotient_term):
    # - Z_h(z_challenge) * [t_1(X) + z_challenge^n * t_2(X) + z_challenge^2n *
    # t_3(X) + z_challenge^3n * t_4(X)]
    one = torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont)
    vanishing_poly_eval = domain.evaluate_vanishing_polynomial(z_challenge)
    
    z_challenge_to_n = F.add_mod(vanishing_poly_eval,one)
    l1_eval = compute_first_lagrange_evaluation(
        domain,
        vanishing_poly_eval,
        z_challenge,
    )
    
    lookup_evals = LookupEvaluations(
        q_lookup_eval,
        z2_next_eval,
        h1_eval,
        h1_next_eval,
        h2_eval,
        f_eval,
        table_eval,
        table_next_eval
    )

    gate_constraints = compute_gate_constraint_satisfiability(
        range_separation_challenge,
        logic_separation_challenge,
        fixed_base_separation_challenge,
        var_base_separation_challenge,
        wire_evals,
        q_arith_eval,
        custom_evals,
        prover_key
    )

    lookup = prover_key.lookup.compute_linearisation(
        l1_eval,
        a_eval,
        b_eval,
        c_eval,
        d_eval,
        f_eval,
        table_eval,
        table_next_eval,
        h1_next_eval,
        h2_eval,
        z2_next_eval,
        delta,
        epsilon,
        zeta,
        z2_poly,
        h1_poly,
        lookup_separation_challenge,
    )
   
    permutation = prover_key.permutation.compute_linearisation(
        n,
        z_challenge,
        (alpha, beta, gamma),
        (a_eval, b_eval, c_eval, d_eval),
        (left_sigma_eval, right_sigma_eval, out_sigma_eval),
        permutation_eval,
        z_poly,
        domain_permutation
    )
    # Calculate t_8_poly * z_challenge_to_n
    term_1 = poly_mul_const(t_8_poly , z_challenge_to_n)

    # Calculate (term_1 + t_7_poly) * z_challenge_to_n
   
    term_2_1 = poly_add_poly(term_1 , t_7_poly)
    term_2 = poly_mul_const(term_2_1, z_challenge_to_n)


    # Calculate (term_2 + t_6_poly) * z_challenge_to_n
    term_3_1 = poly_add_poly(term_2 , t_6_poly)
    term_3 = poly_mul_const(term_3_1, z_challenge_to_n)


    # Calculate (term_3 + t_5_poly) * z_challenge_to_n
    term_4_1 = poly_add_poly(term_3 , t_5_poly)
    term_4 = poly_mul_const(term_4_1, z_challenge_to_n)


    # Calculate (term_4 + t_4_poly) * z_challenge_to_n
    term_5_1 = poly_add_poly(term_4 , t_4_poly)
    term_5 = poly_mul_const(term_5_1, z_challenge_to_n)


    # Calculate (term_5 + t_3_poly) * z_challenge_to_n
    term_6_1 = poly_add_poly(term_5 , t_3_poly)
    term_6 = poly_mul_const(term_6_1, z_challenge_to_n)


    # Calculate (term_6 + t_2_poly) * z_challenge_to_n
    term_7_1 = poly_add_poly(term_6 , t_2_poly)
    term_7 = poly_mul_const(term_7_1, z_challenge_to_n)


    # Calculate (term_7 + t_1_poly) * vanishing_poly_eval
    term_8_1 = poly_add_poly(term_7 , t_1_poly)
    quotient_term = poly_mul_const(term_8_1, vanishing_poly_eval)


    neg_one = neg(one)
    negative_quotient_term = poly_mul_const(quotient_term,neg_one)
    linearisation_polynomial_term_1 = poly_add_poly(gate_constraints, permutation)
    linearisation_polynomial_term_2 = poly_add_poly(lookup, negative_quotient_term)
    linearisation_polynomial = poly_add_poly(linearisation_polynomial_term_1, linearisation_polynomial_term_2)

    proof_evaluations = ProofEvaluations(wire_evals, perm_evals, lookup_evals, custom_evals)
    aa=linearisation_polynomial.tolist()
    return linearisation_polynomial, proof_evaluations

# Computes the gate constraint satisfiability portion of the linearisation polynomial.
def compute_gate_constraint_satisfiability(
    range_separation_challenge: fr.Fr,
    logic_separation_challenge: fr.Fr,
    fixed_base_separation_challenge: fr.Fr,
    var_base_separation_challenge: fr.Fr,
    wire_evals: WireEvaluations,
    q_arith_eval: fr.Fr,
    custom_evals: CustomEvaluations,
    prover_key: Prover_Key,
):
    wit_vals = WitnessValues(
        a_val = wire_evals.a_eval,
        b_val = wire_evals.b_eval,
        c_val = wire_evals.c_eval,
        d_val = wire_evals.d_eval)

    arithmetic = prover_key.arithmetic.compute_linearisation(
        wire_evals.a_eval,
        wire_evals.b_eval,
        wire_evals.c_eval,
        wire_evals.d_eval,
        q_arith_eval,
    )
    range_separation_challenge=torch.tensor(from_gmpy_list_1(range_separation_challenge),dtype=torch.BLS12_381_Fr_G1_Mont)
    range = RangeGate.linearisation_term(
        prover_key.range_selector[0],
        range_separation_challenge,
        wit_vals,
        RangeValues.from_evaluations(custom_evals),
    )

    logic_separation_challenge=torch.tensor(from_gmpy_list_1(logic_separation_challenge),dtype=torch.BLS12_381_Fr_G1_Mont)
    logic = LogicGate.linearisation_term(
        prover_key.logic_selector[0],
        logic_separation_challenge,
        wit_vals,
        LogicValues.from_evaluations(custom_evals),
    )

    fixed_base_separation_challenge=torch.tensor(from_gmpy_list_1(fixed_base_separation_challenge),dtype=torch.BLS12_381_Fr_G1_Mont)
    fixed_base_scalar_mul = FBSMGate.linearisation_term(
        prover_key.fixed_group_add_selector[0],
        fixed_base_separation_challenge,
        wit_vals,
        FBSMValues.from_evaluations(custom_evals),
    )

    var_base_separation_challenge=torch.tensor(from_gmpy_list_1(var_base_separation_challenge),dtype=torch.BLS12_381_Fr_G1_Mont)
    curve_addition = CAGate.linearisation_term(
        prover_key.variable_group_add_selector[0],
        var_base_separation_challenge,
        wit_vals,
        CAValues.from_evaluations(custom_evals),
    )

    mid1 = poly_add_poly(arithmetic, range)  ###返回值为tensor
    mid2 = poly_add_poly(mid1, logic)
    mid3 = poly_add_poly(mid2, fixed_base_scalar_mul)
    res = poly_add_poly(mid3, curve_addition)
    return res

