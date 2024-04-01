from ....domain import Radix2EvaluationDomain
import gmpy2
import torch
import copy
import torch.nn.functional as F
from ....bls12_381 import fr
from ....arithmetic import INTT,coset_NTT,coset_INTT,from_coeff_vec
from ....plonk_core.src.proof_system.widget.mod import WitnessValues
from ....plonk_core.src.proof_system.widget.range import RangeGate,RangeValues
from ....plonk_core.src.proof_system.widget.logic import LogicGate,LogicValues
from ....plonk_core.src.proof_system.widget.fixed_base_scalar_mul import FBSMGate,FBSMValues
from ....plonk_core.src.proof_system.widget.curve_addition import CAGate,CAValues
from ....plonk_core.src.proof_system.mod import CustomEvaluations
from ....arithmetic import INTT,from_coeff_vec,resize,\
                        from_gmpy_list,from_list_gmpy,from_list_tensor,from_tensor_list,from_gmpy_list_1,domian_trans_tensor,calculate_execution_time

# Computes the first lagrange polynomial with the given `scale` over `domain`.
def compute_first_lagrange_poly_scaled(domain: Radix2EvaluationDomain,scale: fr.Fr):
    # x_evals = [fr.Fr.zero() for _ in range(domain.size)]
    
    x_evals=torch.zeros(domain.size,4,dtype=torch.BLS12_381_Fr_G1_Mont)
    x_evals[0] = scale
    # from_gmpy_list(x_evals)
    # x_evals_tensor=from_list_tensor(x_evals)
    x_coeffs = INTT(domain,x_evals)
    result_poly = from_coeff_vec(x_coeffs)
    return result_poly

def compute_gate_constraint_satisfiability(domain, 
    range_challenge, logic_challenge, fixed_base_challenge,
    var_base_challenge, prover_key, wl_eval_8n, wr_eval_8n, 
    wo_eval_8n, w4_eval_8n, pi_poly):

    #get Fr
    params = fr.Fr(gmpy2.mpz(0))
    domain_8n = Radix2EvaluationDomain.new(8 * domain.size,params)
    
    # domian_trans_tensor(domain_8n.group_gen_inv)
    # domian_trans_tensor(domain_8n.size_inv)
    # domian_trans_tensor(domain_8n.group_gen)

    pi_eval_8n = coset_NTT(pi_poly,domain_8n)
    # pi_eval_8n=from_tensor_list(pi_eval_8n)
    # from_list_gmpy(pi_eval_8n)

    gate_contributions = []


    ##TODO 8192
    # from_gmpy_list(prover_key.arithmetic.q_l[1])
    # prover_key.arithmetic.q_l[1][:4096]=from_list_tensor(prover_key.arithmetic.q_l[1][:4096])
    # prover_key.arithmetic.q_l[1][4096:]=from_list_tensor(prover_key.arithmetic.q_l[1][4096:])
    # # self.q_l[1]=q_l1
    # from_gmpy_list(prover_key.arithmetic.q_r[1])
    # prover_key.arithmetic.q_r[1][:4096] = from_list_tensor(prover_key.arithmetic.q_r[1][:4096])
    # prover_key.arithmetic.q_r[1][4096:] = from_list_tensor(prover_key.arithmetic.q_r[1][4096:])

    # from_gmpy_list(prover_key.arithmetic.q_o[1])
    # prover_key.arithmetic.q_o[1][:4096] = from_list_tensor(prover_key.arithmetic.q_o[1][:4096])
    # prover_key.arithmetic.q_o[1][4096:] = from_list_tensor(prover_key.arithmetic.q_o[1][4096:])

    # from_gmpy_list(prover_key.arithmetic.q_4[1])
    # prover_key.arithmetic.q_4[1][:4096] = from_list_tensor(prover_key.arithmetic.q_4[1][:4096])
    # prover_key.arithmetic.q_4[1][4096:] = from_list_tensor(prover_key.arithmetic.q_4[1][4096:])

    # from_gmpy_list(prover_key.arithmetic.q_hl[1])
    # prover_key.arithmetic.q_hl[1][:4096] = from_list_tensor(prover_key.arithmetic.q_hl[1][:4096])
    # prover_key.arithmetic.q_hl[1][4096:] = from_list_tensor(prover_key.arithmetic.q_hl[1][4096:])

    # from_gmpy_list(prover_key.arithmetic.q_hr[1])
    # prover_key.arithmetic.q_hr[1][:4096] = from_list_tensor(prover_key.arithmetic.q_hr[1][:4096])
    # prover_key.arithmetic.q_hr[1][4096:] = from_list_tensor(prover_key.arithmetic.q_hr[1][4096:])

    # from_gmpy_list(prover_key.arithmetic.q_h4[1])
    # prover_key.arithmetic.q_h4[1][:4096] = from_list_tensor(prover_key.arithmetic.q_h4[1][:4096])
    # prover_key.arithmetic.q_h4[1][4096:] = from_list_tensor(prover_key.arithmetic.q_h4[1][4096:])

    # from_gmpy_list(prover_key.arithmetic.q_c[1])
    # prover_key.arithmetic.q_c[1][:4096] = from_list_tensor(prover_key.arithmetic.q_c[1][:4096])
    # prover_key.arithmetic.q_c[1][4096:] = from_list_tensor(prover_key.arithmetic.q_c[1][4096:])

    # from_gmpy_list(prover_key.arithmetic.q_arith[1])
    # prover_key.arithmetic.q_arith[1][:4096] = from_list_tensor(prover_key.arithmetic.q_arith[1][:4096])
    # prover_key.arithmetic.q_arith[1][4096:] = from_list_tensor(prover_key.arithmetic.q_arith[1][4096:])

    # from_gmpy_list(prover_key.arithmetic.q_m[1])
    # prover_key.arithmetic.q_m[1][:4096] = from_list_tensor(prover_key.arithmetic.q_m[1][:4096])
    # prover_key.arithmetic.q_m[1][4096:] = from_list_tensor(prover_key.arithmetic.q_m[1][4096:])


    for i in range(domain_8n.size):
        wit_vals = WitnessValues(
            a_val=wl_eval_8n[i],
            b_val=wr_eval_8n[i],
            c_val=wo_eval_8n[i],
            d_val=w4_eval_8n[i]
        )

        custom_vals = CustomEvaluations(
            vals=[
                ("a_next_eval", wl_eval_8n[i + 8]),
                ("b_next_eval", wr_eval_8n[i + 8]),
                ("d_next_eval", w4_eval_8n[i + 8]),
                ("q_l_eval", copy.deepcopy(prover_key.arithmetic.q_l[1][i])),
                ("q_r_eval", copy.deepcopy(prover_key.arithmetic.q_r[1][i])),
                ("q_c_eval", copy.deepcopy(prover_key.arithmetic.q_c[1][i])),
                # Possibly unnecessary but included nonetheless...
                ("q_hl_eval", copy.deepcopy(prover_key.arithmetic.q_hl[1][i])),
                ("q_hr_eval", copy.deepcopy(prover_key.arithmetic.q_hr[1][i])),
                ("q_h4_eval", copy.deepcopy(prover_key.arithmetic.q_hr[1][i]))
            ]
        )

        arithmetic = prover_key.arithmetic.compute_quotient_i(i, wit_vals)
        # range_selector=torch.tensor(from_gmpy_list_1(prover_key.range_selector[1][i]),dtype=torch.BLS12_381_Fr_G1_Mont)
        range_term = RangeGate.quotient_term(
            prover_key.range_selector[1][i],
            range_challenge,
            wit_vals,
            custom_vals = RangeValues.from_evaluations(custom_vals)
        )
        # logic_selector=torch.tensor(from_gmpy_list_1(prover_key.logic_selector[1][i]),dtype=torch.BLS12_381_Fr_G1_Mont)
        logic_term = LogicGate.quotient_term(
            prover_key.logic_selector[1][i],
            logic_challenge,
            wit_vals,
            LogicValues.from_evaluations(custom_vals)
        )
        # fixed_group_add_selector=torch.tensor(from_gmpy_list_1(prover_key.fixed_group_add_selector[1][i]),dtype=torch.BLS12_381_Fr_G1_Mont)
        fixed_base_scalar_mul_term = FBSMGate.quotient_term(
            prover_key.fixed_group_add_selector[1][i],
            fixed_base_challenge,
            wit_vals,
            FBSMValues.from_evaluations(custom_vals)
        )
        # variable_group_add_selector=torch.tensor(from_gmpy_list_1(prover_key.variable_group_add_selector[1][i]),dtype=torch.BLS12_381_Fr_G1_Mont)
        curve_addition_term = CAGate.quotient_term(
            prover_key.variable_group_add_selector[1][i],
            var_base_challenge,
            wit_vals,
            CAValues.from_evaluations(custom_vals)
        )

        mid1 = F.add_mod(arithmetic ,pi_eval_8n[i])
        mid2 = F.add_mod(mid1, range_term)
        mid3 = F.add_mod(mid2, logic_term)
        mid4 = F.add_mod(mid3, fixed_base_scalar_mul_term)
        gate_i = F.add_mod(mid4, curve_addition_term)
        gate_contributions.append(gate_i)


    return gate_contributions

@calculate_execution_time
def compute_permutation_checks(
    domain:Radix2EvaluationDomain,
    prover_key,
    wl_eval_8n: list[fr.Fr], wr_eval_8n: list[fr.Fr],
    wo_eval_8n: list[fr.Fr], w4_eval_8n: list[fr.Fr],
    z_eval_8n: list[fr.Fr], alpha: fr.Fr, beta: fr.Fr, gamma: fr.Fr):

    #get Fr
    params = fr.Fr(gmpy2.mpz(0))
    #get NTT domain
    domain_8n:Radix2EvaluationDomain = Radix2EvaluationDomain.new(8 * domain.size,params)
    domian_trans_tensor(domain_8n.group_gen_inv)
    domian_trans_tensor(domain_8n.size_inv)
    domian_trans_tensor(domain_8n.group_gen)
    
    # Calculate l1_poly_alpha and l1_alpha_sq_evals
    alpha=torch.tensor(from_gmpy_list_1(alpha),dtype=torch.BLS12_381_Fr_G1_Mont)
    gamma=torch.tensor(from_gmpy_list_1(gamma),dtype=torch.BLS12_381_Fr_G1_Mont)
    beta=torch.tensor(from_gmpy_list_1(beta),dtype=torch.BLS12_381_Fr_G1_Mont)

    alpha2= F.mul_mod(alpha,alpha)
    l1_poly_alpha = compute_first_lagrange_poly_scaled(domain, alpha2)
    l1_alpha_sq_evals = coset_NTT(l1_poly_alpha, domain_8n)

    # Initialize result list
    result = []

    # Calculate permutation contribution for each index
    for i in range(domain_8n.size):
        quotient_i = prover_key.permutation.compute_quotient_i(
            i,
            wl_eval_8n[i],
            wr_eval_8n[i],
            wo_eval_8n[i],
            w4_eval_8n[i],
            z_eval_8n[i],
            z_eval_8n[i + 8],
            alpha,
            l1_alpha_sq_evals[i],
            beta,
            gamma
        )
        result.append(quotient_i)

    return result

@calculate_execution_time
def compute(domain: Radix2EvaluationDomain, 
            prover_key, 
            z_poly, z2_poly, 
            w_l_poly, w_r_poly, w_o_poly, w_4_poly, 
            public_inputs_poly, 
            f_poly, table_poly, h1_poly, h2_poly, 
            alpha: fr.Fr, beta, gamma, delta, epsilon, zeta, 
            range_challenge, logic_challenge, 
            fixed_base_challenge, var_base_challenge, 
            lookup_challenge):
    
    #get Fr
    params = fr.Fr(gmpy2.mpz(0))
    #get NTT domain
    domain_8n = Radix2EvaluationDomain.new(8 * domain.size,params)
    domian_trans_tensor(domain_8n.group_gen_inv)
    domian_trans_tensor(domain_8n.size_inv)
    domian_trans_tensor(domain_8n.group_gen)
    l1_poly = compute_first_lagrange_poly_scaled(domain, torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont)) ########输出为Tensor

    l1_eval_8n = coset_NTT(l1_poly,domain_8n)
    z_eval_8n = coset_NTT(z_poly,domain_8n)
    z_eval_8n=from_tensor_list(z_eval_8n)
    #TODO 解决tensor的维数扩展
    z_eval_8n += z_eval_8n[:8]
    # expanded_tensor = torch.concate((z_eval_8n, z_eval_8n[:8]), dim=0)
    # from_list_gmpy(z_eval_8n)

    wl_eval_8n = coset_NTT(w_l_poly,domain_8n)
    wl_eval_8n=from_tensor_list(wl_eval_8n)
    wl_eval_8n += wl_eval_8n[:8]
    # from_list_gmpy(wl_eval_8n)

    wr_eval_8n = coset_NTT(w_r_poly,domain_8n)
    wr_eval_8n=from_tensor_list(wr_eval_8n)
    wr_eval_8n += wr_eval_8n[:8]
    # from_list_gmpy(wr_eval_8n)

    wo_eval_8n = coset_NTT(w_o_poly,domain_8n)
    wo_eval_8n=from_tensor_list(wo_eval_8n)
    # from_list_gmpy(wo_eval_8n)

    w4_eval_8n = coset_NTT(w_4_poly,domain_8n)
    w4_eval_8n=from_tensor_list(w4_eval_8n)
    w4_eval_8n += w4_eval_8n[:8]
    # from_list_gmpy(w4_eval_8n)
    z2_eval_8n = coset_NTT(z2_poly,domain_8n)
    z2_eval_8n=from_tensor_list(z2_eval_8n)
    z2_eval_8n +=z2_eval_8n[:8]
    # from_list_gmpy(z2_eval_8n)

    f_eval_8n = coset_NTT(f_poly,domain_8n)
    # f_eval_8n=from_tensor_list(f_eval_8n)
    # from_list_gmpy(f_eval_8n)

    table_eval_8n = coset_NTT(table_poly,domain_8n)
    table_eval_8n=from_tensor_list(table_eval_8n)
    table_eval_8n += table_eval_8n[:8]
    # from_list_gmpy(table_eval_8n)

    h1_eval_8n = coset_NTT(h1_poly,domain_8n)
    h1_eval_8n=from_tensor_list(h1_eval_8n)
    h1_eval_8n += h1_eval_8n[:8]
    # from_list_gmpy(h1_eval_8n)

    # h2_poly=from_tensor_list(h2_poly)
    h2_eval_8n = coset_NTT(h2_poly,domain_8n)
    # h2_eval_8n=from_tensor_list(h2_eval_8n)
    # from_list_gmpy(h2_eval_8n)

    #TODO有问题
    z_eval_8n=from_list_tensor(z_eval_8n)
    wl_eval_8n=from_list_tensor(wl_eval_8n)
    wr_eval_8n =from_list_tensor(wr_eval_8n)
    wo_eval_8n=from_list_tensor(wo_eval_8n)
    w4_eval_8n=from_list_tensor(w4_eval_8n)
    z2_eval_8n=from_list_tensor(z2_eval_8n)
    table_eval_8n=from_list_tensor(table_eval_8n)
    h1_eval_8n=from_list_tensor(h1_eval_8n)

    range_challenge=torch.tensor(from_gmpy_list_1(range_challenge),dtype=torch.BLS12_381_Fr_G1_Mont)
    logic_challenge = torch.tensor(from_gmpy_list_1(logic_challenge), dtype=torch.BLS12_381_Fr_G1_Mont)
    fixed_base_challenge = torch.tensor(from_gmpy_list_1(fixed_base_challenge), dtype=torch.BLS12_381_Fr_G1_Mont)
    var_base_challenge = torch.tensor(from_gmpy_list_1(var_base_challenge), dtype=torch.BLS12_381_Fr_G1_Mont)
    lookup_challenge =torch.tensor(from_gmpy_list_1(lookup_challenge),dtype=torch.BLS12_381_Fr_G1_Mont)

    gate_constraints = compute_gate_constraint_satisfiability(
        domain,
        range_challenge,logic_challenge,
        fixed_base_challenge,var_base_challenge,
        prover_key,
        wl_eval_8n,wr_eval_8n,wo_eval_8n,w4_eval_8n,
        public_inputs_poly,
    )

    permutation = compute_permutation_checks(
        domain,
        prover_key,
        wl_eval_8n,wr_eval_8n,wo_eval_8n,w4_eval_8n,z_eval_8n,
        alpha,beta,gamma,
    )

    lookup = prover_key.lookup.compute_lookup_quotient_term(
        domain,
        wl_eval_8n,
        wr_eval_8n,
        wo_eval_8n,
        w4_eval_8n,
        f_eval_8n,
        table_eval_8n,
        h1_eval_8n,
        h2_eval_8n,
        z2_eval_8n,
        l1_eval_8n,
        delta,
        epsilon,
        zeta,
        lookup_challenge,
    )
    quotient = torch.empty(domain_8n.size,4,dtype=torch.BLS12_381_Fr_G1_Mont)
    for i in range(domain_8n.size):
        numerator = F.add_mod(gate_constraints[i],permutation[i])
        numerator = F.add_mod(numerator,lookup[i])
        denominator=F.div_mod(torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont),prover_key.v_h_coset_8n[i])
        res =F.mul_mod(numerator,denominator)
        quotient[i]=res

    quotient_poly = coset_INTT(quotient,domain_8n)
    hx = from_coeff_vec(quotient_poly)

    return hx