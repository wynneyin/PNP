from dataclasses import dataclass
from .....bls12_381 import fr
import torch
import torch.nn.functional as F
from typing import List, Tuple
from .....plonk_core.src.proof_system.widget.mod import WitnessValues
from .....plonk_core.src.constraint_system.hash import SBOX_ALPHA
from .....arithmetic import poly_mul_const,poly_add_poly,from_gmpy_list,from_list_gmpy,from_list_tensor,from_tensor_list,pow_1
@dataclass
class Arith:
    q_m: Tuple[List,List]
    q_l: Tuple[List,List]
    q_r: Tuple[List,List]
    q_o: Tuple[List,List]
    q_4: Tuple[List,List]
    q_hl: Tuple[List,List]
    q_hr: Tuple[List,List]
    q_h4: Tuple[List,List]
    q_c: Tuple[List,List]
    q_arith: Tuple[List,List]

    # Computes the arithmetic gate contribution to the quotient polynomial at
    # the element of the domain at the given `index`.
    def compute_quotient_i(self, index: int, wit_vals: WitnessValues):

        

        mult = F.mul_mod(wit_vals.a_val, wit_vals.b_val)
        mult = F.mul_mod(mult,self.q_m[1][index]) 
        left = F.mul_mod(wit_vals.a_val, self.q_l[1][index])
        right = F.mul_mod(wit_vals.b_val, self.q_r[1][index])
        out = F.mul_mod(wit_vals.c_val, self.q_o[1][index])
        fourth = F.mul_mod(wit_vals.d_val, self.q_4[1][index])
        ###TODO pow not fixed
        # a_high = wit_vals.a_val.pow_1(SBOX_ALPHA)
        # b_high = wit_vals.b_val.pow_1(SBOX_ALPHA)
        # f_high = wit_vals.d_val.pow_1(SBOX_ALPHA)
        a_high = pow_1( wit_vals.a_val,SBOX_ALPHA)
        b_high = pow_1(wit_vals.b_val,SBOX_ALPHA)
        f_high = pow_1(wit_vals.d_val,SBOX_ALPHA)

        a_high = F.mul_mod(a_high, self.q_hl[1][index])
        b_high = F.mul_mod(b_high, self.q_hr[1][index])
        f_high = F.mul_mod(f_high, self.q_h4[1][index])

        mid1 = F.add_mod(mult, left)
        mid2 = F.add_mod(mid1, right)
        mid3 = F.add_mod(mid2, out)
        mid4 = F.add_mod(mid3, fourth)
        mid5 = F.add_mod(mid4, a_high)
        mid6 = F.add_mod(mid5, b_high)
        mid7 = F.add_mod(mid6, f_high)
        mid8 = F.add_mod(mid7, self.q_c[1][index])

        arith_val = F.mul_mod(mid8, self.q_arith[1][index])

        return arith_val
    # Computes the arithmetic gate contribution to the linearisation
    # polynomial at the given evaluation points.
    

    def compute_linearisation(
        self, 
        a_eval: fr.Fr,
        b_eval: fr.Fr, 
        c_eval: fr.Fr, 
        d_eval: fr.Fr, 
        q_arith_eval: fr.Fr):
        mid1_1 =F.mul_mod(a_eval,b_eval)

        # from_gmpy_list(self.q_m[0])
        # # from_gmpy_list(self.q_l[0])
        # # from_gmpy_list(self.q_r[0])
        # from_gmpy_list(self.q_o[0])
        # from_gmpy_list(self.q_4[0])
        # # from_gmpy_list(self.q_hl[0])
        

        # q_l0 = from_list_tensor(self.q_l[0])
        # q_r0 = from_list_tensor(self.q_r[0])
        # q_o0 = from_list_tensor(self.q_o[0])
        # q_40 = from_list_tensor(self.q_4[0])
        # q_hl0=from_list_tensor(self.q_hl[0])
        # q_hr0=from_list_tensor(self.q_hr[0])
        # q_h40=from_list_tensor(self.q_h4[0])
        # q_c0=from_list_tensor(self.q_c[0])

        mid1 = poly_mul_const(self.q_m[0] ,mid1_1)
        mid2 = poly_mul_const(self.q_l[0] ,a_eval)
        mid3 = poly_mul_const(self.q_r[0] ,b_eval)
        mid4 = poly_mul_const(self.q_o[0] ,c_eval)
        mid5 = poly_mul_const(self.q_4[0] ,d_eval)
        mid6_1 = pow_1(a_eval,SBOX_ALPHA)
        mid6 = poly_mul_const(self.q_hl[0] ,mid6_1)
        mid7_1 =pow_1(b_eval,SBOX_ALPHA)
        mid7 = poly_mul_const(self.q_hr[0] ,mid7_1)
        mid8_1 = pow_1(d_eval,SBOX_ALPHA)
        mid8 = poly_mul_const(self.q_h4[0],mid8_1)

        add1 = poly_add_poly(mid1, mid2)   
        add2 = poly_add_poly(add1, mid3)
        add3 = poly_add_poly(add2, mid4)
        add4 = poly_add_poly(add3, mid5)
        add5 = poly_add_poly(add4, mid6)
        add6 = poly_add_poly(add5, mid7)
        add7 = poly_add_poly(add6, mid8)
        add8 = poly_add_poly(add7, self.q_c[0])
        
        result = poly_mul_const(add8, q_arith_eval)
        return result
