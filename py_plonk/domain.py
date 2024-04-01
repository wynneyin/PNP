from dataclasses import dataclass
import gmpy2
from .bls12_381 import fr
import math
import torch
import torch.nn.functional as F

def pow_1(self,exp):
    res = torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont)
    for i in range(63,-1,-1):
        #modsquare
        res = F.mul_mod(res,res)
        if ((exp >> i) & 1) == 1:
            # res = res.mul(self)
            res = F.mul_mod(res,self)
    return res

def neg(self):
    a=torch.tensor([18446744069414584321, 6034159408538082302, 3691218898639771653, 8353516859464449352],dtype=torch.BLS12_381_Fr_G1_Mont)
    if torch.equal(self,torch.tensor([0,0,0,0],dtype=torch.BLS12_381_Fr_G1_Mont)):
        return self
    else:
        res= F.sub_mod(a,self)
        return res

@dataclass
class Radix2EvaluationDomain:
    size: int
    log_size_of_group: int
    size_as_field_element: fr.Fr
    size_inv: fr.Fr
    group_gen: fr.Fr
    group_gen_inv: fr.Fr
    generator_inv: fr.Fr

    @classmethod
    def new_1(cls, num_coeffs: int, params:fr.Fr):
        # Compute the size of our evaluation domain
        size = num_coeffs if num_coeffs & (num_coeffs - 1) == 0 else 2 ** num_coeffs.bit_length()
        log_size_of_group = size.bit_length()-1
        
        # Check if log_size_of_group exceeds TWO_ADICITY
        if log_size_of_group > params.TWO_ADICITY:
            return None

        # Compute the generator for the multiplicative subgroup.
        # It should be the 2^(log_size_of_group) root of unity.
        group_gen = fr.Fr(value=params.get_root_of_unity(size))
        
        # Check that it is indeed the 2^(log_size_of_group) root of unity.
        group_gen_pow = group_gen.pow(size)
        assert torch.equal(group_gen_pow,params.one())

        size_as_field_element=fr.Fr.from_repr(size)
        size_inv = fr.Fr.inverse(size_as_field_element)
        group_gen_inv = fr.Fr.inverse(group_gen)
        generator_inv = fr.Fr.inverse(params.multiplicative_generator())

        return cls(size, log_size_of_group, size_as_field_element, size_inv, group_gen, group_gen_inv, generator_inv)
    
    @classmethod
    def new(cls,num_coeffs:int,params:fr.Fr):
        # Compute the size of our evaluation domain
        size = num_coeffs if num_coeffs & (num_coeffs - 1) == 0 else 2 ** num_coeffs.bit_length()
        log_size_of_group = size.bit_length()-1
        
        # Check if log_size_of_group exceeds TWO_ADICITY
        if log_size_of_group > params.TWO_ADICITY:
            return None

        # Compute the generator for the multiplicative subgroup.
        # It should be the 2^(log_size_of_group) root of unity.
        group_gen = fr.Fr(value=params.get_root_of_unity(size))
        
        # Check that it is indeed the 2^(log_size_of_group) root of unity.
        group_gen_pow = group_gen.pow(size)
        assert group_gen_pow == params.one()

        size_as_field_element=fr.Fr.from_repr(size)
        size_inv = fr.Fr.inverse(size_as_field_element)
        group_gen_inv = fr.Fr.inverse(group_gen)
        generator_inv = fr.Fr.inverse(params.multiplicative_generator())

        return cls(size, log_size_of_group, size_as_field_element, size_inv, group_gen, group_gen_inv, generator_inv)
    
    # Evaluate all Lagrange polynomials at tau to get the lagrange coefficients.
    # Define the following as
    # - H: The coset we are in, with generator g and offset h
    # - m: The size of the coset H
    # - Z_H: The vanishing polynomial for H. Z_H(x) = prod_{i in m} (x - hg^i) = x^m - h^m
    # - v_i: A sequence of values, where v_0 = 1/(m * h^(m-1)), and v_{i + 1} = g * v_i
    #
    # We then compute L_{i,H}(tau) as `L_{i,H}(tau) = Z_H(tau) * v_i / (tau - h g^i)`
    #
    # However, if tau in H, both the numerator and denominator equal 0
    # when i corresponds to the value tau equals, and the coefficient is 0 everywhere else.
    # We handle this case separately, and we can easily detect by checking if the vanishing poly is 0.
    def evaluate_all_lagrange_coefficients(self, tau: fr.Fr):
        from .arithmetic import from_gmpy_list_1
        size = self.size
        t_size = pow_1(tau,size)
        domain_offset = torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont)
        one = torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont)
        self.group_gen_inv=torch.tensor(from_gmpy_list_1(self.group_gen_inv),dtype=torch.BLS12_381_Fr_G1_Mont)
        self.group_gen=torch.tensor(from_gmpy_list_1(self.group_gen),dtype=torch.BLS12_381_Fr_G1_Mont)
        z_h_at_tau = F.sub_mod(t_size, domain_offset)
        zero = torch.tensor([0,0,0,0],dtype=torch.BLS12_381_Fr_G1_Mont)

        if torch.equal(z_h_at_tau,zero):
            u = [zero for _ in range(size)]
            omega_i = domain_offset
            for i in range(size):
                if omega_i == tau:
                    u[i] = one
                    break
                omega_i = F.mul_mod(omega_i, self.group_gen)
            return u
        else:
            # In this case we have to compute `Z_H(tau) * v_i / (tau - h g^i)`
            # for i in 0..size
            # We actually compute this by computing (Z_H(tau) * v_i)^{-1} * (tau - h g^i)
            # and then batch inverting to get the correct lagrange coefficients.
            # We let `l_i = (Z_H(tau) * v_i)^-1` and `r_i = tau - h g^i`
            # Notice that since Z_H(tau) is i-independent,
            # and v_i = g * v_{i-1}, it follows that
            # l_i = g^-1 * l_{i-1}
            # TODO: consider caching the computation of l_i to save N multiplications
            from .arithmetic import batch_inversion,neg

            # v_0_inv = m * h^(m-1)
            f_size = fr.Fr.from_repr(size)
            f_size=torch.tensor(from_gmpy_list_1(f_size),dtype=torch.BLS12_381_Fr_G1_Mont)
            pow_dof = pow_1(domain_offset, size - 1)
            v_0_inv = F.mul_mod(f_size, pow_dof)
            z_h_at_tau_inv= F.div_mod(torch.tensor(from_gmpy_list_1(fr.Fr(value=gmpy2.mpz(10920338887063814464675503992315976177888879664585288394250266608035967270910))),dtype=torch.BLS12_381_Fr_G1_Mont),z_h_at_tau)
            l_i = F.mul_mod(z_h_at_tau_inv, v_0_inv)
            negative_cur_elem = neg(domain_offset)
            lagrange_coefficients_inverse = [zero for _ in range(size)]
            for i in range(size):
                r_i = F.add_mod(tau, negative_cur_elem)
                lagrange_coefficients_inverse[i] = F.mul_mod(l_i, r_i)
                # Increment l_i and negative_cur_elem
                l_i = F.mul_mod(l_i, self.group_gen_inv)
                negative_cur_elem = F.mul_mod(negative_cur_elem, self.group_gen)
            batch_inversion(lagrange_coefficients_inverse)
            return lagrange_coefficients_inverse

        
    # def evaluate_all_lagrange_coefficients(self, tau: field):
    #     size = self.size
    #     t_size = tau.pow(size)
    #     one = tau.one()
    #     zero = field.zero(tau.params)

    #     if t_size.value == one.value:
    #         u = [zero for _ in range(size)]
    #         omega_i = one
    #         for i in range(size):
    #             if omega_i.value == tau.value:
    #                 u[i] = one
    #                 break
    #             omega_i =omega_i.mul(self.group_gen)
    #         return u
    #     else:
    #         from arithmetic import batch_inversion
    #         t_size_sub_one = t_size.sub(one)
    #         l = t_size_sub_one.mul(self.size_inv)
    #         r = one
    #         u = [zero] * size
    #         ls = [zero] * size
    #         for i in range(size):
    #             u[i] = tau.sub(r)
    #             ls[i] = l
    #             l = l.mul(self.group_gen)
    #             r = r.mul(self.group_gen)

    #         batch_inversion(u)

    #         for i in range(size):
    #             u[i] = ls[i].mul(u[i])

    #         return u
    
    # This evaluates the vanishing polynomial for this domain at tau.
    # For multiplicative subgroups, this polynomial is `z(X) = X^self.size - 1`.
    def evaluate_vanishing_polynomial(self, tau: fr.Fr):
        
        def pow_1(self,exp):
            res = torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont)
            for i in range(63,-1,-1):
                #modsquare
                res = F.mul_mod(res,res)
                if ((exp >> i) & 1) == 1:
                    # res = res.mul(self)
                    res = F.mul_mod(res,self)
            return res
        one =torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont)
        pow_tau=pow_1(tau,self.size)
        return F.sub_mod(pow_tau,one)
    
    # Returns the `i`-th element of the domain, where elements are ordered by
    # their power of the generator which they correspond to.
    # e.g. the `i`-th element is g^i
    def element(self, i):
        # TODO: Consider precomputed exponentiation tables if we need this to be faster.
        res = self.group_gen.pow(i)
        return res
    
        