import gmpy2
import copy
from .bls12_381 import fr,fq
from .domain import Radix2EvaluationDomain
from .structure import AffinePointG1
from .jacobian import ProjectivePointG1
import math
import random
import torch
import torch.nn.functional as F
import time
from .ele import into_repr
def pow_1(self,exp):
            
        def square_1(self):
            self = F.mul_mod(self)
            # self = self.mul(self)
            return self
        res = self.clone()
        for i in range(63,-1,-1):
            #modsquare
            res = res.square_1()
            if ((exp >> i) & 1) == 1:
                # res = res.mul(self)
                res = F.mul_mod(self)
        return res

def calculate_execution_time(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()  # 记录函数开始执行的时间
            result = func(*args, **kwargs)
            end_time = time.time()    # 记录函数执行结束的时间
            execution_time = end_time - start_time  # 计算函数执行时间
            print(f"函数 {func.__name__} 执行时间为：{execution_time} 秒")
            return result
        return wrapper

def neg(self):
    a=torch.tensor([18446744069414584321, 6034159408538082302, 3691218898639771653, 8353516859464449352],dtype=torch.BLS12_381_Fr_G1_Mont)
    if torch.equal(self,torch.tensor([0,0,0,0],dtype=torch.BLS12_381_Fr_G1_Mont)):
        return self
    else:
        res= F.sub_mod(a,self)
        return res

def from_list_tensor(input:list, dtype=torch.BLS12_381_Fr_G1_Mont):
    base_input=[]
    for i in range(len(input)):
        # print(input[i].value)
        base_input.append(input[i].value)
    output = torch.tensor(base_input,dtype = dtype,device='cpu')
    return output

def from_tensor_list(input:torch.Tensor):
    output = input.tolist()
    # print("output 值为",output)
    for i in range(len(input)):
        output[i]=fr.Fr(value=output[i])
    return output


def from_list_gmpy(input:list):
    for i in range(len(input)):
        output = 0 
        for j in reversed(input[i].value):
            output = output<<64
            output = output | j
        input[i] =  fr.Fr(value = gmpy2.mpz(output))

def from_gmpy_list(input:list): ##inplace
    if(len(input)==0):
        input=torch.tensor([],dtype=torch.uint64)
    for i in range(len(input)):
        output = []
        for j in range(fr.Fr.Limbs):
            output.append( int(input[i].value & 0xFFFFFFFFFFFFFFFF) )
            input[i].value = input[i].value >> 64
        input[i] = fr.Fr(value = output)

def from_gmpy_list_1(input:fr.Fr):
    output = []
    input_copy=copy.deepcopy(input)
    for j in range(fr.Fr.Limbs):
        output.append( int(input_copy.value & 0xFFFFFFFFFFFFFFFF) )
        input_copy.value = input_copy.value >> 64
    # input=fr.Fr(value=output)
    return output

def from_list_gmpy_1(input:list):
        output = 0 
        for j in reversed(input):
            output = output<<64
            output = output | j
        return fr.Fr(value = gmpy2.mpz(output))

def domian_trans_tensor(domain_ele):
        a=copy.deepcopy(domain_ele)
        temp=from_gmpy_list_1(a)
        xx = torch.tensor(temp, dtype=torch.BLS12_381_Fr_G1_Mont)
        # domain_ele.value=xx
        return xx


def pow_1(self,exp):
    res = torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont)
    for i in range(63,-1,-1):
        #modsquare
        res = F.mul_mod(res,res)
        if ((exp >> i) & 1) == 1:
            # res = res.mul(self)
            res = F.mul_mod(res,self)
    return res

def reverse_bits(operand, bit_count):
    acc = 0
    for i in range(bit_count):
        acc = (acc << 1) | ((operand >> i) & 1)
    return acc

def derange(xi, log_len):
    for idx in range(1, len(xi) - 1):
        ridx = reverse_bits(idx, log_len)
        if idx < ridx:
            xi[idx], xi[ridx] = xi[ridx], xi[idx]
    return xi

# def precompute_twiddles(domain:Radix2EvaluationDomain, root:fr.Fr):
#     log_size = int(math.log2(domain.size))
#     # powers = [root.zero()] * (1 << (log_size - 1))
#     # powers[0] = root.one()
#     powers=torch.zeros(1 << (log_size - 1),4,dtype=torch.BLS12_381_Fr_G1_Mont)
#     powers[0]=torch.tensor(from_gmpy_list_1(root.one()),dtype=torch.BLS12_381_Fr_G1_Mont)

#     for idx in range(1, len(powers)):
#         # powers[idx] = powers[idx - 1].mul(root.value)
#         powers[idx]=F.mul_mod(powers[idx - 1],root.value)
#     return powers

def precompute_twiddles(domain:Radix2EvaluationDomain, root:fr.Fr):
    log_size = int(math.log2(domain.size))
    powers = [root.zero()] * (1 << (log_size - 1))
    powers[0] = root.one()
    for idx in range(1, len(powers)):
        powers[idx] = powers[idx - 1].mul(root)
    return powers

# def  operator(domain:Radix2EvaluationDomain, xi:torch.tensor, root:fr.Fr):
#     log_size = int(math.log2(domain.size))
#     xi=from_tensor_list(xi)
#     from_list_gmpy(xi)
#     xi = derange(xi,log_size)
#     twiddles=precompute_twiddles(domain,root)
#     chunk = 2
#     twiddle_chunk = domain.size // 2

#     twiddles=from_tensor_list(twiddles)
#     from_list_gmpy(twiddles)
#     for i in range(log_size):
#         for j in range(0, domain.size, chunk):
#             t = xi[j + chunk // 2]       # Copy Right[0]
#             xi[j + chunk // 2] = xi[j]   # Right[0] = Left[0]
#             xi[j] = xi[j].add(t)
#             xi[j + chunk // 2] = xi[j + chunk // 2].sub(t)
#             for m in range(chunk // 2 - 1):
#                 twiddle = twiddles[(m + 1) * twiddle_chunk]
#                 t1 = xi[j + chunk // 2 + m + 1]
#                 t1 = t1.mul(twiddle)
#                 xi[j + chunk // 2 + m + 1] = xi[j + m + 1]
#                 xi[j + m + 1] = xi[j + m + 1].add(t1)  # a + b * w
#                 # a - b * w
#                 xi[j + chunk // 2 + m + 1] = xi[j + chunk // 2 + m + 1].sub(t1)  
#         chunk *= 2  # Merge up
#         twiddle_chunk //= 2

#     # for i in range(log_size):
#     #     for j in range(0, domain.size, chunk):
#     #         t = xi[j + chunk // 2]       # Copy Right[0]
#     #         xi[j + chunk // 2] = xi[j]   # Right[0] = Left[0]
#     #         xi[j] = F.add_mod(xi[j].clone(), t.clone())
#     #         xi[j + chunk // 2] = F.sub_mod(xi[j + chunk // 2].clone(), t.clone())
#     #         for m in range(chunk // 2 - 1):
#     #             twiddle = twiddles[(m + 1) * twiddle_chunk]
#     #             t1 = xi[j + chunk // 2 + m + 1]
#     #             t1 = F.mul_mod(t1.clone(), twiddle.clone())
#     #             xi[j + chunk // 2 + m + 1] = xi[j + m + 1]
#     #             xi[j + m + 1] = F.add_mod(xi[j + m + 1].clone(), t1.clone())
#     #             xi[j + chunk // 2 + m + 1] = F.sub_mod(xi[j + chunk // 2 + m + 1].clone(), t1.clone())
#     #     chunk *= 2  # Merge up
#     #     twiddle_chunk //= 2
#     from_gmpy_list(xi)
#     xi=from_list_tensor(xi)
#     return xi

def operator(domain:Radix2EvaluationDomain, xi:list[fr.Fr], root:fr.Fr):
    log_size = int(math.log2(domain.size))
    xi = derange(xi,log_size)
    twiddles=precompute_twiddles(domain,root)
    chunk = 2
    twiddle_chunk = domain.size // 2
    for i in range(log_size):
        for j in range(0, domain.size, chunk):
            t = xi[j + chunk // 2]       # Copy Right[0]
            xi[j + chunk // 2] = xi[j]   # Right[0] = Left[0]
            xi[j] = xi[j].add(t)
            xi[j + chunk // 2] = xi[j + chunk // 2].sub(t)
            for m in range(chunk // 2 - 1):
                twiddle = twiddles[(m + 1) * twiddle_chunk]
                t1 = xi[j + chunk // 2 + m + 1]
                t1 = t1.mul(twiddle)
                xi[j + chunk // 2 + m + 1] = xi[j + m + 1]
                xi[j + m + 1] = xi[j + m + 1].add(t1)  # a + b * w
                # a - b * w
                xi[j + chunk // 2 + m + 1] = xi[j + chunk // 2 + m + 1].sub(t1)  
        chunk *= 2  # Merge up
        twiddle_chunk //= 2
    return xi


# def resize(self, target_len, padding):
#     res = self[:]
#     if len(self) < target_len:
#         num_to_pad = target_len - len(self)
#         res.extend([padding for _ in range(num_to_pad)])
#     return res

def resize_1(self, target_len):
    # res = copy.deepcopy(self)
    # self=from_tensor_list(self)
    # from_list_gmpy(self)
    if isinstance(self, list):
        res = copy.deepcopy(self)
    else:
        res = self.clone()
    
    if len(self) < target_len:
        output=torch.empty(target_len,4,dtype=torch.BLS12_381_Fr_G1_Mont)
        # num_to_pad = target_len - len(self)
        
        output[:len(self)]=res[:]
        # res.extend([padding for _ in range(num_to_pad)])
        return output 
    else :
        return res
def resize(self, target_len, padding):
    res = copy.deepcopy(self)
    if len(self) < target_len:
        num_to_pad = target_len - len(self)
        res.extend([padding for _ in range(num_to_pad)])
    return res


def distribute_powers(coeffs:list[fr.Fr], g):
    g_field = fr.Fr(value = g)
    one = g_field.one()
    distribute_powers_and_mul_by_const(coeffs, g_field, one)

# Multiply the `i`-th element of `coeffs` with `c*g^i`.
def distribute_powers_and_mul_by_const(coeffs, g, c):
    pow = c
    for i in range(len(coeffs)):
        coeffs[i] = coeffs[i].mul(pow)
        pow = pow.mul(g)
def degree(poly):
    if len(poly)==0:
        return 0
    else:
        assert not poly[-1]==0
        return len(poly) - 1

def convert_to_bigints_new(p: list[torch.tensor]):
    # coeffs = [fr.Fr(value = s.into_repr()) for s in p]
    # coeffs = [torch.tensor(into_repr(s)) for s in p]
    if len(p)==0:
        return torch.tensor([],dtype=torch.BLS12_381_Fr_G1_Mont)
    else:
            
        coeffs =torch.zeros(len(p),4,dtype=torch.BLS12_381_Fr_G1_Mont)
        i=0
        for s in p:
            coeffs[i]=into_repr(s)
            i+=1
        return coeffs
    
def convert_to_bigints(p: list[fr.Fr]):
    coeffs = [fr.Fr(value = s.into_repr()) for s in p]
    return coeffs


def skip_leading_zeros_and_convert_to_bigints(p: list[fr.Fr]):
    num_leading_zeros = 0
    while num_leading_zeros < len(p) and p[num_leading_zeros] == 0:
        num_leading_zeros += 1

    coeffs = convert_to_bigints(p[num_leading_zeros:])
    return num_leading_zeros, coeffs

def skip_leading_zeros_and_convert_to_bigints_new(p:list[torch.tensor]):
    num_leading_zeros = 0
    # while num_leading_zeros < len(p) and p[num_leading_zeros] == torch.tensor([0,0,0,0],dtype=torch.BLS12_381_Fr_G1_Mont):
    #TODO 等于号
    while num_leading_zeros < len(p) and torch.equal(p[num_leading_zeros] ,torch.zeros(1,4,dtype=torch.BLS12_381_Fr_G1_Mont)):
        num_leading_zeros += 1

    coeffs = convert_to_bigints_new(p[num_leading_zeros:])  
    return num_leading_zeros, coeffs



def NTT_1(domain,coeffs:torch.Tensor):
    # coeffs_value=from_tensor_list(coeffs)
    # from_list_gmpy(coeffs_value)

    # zero = fr.Fr.zero()
    #add zero to resize
    resize_coeffs_value = resize(coeffs,domain.size)
    # resize_coeffs_value=from_tensor_list(resize_coeffs_value)
    evals = operator(domain,resize_coeffs_value,domain.group_gen)

    # from_gmpy_list(evals)
    # output=from_list_tensor(evals,dtype=torch.BLS12_381_Fr_G1_Mont)

    return evals

def NTT(domain,coeffs:torch.Tensor):
    coeffs_value=from_tensor_list(coeffs)
    from_list_gmpy(coeffs_value)

    zero = fr.Fr.zero()
    #add zero to resize
    resize_coeffs_value = resize(coeffs_value,domain.size,zero)
    # from_gmpy_list(resize_coeffs_value)
    evals = operator(domain,resize_coeffs_value,domain.group_gen)

    from_gmpy_list(evals)
    output=from_list_tensor(evals,dtype=torch.BLS12_381_Fr_G1_Mont)

    return output

def INTT_1(domain,evals:torch.Tensor):
    # temp=from_tensor_list(evals)
    # from_list_gmpy(temp)

    #add zero to resize
    # zero = fr.Fr.zero()
    # b=from_tensor_list(evals)
    a=evals.clone()
    a=from_tensor_list(a)
    from_list_gmpy(a)
    resize_value = resize(evals,domain.size)

    evals = operator(domain,resize_value,domain.group_gen_inv)
    for i in range(len(evals)):
        # temp[i] = temp[i].mul(domain.size_inv)
        evals[i] = F.mul_mod( evals[i] ,domain.size_inv.value)

    # from_gmpy_list(evals)
    # output=from_list_tensor(evals,dtype=torch.BLS12_381_Fr_G1_Mont)

    return evals

def INTT(domain,evals:torch.Tensor):
    temp=from_tensor_list(evals)
    from_list_gmpy(temp)

    #add zero to resize
    zero = fr.Fr.zero()
    resize_value = resize(temp,domain.size,zero)
    # a=from_gmpy_list_1(domain.size_inv)
    temp = operator(domain,resize_value,domain.group_gen_inv)
    temp1=copy.deepcopy(temp)
    for i in range(len(temp)):
        temp[i] = temp[i].mul(domain.size_inv)

    from_gmpy_list(temp)
    output=from_list_tensor(temp,dtype=torch.BLS12_381_Fr_G1_Mont)

    return output
# Compute a NTT over a coset of the domain, modifying the input vector in place.
def coset_NTT(coeffs:torch.tensor, domain):

    coeffs=from_tensor_list(coeffs)
    from_list_gmpy(coeffs)

    modified_coeffs = coeffs[:]
    distribute_powers(modified_coeffs, fr.Fr.GENERATOR)
    from_gmpy_list(modified_coeffs)
    modified_coeffs=from_list_tensor(modified_coeffs)
    evals = NTT(domain,modified_coeffs)
    return evals

# Compute a INTT over a coset of the domain, modifying the input vector in place.
def coset_INTT(evals:torch.tensor, domain):
    evals=from_tensor_list(evals)
    from_list_gmpy(evals)
    #add zero to resize
    zero = fr.Fr.zero()
    resize_evals = resize(evals,domain.size,zero)
    evals = operator(domain,resize_evals,domain.group_gen_inv)
    distribute_powers_and_mul_by_const(evals, domain.generator_inv,domain.size_inv)
    from_gmpy_list(evals)
    evals=from_list_tensor(evals)
    return evals

def from_coeff_vec(coeffs:torch.Tensor):
    coeffs=from_tensor_list(coeffs)  

    result = coeffs[:]
    # print(result[-1])
    #TODO 缺少等于
    while result and result[-1]== 0:
        result.pop()
    
    output=from_list_tensor(result,dtype=torch.BLS12_381_Fr_G1_Mont)
    return output

def from_coeff_vec_list(coeffs:list):
    temp=[]
    for i in range(0,1024):
        temp.append(from_tensor_list(coeffs[i]) )

    result = temp[:]
    # print(result[-1])
    while result and result[-1]== 0:
        result.pop()
    
    output=[]
    for i in range(0,1024):
        output.append(from_list_tensor(result[i]) )
    return output

def poly_add_poly(self: torch.tensor, other: torch.tensor):    #input tensor output tensor
    if len(self) == 0:
        res = other[:]
        return res
    if len(other) == 0:
        res =self[:]
        return res
    elif len(self) >= len(other):
        result=self.clone()
        for i in range(len(other)):
            result[i] = F.add_mod(result[i],other[i])

        result = from_coeff_vec(result)
        return result
    else:
        result=other.clone()
        for i in range(len(self)):
            result[i] = F.add_mod(result[i],self[i])

        result = from_coeff_vec(result)
        return result

def poly_mul_const(poly:torch.tensor,elem:torch.tensor):  ###因为输入poly可能会是空，这样无法转为tensor
    if len(poly) == 0 or torch.equal(elem,torch.zeros(1,4,dtype=torch.BLS12_381_Fr_G1_Mont)):
        return poly
    else:
        result =poly.clone()
        for i in range(len(result)):
            result[i] = F.mul_mod(result[i],elem)
        return result
    
def divide_with_q_and_r(self: list[fr.Fr], divisor: list[fr.Fr]):
    if len(self) == 0:
        return self
    elif len(divisor) == 0:
        raise ValueError("Dividing by zero polynomial")
    elif len(self) < len(divisor):
        zero = fr.Fr.zero()
        return [zero]
    else:
        zero = fr.Fr.zero()
        quotient = [zero for _ in range(len(self) - len(divisor) + 1)]
        remainder = self[:]
        divisor_leading = divisor[-1]
        divisor_leading_inv = fr.Fr.inverse(divisor_leading)
        while len(remainder) != 0 and len(remainder) >= len(divisor):
            remainder_leading = remainder[-1]
            cur_q_coeff = remainder_leading.mul(divisor_leading_inv)
            cur_q_degree = len(remainder) - len(divisor)
            quotient[cur_q_degree] = cur_q_coeff
            for i, div_coeff in enumerate(divisor):
                temp = cur_q_coeff.mul(div_coeff)
                remainder[cur_q_degree + i] = remainder[cur_q_degree + i].sub(temp)
            while remainder[-1] and remainder[-1].value == 0:
                remainder.pop()
        from_gmpy_list(quotient)
        quotient=from_list_tensor(quotient)
        res_quotient = from_coeff_vec(quotient)
        return res_quotient, remainder
            
def poly_div_poly(self: list[fr.Fr], divisor: list[fr.Fr]):
        res, remainder = divide_with_q_and_r(self,divisor)
        return res

def rand_poly(d):
    random.seed(42)
    random_coeffs = [fr.Fr.from_repr(random.random) for _ in range(d + 1)]
    return from_coeff_vec(random_coeffs)

def ln_without_floats(a):
    # log2(a) * ln(2)
    return int(math.log2(a) * 69 / 100)

# Evaluates `self` at the given `point` in `Self::Point`.
def evaluate(self, point: fr.Fr):
    zero =torch.tensor([0,0,0,0],dtype=torch.BLS12_381_Fr_G1_Mont)
    if len(self) == 0:
        return zero
    elif torch.equal(point,zero):
        return self[0]
    return horner_evaluate(self, point)

# Horner's method for polynomial evaluation
def horner_evaluate(poly_coeffs: list, point: fr.Fr):

    # from_list_gmpy(poly_coeffs)
    result =torch.tensor([0,0,0,0],dtype=torch.BLS12_381_Fr_G1_Mont)
    for coeff in reversed(poly_coeffs):
        result = F.mul_mod(result, point)
        result = F.add_mod(result, coeff)
    return result

def poly_add_poly_mul_const(self: list[fr.Fr], f: fr.Fr, other: list[fr.Fr]):
    if len(self) == 0:
        self = other[:]
        for i in range(len(self)):
            self[i] = self[i].mul(f)
        return self
    elif len(other) == 0:
        return self
    elif len(self) >= len(other):
        pass
    else:
        zero = fr.Fr.zero()
        self = resize(self,len(other),zero)
    for i in range(len(other)):
        temp = f.mul(other[i])
        self[i] = self[i].add(temp)

    res=copy.deepcopy(self)
    from_gmpy_list(res)
    res=from_list_tensor(res)

    res = from_coeff_vec(res)

    res=from_tensor_list(res)
    from_list_gmpy(res)
    return res

# Given a vector of field elements {v_i}, compute the vector {coeff * v_i^(-1)}
# This method is explicitly single core.
def serial_batch_inversion_and_mul(v: list[fr.Fr], coeff: fr.Fr):
    prod =torch.zeros(len(v),4,dtype=torch.BLS12_381_Fr_G1_Mont)
    tmp = torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont)

    i=0
    ###TODO 应当去0
    for f in v:
        if not torch.equal(f,torch.zeros(1,4,dtype=torch.BLS12_381_Fr_G1_Mont)):
            tmp = F.mul_mod(tmp,f)
            prod[i]=tmp
            i=i+1

    # Invert `tmp`.
    tmp= F.div_mod(torch.tensor(from_gmpy_list_1(fr.Fr(value=gmpy2.mpz(10920338887063814464675503992315976177888879664585288394250266608035967270910))),dtype=torch.BLS12_381_Fr_G1_Mont),tmp)
    # Multiply product by coeff, so all inverses will be scaled by coeff
    tmp = F.mul_mod(tmp,coeff)
    rev_prod=torch.zeros(len(prod),4,dtype=torch.BLS12_381_Fr_G1_Mont)
    for i in range(0,len(prod)):
        rev_prod[i]=prod[len(prod)-2-i]
    rev_prod[len(prod)-1]=torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont)
    
    aa=rev_prod.tolist()
    # Backwards, skip last element, fill in one for last term.
    for i,(f, s) in enumerate(zip(reversed(v), rev_prod)):
        if not torch.equal(f,torch.zeros(1,4,dtype=torch.BLS12_381_Fr_G1_Mont)):
            # tmp := tmp * f; f := tmp * s = 1/f
            new_tmp=F.mul_mod(tmp,f)
            f= F.mul_mod(tmp,s)
            tmp = new_tmp
            v[len(v) - 1 - i] = f  # Update the value of v with the new result

# Given a vector of field elements {v_i}, compute the vector {coeff * v_i^(-1)}
def batch_inversion_and_mul(v: list[fr.Fr], coeff: fr.Fr):
    serial_batch_inversion_and_mul(v, coeff)

# Given a vector of field elements {v_i}, compute the vector {v_i^(-1)}
def batch_inversion(v: list[fr.Fr]):
    one = torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont)
    batch_inversion_and_mul(v, one)

# The first lagrange polynomial has the expression:
# L_0(X) = mul_from_1_to_(n-1) [(X - omega^i) / (1 - omega^i)]
#
# with `omega` being the generator of the domain (the `n`th root of unity).
#
# We use two equalities:
#   1. `mul_from_2_to_(n-1) [1 / (1 - omega^i)] = 1 / n`
#   2. `mul_from_2_to_(n-1) [(X - omega^i)] = (X^n - 1) / (X - 1)`
# to obtain the expression:
# L_0(X) = (X^n - 1) / n * (X - 1)
def compute_first_lagrange_evaluation(
    domain: Radix2EvaluationDomain, z_h_eval: fr.Fr, z_challenge: fr.Fr):
    one = torch.tensor([[8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911]],dtype=torch.BLS12_381_Fr_G1_Mont)
    n_fr = fr.Fr.from_repr(domain.size)
    n_fr =torch.tensor(from_gmpy_list_1(n_fr),dtype=torch.BLS12_381_Fr_G1_Mont)
    z_challenge_sub_one =F.sub_mod(z_challenge,one)
    denom=F.mul_mod(n_fr,z_challenge_sub_one)
    denom_in =  F.div_mod(torch.tensor(from_gmpy_list_1(fr.Fr(value=gmpy2.mpz(10920338887063814464675503992315976177888879664585288394250266608035967270910))),dtype=torch.BLS12_381_Fr_G1_Mont),denom)
    res= F.mul_mod(z_h_eval,denom_in)
    return res  

def MSM(bases:list[AffinePointG1], scalars:list[fr.Fr], params):
    size = min(len(bases), len(scalars))
    fq_elem = bases[0].x
    scalars = scalars[:size]
    bases = bases[:size]
    scalars_and_bases_iter = [(s, b) for s, b in zip(scalars, bases) if not s==0]

    c = 3 if size < 32 else ln_without_floats(size) + 2
    num_bits = fr.Fr.MODULUS_BITS
    fr_one = params.one()
    fr_one = fr_one.into_repr()

    zero:ProjectivePointG1 = ProjectivePointG1.zero(fq_elem)
    window_starts = list(range(0, num_bits, c))

    window_sums = []
    for w_start in window_starts:
        res = zero
        buckets = [zero for _ in range((1 << c) - 1)]

        for org_scalar, org_base in scalars_and_bases_iter:
            scalar = copy.copy(org_scalar)
            base = copy.copy(org_base)
            if scalar.value == fr_one:
                if w_start == 0:
                    res = res.add_assign_mixed(base)
            else:
                # We right-shift by w_start, thus getting rid of the lower bits
                scalar.value >>= w_start
                # We mod the remaining bits by 2^{window size}, thus taking `c` bits.
                scalar.value %= (1 << c)
                if scalar.value != 0:
                    buckets[scalar.value - 1] = buckets[scalar.value - 1].add_assign_mixed(base)

        running_sum:ProjectivePointG1 = ProjectivePointG1.zero(fq_elem)
        for b in reversed(buckets):
            running_sum = running_sum.add_assign(b)
            res = res.add_assign(running_sum)

        window_sums.append(res)

    lowest = window_sums[0]

    total:ProjectivePointG1 = zero
    for sum_i in reversed(window_sums[1:]):
        total = total.add_assign(sum_i)
        for _ in range(c):
            total = total.double()
    total = lowest.add_assign(total)
    return total
