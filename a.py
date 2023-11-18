import torch
import torch.nn.functional as F

print(torch.__path__)

def split_into_64_bits(integer):
    # 将整数转换为二进制字符串
    binary_representation = bin(integer)[2:]

    # 补零，使其长度为64的倍数
    binary_representation = '0' * (64 - len(binary_representation) % 64) + binary_representation

    # 将二进制字符串拆分成64比特块
    blocks = [binary_representation[i:i+64] for i in range(0, len(binary_representation), 64)]

    # 将每个64比特块转换回十六进制
    hex_blocks = ['TO_CUDA_T('+'0x'+ format(int(block, 2),'016x') + ')' for block in blocks[::-1]]

    return hex_blocks

def extended_gcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, x, y = extended_gcd(b % a, a)
        return (g, y - (b // a) * x, x)
    
def modinv(a, m):
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise Exception('Modular inverse does not exist')
    else:
        return x % m
    
# 获取用户输入的大整数
user_input = int(0x01C4C62D92C41110229022EEE2CDADB7F997505B8FAFED5EB7E8F96C97D87307FDB925E8A0ED8D99D124D9A15AF79DB26C5C28C859A99B3EEBCA9429212636B9DFF97634993AA4D6C381BC3F0057974EA099170FA13A4FD90776E240000001)
user_input = user_input << 15

# 将大整数拆分成64比特数的组合，并转换为十六进制
result = split_into_64_bits(user_input)

# 打印结果
print("拆分成64比特数的十六进制组合:", result)




# x = torch.tensor([[9223372036854772, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.BLS12_377_Fr_G1_Base)
# xq = torch.tensor([[9223372036854772, 2, 3, 4, 5, 6], [1, 2, 5, 6, 7, 8]], dtype=torch.BLS12_377_Fq_G1_Base)
# # x.to("cuda")
# print("===========")
# print(x)
# y = F.to_mont(x)
# print(y)
# z = F.to_base(y)
# print(z)

# a = y.clone()
# print(a)




# y = torch.tensor([[9223372036854772, 2, 3, 10], [4, 5, 6, 8], [4, 5, 6, 8]], dtype=torch.big_integer)


# y = torch.tensor([
#     [[922337203685477, 2, 3, 10], [4, 5, 6, 8], [4, 5, 6, 8]],
#     [[922337203685477, 2, 3, 10], [4, 5, 6, 8], [4, 5, 6, 8]]
# ], dtype=torch.big_integer)

# print(x)
# print(y)

# print(type(x.type()))

# # x.to_Fq(torch.uint192)


# # x.to("CUDA")

# print(x.shape)
# print(y.shape)

# print("===========")
# print(x)
# y = F.to_mont(x)
# print(y)

# # # x = torch.tensor([[9223372036854775809, 2, 3], [4, 5, 6]], dtype=torch.uint64)
# # # y = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.uint64)
# # # z = x

# # # x = torch.tensor([[922337203685477, 2, 3], [4, 5, 6]], dtype=torch.float8_e5m2)
# y = torch.tensor([[922337203685477, 2, 3, 10], [4, 5, 6, 8]], dtype=torch.big_integer)
# # x = torch.tensor([[922337203685477, 2, 3], [4, 5, 6]], dtype=torch.field64)

# print(y)

# # z = torch.add(x, y, alpha=2)

# # print(z) 

# # m = torch.nn.MyFUNC(inplace=True)



# # x = m(x)
# # print(x.cpu())


# # x = x.cuda()
# # x = m(x) #now in GPU
# # print(x.cpu())

# # c = x.tolist()

# # d = torch.tensor(c, dtype=torch.uint64)

# # print(d)



