import torch
import random
import torch.nn.functional as F

random_list = [[random.randint(1, 1000) for _ in range(4)] for _ in range(1024)]
x = torch.tensor(random_list, dtype=torch.BLS12_381_Fr_G1_Mont)
x_cpu = torch.tensor(random_list,dtype=torch.uint64)
#x = torch.tensor([[9223372036854772, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.BLS12_381_Fr_G1_Base)

x_gpu = x.to("cuda")
# input = F.to_mont(x_gpu)
# y = y_gpu.to("cpu")
# y_list = y.tolist()
# print(y_list==random_list) 

output_cpu = torch.intt_zkp(x_cpu)
z1 = output_cpu.tolist()

output = torch.intt_zkp(x_gpu)
z2 = output.to("cpu")
z2 = z2.tolist()
print(random_list[:5])
print(z2[:5])
# z_gpu = F.to_base(y_gpu)
# z = z_gpu.to("cpu")
# z_list = z.tolist()
# print(z_list==random_list) 
print(z1==z2)

