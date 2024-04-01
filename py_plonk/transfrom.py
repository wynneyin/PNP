import torch
def from_list_tensor(input:list, dtype=torch.BLS12_381_Fr_G1_Mont):
    base_input=[]
    for i in range(len(input)):
        # print(input[i].value)
        base_input.append(input[i].value)
    output = torch.tensor(base_input,dtype = dtype,device='cpu')
    return output

def from_list_tensor_new(input:list, dtype=torch.BLS12_381_Fr_G1_Mont):
    base_input=[]
    for i in range(len(input)):
        # print(input[i].value)
        base_input.append(torch.tensor(input[i].value,dtype=dtype))
    return base_input

def from_tensor_list(input:torch.Tensor):
    output = input.tolist()
    # print("output 值为",output)
    for i in range(len(input)):
        output[i]=fr.Fr(value=output[i])
    return output

def from_tensor_list_new(input:torch.Tensor):
    output=[]
    for i in range(len(input)):
        output.append(fr.Fr(value=input[i].tolist()))
    return output