# Linear combination of a series of values
# For values [v_0, v_1,... v_k] returns:
# v_0 + challenge * v_1 + ... + challenge^k  * v_k
import torch.nn.functional as F
def Multiset_lc(values, challenge):
    kth_val = values.elements[-1]
    for val in reversed(values.elements[:-1]):
        for i in range(len(kth_val)):
            # kth_val[i] = kth_val[i].mul(challenge)
            kth_val[i]=F.mul_mod(kth_val[i],challenge)
            try:
                # kth_val[i] = kth_val[i].add(val[i])
                kth_val[i]=F.add_mod(kth_val[i],val[i])
            except Exception as e:
                print(i,kth_val[i].value,val[i].value)
                print(e)
            if(i==1020):
                pass
    return kth_val

def lc(values:list, challenge):
    kth_val = values[-1]
    for val in reversed(values[:-1]):
        kth_val = F.mul_mod(kth_val, challenge)
        kth_val = F.add_mod(kth_val, val)

    return kth_val

