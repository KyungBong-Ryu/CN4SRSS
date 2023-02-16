from torch.utils.data import DataLoader


def DataLoader_multi_worker_FIX(**kargs):
    return DataLoader(**kargs)


if __name__ == '__main__':
    print("")
    print("#==============================================================================")
    print("     windows 10 pytorch dataloader multi-worker fix trick")
    print("     ")
    print("     written by <Kyung Bong Ryu> who want to use pytorch on windows 10")
    print("     ")
    print("     [ How to use ]")
    print("     <original pytorch>")
    print("     dataloader_train = torch.utils.data.DataLoader(**inputs)")
    print("     ")
    print("     <multi-worker FIX>")
    print("     dataloader_train = DataLoader_multi_worker_FIX(**inputs)")
    print("     ")
    print("     Note: you can use same form for **inputs")
    print("     ")
    print("#==============================================================================")
    print("")
    print("EoF: mp_dataloader.py")