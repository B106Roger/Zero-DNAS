def counting_forward_hook(module, inp, out):
    # try:
    # if not module.visited_backwards:
    #     return
    if isinstance(inp, tuple):
        inp = inp[0]
    inp = inp.view(inp.size(0), -1)
    x = (inp > 0).float()
    K = x @ x.t()
    K2 = (1.-x) @ (1.-x.t())
    print(K.cpu().numpy() + K2.cpu().numpy())
    # network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()
    # except:
    #     pass


def counting_backward_hook(module, inp, out):
    module.visited_backwards = True