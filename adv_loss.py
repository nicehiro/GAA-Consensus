from utils import collect_grads


def adv_loss1(model, loss1, loss2, iter_no):
    return collect_grads(model, -loss1 + 10 * loss2)


def adv_loss2(model, loss1, loss2, iter_no):
    if iter_no <= 1000:
        grads = collect_grads(model, loss2)
    else:
        grads = collect_grads(model, -loss1 + 2 * loss2)
    return grads


def adv_loss2n(model, loss1, loss2, iter_no):
    if iter_no <= 4000:
        grads = collect_grads(model, loss2)
    else:
        grads = collect_grads(model, -loss1 + 2 * loss2)
    return grads


def adv_loss3(model, loss1, loss2, iter_no):
    if iter_no % 1000 < 996:
        grads = collect_grads(model, loss2)
    else:
        grads = collect_grads(model, -loss1)
    return grads


def adv_loss4(model, loss1, loss2, iter_no):
    return collect_grads(model, loss2)


localVals = dict(**locals())
methodNames = [x for x in localVals if "_loss" in x]
adv_losses = {x: localVals[x] for x in methodNames}
