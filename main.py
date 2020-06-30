import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.distributed as dist
import torchvision
from scipy.sparse.linalg import cg, lsmr
from time import time
from torch.nn.functional import softmax, softplus
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR10, EMNIST, MNIST
from torchvision.transforms import Compose, Lambda, Normalize, ToTensor


def _cg(A, b, rtol, maxit, M_inv=None):
    """The conjugate gradient method to approximately solve ``Ax = b`` for
    ``x``, where ``A`` is symmetric and positive-definite.

    Args:
        A (Tensor): matrix ``A`` in ``Ax = b``. Matrix must be symmetric and
            positive-definite.
        b (Tensor): vector ``b`` in ``Ax = b``.
        rtol (double): termination tolerance for relative residual.
        maxit (int): maximum iterations.
        M_inv (Tensor, optional): inverse of preconditioner matrix ``M``.
            Defaults to ``None``.
    """
    if rtol < 0.0:
        raise ValueError("Invalid termination tolerance: {}."
            " It must be non-negative.".format(rtol))
    if maxit < 0:
        raise ValueError("Invalid maximum iterations: {}."
            " It must be non-negative.".format(maxit))
    b = b.reshape(-1)
    m = len(b)
    device = b.device
    iters = 0
    x = torch.zeros(m, device=device).double()
    bb = torch.dot(b,b).item()
    if bb == 0:
        return x
    r = b.clone()
    if M_inv is None:
        z = r
        assert(id(z) == id(r))
    else:
        z = M_inv(r)
    p = z.clone()
    rz_old = torch.dot(r,z).item()
    while iters < maxit:
        iters += 1
        Ap = A.mv(p).reshape(-1)
        pAp = torch.dot(p,Ap).item()
        assert pAp > 0, "A is not positive-definite."
        alpha = rz_old / pAp
        x.add_(p,  alpha=alpha)
        r.sub_(Ap, alpha=alpha)
        rr = torch.dot(r,r).item()
        if np.sqrt(rr/bb) <= rtol:
            return x, 0
        if M_inv is None:
            assert(id(z) == id(r))
            rz_new = rr
        else:
            z = M_inv(r)
            rz_new = torch.dot(r,z).item()
        beta = rz_new/rz_old
        p.mul_(beta).add_(z)
        rz_old = rz_new
    return x, 1


def _obj_fun(model_name, dataloader, weights, device, dtype, comp_func=True,
             comp_grad=True, comp_hess=True, use_avg_all_reduce=True,
             use_regularization=False):
    loss = 0
    grad = 0
    hess = 0
    for i, data in enumerate(dataloader):
        inputs = data[0].to(device=device, dtype=dtype)
        inputs = inputs.view(inputs.size(0), -1)
        targets = data[1].to(device)
        if model_name == 'NLLS':
            num_samples = len(targets)
            iw = inputs.mv(weights)
            so = torch.nn.functional.softplus(iw).sub(targets.to(dtype))
            if comp_func: ## function value
                f = so.pow(2).sum()
                loss += f.div(num_samples)
            if comp_grad or comp_hess:
                si = torch.sigmoid(iw)
            if comp_grad: ## gradient
                g = 2*inputs.t().mv(so*si)
                grad += g.div(num_samples)
            if comp_hess: ## Hessian
                H = 2*inputs.t().mul(so*si*(1-si)+si*si).mm(inputs)
                hess += H.div(num_samples)
        else:
            num_samples, num_features = inputs.size()
            num_classes = weights.numel() // num_features
            # turn targets into matrix
            rows = torch.linspace(0, num_samples-1, num_samples).long()
            cols = targets.cpu()
            targets_I = torch.cat([rows.view(1,-1), cols.view(1,-1)], dim=0)
            targets_V = torch.ones(num_samples).long()
            size = torch.Size([num_samples, num_classes])
            targets_matrix = torch.sparse_coo_tensor(targets_I, targets_V,
                                                     size=size, device=device,
                                                     dtype=dtype)
            # use "log sum exp trick" for numerical stability
            IW = inputs.mm(weights.view(num_classes,num_features).t())
            large_vals, _ = IW.max(dim=1)
            large_vals = torch.max(torch.zeros_like(large_vals), large_vals)
            IW_trick = IW.sub(large_vals.view(-1,1))
            sum_exp_trick = torch.cat([-large_vals.view(-1,1), IW_trick], dim=1)
            sum_exp_trick = sum_exp_trick.exp().sum(dim=1)
            log_sum_exp_trick = torch.log(sum_exp_trick).add(large_vals)
            if comp_func: ## function value
                f = log_sum_exp_trick.sum().sub(IW[rows,cols].sum())
                loss += f.div(num_samples)
            if comp_grad or comp_hess:
                S = IW_trick.exp().div(sum_exp_trick.view(-1,1))
            if comp_grad: ## gradient
                g = inputs.t().mm(S-targets_matrix).t().flatten()
                grad += g.div(num_samples)
            if comp_hess: ## Hessian
                if i == 0: # add a fixed size sparse matrix to block diagonals
                    base = torch.linspace(0,num_classes-1,num_classes).long()
                    base = base.repeat_interleave(num_features**2)
                    base.mul_(num_features)
                    rows = torch.linspace(0,num_features-1,num_features).long()
                    rows = rows.repeat_interleave(num_features)
                    rows = rows.repeat(num_classes).add(base)
                    cols = torch.linspace(0,num_features-1,num_features).long()
                    cols = cols.repeat(num_features).repeat(num_classes)
                    cols.add_(base)
                    I = torch.cat([rows.view(1,-1), cols.view(1,-1)], dim=0)
                    H_size = torch.Size([weights.numel(), weights.numel()])
                S = S.t().view(num_classes, num_samples, 1).mul(inputs)
                H = torch.einsum('ij,bjk->bik', inputs.t(), S).flatten()
                H = torch.sparse_coo_tensor(I, H, size=H_size, device=device,
                                            dtype=dtype)
                S = S.transpose(1,2).view(weights.numel(),-1)
                H = S.mm(S.t()).mul(-1).add(H)
                hess += H.div(num_samples)
        if comp_func and use_regularization:
            loss.add_(weights.norm().pow(2).div(2*num_samples))
        if comp_grad and use_regularization:
            grad.add_(weights.div(num_samples))
        if comp_hess and use_regularization:
            hess.diagonal().add_(1.0/num_samples)
    loss /= len(dataloader)
    grad /= len(dataloader)
    hess /= len(dataloader)
    if comp_func and use_avg_all_reduce:
        dist.all_reduce(loss)
        loss.div_(dist.get_world_size())
    if comp_grad and use_avg_all_reduce:
        dist.all_reduce(grad)
        grad.div_(dist.get_world_size())
    if comp_hess and use_avg_all_reduce:
        dist.all_reduce(hess)
        hess.div_(dist.get_world_size())
    return loss, grad, hess


def _parse_args():
    data_choices = ['CIFAR10','EMNIST','MNIST']
    model_choices = ['NLLS','Softmax']
    parser = argparse.ArgumentParser(description='DINO on CIFAR10 NLLS')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--data-name', type=str, default='MNIST',
                        help='name of dataset to use', choices=data_choices)
    parser.add_argument('--data-root', default='~/Downloads/', type=str,
                        help='root directory of CIFAR10 dataset')
    parser.add_argument('--dataloader-num-workers', default=16, type=int,
                        help='number of data loading workers')
    parser.add_argument("--distributed-backend", type=str, default="nccl",
                        help="the backend to use in ``init_process_group``")
    parser.add_argument("--distributed-init-method", type=str,
                        default="env://",
                        help="init method for ``init_process_group``")
    parser.add_argument("--distributed-rank", type=int, default=0,
                        help="rank of the current process")
    parser.add_argument("--distributed-world-size", type=int, default=1,
                        help="number of processes participating in the job")
    parser.add_argument('--max-communication-rounds', default=520, type=int,
                        help='max number of communication rounds to perform')
    parser.add_argument('--model-name', type=str, default='Softmax',
                        help='name of model to train', choices=model_choices)
    parser.add_argument('--phi', default=1e-6, type=float,
                        help='the hyper-parameter phi')
    parser.add_argument('--rho', default=1e-4, type=float,
                        help='the line-search parameter rho')
    parser.add_argument('--subproblem-max-iter', default=50, type=int,
                        help='max number of subproblem solver iterations,'
                        ' if not using exact subproblem solve')
    parser.add_argument('--subproblem-tol', default=1e-4, type=float,
                        help='subproblem solver relative residual tolerance,'
                        ' if not using exact subproblem solve')
    parser.add_argument('--theta', default=1e-4, type=float,
                        help='the hyper-parameter theta')
    parser.add_argument('--use-cuda', action='store_true',
                        help='if to use CUDA')
    parser.add_argument('--use-double-precision', action='store_true',
                        help='if to use float64')
    parser.add_argument('--use-exact-solve', action='store_true',
                        help='if to use exact subproblem solutions')
    parser.add_argument("--use-regularization", action='store_true',
                        help='if to use l2 regularization')
    args = parser.parse_args()
    return args


def _get_step_size(model_name, dataloader, weights, device, dtype, global_loss,
                   global_grad, update_direction, rho, use_regularization):
    step_size = 1.0
    direction_dot_grad = update_direction.dot(global_grad)
    while True:
        new_loss, _,_ = _obj_fun(model_name, dataloader,
                                 weights.add(update_direction, alpha=step_size),
                                 device, dtype, comp_func=True, comp_grad=False,
                                 comp_hess=False, use_avg_all_reduce=True,
                                 use_regularization=use_regularization)
        if new_loss <= global_loss + step_size * rho * direction_dot_grad:
            break
        else:
            step_size *= 0.5
    return step_size


def _get_test_val(model_name, dataloader, weights, device, dtype):
    loss = torch.zeros(1, device=device, dtype=dtype)
    correct = torch.zeros(1, device=device, dtype=dtype)
    total = torch.zeros(1, device=device, dtype=dtype)
    for data in iter(dataloader):
        inputs = data[0].to(device=device, dtype=dtype)
        inputs = inputs.view(inputs.size(0), -1)
        targets = data[1].to(device)
        if model_name == 'NLLS':
            num_samples = len(targets)
            iw = inputs.mv(weights)
            so = torch.nn.functional.softplus(iw).sub(targets.to(dtype))
            loss += so.pow(2).sum().div(num_samples)
        else:
            num_samples, num_features = inputs.size()
            num_classes = weights.numel() // num_features
            # use "log sum exp trick" for numerical stability
            IW = inputs.mm(weights.view(num_classes,num_features).t())
            large_vals, _ = IW.max(dim=1)
            large_vals = torch.max(torch.zeros_like(large_vals), large_vals)
            IW_trick = IW.sub(large_vals.view(-1,1))
            sum_exp_trick = torch.cat([-large_vals.view(-1,1), IW_trick], dim=1)
            sum_exp_trick = sum_exp_trick.exp().sum(dim=1)
            S = IW_trick.exp().div(sum_exp_trick.view(-1,1))
            _, predicted = S.max(dim=1)
            correct += predicted.eq(targets).sum()
            total += targets.numel()
    if model_name == 'NLLS':
        dist.all_reduce(loss)
        loss.div_(dist.get_world_size())
        return loss
    else:
        dist.all_reduce(correct)
        dist.all_reduce(total)
        return 100*correct/total


def _get_update_direction(model_name, dataloader, weights, device, dtype,
                          global_grad, phi, theta, use_exact_solve,
                          subproblem_tol, subproblem_max_iter,
                          use_regularization):
    condition = theta * global_grad.norm().pow(2).item()
    _,_, local_hess = _obj_fun(model_name, dataloader, weights, device, dtype,
                               comp_func=False, comp_grad=False, comp_hess=True,
                               use_avg_all_reduce=False,
                               use_regularization=use_regularization)
    eye = torch.eye(local_hess.size(0), device=device, dtype=dtype)
    ## vector v^{(1)}
    if use_exact_solve:
        grad_tilde = torch.cat([global_grad, torch.zeros_like(global_grad)])
        local_hess_tilde = torch.cat([local_hess, phi*eye], dim=0)
        v1 = local_hess_tilde.pinverse().mv(grad_tilde)
    else:
        tmp = lsmr(local_hess.cpu().numpy(), global_grad.cpu().numpy(),
                   damp=phi, atol=0, btol=subproblem_tol,
                   maxiter=subproblem_max_iter)
        v1, istop = tmp[0], tmp[1]
        # assert istop == 1
        v1 = torch.from_numpy(v1).to(device=device, dtype=dtype)
    v1_dot_grad = v1.dot(global_grad).item()
    if v1_dot_grad >= condition:
        update_direction = -v1
    else: ## vector v^{(2)}
        if use_exact_solve:
            HH = local_hess.t().mm(local_hess).add(phi*phi*eye)
            u = torch.cholesky(HH)
            v2 = torch.cholesky_solve(global_grad, u)
        else:
            v2, istop = _cg(HH, global_grad, rtol=subproblem_tol,
                            maxit=subproblem_max_iter)
            # assert istop == 0
        v2_dot_grad = v2.dot(global_grad).item()
        assert v2_dot_grad > 0
        lamda = (-v1_dot_grad + condition) / v2_dot_grad
        update_direction = -v1-lamda*v2
    dist.all_reduce(update_direction)
    update_direction.div_(dist.get_world_size())
    return update_direction


@torch.no_grad()
def derivative_test(): # Test accuracy of gradient and Hessian computations.
    ## reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args = _parse_args()
    dist.init_process_group(backend=args.distributed_backend,
                            init_method=args.distributed_init_method,
                            world_size=args.distributed_world_size,
                            rank=args.distributed_rank)

    ## data
    if args.data_name == "CIFAR10":
        dataset = CIFAR10(args.data_root, transform=ToTensor())
        num_classes = 10
        num_features = 32*32*3
    elif args.data_name == "EMNIST":
        dataset = EMNIST(args.data_root, transform=ToTensor(), split="digits")
        num_classes = 10
        num_features = 28*28
    else:
        dataset = MNIST(args.data_root, transform=ToTensor())
        num_classes = 10
        num_features = 28*28
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset,batch_size=args.batch_size,sampler=sampler,
                            num_workers=args.dataloader_num_workers)

    ## model
    device = torch.device("cuda" if args.use_cuda else "cpu")
    dtype = torch.float64 if args.use_double_precision else torch.float32
    d = num_features if args.model_name == "NLLS" else num_features*num_classes
    weights = torch.randn(d, device=device, dtype=dtype)
    dist.all_reduce(weights)
    weights.div(dist.get_world_size())

    ## derivative test
    start_time = time()
    f0, g0, H0 = _obj_fun(args.model_name, dataloader, weights, device, dtype,
                          use_regularization=args.use_regularization)
    dw = torch.randn_like(weights)
    dist.all_reduce(dw)
    dw.div(dist.get_world_size())
    M = 30
    dws = np.zeros((M,1))
    firsterror = np.zeros((M,1))
    order1 = np.zeros((M-1,1))
    seconderror = np.zeros((M,1))
    order2 = np.zeros((M-1,1))
    for i in range(M):
        st = time()
        f1, _, _ = _obj_fun(args.model_name, dataloader, weights.add(dw),
                            device, dtype, comp_grad=False, comp_hess=False,
                            use_regularization=args.use_regularization)
        # f(w0+dw) ≈ f(w0) + ∇f(w0)^T dw
        firsterror[i] = f1.sub(f0 + dw.dot(g0)).div(f0).abs().item()
        # f(w0+dw) ≈ f(w0) + ∇f(w0)^T dw + 0.5 dw^T ∇^2f(w0) dw
        seconderror[i] = (
            f1.sub(f0 + dw.dot(g0) + H0.mv(dw).dot(dw)/2).div(f0).abs().item())
        if i > 0:
            order1[i-1] = np.log2(firsterror[i-1]/firsterror[i])
            order2[i-1] = np.log2(seconderror[i-1]/seconderror[i])
        dws[i] = torch.norm(dw).item()
        dw.mul_(0.5)
        print("Iter {}, Time {:.2f}s".format(i, time()-st))
    if dist.get_rank() == 0:
        step = [2**(-i-1) for i in range(M)]
        plt.figure(figsize=(12,8))
        plt.subplot(221)
        plt.loglog(step, abs(firsterror),'b', label = '1st Order Err')
        plt.loglog(step, dws**2,'r', label = 'order')
        plt.gca().invert_xaxis()
        plt.legend()
        plt.subplot(222)
        plt.semilogx(step[1:], order1,'b', label = '1st Order')
        plt.gca().invert_xaxis()
        plt.legend()
        plt.subplot(223)
        plt.loglog(step, abs(seconderror),'b', label = '2nd Order Err')
        plt.loglog(step, dws**3,'r', label = 'Order')
        plt.gca().invert_xaxis()
        plt.legend()
        plt.subplot(224)
        plt.semilogx(step[1:], order2,'b', label = '2nd Order')
        plt.gca().invert_xaxis()
        plt.legend()
        plt.savefig('derivative_test.pdf')
    print("Total Time: {:.2f}s".format(time()-start_time))


@torch.no_grad()
def main():
    ## reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args = _parse_args()
    dist.init_process_group(backend=args.distributed_backend,
                            init_method=args.distributed_init_method,
                            world_size=args.distributed_world_size,
                            rank=args.distributed_rank)

    ## data
    if args.data_name == "CIFAR10":
        transform = Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_dataset = CIFAR10(args.data_root, transform=transform)
        test_dataset = CIFAR10(args.data_root, transform=transform, train=False)
        num_classes = 10
        num_features = 32*32*3
    elif args.data_name == "EMNIST":
        # transform = Compose([ToTensor(), Normalize([0.1732], [0.3317])])
        transform = ToTensor()
        train_dataset = EMNIST(args.data_root, transform=transform,
                               split="digits")
        test_dataset = EMNIST(args.data_root, transform=transform,
                              split="digits", train=False)
        num_classes = 10
        num_features = 28*28
    else:
        transform = Compose([ToTensor(), Normalize([0.1732], [0.3317])])
        train_dataset = MNIST(args.data_root, transform=transform)
        test_dataset = MNIST(args.data_root, transform=transform, train=False)
        num_classes = 10
        num_features = 28*28
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=args.dataloader_num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             sampler=test_sampler,
                             num_workers=args.dataloader_num_workers)

    ## model
    device = torch.device("cuda" if args.use_cuda else "cpu")
    dtype = torch.float64 if args.use_double_precision else torch.float32
    d = num_features if args.model_name == "NLLS" else num_features*num_classes
    weights = torch.zeros(d, device=device, dtype=dtype)

    ## run
    header = ["iter", "ccr", "loss", "grad", "test", "alpha"]
    print(("{:^16s}"*len(header)).format(*header))
    iterations_list = []
    communication_rounds_list = []
    loss_list = []
    grad_norm_list = []
    test_val_list = []
    step_size_list = []
    communication_rounds = 0
    iteration = 0
    while communication_rounds < args.max_communication_rounds:
        iterations_list.append(iteration)
        communication_rounds_list.append(communication_rounds)
        loss, grad, _ = _obj_fun(args.model_name, train_loader, weights, device,
                                 dtype, comp_hess=False,
                                 use_regularization=args.use_regularization)
        loss_list.append(loss)
        grad_norm_list.append(grad.norm().item())
        test_val = _get_test_val(args.model_name, test_loader, weights, device,
                                 dtype)
        test_val_list.append(test_val.item())
        update_direction = _get_update_direction(args.model_name, train_loader,
                                                 weights, device, dtype, grad,
                                                 args.phi, args.theta,
                                                 args.use_exact_solve,
                                                 args.subproblem_tol,
                                                 args.subproblem_max_iter,
                                                 args.use_regularization)
        step_size = _get_step_size(args.model_name, train_loader, weights,
                                   device, dtype, loss, grad, update_direction,
                                   args.rho, args.use_regularization)
        step_size_list.append(step_size)
        weights.add_(update_direction, alpha=step_size)
        # code can be changed to have 5 or 6 communication rounds per iteration
        communication_rounds += (5 if iteration == 0 else 6)
        iteration += 1
        print("{:^16g}{:^16g}{:^16.2e}{:^16.2e}{:^16.2e}{:^16.2e}".format(
              iterations_list[-1], communication_rounds_list[-1], loss_list[-1],
              grad_norm_list[-1], test_val_list[-1], step_size_list[-1]))
    if dist.get_rank() == 0:
        data = zip(iterations_list, communication_rounds_list, loss_list,
                   grad_norm_list, test_val_list, step_size_list)
        np.savetxt('DINO.csv',list(data),delimiter=',',header=",".join(header))


if __name__ == '__main__':
    main()
