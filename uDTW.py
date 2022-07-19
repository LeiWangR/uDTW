import numpy as np
import torch
import torch.cuda
from numba import jit
from torch.autograd import Function
from numba import cuda
import math


# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_cuda(D, Sig, D_new, Sig_new, gamma, bandwidth, max_i, max_j, n_passes, R, SigR):
    """
    :param seq_len: The length of the sequence (both inputs are assumed to be of the same size)
    :param n_passes: 2 * seq_len - 1 (The number of anti-diagonals)
    """
    # Each block processes one pair of examples
    b = cuda.blockIdx.x
    # We have as many threads as seq_len, because the most number of threads we need
    # is equal to the number of elements on the largest anti-diagonal
    tid = cuda.threadIdx.x

    # Compute I, J, the indices from [0, seq_len)

    # The row index is always the same as tid
    I = tid
    
    inv_gamma = 1.0 / gamma

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_passes):

        # The index is actually 'p - tid' but need to force it in-bounds
        J = max(0, min(p - tid, max_j - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == p and (I < max_i and J < max_j):
            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                r0 = -R[b, i - 1, j - 1] * inv_gamma
                r1 = -R[b, i - 1, j] * inv_gamma
                r2 = -R[b, i, j - 1] * inv_gamma
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)

                ratio0 = math.exp(r0 - rmax) / rsum
                ratio1 = math.exp(r1 - rmax) / rsum
                ratio2 = math.exp(r2 - rmax) / rsum

                softmind = ratio0 * D_new[b, i - 1, j - 1] + ratio1 * D_new[b, i - 1, j] + ratio2 * D_new[b, i, j - 1]
                R[b, i, j] = D[b, i - 1, j - 1] + softmind

                softmins = ratio0 * Sig_new[b, i - 1, j - 1] + ratio1 * Sig_new[b, i - 1, j] + ratio2 * Sig_new[
                    b, i, j - 1]
                SigR[b, i, j] = Sig[b, i - 1, j - 1] + softmins

        # Wait for other threads in this block
        cuda.syncthreads()


# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_backward_cuda(D, R, Sig, SigR, inv_gamma, bandwidth, max_i, max_j, n_passes, E, ES):
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    # Indexing logic is the same as above, however, the anti-diagonal needs to
    # progress backwards
    I = tid

    for p in range(n_passes):
        # Reverse the order to make the loop go backward
        rev_p = n_passes - p - 1

        # convert tid to I, J, then i, j
        J = max(0, min(rev_p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == rev_p and (I < max_i and J < max_j):

            if math.isinf(R[k, i, j]):
                R[k, i, j] = -math.inf

            if math.isinf(SigR[k, i, j]):
                SigR[k, i, j] = -math.inf

            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                r01 = -R[k, i, j - 1] * inv_gamma
                r11 = -R[k, i, j] * inv_gamma
                r21 = -R[k, i + 1, j - 1] * inv_gamma
                rmax1 = max(max(r01, r11), r21)
                rsum1 = math.exp(r01 - rmax1) + math.exp(r11 - rmax1) + math.exp(r21 - rmax1)
                ratio1 = math.exp(r11) / rsum1

                a0 = (R[k, i + 1, j] - D[k, i, j] - D[k, i + 1, j]) * inv_gamma * ratio1

                r02 = -R[k, i - 1, j] * inv_gamma
                r12 = -R[k, i, j] * inv_gamma
                r22 = -R[k, i - 1, j + 1] * inv_gamma
                rmax2 = max(max(r02, r12), r22)
                rsum2 = math.exp(r02 - rmax2) + math.exp(r12 - rmax2) + math.exp(r22 - rmax2)
                ratio2 = math.exp(r12) / rsum2

                b0 = (R[k, i, j + 1] - D[k, i, j] - D[k, i, j + 1]) * inv_gamma * ratio2

                r03 = -R[k, i + 1, j] * inv_gamma
                r13 = -R[k, i, j] * inv_gamma
                r23 = -R[k, i, j + 1] * inv_gamma
                rmax3 = max(max(r03, r13), r23)
                rsum3 = math.exp(r03 - rmax3) + math.exp(r13 - rmax3) + math.exp(r23 - rmax3)
                ratio3 = math.exp(r13) / rsum3

                c0 = (R[k, i + 1, j + 1] - D[k, i, j] - D[k, i + 1, j + 1]) * inv_gamma * ratio3

                E[k, i, j] = E[k, i + 1, j] * a0 + E[k, i, j + 1] * b0 + E[k, i + 1, j + 1] * c0

                # compute derivative
                sa0 = (SigR[k, i + 1, j] - Sig[k, i, j] - Sig[k, i + 1, j]) * inv_gamma * ratio1
                sb0 = (SigR[k, i, j + 1] - Sig[k, i, j] - Sig[k, i, j + 1]) * inv_gamma * ratio2
                sc0 = (SigR[k, i + 1, j + 1] - Sig[k, i, j] - Sig[k, i + 1, j + 1]) * inv_gamma * ratio3

                ES[k, i, j] = ES[k, i + 1, j] * sa0 + ES[k, i, j + 1] * sb0 + ES[k, i + 1, j + 1] * sc0

        # Wait for other threads in this block
        cuda.syncthreads()


# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTWCUDA(Function):
    """
    CUDA implementation is inspired by the diagonal one proposed in https://ieeexplore.ieee.org/document/8400444:
    "Developing a pattern discovery method in time series data and its GPU acceleration"
    """

    @staticmethod
    def forward(ctx, D, Sig, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.cuda.FloatTensor([gamma])
        bandwidth = torch.cuda.FloatTensor([bandwidth])

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        # Prepare the output array
        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0] = 0

        SigR = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * math.inf
        SigR[:, 0, 0] = 0

        D_new = torch.zeros((B, N + 2, M + 2), device=dev, dtype=dtype)
        D_new[:, 1:N+1, 1:M+1] = D
        Sig_new = torch.zeros((B, N + 2, M + 2), device=dev, dtype=dtype)
        Sig_new[:, 1:N+1, 1:M+1] = Sig

        # Run the CUDA kernel.
        # Set CUDA's grid size to be equal to the batch size (every CUDA block processes one sample pair)
        # Set the CUDA block size to be equal to the length of the longer sequence (equal to the size of the largest diagonal)
        compute_softdtw_cuda[B, threads_per_block](cuda.as_cuda_array(D.detach()), cuda.as_cuda_array(Sig.detach()),cuda.as_cuda_array(D_new.detach()), cuda.as_cuda_array(Sig_new.detach()),
                                                   gamma.item(), bandwidth.item(), N, M, n_passes,
                                                   cuda.as_cuda_array(R), cuda.as_cuda_array(SigR))

        ctx.save_for_backward(D, Sig, R.clone(), SigR.clone(), gamma, bandwidth)


        l1 = D.shape[1]
        l2 = D.shape[2]
        bw = int(bandwidth.item())

        if l1 < l2:
            if bw >= abs(l1 - l2) or bw == 0:
                return R[:, -2, -2], SigR[:, -2, -2]
            else:
                return R[:, -2, l1 - l2 + int(bandwidth.item()) - 2], SigR[:, -2, l1 - l2 + int(bandwidth.item()) - 2]
        elif l1 > l2:
            if bw >= abs(l1 - l2) or bw == 0:
                return R[:, -2, -2], SigR[:, -2, -2]
            else:
                return R[:, l2 - l1 + int(bandwidth.item()) - 2, -2], SigR[:, l2 - l1 + int(bandwidth.item()) - 2, -2]

        else:
            return R[:, -2, -2], SigR[:, -2, -2]




    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        dev = grad_output1.device
        dtype = grad_output1.dtype
        D, Sig, R, SigR, gamma, bandwidth = ctx.saved_tensors

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D

        Sig_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        Sig_[:, 1:N + 1, 1:M + 1] = Sig

        E = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        ES = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)

        l1 = D.shape[1]
        l2 = D.shape[2]
        bw = int(bandwidth.item())

        # print(l1, l2, '---')

        if l1 < l2:
            if bw >= abs(l1 - l2) or bw == 0:
                E[:, -1, -1] = 1
                R[:, :, -1] = 0
                R[:, -1, :] = 0

                R[:, -1, -1] = R[:, -2, -2]
                # ----------
                ES[:, -1, -1] = 1
                SigR[:, :, -1] = 0
                SigR[:, -1, :] = 0

                SigR[:, -1, -1] = SigR[:, -2, -2]
            else:
                E[:, -1, l1 - l2 + int(bandwidth.item()) - 1] = 1
                R[:, :, l1 - l2 + int(bandwidth.item()) - 1] = 0
                R[:, -1, :] = 0

                V = R[:, -2, l1 - l2 + int(bandwidth.item()) - 2]
                R[:, -1, l1 - l2 + int(bandwidth.item()) - 1] = V
                # -----------
                ES[:, -1, l1 - l2 + int(bandwidth.item()) - 1] = 1
                SigR[:, :, l1 - l2 + int(bandwidth.item()) - 1] = 0
                SigR[:, -1, :] = 0

                SigR[:, -1, l1 - l2 + int(bandwidth.item()) - 1] = SigR[:, -2, l1 - l2 + int(bandwidth.item()) - 2]

        elif l1 > l2:
            if bw >= abs(l1 - l2) or bw == 0:
                E[:, -1, -1] = 1
                R[:, :, -1] = 0
                R[:, -1, :] = 0

                R[:, -1, -1] = R[:, -2, -2]
                # ----------
                ES[:, -1, -1] = 1
                SigR[:, :, -1] = 0
                SigR[:, -1, :] = 0

                SigR[:, -1, -1] = SigR[:, -2, -2]

            else:
                E[:, l2 - l1 + int(bandwidth.item()) - 1, -1] = 1
                R[:, :, -1] = 0
                R[:, l2 - l1 + int(bandwidth.item()) - 1, :] = 0

                V = R[:, l2 - l1 + int(bandwidth.item()) - 2, -2]
                R[:, l2 - l1 + int(bandwidth.item()) - 1, -1] = V
                # -----------
                ES[:, l2 - l1 + int(bandwidth.item()) - 1, -1] = 1
                SigR[:, :, -1] = 0
                SigR[:, l2 - l1 + int(bandwidth.item()) - 1, :] = 0

                SigR[:, l2 - l1 + int(bandwidth.item()) - 1, -1] = SigR[:, l2 - l1 + int(bandwidth.item()) - 2, -2]

        else:
            E[:, -1, -1] = 1
            R[:, :, -1] = 0
            R[:, -1, :] = 0

            R[:, -1, -1] = R[:, -2, -2]
            # ----------
            ES[:, -1, -1] = 1
            SigR[:, :, -1] = 0
            SigR[:, -1, :] = 0

            SigR[:, -1, -1] = SigR[:, -2, -2]



        # Grid and block sizes are set same as done above for the forward() call
        compute_softdtw_backward_cuda[B, threads_per_block](cuda.as_cuda_array(D_), cuda.as_cuda_array(R), cuda.as_cuda_array(Sig_), cuda.as_cuda_array(SigR),
                                                            1.0 / gamma.item(), bandwidth.item(), N, M, n_passes,
                                                            cuda.as_cuda_array(E), cuda.as_cuda_array(ES))
        E = E[:, 1:N + 1, 1:M + 1]
        ES = ES[:, 1:N + 1, 1:M + 1]
        return grad_output1.view(-1, 1, 1).expand_as(E) * E, grad_output2.view(-1, 1, 1).expand_as(ES) * ES, None, None


# ----------------------------------------------------------------------------------------------------------------------
#
# The following is the CPU implementation based on
# https://github.com/Sleepwalking/pytorch-softdtw
# and https://github.com/Maghoumi/pytorch-softdtw-cuda
# Credit goes to Kanru Hua and Maghoumi.
#
# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True)
def compute_softdtw(D, Sig, gamma, bandwidth):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]

    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0

    SigR = np.ones((B, N + 2, M + 2)) * np.inf
    SigR[:, 0, 0] = 0

    alpha = 1.0 / gamma

    D_new = np.zeros((B, N + 2, M + 2))
    D_new[:, 1:N+1, 1:M+1] = D

    Sig_new = np.zeros((B, N + 2, M + 2))
    Sig_new[:, 1:N+1, 1:M+1] = Sig

    for b in range(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                r0 = -alpha * R[b, i - 1, j - 1]
                r1 = -alpha * R[b, i - 1, j]
                r2 = -alpha * R[b, i, j - 1]
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)

                ratio0 = np.exp(r0-rmax) / rsum
                ratio1 = np.exp(r1-rmax) / rsum
                ratio2 = np.exp(r2-rmax) / rsum

                softmind =  ratio0 * D_new[b, i - 1, j - 1] + ratio1 * D_new[b, i - 1, j] + ratio2 * D_new[b, i, j - 1]
                R[b, i, j] = D[b, i - 1, j - 1] + softmind

                softmins = ratio0 * Sig_new[b, i-1, j-1] + ratio1 * Sig_new[b, i-1, j] + ratio2 * Sig_new[b, i, j-1]
                SigR[b, i, j] = Sig[b, i - 1, j - 1] + softmins

    return R, SigR


# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True)
def compute_softdtw_backward(D_, Sig_, R, SigR, gamma, bandwidth):
    # print(D_.shape, R.shape)
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]

    D = np.zeros((B, N + 2, M + 2))
    Sig = np.zeros((B, N + 2, M + 2))

    E = np.zeros((B, N + 2, M + 2))
    ES = np.zeros((B, N + 2, M + 2))

    D[:, 1:N + 1, 1:M + 1] = D_
    Sig[:, 1:N + 1, 1:M + 1] = Sig_

    l1 = D_.shape[1]
    l2 = D_.shape[2]
    bw = int(bandwidth.item())

    alpha = 1.0 / gamma

    if l1 < l2:
        if bw >= abs(l1 - l2) or bw == 0:
            E[:, -1, -1] = 1
            R[:, :, -1] = 0
            R[:, -1, :] = 0

            R[:, -1, -1] = R[:, -2, -2]
            # ----------
            ES[:, -1, -1] = 1
            SigR[:, :, -1] = 0
            SigR[:, -1, :] = 0

            SigR[:, -1, -1] = SigR[:, -2, -2]
        else:
            E[:, -1, l1 - l2 + int(bandwidth.item())-1] = 1
            R[:, :, l1 - l2 + int(bandwidth.item())-1] = 0
            R[:, -1, :] = 0

            V = R[:, -2, l1 - l2 + int(bandwidth.item()) - 2]
            R[:, -1, l1 - l2 + int(bandwidth.item())-1] = V
            # -----------
            ES[:, -1, l1 - l2 + int(bandwidth.item()) - 1] = 1
            SigR[:, :, l1 - l2 + int(bandwidth.item()) - 1] =0
            SigR[:, -1, :] = 0

            SigR[:, -1, l1 - l2 + int(bandwidth.item()) - 1] = SigR[:, -2, l1 - l2 + int(bandwidth.item()) - 2]

    elif l1 > l2:
        if bw >= abs(l1 - l2) or bw == 0:
            E[:, -1, -1] = 1
            R[:, :, -1] = 0
            R[:, -1, :] = 0

            R[:, -1, -1] = R[:, -2, -2]
            # ----------
            ES[:, -1, -1] = 1
            SigR[:, :, -1] = 0
            SigR[:, -1, :] = 0

            SigR[:, -1, -1] = SigR[:, -2, -2]

        else:
            E[:, l2 - l1 + int(bandwidth.item())-1, -1] = 1
            R[:, :, -1] = 0
            R[:, l2 - l1 + int(bandwidth.item())-1, :] = 0

            V = R[:, l2 - l1 + int(bandwidth.item()) - 2, -2]
            R[:, l2 - l1 + int(bandwidth.item())-1, -1] = V
            # -----------
            ES[:, l2 - l1 + int(bandwidth.item()) - 1, -1] = 1
            SigR[:, :, -1] = 0
            SigR[:, l2 - l1 + int(bandwidth.item()) - 1, :] = 0

            SigR[:, l2 - l1 + int(bandwidth.item()) - 1, -1] = SigR[:, l2 - l1 + int(bandwidth.item()) - 2, -2]

    else:
        E[:, -1, -1] = 1
        R[:, :, -1] = 0
        R[:, -1, :] = 0

        R[:, -1, -1] = R[:, -2, -2]
        # ----------
        ES[:, -1, -1] = 1
        SigR[:, :, -1] = 0
        SigR[:, -1, :] = 0

        SigR[:, -1, -1] = SigR[:, -2, -2]

    for k in range(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):

                if np.isinf(R[k, i, j]):
                    R[k, i, j] = 0
                if np.isinf(SigR[k, i, j]):
                    SigR[k, i, j] = 0

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                r01 = -R[k, i, j - 1] * alpha
                r11 = -R[k, i, j] * alpha
                r21 = -R[k, i + 1, j - 1] * alpha
                rmax1 = max(max(r01, r11), r21)
                rsum1 = np.exp(r01 - rmax1) + np.exp(r11 - rmax1) + np.exp(r21 - rmax1)
                ratio1 = np.exp(r11) / rsum1

                a0 = (R[k, i + 1, j] - D[k, i, j] - D[k, i + 1, j]) * alpha * ratio1

                r02 = -R[k, i - 1, j] * alpha
                r12 = -R[k, i, j] * alpha
                r22 = -R[k, i - 1, j + 1] * alpha
                rmax2 = max(max(r02, r12), r22)
                rsum2 = np.exp(r02 - rmax2) + np.exp(r12 - rmax2) + np.exp(r22 - rmax2)
                ratio2 = np.exp(r12) / rsum2

                b0 = (R[k, i, j + 1] - D[k, i, j] - D[k, i, j + 1]) * alpha * ratio2

                r03 = -R[k, i + 1, j] * alpha
                r13 = -R[k, i, j] * alpha
                r23 = -R[k, i, j + 1] * alpha
                rmax3 = max(max(r03, r13), r23)
                rsum3 = np.exp(r03 - rmax3) + np.exp(r13 - rmax3) + np.exp(r23 - rmax3)
                ratio3 = np.exp(r13) / rsum3

                c0 = (R[k, i + 1, j + 1] - D[k, i, j] - D[k, i + 1, j + 1]) * alpha * ratio3

                E[k, i, j] = E[k, i + 1, j] * a0 + E[k, i, j + 1] * b0 + E[k, i + 1, j + 1] * c0

                # compute derivative
                sa0 = (SigR[k, i + 1, j] - Sig[k, i, j] - Sig[k, i + 1, j]) * alpha * ratio1
                sb0 = (SigR[k, i, j + 1] - Sig[k, i, j] - Sig[k, i, j + 1]) * alpha * ratio2
                sc0 = (SigR[k, i + 1, j + 1] - Sig[k, i, j] - Sig[k, i + 1, j + 1]) * alpha * ratio3

                ES[k, i, j] = ES[k, i + 1, j] * sa0 + ES[k, i, j + 1] * sb0 + ES[k, i + 1, j + 1] * sc0

    return E[:, 1:N + 1, 1:M + 1], ES[:, 1:N + 1, 1:M + 1]


# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTW(Function):
    """
    CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
    """

    @staticmethod
    def forward(ctx, D, Sig, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)  # dtype fixed
        bandwidth = torch.Tensor([bandwidth]).to(dev).type(dtype)
        D_ = D.detach().cpu().numpy()
        Sig_ = Sig.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        R, SigR = torch.Tensor(compute_softdtw(D_, Sig_, g_, b_)).to(dev).type(dtype)
        ctx.save_for_backward(D, Sig, R, SigR, gamma, bandwidth)

        l1 = D.shape[1]
        l2 = D.shape[2]
        bw = int(bandwidth.item())

        if l1 < l2:
            if bw >= abs(l1 - l2) or bw == 0:
                V = R[:, -2, -2]
                VS = SigR[:, -2, -2]
            else:
                V = R[:, -2, l1-l2+int(bandwidth.item())-2]
                VS = SigR[:, -2, l1 - l2 + int(bandwidth.item()) - 2]
        elif l1 > l2:
            if bw >= abs(l1 - l2) or bw == 0:
                V = R[:, -2, -2]
                VS = SigR[:, -2, -2]
            else:
                V = R[:, l2-l1+int(bandwidth.item())-2, -2]
                VS = SigR[:, l2 - l1 + int(bandwidth.item()) - 2, -2]
        else:
            V = R[:, -2, -2]
            VS = SigR[:, -2, -2]

        return V, VS

    @staticmethod
    def backward(ctx, grad_output1, grad_output2):

        dev = grad_output1.device
        dtype = grad_output1.dtype
        D, Sig, R, SigR, gamma, bandwidth = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        Sig_ = Sig.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        SigR_ = SigR.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        E, ES = torch.Tensor(compute_softdtw_backward(D_, Sig_, R_, SigR_, g_, b_)).to(dev).type(dtype)

        return grad_output1.view(-1, 1, 1).expand_as(E) * E, grad_output2.view(-1, 1, 1).expand_as(ES) * ES, None, None


# ----------------------------------------------------------------------------------------------------------------------
class SoftDTW(torch.nn.Module):
    """
    The soft DTW implementation that optionally supports CUDA
    """

    def __init__(self, use_cuda, gamma=1.0, normalize=False, bandwidth=None):
        """
        Initializes a new instance using the supplied parameters
        :param use_cuda: Flag indicating whether the CUDA implementation should be used
        :param gamma: sDTW's gamma parameter
        :param normalize: Flag indicating whether to perform normalization
                          (as discussed in https://github.com/mblondel/soft-dtw/issues/10#issuecomment-383564790)
        :param bandwidth: Sakoe-Chiba bandwidth for pruning. Passing 'None' will disable pruning.
        """
        super(SoftDTW, self).__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda

    def _get_func_dtw(self, x, y):
        """
        Checks the inputs and selects the proper implementation to use.
        """
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        # Make sure the dimensions match
        assert bx == by  # Equal batch sizes
        assert dx == dy  # Equal feature dimensions

        use_cuda = self.use_cuda

        if use_cuda and (lx > 1024 or ly > 1024):  # We should be able to spawn enough threads in CUDA
            print(
                "SoftDTW: Cannot use CUDA because the sequence length > 1024 (the maximum block size supported by CUDA)")
            use_cuda = False

        # Finally, return the correct function
        return _SoftDTWCUDA.apply if use_cuda else _SoftDTW.apply

    def _calc_distance_matrix(self, x, y, sigma_x, sigma_y, beta):
        """
        Calculates the Euclidean distance between each element in x and y per timestep
        """
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)

        sigmaX = sigma_x.repeat(1, 1, m)
        sigmaY = sigma_y.repeat(1, 1, n).permute(0, 2, 1)

        # sigma_xy = torch.mul(sigmaX, sigmaY)
        # sigma_xy = torch.add(sigmaX, sigmaY)
        # sigma_xy = torch.mul(torch.pow(sigmaX, 2), torch.pow(sigmaY, 2))
        sigma_xy = torch.add(torch.pow(sigmaX, 2), torch.pow(sigmaY, 2))

        # normalize between 0 and 1
        # sigma_xy = sigma_xy  / sigma_xy.max()

        return torch.pow(x - y, 2).sum(3) / sigma_xy / d, (1 / 2) * beta * torch.log(sigma_xy)

    def forward(self, X, Y, Sigma_X, Sigma_Y, beta):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed results
        """

        # Check the inputs and get the correct implementation
        func_dtw = self._get_func_dtw(X, Y)

        if self.normalize:
            # Stack everything up and run
            D_xy, S_xy = self._calc_distance_matrix(X, Y, Sigma_X, Sigma_Y, beta)
            D_xx, S_xx = self._calc_distance_matrix(X, X, Sigma_X, Sigma_X, beta)
            D_yy, S_yy = self._calc_distance_matrix(Y, Y, Sigma_Y, Sigma_Y, beta)

            out_xy, outs_xy = func_dtw(D_xy, S_xy, self.gamma, self.bandwidth)
            out_xx, outs_xx = func_dtw(D_xx, S_xx, self.gamma, self.bandwidth)
            out_yy, outs_yy = func_dtw(D_yy, S_yy, self.gamma, self.bandwidth)

            return out_xy - 1 / 2 * (out_xx + out_yy), outs_xy - 1 / 2 * (outs_xx + outs_yy)
        else:
            D_xy, S_xy = self._calc_distance_matrix(X, Y, Sigma_X, Sigma_Y, beta)
            out_xy, outs_xy = func_dtw(D_xy, S_xy, self.gamma, self.bandwidth)
            return out_xy, outs_xy
