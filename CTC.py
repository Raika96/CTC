import numpy as np


class CTC:

    def __init__(self, params, seq):
        self.params = params
        self.seq = seq
        self.L = 2 * seq.shape[0] + 1  # Length of label sequence with blanks
        self.T = params.shape[1]
        # self.forward_backward()

    @staticmethod
    def forward(params, seq):
        L = 2 * seq.shape[0] + 1  # Length of label sequence with blanks
        T = params.shape[1]  # Length of utterance (time)
        alphas = np.zeros((L, T))

        # Forward pass
        # Initialisation
        alphas[0, 0] = params[0, 0]
        alphas[1, 0] = params[seq[0], 0]

        # Rescaling for avoiding underflow
        c = np.sum(alphas[:, 0])
        alphas[:, 0] = alphas[:, 0] / c
        forward_loss = np.log(c)

        def alpha_bar(alpha, ind1, ind2):
            first = 0 if (ind2 - 1 < 0) else alpha[ind1, ind2 - 1]
            second = 0 if (ind2 - 1 < 0 or ind1 - 1 < 0) else alpha[ind1 - 1, ind2 - 1]
            return first + second

        # Recursion for forward Pass
        for t in range(1, T):
            for l in range(L):
                if l % 2 == 0 or (seq[l // 2] == seq[l // 2 - 1] if l > 1 else True):
                    paramInd = 0 if l % 2 == 0 else seq[l // 2]
                    alphas[l, t] = (alpha_bar(alphas, l, t)) * params[paramInd, t]
                else:
                    alphas[l, t] = (alpha_bar(alphas, l, t) +
                                    alphas[l - 2, t - 1]) * params[paramInd, t]
            c = np.sum(alphas[:, t])
            alphas[:, t] = alphas[:, t] / c
            forward_loss += np.log(c)
        return alphas, forward_loss

    @staticmethod
    def backward(params, seq):
        L = 2 * seq.shape[0] + 1  # Length of label sequence with blanks
        T = params.shape[1]  # Length of utterance (time)
        # Backward pass
        # Initialisation
        betas = np.zeros((L, T))
        betas[-1, -1] = params[0, -1]
        betas[-2, -1] = params[seq[-1], -1]
        # Rescaling for avoiding underflow
        d = np.sum(betas[:, -1])
        betas[:, -1] = betas[:, -1] / d
        backward_loss = np.log(d)

        def beta_bar(beta, ind1, ind2):
            first = 0 if (ind2 + 1 >= beta.shape[1]) else beta[ind1, ind2 + 1]
            second = 0 if (ind2 + 1 >= beta.shape[1] or ind1 + 1 >= beta.shape[0]) else beta[ind1 + 1, ind2 + 1]
            return first + second

        for t in range(T - 2, 0, -1):
            for l in range(L - 1, 0, -1):
                if l % 2 == 0 or (seq[l // 2] == seq[l // 2 + 1] if l < L - 2 else True):
                    paramInd = 0 if l % 2 == 0 else seq[l // 2]
                    betas[l, t] = (beta_bar(betas, l, t)) * params[paramInd, t]
                else:
                    betas[l, t] = (beta_bar(betas, l, t) + betas[l + 2, t + 1]) * params[paramInd, t]
            d = np.sum(betas[:, t])
            betas[:, t] = betas[:, t] / d
            backward_loss += np.log(d)
        return betas, backward_loss

    def ctc_grad(self, params, seq, alpha_beta):
        """Return Objective function derivatives with respect to the unnormalised outputs"""
        grad = np.zeros(self.params.shape)
        for i in range(alpha_beta.shape[0]):
            if i % 2 == 0:
                grad[0, :] += alpha_beta[i, :]
                alpha_beta[i, :] = alpha_beta[i, :] / params[0, :]
            else:
                grad[seq[i // 2], :] += alpha_beta[i, :]
                alpha_beta[i, :] = alpha_beta[i, :] / (params[seq[i // 2], :])
        alpha_beta_sum = np.sum(alpha_beta, axis=0)
        grad = params - grad / (params * alpha_beta_sum)
        return grad

    def forward_backward(self):
        """
        CTC loss function.
        params - n x m matrix of n-D probability distributions over m frames.
        seq - sequence of phone id's for given example.
        Returns objective and gradient.
        """
        alphas, forward_loss = self.forward(self.params, self.seq)
        betas, _ = self.backward(self.params, self.seq)
        alphaBeta = alphas * betas
        grad = self.ctc_grad(self.params, self.seq, alphaBeta)
        return (-forward_loss), grad

    def edit_distance(self, str1, str2):
        """Return the edit distance between two string"""
        if len(str1) == 0:
            return len(str2)

        if len(str2) == 0:
            return len(str1)

        if str1[0] == str2[0]:
            return self.edit_distance(str1[1:], str2[1:])
        else:
            return 1 + min(self.edit_distance(str1[1:], str2[1:]),
                           self.edit_distance(str1, str2[1:]),
                           self.edit_distance(str1[1:], str2))


