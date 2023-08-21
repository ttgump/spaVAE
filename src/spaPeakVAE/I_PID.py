from math import exp

class PIDControl():
    """incremental PID controller"""
    def __init__(self, Kp, Ki, init_beta, min_beta, max_beta):
        """define them out of loop"""
        self.W_k1 = init_beta
        self.W_min = min_beta
        self.W_max = max_beta
        self.e_k1 = 0.0
        self.Kp = Kp
        self.Ki = Ki

    def _Kp_fun(self, Err, scale=1):
        return 1.0/(1.0 + float(scale)*exp(Err))

    def pid(self, exp_KL, kl_loss):
        """
        Incremental PID algorithm
        Input: KL_loss
        return: weight for KL divergence, beta
        """
        error_k = (exp_KL - kl_loss) * 5.   # we enlarge the error 5 times to allow faster tuning of beta
        ## comput U as the control factor
        dP = self.Kp * (self._Kp_fun(error_k) - self._Kp_fun(self.e_k1))
        dI = self.Ki * error_k

        if self.W_k1 < self.W_min:
            dI = 0
        dW = dP + dI
        ## update with previous W_k1
        Wk = dW + self.W_k1
        self.W_k1 = Wk
        self.e_k1 = error_k

        ## min and max value
        if Wk < self.W_min:
            Wk = self.W_min
        if Wk > self.W_max:
            Wk = self.W_max

        return Wk, error_k
