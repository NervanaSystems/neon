"""
This is a reference LSTM numpy implementation adapted from Karpathy's code:

The adaptation includes
  - interface to use the same initialization values
  - being able to read out intermediate values to compare with another LSTM
    implementation
"""
from builtins import input
import numpy as np
from neon import logger as neon_logger
from utils import allclose_with_out


class LSTM(object):

    @staticmethod
    def init(input_size, hidden_size):
        """
        Initialize parameters of the LSTM (both weights and biases in one matrix)
        to be ones
        """
        a = input_size + hidden_size + 1
        b = 4 * hidden_size
        # c = np.sqrt(input_size + hidden_size)
        WLSTM = np.ones((a, b))
        return WLSTM

    @staticmethod
    def forward(X, WLSTM, c0=None, h0=None):
        """
        X should be of shape (n,b,input_size), where n = length of sequence, b = batch size
        """
        n, b, input_size = X.shape
        d = WLSTM.shape[1] // 4  # hidden size
        if c0 is None:
            c0 = np.zeros((b, d))
        if h0 is None:
            h0 = np.zeros((b, d))

        # Perform the LSTM forward pass with X as the input
        xphpb = WLSTM.shape[0]  # x plus h plus bias, lol
        # input [1, xt, ht-1] to each tick of the LSTM
        Hin = np.zeros((n, b, xphpb))
        # hidden representation of the LSTM (gated cell content)
        Hout = np.zeros((n, b, d))
        IFOG = np.zeros((n, b, d * 4))  # input, forget, output, gate (IFOG)
        IFOGf = np.zeros((n, b, d * 4))  # after nonlinearity
        C = np.zeros((n, b, d))  # cell content
        Ct = np.zeros((n, b, d))  # tanh of cell content
        for t in range(n):
            # concat [x,h] as input to the LSTM
            prevh = Hout[t - 1] if t > 0 else h0
            Hin[t, :, 0] = 1  # bias
            Hin[t, :, 1:input_size + 1] = X[t]
            Hin[t, :, input_size + 1:] = prevh
            # compute all gate activations. dots: (most work is this line)

            IFOG[t] = Hin[t].dot(WLSTM)
            # non-linearities
            # sigmoids; these are the gates
            IFOGf[t, :, :3 * d] = 1.0 / (1.0 + np.exp(-IFOG[t, :, :3 * d]))
            IFOGf[t, :, 3 * d:] = np.tanh(IFOG[t, :, 3 * d:])  # tanh
            # compute the cell activation
            prevc = C[t - 1] if t > 0 else c0
            C[t] = IFOGf[t, :, :d] * IFOGf[t, :, 3 * d:] + \
                IFOGf[t, :, d:2 * d] * prevc
            Ct[t] = np.tanh(C[t])
            Hout[t] = IFOGf[t, :, 2 * d:3 * d] * Ct[t]

        cache = {}
        cache['WLSTM'] = WLSTM
        cache['Hout'] = Hout
        cache['IFOGf'] = IFOGf
        cache['IFOG'] = IFOG
        cache['C'] = C
        cache['Ct'] = Ct
        cache['Hin'] = Hin
        cache['c0'] = c0
        cache['h0'] = h0

        # return C[t], as well so we can continue LSTM with prev state init if
        # needed
        return Hout, C[t], Hout[t], cache

    @staticmethod
    def backward(dHout_in, cache, dcn=None, dhn=None):

        WLSTM = cache['WLSTM']
        Hout = cache['Hout']
        IFOGf = cache['IFOGf']
        IFOG = cache['IFOG']
        C = cache['C']
        Ct = cache['Ct']
        Hin = cache['Hin']
        c0 = cache['c0']
        # h0 = cache['h0']
        n, b, d = Hout.shape
        input_size = WLSTM.shape[0] - d - 1  # -1 due to bias

        # backprop the LSTM
        dIFOG = np.zeros(IFOG.shape)
        dIFOGf = np.zeros(IFOGf.shape)
        dWLSTM = np.zeros(WLSTM.shape)
        dHin = np.zeros(Hin.shape)
        dC = np.zeros(C.shape)
        dX = np.zeros((n, b, input_size))
        dh0 = np.zeros((b, d))
        dc0 = np.zeros((b, d))
        dHout = dHout_in.copy()  # make a copy so we don't have any funny side effects
        if dcn is not None:
            dC[n - 1] += dcn.copy()  # carry over gradients from later
        if dhn is not None:
            dHout[n - 1] += dhn.copy()

        for t in reversed(range(n)):
            tanhCt = Ct[t]
            dIFOGf[t, :, 2 * d:3 * d] = tanhCt * dHout[t]
            # backprop tanh non-linearity first then continue backprop
            dC[t] += (1 - tanhCt ** 2) * (IFOGf[t, :, 2 * d:3 * d] * dHout[t])

            if t > 0:
                dIFOGf[t, :, d:2 * d] = C[t - 1] * dC[t]
                dC[t - 1] += IFOGf[t, :, d:2 * d] * dC[t]
            else:
                dIFOGf[t, :, d:2 * d] = c0 * dC[t]
                dc0 = IFOGf[t, :, d:2 * d] * dC[t]
            dIFOGf[t, :, :d] = IFOGf[t, :, 3 * d:] * dC[t]
            dIFOGf[t, :, 3 * d:] = IFOGf[t, :, :d] * dC[t]

            # backprop activation functions
            dIFOG[t, :, 3 * d:] = (1 - IFOGf[t, :, 3 * d:] **
                                   2) * dIFOGf[t, :, 3 * d:]
            y = IFOGf[t, :, :3 * d]
            dIFOG[t, :, :3 * d] = (y * (1.0 - y)) * dIFOGf[t, :, :3 * d]

            # backprop matrix multiply
            dWLSTM += np.dot(Hin[t].transpose(), dIFOG[t])
            dHin[t] = dIFOG[t].dot(WLSTM.transpose())

            # backprop the identity transforms into Hin
            dX[t] = dHin[t, :, 1:input_size + 1]
            if t > 0:
                dHout[t - 1, :] += dHin[t, :, input_size + 1:]
            else:
                dh0 += dHin[t, :, input_size + 1:]

            # for debugging

            # hidden_size = WLSTM.shape[0] - input_size - 1
            # dWrecur = dWLSTM[-hidden_size:, :]
            # dWinput = dWLSTM[1:input_size + 1, :]
            # db = dWLSTM[0, :]
        return dX, dWLSTM, dc0, dh0

    @staticmethod
    def runBatchFpropWithGivenInput(hidden_size, X):
        """
        run the LSTM model through the given input data. The data has dimension
        (seq_len, batch_size, hidden_size)

        """
        # seq_len = X.shape[0]
        # batch_size = X.shape[1]
        input_size = X.shape[2]

        WLSTM = LSTM.init(input_size, hidden_size)

        # batch forward
        Hout, cprev, hprev, batch_cache = LSTM.forward(X, WLSTM)

        IFOGf = batch_cache['IFOGf']
        Ct = batch_cache['Ct']

        return Hout, IFOGf, Ct, batch_cache

    @staticmethod
    def runBatchBpropWithGivenDelta(hidden_size, batch_cache, delta):
        """
        run the LSTM model through the given input errors. The data has dimension
        (seq_len, batch_size, hidden_size)

        """
        dH = delta

        # get the batched version gradients
        dX, dWLSTM, dc0, dh0 = LSTM.backward(dH, batch_cache)

        input_size = dWLSTM.shape[0] - hidden_size - 1
        dWrecur = dWLSTM[-hidden_size:, :]
        dWinput = dWLSTM[1:input_size + 1, :]
        db = dWLSTM[0, :]

        return dX, dWrecur, dWinput, db, dWLSTM

# -------------------
# TEST CASES
# -------------------


def checkSequentialMatchesBatch():
    """ check LSTM I/O forward/backward interactions """

    n, b, d = (5, 3, 4)  # sequence length, batch size, hidden size
    input_size = 10
    WLSTM = LSTM.init(input_size, d)  # input size, hidden size
    X = np.random.randn(n, b, input_size)
    h0 = np.random.randn(b, d)
    c0 = np.random.randn(b, d)

    # sequential forward
    cprev = c0
    hprev = h0
    caches = [{} for t in range(n)]
    Hcat = np.zeros((n, b, d))
    for t in range(n):
        xt = X[t:t + 1]
        _, cprev, hprev, cache = LSTM.forward(xt, WLSTM, cprev, hprev)
        caches[t] = cache
        Hcat[t] = hprev

    # sanity check: perform batch forward to check that we get the same thing
    H, _, _, batch_cache = LSTM.forward(X, WLSTM, c0, h0)
    assert allclose_with_out(H, Hcat), 'Sequential and Batch forward don''t match!'

    # eval loss
    wrand = np.random.randn(*Hcat.shape)
    # loss = np.sum(Hcat * wrand)
    dH = wrand

    # get the batched version gradients
    BdX, BdWLSTM, Bdc0, Bdh0 = LSTM.backward(dH, batch_cache)

    # now perform sequential backward
    dX = np.zeros_like(X)
    dWLSTM = np.zeros_like(WLSTM)
    dc0 = np.zeros_like(c0)
    dh0 = np.zeros_like(h0)
    dcnext = None
    dhnext = None
    for t in reversed(range(n)):
        dht = dH[t].reshape(1, b, d)
        dx, dWLSTMt, dcprev, dhprev = LSTM.backward(
            dht, caches[t], dcnext, dhnext)
        dhnext = dhprev
        dcnext = dcprev

        dWLSTM += dWLSTMt  # accumulate LSTM gradient
        dX[t] = dx[0]
        if t == 0:
            dc0 = dcprev
            dh0 = dhprev

    # and make sure the gradients match
    neon_logger.display('Making sure batched version agrees with sequential version: '
                        '(should all be True)')
    neon_logger.display(np.allclose(BdX, dX))
    neon_logger.display(np.allclose(BdWLSTM, dWLSTM))
    neon_logger.display(np.allclose(Bdc0, dc0))
    neon_logger.display(np.allclose(Bdh0, dh0))


def checkBatchGradient():
    """ check that the batch gradient is correct """

    # lets gradient check this beast
    n, b, d = (5, 3, 4)  # sequence length, batch size, hidden size
    input_size = 10
    WLSTM = LSTM.init(input_size, d)  # input size, hidden size
    X = np.random.randn(n, b, input_size)
    h0 = np.random.randn(b, d)
    c0 = np.random.randn(b, d)

    # batch forward backward
    H, Ct, Ht, cache = LSTM.forward(X, WLSTM, c0, h0)
    wrand = np.random.randn(*H.shape)
    # loss = np.sum(H * wrand)  # weighted sum is a nice hash to use I think
    dH = wrand
    dX, dWLSTM, dc0, dh0 = LSTM.backward(dH, cache)

    def fwd():
        h, _, _, _ = LSTM.forward(X, WLSTM, c0, h0)
        return np.sum(h * wrand)

    # now gradient check all
    delta = 1e-5
    rel_error_thr_warning = 1e-2
    rel_error_thr_error = 1
    tocheck = [X, WLSTM, c0, h0]
    grads_analytic = [dX, dWLSTM, dc0, dh0]
    names = ['X', 'WLSTM', 'c0', 'h0']
    for j in range(len(tocheck)):
        mat = tocheck[j]
        dmat = grads_analytic[j]
        name = names[j]
        # gradcheck
        for i in range(mat.size):
            old_val = mat.flat[i]
            mat.flat[i] = old_val + delta
            loss0 = fwd()
            mat.flat[i] = old_val - delta
            loss1 = fwd()
            mat.flat[i] = old_val

            grad_analytic = dmat.flat[i]
            grad_numerical = (loss0 - loss1) / float(2 * delta)

            if grad_numerical == 0 and grad_analytic == 0:
                rel_error = 0  # both are zero, OK.
                status = 'OK'
            elif abs(grad_numerical) < 1e-7 and abs(grad_analytic) < 1e-7:
                rel_error = 0  # not enough precision to check this
                status = 'VAL SMALL WARNING'
            else:
                rel_error = (abs(grad_analytic - grad_numerical) /
                             abs(float(grad_numerical + grad_analytic)))
                status = 'OK'
                if rel_error > rel_error_thr_warning:
                    status = 'WARNING'
                if rel_error > rel_error_thr_error:
                    status = '!!!!! NOTOK'

            # print stats
            neon_logger.display('%s checking param %s index %s (val = %+8f), analytic = %+8f,' +
                                'numerical = %+8f, relative error = %+8f'
                                % (status, name, repr(np.unravel_index(i, mat.shape)), old_val,
                                   grad_analytic, grad_numerical, rel_error))

if __name__ == "__main__":
    checkSequentialMatchesBatch()
    input('check OK, press key to continue to gradient check')
    checkBatchGradient()
    neon_logger.display('every line should start with OK. Have a nice day!')
