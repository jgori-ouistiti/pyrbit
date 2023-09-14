
from pyrbit.actr import (actr_dq1_ds_sample, actr_dq1_dtau_sample, actr_dq1_dd_sample, actr_dq0_ds_sample, actr_dq0_dtau_sample, actr_dq0_dd_sample,actr_ddq1_ds_ds_sample, actr_ddq1_dtau_dtau_sample,actr_ddq1_dd_dd_sample, actr_ddq1_ds_dtau_sample, actr_ddq1_dd_dtau_sample,actr_ddq1_dd_ds_sample,actr_ddq0_ds_ds_sample,actr_ddq0_dtau_dtau_sample,actr_ddq0_dd_dd_sample,actr_ddq0_ds_dtau_sample,actr_ddq0_dd_dtau_sample,actr_ddq0_dd_ds_sample, da_dd, dda_ddd)

from sympy import symbols, exp, log, diff, lambdify


s, tau, d, t1,t2,t3 = symbols('s tau d t1 t2 t3')


q_1 = -log(1+exp((-(log(t1**(-d) + t2**(-d) + t3**(-d))-tau)/s)))
a = log(t1**(-d) + t2**(-d) + t3**(-d))
q_0 = -log(1+exp(((log(t1**(-d) + t2**(-d) + t3**(-d))-tau)/s)))
a = log(t1**(-d) + t2**(-d) + t3**(-d))

# First order

_da_dd = lambdify((s, tau, d, t1,t2,t3), diff(a, d), 'numpy')
dq1_ds = lambdify((s, tau, d, t1,t2,t3), diff(q_1, s),'numpy')
dq1_dtau = lambdify((s, tau, d, t1,t2,t3), diff(q_1, tau), 'numpy')
dq1_dd = lambdify((s, tau, d, t1,t2,t3), diff(q_1, d), 'numpy')
dq0_ds = lambdify((s, tau, d, t1,t2,t3), diff(q_0, s), 'numpy')
dq0_dtau = lambdify((s, tau, d, t1,t2,t3), diff(q_0, tau), 'numpy')
dq0_dd = lambdify((s, tau, d, t1,t2,t3), diff(q_0, d), 'numpy')

# Second order
_dda_ddd = lambdify((s, tau, d, t1,t2,t3), diff(diff(a, d),d), 'numpy')

ddq1_dss = lambdify((s, tau, d, t1,t2,t3), diff(diff(q_1, s),s), 'numpy')
ddq1_dtautau = lambdify((s, tau, d, t1,t2,t3), diff(diff(q_1, tau),tau), 'numpy')
ddq1_ddd = lambdify((s, tau, d, t1,t2,t3), diff(diff(q_1, d),d), 'numpy')
ddq1_ddtau = lambdify((s, tau, d, t1,t2,t3), diff(diff(q_1, d),tau), 'numpy')
ddq1_dds = lambdify((s, tau, d, t1,t2,t3), diff(diff(q_1, d),s), 'numpy')
ddq1_dstau = lambdify((s, tau, d, t1,t2,t3), diff(diff(q_1, s),tau), 'numpy')

ddq0_dss = lambdify((s, tau, d, t1,t2,t3), diff(diff(q_0, s),s), 'numpy')
ddq0_dtautau = lambdify((s, tau, d, t1,t2,t3), diff(diff(q_0, tau),tau), 'numpy')
ddq0_ddd = lambdify((s, tau, d, t1,t2,t3), diff(diff(q_0, d),d), 'numpy')
ddq0_ddtau = lambdify((s, tau, d, t1,t2,t3), diff(diff(q_0, d),tau), 'numpy')
ddq0_dds = lambdify((s, tau, d, t1,t2,t3), diff(diff(q_0, d),s), 'numpy')
ddq0_dstau = lambdify((s, tau, d, t1,t2,t3), diff(diff(q_0, s),tau), 'numpy')


if __name__ == '__main__':
    deltati = [9.87, 5.84, 2.54]
    d = 0.5
    tau = 0.1
    s = 1.3


    
    ## ===================== First order
    assert abs(da_dd(deltati,d) - _da_dd(s, tau, d, *deltati)) < 1e-6

    # omega = 1
    # dq1_ds
    assert abs(actr_dq1_ds_sample(tau, s, d, deltati) - dq1_ds(s, tau, d,*deltati))< 1e-6
     #dq1_dtau
    assert abs(actr_dq1_dtau_sample(tau, s, d, deltati) - dq1_dtau(s, tau, d, *deltati))< 1e-6
    #dq1_dd
    assert abs(actr_dq1_dd_sample(tau, s, d, deltati) - dq1_dd(s, tau, d, *deltati))< 1e-6

    # omega = 0
    # dq0_ds
    assert abs(actr_dq0_ds_sample(tau, s, d, deltati) - dq0_ds(s, tau, d, *deltati))< 1e-6
    #dq1_dtau
    assert abs(actr_dq0_dtau_sample(tau, s, d, deltati) - dq0_dtau(s, tau, d, *deltati))< 1e-6
    #dq1_dd
    assert abs(actr_dq0_dd_sample(tau, s, d, deltati) - dq0_dd(s, tau, d, *deltati))< 1e-6

    ## ==================== Second order

    assert abs(dda_ddd(deltati, d) - _dda_ddd(s, tau, d, *deltati)) < 1e-6

    #ddq1_dss
    assert abs(actr_ddq1_ds_ds_sample(tau, s, d, deltati) - ddq1_dss(s, tau, d, *deltati))< 1e-6
    #ddq1_dtautau
    assert abs(actr_ddq1_dtau_dtau_sample(tau, s, d, deltati) - ddq1_dtautau(s, tau, d, *deltati))< 1e-6
    #ddq1_ddd
    assert abs(actr_ddq1_dd_dd_sample(tau, s, d, deltati) - ddq1_ddd(s, tau, d, *deltati))< 1e-6
    #ddq1_dstau
    assert abs(actr_ddq1_ds_dtau_sample(tau, s, d, deltati) - ddq1_dstau(s, tau, d, *deltati))< 1e-6
    #ddq1_ddtau
    assert abs(actr_ddq1_dd_dtau_sample(tau, s, d, deltati) - ddq1_ddtau(s, tau, d, *deltati))< 1e-6
    #ddq1_ddd
    assert abs(actr_ddq1_dd_ds_sample(tau, s, d, deltati) - ddq1_dds(s, tau, d, *deltati))< 1e-6


    #ddq0_dss
    assert abs(actr_ddq0_ds_ds_sample(tau, s, d, deltati) - ddq0_dss(s, tau, d, *deltati))< 1e-6
    #ddq0_dtautau
    assert abs(actr_ddq0_dtau_dtau_sample(tau, s, d, deltati) - ddq0_dtautau(s, tau, d, *deltati))< 1e-6
    #ddq0_ddd
    assert abs(actr_ddq0_dd_dd_sample(tau, s, d, deltati) - ddq0_ddd(s, tau, d, *deltati))< 1e-6
    #ddq0_dstau
    assert abs(actr_ddq0_ds_dtau_sample(tau, s, d, deltati) - ddq0_dstau(s, tau, d, *deltati))< 1e-6
    #ddq0_ddtau
    assert abs(actr_ddq0_dd_dtau_sample(tau, s, d, deltati) - ddq0_ddtau(s, tau, d, *deltati))< 1e-6
    #ddq0_ddd
    assert abs(actr_ddq0_dd_ds_sample(tau, s, d, deltati) - ddq0_dds(s, tau, d, *deltati))< 1e-6



