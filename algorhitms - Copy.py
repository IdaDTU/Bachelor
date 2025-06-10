import numpy as np
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.pyplot as plt 
from dictionaries import MWI,CIMR, OW_tiepoints
import numpy as np


def bristol(tb18v, tb37v, tb37h):
    # Water (OW)
    tw37v = OW_tiepoints["31.4V"]
    tw37h = OW_tiepoints["31.4H"]
    tw18v = OW_tiepoints["18.7V"]

    # FYI
    tfy37v = MWI["FYI"]["31.4V"]["tiepoint"]
    tfy37h = MWI["FYI"]["31.4H"]["tiepoint"]
    tfy18v = MWI["FYI"]["18.7V"]["tiepoint"]

    # MYI
    tmy37v = MWI["MYI"]["31.4V"]["tiepoint"]
    tmy37h = MWI["MYI"]["31.4H"]["tiepoint"]
    tmy18v = MWI["MYI"]["18.7V"]["tiepoint"]

    # Scalars for tie points
    xa = tmy37v + 1.045 * tmy37h + 0.525 * tmy18v
    xd = tfy37v + 1.045 * tfy37h + 0.525 * tfy18v
    xh = tw37v + 1.045 * tw37h + 0.525 * tw18v

    ya = 0.9164 * tmy18v - tmy37v + 0.4965 * tmy37h
    yd = 0.9164 * tfy18v - tfy37v + 0.4965 * tfy37h
    yh = 0.9164 * tw18v  - tw37v  + 0.4965 * tw37h

    # Observed maps
    xt = tb37v + 1.045 * tb37h + 0.525 * tb18v
    yt = 0.9164 * tb18v - tb37v + 0.4965 * tb37h

    with np.errstate(divide='ignore', invalid='ignore'):
        a_ht = (yt - yh) / (xt - xh)
        b_ht = yh - a_ht * xh
        a_da = (ya - yd) / (xa - xd) if xa != xd else np.nan
        b_da = yd - a_da * xd

        denom_intersection = a_ht - a_da
        xi = np.where(denom_intersection != 0, (b_da - b_ht) / denom_intersection, np.nan)

        denom_fraction = xi - xh
        cf = np.where(denom_fraction != 0, (xt - xh) / denom_fraction, 0.0)

        # Set any invalid (nan or inf) values explicitly to 0
        cf = np.where(np.isfinite(cf), cf, 0.0)

    return cf


def bristol_CIMR(tb18v, tb37v, tb37h):
    # Water (OW)
    tw37v = OW_tiepoints["36.5V"]
    tw37h = OW_tiepoints["36.5H"]
    tw18v = OW_tiepoints["18.7V"]

    # FYI
    tfy37v = CIMR["FYI"]["36.5V"]["tiepoint"]
    tfy37h = CIMR["FYI"]["36.5H"]["tiepoint"]
    tfy18v = CIMR["FYI"]["18.7V"]["tiepoint"]

    # MYI
    tmy37v = CIMR["MYI"]["36.5V"]["tiepoint"]
    tmy37h = CIMR["MYI"]["36.5H"]["tiepoint"]
    tmy18v = CIMR["MYI"]["18.7V"]["tiepoint"]

    # Scalars for tie points
    xa = tmy37v + 1.045 * tmy37h + 0.525 * tmy18v
    xd = tfy37v + 1.045 * tfy37h + 0.525 * tfy18v
    xh = tw37v + 1.045 * tw37h + 0.525 * tw18v

    ya = 0.9164 * tmy18v - tmy37v + 0.4965 * tmy37h
    yd = 0.9164 * tfy18v - tfy37v + 0.4965 * tfy37h
    yh = 0.9164 * tw18v  - tw37v  + 0.4965 * tw37h

    # Observed maps
    xt = tb37v + 1.045 * tb37h + 0.525 * tb18v
    yt = 0.9164 * tb18v - tb37v + 0.4965 * tb37h

    with np.errstate(divide='ignore', invalid='ignore'):
        a_ht = (yt - yh) / (xt - xh)
        b_ht = yh - a_ht * xh
        a_da = (ya - yd) / (xa - xd) if xa != xd else np.nan
        b_da = yd - a_da * xd

        denom_intersection = a_ht - a_da
        xi = np.where(denom_intersection != 0, (b_da - b_ht) / denom_intersection, np.nan)

        denom_fraction = xi - xh
        cf = np.where(denom_fraction != 0, (xt - xh) / denom_fraction, 0.0)

        # Set any invalid (nan or inf) values explicitly to 0
        cf = np.where(np.isfinite(cf), cf, 0.0)

    return cf


# -------------- snow-depth ------------------
# made from: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018JC014028

def snow_depth(tb7v, tb19v, sic, icetype):
    
    # skal tie-points være vertical også? Fra ivanova
    tp19v = 183.72
    tp7v = 161.35
    
    k1 = tp19v - tp7v 
    k2 = tp19v + tp7v
    
    GR = (tb19v - tb7v - k1*(1-sic)) / (tb19v + tb7v - k2*(1-sic))
    
    # linear coefficients determined in paper
    if icetype == 'FY':
        a = 19.26
        b = 553
    elif icetype == 'MY':
        a = 19.34
        b = 368
    else:
        print('Icetype must be either FY or MY')

    sd = a - b * GR
    
    return sd #in cm


# ----------- Snow/ice Interface Temperature --------------------
# https://ieeexplore.ieee.org/document/4510757

def compute_interface_temperature(TB_6V, Ti, SIC):
    """
    Compute ice temperature using the AMSR-E algorithm. Use nearly 100% ice

    Parameters:
    - TB_6V: Brightness temperature at 6 GHz vertical polarization (float or array)
    - SIC: Ice concentration (0 to 1, float or array)

    Returns:
    - T_i: Estimated ice temperature in Kelvin
    """

    T_w = 271.35
    eps6V = TB_6V / Ti    #exprected to bea round 0.95
    T_p = TB_6V / eps6V

    T_si = (T_p - T_w * (1 - SIC)) / SIC

    return T_si









    
    
    




