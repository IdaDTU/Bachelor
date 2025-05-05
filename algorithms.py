import numpy as np

#tp = {'fy18v': 218.77, 'my18v': -, 'ow18v': 183.72,
 #     'fy37v': 210.55, 'my37v': -, 'ow37v': 209.81,
  #    'fy37h': 189.33, 'my37h': 190.22, 'ow37h': 145.29}

# ------------------ SIC -------------------------

def bristol(tb18v, tb37v, tb37h):
    """
    Compute sea ice concentration using the Bristol algorithm.
    Based on Ivanova et al. (2015) SH FY tiepoints.
    
    Parameters:
        tb18v : array-like
            Brightness temperatures at 18 GHz vertical polarization
        tb37v : array-like
            Brightness temperatures at 37 GHz vertical polarization
        tb37h : array-like
            Brightness temperatures at 37 GHz horizontal polarization

    Returns:
        Sea ice concentration (array-like, clipped to [0,1])
    """
    # updated to the northen hemisphere - WE NEED OUR OWN TIEPOINTS
    tp = {'fy18v': 218.77, 'my18v': 226.26, 'ow18v': 183.72,
          'fy37v': 210.55, 'my37v': 196.91, 'ow37v': 209.81,
          'fy37h': 189.33, 'my37h': 190.22, 'ow37h': 145.29}
    
    # Tie-points
    tw18v, tw37v, tw37h = tp['ow18v'], tp['ow37v'], tp['ow37h']
    tfy18v, tfy37v, tfy37h = tp['fy18v'], tp['fy37v'], tp['fy37h']
    tmy18v, tmy37v, tmy37h = tp['my18v'], tp['my37v'], tp['my37h']

    # Calculate intermediate values
    XA = tmy37v + (1.045 * tmy37h) + (0.525 * tmy18v)
    XD = tfy37v + (1.045 * tfy37h) + (0.525 * tfy18v)
    XH = tw37v + (1.045 * tw37h) + (0.525 * tw18v)
    XT = tb37v + (1.045 * tb37h) + (0.525 * tb18v)

    YA = (0.9164 * tmy18v) - tmy37v + (0.4965 * tmy37h)
    YD = (0.9164 * tfy18v) - tfy37v + (0.4965 * tfy37h)
    YH = (0.9164 * tw18v) - tw37v + (0.4965 * tw37h)
    YT = (0.9164 * tb18v) - tb37v + (0.4965 * tb37h)

    # Line equations
    A_HT = (YT - YH) / (XT - XH)
    B_HT = YH - (A_HT * XH)

    A_DA = (YA - YD) / (XA - XD)
    B_DA = YD - (A_DA * XD)

    XI = (B_DA - B_HT) / (A_HT - A_DA)
    CF = (XT - XH) / (XI - XH)

    # Clip concentration to [0, 1]
    C = np.clip(CF, 0.0, 1.0)

    return C

def tud(tb18v, tb37v, tb85v, tb85h):
# bootstrap winter tie points (Comiso, 1997)
# (ice tie points represent slopes and offsets only)
#Ivanova et al. 2015 SH FY tiepoints
    # updated to northern hemisphere - WE NEED OUR OWN TIEPOINTS
    tp = {'fy18v': 218.77, 'my18v': 226.26, 'ow18v': 183.72,
          'fy37v': 210.55, 'my37v': 196.91, 'ow37v': 209.81,
          'fy37h': 189.33, 'my37h': 190.22, 'ow37h': 145.29}

    tw18v = tp['ow18v']
    tw37h = tp['ow37h']
    tw37v = tp['ow37v']

    tfy18v= tp['fy18v']
    tfy37h= tp['fy37h']
    tfy37v= tp['fy37v']

    tmy18v= tp['my18v']
    tmy37h= tp['my37h']
    tmy37v= tp['my37v']
        
    # WHERE DOES THESE COME FROM?
    a1 = 1.35
    a2 = -1.0/40.0
    a3 = -0.03

    af = (tfy37v - tmy37v)/(tfy18v - tmy18v)
    bf = (tmy37v - af*tmy18v)
    qf = (tb37v - tw37v)/(tb18v - tw18v)
    wf = (tw37v - qf*tw18v)
    ti18vf = (bf - wf)/(qf - af)
    cf = (tb18v - tw18v)/(ti18vf - tw18v)
    c=cf
    c85 = a1 + (tb85v - tb85h)*a2
    c=cf*c85
    idx=(cf < 0.0)
    c[idx]=0.0
    ct = np.sqrt(c)+a3
    return ct

def esmr_sic(tb19h):
    tp = {'fy18v': 218.77, 'ow18v': 183.72,
          'fy18h': 237.54, 'ow18h':108.46,
          'fy37v': 210.55, 'ow37v': 209.81,
          'fy37h': 189.33, 'ow37h': 145.29}
    tw19h = tp['ow18h']
    tfy19h = tp['fy18h']
    sic=(tb19h-tw19h)/(tfy19h-tw19h)
    return sic

# -------------- snow-density ------------------
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








