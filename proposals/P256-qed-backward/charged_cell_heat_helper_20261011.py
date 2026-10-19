"""Pure exact/outward whole-spectrum charged heat evaluation. No solver."""
from fractions import Fraction as F
import charged_output_lower_certificate_20261011 as iv

Z=iv.iv(0)
ONE=iv.iv(1)

def expminus(x):
    """Rational alternating Taylor brackets, range reduction, outward squaring."""
    x=F(x)
    if x<0: raise ValueError('nonnegative argument required')
    k=0
    while x>F(1,8): x/=2; k+=1
    term=F(1); total=term
    for j in range(1,42):
        term*=-x/j; total+=term
        if j==40: upper=total
    out=(iv.iv(total)[0],iv.iv(upper)[1])
    for _ in range(k): out=iv.mul(out,out)
    return out

def decode(entry):
    return tuple((iv.iv(F(entry[key][0]))[0],iv.iv(F(entry[key][1]))[1]) for key in ('re','im'))

def encode(z):
    return {'re':iv.exact_endpoints(z[0]),'im':iv.exact_endpoints(z[1])}

def cmulreal(z,r): return iv.mul(z[0],r),iv.mul(z[1],r)
def cadd(a,b): return iv.add(a[0],b[0]),iv.add(a[1],b[1])
def widen(a,r): return a[0]-r[1],a[1]+r[1]
def intersect(a,b):
    c=max(a[0],b[0]),min(a[1],b[1])
    if c[0]>c[1]: raise ArithmeticError('empty rigorous intersection')
    return c

def evaluate(interface,role,orientation,t,method='floor_hermite'):
    """Return actual native G[q,p] for creation; annihilation is transpose Gram."""
    t=F(t)
    if t<0: raise ValueError('heat time must be nonnegative')
    row=next(r for r in interface['roles'] if r['role']==role and r['orientation']==orientation)
    M=[[[decode(v) for v in rr] for rr in row['moments'][name]] for name in ('M0','M1','M2')]
    b=F(interface['charged_floor_exact']); decay=expminus(t*b)
    eta=(iv.iv(F(interface['ground_distance_exact'][0]))[0],iv.iv(F(interface['ground_distance_exact'][1]))[1])
    price=iv.scale(iv.mul(eta,decay),2)
    coarse=decay[1]
    result=[[( (0,coarse), Z) if i==j else ((-coarse//2,(coarse+1)//2),(-coarse//2,(coarse+1)//2)) for j in range(2)] for i in range(2)]
    trace0=sum(F(z[0][0]+z[0][1],2*iv.Q) for z in (M[0][0][0],M[0][1][1]))
    trace1=sum(F(z[0][0]+z[0][1],2*iv.Q) for z in (M[1][0][0],M[1][1][1]))
    center=trace1/trace0
    anchors=[max(b+1,center*f) for f in (F(1,2),F(1),F(2))]
    if method=='baseline_quadratic': anchors=[max(b+1,center)]
    elif method!='floor_hermite': raise ValueError('unknown global majorant')
    if t==0:
        for i in range(2):
            for j in range(2):
                result[i][j]=(intersect(result[i][j][0],widen(M[0][i][j][0],price)),intersect(result[i][j][1],widen(M[0][i][j][1],price)))
    else:
        for a in anchors:
            ea=expminus(t*a)
            # The second divided difference of exp(-tx) is decreasing in x.
            numerator=iv.sub(decay,iv.mul(ea,iv.iv(1+t*(a-b))))
            coeff=iv.mul(numerator,iv.iv(1/(a-b)**2))
            coeff=(max(0,coeff[0]),max(0,coeff[1]))
            if method=='baseline_quadratic': coeff=iv.mul(decay,iv.iv(t*t/2))
            L=[[cadd(cmulreal(M[0][i][j],iv.mul(ea,iv.iv(1+t*a))),cmulreal(M[1][i][j],iv.mul(ea,iv.iv(-t)))) for j in range(2)] for i in range(2)]
            V=[[cadd(cadd(M[2][i][j],cmulreal(M[1][i][j],iv.iv(-2*a))),cmulreal(M[0][i][j],iv.iv(a*a))) for j in range(2)] for i in range(2)]
            D=[[cmulreal(V[i][j],coeff) for j in range(2)] for i in range(2)]
            diag=[(0,max(0,D[i][i][0][1])) for i in range(2)]
            for i in range(2):
                for j in range(2):
                    if i==j:
                        bound=(L[i][i][0][0],L[i][i][0][1]+diag[i][1])
                        candidates=(widen(bound,price),Z)
                    else:
                        mid=cadd(L[i][j],cmulreal(D[i][j],iv.iv(F(1,2))))
                        radius=iv.add(iv.mul(iv.sqrt(iv.mul(diag[i],diag[j])),iv.iv(F(1,2))),price)
                        candidates=tuple(widen(v,radius) for v in mid)
                    result[i][j]=(intersect(result[i][j][0],candidates[0]),intersect(result[i][j][1],candidates[1]))
    if orientation=='annihilation': result=[[result[j][i] for j in range(2)] for i in range(2)]
    return [[encode(v) for v in rr] for rr in result]

def derivative_bound(interface,role,orientation,t,order=1):
    """Operator norm upper; first/second derivatives also valid at time zero."""
    t=F(t)
    if t<0 or order not in (0,1,2): raise ValueError('order must be 0,1,2; time nonnegative')
    row=next(r for r in interface['roles'] if r['role']==role and r['orientation']==orientation)
    cap=iv.iv(F(row['ground_second_moment_trace_upper_exact']))
    if order==0: cap=ONE
    elif order==1: cap=iv.sqrt(cap)
    bound=iv.mul(cap,expminus(t*F(interface['charged_floor_exact'])))
    return iv.exact_endpoints((0,bound[1]))

def tail_bound(interface,T):
    """Integral_T^infinity ||G(t)||dt <= exp(-bT)/b."""
    b=F(interface['charged_floor_exact']);T=F(T)
    if T<0: raise ValueError('nonnegative tail time required')
    value=iv.mul(expminus(b*T),iv.iv(1/b))
    return iv.exact_endpoints((0,value[1]))
