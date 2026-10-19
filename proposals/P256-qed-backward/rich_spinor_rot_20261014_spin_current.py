"""Physical charged-spin flux of the frozen rotational/cavity candidate.
Primitive spin transfers torque to orientor; this is NOT a spin Ward identity.
Use the same solved/built model as full-current/contact/scattering consumers.
"""
from rich_spinor_rot_20261014_model import current


def spin_current(model,q,epsilon,axis):
    """{S_axis,J_physical(q,epsilon)}/2 including actual fieldfeedback current."""
    s=model['spin'][axis]
    j=current(model,q,epsilon)
    return (s@j+j@s)/2
