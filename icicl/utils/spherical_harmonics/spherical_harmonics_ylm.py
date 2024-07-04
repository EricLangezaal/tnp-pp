"""
analytic expressions of spherical harmonics generated with sympy file
Marc Russwurm generated 2024-07-04

run
python spherical_harmonics_generate_ylms.py > spherical_harmonics_ylm.py

to generate the source code
"""

# pylint: skip-file

import torch
from torch import cos, sin


def get_spherical_harmonics(m: int, l: int):
    fname = f"Yl{l}_m{m}".replace("-", "_minus_")
    return globals()[fname]


def spherical_harmonics(
    m: int, l: int, phi: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    ylm = get_spherical_harmonics(m, l)
    return ylm(theta, phi)


@torch.jit.script
def Yl0_m0(theta: torch.Tensor, phi: torch.Tensor):
    return 0.282094791773878


@torch.jit.script
def Yl1_m_minus_1(theta: torch.Tensor, phi: torch.Tensor):
    return 0.48860251190292 * (1.0 - cos(theta) ** 2) ** 0.5 * sin(phi)


@torch.jit.script
def Yl1_m0(theta: torch.Tensor, phi: torch.Tensor):
    return 0.48860251190292 * cos(theta)


@torch.jit.script
def Yl1_m1(theta: torch.Tensor, phi: torch.Tensor):
    return 0.48860251190292 * (1.0 - cos(theta) ** 2) ** 0.5 * cos(phi)


@torch.jit.script
def Yl2_m_minus_2(theta: torch.Tensor, phi: torch.Tensor):
    return 0.18209140509868 * (3.0 - 3.0 * cos(theta) ** 2) * sin(2 * phi)


@torch.jit.script
def Yl2_m_minus_1(theta: torch.Tensor, phi: torch.Tensor):
    return 1.09254843059208 * (1.0 - cos(theta) ** 2) ** 0.5 * sin(phi) * cos(theta)


@torch.jit.script
def Yl2_m0(theta: torch.Tensor, phi: torch.Tensor):
    return 0.94617469575756 * cos(theta) ** 2 - 0.31539156525252


@torch.jit.script
def Yl2_m1(theta: torch.Tensor, phi: torch.Tensor):
    return 1.09254843059208 * (1.0 - cos(theta) ** 2) ** 0.5 * cos(phi) * cos(theta)


@torch.jit.script
def Yl2_m2(theta: torch.Tensor, phi: torch.Tensor):
    return 0.18209140509868 * (3.0 - 3.0 * cos(theta) ** 2) * cos(2 * phi)


@torch.jit.script
def Yl3_m_minus_3(theta: torch.Tensor, phi: torch.Tensor):
    return 0.590043589926644 * (1.0 - cos(theta) ** 2) ** 1.5 * sin(3 * phi)


@torch.jit.script
def Yl3_m_minus_2(theta: torch.Tensor, phi: torch.Tensor):
    return 1.44530572132028 * (1.0 - cos(theta) ** 2) * sin(2 * phi) * cos(theta)


@torch.jit.script
def Yl3_m_minus_1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.304697199642977
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (7.5 * cos(theta) ** 2 - 1.5)
        * sin(phi)
    )


@torch.jit.script
def Yl3_m0(theta: torch.Tensor, phi: torch.Tensor):
    return 1.86588166295058 * cos(theta) ** 3 - 1.11952899777035 * cos(theta)


@torch.jit.script
def Yl3_m1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.304697199642977
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (7.5 * cos(theta) ** 2 - 1.5)
        * cos(phi)
    )


@torch.jit.script
def Yl3_m2(theta: torch.Tensor, phi: torch.Tensor):
    return 1.44530572132028 * (1.0 - cos(theta) ** 2) * cos(2 * phi) * cos(theta)


@torch.jit.script
def Yl3_m3(theta: torch.Tensor, phi: torch.Tensor):
    return 0.590043589926644 * (1.0 - cos(theta) ** 2) ** 1.5 * cos(3 * phi)


@torch.jit.script
def Yl4_m_minus_4(theta: torch.Tensor, phi: torch.Tensor):
    return 0.625835735449176 * (1.0 - cos(theta) ** 2) ** 2 * sin(4 * phi)


@torch.jit.script
def Yl4_m_minus_3(theta: torch.Tensor, phi: torch.Tensor):
    return 1.77013076977993 * (1.0 - cos(theta) ** 2) ** 1.5 * sin(3 * phi) * cos(theta)


@torch.jit.script
def Yl4_m_minus_2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.063078313050504
        * (1.0 - cos(theta) ** 2)
        * (52.5 * cos(theta) ** 2 - 7.5)
        * sin(2 * phi)
    )


@torch.jit.script
def Yl4_m_minus_1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.267618617422916
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (17.5 * cos(theta) ** 3 - 7.5 * cos(theta))
        * sin(phi)
    )


@torch.jit.script
def Yl4_m0(theta: torch.Tensor, phi: torch.Tensor):
    return (
        3.70249414203215 * cos(theta) ** 4
        - 3.17356640745613 * cos(theta) ** 2
        + 0.317356640745613
    )


@torch.jit.script
def Yl4_m1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.267618617422916
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (17.5 * cos(theta) ** 3 - 7.5 * cos(theta))
        * cos(phi)
    )


@torch.jit.script
def Yl4_m2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.063078313050504
        * (1.0 - cos(theta) ** 2)
        * (52.5 * cos(theta) ** 2 - 7.5)
        * cos(2 * phi)
    )


@torch.jit.script
def Yl4_m3(theta: torch.Tensor, phi: torch.Tensor):
    return 1.77013076977993 * (1.0 - cos(theta) ** 2) ** 1.5 * cos(3 * phi) * cos(theta)


@torch.jit.script
def Yl4_m4(theta: torch.Tensor, phi: torch.Tensor):
    return 0.625835735449176 * (1.0 - cos(theta) ** 2) ** 2 * cos(4 * phi)


@torch.jit.script
def Yl5_m_minus_5(theta: torch.Tensor, phi: torch.Tensor):
    return 0.65638205684017 * (1.0 - cos(theta) ** 2) ** 2.5 * sin(5 * phi)


@torch.jit.script
def Yl5_m_minus_4(theta: torch.Tensor, phi: torch.Tensor):
    return 2.07566231488104 * (1.0 - cos(theta) ** 2) ** 2 * sin(4 * phi) * cos(theta)


@torch.jit.script
def Yl5_m_minus_3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.00931882475114763
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (472.5 * cos(theta) ** 2 - 52.5)
        * sin(3 * phi)
    )


@torch.jit.script
def Yl5_m_minus_2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0456527312854602
        * (1.0 - cos(theta) ** 2)
        * (157.5 * cos(theta) ** 3 - 52.5 * cos(theta))
        * sin(2 * phi)
    )


@torch.jit.script
def Yl5_m_minus_1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.241571547304372
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (39.375 * cos(theta) ** 4 - 26.25 * cos(theta) ** 2 + 1.875)
        * sin(phi)
    )


@torch.jit.script
def Yl5_m0(theta: torch.Tensor, phi: torch.Tensor):
    return (
        7.36787031456569 * cos(theta) ** 5
        - 8.18652257173965 * cos(theta) ** 3
        + 1.75425483680135 * cos(theta)
    )


@torch.jit.script
def Yl5_m1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.241571547304372
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (39.375 * cos(theta) ** 4 - 26.25 * cos(theta) ** 2 + 1.875)
        * cos(phi)
    )


@torch.jit.script
def Yl5_m2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0456527312854602
        * (1.0 - cos(theta) ** 2)
        * (157.5 * cos(theta) ** 3 - 52.5 * cos(theta))
        * cos(2 * phi)
    )


@torch.jit.script
def Yl5_m3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.00931882475114763
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (472.5 * cos(theta) ** 2 - 52.5)
        * cos(3 * phi)
    )


@torch.jit.script
def Yl5_m4(theta: torch.Tensor, phi: torch.Tensor):
    return 2.07566231488104 * (1.0 - cos(theta) ** 2) ** 2 * cos(4 * phi) * cos(theta)


@torch.jit.script
def Yl5_m5(theta: torch.Tensor, phi: torch.Tensor):
    return 0.65638205684017 * (1.0 - cos(theta) ** 2) ** 2.5 * cos(5 * phi)


@torch.jit.script
def Yl6_m_minus_6(theta: torch.Tensor, phi: torch.Tensor):
    return 0.683184105191914 * (1.0 - cos(theta) ** 2) ** 3 * sin(6 * phi)


@torch.jit.script
def Yl6_m_minus_5(theta: torch.Tensor, phi: torch.Tensor):
    return 2.36661916223175 * (1.0 - cos(theta) ** 2) ** 2.5 * sin(5 * phi) * cos(theta)


@torch.jit.script
def Yl6_m_minus_4(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0010678622237645
        * (1.0 - cos(theta) ** 2) ** 2
        * (5197.5 * cos(theta) ** 2 - 472.5)
        * sin(4 * phi)
    )


@torch.jit.script
def Yl6_m_minus_3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.00584892228263444
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (1732.5 * cos(theta) ** 3 - 472.5 * cos(theta))
        * sin(3 * phi)
    )


@torch.jit.script
def Yl6_m_minus_2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0350935336958066
        * (1.0 - cos(theta) ** 2)
        * (433.125 * cos(theta) ** 4 - 236.25 * cos(theta) ** 2 + 13.125)
        * sin(2 * phi)
    )


@torch.jit.script
def Yl6_m_minus_1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.221950995245231
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (86.625 * cos(theta) ** 5 - 78.75 * cos(theta) ** 3 + 13.125 * cos(theta))
        * sin(phi)
    )


@torch.jit.script
def Yl6_m0(theta: torch.Tensor, phi: torch.Tensor):
    return (
        14.6844857238222 * cos(theta) ** 6
        - 20.024298714303 * cos(theta) ** 4
        + 6.67476623810098 * cos(theta) ** 2
        - 0.317846011338142
    )


@torch.jit.script
def Yl6_m1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.221950995245231
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (86.625 * cos(theta) ** 5 - 78.75 * cos(theta) ** 3 + 13.125 * cos(theta))
        * cos(phi)
    )


@torch.jit.script
def Yl6_m2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0350935336958066
        * (1.0 - cos(theta) ** 2)
        * (433.125 * cos(theta) ** 4 - 236.25 * cos(theta) ** 2 + 13.125)
        * cos(2 * phi)
    )


@torch.jit.script
def Yl6_m3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.00584892228263444
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (1732.5 * cos(theta) ** 3 - 472.5 * cos(theta))
        * cos(3 * phi)
    )


@torch.jit.script
def Yl6_m4(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0010678622237645
        * (1.0 - cos(theta) ** 2) ** 2
        * (5197.5 * cos(theta) ** 2 - 472.5)
        * cos(4 * phi)
    )


@torch.jit.script
def Yl6_m5(theta: torch.Tensor, phi: torch.Tensor):
    return 2.36661916223175 * (1.0 - cos(theta) ** 2) ** 2.5 * cos(5 * phi) * cos(theta)


@torch.jit.script
def Yl6_m6(theta: torch.Tensor, phi: torch.Tensor):
    return 0.683184105191914 * (1.0 - cos(theta) ** 2) ** 3 * cos(6 * phi)


@torch.jit.script
def Yl7_m_minus_7(theta: torch.Tensor, phi: torch.Tensor):
    return 0.707162732524596 * (1.0 - cos(theta) ** 2) ** 3.5 * sin(7 * phi)


@torch.jit.script
def Yl7_m_minus_6(theta: torch.Tensor, phi: torch.Tensor):
    return 2.6459606618019 * (1.0 - cos(theta) ** 2) ** 3 * sin(6 * phi) * cos(theta)


@torch.jit.script
def Yl7_m_minus_5(theta: torch.Tensor, phi: torch.Tensor):
    return (
        9.98394571852353e-5
        * (1.0 - cos(theta) ** 2) ** 2.5
        * (67567.5 * cos(theta) ** 2 - 5197.5)
        * sin(5 * phi)
    )


@torch.jit.script
def Yl7_m_minus_4(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.000599036743111412
        * (1.0 - cos(theta) ** 2) ** 2
        * (22522.5 * cos(theta) ** 3 - 5197.5 * cos(theta))
        * sin(4 * phi)
    )


@torch.jit.script
def Yl7_m_minus_3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.00397356022507413
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (5630.625 * cos(theta) ** 4 - 2598.75 * cos(theta) ** 2 + 118.125)
        * sin(3 * phi)
    )


@torch.jit.script
def Yl7_m_minus_2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0280973138060306
        * (1.0 - cos(theta) ** 2)
        * (1126.125 * cos(theta) ** 5 - 866.25 * cos(theta) ** 3 + 118.125 * cos(theta))
        * sin(2 * phi)
    )


@torch.jit.script
def Yl7_m_minus_1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.206472245902897
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (
            187.6875 * cos(theta) ** 6
            - 216.5625 * cos(theta) ** 4
            + 59.0625 * cos(theta) ** 2
            - 2.1875
        )
        * sin(phi)
    )


@torch.jit.script
def Yl7_m0(theta: torch.Tensor, phi: torch.Tensor):
    return (
        29.2939547952501 * cos(theta) ** 7
        - 47.3210039000194 * cos(theta) ** 5
        + 21.5095472272816 * cos(theta) ** 3
        - 2.38994969192017 * cos(theta)
    )


@torch.jit.script
def Yl7_m1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.206472245902897
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (
            187.6875 * cos(theta) ** 6
            - 216.5625 * cos(theta) ** 4
            + 59.0625 * cos(theta) ** 2
            - 2.1875
        )
        * cos(phi)
    )


@torch.jit.script
def Yl7_m2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0280973138060306
        * (1.0 - cos(theta) ** 2)
        * (1126.125 * cos(theta) ** 5 - 866.25 * cos(theta) ** 3 + 118.125 * cos(theta))
        * cos(2 * phi)
    )


@torch.jit.script
def Yl7_m3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.00397356022507413
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (5630.625 * cos(theta) ** 4 - 2598.75 * cos(theta) ** 2 + 118.125)
        * cos(3 * phi)
    )


@torch.jit.script
def Yl7_m4(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.000599036743111412
        * (1.0 - cos(theta) ** 2) ** 2
        * (22522.5 * cos(theta) ** 3 - 5197.5 * cos(theta))
        * cos(4 * phi)
    )


@torch.jit.script
def Yl7_m5(theta: torch.Tensor, phi: torch.Tensor):
    return (
        9.98394571852353e-5
        * (1.0 - cos(theta) ** 2) ** 2.5
        * (67567.5 * cos(theta) ** 2 - 5197.5)
        * cos(5 * phi)
    )


@torch.jit.script
def Yl7_m6(theta: torch.Tensor, phi: torch.Tensor):
    return 2.6459606618019 * (1.0 - cos(theta) ** 2) ** 3 * cos(6 * phi) * cos(theta)


@torch.jit.script
def Yl7_m7(theta: torch.Tensor, phi: torch.Tensor):
    return 0.707162732524596 * (1.0 - cos(theta) ** 2) ** 3.5 * cos(7 * phi)


@torch.jit.script
def Yl8_m_minus_8(theta: torch.Tensor, phi: torch.Tensor):
    return 0.72892666017483 * (1.0 - cos(theta) ** 2) ** 4 * sin(8 * phi)


@torch.jit.script
def Yl8_m_minus_7(theta: torch.Tensor, phi: torch.Tensor):
    return 2.91570664069932 * (1.0 - cos(theta) ** 2) ** 3.5 * sin(7 * phi) * cos(theta)


@torch.jit.script
def Yl8_m_minus_6(theta: torch.Tensor, phi: torch.Tensor):
    return (
        7.87853281621404e-6
        * (1.0 - cos(theta) ** 2) ** 3
        * (1013512.5 * cos(theta) ** 2 - 67567.5)
        * sin(6 * phi)
    )


@torch.jit.script
def Yl8_m_minus_5(theta: torch.Tensor, phi: torch.Tensor):
    return (
        5.10587282657803e-5
        * (1.0 - cos(theta) ** 2) ** 2.5
        * (337837.5 * cos(theta) ** 3 - 67567.5 * cos(theta))
        * sin(5 * phi)
    )


@torch.jit.script
def Yl8_m_minus_4(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.000368189725644507
        * (1.0 - cos(theta) ** 2) ** 2
        * (84459.375 * cos(theta) ** 4 - 33783.75 * cos(theta) ** 2 + 1299.375)
        * sin(4 * phi)
    )


@torch.jit.script
def Yl8_m_minus_3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0028519853513317
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (
            16891.875 * cos(theta) ** 5
            - 11261.25 * cos(theta) ** 3
            + 1299.375 * cos(theta)
        )
        * sin(3 * phi)
    )


@torch.jit.script
def Yl8_m_minus_2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0231696385236779
        * (1.0 - cos(theta) ** 2)
        * (
            2815.3125 * cos(theta) ** 6
            - 2815.3125 * cos(theta) ** 4
            + 649.6875 * cos(theta) ** 2
            - 19.6875
        )
        * sin(2 * phi)
    )


@torch.jit.script
def Yl8_m_minus_1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.193851103820053
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (
            402.1875 * cos(theta) ** 7
            - 563.0625 * cos(theta) ** 5
            + 216.5625 * cos(theta) ** 3
            - 19.6875 * cos(theta)
        )
        * sin(phi)
    )


@torch.jit.script
def Yl8_m0(theta: torch.Tensor, phi: torch.Tensor):
    return (
        58.4733681132208 * cos(theta) ** 8
        - 109.150287144679 * cos(theta) ** 6
        + 62.9713195065454 * cos(theta) ** 4
        - 11.4493308193719 * cos(theta) ** 2
        + 0.318036967204775
    )


@torch.jit.script
def Yl8_m1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.193851103820053
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (
            402.1875 * cos(theta) ** 7
            - 563.0625 * cos(theta) ** 5
            + 216.5625 * cos(theta) ** 3
            - 19.6875 * cos(theta)
        )
        * cos(phi)
    )


@torch.jit.script
def Yl8_m2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0231696385236779
        * (1.0 - cos(theta) ** 2)
        * (
            2815.3125 * cos(theta) ** 6
            - 2815.3125 * cos(theta) ** 4
            + 649.6875 * cos(theta) ** 2
            - 19.6875
        )
        * cos(2 * phi)
    )


@torch.jit.script
def Yl8_m3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0028519853513317
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (
            16891.875 * cos(theta) ** 5
            - 11261.25 * cos(theta) ** 3
            + 1299.375 * cos(theta)
        )
        * cos(3 * phi)
    )


@torch.jit.script
def Yl8_m4(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.000368189725644507
        * (1.0 - cos(theta) ** 2) ** 2
        * (84459.375 * cos(theta) ** 4 - 33783.75 * cos(theta) ** 2 + 1299.375)
        * cos(4 * phi)
    )


@torch.jit.script
def Yl8_m5(theta: torch.Tensor, phi: torch.Tensor):
    return (
        5.10587282657803e-5
        * (1.0 - cos(theta) ** 2) ** 2.5
        * (337837.5 * cos(theta) ** 3 - 67567.5 * cos(theta))
        * cos(5 * phi)
    )


@torch.jit.script
def Yl8_m6(theta: torch.Tensor, phi: torch.Tensor):
    return (
        7.87853281621404e-6
        * (1.0 - cos(theta) ** 2) ** 3
        * (1013512.5 * cos(theta) ** 2 - 67567.5)
        * cos(6 * phi)
    )


@torch.jit.script
def Yl8_m7(theta: torch.Tensor, phi: torch.Tensor):
    return 2.91570664069932 * (1.0 - cos(theta) ** 2) ** 3.5 * cos(7 * phi) * cos(theta)


@torch.jit.script
def Yl8_m8(theta: torch.Tensor, phi: torch.Tensor):
    return 0.72892666017483 * (1.0 - cos(theta) ** 2) ** 4 * cos(8 * phi)


@torch.jit.script
def Yl9_m_minus_9(theta: torch.Tensor, phi: torch.Tensor):
    return 0.748900951853188 * (1.0 - cos(theta) ** 2) ** 4.5 * sin(9 * phi)


@torch.jit.script
def Yl9_m_minus_8(theta: torch.Tensor, phi: torch.Tensor):
    return 3.1773176489547 * (1.0 - cos(theta) ** 2) ** 4 * sin(8 * phi) * cos(theta)


@torch.jit.script
def Yl9_m_minus_7(theta: torch.Tensor, phi: torch.Tensor):
    return (
        5.37640612566745e-7
        * (1.0 - cos(theta) ** 2) ** 3.5
        * (17229712.5 * cos(theta) ** 2 - 1013512.5)
        * sin(7 * phi)
    )


@torch.jit.script
def Yl9_m_minus_6(theta: torch.Tensor, phi: torch.Tensor):
    return (
        3.72488342871223e-6
        * (1.0 - cos(theta) ** 2) ** 3
        * (5743237.5 * cos(theta) ** 3 - 1013512.5 * cos(theta))
        * sin(6 * phi)
    )


@torch.jit.script
def Yl9_m_minus_5(theta: torch.Tensor, phi: torch.Tensor):
    return (
        2.88528229719329e-5
        * (1.0 - cos(theta) ** 2) ** 2.5
        * (1435809.375 * cos(theta) ** 4 - 506756.25 * cos(theta) ** 2 + 16891.875)
        * sin(5 * phi)
    )


@torch.jit.script
def Yl9_m_minus_4(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.000241400036332803
        * (1.0 - cos(theta) ** 2) ** 2
        * (
            287161.875 * cos(theta) ** 5
            - 168918.75 * cos(theta) ** 3
            + 16891.875 * cos(theta)
        )
        * sin(4 * phi)
    )


@torch.jit.script
def Yl9_m_minus_3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.00213198739401417
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (
            47860.3125 * cos(theta) ** 6
            - 42229.6875 * cos(theta) ** 4
            + 8445.9375 * cos(theta) ** 2
            - 216.5625
        )
        * sin(3 * phi)
    )


@torch.jit.script
def Yl9_m_minus_2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0195399872275232
        * (1.0 - cos(theta) ** 2)
        * (
            6837.1875 * cos(theta) ** 7
            - 8445.9375 * cos(theta) ** 5
            + 2815.3125 * cos(theta) ** 3
            - 216.5625 * cos(theta)
        )
        * sin(2 * phi)
    )


@torch.jit.script
def Yl9_m_minus_1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.183301328077446
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (
            854.6484375 * cos(theta) ** 8
            - 1407.65625 * cos(theta) ** 6
            + 703.828125 * cos(theta) ** 4
            - 108.28125 * cos(theta) ** 2
            + 2.4609375
        )
        * sin(phi)
    )


@torch.jit.script
def Yl9_m0(theta: torch.Tensor, phi: torch.Tensor):
    return (
        116.766123398619 * cos(theta) ** 9
        - 247.269437785311 * cos(theta) ** 7
        + 173.088606449718 * cos(theta) ** 5
        - 44.3816939614661 * cos(theta) ** 3
        + 3.02602458828178 * cos(theta)
    )


@torch.jit.script
def Yl9_m1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.183301328077446
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (
            854.6484375 * cos(theta) ** 8
            - 1407.65625 * cos(theta) ** 6
            + 703.828125 * cos(theta) ** 4
            - 108.28125 * cos(theta) ** 2
            + 2.4609375
        )
        * cos(phi)
    )


@torch.jit.script
def Yl9_m2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0195399872275232
        * (1.0 - cos(theta) ** 2)
        * (
            6837.1875 * cos(theta) ** 7
            - 8445.9375 * cos(theta) ** 5
            + 2815.3125 * cos(theta) ** 3
            - 216.5625 * cos(theta)
        )
        * cos(2 * phi)
    )


@torch.jit.script
def Yl9_m3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.00213198739401417
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (
            47860.3125 * cos(theta) ** 6
            - 42229.6875 * cos(theta) ** 4
            + 8445.9375 * cos(theta) ** 2
            - 216.5625
        )
        * cos(3 * phi)
    )


@torch.jit.script
def Yl9_m4(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.000241400036332803
        * (1.0 - cos(theta) ** 2) ** 2
        * (
            287161.875 * cos(theta) ** 5
            - 168918.75 * cos(theta) ** 3
            + 16891.875 * cos(theta)
        )
        * cos(4 * phi)
    )


@torch.jit.script
def Yl9_m5(theta: torch.Tensor, phi: torch.Tensor):
    return (
        2.88528229719329e-5
        * (1.0 - cos(theta) ** 2) ** 2.5
        * (1435809.375 * cos(theta) ** 4 - 506756.25 * cos(theta) ** 2 + 16891.875)
        * cos(5 * phi)
    )


@torch.jit.script
def Yl9_m6(theta: torch.Tensor, phi: torch.Tensor):
    return (
        3.72488342871223e-6
        * (1.0 - cos(theta) ** 2) ** 3
        * (5743237.5 * cos(theta) ** 3 - 1013512.5 * cos(theta))
        * cos(6 * phi)
    )


@torch.jit.script
def Yl9_m7(theta: torch.Tensor, phi: torch.Tensor):
    return (
        5.37640612566745e-7
        * (1.0 - cos(theta) ** 2) ** 3.5
        * (17229712.5 * cos(theta) ** 2 - 1013512.5)
        * cos(7 * phi)
    )


@torch.jit.script
def Yl9_m8(theta: torch.Tensor, phi: torch.Tensor):
    return 3.1773176489547 * (1.0 - cos(theta) ** 2) ** 4 * cos(8 * phi) * cos(theta)


@torch.jit.script
def Yl9_m9(theta: torch.Tensor, phi: torch.Tensor):
    return 0.748900951853188 * (1.0 - cos(theta) ** 2) ** 4.5 * cos(9 * phi)


@torch.jit.script
def Yl10_m_minus_10(theta: torch.Tensor, phi: torch.Tensor):
    return 0.76739511822199 * (1.0 - cos(theta) ** 2) ** 5 * sin(10 * phi)


@torch.jit.script
def Yl10_m_minus_9(theta: torch.Tensor, phi: torch.Tensor):
    return 3.43189529989171 * (1.0 - cos(theta) ** 2) ** 4.5 * sin(9 * phi) * cos(theta)


@torch.jit.script
def Yl10_m_minus_8(theta: torch.Tensor, phi: torch.Tensor):
    return (
        3.23120268385452e-8
        * (1.0 - cos(theta) ** 2) ** 4
        * (327364537.5 * cos(theta) ** 2 - 17229712.5)
        * sin(8 * phi)
    )


@torch.jit.script
def Yl10_m_minus_7(theta: torch.Tensor, phi: torch.Tensor):
    return (
        2.37443934928654e-7
        * (1.0 - cos(theta) ** 2) ** 3.5
        * (109121512.5 * cos(theta) ** 3 - 17229712.5 * cos(theta))
        * sin(7 * phi)
    )


@torch.jit.script
def Yl10_m_minus_6(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.95801284774625e-6
        * (1.0 - cos(theta) ** 2) ** 3
        * (27280378.125 * cos(theta) ** 4 - 8614856.25 * cos(theta) ** 2 + 253378.125)
        * sin(6 * phi)
    )


@torch.jit.script
def Yl10_m_minus_5(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.75129993135143e-5
        * (1.0 - cos(theta) ** 2) ** 2.5
        * (
            5456075.625 * cos(theta) ** 5
            - 2871618.75 * cos(theta) ** 3
            + 253378.125 * cos(theta)
        )
        * sin(5 * phi)
    )


@torch.jit.script
def Yl10_m_minus_4(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.000166142899475011
        * (1.0 - cos(theta) ** 2) ** 2
        * (
            909345.9375 * cos(theta) ** 6
            - 717904.6875 * cos(theta) ** 4
            + 126689.0625 * cos(theta) ** 2
            - 2815.3125
        )
        * sin(4 * phi)
    )


@torch.jit.script
def Yl10_m_minus_3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.00164473079210685
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (
            129906.5625 * cos(theta) ** 7
            - 143580.9375 * cos(theta) ** 5
            + 42229.6875 * cos(theta) ** 3
            - 2815.3125 * cos(theta)
        )
        * sin(3 * phi)
    )


@torch.jit.script
def Yl10_m_minus_2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0167730288071195
        * (1.0 - cos(theta) ** 2)
        * (
            16238.3203125 * cos(theta) ** 8
            - 23930.15625 * cos(theta) ** 6
            + 10557.421875 * cos(theta) ** 4
            - 1407.65625 * cos(theta) ** 2
            + 27.0703125
        )
        * sin(2 * phi)
    )


@torch.jit.script
def Yl10_m_minus_1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.174310428544485
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (
            1804.2578125 * cos(theta) ** 9
            - 3418.59375 * cos(theta) ** 7
            + 2111.484375 * cos(theta) ** 5
            - 469.21875 * cos(theta) ** 3
            + 27.0703125 * cos(theta)
        )
        * sin(phi)
    )


@torch.jit.script
def Yl10_m0(theta: torch.Tensor, phi: torch.Tensor):
    return (
        233.240148813258 * cos(theta) ** 10
        - 552.410878768242 * cos(theta) ** 8
        + 454.926606044435 * cos(theta) ** 6
        - 151.642202014812 * cos(theta) ** 4
        + 17.4971771555552 * cos(theta) ** 2
        - 0.318130493737367
    )


@torch.jit.script
def Yl10_m1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.174310428544485
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (
            1804.2578125 * cos(theta) ** 9
            - 3418.59375 * cos(theta) ** 7
            + 2111.484375 * cos(theta) ** 5
            - 469.21875 * cos(theta) ** 3
            + 27.0703125 * cos(theta)
        )
        * cos(phi)
    )


@torch.jit.script
def Yl10_m2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0167730288071195
        * (1.0 - cos(theta) ** 2)
        * (
            16238.3203125 * cos(theta) ** 8
            - 23930.15625 * cos(theta) ** 6
            + 10557.421875 * cos(theta) ** 4
            - 1407.65625 * cos(theta) ** 2
            + 27.0703125
        )
        * cos(2 * phi)
    )


@torch.jit.script
def Yl10_m3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.00164473079210685
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (
            129906.5625 * cos(theta) ** 7
            - 143580.9375 * cos(theta) ** 5
            + 42229.6875 * cos(theta) ** 3
            - 2815.3125 * cos(theta)
        )
        * cos(3 * phi)
    )


@torch.jit.script
def Yl10_m4(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.000166142899475011
        * (1.0 - cos(theta) ** 2) ** 2
        * (
            909345.9375 * cos(theta) ** 6
            - 717904.6875 * cos(theta) ** 4
            + 126689.0625 * cos(theta) ** 2
            - 2815.3125
        )
        * cos(4 * phi)
    )


@torch.jit.script
def Yl10_m5(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.75129993135143e-5
        * (1.0 - cos(theta) ** 2) ** 2.5
        * (
            5456075.625 * cos(theta) ** 5
            - 2871618.75 * cos(theta) ** 3
            + 253378.125 * cos(theta)
        )
        * cos(5 * phi)
    )


@torch.jit.script
def Yl10_m6(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.95801284774625e-6
        * (1.0 - cos(theta) ** 2) ** 3
        * (27280378.125 * cos(theta) ** 4 - 8614856.25 * cos(theta) ** 2 + 253378.125)
        * cos(6 * phi)
    )


@torch.jit.script
def Yl10_m7(theta: torch.Tensor, phi: torch.Tensor):
    return (
        2.37443934928654e-7
        * (1.0 - cos(theta) ** 2) ** 3.5
        * (109121512.5 * cos(theta) ** 3 - 17229712.5 * cos(theta))
        * cos(7 * phi)
    )


@torch.jit.script
def Yl10_m8(theta: torch.Tensor, phi: torch.Tensor):
    return (
        3.23120268385452e-8
        * (1.0 - cos(theta) ** 2) ** 4
        * (327364537.5 * cos(theta) ** 2 - 17229712.5)
        * cos(8 * phi)
    )


@torch.jit.script
def Yl10_m9(theta: torch.Tensor, phi: torch.Tensor):
    return 3.43189529989171 * (1.0 - cos(theta) ** 2) ** 4.5 * cos(9 * phi) * cos(theta)


@torch.jit.script
def Yl10_m10(theta: torch.Tensor, phi: torch.Tensor):
    return 0.76739511822199 * (1.0 - cos(theta) ** 2) ** 5 * cos(10 * phi)


@torch.jit.script
def Yl11_m_minus_11(theta: torch.Tensor, phi: torch.Tensor):
    return 0.784642105787197 * (1.0 - cos(theta) ** 2) ** 5.5 * sin(11 * phi)


@torch.jit.script
def Yl11_m_minus_10(theta: torch.Tensor, phi: torch.Tensor):
    return 3.68029769880531 * (1.0 - cos(theta) ** 2) ** 5 * sin(10 * phi) * cos(theta)


@torch.jit.script
def Yl11_m_minus_9(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.73470916587426e-9
        * (1.0 - cos(theta) ** 2) ** 4.5
        * (6874655287.5 * cos(theta) ** 2 - 327364537.5)
        * sin(9 * phi)
    )


@torch.jit.script
def Yl11_m_minus_8(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.34369994198887e-8
        * (1.0 - cos(theta) ** 2) ** 4
        * (2291551762.5 * cos(theta) ** 3 - 327364537.5 * cos(theta))
        * sin(8 * phi)
    )


@torch.jit.script
def Yl11_m_minus_7(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.17141045151419e-7
        * (1.0 - cos(theta) ** 2) ** 3.5
        * (
            572887940.625 * cos(theta) ** 4
            - 163682268.75 * cos(theta) ** 2
            + 4307428.125
        )
        * sin(7 * phi)
    )


@torch.jit.script
def Yl11_m_minus_6(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.11129753051333e-6
        * (1.0 - cos(theta) ** 2) ** 3
        * (
            114577588.125 * cos(theta) ** 5
            - 54560756.25 * cos(theta) ** 3
            + 4307428.125 * cos(theta)
        )
        * sin(6 * phi)
    )


@torch.jit.script
def Yl11_m_minus_5(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.12235548974089e-5
        * (1.0 - cos(theta) ** 2) ** 2.5
        * (
            19096264.6875 * cos(theta) ** 6
            - 13640189.0625 * cos(theta) ** 4
            + 2153714.0625 * cos(theta) ** 2
            - 42229.6875
        )
        * sin(5 * phi)
    )


@torch.jit.script
def Yl11_m_minus_4(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0001187789403385
        * (1.0 - cos(theta) ** 2) ** 2
        * (
            2728037.8125 * cos(theta) ** 7
            - 2728037.8125 * cos(theta) ** 5
            + 717904.6875 * cos(theta) ** 3
            - 42229.6875 * cos(theta)
        )
        * sin(4 * phi)
    )


@torch.jit.script
def Yl11_m_minus_3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.00130115809959914
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (
            341004.7265625 * cos(theta) ** 8
            - 454672.96875 * cos(theta) ** 6
            + 179476.171875 * cos(theta) ** 4
            - 21114.84375 * cos(theta) ** 2
            + 351.9140625
        )
        * sin(3 * phi)
    )


@torch.jit.script
def Yl11_m_minus_2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0146054634441776
        * (1.0 - cos(theta) ** 2)
        * (
            37889.4140625 * cos(theta) ** 9
            - 64953.28125 * cos(theta) ** 7
            + 35895.234375 * cos(theta) ** 5
            - 7038.28125 * cos(theta) ** 3
            + 351.9140625 * cos(theta)
        )
        * sin(2 * phi)
    )


@torch.jit.script
def Yl11_m_minus_1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.166527904912351
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (
            3788.94140625 * cos(theta) ** 10
            - 8119.16015625 * cos(theta) ** 8
            + 5982.5390625 * cos(theta) ** 6
            - 1759.5703125 * cos(theta) ** 4
            + 175.95703125 * cos(theta) ** 2
            - 2.70703125
        )
        * sin(phi)
    )


@torch.jit.script
def Yl11_m0(theta: torch.Tensor, phi: torch.Tensor):
    return (
        465.998147319252 * cos(theta) ** 11
        - 1220.47133821709 * cos(theta) ** 9
        + 1156.23600462672 * cos(theta) ** 7
        - 476.097178375706 * cos(theta) ** 5
        + 79.3495297292844 * cos(theta) ** 3
        - 3.66228598750543 * cos(theta)
    )


@torch.jit.script
def Yl11_m1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.166527904912351
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (
            3788.94140625 * cos(theta) ** 10
            - 8119.16015625 * cos(theta) ** 8
            + 5982.5390625 * cos(theta) ** 6
            - 1759.5703125 * cos(theta) ** 4
            + 175.95703125 * cos(theta) ** 2
            - 2.70703125
        )
        * cos(phi)
    )


@torch.jit.script
def Yl11_m2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0146054634441776
        * (1.0 - cos(theta) ** 2)
        * (
            37889.4140625 * cos(theta) ** 9
            - 64953.28125 * cos(theta) ** 7
            + 35895.234375 * cos(theta) ** 5
            - 7038.28125 * cos(theta) ** 3
            + 351.9140625 * cos(theta)
        )
        * cos(2 * phi)
    )


@torch.jit.script
def Yl11_m3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.00130115809959914
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (
            341004.7265625 * cos(theta) ** 8
            - 454672.96875 * cos(theta) ** 6
            + 179476.171875 * cos(theta) ** 4
            - 21114.84375 * cos(theta) ** 2
            + 351.9140625
        )
        * cos(3 * phi)
    )


@torch.jit.script
def Yl11_m4(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0001187789403385
        * (1.0 - cos(theta) ** 2) ** 2
        * (
            2728037.8125 * cos(theta) ** 7
            - 2728037.8125 * cos(theta) ** 5
            + 717904.6875 * cos(theta) ** 3
            - 42229.6875 * cos(theta)
        )
        * cos(4 * phi)
    )


@torch.jit.script
def Yl11_m5(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.12235548974089e-5
        * (1.0 - cos(theta) ** 2) ** 2.5
        * (
            19096264.6875 * cos(theta) ** 6
            - 13640189.0625 * cos(theta) ** 4
            + 2153714.0625 * cos(theta) ** 2
            - 42229.6875
        )
        * cos(5 * phi)
    )


@torch.jit.script
def Yl11_m6(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.11129753051333e-6
        * (1.0 - cos(theta) ** 2) ** 3
        * (
            114577588.125 * cos(theta) ** 5
            - 54560756.25 * cos(theta) ** 3
            + 4307428.125 * cos(theta)
        )
        * cos(6 * phi)
    )


@torch.jit.script
def Yl11_m7(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.17141045151419e-7
        * (1.0 - cos(theta) ** 2) ** 3.5
        * (
            572887940.625 * cos(theta) ** 4
            - 163682268.75 * cos(theta) ** 2
            + 4307428.125
        )
        * cos(7 * phi)
    )


@torch.jit.script
def Yl11_m8(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.34369994198887e-8
        * (1.0 - cos(theta) ** 2) ** 4
        * (2291551762.5 * cos(theta) ** 3 - 327364537.5 * cos(theta))
        * cos(8 * phi)
    )


@torch.jit.script
def Yl11_m9(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.73470916587426e-9
        * (1.0 - cos(theta) ** 2) ** 4.5
        * (6874655287.5 * cos(theta) ** 2 - 327364537.5)
        * cos(9 * phi)
    )


@torch.jit.script
def Yl11_m10(theta: torch.Tensor, phi: torch.Tensor):
    return 3.68029769880531 * (1.0 - cos(theta) ** 2) ** 5 * cos(10 * phi) * cos(theta)


@torch.jit.script
def Yl11_m11(theta: torch.Tensor, phi: torch.Tensor):
    return 0.784642105787197 * (1.0 - cos(theta) ** 2) ** 5.5 * cos(11 * phi)


@torch.jit.script
def Yl12_m_minus_12(theta: torch.Tensor, phi: torch.Tensor):
    return 0.800821995783972 * (1.0 - cos(theta) ** 2) ** 6 * sin(12 * phi)


@torch.jit.script
def Yl12_m_minus_11(theta: torch.Tensor, phi: torch.Tensor):
    return (
        3.92321052893598 * (1.0 - cos(theta) ** 2) ** 5.5 * sin(11 * phi) * cos(theta)
    )


@torch.jit.script
def Yl12_m_minus_10(theta: torch.Tensor, phi: torch.Tensor):
    return (
        8.4141794839602e-11
        * (1.0 - cos(theta) ** 2) ** 5
        * (158117071612.5 * cos(theta) ** 2 - 6874655287.5)
        * sin(10 * phi)
    )


@torch.jit.script
def Yl12_m_minus_9(theta: torch.Tensor, phi: torch.Tensor):
    return (
        6.83571172711927e-10
        * (1.0 - cos(theta) ** 2) ** 4.5
        * (52705690537.5 * cos(theta) ** 3 - 6874655287.5 * cos(theta))
        * sin(9 * phi)
    )


@torch.jit.script
def Yl12_m_minus_8(theta: torch.Tensor, phi: torch.Tensor):
    return (
        6.26503328368427e-9
        * (1.0 - cos(theta) ** 2) ** 4
        * (
            13176422634.375 * cos(theta) ** 4
            - 3437327643.75 * cos(theta) ** 2
            + 81841134.375
        )
        * sin(8 * phi)
    )


@torch.jit.script
def Yl12_m_minus_7(theta: torch.Tensor, phi: torch.Tensor):
    return (
        6.26503328368427e-8
        * (1.0 - cos(theta) ** 2) ** 3.5
        * (
            2635284526.875 * cos(theta) ** 5
            - 1145775881.25 * cos(theta) ** 3
            + 81841134.375 * cos(theta)
        )
        * sin(7 * phi)
    )


@torch.jit.script
def Yl12_m_minus_6(theta: torch.Tensor, phi: torch.Tensor):
    return (
        6.68922506214776e-7
        * (1.0 - cos(theta) ** 2) ** 3
        * (
            439214087.8125 * cos(theta) ** 6
            - 286443970.3125 * cos(theta) ** 4
            + 40920567.1875 * cos(theta) ** 2
            - 717904.6875
        )
        * sin(6 * phi)
    )


@torch.jit.script
def Yl12_m_minus_5(theta: torch.Tensor, phi: torch.Tensor):
    return (
        7.50863650967357e-6
        * (1.0 - cos(theta) ** 2) ** 2.5
        * (
            62744869.6875 * cos(theta) ** 7
            - 57288794.0625 * cos(theta) ** 5
            + 13640189.0625 * cos(theta) ** 3
            - 717904.6875 * cos(theta)
        )
        * sin(5 * phi)
    )


@torch.jit.script
def Yl12_m_minus_4(theta: torch.Tensor, phi: torch.Tensor):
    return (
        8.75649965675714e-5
        * (1.0 - cos(theta) ** 2) ** 2
        * (
            7843108.7109375 * cos(theta) ** 8
            - 9548132.34375 * cos(theta) ** 6
            + 3410047.265625 * cos(theta) ** 4
            - 358952.34375 * cos(theta) ** 2
            + 5278.7109375
        )
        * sin(4 * phi)
    )


@torch.jit.script
def Yl12_m_minus_3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.00105077995881086
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (
            871456.5234375 * cos(theta) ** 9
            - 1364018.90625 * cos(theta) ** 7
            + 682009.453125 * cos(theta) ** 5
            - 119650.78125 * cos(theta) ** 3
            + 5278.7109375 * cos(theta)
        )
        * sin(3 * phi)
    )


@torch.jit.script
def Yl12_m_minus_2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0128693736551466
        * (1.0 - cos(theta) ** 2)
        * (
            87145.65234375 * cos(theta) ** 10
            - 170502.36328125 * cos(theta) ** 8
            + 113668.2421875 * cos(theta) ** 6
            - 29912.6953125 * cos(theta) ** 4
            + 2639.35546875 * cos(theta) ** 2
            - 35.19140625
        )
        * sin(2 * phi)
    )


@torch.jit.script
def Yl12_m_minus_1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.159704727088682
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (
            7922.33203125 * cos(theta) ** 11
            - 18944.70703125 * cos(theta) ** 9
            + 16238.3203125 * cos(theta) ** 7
            - 5982.5390625 * cos(theta) ** 5
            + 879.78515625 * cos(theta) ** 3
            - 35.19140625 * cos(theta)
        )
        * sin(phi)
    )


@torch.jit.script
def Yl12_m0(theta: torch.Tensor, phi: torch.Tensor):
    return (
        931.186918632914 * cos(theta) ** 12
        - 2672.1015925988 * cos(theta) ** 10
        + 2862.96599207014 * cos(theta) ** 8
        - 1406.36925926252 * cos(theta) ** 6
        + 310.228513072616 * cos(theta) ** 4
        - 24.8182810458093 * cos(theta) ** 2
        + 0.318183090330888
    )


@torch.jit.script
def Yl12_m1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.159704727088682
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (
            7922.33203125 * cos(theta) ** 11
            - 18944.70703125 * cos(theta) ** 9
            + 16238.3203125 * cos(theta) ** 7
            - 5982.5390625 * cos(theta) ** 5
            + 879.78515625 * cos(theta) ** 3
            - 35.19140625 * cos(theta)
        )
        * cos(phi)
    )


@torch.jit.script
def Yl12_m2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0128693736551466
        * (1.0 - cos(theta) ** 2)
        * (
            87145.65234375 * cos(theta) ** 10
            - 170502.36328125 * cos(theta) ** 8
            + 113668.2421875 * cos(theta) ** 6
            - 29912.6953125 * cos(theta) ** 4
            + 2639.35546875 * cos(theta) ** 2
            - 35.19140625
        )
        * cos(2 * phi)
    )


@torch.jit.script
def Yl12_m3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.00105077995881086
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (
            871456.5234375 * cos(theta) ** 9
            - 1364018.90625 * cos(theta) ** 7
            + 682009.453125 * cos(theta) ** 5
            - 119650.78125 * cos(theta) ** 3
            + 5278.7109375 * cos(theta)
        )
        * cos(3 * phi)
    )


@torch.jit.script
def Yl12_m4(theta: torch.Tensor, phi: torch.Tensor):
    return (
        8.75649965675714e-5
        * (1.0 - cos(theta) ** 2) ** 2
        * (
            7843108.7109375 * cos(theta) ** 8
            - 9548132.34375 * cos(theta) ** 6
            + 3410047.265625 * cos(theta) ** 4
            - 358952.34375 * cos(theta) ** 2
            + 5278.7109375
        )
        * cos(4 * phi)
    )


@torch.jit.script
def Yl12_m5(theta: torch.Tensor, phi: torch.Tensor):
    return (
        7.50863650967357e-6
        * (1.0 - cos(theta) ** 2) ** 2.5
        * (
            62744869.6875 * cos(theta) ** 7
            - 57288794.0625 * cos(theta) ** 5
            + 13640189.0625 * cos(theta) ** 3
            - 717904.6875 * cos(theta)
        )
        * cos(5 * phi)
    )


@torch.jit.script
def Yl12_m6(theta: torch.Tensor, phi: torch.Tensor):
    return (
        6.68922506214776e-7
        * (1.0 - cos(theta) ** 2) ** 3
        * (
            439214087.8125 * cos(theta) ** 6
            - 286443970.3125 * cos(theta) ** 4
            + 40920567.1875 * cos(theta) ** 2
            - 717904.6875
        )
        * cos(6 * phi)
    )


@torch.jit.script
def Yl12_m7(theta: torch.Tensor, phi: torch.Tensor):
    return (
        6.26503328368427e-8
        * (1.0 - cos(theta) ** 2) ** 3.5
        * (
            2635284526.875 * cos(theta) ** 5
            - 1145775881.25 * cos(theta) ** 3
            + 81841134.375 * cos(theta)
        )
        * cos(7 * phi)
    )


@torch.jit.script
def Yl12_m8(theta: torch.Tensor, phi: torch.Tensor):
    return (
        6.26503328368427e-9
        * (1.0 - cos(theta) ** 2) ** 4
        * (
            13176422634.375 * cos(theta) ** 4
            - 3437327643.75 * cos(theta) ** 2
            + 81841134.375
        )
        * cos(8 * phi)
    )


@torch.jit.script
def Yl12_m9(theta: torch.Tensor, phi: torch.Tensor):
    return (
        6.83571172711927e-10
        * (1.0 - cos(theta) ** 2) ** 4.5
        * (52705690537.5 * cos(theta) ** 3 - 6874655287.5 * cos(theta))
        * cos(9 * phi)
    )


@torch.jit.script
def Yl12_m10(theta: torch.Tensor, phi: torch.Tensor):
    return (
        8.4141794839602e-11
        * (1.0 - cos(theta) ** 2) ** 5
        * (158117071612.5 * cos(theta) ** 2 - 6874655287.5)
        * cos(10 * phi)
    )


@torch.jit.script
def Yl12_m11(theta: torch.Tensor, phi: torch.Tensor):
    return (
        3.92321052893598 * (1.0 - cos(theta) ** 2) ** 5.5 * cos(11 * phi) * cos(theta)
    )


@torch.jit.script
def Yl12_m12(theta: torch.Tensor, phi: torch.Tensor):
    return 0.800821995783972 * (1.0 - cos(theta) ** 2) ** 6 * cos(12 * phi)


@torch.jit.script
def Yl13_m_minus_13(theta: torch.Tensor, phi: torch.Tensor):
    return 0.816077118837628 * (1.0 - cos(theta) ** 2) ** 6.5 * sin(13 * phi)


@torch.jit.script
def Yl13_m_minus_12(theta: torch.Tensor, phi: torch.Tensor):
    return 4.16119315354964 * (1.0 - cos(theta) ** 2) ** 6 * sin(12 * phi) * cos(theta)


@torch.jit.script
def Yl13_m_minus_11(theta: torch.Tensor, phi: torch.Tensor):
    return (
        3.72180924766049e-12
        * (1.0 - cos(theta) ** 2) ** 5.5
        * (3952926790312.5 * cos(theta) ** 2 - 158117071612.5)
        * sin(11 * phi)
    )


@torch.jit.script
def Yl13_m_minus_10(theta: torch.Tensor, phi: torch.Tensor):
    return (
        3.15805986876424e-11
        * (1.0 - cos(theta) ** 2) ** 5
        * (1317642263437.5 * cos(theta) ** 3 - 158117071612.5 * cos(theta))
        * sin(10 * phi)
    )


@torch.jit.script
def Yl13_m_minus_9(theta: torch.Tensor, phi: torch.Tensor):
    return (
        3.02910461422567e-10
        * (1.0 - cos(theta) ** 2) ** 4.5
        * (
            329410565859.375 * cos(theta) ** 4
            - 79058535806.25 * cos(theta) ** 2
            + 1718663821.875
        )
        * sin(9 * phi)
    )


@torch.jit.script
def Yl13_m_minus_8(theta: torch.Tensor, phi: torch.Tensor):
    return (
        3.17695172143292e-9
        * (1.0 - cos(theta) ** 2) ** 4
        * (
            65882113171.875 * cos(theta) ** 5
            - 26352845268.75 * cos(theta) ** 3
            + 1718663821.875 * cos(theta)
        )
        * sin(8 * phi)
    )


@torch.jit.script
def Yl13_m_minus_7(theta: torch.Tensor, phi: torch.Tensor):
    return (
        3.5661194627771e-8
        * (1.0 - cos(theta) ** 2) ** 3.5
        * (
            10980352195.3125 * cos(theta) ** 6
            - 6588211317.1875 * cos(theta) ** 4
            + 859331910.9375 * cos(theta) ** 2
            - 13640189.0625
        )
        * sin(7 * phi)
    )


@torch.jit.script
def Yl13_m_minus_6(theta: torch.Tensor, phi: torch.Tensor):
    return (
        4.21948945157073e-7
        * (1.0 - cos(theta) ** 2) ** 3
        * (
            1568621742.1875 * cos(theta) ** 7
            - 1317642263.4375 * cos(theta) ** 5
            + 286443970.3125 * cos(theta) ** 3
            - 13640189.0625 * cos(theta)
        )
        * sin(6 * phi)
    )


@torch.jit.script
def Yl13_m_minus_5(theta: torch.Tensor, phi: torch.Tensor):
    return (
        5.2021359721285e-6
        * (1.0 - cos(theta) ** 2) ** 2.5
        * (
            196077717.773438 * cos(theta) ** 8
            - 219607043.90625 * cos(theta) ** 6
            + 71610992.578125 * cos(theta) ** 4
            - 6820094.53125 * cos(theta) ** 2
            + 89738.0859375
        )
        * sin(5 * phi)
    )


@torch.jit.script
def Yl13_m_minus_4(theta: torch.Tensor, phi: torch.Tensor):
    return (
        6.62123812058377e-5
        * (1.0 - cos(theta) ** 2) ** 2
        * (
            21786413.0859375 * cos(theta) ** 9
            - 31372434.84375 * cos(theta) ** 7
            + 14322198.515625 * cos(theta) ** 5
            - 2273364.84375 * cos(theta) ** 3
            + 89738.0859375 * cos(theta)
        )
        * sin(4 * phi)
    )


@torch.jit.script
def Yl13_m_minus_3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.000863303829622583
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (
            2178641.30859375 * cos(theta) ** 10
            - 3921554.35546875 * cos(theta) ** 8
            + 2387033.0859375 * cos(theta) ** 6
            - 568341.2109375 * cos(theta) ** 4
            + 44869.04296875 * cos(theta) ** 2
            - 527.87109375
        )
        * sin(3 * phi)
    )


@torch.jit.script
def Yl13_m_minus_2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0114530195317401
        * (1.0 - cos(theta) ** 2)
        * (
            198058.30078125 * cos(theta) ** 11
            - 435728.26171875 * cos(theta) ** 9
            + 341004.7265625 * cos(theta) ** 7
            - 113668.2421875 * cos(theta) ** 5
            + 14956.34765625 * cos(theta) ** 3
            - 527.87109375 * cos(theta)
        )
        * sin(2 * phi)
    )


@torch.jit.script
def Yl13_m_minus_1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.153658381323621
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (
            16504.8583984375 * cos(theta) ** 12
            - 43572.826171875 * cos(theta) ** 10
            + 42625.5908203125 * cos(theta) ** 8
            - 18944.70703125 * cos(theta) ** 6
            + 3739.0869140625 * cos(theta) ** 4
            - 263.935546875 * cos(theta) ** 2
            + 2.9326171875
        )
        * sin(phi)
    )


@torch.jit.script
def Yl13_m0(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1860.99583201813 * cos(theta) ** 13
        - 5806.30699589657 * cos(theta) ** 11
        + 6942.32358205025 * cos(theta) ** 9
        - 3967.04204688585 * cos(theta) ** 7
        + 1096.15635506056 * cos(theta) ** 5
        - 128.959571183596 * cos(theta) ** 3
        + 4.29865237278653 * cos(theta)
    )


@torch.jit.script
def Yl13_m1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.153658381323621
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (
            16504.8583984375 * cos(theta) ** 12
            - 43572.826171875 * cos(theta) ** 10
            + 42625.5908203125 * cos(theta) ** 8
            - 18944.70703125 * cos(theta) ** 6
            + 3739.0869140625 * cos(theta) ** 4
            - 263.935546875 * cos(theta) ** 2
            + 2.9326171875
        )
        * cos(phi)
    )


@torch.jit.script
def Yl13_m2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0114530195317401
        * (1.0 - cos(theta) ** 2)
        * (
            198058.30078125 * cos(theta) ** 11
            - 435728.26171875 * cos(theta) ** 9
            + 341004.7265625 * cos(theta) ** 7
            - 113668.2421875 * cos(theta) ** 5
            + 14956.34765625 * cos(theta) ** 3
            - 527.87109375 * cos(theta)
        )
        * cos(2 * phi)
    )


@torch.jit.script
def Yl13_m3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.000863303829622583
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (
            2178641.30859375 * cos(theta) ** 10
            - 3921554.35546875 * cos(theta) ** 8
            + 2387033.0859375 * cos(theta) ** 6
            - 568341.2109375 * cos(theta) ** 4
            + 44869.04296875 * cos(theta) ** 2
            - 527.87109375
        )
        * cos(3 * phi)
    )


@torch.jit.script
def Yl13_m4(theta: torch.Tensor, phi: torch.Tensor):
    return (
        6.62123812058377e-5
        * (1.0 - cos(theta) ** 2) ** 2
        * (
            21786413.0859375 * cos(theta) ** 9
            - 31372434.84375 * cos(theta) ** 7
            + 14322198.515625 * cos(theta) ** 5
            - 2273364.84375 * cos(theta) ** 3
            + 89738.0859375 * cos(theta)
        )
        * cos(4 * phi)
    )


@torch.jit.script
def Yl13_m5(theta: torch.Tensor, phi: torch.Tensor):
    return (
        5.2021359721285e-6
        * (1.0 - cos(theta) ** 2) ** 2.5
        * (
            196077717.773438 * cos(theta) ** 8
            - 219607043.90625 * cos(theta) ** 6
            + 71610992.578125 * cos(theta) ** 4
            - 6820094.53125 * cos(theta) ** 2
            + 89738.0859375
        )
        * cos(5 * phi)
    )


@torch.jit.script
def Yl13_m6(theta: torch.Tensor, phi: torch.Tensor):
    return (
        4.21948945157073e-7
        * (1.0 - cos(theta) ** 2) ** 3
        * (
            1568621742.1875 * cos(theta) ** 7
            - 1317642263.4375 * cos(theta) ** 5
            + 286443970.3125 * cos(theta) ** 3
            - 13640189.0625 * cos(theta)
        )
        * cos(6 * phi)
    )


@torch.jit.script
def Yl13_m7(theta: torch.Tensor, phi: torch.Tensor):
    return (
        3.5661194627771e-8
        * (1.0 - cos(theta) ** 2) ** 3.5
        * (
            10980352195.3125 * cos(theta) ** 6
            - 6588211317.1875 * cos(theta) ** 4
            + 859331910.9375 * cos(theta) ** 2
            - 13640189.0625
        )
        * cos(7 * phi)
    )


@torch.jit.script
def Yl13_m8(theta: torch.Tensor, phi: torch.Tensor):
    return (
        3.17695172143292e-9
        * (1.0 - cos(theta) ** 2) ** 4
        * (
            65882113171.875 * cos(theta) ** 5
            - 26352845268.75 * cos(theta) ** 3
            + 1718663821.875 * cos(theta)
        )
        * cos(8 * phi)
    )


@torch.jit.script
def Yl13_m9(theta: torch.Tensor, phi: torch.Tensor):
    return (
        3.02910461422567e-10
        * (1.0 - cos(theta) ** 2) ** 4.5
        * (
            329410565859.375 * cos(theta) ** 4
            - 79058535806.25 * cos(theta) ** 2
            + 1718663821.875
        )
        * cos(9 * phi)
    )


@torch.jit.script
def Yl13_m10(theta: torch.Tensor, phi: torch.Tensor):
    return (
        3.15805986876424e-11
        * (1.0 - cos(theta) ** 2) ** 5
        * (1317642263437.5 * cos(theta) ** 3 - 158117071612.5 * cos(theta))
        * cos(10 * phi)
    )


@torch.jit.script
def Yl13_m11(theta: torch.Tensor, phi: torch.Tensor):
    return (
        3.72180924766049e-12
        * (1.0 - cos(theta) ** 2) ** 5.5
        * (3952926790312.5 * cos(theta) ** 2 - 158117071612.5)
        * cos(11 * phi)
    )


@torch.jit.script
def Yl13_m12(theta: torch.Tensor, phi: torch.Tensor):
    return 4.16119315354964 * (1.0 - cos(theta) ** 2) ** 6 * cos(12 * phi) * cos(theta)


@torch.jit.script
def Yl13_m13(theta: torch.Tensor, phi: torch.Tensor):
    return 0.816077118837628 * (1.0 - cos(theta) ** 2) ** 6.5 * cos(13 * phi)


@torch.jit.script
def Yl14_m_minus_14(theta: torch.Tensor, phi: torch.Tensor):
    return 0.830522083064524 * (1.0 - cos(theta) ** 2) ** 7 * sin(14 * phi)


@torch.jit.script
def Yl14_m_minus_13(theta: torch.Tensor, phi: torch.Tensor):
    return (
        4.39470978027212 * (1.0 - cos(theta) ** 2) ** 6.5 * sin(13 * phi) * cos(theta)
    )


@torch.jit.script
def Yl14_m_minus_12(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.51291507116349e-13
        * (1.0 - cos(theta) ** 2) ** 6
        * (106729023338438.0 * cos(theta) ** 2 - 3952926790312.5)
        * sin(12 * phi)
    )


@torch.jit.script
def Yl14_m_minus_11(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.33617041195793e-12
        * (1.0 - cos(theta) ** 2) ** 5.5
        * (35576341112812.5 * cos(theta) ** 3 - 3952926790312.5 * cos(theta))
        * sin(11 * phi)
    )


@torch.jit.script
def Yl14_m_minus_10(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.33617041195793e-11
        * (1.0 - cos(theta) ** 2) ** 5
        * (
            8894085278203.13 * cos(theta) ** 4
            - 1976463395156.25 * cos(theta) ** 2
            + 39529267903.125
        )
        * sin(10 * phi)
    )


@torch.jit.script
def Yl14_m_minus_9(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.46370135060066e-10
        * (1.0 - cos(theta) ** 2) ** 4.5
        * (
            1778817055640.63 * cos(theta) ** 5
            - 658821131718.75 * cos(theta) ** 3
            + 39529267903.125 * cos(theta)
        )
        * sin(9 * phi)
    )


@torch.jit.script
def Yl14_m_minus_8(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.71945976061531e-9
        * (1.0 - cos(theta) ** 2) ** 4
        * (
            296469509273.438 * cos(theta) ** 6
            - 164705282929.688 * cos(theta) ** 4
            + 19764633951.5625 * cos(theta) ** 2
            - 286443970.3125
        )
        * sin(8 * phi)
    )


@torch.jit.script
def Yl14_m_minus_7(theta: torch.Tensor, phi: torch.Tensor):
    return (
        2.13379344766496e-8
        * (1.0 - cos(theta) ** 2) ** 3.5
        * (
            42352787039.0625 * cos(theta) ** 7
            - 32941056585.9375 * cos(theta) ** 5
            + 6588211317.1875 * cos(theta) ** 3
            - 286443970.3125 * cos(theta)
        )
        * sin(7 * phi)
    )


@torch.jit.script
def Yl14_m_minus_6(theta: torch.Tensor, phi: torch.Tensor):
    return (
        2.76571240765567e-7
        * (1.0 - cos(theta) ** 2) ** 3
        * (
            5294098379.88281 * cos(theta) ** 8
            - 5490176097.65625 * cos(theta) ** 6
            + 1647052829.29688 * cos(theta) ** 4
            - 143221985.15625 * cos(theta) ** 2
            + 1705023.6328125
        )
        * sin(6 * phi)
    )


@torch.jit.script
def Yl14_m_minus_5(theta: torch.Tensor, phi: torch.Tensor):
    return (
        3.71059256983961e-6
        * (1.0 - cos(theta) ** 2) ** 2.5
        * (
            588233153.320313 * cos(theta) ** 9
            - 784310871.09375 * cos(theta) ** 7
            + 329410565.859375 * cos(theta) ** 5
            - 47740661.71875 * cos(theta) ** 3
            + 1705023.6328125 * cos(theta)
        )
        * sin(5 * phi)
    )


@torch.jit.script
def Yl14_m_minus_4(theta: torch.Tensor, phi: torch.Tensor):
    return (
        5.11469888818129e-5
        * (1.0 - cos(theta) ** 2) ** 2
        * (
            58823315.3320313 * cos(theta) ** 10
            - 98038858.8867188 * cos(theta) ** 8
            + 54901760.9765625 * cos(theta) ** 6
            - 11935165.4296875 * cos(theta) ** 4
            + 852511.81640625 * cos(theta) ** 2
            - 8973.80859375
        )
        * sin(4 * phi)
    )


@torch.jit.script
def Yl14_m_minus_3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.000719701928156307
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (
            5347574.12109375 * cos(theta) ** 11
            - 10893206.5429688 * cos(theta) ** 9
            + 7843108.7109375 * cos(theta) ** 7
            - 2387033.0859375 * cos(theta) ** 5
            + 284170.60546875 * cos(theta) ** 3
            - 8973.80859375 * cos(theta)
        )
        * sin(3 * phi)
    )


@torch.jit.script
def Yl14_m_minus_2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0102793996196251
        * (1.0 - cos(theta) ** 2)
        * (
            445631.176757813 * cos(theta) ** 12
            - 1089320.65429688 * cos(theta) ** 10
            + 980388.588867188 * cos(theta) ** 8
            - 397838.84765625 * cos(theta) ** 6
            + 71042.6513671875 * cos(theta) ** 4
            - 4486.904296875 * cos(theta) ** 2
            + 43.9892578125
        )
        * sin(2 * phi)
    )


@torch.jit.script
def Yl14_m_minus_1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.148251609638173
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (
            34279.3212890625 * cos(theta) ** 13
            - 99029.150390625 * cos(theta) ** 11
            + 108932.065429688 * cos(theta) ** 9
            - 56834.12109375 * cos(theta) ** 7
            + 14208.5302734375 * cos(theta) ** 5
            - 1495.634765625 * cos(theta) ** 3
            + 43.9892578125 * cos(theta)
        )
        * sin(phi)
    )


@torch.jit.script
def Yl14_m0(theta: torch.Tensor, phi: torch.Tensor):
    return (
        3719.61718745389 * cos(theta) ** 14
        - 12536.487557715 * cos(theta) ** 12
        + 16548.1635761838 * cos(theta) ** 10
        - 10792.2805931633 * cos(theta) ** 8
        + 3597.42686438778 * cos(theta) ** 6
        - 568.014768061228 * cos(theta) ** 4
        + 33.4126334153663 * cos(theta) ** 2
        - 0.318215556336822
    )


@torch.jit.script
def Yl14_m1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.148251609638173
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (
            34279.3212890625 * cos(theta) ** 13
            - 99029.150390625 * cos(theta) ** 11
            + 108932.065429688 * cos(theta) ** 9
            - 56834.12109375 * cos(theta) ** 7
            + 14208.5302734375 * cos(theta) ** 5
            - 1495.634765625 * cos(theta) ** 3
            + 43.9892578125 * cos(theta)
        )
        * cos(phi)
    )


@torch.jit.script
def Yl14_m2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.0102793996196251
        * (1.0 - cos(theta) ** 2)
        * (
            445631.176757813 * cos(theta) ** 12
            - 1089320.65429688 * cos(theta) ** 10
            + 980388.588867188 * cos(theta) ** 8
            - 397838.84765625 * cos(theta) ** 6
            + 71042.6513671875 * cos(theta) ** 4
            - 4486.904296875 * cos(theta) ** 2
            + 43.9892578125
        )
        * cos(2 * phi)
    )


@torch.jit.script
def Yl14_m3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.000719701928156307
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (
            5347574.12109375 * cos(theta) ** 11
            - 10893206.5429688 * cos(theta) ** 9
            + 7843108.7109375 * cos(theta) ** 7
            - 2387033.0859375 * cos(theta) ** 5
            + 284170.60546875 * cos(theta) ** 3
            - 8973.80859375 * cos(theta)
        )
        * cos(3 * phi)
    )


@torch.jit.script
def Yl14_m4(theta: torch.Tensor, phi: torch.Tensor):
    return (
        5.11469888818129e-5
        * (1.0 - cos(theta) ** 2) ** 2
        * (
            58823315.3320313 * cos(theta) ** 10
            - 98038858.8867188 * cos(theta) ** 8
            + 54901760.9765625 * cos(theta) ** 6
            - 11935165.4296875 * cos(theta) ** 4
            + 852511.81640625 * cos(theta) ** 2
            - 8973.80859375
        )
        * cos(4 * phi)
    )


@torch.jit.script
def Yl14_m5(theta: torch.Tensor, phi: torch.Tensor):
    return (
        3.71059256983961e-6
        * (1.0 - cos(theta) ** 2) ** 2.5
        * (
            588233153.320313 * cos(theta) ** 9
            - 784310871.09375 * cos(theta) ** 7
            + 329410565.859375 * cos(theta) ** 5
            - 47740661.71875 * cos(theta) ** 3
            + 1705023.6328125 * cos(theta)
        )
        * cos(5 * phi)
    )


@torch.jit.script
def Yl14_m6(theta: torch.Tensor, phi: torch.Tensor):
    return (
        2.76571240765567e-7
        * (1.0 - cos(theta) ** 2) ** 3
        * (
            5294098379.88281 * cos(theta) ** 8
            - 5490176097.65625 * cos(theta) ** 6
            + 1647052829.29688 * cos(theta) ** 4
            - 143221985.15625 * cos(theta) ** 2
            + 1705023.6328125
        )
        * cos(6 * phi)
    )


@torch.jit.script
def Yl14_m7(theta: torch.Tensor, phi: torch.Tensor):
    return (
        2.13379344766496e-8
        * (1.0 - cos(theta) ** 2) ** 3.5
        * (
            42352787039.0625 * cos(theta) ** 7
            - 32941056585.9375 * cos(theta) ** 5
            + 6588211317.1875 * cos(theta) ** 3
            - 286443970.3125 * cos(theta)
        )
        * cos(7 * phi)
    )


@torch.jit.script
def Yl14_m8(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.71945976061531e-9
        * (1.0 - cos(theta) ** 2) ** 4
        * (
            296469509273.438 * cos(theta) ** 6
            - 164705282929.688 * cos(theta) ** 4
            + 19764633951.5625 * cos(theta) ** 2
            - 286443970.3125
        )
        * cos(8 * phi)
    )


@torch.jit.script
def Yl14_m9(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.46370135060066e-10
        * (1.0 - cos(theta) ** 2) ** 4.5
        * (
            1778817055640.63 * cos(theta) ** 5
            - 658821131718.75 * cos(theta) ** 3
            + 39529267903.125 * cos(theta)
        )
        * cos(9 * phi)
    )


@torch.jit.script
def Yl14_m10(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.33617041195793e-11
        * (1.0 - cos(theta) ** 2) ** 5
        * (
            8894085278203.13 * cos(theta) ** 4
            - 1976463395156.25 * cos(theta) ** 2
            + 39529267903.125
        )
        * cos(10 * phi)
    )


@torch.jit.script
def Yl14_m11(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.33617041195793e-12
        * (1.0 - cos(theta) ** 2) ** 5.5
        * (35576341112812.5 * cos(theta) ** 3 - 3952926790312.5 * cos(theta))
        * cos(11 * phi)
    )


@torch.jit.script
def Yl14_m12(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.51291507116349e-13
        * (1.0 - cos(theta) ** 2) ** 6
        * (106729023338438.0 * cos(theta) ** 2 - 3952926790312.5)
        * cos(12 * phi)
    )


@torch.jit.script
def Yl14_m13(theta: torch.Tensor, phi: torch.Tensor):
    return (
        4.39470978027212 * (1.0 - cos(theta) ** 2) ** 6.5 * cos(13 * phi) * cos(theta)
    )


@torch.jit.script
def Yl14_m14(theta: torch.Tensor, phi: torch.Tensor):
    return 0.830522083064524 * (1.0 - cos(theta) ** 2) ** 7 * cos(14 * phi)


@torch.jit.script
def Yl15_m_minus_15(theta: torch.Tensor, phi: torch.Tensor):
    return 0.844250650857373 * (1.0 - cos(theta) ** 2) ** 7.5 * sin(15 * phi)


@torch.jit.script
def Yl15_m_minus_14(theta: torch.Tensor, phi: torch.Tensor):
    return 4.62415125663001 * (1.0 - cos(theta) ** 2) ** 7 * sin(14 * phi) * cos(theta)


@torch.jit.script
def Yl15_m_minus_13(theta: torch.Tensor, phi: torch.Tensor):
    return (
        5.68899431025918e-15
        * (1.0 - cos(theta) ** 2) ** 6.5
        * (3.09514167681469e15 * cos(theta) ** 2 - 106729023338438.0)
        * sin(13 * phi)
    )


@torch.jit.script
def Yl15_m_minus_12(theta: torch.Tensor, phi: torch.Tensor):
    return (
        5.21404941098716e-14
        * (1.0 - cos(theta) ** 2) ** 6
        * (1.03171389227156e15 * cos(theta) ** 3 - 106729023338438.0 * cos(theta))
        * sin(12 * phi)
    )


@torch.jit.script
def Yl15_m_minus_11(theta: torch.Tensor, phi: torch.Tensor):
    return (
        5.4185990958026e-13
        * (1.0 - cos(theta) ** 2) ** 5.5
        * (
            257928473067891.0 * cos(theta) ** 4
            - 53364511669218.8 * cos(theta) ** 2
            + 988231697578.125
        )
        * sin(11 * phi)
    )


@torch.jit.script
def Yl15_m_minus_10(theta: torch.Tensor, phi: torch.Tensor):
    return (
        6.17815352749854e-12
        * (1.0 - cos(theta) ** 2) ** 5
        * (
            51585694613578.1 * cos(theta) ** 5
            - 17788170556406.3 * cos(theta) ** 3
            + 988231697578.125 * cos(theta)
        )
        * sin(10 * phi)
    )


@torch.jit.script
def Yl15_m_minus_9(theta: torch.Tensor, phi: torch.Tensor):
    return (
        7.56666184747369e-11
        * (1.0 - cos(theta) ** 2) ** 4.5
        * (
            8597615768929.69 * cos(theta) ** 6
            - 4447042639101.56 * cos(theta) ** 4
            + 494115848789.063 * cos(theta) ** 2
            - 6588211317.1875
        )
        * sin(9 * phi)
    )


@torch.jit.script
def Yl15_m_minus_8(theta: torch.Tensor, phi: torch.Tensor):
    return (
        9.80751467720255e-10
        * (1.0 - cos(theta) ** 2) ** 4
        * (
            1228230824132.81 * cos(theta) ** 7
            - 889408527820.313 * cos(theta) ** 5
            + 164705282929.688 * cos(theta) ** 3
            - 6588211317.1875 * cos(theta)
        )
        * sin(8 * phi)
    )


@torch.jit.script
def Yl15_m_minus_7(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.33035601710264e-8
        * (1.0 - cos(theta) ** 2) ** 3.5
        * (
            153528853016.602 * cos(theta) ** 8
            - 148234754636.719 * cos(theta) ** 6
            + 41176320732.4219 * cos(theta) ** 4
            - 3294105658.59375 * cos(theta) ** 2
            + 35805496.2890625
        )
        * sin(7 * phi)
    )


@torch.jit.script
def Yl15_m_minus_6(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.87197684863824e-7
        * (1.0 - cos(theta) ** 2) ** 3
        * (
            17058761446.2891 * cos(theta) ** 9
            - 21176393519.5313 * cos(theta) ** 7
            + 8235264146.48438 * cos(theta) ** 5
            - 1098035219.53125 * cos(theta) ** 3
            + 35805496.2890625 * cos(theta)
        )
        * sin(6 * phi)
    )


@torch.jit.script
def Yl15_m_minus_5(theta: torch.Tensor, phi: torch.Tensor):
    return (
        2.71275217737612e-6
        * (1.0 - cos(theta) ** 2) ** 2.5
        * (
            1705876144.62891 * cos(theta) ** 10
            - 2647049189.94141 * cos(theta) ** 8
            + 1372544024.41406 * cos(theta) ** 6
            - 274508804.882813 * cos(theta) ** 4
            + 17902748.1445313 * cos(theta) ** 2
            - 170502.36328125
        )
        * sin(5 * phi)
    )


@torch.jit.script
def Yl15_m_minus_4(theta: torch.Tensor, phi: torch.Tensor):
    return (
        4.02366171874445e-5
        * (1.0 - cos(theta) ** 2) ** 2
        * (
            155079649.511719 * cos(theta) ** 11
            - 294116576.660156 * cos(theta) ** 9
            + 196077717.773438 * cos(theta) ** 7
            - 54901760.9765625 * cos(theta) ** 5
            + 5967582.71484375 * cos(theta) ** 3
            - 170502.36328125 * cos(theta)
        )
        * sin(4 * phi)
    )


@torch.jit.script
def Yl15_m_minus_3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.000607559596001151
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (
            12923304.1259766 * cos(theta) ** 12
            - 29411657.6660156 * cos(theta) ** 10
            + 24509714.7216797 * cos(theta) ** 8
            - 9150293.49609375 * cos(theta) ** 6
            + 1491895.67871094 * cos(theta) ** 4
            - 85251.181640625 * cos(theta) ** 2
            + 747.8173828125
        )
        * sin(3 * phi)
    )


@torch.jit.script
def Yl15_m_minus_2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.00929387470704126
        * (1.0 - cos(theta) ** 2)
        * (
            994100.317382813 * cos(theta) ** 13
            - 2673787.06054688 * cos(theta) ** 11
            + 2723301.63574219 * cos(theta) ** 9
            - 1307184.78515625 * cos(theta) ** 7
            + 298379.135742188 * cos(theta) ** 5
            - 28417.060546875 * cos(theta) ** 3
            + 747.8173828125 * cos(theta)
        )
        * sin(2 * phi)
    )


@torch.jit.script
def Yl15_m_minus_1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.143378915753688
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (
            71007.1655273438 * cos(theta) ** 14
            - 222815.588378906 * cos(theta) ** 12
            + 272330.163574219 * cos(theta) ** 10
            - 163398.098144531 * cos(theta) ** 8
            + 49729.8559570313 * cos(theta) ** 6
            - 7104.26513671875 * cos(theta) ** 4
            + 373.90869140625 * cos(theta) ** 2
            - 3.14208984375
        )
        * sin(phi)
    )


@torch.jit.script
def Yl15_m0(theta: torch.Tensor, phi: torch.Tensor):
    return (
        7435.10031825349 * cos(theta) ** 15
        - 26920.1908074695 * cos(theta) ** 13
        + 38884.7200552338 * cos(theta) ** 11
        - 28515.4613738381 * cos(theta) ** 9
        + 11158.2240158497 * cos(theta) ** 7
        - 2231.64480316994 * cos(theta) ** 5
        + 195.758316067539 * cos(theta) ** 3
        - 4.93508359834131 * cos(theta)
    )


@torch.jit.script
def Yl15_m1(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.143378915753688
        * (1.0 - cos(theta) ** 2) ** 0.5
        * (
            71007.1655273438 * cos(theta) ** 14
            - 222815.588378906 * cos(theta) ** 12
            + 272330.163574219 * cos(theta) ** 10
            - 163398.098144531 * cos(theta) ** 8
            + 49729.8559570313 * cos(theta) ** 6
            - 7104.26513671875 * cos(theta) ** 4
            + 373.90869140625 * cos(theta) ** 2
            - 3.14208984375
        )
        * cos(phi)
    )


@torch.jit.script
def Yl15_m2(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.00929387470704126
        * (1.0 - cos(theta) ** 2)
        * (
            994100.317382813 * cos(theta) ** 13
            - 2673787.06054688 * cos(theta) ** 11
            + 2723301.63574219 * cos(theta) ** 9
            - 1307184.78515625 * cos(theta) ** 7
            + 298379.135742188 * cos(theta) ** 5
            - 28417.060546875 * cos(theta) ** 3
            + 747.8173828125 * cos(theta)
        )
        * cos(2 * phi)
    )


@torch.jit.script
def Yl15_m3(theta: torch.Tensor, phi: torch.Tensor):
    return (
        0.000607559596001151
        * (1.0 - cos(theta) ** 2) ** 1.5
        * (
            12923304.1259766 * cos(theta) ** 12
            - 29411657.6660156 * cos(theta) ** 10
            + 24509714.7216797 * cos(theta) ** 8
            - 9150293.49609375 * cos(theta) ** 6
            + 1491895.67871094 * cos(theta) ** 4
            - 85251.181640625 * cos(theta) ** 2
            + 747.8173828125
        )
        * cos(3 * phi)
    )


@torch.jit.script
def Yl15_m4(theta: torch.Tensor, phi: torch.Tensor):
    return (
        4.02366171874445e-5
        * (1.0 - cos(theta) ** 2) ** 2
        * (
            155079649.511719 * cos(theta) ** 11
            - 294116576.660156 * cos(theta) ** 9
            + 196077717.773438 * cos(theta) ** 7
            - 54901760.9765625 * cos(theta) ** 5
            + 5967582.71484375 * cos(theta) ** 3
            - 170502.36328125 * cos(theta)
        )
        * cos(4 * phi)
    )


@torch.jit.script
def Yl15_m5(theta: torch.Tensor, phi: torch.Tensor):
    return (
        2.71275217737612e-6
        * (1.0 - cos(theta) ** 2) ** 2.5
        * (
            1705876144.62891 * cos(theta) ** 10
            - 2647049189.94141 * cos(theta) ** 8
            + 1372544024.41406 * cos(theta) ** 6
            - 274508804.882813 * cos(theta) ** 4
            + 17902748.1445313 * cos(theta) ** 2
            - 170502.36328125
        )
        * cos(5 * phi)
    )


@torch.jit.script
def Yl15_m6(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.87197684863824e-7
        * (1.0 - cos(theta) ** 2) ** 3
        * (
            17058761446.2891 * cos(theta) ** 9
            - 21176393519.5313 * cos(theta) ** 7
            + 8235264146.48438 * cos(theta) ** 5
            - 1098035219.53125 * cos(theta) ** 3
            + 35805496.2890625 * cos(theta)
        )
        * cos(6 * phi)
    )


@torch.jit.script
def Yl15_m7(theta: torch.Tensor, phi: torch.Tensor):
    return (
        1.33035601710264e-8
        * (1.0 - cos(theta) ** 2) ** 3.5
        * (
            153528853016.602 * cos(theta) ** 8
            - 148234754636.719 * cos(theta) ** 6
            + 41176320732.4219 * cos(theta) ** 4
            - 3294105658.59375 * cos(theta) ** 2
            + 35805496.2890625
        )
        * cos(7 * phi)
    )


@torch.jit.script
def Yl15_m8(theta: torch.Tensor, phi: torch.Tensor):
    return (
        9.80751467720255e-10
        * (1.0 - cos(theta) ** 2) ** 4
        * (
            1228230824132.81 * cos(theta) ** 7
            - 889408527820.313 * cos(theta) ** 5
            + 164705282929.688 * cos(theta) ** 3
            - 6588211317.1875 * cos(theta)
        )
        * cos(8 * phi)
    )


@torch.jit.script
def Yl15_m9(theta: torch.Tensor, phi: torch.Tensor):
    return (
        7.56666184747369e-11
        * (1.0 - cos(theta) ** 2) ** 4.5
        * (
            8597615768929.69 * cos(theta) ** 6
            - 4447042639101.56 * cos(theta) ** 4
            + 494115848789.063 * cos(theta) ** 2
            - 6588211317.1875
        )
        * cos(9 * phi)
    )


@torch.jit.script
def Yl15_m10(theta: torch.Tensor, phi: torch.Tensor):
    return (
        6.17815352749854e-12
        * (1.0 - cos(theta) ** 2) ** 5
        * (
            51585694613578.1 * cos(theta) ** 5
            - 17788170556406.3 * cos(theta) ** 3
            + 988231697578.125 * cos(theta)
        )
        * cos(10 * phi)
    )


@torch.jit.script
def Yl15_m11(theta: torch.Tensor, phi: torch.Tensor):
    return (
        5.4185990958026e-13
        * (1.0 - cos(theta) ** 2) ** 5.5
        * (
            257928473067891.0 * cos(theta) ** 4
            - 53364511669218.8 * cos(theta) ** 2
            + 988231697578.125
        )
        * cos(11 * phi)
    )


@torch.jit.script
def Yl15_m12(theta: torch.Tensor, phi: torch.Tensor):
    return (
        5.21404941098716e-14
        * (1.0 - cos(theta) ** 2) ** 6
        * (1.03171389227156e15 * cos(theta) ** 3 - 106729023338438.0 * cos(theta))
        * cos(12 * phi)
    )


@torch.jit.script
def Yl15_m13(theta: torch.Tensor, phi: torch.Tensor):
    return (
        5.68899431025918e-15
        * (1.0 - cos(theta) ** 2) ** 6.5
        * (3.09514167681469e15 * cos(theta) ** 2 - 106729023338438.0)
        * cos(13 * phi)
    )


@torch.jit.script
def Yl15_m14(theta: torch.Tensor, phi: torch.Tensor):
    return 4.62415125663001 * (1.0 - cos(theta) ** 2) ** 7 * cos(14 * phi) * cos(theta)


@torch.jit.script
def Yl15_m15(theta: torch.Tensor, phi: torch.Tensor):
    return 0.844250650857373 * (1.0 - cos(theta) ** 2) ** 7.5 * cos(15 * phi)
