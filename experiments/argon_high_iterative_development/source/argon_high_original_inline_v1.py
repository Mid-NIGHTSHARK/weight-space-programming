"""Reconstruction of the user-supplied first-prompt Argon High assignments.

The assignment block supplied in the conversation did not include its array
initialization or complete provider response. The project architecture uses
zero-initialized tensors, so this function applies those assignments to zero
arrays of the documented 600-32-32-6 shape. This is a preserved transcription
of the supplied assignments, not an authenticated provider export.
"""

import numpy as np


def build_network():
    W1 = np.zeros((600, 32), dtype=np.float64)
    b1 = np.zeros((32,), dtype=np.float64)
    W2 = np.zeros((32, 32), dtype=np.float64)
    b2 = np.zeros((32,), dtype=np.float64)
    W3 = np.zeros((32, 6), dtype=np.float64)
    b3 = np.zeros((6,), dtype=np.float64)

    # Paddle position P(pos) = -30 * pos on row 9, channel 0, units 0..3.
    W1[540:600:6, 0] = [0, -30, -60, -90, -120, -150, -180, -210, -240, -270]
    W1[540:600:6, 1] = [0, -30, -60, -90, -120, -150, -180, -210, -240, -270]
    W1[540:600:6, 2] = [0, -30, -60, -90, -120, -150, -180, -210, -240, -270]
    W1[540:600:6, 3] = [0, -30, -60, -90, -120, -150, -180, -210, -240, -270]

    # Trail position T(last_x) = -30 * last_x on channel 4, units 0 and 1.
    W1[10:600:60, 0:2] = -30
    W1[16:600:60, 0:2] = -60
    W1[22:600:60, 0:2] = -90
    W1[28:600:60, 0:2] = -120
    W1[34:600:60, 0:2] = -150
    W1[40:600:60, 0:2] = -180
    W1[46:600:60, 0:2] = -210
    W1[52:600:60, 0:2] = -240
    W1[58:600:60, 0:2] = -270

    # Ball position B(ball_x) = 30 * [1,2,4,6,8,10,12,14,16,17].
    W1[3:600:60, 0:2] = 30
    W1[9:600:60, 0:2] = 60
    W1[15:600:60, 0:2] = 120
    W1[21:600:60, 0:2] = 180
    W1[27:600:60, 0:2] = 240
    W1[33:600:60, 0:2] = 300
    W1[39:600:60, 0:2] = 360
    W1[45:600:60, 0:2] = 420
    W1[51:600:60, 0:2] = 480
    W1[57:600:60, 0:2] = 510

    # Disable the general predictor when the ball is on row 8 or 9.
    W1[483:600:6, 0] = -1000
    W1[483:600:6, 1] = 1000

    # Units 2 and 3 compare ball_x with paddle position only on row 8.
    W1[3:483:6, 2] = -1000
    W1[3:483:6, 3] = 1000
    W1[543:600:6, 2] = -1000
    W1[543:600:6, 3] = 1000
    W1[483:543:6, 2] = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]
    W1[483:543:6, 3] = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]

    # Unit 4: Start-1 midpoint detector (ball=(5,2), trail=(4,1), pos=4).
    W1[315, 4] = 20
    W1[250, 4] = 20
    W1[564, 4] = 20

    b1[0:5] = [-15, 15, -15, 15, -50]

    # Layer 2: h2[0] = Move Right, h2[1] = Move Left.
    W2[0, 0] = 20
    W2[2, 0] = 20
    b2[0] = 10

    W2[1, 1] = -20
    W2[3, 1] = -20
    W2[4, 1] = -20
    b2[1] = -10

    # Output layer: action 0 = no-op, action 1 = left, action 3 = right.
    W3[0, 0] = -50
    W3[1, 0] = -50
    W3[1, 1] = 50
    W3[0, 3] = 50
    b3[0:6] = [-50, 0, -100, 0, -100, -100]

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
