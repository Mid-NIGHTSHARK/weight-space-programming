import numpy as np

def build_network():
    W1 = np.zeros((600, 32), dtype=np.float64)
    b1 = np.zeros((32,), dtype=np.float64)
    W2 = np.zeros((32, 32), dtype=np.float64)
    b2 = np.zeros((32,), dtype=np.float64)
    W3 = np.zeros((32, 6), dtype=np.float64)
    b3 = np.zeros((6,), dtype=np.float64)

    # Apply exact corrected edits
    W1[:, :] = 0
    b1[:] = 0
    W2[:, :] = 0
    b2[:] = 0
    W3[:, :] = 0
    b3[:] = 0

    W1[540:600:6, 0] = [0, -30, -60, -90, -120, -150, -180, -210, -240, -270]
    W1[540:600:6, 1] = [0, -30, -60, -90, -120, -150, -180, -210, -240, -270]
    W1[540:600:6, 2] = [0, -30, -60, -90, -120, -150, -180, -210, -240, -270]
    W1[540:600:6, 3] = [0, -30, -60, -90, -120, -150, -180, -210, -240, -270]

    W1[10:600:60, 0:2] = -30
    W1[16:600:60, 0:2] = -60
    W1[22:600:60, 0:2] = -90
    W1[28:600:60, 0:2] = -120
    W1[34:600:60, 0:2] = -150
    W1[40:600:60, 0:2] = -180
    W1[46:600:60, 0:2] = -210
    W1[52:600:60, 0:2] = -240
    W1[58:600:60, 0:2] = -270

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

    W1[483:600:6, 0] = -1000
    W1[483:600:6, 1] = 1000

    W1[3:483:6, 2] = -1000
    W1[3:483:6, 3] = 1000
    W1[543:600:6, 2] = -1000
    W1[543:600:6, 3] = 1000
    W1[483:543:6, 2] = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]
    W1[483:543:6, 3] = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]

    W1[183, 4] = 20
    W1[184, 4] = 20
    W1[564, 4] = 20

    W1[249, 5] = 20
    W1[184, 5] = 20
    W1[564, 5] = 20

    W1[315, 6] = 20
    W1[250, 6] = 20
    W1[564, 6] = 20

    b1[0:7] = [-15, 15, -15, 15, -50, -50, -50]

    W2[0, 0] = 20
    W2[2, 0] = 20
    b2[0] = 10

    W2[1, 1] = -20
    W2[3, 1] = -20
    W2[4, 1] = -20
    W2[5, 1] = -20
    W2[6, 1] = -20
    b2[1] = -50

    W3[0, 0] = -50
    W3[1, 0] = -50
    W3[1, 1] = 50
    W3[0, 3] = 50
    b3[0:6] = [-50, 0, -100, 0, -100, -100]

    return W1, b1, W2, b2, W3, b3

def encode_obs(ball_y, ball_x, trail_y, trail_x, paddle_x, bricks=True):
    x = np.zeros((600,), dtype=np.float64)
    # Channel 0: paddle on row 9
    x[(9 * 10 + paddle_x) * 6 + 0] = 1.0
    # Channel 2: bricks on rows 1..3
    if bricks:
        for r in range(1, 4):
            for c in range(10):
                x[(r * 10 + c) * 6 + 2] = 1.0
    # Channel 3: ball
    x[(ball_y * 10 + ball_x) * 6 + 3] = 1.0
    # Channel 4: trail
    x[(trail_y * 10 + trail_x) * 6 + 4] = 1.0
    return x

def forward(x, W1, b1, W2, b2, W3, b3):
    h1 = np.tanh(x @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    logits = h2 @ W3 + b3
    shifted = logits - np.max(logits)
    probs = np.exp(shifted) / np.sum(np.exp(shifted))
    return h1, h2, logits, probs

def main():
    W1, b1, W2, b2, W3, b3 = build_network()

    # 1. Print nonzero parameter counts
    tensors = [("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2), ("W3", W3), ("b3", b3)]
    total_nz = 0
    print("=== NONZERO PARAMETER COUNTS ===")
    for name, arr in tensors:
        nz = int(np.count_nonzero(arr))
        total_nz += nz
        print(f"{name:3s}: {nz:4d} / {arr.size:5d}")
    print(f"Total nonzero parameters: {total_nz} / {sum(arr.size for _, arr in tensors)}\n")

    # 2. Print all nonzero input indices for each first-layer unit
    print("=== NONZERO INPUT INDICES PER FIRST-LAYER UNIT ===")
    for u in range(32):
        idx = np.nonzero(W1[:, u])[0].tolist()
        if idx:
            print(f"Unit {u:2d} ({len(idx):3d} nonzero): {idx}")
    print()

    # 3. Verify positive and nearby negative cases for special detectors (Units 4, 5, 6)
    print("=== DETECTOR VERIFICATION (UNITS 4, 5, 6) ===")
    detector_tests = [
        # Positive cases
        ("Unit 4 POS: ball=(3,0), trail=(3,0), pos=4", (3, 0, 3, 0, 4), (1.0, -1.0, -1.0)),
        ("Unit 5 POS: ball=(4,1), trail=(3,0), pos=4", (4, 1, 3, 0, 4), (-1.0, 1.0, -1.0)),
        ("Unit 6 POS: ball=(5,2), trail=(4,1), pos=4", (5, 2, 4, 1, 4), (-1.0, -1.0, 1.0)),
        # Nearby negative cases (1 feature changed)
        ("Unit 4 NEG (wrong pos=3): ball=(3,0), trail=(3,0), pos=3", (3, 0, 3, 0, 3), (-1.0, -1.0, -1.0)),
        ("Unit 4 NEG (wrong trail=(4,1)): ball=(3,0), trail=(4,1), pos=4", (3, 0, 4, 1, 4), (-1.0, -1.0, -1.0)),
        ("Unit 5 NEG (wrong ball=(4,2)): ball=(4,2), trail=(3,0), pos=4", (4, 2, 3, 0, 4), (-1.0, -1.0, -1.0)),
        ("Unit 5 NEG (wrong pos=2): ball=(4,1), trail=(3,0), pos=2", (4, 1, 3, 0, 2), (-1.0, -1.0, -1.0)),
        ("Unit 6 NEG (wrong trail=(6,3)): ball=(5,2), trail=(6,3), pos=4", (5, 2, 6, 3, 4), (-1.0, -1.0, -1.0)),
        ("Unit 6 NEG (wrong pos=5): ball=(5,2), trail=(4,1), pos=5", (5, 2, 4, 1, 5), (-1.0, -1.0, -1.0)),
    ]
    for label, state_args, expected in detector_tests:
        x = encode_obs(*state_args)
        h1, _, _, _ = forward(x, W1, b1, W2, b2, W3, b3)
        actual = (round(float(h1[4]), 4), round(float(h1[5]), 4), round(float(h1[6]), 4))
        assert actual == expected, f"Mismatch on {label}: got {actual}, expected {expected}"
        print(f"[PASS] {label:58s} -> h1[4:7] = {actual}")
    print()

    # 4. Evaluate representative board states
    action_names = ["0:No-op", "1:Left", "2:Up", "3:Right", "4:Down", "5:Fire"]
    print("=== REPRESENTATIVE BOARD STATE EVALUATIONS ===")
    rep_states = [
        ("Start 1 t=0 (ball=(3,0), trail=(3,0), pos=4 -> suppress L)", (3, 0, 3, 0, 4), 0),
        ("Start 1 t=1 (ball=(4,1), trail=(3,0), pos=4 -> suppress L)", (4, 1, 3, 0, 4), 0),
        ("Start 1 t=2 (ball=(5,2), trail=(4,1), pos=4 -> suppress L)", (5, 2, 4, 1, 4), 0),
        ("Start 1 t=3 (ball=(6,3), trail=(5,2), pos=4 -> next_x=4==pos)", (6, 3, 5, 2, 4), 0),
        ("Start 1 t=4 (ball=(7,4), trail=(6,3), pos=4 -> next_x=5>pos)", (7, 4, 6, 3, 4), 3),
        ("Start 1 t=5 (ball=(8,5), trail=(7,4), pos=5 -> catch row=8)", (8, 5, 7, 4, 5), 0),
        ("Start 2 t=0 (ball=(3,9), trail=(3,9), pos=4 -> next_x=8>pos)", (3, 9, 3, 9, 4), 3),
        ("Left wall arrival (ball=(5,0), trail=(4,1), pos=1 -> next_x=0<pos)", (5, 0, 4, 1, 1), 1),
        ("Left wall clamp (ball=(6,0), trail=(5,0), pos=0 -> next_x=1>pos)", (6, 0, 5, 0, 0), 3),
        ("Right wall arrival (ball=(5,9), trail=(4,8), pos=8 -> next_x=9>pos)", (5, 9, 4, 8, 8), 3),
        ("Right wall clamp (ball=(6,9), trail=(5,9), pos=9 -> next_x=8<pos)", (6, 9, 5, 9, 9), 1),
        ("Row 8 catch lag left (ball=(8,6), trail=(7,5), pos=5 -> ball_x=6>pos)", (8, 6, 7, 5, 5), 3),
        ("Row 8 catch lag right (ball=(8,3), trail=(7,4), pos=4 -> ball_x=3<pos)", (8, 3, 7, 4, 4), 1),
    ]
    for desc, state_args, expected_action in rep_states:
        x = encode_obs(*state_args)
        _, _, logits, probs = forward(x, W1, b1, W2, b2, W3, b3)
        best_action = int(np.argmax(logits))
        assert best_action == expected_action, f"Action mismatch on {desc}"
        logits_str = "[" + ", ".join(f"{v:6.1f}" for v in logits) + "]"
        probs_str = "[" + ", ".join(f"{p:.2e}" for p in probs) + "]"
        print(f"{desc}")
        print(f"  Logits: {logits_str}")
        print(f"  Probs:  {probs_str} -> Selected: {action_names[best_action]}")

if __name__ == "__main__":
    main()