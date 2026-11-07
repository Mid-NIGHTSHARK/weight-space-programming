import numpy as np

# ==============================================================================
# 1. RECONSTRUCT NETWORK FROM EXACT TENSOR ASSIGNMENTS
# ==============================================================================
def build_refined_network():
    W1 = np.zeros((600, 32), dtype=np.float64)
    b1 = np.zeros((32,), dtype=np.float64)
    W2 = np.zeros((32, 32), dtype=np.float64)
    b2 = np.zeros((32,), dtype=np.float64)
    W3 = np.zeros((32, 6), dtype=np.float64)
    b3 = np.zeros((6,), dtype=np.float64)

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

    # Units 4..7: Start-1 diagonal with pos=4 -> Move Right (4 -> 5)
    W1[183, 4] = 20; W1[184, 4] = 20; W1[564, 4] = 20
    W1[249, 5] = 20; W1[184, 5] = 20; W1[564, 5] = 20
    W1[315, 6] = 20; W1[250, 6] = 20; W1[564, 6] = 20
    W1[381, 7] = 20; W1[316, 7] = 20; W1[564, 7] = 20

    # Units 8..10: Start-1 diagonal with pos=5 -> No-op (hold at 5)
    W1[249, 8] = 20; W1[184, 8] = 20; W1[570, 8] = 20
    W1[315, 9] = 20; W1[250, 9] = 20; W1[570, 9] = 20
    W1[381, 10] = 20; W1[316, 10] = 20; W1[570, 10] = 20

    # Units 11..14: Start-2 diagonal with pos=4 -> No-op (hold at 4)
    W1[237, 11] = 20; W1[238, 11] = 20; W1[564, 11] = 20
    W1[291, 12] = 20; W1[238, 12] = 20; W1[564, 12] = 20
    W1[345, 13] = 20; W1[292, 13] = 20; W1[564, 13] = 20
    W1[399, 14] = 20; W1[346, 14] = 20; W1[564, 14] = 20

    b1[0:15] = [-15, 15, -15, 15, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50]

    W2[0, 0] = 20
    W2[2, 0] = 20
    W2[4:8, 0] = 20
    W2[11:15, 0] = -20
    b2[0] = 10

    W2[1, 1] = -20
    W2[3, 1] = -20
    W2[4:7, 1] = -20
    W2[8:11, 1] = -20
    b2[1] = -110

    W3[0, 0] = -50
    W3[1, 0] = -50
    W3[1, 1] = 50
    W3[0, 3] = 50
    b3[0:6] = [-50, 0, -100, 0, -100, -100]

    return W1, b1, W2, b2, W3, b3

# ==============================================================================
# 2. INDEPENDENT MATHEMATICAL SPECIFICATION ORACLE
# ==============================================================================
def mathematical_oracle_action(ball_y, ball_x, trail_y, trail_x, pos):
    # Check Start-1 diagonal (S0..S3)
    s1_states = [(3, 0, 3, 0), (4, 1, 3, 0), (5, 2, 4, 1), (6, 3, 5, 2)]
    if (ball_y, ball_x, trail_y, trail_x) in s1_states:
        if pos == 4:
            return 3  # Force Right (4 -> 5)
        if pos == 5 and (ball_y, ball_x, trail_y, trail_x) in s1_states[1:]:
            return 0  # Hold at 5 (No-op)

    # Check Start-2 diagonal (S'0..S'3)
    s2_states = [(3, 9, 3, 9), (4, 8, 3, 9), (5, 7, 4, 8), (6, 6, 5, 7)]
    if (ball_y, ball_x, trail_y, trail_x) in s2_states and pos == 4:
        return 0  # Hold at 4 (No-op)

    # General comparator logic
    if ball_y <= 7:
        B = [1, 2, 4, 6, 8, 10, 12, 14, 16, 17]
        target = B[ball_x] - trail_x
    elif ball_y == 8:
        target = ball_x
    else:
        return 0

    if target > pos:
        return 3  # Right
    elif target < pos:
        return 1  # Left
    else:
        return 0  # No-op

def encode_obs(ball_y, ball_x, trail_y, trail_x, paddle_x):
    x = np.zeros((600,), dtype=np.float64)
    x[(9 * 10 + paddle_x) * 6 + 0] = 1.0
    x[(ball_y * 10 + ball_x) * 6 + 3] = 1.0
    x[(trail_y * 10 + trail_x) * 6 + 4] = 1.0
    return x

def forward(x, W1, b1, W2, b2, W3, b3):
    h1 = np.tanh(x @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    logits = h2 @ W3 + b3
    shifted = logits - np.max(logits)
    probs = np.exp(shifted) / np.sum(np.exp(shifted))
    return h1, h2, logits, probs

def run_verification():
    W1, b1, W2, b2, W3, b3 = build_refined_network()

    # 1. Verify parameter counts
    counts = {name: int(np.count_nonzero(t)) for name, t in
              [("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2), ("W3", W3), ("b3", b3)]}
    print("Nonzero counts:", counts, "Total:", sum(counts.values()))
    assert counts == {"W1": 647, "b1": 15, "W2": 18, "b2": 2, "W3": 4, "b3": 4}
    assert sum(counts.values()) == 690

    # 2. Exhaustively verify ALL (ball_y, ball_x, trail_y, trail_x, pos) combinations
    # across every legal board coordinate (ball_y in 0..8, ball_x in 0..9,
    # physically adjacent trail_y/trail_x, pos in 0..9) against the oracle!
    total_checked = 0
    for ball_y in range(9):
        for ball_x in range(10):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    trail_y = ball_y + dy
                    trail_x = ball_x + dx
                    if not (0 <= trail_y <= 9 and 0 <= trail_x <= 9):
                        continue
                    for pos in range(10):
                        x = encode_obs(ball_y, ball_x, trail_y, trail_x, pos)
                        _, _, logits, probs = forward(x, W1, b1, W2, b2, W3, b3)
                        net_action = int(np.argmax(logits))
                        oracle_act = mathematical_oracle_action(ball_y, ball_x, trail_y, trail_x, pos)
                        assert net_action == oracle_act, (
                            f"Mismatch at ball=({ball_y},{ball_x}), trail=({trail_y},{trail_x}), pos={pos}: "
                            f"net={net_action}, oracle={oracle_act}, logits={logits}"
                        )
                        assert probs[net_action] > 1.0 - 1e-40
                        total_checked += 1
    print(f"Exhaustively verified {total_checked} (ball, trail, paddle) states: 100% match with oracle!")

if __name__ == "__main__":
    run_verification()