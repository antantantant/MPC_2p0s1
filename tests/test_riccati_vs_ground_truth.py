"""
Test: Compare Riccati implementation vs ground truth for a single-player case.

This isolates the Riccati solver by testing it on a case where
we have an analytical solution.
"""
import torch

def discrete_lqr_ground_truth(Ad, Bd, Q, R, Qf, N):
    """
    Ground truth discrete-time LQR.
    Returns K matrices such that u_k = -K_k x_k.
    """
    K_matrices = []
    Pk = Qf.clone()
    
    for k in range(N, 0, -1):
        # Gain: K = (R + B^T P B)^{-1} B^T P A
        M = R + Bd.T @ Pk @ Bd
        Fk = torch.linalg.solve(M, Bd.T @ Pk @ Ad)
        
        # P update: P = K^T R K + (A - B K)^T P (A - B K)
        A_cl = Ad - Bd @ Fk
        Pk_new = Fk.T @ R @ Fk + A_cl.T @ Pk @ A_cl
        Pk = Pk_new
        
        K_matrices.insert(0, Fk)
    
    return K_matrices, Pk


def your_riccati_single_step(A, B, tau, R, P_plus):
    """
    Your Riccati step (single player, no affine terms).
    Adapted from _local_lq_saddle.
    """
    du = B.shape[1]
    dx = A.shape[0]
    
    # Hessian
    H_uu = tau * R + B.T @ P_plus @ B
    
    # Cross term
    F_u = B.T @ P_plus @ A
    
    # Solve for gain
    K_u = -torch.linalg.solve(H_uu, F_u)  # Note: negative
    
    # P update
    Q = A.T @ P_plus @ A
    sol_F = torch.linalg.solve(H_uu, F_u)
    P_loc = Q - F_u.T @ sol_F
    
    return P_loc, -K_u  # Return positive K for comparison


def test_single_player_riccati():
    """Test single-player LQR against ground truth."""
    dtype = torch.float64
    device = torch.device("cpu")
    
    # Simple double integrator
    tau = 0.1
    A = torch.tensor([
        [1.0, tau],
        [0.0, 1.0]
    ], dtype=dtype, device=device)
    
    B = torch.tensor([
        [0.5 * tau**2],
        [tau]
    ], dtype=dtype, device=device)
    
    Q_running = torch.zeros(2, 2, dtype=dtype, device=device)  # No running state cost
    R_base = torch.tensor([[1.0]], dtype=dtype, device=device)
    Qf = torch.eye(2, dtype=dtype, device=device)  # Terminal cost
    
    N = 5
    
    # Ground truth uses R * tau (to match your implementation's tau * R)
    R_scaled = R_base * tau
    K_gt, P0_gt = discrete_lqr_ground_truth(A, B, Q_running, R_scaled, Qf, N)
    
    print("=" * 60)
    print("SINGLE-PLAYER LQR TEST")
    print("=" * 60)
    print(f"\nGround truth K matrices (using R * tau):")
    for k, K in enumerate(K_gt):
        print(f"  K[{k}] = {K.numpy()}")
    print(f"\nGround truth P[0] = \n{P0_gt.numpy()}")
    
    # Your implementation (backward pass) uses tau * R internally
    P = Qf.clone()
    K_yours = []
    
    for k in range(N-1, -1, -1):
        P_new, K_k = your_riccati_single_step(A, B, tau, R_base, P)
        K_yours.insert(0, K_k)
        P = P_new
    
    print(f"\nYour K matrices (using tau * R):")
    for k, K in enumerate(K_yours):
        print(f"  K[{k}] = {K.numpy()}")
    print(f"\nYour P[0] = \n{P.numpy()}")
    
    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    K_match = all(torch.allclose(k1, k2, atol=1e-10) for k1, k2 in zip(K_gt, K_yours))
    P_match = torch.allclose(P0_gt, P, atol=1e-10)
    
    print(f"K matrices match: {K_match}")
    print(f"P[0] matches: {P_match}")
    
    if not K_match:
        print("\nK differences:")
        for k, (k1, k2) in enumerate(zip(K_gt, K_yours)):
            diff = (k1 - k2).abs().max()
            print(f"  K[{k}] max diff: {diff:.6e}")
    
    if not P_match:
        print(f"\nP[0] difference: {(P0_gt - P).abs().max():.6e}")
    
    return K_match and P_match


def test_with_tau_scaling():
    """
    Test if tau scaling in ground truth vs implementation matches.
    
    Ground truth uses: R_eff = R * tau
    Your code uses: H_uu = tau * R_bar + B^T P B
    
    These should be equivalent if R_bar = R.
    """
    dtype = torch.float64
    device = torch.device("cpu")
    
    tau = 0.1
    A = torch.tensor([
        [1.0, tau],
        [0.0, 1.0]
    ], dtype=dtype, device=device)
    
    B = torch.tensor([
        [0.5 * tau**2],
        [tau]
    ], dtype=dtype, device=device)
    
    Qf = torch.eye(2, dtype=dtype, device=device)
    N = 5
    
    # Ground truth with R * tau
    R_base = torch.tensor([[1.0]], dtype=dtype, device=device)
    R_scaled = R_base * tau
    K_gt, P0_gt = discrete_lqr_ground_truth(A, B, torch.zeros(2,2), R_scaled, Qf, N)
    
    # Your implementation with tau * R_bar
    P = Qf.clone()
    K_yours = []
    for k in range(N-1, -1, -1):
        P_new, K_k = your_riccati_single_step(A, B, tau, R_base, P)
        K_yours.insert(0, K_k)
        P = P_new
    
    print("\n" + "=" * 60)
    print("TAU SCALING TEST")
    print("=" * 60)
    print("Ground truth: R_eff = R * tau")
    print("Your code: H_uu = tau * R + B^T P B")
    
    K_match = all(torch.allclose(k1, k2, atol=1e-10) for k1, k2 in zip(K_gt, K_yours))
    P_match = torch.allclose(P0_gt, P, atol=1e-10)
    
    print(f"\nK matrices match: {K_match}")
    print(f"P[0] matches: {P_match}")
    
    if not K_match:
        print("\nK differences:")
        for k, (k1, k2) in enumerate(zip(K_gt, K_yours)):
            print(f"  GT K[{k}] = {k1.numpy()}")
            print(f"  Yours K[{k}] = {k2.numpy()}")
            print(f"  Diff: {(k1 - k2).abs().max():.6e}")
    
    return K_match and P_match


def test_factor_of_2():
    """
    Test if factor of 2 in HexnerGame matches ground truth.
    
    HexnerGame stores: R = 2 * R1_base
    Then Riccati uses: tau * R = tau * 2 * R1_base
    
    Ground truth uses: R1_base * tau
    
    These differ by factor of 2!
    """
    dtype = torch.float64
    device = torch.device("cpu")
    
    tau = 0.1
    A = torch.tensor([
        [1.0, tau],
        [0.0, 1.0]
    ], dtype=dtype, device=device)
    
    B = torch.tensor([
        [0.5 * tau**2],
        [tau]
    ], dtype=dtype, device=device)
    
    Qf = torch.eye(2, dtype=dtype, device=device)
    N = 5
    
    R_base = torch.tensor([[0.05]], dtype=dtype, device=device)
    
    print("\n" + "=" * 60)
    print("FACTOR OF 2 TEST")
    print("=" * 60)
    
    # Ground truth: uses R_base * tau directly
    R_gt = R_base * tau
    K_gt, P0_gt = discrete_lqr_ground_truth(A, B, torch.zeros(2,2), R_gt, Qf, N)
    print(f"Ground truth with R = R_base * tau = {R_gt.item():.6f}")
    print(f"  K[0] = {K_gt[0].numpy()}")
    
    # HexnerGame style: stores 2 * R_base, then Riccati uses tau * R
    R_hexner = 2.0 * R_base  # What HexnerGame stores
    P = Qf.clone()
    K_hexner = []
    for k in range(N-1, -1, -1):
        P_new, K_k = your_riccati_single_step(A, B, tau, R_hexner, P)
        K_hexner.insert(0, K_k)
        P = P_new
    print(f"\nHexnerGame style with R_stored = 2 * R_base = {R_hexner.item():.6f}")
    print(f"  (then tau * R_stored = {(tau * R_hexner).item():.6f})")
    print(f"  K[0] = {K_hexner[0].numpy()}")
    
    # What it should be: tau * 2 * R_base in ground truth
    R_gt_fixed = 2.0 * R_base * tau
    K_gt_fixed, _ = discrete_lqr_ground_truth(A, B, torch.zeros(2,2), R_gt_fixed, Qf, N)
    print(f"\nGround truth with R = 2 * R_base * tau = {R_gt_fixed.item():.6f}")
    print(f"  K[0] = {K_gt_fixed[0].numpy()}")
    
    # Compare
    match_original = torch.allclose(K_gt[0], K_hexner[0], atol=1e-10)
    match_fixed = torch.allclose(K_gt_fixed[0], K_hexner[0], atol=1e-10)
    
    print(f"\nHexnerGame matches original GT: {match_original}")
    print(f"HexnerGame matches fixed GT (2x): {match_fixed}")
    
    if match_fixed:
        print("\n✓ CONFIRMED: Ground truth should use 2 * R_base * tau to match HexnerGame!")
    
    return match_fixed


if __name__ == "__main__":
    test1 = test_single_player_riccati()
    test2 = test_with_tau_scaling()
    test3 = test_factor_of_2()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Single-player test: {'PASS' if test1 else 'FAIL'}")
    print(f"Tau scaling test: {'PASS' if test2 else 'FAIL'}")
    print(f"Factor of 2 test: {'PASS' if test3 else 'FAIL'}")
