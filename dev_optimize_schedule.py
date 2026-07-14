import numpy as np
import itertools
import pandas as pd
import numpy.typing as npt

def build_design_matrix(fixed_X: npt.NDArray, candidate_X: npt.NDArray) -> npt.NDArray:
    """
    Combines the 16 completed runs with a proposed 8-run schedule.
    Returns a (24, 5) matrix: [Intercept, Stress, Quality, Temp, Age]
    """
    full_X = np.vstack([fixed_X, candidate_X])
    return full_X

def calculate_d_score(X: npt.NDArray) -> float:
    """
    Calculates the D-Optimality score of a design matrix.
    """
    X_mean = np.mean(X[:, 1:], axis=0)
    X_std = np.std(X[:, 1:], axis=0, ddof=1)
    X_scaled = (X[:, 1:] - X_mean) / X_std
    X_scaled = np.column_stack((X[:, 0], X_scaled))
    Information_matrix = X_scaled.T @ X_scaled
    return np.linalg.det(Information_matrix)

def optimize_schedule() -> None:
    # 1. Define your 16 fixed rows based on your logbook
    fixed_X = np.array([
        [1.0, 10, 1, 21.0, 2],   # Run 1
        [1.0, 20, 0, 22.0, 2],   # Run 2
        [1.0, 30, 1, 21.0, 3],   # Run 3
        [1.0, 20, 1, 22.0, 3],   # Run 4
        [1.0, 30, 1, 22.5, 3],   # Run 5
        [1.0, 20, 0, 20.5, 4],   # Run 6
        [1.0, 10, 1, 21.5, 4],   # Run 7
        [1.0, 20, 0, 22.0, 4],   # Run 8
        [1.0, 20, 1, 24.5, 7],   # Run 9
        [1.0, 10, 0, 25.0, 7],   # Run 10
        [1.0, 10, 0, 25.0, 8],   # Run 11
        [1.0, 10, 1, 26.0, 8],   # Run 12
        [1.0, 20, 1, 25.5, 9],   # Run 13
        [1.0, 30, 0, 26.5, 9],   # Run 14
        [1.0, 30, 0, 28.0, 10],  # Run 15
        [1.0, 30, 1, 30.0, 10],  # Run 16  
    ])

    # 2. Define the 8 remaining tests [Stress, Quality_Binary]
    remaining_tests = [
        [10, 0],  # S.10.3
        [30, 0],  # S.30.3
        [30, 0],  # S.30.4  
        [20, 1],  # H.20.4
        [20, 0],  # S.20.4  
        [30, 1],  # H.30.4
        [10, 1],  # H.10.4
        [10, 0],  # S.10.4
    ]

    # 3. Define the conditions of the 8 upcoming slots [Forecasted_Temp, Age_Days]
    upcoming_slots = [
        [27.0, 11],  # Run 17 (Friday Heat Wave)
        [27.0, 11],  # Run 18 (Friday Heat Wave)
        [21.0, 14],  # Run 19 (Monday Normal)
        [21.0, 14],  # Run 20 (Monday Normal)
        [20.0, 15],  # Run 21 (Tuesday Normal)
        [20.0, 15],  # Run 22 (Tuesday Normal)
        [19.0, 16],  # Run 23 (Wednesday Normal)
        [19.0, 16],  # Run 24 (Wednesday Normal)
    ]

    slot_labels = [
        "Run 17 (Friday Heat Wave)",
        "Run 18 (Friday Heat Wave)",
        "Run 19 (Monday Normal)",
        "Run 20 (Monday Normal)",
        "Run 21 (Tuesday Normal)",
        "Run 22 (Tuesday Normal)",
        "Run 23 (Wednesday Normal)",
        "Run 24 (Wednesday Normal)",
    ]

    best_score = -np.inf
    best_schedule = None
    best_permutation = None

    # 4. Generate all 40,320 possible ways to assign the 8 tests to the 8 slots
    for test_permutation in itertools.permutations(range(len(remaining_tests))):
        tests = [remaining_tests[i] for i in test_permutation]

        candidate_X = np.array([
            [1.0, test[0], test[1], slot[0], slot[1]]
            for test, slot in zip(tests, upcoming_slots)
        ])

        X_full = build_design_matrix(fixed_X, candidate_X)
        score = calculate_d_score(X_full)

        if score > best_score:
            best_score = score
            best_schedule = candidate_X
            best_permutation = test_permutation

    # 5. Output the optimal schedule 
    print("=" * 55)
    print("         OPTIMAL SCHEDULE FOUND")
    print("=" * 55)
    print(f"{'Run':<28} {'Stress':>7} {'Quality':>8} {'Temp':>6} {'Age':>5}")
    print("-" * 55)
    for i, (row, label) in enumerate(zip(best_schedule, slot_labels)):
        quality_label = "High" if row[2] == 1 else "Std"
        print(f"{label:<28} {int(row[1]):>7} {quality_label:>8} {row[3]:>6.1f} {int(row[4]):>5}")
    print("=" * 55)
    print(f"D-Optimality Score: {best_score:.4f}")

if __name__ == "__main__":
    optimize_schedule()