def matrix_rank(A):
    m = len(A)
    n = len(A[0])
    r = 0
    for j in range(n):
        pivot = None
        for i in range(r, m):
            if abs(A[i][j]) > 1e-9:
                pivot = i
                break
        if pivot is not None:
            A[r], A[pivot] = A[pivot], A[r]
            for i in range(r + 1, m):
                factor = A[i][j] / A[r][j]
                for k in range(j, n):
                    A[i][k] -= factor * A[r][k]
            r += 1
    return r

def double_cover(A):
    n = len(A)
    N = 2 * n + 1
    B = [[0] * N for _ in range(N)]
    for i in range(n):
        for j in range(n):
            B[i][j] = A[i][j]
            B[n+i][n+j] = A[i][j]
    for i in range(n):
        B[2*n][n+i] = 1
        B[n+i][2*n] = 1
    return B

A = [[0]]
for n in range(8):
    # Deep copy A to avoid mutating it during rank computation
    A_copy = [[float(x) for x in row] for row in A]
    r = matrix_rank(A_copy)
    print(f"Gamma_{n}: size={len(A)}, rank={r}")
    A = double_cover(A)
