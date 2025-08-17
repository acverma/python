"""
✅ Features:

Supports three methods:

Pure Python (manual Gaussian elimination)

NumPy (fast, numerical)

SymPy (exact symbolic math)

Provides a clean unified interface

Includes auto-benchmarking to compare performance

Returns a summary of results and performance

✅ Unified Tool: VectorDependencyTool
✅ Summary Comparison (Integrated in Tool):
Method	Symbolic	External Dependency	Precision	Speed (usually)	Use Case
Pure Python	❌	No	Approx (float)	Medium	Embedded / No dependencies
NumPy	❌	Yes (NumPy)	Float64	🔥 Fastest	Numerical computing, real apps
SymPy	✅	Yes (SymPy)	Exact Math	Slower	Symbolic math, algebra, CAS
✅ Requirements

NumPy (for fast numeric ops)

SymPy (optional, for symbolic ops)

Install missing ones via pip:

=== Linear Dependence Summary ===
Pure Python => Dependent    | Time: 0.000045s
NumPy      => Dependent    | Time: 0.000017s
SymPy      => Dependent    | Time: 0.000142s

⚡ Fastest Method: NumPy (0.000017s)


"""
import time
import numpy as np

try:
    from sympy import Matrix, symbols
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

class VectorDependencyTool:
    def __init__(self, vectors):
        self.vectors = vectors
        self.num_vectors = len(vectors)
        self.dimension = len(vectors[0])
        self.methods = {
            'pure_python': self._check_pure_python,
            'numpy': self._check_numpy,
            'sympy': self._check_sympy if HAS_SYMPY else None
        }
        self.results = {}

    def _check_pure_python(self):
        def gaussian_rank(mat):
            m = [row[:] for row in mat]
            rows, cols = len(m), len(m[0])
            rank = 0
            for col in range(cols):
                pivot_row = None
                for row in range(rank, rows):
                    if abs(m[row][col]) > 1e-10:
                        pivot_row = row
                        break
                if pivot_row is not None:
                    m[rank], m[pivot_row] = m[pivot_row], m[rank]
                    pivot = m[rank][col]
                    m[rank] = [x / pivot for x in m[rank]]
                    for r in range(rows):
                        if r != rank and abs(m[r][col]) > 1e-10:
                            factor = m[r][col]
                            m[r] = [m[r][i] - factor * m[rank][i] for i in range(cols)]
                    rank += 1
            return rank

        start = time.perf_counter()
        rank = gaussian_rank(self.vectors)
        end = time.perf_counter()
        is_independent = rank == self.num_vectors if self.num_vectors <= self.dimension else False
        return {
            'method': 'Pure Python',
            'independent': is_independent,
            'time': end - start
        }

    def _check_numpy(self):
        arr = np.array(self.vectors)
        start = time.perf_counter()
        rank = np.linalg.matrix_rank(arr)
        end = time.perf_counter()
        is_independent = rank == arr.shape[0] if arr.shape[0] <= arr.shape[1] else False
        return {
            'method': 'NumPy',
            'independent': is_independent,
            'time': end - start
        }

    def _check_sympy(self):
        mat = Matrix(self.vectors)
        start = time.perf_counter()
        rank = mat.rank()
        end = time.perf_counter()
        is_independent = rank == mat.rows if mat.rows <= mat.cols else False
        return {
            'method': 'SymPy',
            'independent': is_independent,
            'time': end - start
        }

    def run_all(self):
        for key, method in self.methods.items():
            if method:
                self.results[key] = method()

    def print_summary(self):
        print("=== Linear Dependence Summary ===")
        for key, res in self.results.items():
            status = "Independent" if res['independent'] else "Dependent"
            print(f"{res['method']:<10} => {status:<12} | Time: {res['time']:.6f}s")

        # Best time
        best_method = min(self.results.items(), key=lambda x: x[1]['time'])
        print(f"\n⚡ Fastest Method: {best_method[1]['method']} ({best_method[1]['time']:.6f}s)")

    def get_summary_dict(self):
        return self.results
if __name__ == "__main__":
    # Example usage
    print("Running demo with 3 vectors in 3D...")
    vectors = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]  # Dependent (rows are linearly dependent)
    ]

    tool = VectorDependencyTool(vectors)
    tool.run_all()
    tool.print_summary()
