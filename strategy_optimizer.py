"""
Strategy Optimizer — Genetic Algorithm + Bayesian Optimization for trading parameters.
Automatically finds optimal RSI thresholds, EMA periods, stop-loss levels, etc.
Runs on 1GB RAM, no heavy dependencies (numpy only).
"""
import random
import json
import numpy as np
from typing import Dict, List, Callable, Optional, Tuple


class GeneticOptimizer:
    """
    Genetic Algorithm for optimizing trading strategy parameters.
    
    Usage:
        def fitness(params):
            # Run backtest with these params, return Sharpe ratio
            return sharpe
        
        opt = GeneticOptimizer(
            param_ranges={"rsi_period": (5, 30), "rsi_overbought": (60, 85)},
            fitness_func=fitness,
        )
        best = opt.optimize(generations=20, population=30)
    """
    
    def __init__(self, param_ranges: Dict, fitness_func: Callable,
                 population_size: int = 30, mutation_rate: float = 0.2,
                 elite_ratio: float = 0.2):
        self.param_ranges = param_ranges
        self.fitness_func = fitness_func
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_count = max(1, int(population_size * elite_ratio))
        self.history = []
    
    def _random_individual(self) -> Dict:
        ind = {}
        for name, range_vals in self.param_ranges.items():
            if isinstance(range_vals, list):
                ind[name] = random.choice(range_vals)
            else:
                ind[name] = random.uniform(range_vals[0], range_vals[1])
        return ind
    
    def _crossover(self, p1: Dict, p2: Dict) -> Dict:
        child = {}
        for name in self.param_ranges:
            child[name] = p1[name] if random.random() < 0.5 else p2[name]
        return child
    
    def _mutate(self, ind: Dict) -> Dict:
        mutated = ind.copy()
        for name, range_vals in self.param_ranges.items():
            if random.random() < self.mutation_rate:
                if isinstance(range_vals, list):
                    mutated[name] = random.choice(range_vals)
                else:
                    mutated[name] = random.uniform(range_vals[0], range_vals[1])
        return mutated
    
    def optimize(self, generations: int = 20, verbose: bool = True) -> Dict:
        population = [self._random_individual() for _ in range(self.population_size)]
        best_overall = None
        best_fitness_overall = -999
        
        for gen in range(generations):
            fitnesses = [self.fitness_func(ind) for ind in population]
            best_idx = int(np.argmax(fitnesses))
            best_f = fitnesses[best_idx]
            avg_f = float(np.mean(fitnesses))
            
            if best_f > best_fitness_overall:
                best_fitness_overall = best_f
                best_overall = population[best_idx].copy()
            
            self.history.append({
                "gen": gen + 1, "best": best_f, "avg": avg_f,
                "best_params": population[best_idx],
            })
            
            if verbose:
                print(f"  Gen {gen+1}/{generations}: best={best_f:.4f} avg={avg_f:.4f}")
            
            # Sort by fitness (descending)
            sorted_indices = np.argsort(fitnesses)[::-1]
            population = [population[i] for i in sorted_indices]
            
            # Elitism: keep best individuals
            next_gen = population[:self.elite_count]
            
            # Fill rest with crossover + mutation
            while len(next_gen) < self.population_size:
                p1 = random.choice(population[:self.population_size // 2])
                p2 = random.choice(population[:self.population_size // 2])
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                next_gen.append(child)
            
            population = next_gen
        
        return {"best_params": best_overall, "best_fitness": best_fitness_overall,
                "history": self.history}


class BayesianOptimizer:
    """
    Simple Bayesian Optimization using random sampling + Gaussian Process.
    Fits a GP surrogate model and uses expected improvement to select next params.
    """
    
    def __init__(self, param_ranges: Dict, fitness_func: Callable,
                 n_initial: int = 10, n_iterations: int = 30):
        self.param_ranges = param_ranges
        self.fitness_func = fitness_func
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.X = []  # sampled points
        self.y = []  # observed values
    
    def _sample_random_point(self) -> Dict:
        return {n: random.uniform(r[0], r[1]) if not isinstance(r, list)
                else random.choice(r)
                for n, r in self.param_ranges.items()}
    
    def _dict_to_array(self, d: Dict) -> np.ndarray:
        return np.array([d[n] for n in self.param_ranges])
    
    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray, length: float = 1.0) -> float:
        """Radial basis function kernel"""
        dist = np.linalg.norm(x1 - x2)
        return np.exp(-0.5 * (dist / length) ** 2)
    
    def _expected_improvement(self, x: np.ndarray, xi: float = 0.01) -> float:
        """Expected Improvement acquisition function"""
        if len(self.X) < 2:
            return random.random()
        
        X_arr = np.array([self._dict_to_array(p) for p in self.X])
        y_arr = np.array(self.y)
        
        y_best = np.max(y_arr)
        n = len(self.X)
        
        # Simple GP mean and variance at x
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self._rbf_kernel(X_arr[i], X_arr[j])
        
        k = np.array([self._rbf_kernel(X_arr[i], x) for i in range(n)])
        
        K_inv = np.linalg.pinv(K + 1e-6 * np.eye(n))
        mu = k.T @ K_inv @ y_arr
        sigma2 = 1.0 - k.T @ K_inv @ k
        sigma = np.sqrt(max(sigma2, 1e-10))
        
        # Expected Improvement
        imp = mu - y_best - xi
        Z = imp / sigma
        ei = imp * (0.5 * (1.0 + np.math.erf(Z / np.sqrt(2.0)))) + \
             sigma * np.exp(-0.5 * Z ** 2) / np.sqrt(2 * np.pi)
        
        return max(float(ei), 0)
    
    def optimize(self, verbose: bool = True) -> Dict:
        # Initial random sampling
        for i in range(self.n_initial):
            pt = self._sample_random_point()
            val = self.fitness_func(pt)
            self.X.append(pt)
            self.y.append(val)
            if verbose:
                print(f"  Init {i+1}/{self.n_initial}: {val:.4f}")
        
        # Bayesian iterations
        for i in range(self.n_iterations):
            best_ei = -1
            best_pt = None
            
            for _ in range(50):  # random search for best EI
                pt = self._sample_random_point()
                x = self._dict_to_array(pt)
                ei = self._expected_improvement(x)
                if ei > best_ei:
                    best_ei = ei
                    best_pt = pt
            
            if best_pt:
                val = self.fitness_func(best_pt)
                self.X.append(best_pt)
                self.y.append(val)
                if verbose:
                    print(f"  Bayes {i+1}/{self.n_iterations}: {val:.4f} (EI={best_ei:.4f})")
        
        best_idx = int(np.argmax(self.y))
        return {"best_params": self.X[best_idx], "best_fitness": self.y[best_idx],
                "all_params": self.X, "all_values": self.y}


if __name__ == "__main__":
    # Example: optimize RSI strategy parameters
    def fake_fitness(params):
        rsi_period = int(params.get("rsi_period", 14))
        overbought = params.get("rsi_overbought", 70)
        return -abs(rsi_period - 14) * 0.1 - abs(overbought - 70) * 0.05 + random.gauss(0, 0.1)
    
    opt = GeneticOptimizer(
        param_ranges={"rsi_period": (5, 30), "rsi_overbought": (60, 85)},
        fitness_func=fake_fitness,
        population_size=20,
    )
    result = opt.optimize(generations=10, verbose=True)
    print(f"\nBest: {result['best_params']} (fitness={result['best_fitness']:.4f})")