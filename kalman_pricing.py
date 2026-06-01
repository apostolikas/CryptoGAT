import numpy as np
import polars as pl

class KalmanPricer:
    def __init__(self, n_features: int, q_var: float = 1e-5, r_var: float = 1e-3):
        """
        n_features: Number of predictors (e.g. 1 for intercept + N macro drivers)
        q_var: Process noise variance (how fast the true state changes)
        r_var: Measurement noise variance (how noisy the price observations are)
        """
        self.n = n_features
        self.theta = np.zeros(self.n)  # State vector [beta_0, beta_1, ...]
        self.P = np.eye(self.n) * 1.0  # State covariance matrix
        
        self.Q = np.eye(self.n) * q_var
        self.R = r_var
        
    def step(self, x: np.ndarray, y: float) -> float:
        """
        x: array of predictors at time t, shape (n_features,)
        y: observed mid-price of asset at time t
        Returns the innovation error (epsilon_t)
        """
        # 1. Predict (A priori)
        # Assuming state transition matrix is Identity (random walk)
        y_hat = np.dot(x, self.theta)
        
        # 2. Compute Innovation
        epsilon = y - y_hat
        
        # 3. Kalman Gain
        S = np.dot(np.dot(x, self.P), x.T) + self.R
        K = np.dot(self.P, x.T) / S
        
        # 4. Update State (A posteriori)
        self.theta = self.theta + K * epsilon
        self.P = self.P - np.outer(K, x).dot(self.P) + self.Q
        
        return epsilon

def apply_kalman_filter(df: pl.DataFrame | pl.LazyFrame, asset_col: str, macro_cols: list[str], window_size: int = 60) -> pl.DataFrame | pl.LazyFrame:
    """
    Step 3: Implement Macro-Relative Kalman Filter Pricing
    Expects a Polars DataFrame with the asset price column and macro driver columns.
    Iterates sequentially per symbol to strip out systemic components and leave asset-specific idiosyncratic variance.
    Returns the original DataFrame with 'kalman_epsilon' and 'kalman_z_score' added.
    """
    is_lazy = isinstance(df, pl.LazyFrame)
    dff = df.collect() if is_lazy else df
    
    n_macros = len(macro_cols)
    symbols = dff["symbol"].unique().to_list()
    out_dfs = []
    
    for sym in symbols:
        sym_df = dff.filter(pl.col("symbol") == sym)
        n_samples = len(sym_df)
        
        pricer = KalmanPricer(n_features=n_macros + 1)
        epsilons = np.zeros(n_samples)
        
        y_vals = sym_df[asset_col].to_numpy()
        x_vals = sym_df.select(macro_cols).to_numpy()
        
        for t in range(n_samples):
            x_t = np.concatenate([[1.0], x_vals[t]])
            y_t = y_vals[t]
            
            if np.isnan(y_t) or np.any(np.isnan(x_t)):
                epsilons[t] = 0.0
                continue
                
            eps_t = pricer.step(x_t, y_t)
            epsilons[t] = eps_t
            
        sym_df = sym_df.with_columns(pl.Series("kalman_epsilon", epsilons))
        out_dfs.append(sym_df)
        
    # Reassemble the pieces
    res_df = pl.concat(out_dfs).sort(["symbol", "ts_event"])
    
    # Calculate rolling std per symbol for z-score using polars EWMA for stationarity matching
    res_df = res_df.with_columns([
        pl.col("kalman_epsilon").ewm_std(half_life=window_size).over("symbol").alias("rolling_std")
    ])
    
    res_df = res_df.with_columns([
        pl.when(pl.col("rolling_std").is_null() | pl.col("rolling_std").is_nan() | (pl.col("rolling_std") == 0.0))
          .then(0.0)
          .otherwise(pl.col("kalman_epsilon") / pl.col("rolling_std"))
          .fill_null(0.0)
          .alias("kalman_z_score")
    ]).drop("rolling_std")
    
    return res_df.lazy() if is_lazy else res_df

if __name__ == "__main__":
    pass
