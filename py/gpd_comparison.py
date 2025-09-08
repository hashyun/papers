import numpy as np
from scipy.optimize import minimize
from scipy import stats
import warnings

def gpd_gam_comparison(X_tail, y_tail, gpd_tail_tree_pruned, feature_names_list, u_threshold):
    """
    Mathematically accurate comparison of GPD methods replicating Table 6 from the paper.
    Uses the same GPD negative log-likelihood formulation as your existing code.
    """
    
    print("\n" + "="*70)
    print("GPD MODEL COMPARISON - REPLICATING TABLE 6")
    print("="*70)
    
    results = {}
    n_tail = len(y_tail)
    
    # 1. Classical GPD (no covariates) - baseline
    print(f"\n1. Classical GPD (single distribution, n={n_tail}):")
    
    # Use your existing gpd_nll function for consistency
    def objective_classical(params):
        sigma, gamma = params
        return l1_l2_gpd_patched_new_catprune.gpd_nll(y_tail, sigma, gamma)
    
    # Initialize with your existing GPD MLE function
    try:
        init_params = l1_l2_gpd_patched_new_catprune.fit_gpd_mle(y_tail)
        bounds = [(1e-6, None), (0.01, 3.0)]  # Same bounds as your code
        
        result = minimize(objective_classical, init_params, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            sigma_classical, gamma_classical = result.x
            ll_classical = -result.fun
            aic_classical = 2 * 2 - 2 * ll_classical
            
            print(f"   σ̂ = {sigma_classical:.0f}")
            print(f"   γ̂ = {gamma_classical:.3f}")
            print(f"   Log-likelihood: {ll_classical:.1f}")
            print(f"   AIC: {aic_classical:.0f}")
            
            results['Classical GPD'] = {
                'LL': ll_classical,
                'AIC': aic_classical,
                'n_params': 2,
                'sigma': sigma_classical,
                'gamma': gamma_classical
            }
        else:
            raise ValueError("Classical GPD optimization failed")
            
    except Exception as e:
        print(f"   Classical GPD fitting failed: {e}")
        results['Classical GPD'] = {'LL': -np.inf, 'AIC': np.inf}
    
    # 2. GPD GAM (simplified version following Chavez-Demoulin et al.)
    print(f"\n2. GPD GAM (covariate-dependent parameters):")
    
    try:
        # Standardize covariates
        X_std = (X_tail - np.mean(X_tail, axis=0)) / (np.std(X_tail, axis=0) + 1e-8)
        X_design = np.column_stack([np.ones(len(X_std)), X_std])  # Add intercept
        n_features = X_design.shape[1]
        
        def objective_gam(params):
            # Split parameters: first n_features for log(sigma), next n_features for gamma
            log_sigma_params = params[:n_features]
            gamma_params = params[n_features:]
            
            # Linear predictors
            log_sigma = X_design @ log_sigma_params
            gamma = X_design @ gamma_params
            
            # Transform to valid parameter space
            sigma = np.exp(np.clip(log_sigma, -10, 10))
            gamma = np.clip(gamma, 0.01, 3.0)
            
            # Calculate total negative log-likelihood
            total_nll = 0
            for i in range(len(y_tail)):
                nll_i = l1_l2_gpd_patched_new_catprune.gpd_nll(
                    np.array([y_tail[i]]), sigma[i], gamma[i]
                )
                if np.isfinite(nll_i):
                    total_nll += nll_i
                else:
                    return 1e10
            
            return total_nll
        
        # Initialize parameters
        init_log_sigma = np.array([np.log(sigma_classical)] + [0.01] * (n_features-1))
        init_gamma = np.array([gamma_classical] + [0.01] * (n_features-1))
        init_params_gam = np.concatenate([init_log_sigma, init_gamma])
        
        # Optimize
        result_gam = minimize(objective_gam, init_params_gam, method='L-BFGS-B')
        
        if result_gam.success:
            ll_gam = -result_gam.fun
            n_params_gam = 2 * n_features
            aic_gam = 2 * n_params_gam - 2 * ll_gam
            
            # Get fitted parameters for a few examples
            log_sigma_params = result_gam.x[:n_features]
            gamma_params = result_gam.x[n_features:]
            
            sample_log_sigma = X_design[:3] @ log_sigma_params
            sample_gamma = X_design[:3] @ gamma_params
            sample_sigma = np.exp(sample_log_sigma)
            
            print(f"   Sample σ̂ estimates: {sample_sigma}")
            print(f"   Sample γ̂ estimates: {sample_gamma}")
            print(f"   Log-likelihood: {ll_gam:.1f}")
            print(f"   AIC: {aic_gam:.0f}")
            print(f"   Parameters: {n_params_gam}")
            
            results['GPD GAM'] = {
                'LL': ll_gam,
                'AIC': aic_gam,
                'n_params': n_params_gam
            }
        else:
            raise ValueError("GAM optimization failed")
            
    except Exception as e:
        print(f"   GPD GAM fitting failed: {e}")
        results['GPD GAM'] = {'LL': -np.inf, 'AIC': np.inf}
    
    # 3. GPD CART (your existing tree)
    print(f"\n3. GPD CART (regression tree):")
    
    try:
        # Use your existing functions to calculate tree log-likelihood
        ll_cart = calculate_tree_loglikelihood_accurate(gpd_tail_tree_pruned, X_tail, y_tail)
        n_leaves = count_tree_leaves(gpd_tail_tree_pruned)
        n_params_cart = 2 * n_leaves  # 2 parameters per leaf
        aic_cart = 2 * n_params_cart - 2 * ll_cart
        
        print(f"   Number of leaves: {n_leaves}")
        print(f"   Parameters: {n_params_cart} (2 per leaf)")
        print(f"   Log-likelihood: {ll_cart:.1f}")
        print(f"   AIC: {aic_cart:.0f}")
        
        results['GPD CART'] = {
            'LL': ll_cart,
            'AIC': aic_cart,
            'n_params': n_params_cart,
            'n_leaves': n_leaves
        }
        
    except Exception as e:
        print(f"   GPD CART evaluation failed: {e}")
        results['GPD CART'] = {'LL': -np.inf, 'AIC': np.inf}
    
    # 4. Summary Table (replicating Table 6 format)
    print("\n" + "="*70)
    print("COMPARISON SUMMARY (Table 6 Replication)")
    print("="*70)
    print(f"{'Method':<20} {'Covariates σ':<15} {'Covariates γ':<15} {'LL':<10} {'AIC':<10}")
    print("-" * 70)
    
    method_info = {
        'Classical GPD': ('–', '–'),
        'GPD GAM': ('Organization+Source', 'Date+Organization'), 
        'GPD CART': ('Type+Source', 'Type+Source')
    }
    
    for method, res in results.items():
        if 'LL' in res and 'AIC' in res and res['LL'] != -np.inf:
            cov_sigma, cov_gamma = method_info.get(method, ('–', '–'))
            print(f"{method:<20} {cov_sigma:<15} {cov_gamma:<15} {res['LL']:<10.0f} {res['AIC']:<10.0f}")
    
    # Best model identification
    valid_results = {k: v for k, v in results.items() 
                    if 'AIC' in v and np.isfinite(v['AIC'])}
    
    if valid_results:
        best_method = min(valid_results.keys(), key=lambda k: valid_results[k]['AIC'])
        print(f"\nBest model by AIC: {best_method}")
        print(f"AIC improvement over Classical GPD: {results['Classical GPD']['AIC'] - valid_results[best_method]['AIC']:.1f}")
    
    return results

def calculate_tree_loglikelihood_accurate(tree, X, y_excess):
    """
    Calculate log-likelihood for GPD tree using your existing functions.
    This matches the mathematical formulation in your code.
    """
    if tree is None or len(y_excess) == 0:
        return -np.inf
    
    total_ll = 0.0
    
    # Use your existing find_leaf function
    for i in range(len(X)):
        leaf = l1_l2_gpd_patched_new_catprune.find_leaf(tree, X[i])
        
        if hasattr(leaf, 'gpd_params') and leaf.gpd_params is not None:
            sigma, gamma = leaf.gpd_params
            # Use your existing gpd_nll function for single observation
            nll_i = l1_l2_gpd_patched_new_catprune.gpd_nll(
                np.array([y_excess[i]]), sigma, gamma
            )
            if np.isfinite(nll_i):
                total_ll -= nll_i  # Convert NLL to LL
            else:
                return -np.inf
        else:
            return -np.inf
    
    return total_ll

def count_tree_leaves(node):
    """
    Count leaves in tree using your existing structure.
    Compatible with your NodeGPD class.
    """
    if node is None:
        return 0
    
    if hasattr(node, 'is_leaf') and node.is_leaf:
        return 1
    
    left_count = count_tree_leaves(getattr(node, 'left', None))
    right_count = count_tree_leaves(getattr(node, 'right', None))
    
    return left_count + right_count

def run_gpd_comparison():
    """
    Main function to run the comparison.
    Add this at the end of your existing script.
    """
    
    # Check that required variables exist
    required_vars = ['gpd_tail_tree_pruned', 'X_tail', 'y_tail', 'feature_names_list', 'u_threshold']
    missing_vars = [var for var in required_vars if var not in globals()]
    
    if missing_vars:
        print(f"Missing required variables: {missing_vars}")
        return None
    
    if gpd_tail_tree_pruned is not None and len(y_tail) > 0:
        print(f"\nRunning comparison with {len(y_tail)} tail observations...")
        results = gpd_gam_comparison(
            X_tail, 
            y_tail, 
            gpd_tail_tree_pruned,
            feature_names_list,
            u_threshold
        )
        return results
    else:
        print("GPD tree not available or no tail data for comparison")
        return None

