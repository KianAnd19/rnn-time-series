import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import stats
import scikit_posthocs as sp
import pandas as pd


# https://github.com/yesteryearer/Automated-Analysis-of-Metaheuristics
def plot_critical_difference_diagram(ranks, sig_matrix):
    fig, ax = plt.subplots(figsize=(11.5, 5))
    
    sp.critical_difference_diagram(
        ranks, 
        sig_matrix, 
        ax=ax, 
        label_fmt_left='{label} [{rank:.3f}]  ',
        label_fmt_right='  [{rank:.3f}] {label}',
        text_h_margin=0.15,
        label_props={'color': 'black', 'fontweight': 'bold', 'fontsize': 10},
        crossbar_props={'color': None, 'marker': 'o', 'linewidth': 1.5},
        elbow_props={'color': 'gray', 'linestyle': '--'},
    )

    plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.3)
    plt.savefig('critical_difference.pdf')

def friedman_test(data):
    k = data.shape[1]  # number of models
    n = data.shape[0]  # number of datasets
    
    # Convert data to ranks
    ranks = np.zeros_like(data)
    for i in range(n):
        ranks[i] = stats.rankdata(data[i])
    
    # Calculate R_j (sum of ranks for each model)
    R_j = np.sum(ranks, axis=0)
    
    # Calculate Friedman statistic
    chi2 = (12 * n) / (k * (k + 1)) * (np.sum(R_j ** 2) - (k * (k + 1) ** 2) / 4)
    
    # Degrees of freedom
    df = k - 1
    
    # Calculate p-value
    p_value = stats.chi2.sf(chi2, df)
    
    return chi2, p_value, df, R_j

def compute_CD(n_datasets, n_models, alpha=0.05):
    q = stats.studentized_range.ppf(q=1-alpha, k=n_models, df=np.inf)
    CD = q * np.sqrt((n_models * (n_models + 1)) / (6 * n_datasets))
    return CD

def plot_CD(average_ranks, names, cd):
    plt.figure(figsize=(10, 6))
    
    ax = plt.gca()
    ax.set_xlim(0, max(average_ranks) + 1)
    ax.set_ylim(0, len(names) * 1.5)
    ax.invert_yaxis()
    ax.axis('off')
    
    for i, (rank, name) in enumerate(sorted(zip(average_ranks, names))):
        ax.plot([0, rank], [i * 1.5, i * 1.5], 'k')
        ax.plot([rank, rank], [i * 1.5 - 0.1, i * 1.5 + 0.1], 'k')
        ax.text(rank, i * 1.5 + 0.1, f"{name} ({rank:.2f})", ha='center', va='bottom')
    
    min_rank, max_rank = min(average_ranks), max(average_ranks)
    cd_min, cd_max = max(min_rank, max_rank - cd), min(max_rank, min_rank + cd)
    ax.plot([cd_min, cd_max], [-0.5, -0.5], 'k-', linewidth=2)
    ax.text((cd_min + cd_max) / 2, -0.75, f"CD = {cd:.2f}", ha='center', va='top')
    
    plt.title("Critical Difference Plot")
    plt.tight_layout()
    plt.savefig('custom_critical_difference_plot.png')
    plt.close()

def pairwise_comparison(R_j, n_datasets, n_models):
    se = np.sqrt((n_models * (n_models + 1)) / (6 * n_datasets))
    
    comparisons = []
    for (i, Ri), (j, Rj) in combinations(enumerate(R_j), 2):
        z = (Ri - Rj) / (np.sqrt(2) * se)
        p_unadj = 2 * (1 - stats.norm.cdf(abs(z)))
        comparisons.append((i, j, z, p_unadj))
    
    # Manual Bonferroni correction
    n_comparisons = len(comparisons)
    p_adj = [min(p * n_comparisons, 1.0) for _, _, _, p in comparisons]
    
    return [(c[0], c[1], c[2], c[3], p) for c, p in zip(comparisons, p_adj)]


# Load data
data = []
with open('results/final.csv', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(',')
        line = [float(x) for x in line]
        data.append(line)
data = np.array(data)

# Perform Friedman test
chi2, p_value, df, R_j = friedman_test(data)

print(f"Friedman test results:")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {df}")

if p_value < 0.05:
    print("There are significant differences among the models.")
else:
    print("There are no significant differences among the models.")

# Prepare data for critical difference plot
n_datasets, n_models = data.shape
model_names = ['Elman', 'Jordan', 'Multi']  # Replace with your actual model names
average_ranks = R_j / n_datasets

# Compute Critical Difference
cd = compute_CD(n_datasets, n_models)

comparisons = pairwise_comparison(R_j, n_datasets, n_models)

# Create a symmetric matrix for p-values (significance matrix)
sig_matrix = np.ones((n_models, n_models))  # Initialize with 1s (no significance)
for i, j, z, p_unadj, p_adj in comparisons:
    sig_matrix[i, j] = p_adj
    sig_matrix[j, i] = p_adj  # Symmetry

# Convert significance matrix to DataFrame
sig_matrix_df = pd.DataFrame(sig_matrix, columns=model_names, index=model_names)

# Plot critical difference diagram using scikit_posthocs
plot_critical_difference_diagram(dict(zip(model_names, average_ranks)), sig_matrix_df)

# Generate LaTeX table
latex_table = r"""\begin{table}[htbp]
\caption{Pairwise Comparison of Architectures}
\begin{center}
\begin{tabular}{c|c c c c c c}
Model A & Model B & z-value & p-value(unadj.) & p-value(adj.) & Null Hypoth. & Better Model \\
\hline
"""

for i, j, z, p_unadj, p_adj in comparisons:
    null_hypoth = "Retained" if p_adj >= 0.05 else "Rejected"
    better_model = model_names[i] if R_j[i] < R_j[j] else model_names[j]
    latex_table += f"{model_names[i]} & {model_names[j]} & {z:.5f} & {p_unadj:.5f} & {p_adj:.5f} & {null_hypoth} & {better_model} \\\\\n"

latex_table += r"""\end{tabular}
\label{tab_pairwise}
\end{center}
\end{table}"""

print(latex_table)

# Save to a file
with open('pairwise_comparison_results.tex', 'w') as f:
    f.write(latex_table)

