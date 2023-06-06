from M76_final_functions import *

# Generate a community matrix size S and connectance C, then print the matrix and plot the food web
M = generate_matrix(10, 0.4)
print(M)
plot_food_web(M)

# Print the stability of matrix M
stab = web_stability(M)
print(stab)

# Add an invasive species to M
I = add_invasive(M, 0.4)
print(I)

# Evaluate web stability 100 times for S = 10, C = 0.25
r = evaluate_web_stability(20, 0.25, 100)
print(r)

# Creates a plot of web stability proportions for many S and C values
plot_web_stability_multi(10, 40, 2, 0.1, 0.4, 0.02, 1)

# Regressions for the effects of predation, closeness centrality, and degree centrality
# For small food webs

log_reg_predation(10, 20, 1, 0.1, 0.2, 0.01, 100)
log_reg_predation(10, 20, 1, 0.3, 0.4, 0.01, 100)
log_reg_predation(30, 40, 1, 0.1, 0.2, 0.01, 100)
log_reg_predation(30, 40, 1, 0.3, 0.4, 0.01, 100)

log_reg_closeness(10, 20, 1, 0.1, 0.2, 0.01, 100)
log_reg_closeness(10, 20, 1, 0.3, 0.4, 0.01, 100)
log_reg_closeness(30, 40, 1, 0.1, 0.2, 0.01, 100)
log_reg_closeness(30, 40, 1, 0.3, 0.4, 0.01, 100)

log_reg_degree(10, 20, 1, 0.1, 0.2, 0.01, 100)
log_reg_degree(10, 20, 1, 0.3, 0.4, 0.01, 100)
log_reg_degree(30, 40, 1, 0.1, 0.2, 0.01, 100)
log_reg_degree(30, 40, 1, 0.3, 0.4, 0.01, 100)
