import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def generate_matrix(S, C):

    # Generates a matrix of size S by S with connectivity C.

    # Parameters:
    # S (int): size of the square matrix.
    # C (float): connectivity of the matrix.

    # Returns:
    # A (numpy.ndarray): the generated matrix.

    A = np.zeros((S, S))
    p = 1 - C

    for i in range(S):
        for j in range(S):
            if i == j:
                A[i, j] = np.random.normal()
                A[i, j] = -np.abs(A[i, j])
            elif A[j,i] > 0:
                A[i, j] = np.random.normal()
                A[i, j] = -np.abs(A[i, j])
            elif A[j, i] < 0:
                A[i, j] = np.random.normal()
                A[i, j] = np.abs(A[i, j])
            elif j < i:
                A[i, j] = 0
            else:
                if np.random.random() > p:
                    A[i, j] = np.random.normal()

    return A


def plot_food_web(A):

    # Create a directed graph from the adjacency matrix
    G = nx.DiGraph(A)

    # Set the positions of the nodes in the graph
    pos = nx.circular_layout(G)

    # Draw the nodes and edges of the graph
    nx.draw_networkx_nodes(G, pos, node_size=400, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=1, edge_color='gray')

    # Show the graph
    plt.axis('off')
    plt.show()


def web_stability(A):

    # Checks if all eigenvalues of the matrix A have negative real parts.

    # Parameters:
    # A (numpy.ndarray): the matrix to check.

    # Returns:
    # bool: True if all eigenvalues of A have negative real parts, False otherwise.

    eigenvalues = np.linalg.eigvals(A)
    real_parts = eigenvalues.real

    return all(real_parts < 0)


def add_invasive(A, C):

    # Adds a row and column to the adjacency matrix A, with entries chosen
    # from a normal distribution with mean 0 and variance 1, with probability
    # C, and filled with zeros otherwise.

    # Parameters:
    # A (numpy.ndarray): the original adjacency matrix.
    # C (float): the connectivity parameter.

    # Returns:
    # numpy.ndarray: the modified adjacency matrix A_new.

    S = A.shape[0]  # get the dimension of the matrix A

    # create a new row and column with entries chosen from a normal distribution
    # with mean 0 and variance 1, with probability C
    new_row = np.random.normal(0, 1, S + 1)
    new_row[np.random.random(S + 1) > C] = 0

    new_col = np.zeros(S + 1)

    for i in range(S + 1):
        if new_row[i] == 0:
            new_col[i] = 0
        else:
            if new_row[i] < 0:
                new_col[i] = np.abs(np.random.normal())
            else:
                new_col[i] = -np.abs(np.random.normal())

    # create the new adjacency matrix A_new by appending the new row and column
    A_new = np.zeros((S + 1, S + 1))
    A_new[:S, :S] = A
    A_new[:S, -1] = new_col[:S]
    A_new[-1, :S] = new_row[:S]
    A_new[-1, -1] = -np.abs(np.random.normal())

    return A_new


def evaluate_web_stability(S, C, n):
    """
    Evaluates the stability of food webs by generating matrices, checking stability,
    adding invasive species, and rechecking stability.

    Parameters:
    S (int): dimension of the matrix.
    C (float): conductance of the matrix.
    n (int): number of simulations to run.

    Returns:
    float: proportion of stable matrices that became unstable after adding an invasive species.
    """

    stable_count = 0  # Counter for the number of stable matrices

    for _ in range(n):
        stable = False

        # Generate an initial matrix until it is stable
        while not stable:
            init_matrix = generate_matrix(S, C)
            stable = web_stability(init_matrix)

        # Add an invasive species to the stable matrix
        invasive_matrix = add_invasive(init_matrix, C)

        # Check stability after adding the invasive species
        stable_after_invasive = web_stability(invasive_matrix)

        if stable_after_invasive:
            stable_count += 1

    return stable_count / n


def plot_web_stability_multi(S_min, S_max, S_step, C_min, C_max, C_step, n):
    S_values = np.arange(S_min, S_max, S_step)
    C_values = np.arange(C_min, C_max, C_step)
    data_matrix = np.zeros((len(S_values), len(C_values)))
    for i, S in enumerate(S_values):
        for j, C in enumerate(C_values):
            data_matrix[i, j] = evaluate_web_stability(S, C, n)
            print(j)
        print(i)

    plt.figure(figsize=(10, 8))  # Set the figure size as desired (10 inches wide and 8 inches tall)
    plt.imshow(data_matrix, origin='lower', extent=[C_values.min(), C_values.max(), S_values.min(), S_values.max()],
               cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Stability')
    plt.xlabel('C')
    plt.ylabel('S')
    plt.title('Web Stability')
    plt.tight_layout()
    plt.show()


def calculate_predation_index(adjacency_matrix):
    num_species = adjacency_matrix.shape[0]
    predation_index = np.zeros(num_species, dtype=float)

    for species in range(num_species):
        pred_weight = np.sum(adjacency_matrix[species])
        predation_index[species] = pred_weight

    max_value_pred = np.max(predation_index)
    min_value_pred = np.min(predation_index)

    scaling_factor_pred = max_value_pred - min_value_pred

    predation_index = (predation_index - min_value_pred) / scaling_factor_pred

    predation_index = predation_index * 2 - 1

    return predation_index


def evaluate_predation_effect(S, C, n):
    """
    Evaluates the stability of food webs by generating matrices, checking stability,
    adding invasive species, and rechecking stability.

    Parameters:
    S (int): dimension of the matrix.
    C (float): conductance of the matrix.
    n (int): number of simulations to run.

    Returns:
    float: proportion of stable matrices that became unstable after adding an invasive species.
    """

    stable_count = np.zeros(n, dtype=int)
    invasive_predation_indexes = np.zeros(n, dtype=float)

    for i in range(n):
        stable = False

        # Generate an initial matrix until it is stable
        while not stable:
            initial_matrix = generate_matrix(S, C)
            stable = web_stability(initial_matrix)

        # Add an invasive species to the stable matrix
        invasive_matrix = add_invasive(initial_matrix, C)

        # Check stability after adding the invasive species
        stable_after_invasive = web_stability(invasive_matrix)
        predation_index = calculate_predation_index(invasive_matrix)
        invasive_predation_index = predation_index[-1]  # Get the predation index of the invasive species
        invasive_predation_indexes[i] = (invasive_predation_index)

        if stable_after_invasive:
            stable_count[i] = 1
        else:
            stable_count[i] = 0
    return stable_count, invasive_predation_indexes


def iterate_predation(S_min, S_max, S_step, C_min, C_max, C_step, n):
    S_values = np.arange(S_min, S_max, S_step)
    C_values = np.arange(C_min, C_max, C_step)

    stab_vec = np.zeros(n * len(S_values) * len(C_values), dtype=int)
    pred_vec = np.zeros(n * len(S_values) * len(C_values), dtype=int)

    count = 0  # Index counter for stab_vec and pred_vec

    for S in S_values:
        for C in C_values:
            A, B = evaluate_predation_effect(S, C, n)
            stab_vec[count:count + n] = A
            pred_vec[count:count + n] = B
            count += n

    return stab_vec, pred_vec


def log_reg_predation(S_min, S_max, S_step, C_min, C_max, C_step, n):
    A, B = iterate_predation(S_min, S_max, S_step, C_min, C_max, C_step, n)

    # Reshape B to a 2D array for logistic regression input
    B = B.reshape(-1, 1)

    # Create a logistic regression model
    model = LogisticRegression()

    # Fit the model using B as the predictor and A as the target
    model.fit(B, A)

    # Predict the stability based on the predation index
    predictions = model.predict(B)

    # Calculate the accuracy of the logistic regression model
    accuracy = (predictions == A).mean()
    print(f"Accuracy: {accuracy}")

    coef = model.coef_[0]
    # Determine the relationship
    if coef > 0:
        relationship = "Higher predation predicts instability"
    elif coef < 0:
        relationship = "Higher predation predicts stability"
    else:
        relationship = "Predation has no effect on Stability"

    print("Coefficient:", coef)
    print("Relationship:", relationship)


def calculate_closeness_centrality(adjacency_matrix):
    # Create a graph from the adjacency matrix
    graph = nx.Graph(adjacency_matrix)

    # Remove edge weights
    for edge in graph.edges():
        del graph[edge[0]][edge[1]]['weight']

    # Calculate betweenness centrality
    closeness_centrality = nx.closeness_centrality(graph)

    # Convert the dictionary of closeness centrality values to a vector
    closeness_values = []
    for node in graph.nodes():
        if node in closeness_centrality:
            closeness_values.append(closeness_centrality[node])
        else:
            closeness_values.append(0)

    return closeness_values


def evaluate_closeness_effect(S, C, n):

    stable_count = np.zeros(n, dtype=int)
    invasive_closeness_indexes = np.zeros(n, dtype=float)

    for i in range(n):
        stable = False

        # Generate an initial matrix until it is stable
        while not stable:
            initial_matrix = generate_matrix(S, C)
            stable = web_stability(initial_matrix)

        # Add an invasive species to the stable matrix
        invasive_matrix = add_invasive(initial_matrix, C)

        # Check stability after adding the invasive species
        stable_after_invasive = web_stability(invasive_matrix)
        closeness_index = calculate_closeness_centrality(invasive_matrix)
        invasive_closeness_index = closeness_index[-1]  # Get the closeness index of the invasive species
        invasive_closeness_indexes[i] = (invasive_closeness_index)

        if stable_after_invasive:
            stable_count[i] = 1
        else:
            stable_count[i] = 0

    return stable_count, invasive_closeness_indexes


def iterate_closeness(S_min, S_max, S_step, C_min, C_max, C_step, n):
    S_values = np.arange(S_min, S_max, S_step)
    C_values = np.arange(C_min, C_max, C_step)

    stab_vec = np.zeros(n * len(S_values) * len(C_values), dtype=int)
    cent_vec = np.zeros(n * len(S_values) * len(C_values), dtype=int)

    count = 0  # Index counter for stab_vec and cent_vec

    for S in S_values:
        for C in C_values:
            A, B = evaluate_closeness_effect(S, C, n)
            stab_vec[count:count + n] = A
            cent_vec[count:count + n] = B
            count += n

    return stab_vec, cent_vec

def log_reg_closeness(S_min, S_max, S_step, C_min, C_max, C_step, n):
    A, B = iterate_closeness(S_min, S_max, S_step, C_min, C_max, C_step, n)

    # Reshape B to a 2D array for logistic regression input
    B = B.reshape(-1, 1)

    # Create a logistic regression model
    model = LogisticRegression()

    # Fit the model using B as the predictor and A as the target
    model.fit(B, A)

    # Predict the stability based on the predation index
    predictions = model.predict(B)

    # Calculate the accuracy of the logistic regression model
    accuracy = (predictions == A).mean()
    print(f"Accuracy: {accuracy}")

    coef = model.coef_[0]
    # Determine the relationship
    if coef > 0:
        relationship = "Higher closeness predicts instability"
    elif coef < 0:
        relationship = "Higher closeness predicts stability"
    else:
        relationship = "Closeness has no effect on Stability"

    print("Coefficient:", coef)
    print("Relationship:", relationship)






def calculate_degree_centrality(adjacency_matrix):
    # Create a graph from the adjacency matrix
    graph = nx.Graph(adjacency_matrix)

    # Remove edge weights
    for edge in graph.edges():
        del graph[edge[0]][edge[1]]['weight']

    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(graph)

    # Convert the dictionary of degree centrality values to a vector
    degree_values = []
    for node in graph.nodes():
        if node in degree_centrality:
            degree_values.append(degree_centrality[node])
        else:
            degree_values.append(0)

    return degree_values


def evaluate_degree_effect(S, C, n):

    stable_count = np.zeros(n, dtype=int)
    invasive_degree_indexes = np.zeros(n, dtype=float)

    for i in range(n):
        stable = False

        # Generate an initial matrix until it is stable
        while not stable:
            initial_matrix = generate_matrix(S, C)
            stable = web_stability(initial_matrix)

        # Add an invasive species to the stable matrix
        invasive_matrix = add_invasive(initial_matrix, C)

        # Check stability after adding the invasive species
        stable_after_invasive = web_stability(invasive_matrix)
        degree_index = calculate_degree_centrality(invasive_matrix)
        invasive_degree_index = degree_index[-1]  # Get the closeness index of the invasive species
        invasive_degree_indexes[i] = (invasive_degree_index)

        if stable_after_invasive:
            stable_count[i] = 1
        else:
            stable_count[i] = 0

    return stable_count, invasive_degree_indexes


def iterate_degree(S_min, S_max, S_step, C_min, C_max, C_step, n):
    S_values = np.arange(S_min, S_max, S_step)
    C_values = np.arange(C_min, C_max, C_step)

    stab_vec = np.zeros(n * len(S_values) * len(C_values), dtype=int)
    deg_vec = np.zeros(n * len(S_values) * len(C_values), dtype=int)

    count = 0  # Index counter for stab_vec and deg_vec

    for S in S_values:
        for C in C_values:
            A, B = evaluate_degree_effect(S, C, n)
            stab_vec[count:count + n] = A
            deg_vec[count:count + n] = B
            count += n
    return stab_vec, deg_vec

def log_reg_degree(S_min, S_max, S_step, C_min, C_max, C_step, n):
    A, B = iterate_degree(S_min, S_max, S_step, C_min, C_max, C_step, n)

    # Reshape B to a 2D array for logistic regression input
    B = B.reshape(-1, 1)

    # Create a logistic regression model
    model = LogisticRegression()

    # Fit the model using B as the predictor and A as the target
    model.fit(B, A)

    # Predict the stability based on the predation index
    predictions = model.predict(B)

    # Calculate the accuracy of the logistic regression model
    accuracy = (predictions == A).mean()
    print(f"Accuracy: {accuracy}")

    coef = model.coef_[0]
    # Determine the relationship
    if coef > 0:
        relationship = "Higher degree predicts instability"
    elif coef < 0:
        relationship = "Higher degree predicts stability"
    else:
        relationship = "Degree has no effect on Stability"

    print("Coefficient:", coef)
    print("Relationship:", relationship)
