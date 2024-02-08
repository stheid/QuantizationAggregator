from mip import Model, xsum, minimize, BINARY

def mip_counting_problem(X, B):
    n = len(X)

    # Create MIP model
    model = Model()

    # Binary decision variables
    y = [model.add_var(var_type=BINARY) for _ in range(n)]

    # Objective function: minimize the difference from the target count B
    model.objective = minimize(xsum((y[i] - B)*(y[i] - B)
                                    for i in range(n)))

    # Counting constraint
    model += xsum(y) == B

    # Solve the MIP
    model.optimize()

    # Extract the solution
    solution = [int(y[i].x) for i in range(n)]

    return solution

if __name__ == '__main__':
    # Example usage
    X = [1.5, 2.8, 3.2, 1.0, 2.0]
    B = 3

    solution = mip_counting_problem(X, B)

    print("X =", X)
    print("Solution (y) =", solution)
