
import numpy as np
import matplotlib.pyplot as plt
import random
import math

def distance(city1, city2):
    return np.linalg.norm(np.array(city1) - np.array(city2))

def total_distance(route, cities):
    return sum(distance(cities[route[i]], cities[route[(i+1) % len(route)]]) for i in range(len(route)))

def random_neighbor(route):
    new_route = route[:]
    i, j = random.sample(range(len(route)), 2)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

def simulated_annealing(cities, initial_temp=10000, cooling_rate=0.9995, stopping_temp=0.0001):
    n = len(cities)
    current_route = list(range(n))
    random.shuffle(current_route)

    current_distance = total_distance(current_route, cities)
    best_route = current_route[:]
    best_distance = current_distance

    temp = initial_temp
    distances = []

    while temp > stopping_temp:
        new_route = random_neighbor(current_route)
        new_distance = total_distance(new_route, cities)

        if new_distance < current_distance or random.uniform(0, 1) < math.exp((current_distance - new_distance) / temp):
            current_route, current_distance = new_route, new_distance
            if new_distance < best_distance:
                best_route, best_distance = new_route[:], new_distance

        distances.append(best_distance)
        temp *= cooling_rate

    return best_route, best_distance, distances

if __name__ == "__main__":
    np.random.seed(42)
    num_cities = 100
    cities = np.random.rand(num_cities, 2) * 1000

    best_route, best_distance, distances = simulated_annealing(cities)

    plt.figure(figsize=(12, 6))
    plt.plot(distances, color="royalblue")
    plt.title("Best Distance Over Time (Simulated Annealing Convergence)")
    plt.xlabel("Iteration")
    plt.ylabel("Best Distance (km)")
    plt.grid(True)
    plt.savefig("../images/convergence_plot.png")

    initial_route = list(range(num_cities))
    initial_distance = total_distance(initial_route, cities)
    ordered_cities_initial = np.array([cities[i] for i in initial_route] + [cities[initial_route[0]]])
    plt.figure(figsize=(10, 10))
    plt.plot(ordered_cities_initial[:, 0], ordered_cities_initial[:, 1], marker="o", linestyle="-", color="red")
    plt.title(f"Initial Random Route - Distance: {initial_distance:.2f} km")
    plt.grid(True)
    plt.savefig("../images/initial_route.png")

    ordered_cities_best = np.array([cities[i] for i in best_route] + [cities[best_route[0]]])
    plt.figure(figsize=(10, 10))
    plt.plot(ordered_cities_best[:, 0], ordered_cities_best[:, 1], marker="o", linestyle="-", color="green")
    plt.title(f"Optimized Route - Distance: {best_distance:.2f} km")
    plt.grid(True)
    plt.savefig("../images/optimized_route.png")
