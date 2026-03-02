#Requires "tsp.dat" to be in the same file directory
#Plots of Best and Average Fitness over generationswill appear in 'chart.png'
#Map of route taken by best solution will appear in "best.png"
#Both figures will appear in the "figures" folder
#Run code with "python ga.py"

import random
import math

from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

N = 49
EARTH_RADIUS = 3963
RUNS = 50

POPULATION_SIZE = 50
MAX_NUMBER_OF_GENERATIONS = 10
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.2

df = pd.read_fwf(
    'tsp.dat',
    colspecs=[(0, 20), (20, 32), (33, 45)],
    names=["Capital", "Latitude", "Longitude"]
)

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return EARTH_RADIUS * c

dist_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        dist_matrix[i][j] = haversine(
            df.at[i, "Latitude"], df.at[i, "Longitude"],
            df.at[j, "Latitude"], df.at[j, "Longitude"]
        )

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def init_individual():
    return creator.Individual(random.sample(range(N), N))

toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def fitness_fn(ind):
    return (sum([dist_matrix[ind[i]][ind[(i+1) % N]] for i in range(N)]), )

toolbox.register("evaluate", fitness_fn)

def two_opt(ind):
    best = ind[:]
    best_fitness = fitness_fn(best)
    improved = True
    
    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                new_ind = best[:]
                new_ind[i:j] = best[i:j][::-1]
                new_fitness = fitness_fn(new_ind)
                if new_fitness < best_fitness:
                    best = new_ind
                    best_fitness = new_fitness
                    improved = True

    ind[:] = best
    return (ind, )

toolbox.register("two_opt", two_opt)

toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_once(seed):
    random.seed(seed)
    np.random.seed(seed)

    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    fits = [ind.fitness.values[0] for ind in pop]
    mean = sum(fits) / len(fits)
    best = min(fits)

    best_list = [best]
    mean_list = [mean]

    for _ in range(MAX_NUMBER_OF_GENERATIONS):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=CROSSOVER_RATE, mutpb=MUTATION_RATE)

        for ind in offspring:
            toolbox.two_opt(ind)
            del ind.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        hof.update(offspring)

        fits = [ind.fitness.values[0] for ind in offspring]
        mean = sum(fits) / len(fits)
        best = min(fits)

        best_list.append(best)
        mean_list.append(mean)

        pop[:] = toolbox.select(offspring, k=len(pop) - 1)
        pop.append(toolbox.clone(hof[0]))

    return {
        "best": hof[0],
        "best_fitness": best_list,
        "mean_fitness": mean_list
    }

best_fitness_list = []
mean_fitness_list = []

best_ind = None

for i in tqdm(range(RUNS)):
    run = run_once(i)
    best_fitness_list.append(run["best_fitness"])
    mean_fitness_list.append(run["mean_fitness"])
    if best_ind is None or run["best"].fitness.values[0] < best_ind.fitness.values[0]:
        best_ind = run["best"]

def plot_best(file_name, best):
    _, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.Robinson()})

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_extent([-130, -65, 24, 50], crs=ccrs.PlateCarree())

    ax.scatter(df["Longitude"], df["Latitude"], c='red', zorder=5, transform=ccrs.PlateCarree())

    for _, row in df.iterrows():
        ax.annotate(row["Capital"], (row["Longitude"]+0.5, row["Latitude"]+0.25), fontsize=7, transform=ccrs.PlateCarree())

    for i in range(len(best)):
        ax.plot(
            [df.at[best[i], "Longitude"], df.at[best[(i+1) % N], "Longitude"]],
            [df.at[best[i], "Latitude"], df.at[best[(i+1) % N], "Latitude"]],
            'b-', linewidth=0.8, transform=ccrs.Geodetic()
        )

    plt.savefig(f"figures/{file_name}.png", dpi=300)

def plot_graphs(file_name, best_fitness_list, mean_fitness_list):
    best_fitness_arr = np.array(best_fitness_list)
    mean_fitness_arr = np.array(mean_fitness_list)

    x = np.arange(MAX_NUMBER_OF_GENERATIONS+1)

    _, ax = plt.subplots(1, 2, figsize=(10, 5))

    mean = np.mean(best_fitness_arr, axis=0)

    std = np.std(best_fitness_arr, axis=0, ddof=1)
    conf = 1.96 * std / np.sqrt(RUNS)

    print(f"Best Fitness: {mean[-1]}\nStandard Deviation: {std[-1]}\nConfidence Interval: [{mean[-1] - conf[-1]}, {mean[-1] + conf[-1]}]")

    ax[0].plot(x, mean)
    ax[0].fill_between(x, mean - conf, mean + conf, alpha=0.3)

    ax[0].set_title("Best Fitness with 95% Confidence", fontsize=16)
    ax[0].set_xlabel("Generation", fontsize=14)
    ax[0].set_ylabel("Fitness Score", fontsize=14)
    ax[0].tick_params(axis="both", labelsize=14)

    mean = np.mean(mean_fitness_arr, axis=0)

    std = np.std(mean_fitness_arr, axis=0, ddof=1)
    conf = 1.96 * std / np.sqrt(RUNS)

    ax[1].plot(x, mean)
    ax[1].fill_between(x, mean - conf, mean + conf, alpha=0.3)
    ax[1].set_title("Mean Fitness with 95% Confidence", fontsize=16)
    ax[1].set_xlabel("Generation", fontsize=14)
    ax[1].set_ylabel("Fitness Score", fontsize=14)
    ax[1].tick_params(axis="both", labelsize=14)

    plt.tight_layout()
    plt.savefig(f"figures/{file_name}.png", dpi=300)

plot_best("best", best_ind)
plot_graphs("chart", best_fitness_list, mean_fitness_list)
