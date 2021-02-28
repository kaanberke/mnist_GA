from network import init, train
import random

no_of_generations = 10
no_of_individuals = 10
mutate_factor = 0.05
individuals = []

layers = [0, 3, 5]


def mutate(new_individual):
    for layer in layers:
        for bias in range(len(new_individual.layers[layer].get_weights()[1])):
            n = random.random()
            if n < mutate_factor:
                new_individual.layers[layer].get_weights(
                )[1][bias] *= random.uniform(-0.5, 0.5)

    for layer in layers:
        for weight in new_individual.layers[layer].get_weights()[0]:
            n = random.random()
            if n < mutate_factor:
                for j in range(len(weight)):
                    if random.random() < mutate_factor:
                        new_individual.layers[layer].get_weights(
                        )[0][j] *= random.uniform(-0.5, 0.5)

    return new_individual


def crossover(individuals_param):
    new_individuals = [individuals_param[0], individuals_param[1]]

    for j in range(2, no_of_individuals):
        if j < (no_of_individuals - 2):
            if j == 2:
                parentA = random.choice(individuals_param[:3])
                parentB = random.choice(individuals_param[:3])
            else:
                parentA = random.choice(individuals_param[:])
                parentB = random.choice(individuals_param[:])

            for j in layers:
                temp = parentA.layers[j].get_weights()[1]
                parentA.layers[j].get_weights(
                )[1] = parentB.layers[j].get_weights()[1]
                parentB.layers[j].get_weights()[1] = temp

            new_individual = random.choice([parentA, parentB])

        else:
            new_individual = random.choice(individuals_param[:])

        new_individuals.append(mutate(new_individual))
        # new_individuals.append(new_individual)

    return new_individuals


def evolve(individuals_param, losses_param):
    sorted_y_idx_list = sorted(range(len(losses_param)), key=lambda x: losses_param[x])
    individuals_param = [individuals_param[x] for x in sorted_y_idx_list]

    # winners = individuals[:6]

    new_individuals = crossover(individuals_param)

    return new_individuals


if __name__ == "__main__":

    for i in range(no_of_individuals):
        individuals.append(init())

    for generation in range(no_of_generations):
        individuals, losses = train(individuals)
        print(losses)

        individuals = evolve(individuals, losses)

    individuals[0].save("cnn.h5")
