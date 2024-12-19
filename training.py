from metawars_api import *
import pandas as pd
import numpy as np
from sortedcontainers import SortedList
from tqdm import tqdm

unit_types = ["peasant", "swordman", "spearman", "archer", "defender", "horseman", "sniper", "knight", "elefant"]
equipment_types = ["wood", "steel", "diamond"]

units = []

c = 1
for ut in unit_types:
    for wt in equipment_types:
        for at in equipment_types:
            l = 1
            while True:
                u = Unit(ut, wt, at, l)
                
                if (u.cost) > 100000:
                    break

                units.append((c, u.cost, u, u.unit_type))

                l, c = l + 1, c + 1

by_cost_list = SortedList(units, key=lambda x: x[1])



class GeneticAlgorithm()


class GeneticAlgorithm():
    
    def __init__(self, 
                army_budget,
                population_size, 
                alpha_s_min, 
                alpha_s_max, 
                alpha_r_min, 
                alpha_r_max,
                alpha_m_min,
                alpha_m_max,
                alpha_i_min,
                alpha_i_max):
        super().__init__()
        self.population_size = population_size
        self.population = None
        self.alpha_s_cur = alpha_s_min # survival rate
        self.alpha_s_max = alpha_s_max # max survival rate
        self.alpha_r_cur = alpha_r_min # reproduction rate
        self.alpha_r_max = alpha_r_max #
        self.alpha_m_cur = alpha_m_max # mutation rate
        self.alpha_m_min = alpha_m_min 
        self.alpha_i_cur = alpha_i_max # innovation rate
        self.alpha_i_min = alpha_i_min
        self.param_range = alpha_s_max - alpha_s_min
        self.mu_c_cur = 1 # combination rate
        self.army_budget = army_budget
        self.init_population()


    def update_parameters(self, gradient):
        print("aquiii cojoneeee")
        self.alpha_s_cur = max(self.alpha_s_cur + gradient, self.alpha_s_max)
        self.alpha_r_cur = max(self.alpha_r_cur + gradient, self.alpha_r_max)
        self.alpha_m_cur = min(self.alpha_m_cur - gradient, self.alpha_m_min)
        self.alpha_i_cur = min(self.alpha_i_cur - gradient, self.alpha_i_min)
        



    def init_population(self):
        self.population = [self.get_random_army() for i in range(self.population_size)]

    
    def get_random_army(self):
        budget = self.army_budget
        army = []
        while budget > 0:

            limit = by_cost_list.bisect_key_right(budget) - 1 # get the index of the most expensive unit cheaper than cost
            
            if (limit < 0): 
                return army
            
            sel = random.randint(0, limit)
            
            unit = by_cost_list[sel][2]
            unit.id = by_cost_list[sel][0]

            budget -= unit.cost
            army.append(unit)
        return army
    
    
    def make_tournament(self, armies):
        results = np.zeros(len(armies))
        

        with tqdm(total=self.population_size * (self.population_size - 1) / 2, desc="Tournament", colour='green') as pbar:
            for i in range(len(armies)):
                for j in range(i+1, len(armies)):
                    res = simulation(armies[i], armies[j], False)
                    if (res[0] > res[1]):
                        results[i] += 1
                    else:
                        results[j] += 1
                    
                    pbar.update(1)
        return results


    def reproduce(self, armya: List[Unit], armyb: List[Unit]):
        cost = 5000 + int(3000 * self.mu_c_cur)
        new_army = []
        for u in random.sample(armya, len(armya)):
            if (u.cost <= cost):
                new_army.append(u)
                cost -= u.cost
        
        cost = 10000 - army_cost(new_army)
        for u in random.sample(armyb, len(armyb)):
            if (u.cost <= cost):
                new_army.append(u)
                cost -= u.cost

        return new_army     

    def mutate(self, army):
        army = army.copy()
        if random.random() <= 0.5:
            for i in range(3):
                j, k = np.random.randint(0,len(army)-1),np.random.randint(0,len(army)-1)
                army[j], army[k] = army[k], army[j]
        else:
            budget = 10000 - army_cost(army)
            for i in range(3):
                j = np.random.randint(0, len(army)-1)
                budget += army[j].cost
                limit = by_cost_list.bisect_key_right(budget) - 1
                if (limit < 0):
                    budget -= army[j].cost
                    continue
                army[j] = by_cost_list[limit][2]
        return army

                
            


    def evaluate(self, iters=120):

        gradient = self.param_range / 5

        for i in tqdm(range(iters)):
            results = self.make_tournament(self.population)
            
            sorted_armies = list(sorted(enumerate(self.population), key=lambda x: results[x[0]]))

            survivors = int(self.population_size * self.alpha_s_cur)
            parents = int(self.population_size * self.alpha_r_cur)
            mutation = int(self.population_size * self.alpha_m_cur)

            new_population = [x[1] for x in sorted_armies[:survivors]]

            for i in range(parents):
                new_population.append(self.reproduce(self.population[random.randint(0, parents-1)],
                self.population[random.randint(0, parents-1)]))

            try:
                for s in random.sample(self.population, mutation):
                    new_population.append(self.mutate(s))
            except:
                print(self.population_size)
                print(self.alpha_m_cur)
                print(mutation)
                return

            rem = self.population_size - len(new_population)

            for i in range(rem):
                new_population.append(self.get_random_army())

            if (i > 20 and i % 20 == 0):
                self.update_parameters(gradient)

            self.population = new_population


    
        

alg = GeneticAlgorithm(100000, 100, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.1, 0.2)
alg.evaluate()

    