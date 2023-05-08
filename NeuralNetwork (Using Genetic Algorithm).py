import random
import numpy as np
import matplotlib.pyplot as plt
#from IPython.display import clear_output

def sigmoid(x):
    return 1/(1+np.exp(-x))

class genetic_algorithm:
    @classmethod   
    def execute(self,pop_size,generations,threshold,X,y,network):
        class Agent:
            def __init__(self,network):
                class neural_network:
                    def __init__(self,network):
                        self.weights = []
                        self.activations = []
                        for layer in network:
                            '''
                            if layer[0] != None:
                                input_size = layer[0]
                            else:
                                input_size = network[network.index(layer)-1][1]
                            '''
                            input_size = layer[0]
                            output_size = layer[1]
                            activation = layer[2]
                            self.weights.append(np.random.randn(input_size,output_size))
                            self.activations.append(activation)
                    def propagate(self,data):
                        input_data = data
                        for i in range(len(self.weights)):
                            z = np.dot(input_data,self.weights[i])
                            a = self.activations[i](z)
                            input_data = a
                        yhat = a
                        return yhat
                self.neural_network = neural_network(network)
                self.fitness = 0
            def __str__(self):
                    return 'Loss: ' + str(self.fitness[0])
                
        
                
        def generate_agents(population, network):
            return [Agent(network) for _ in range(population)]
        
        def fitness(agents,X,y):
            for agent in agents:
                yhat = agent.neural_network.propagate(X)
                cost = (yhat - y)**2
                agent.fitness = sum(cost)
            return agents
        
        def selection(agents):
            agents = sorted(agents, key=lambda agent: agent.fitness.any(), reverse=False)
            print('\n'.join(map(str, agents)))
            agents = agents[:int(0.2 * len(agents))]
            return agents
        
        def unflatten(flattened,shapes):
            newarray = []
            index = 0
            for shape in shapes:
                size = np.product(shape)
                newarray.append(flattened[index : index + size].reshape(shape))
                index += size
            return newarray
        
        def crossover(agents,network,pop_size):
            offspring = []
            for _ in range((pop_size - len(agents)) // 2):
                parent1 = random.choice(agents)
                parent2 = random.choice(agents)
                child1 = Agent(network)
                child2 = Agent(network)
                
                shapes = [a.shape for a in parent1.neural_network.weights]
                
                genes1 = np.concatenate([a.flatten() for a in parent1.neural_network.weights])
                genes2 = np.concatenate([a.flatten() for a in parent2.neural_network.weights])
                
                split = random.randint(0,len(genes1)-1)
                child1_genes = np.array(genes1[0:split].tolist() + genes2[split:].tolist())
                child2_genes = np.array(genes1[0:split].tolist() + genes2[split:].tolist())
                
                child1.neural_network.weights = unflatten(child1_genes,shapes)
                child2.neural_network.weights = unflatten(child2_genes,shapes)
                
                offspring.append(child1)
                offspring.append(child2)
            agents.extend(offspring)
            return agents
        
        def mutation(agents):
            for agent in agents:
                if random.uniform(0.0, 1.0) <= 0.1:
                    weights = agent.neural_network.weights
                    shapes = [a.shape for a in weights]
                    flattened = np.concatenate([a.flatten() for a in weights])
                    randint = random.randint(0,len(flattened)-1)
                    flattened[randint] = np.random.randn()
                    newarray = []
                    indeweights = 0
                    for shape in shapes:
                        size = np.product(shape)
                        newarray.append(flattened[indeweights : indeweights + size].reshape(shape))
                        indeweights += size
                    agent.neural_network.weights = newarray
            return agents
        
        for i in range(generations):
            print('Generation',str(i),':')
            Iteration.append(i)
            agents = generate_agents(pop_size,network)
            agents = fitness(agents,X,y)
            fitness_scores = [agent.fitness for agent in agents]
            mean_fitness = np.mean(fitness_scores)
            max_fitness = np.max(fitness_scores)
            min_fitness = np.min(fitness_scores)
            convergence_data.append([mean_fitness, max_fitness, min_fitness])
            convergence.append(mean_fitness)
            agents = selection(agents)
            agents = crossover(agents,network,pop_size)
            agents = mutation(agents)
            agents = fitness(agents,X,y)
           
            
            
            if any(agent.fitness.any() < threshold for agent in agents):
                print('Threshold met at generation '+str(i)+' !')
                break
                
            #if i % 100:
                #clear_output()
                
        return agents[0]
        #plt.plot(Iteration, convergence_data[:, 0], label='Mean Fitness')
        #plt.plot(Iteration, convergence_data[:, 1], label='Max Fitness')
        #plt.plot(Iteration, convergence_data[:, 2], label='Min Fitness')
                
        #return agents
X = np.array([1,0.6, 2.2, 4.2])
y = np.array([[0.7,0.2]])
network = [[4,2,sigmoid]]

Iteration = []
convergence_data = []
convergence = []
new_Convergence = sorted(convergence)
convergence.sort(reverse=True)
print(new_Convergence )

ga = genetic_algorithm
#execute(pop_size,generations,threshold,X,y,network)
agent = ga.execute(100,5000, 0.1 ,X ,y ,network)
weights = agent.neural_network.weights
print(weights)
agent.fitness.any()
agent.neural_network.propagate(X)

# plotting the points  
plt.plot(Iteration ,sorted(convergence , reverse=True)) 

#plt.legend()
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()

