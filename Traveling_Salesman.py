#
#   The purpose of this program is an introduction to genetic algorithms and using it to solve the traveling salesman
#   problem in a 2D coordinate plane
#   1:  Creates a list of cities with coordinate locations
#   2:  Creates a list of routes as the first generation
#   3:  Sorts the generation by the each route's fitness value
#   4:  Pulls the top x number (elite size) of the current generation and uses them as a mating pool
#   5:  Creates new generation from mating pool and elite size
#   6:  Repeat for specified number of generations
#
#   Notes:
#   1:  This project was made for a Genetic Algorithm class in a Jupyter Notebook and some formatting had to be done
#       in order to make it compatible with traditional IDEs
#
#   Future Plans:
#   1:  Further optimize the program to reduce runtime
#   2:  Reimplement matplotlib portion to generate the animation
#

#--------------------------------------------CODE STARTS HERE----------------------------------------------------------

import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

''' Class that handles each city and turns a x,y coordinate pair into a city object with a name and coordinates
    @Class:     City        |   Object      |   A coordinate pair that denotes the location of a city on a 2-Dimensional map
    @Params:    x           |   Integer     |   X coordinate
                y           |   Integer     |   Y coordinate
    @Functions: distance    |   Function    |   Calculates the distance between two City objects'''    
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return (str(self.x),str(self.y))


''' Class that handles the fitness of each route
    @Class:     Fitness         |   Object      |   A float that denotes the value of a route by it's length (shorter route -> higher fitness)
    @Params:    route           |   List        |   A list of City class objects
    @Functions: routeDistance   |   Function    |   '''
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0

    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
        for i in range(0, len(self.route)):
            fromCity = self.route[i]
            toCity = None
            if i + 1 < len(self.route):
                toCity = self.route[i + 1]
            else:
                toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


''' Creates a random individual.
    @Params:        cityList    |   List    |   A List of City class objects (unshuffled)
    @Returns:       route       |   List    |   A List of City class objects (shuffled)'''
def createRoute(cityList):
    return random.sample(cityList, len(cityList))

'''Create inital population
    @Params:        popSize     |   Integer |   Population size
                    cityList    |   List    |   A List of City class objects (unshuffled)
    @Returns:       population  |   List    |   A List of routes'''
def initialPopulation(popSize, cityList):
    population = []
    for i in range(popSize):
        population.append(createRoute(cityList))
    return population

'''This function sorts the given population in decreasing order of the fitness score.
    @Params:        population  |   List    |   A List of routes
    @Returns:       popRanked   |   List    |   A List of ordered pairs of (index, fitness) where index is the index of the individual within the population'''
def rankRoutes(population):
    fitnessResults = []
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

'''This function takes in a population sorted in decreasing order of fitness score, and chooses a mating pool from it.
    @Params:        popRanked           |   List    |   A List of ordered pairs of (index, fitness) where index is the index of the individual within the population
                    eliteSize           |   Integer |   Number of highest ranked individuals we will retain in the next generation
    @Returns:       selectionResults    |   List    |   A List of indices of the chosen mating pool in the given population'''
def selection(popRanked, eliteSize):
    selectionResults = []
    fitness_values = []
    for i in popRanked:
        fitness_values.append(i[1])
        weights = []
        iterator1 = 0
        while iterator1 < len(fitness_values):
            if iterator1 == 0:
                weights.append(fitness_values[iterator1])
            else:
                iterator2 = iterator1
                total = 0
                while iterator2 >= 0:
                    total += fitness_values[iterator2]
                    iterator2 -= 1
                weights.append(total)
            iterator1 += 1
    pk = weights[eliteSize-1]
    selectionResults = popRanked[0:eliteSize]
    return selectionResults

'''This function takes in a population and returns the chosen mating pool, or a list of routes.
    @Params:        population          |   List    |   A List of routes
                    selectionResults    |   List    |   A List of indices of the chosen mating pool in the given population
    @Returns:       matingpool          |   List    |   A List of routes to be used to create the new generation'''
def matingPool(population, selectionResults):
    matingpool = []
    indices = []
    for i in selectionResults:
        indices.append(i[0])
    for i in indices:
        matingpool.append(population[i])
    return matingpool

'''This function should breed both parents (routes) and return a route according to the ordered crossover algorithm mentioned above
    @Params         parent1     |   List    |   A route
                    parent2     |   List    |   A route
    @Returns        child       |   List    |   A route'''
def breed(parent1, parent2):
    child = None
    length1 = len(parent1)
    num1 = random.randint(0,length1-1)
    num2 = random.randint(0,length1-1)
    if num1 <= num2:
        lower = num1
        upper = num2
    else:
        lower = num2
        upper = num1
    child = parent2
    iterator = lower
    while iterator <= upper:
        child.remove(parent1[iterator])
        child.insert(iterator, parent1[iterator])
        iterator += 1    
    return child

'''This function should return the offspring population from the current population using the breed function. It should
        retain the eliteSize best routes from the current population. Then it should use the breed function to mate
        members of the population, to fill out the rest of the next generation.
        You have the freedom to decide how to choose mates for individuals.
    @Params:        matingpool  |   List    |   A List of routes to be bred
                    elitesize   |   Integer |   Number of highest ranked individuals we will retain in the next generation
                    popsize     |   Integer |   Population size
    @Returns:       children    |   List    |   A List of Routes'''
def breedPopulation(matingpool, eliteSize, popSize):
    children = []
    i = 0
    lengthpool = len(matingpool)-1
    while i < popSize-eliteSize:
        parent1 = matingpool[random.randrange(0,lengthpool,1)]
        parent2 = matingpool[random.randrange(0,lengthpool,1)]
        child = breed(list(parent1), list(parent2))
        children.append(child)
        i+=1
    return children

'''This function should take in an individual (route) and return a mutated individual. Assume mutationRate is a probability
        between 0 and 1. Use the swap mutation described above to mutate the individual according to the mutationRate. Iterate
        through each of the cities and swap it with another city according to the given probability.
    @Params:        individual      |   List    |   A Route
                    mutationRate    |   Float   |   Rate of mutation
    @Returns:       individual      |   List    |   A Route that has (or has not) been mutated'''
def mutate(individual, mutationRate):
    individual = list(individual)
    for i in individual:
        if random.random() <= mutationRate:
            place1 = individual.index(i)
            placeholder = individual[place1]
            place2 = random.randint(0,len(individual)-1)
            individual[place1] = individual[place2]
            individual[place2] = placeholder
    individual = tuple(individual)
    return individual

'''This function should use the above mutate function to mutate each member of the population. Simply iterate over the
        population and mutate each individual using the mutationRate.
    @Params:        population      |   List    |   A List of routes
                    mutationRate    |   Float   |   Rate of mutation
                    matingpool      |   List    |   A List of routes to be bred
    @Returns:       mutatedPop      |   List    |   A List of routes from the new generation that have been mutated'''
def mutatePopulation(population, mutationRate, matingpool):
    i = 0
    while i < len(population)-1:
        if i not in matingpool:
            population[i] = mutate(population[i], mutationRate)
        i += 1
    mutatedPop = population
    return mutatedPop

'''This function takes in the current generation, eliteSize and mutationRate and should return the next generation
    @Params:        currentGen      |   List    |   Routes for the current generation
                    elitesize       |   Integer |   Number of highest ranked individuals we will retain in the next generation
                    mutationRate    |   Float   |   Rate of mutation
                    popSize         |   Integer |   Population size
    @Returns:       nextGen         |   List    |   Routes for the new generation'''
def nextGeneration(currentGen, eliteSize, mutationRate, popSize):
    rankcGen = rankRoutes(currentGen)
    selectionResults = selection(rankcGen, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize, popSize)
    nextGen = matingpool + mutatePopulation(children, mutationRate, matingpool)
    return nextGen

'''This function creates an initial population, then runs the genetic algorithm according to the given parameters
    @Params:        population              |   List    |   A List of routes
                    popSize                 |   Integer |   Population size
                    eliteSize               |   Integer |   Number of highest ranked individuals we will retain in the next generation
                    mutationRate            |   Float   |   Rate of mutation
                    generations             |   Integer |   Number of generations the algorithm will run
                    max_num_gen_no_improv   |   Integer |   Number of generations the algorithm will stop if there's no improvement to the best
    @Returns:       bestRoute               |   List    |   The best route in the current generation
                    bestRoute_list          |   List    |   The list of the best routes that the new best route gets added to
                    fitness_record          |   List    |   The list of best fitness values'''
def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations, max_num_gen_no_improv):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRoute_list = []
    fitness_record = []
    #YOUR CODE HERE
    num_generations = 1
    num_gen_no_improv = 0
    new_generation = pop
    while (num_gen_no_improv < max_num_gen_no_improv) and (num_generations <= generations):
        new_generation = nextGeneration(new_generation, eliteSize, mutationRate, popSize)
        ranked_gen = rankRoutes(new_generation)                   #Rank routes provides (index, fitness)
        best_index = ranked_gen[0][0]
        best_fitness = ranked_gen[0][1]
        bestRoute_list.append(new_generation[best_index])
        fitness_record.append(best_fitness)
        if (fitness_record[num_generations-1] == fitness_record[num_generations-2]) and (num_generations > 1):
            num_gen_no_improv += 1
            num_generations += 1
    bestRoute = bestRoute_list[rankRoutes(bestRoute_list)[0][0]]
    print("Final distance: " + str(1 / fitness_record[0]))
    return bestRoute, bestRoute_list, fitness_record

#Creates a short citylist to test the functionality of the genetic algorithm
def create_test_city_list():
    cityList = []
    NewYork= City( int(40.71 * 100), int( -74.00 * 100))
    LA= City(3405,-11824)
    Chicago=City(4187,-8762)
    Houston=City(2976,-9536)
    Philly=City(3995,-7516)
    cityList.append(NewYork)
    cityList.append(LA)
    cityList.append(Chicago)
    cityList.append(Houston)
    cityList.append(Philly)
    return cityList

#Tests the genetic algorithm with the test city list and plots it on a graph
def test_genetic_algorithm():
    gaResults = geneticAlgorithm(create_test_city_list(), 300, 20, 0.01, 30, 9999)
    print(gaResults[0])
    fig, ax1 = plt.subplots(ncols=1)
    y=[4071,4187,3405,2976,3995]
    x=[-7400,-8762,-11824,-9536,-7516]
    n=['New York','Chicago','LA','Houston','Philly']
    ax1.plot(x, y, marker="o", markerfacecolor="r")
    for i, txt in enumerate(n):
        ax1.annotate(txt, (x[i], y[i]))

#Creates the real city list for the genetic algorithm
def create_city_list():
    NewYork=City(4069, -7392)
    LosAngeles=City(3411, -11841)
    Chicago=City(4184, -8769)
    Miami=City(2578, -8021)
    Dallas=City(3279, -9677)
    Philadelphia=City(4001, -7513)
    Houston=City(2979, -9539)
    Atlanta=City(3376, -8442)
    Washington=City(3890, -7702)
    Boston=City(4232, -7108)
    Phoenix=City(3357, -11209)
    Seattle=City(4762, -12232)
    SanFrancisco=City(3776, -12244)
    Detroit=City(4238, -8310)
    SanDiego=City(3283, -11712)
    Minneapolis=City(4496, -9327)
    Tampa=City(2799, -8245)
    Denver=City(3976, -10488)
    Brooklyn=City(4065, -7395)
    Queens=City(4075, -7380)
    Riverside=City(3394, -11739)
    Baltimore=City(3931, -7661)
    LasVegas=City(3623, -11527)
    Portland=City(4554, -12265)
    SanAntonio=City(2947, -9853)
    StLouis=City(3864, -9025)
    Sacramento=City(3857, -12147)
    Orlando=City(2848, -8134)
    SanJose=City(3730, -12185)
    Cleveland=City(4148, -8168)
    Pittsburgh=City(4044, -7998)
    Austin=City(3030, -9775)
    Cincinnati=City(3914, -8451)
    KansasCity=City(3912, -9455)
    Manhattan=City(4078, -7397)
    Indianapolis=City(3978, -8615)
    Columbus=City(3999, -8299)
    Charlotte=City(3521, -8083)
    VirginiaBeach=City(3673, -7604)
    Bronx=City(4085, -7387)
    Milwaukee=City(4306, -8797)
    Providence=City(4182, -7142)
    Jacksonville=City(3033, -8167)
    SaltLakeCity=City(4078, -11193)
    Nashville=City(3617, -8678)
    Richmond=City(3753, -7748)
    Memphis=City(3510, -8998)
    Raleigh=City(3583, -7864)
    NewOrleans=City(3007, -8993)
    Louisville=City(3817, -8565)
    OklahomaCity=City(3547, -9751)
    Bridgeport=City(4119, -7320)
    Buffalo=City(4290, -7885)
    FortWorth=City(3278, -9735)
    Hartford=City(4177, -7268)
    Tucson=City(3215, -11088)
    Omaha=City(4126, -9605)
    ElPaso=City(3185, -10643)
    McAllen=City(2623, -9825)
    Albuquerque=City(3511, -10665)
    Birmingham=City(3353, -8680)
    Sarasota=City(2734, -8254)
    Dayton=City(3978, -8420)
    Rochester=City(4317, -7762)
    Fresno=City(3678, -11979)
    Allentown=City(4060, -7548)
    Tulsa=City(3613, -9590)
    CapeCoral=City(2664, -8200)
    Concord=City(3797, -12200)
    ColoradoSprings=City(3887, -10476)
    Charleston=City(3282, -7996)
    Springfield=City(4212, -7254)
    GrandRapids=City(4296, -8566)
    MissionViejo=City(3361, -11766)
    Albany=City(4267, -7380)
    Knoxville=City(3597, -8395)
    Bakersfield=City(3535, -11904)
    Ogden=City(4123, -11197)
    BatonRouge=City(3044, -9113)
    Akron=City(4108, -8152)
    NewHaven=City(4131, -7292)
    Columbia=City(3404, -8090)
    Mesa=City(3340, -11172)
    PalmBay=City(2796, -8066)
    Provo=City(4025, -11165)
    Worcester=City(4227, -7181)
    Murrieta=City(3357, -11719)
    Greenville=City(3484, -8236)
    Wichita=City(3769, -9734)
    Toledo=City(4166, -8358)
    StatenIsland=City(4058, -7415)
    DesMoines=City(4157, -9361)
    LongBeach=City(3380, -11817)
    PortStLucie=City(2728, -8039)
    Denton=City(3322, -9714)
    Madison=City(4308, -8939)
    cityList= []
    cityList.append(NewYork)
    cityList.append(LosAngeles)
    cityList.append(Chicago)
    cityList.append(Miami)
    cityList.append(Dallas)
    cityList.append(Philadelphia)
    cityList.append(Houston)
    cityList.append(Atlanta)
    cityList.append(Washington)
    cityList.append(Boston)
    cityList.append(Phoenix)
    cityList.append(Seattle)
    cityList.append(SanFrancisco)
    cityList.append(Detroit)
    cityList.append(SanDiego)
    cityList.append(Minneapolis)
    cityList.append(Tampa)
    cityList.append(Denver)
    cityList.append(Brooklyn)
    cityList.append(Queens)
    cityList.append(Riverside)
    cityList.append(Baltimore)
    cityList.append(LasVegas)
    cityList.append(Portland)
    cityList.append(SanAntonio)
    cityList.append(StLouis)
    cityList.append(Sacramento)
    cityList.append(Orlando)
    cityList.append(SanJose)
    cityList.append(Cleveland)
    cityList.append(Pittsburgh)
    cityList.append(Austin)
    cityList.append(Cincinnati)
    cityList.append(KansasCity)
    cityList.append(Manhattan)
    cityList.append(Indianapolis)
    cityList.append(Columbus)
    cityList.append(Charlotte)
    cityList.append(VirginiaBeach)
    cityList.append(Bronx)
    cityList.append(Milwaukee)
    cityList.append(Providence)
    cityList.append(Jacksonville)
    cityList.append(SaltLakeCity)
    cityList.append(Nashville)
    cityList.append(Richmond)
    cityList.append(Memphis)
    cityList.append(Raleigh)
    cityList.append(NewOrleans)
    cityList.append(Louisville)
    cityList.append(OklahomaCity)
    cityList.append(Bridgeport)
    cityList.append(Buffalo)
    cityList.append(FortWorth)
    cityList.append(Hartford)
    cityList.append(Tucson)
    cityList.append(Omaha)
    cityList.append(ElPaso)
    cityList.append(McAllen)
    cityList.append(Albuquerque)
    cityList.append(Birmingham)
    cityList.append(Sarasota)
    cityList.append(Dayton)
    cityList.append(Rochester)
    cityList.append(Fresno)
    cityList.append(Allentown)
    cityList.append(Tulsa)
    cityList.append(CapeCoral)
    cityList.append(Concord)
    cityList.append(ColoradoSprings)
    cityList.append(Charleston)
    cityList.append(Springfield)
    cityList.append(GrandRapids)
    cityList.append(MissionViejo)
    cityList.append(Albany)
    cityList.append(Knoxville)
    cityList.append(Bakersfield)
    cityList.append(Ogden)
    cityList.append(BatonRouge)
    cityList.append(Akron)
    cityList.append(NewHaven)
    cityList.append(Columbia)
    cityList.append(Mesa)
    cityList.append(PalmBay)
    cityList.append(Provo)
    cityList.append(Worcester)
    cityList.append(Murrieta)
    cityList.append(Greenville)
    cityList.append(Wichita)
    cityList.append(Toledo)
    cityList.append(StatenIsland)
    cityList.append(DesMoines)
    cityList.append(LongBeach)
    cityList.append(PortStLucie)
    cityList.append(Denton)
    cityList.append(Madison)
    city_l = cityList
    print(len(city_l))
    return cityList, len(cityList)

#Runs the genetic algorithm with the full length city list and plots it on a graph
def run_genetic_algorithm():
    travelResults = geneticAlgorithm(create_city_list(), 500, 5, 0.01, 1000, 100)
    best_route = travelResults[0]
    fig,ax = plt.subplots()
    ax.set_xlim([2500,4900])
    ax.set_ylim([-13000,-6000])
    x=[]
    for i in best_route:
        x.append(i.x)
    y=[]
    for i in best_route:
        y.append(i.y) 
    ax.plot(x, y, marker="o", markerfacecolor="r")

run_genetic_algorithm()

#Does not have the animation part implemented yet
'''Input:  route_list : A list of routes. Its lenghth is the number of generations.
           fitness_record: A list of fitness scores. Its length is also the number of generations'''
'''def make_animation(self, route_list, fitness_record, Name ="Your Name"):
        
    fig,ax = plt.subplots()
    ax.set_xlim([2500,4900])
    ax.set_ylim([-13000,-6000])
    
    if len(route_list)<=50:
        divisor = 1 
    else:
        divisor= len(route_list)//50

def initial():
    ax.set_xlim([2500,4900])
    ax.set_ylim([-13000,-6000])
    route_plot, = ax.plot([], [], marker="o", markerfacecolor="r")
    return route_plot

def update(n):
    x=[]
    if n >= (len(route_list)//divisor):
        for i in route_list[-1]:
            x.append(i.x)
        y=[]
        for i in route_list[-1]:
            y.append(i.y) 
    else:
        print("Producing plot for generation: "  ({int(np.floor(n*divisor))}, end='r'))
        for i in route_list[int(np.floor(n*divisor))]:
            x.append(i.x)
        y=[]
        for i in route_list[int(np.floor(n*divisor))]:
            y.append(i.y)  
    
    ax = plt.axes(xlim=(2500,4900), ylim=(-13000,-6000))
    
    route_plot, = ax.plot(x, y, marker="o", markerfacecolor="r")
    ax.set_title("Solving traveling salesman problem using genetic algorithms " ({Name}))
    route_plot.set_label("Generation: " ({int(np.floor(n*divisor))}), "Distance : " ({1 / fitness_record[int(np.floor(n*divisor))]}))
    ax.legend()
    return route_plot,

anim = animation.FuncAnimation(fig, update, init_func=None, frames=int(len(route_list)//divisor), blit=True)    
anim.save('animation.gif', fps=4, writer="avconv", codec="libx264",dpi=600)
make_animation(travelResults[1],travelResults[2], Name ="Brent Kimura")'''



'''#The shortest route distance using a greedy algorithm :  22467.238194193524
def greedy(cityList):
    mindist = 1E10
    mindr = None
    for ind in range(len(cityList)):
        copy = list(cityList*1)
        out = []
        out.append(copy[ind])
        copy.pop(ind)
        for i in range(len(cityList)-1):
            distances = {}
            for j in range(len(copy)):
                distances[j] = out[i].distance(copy[j])
            s = sorted(distances.items(), key = operator.itemgetter(1), reverse = False)
            out.append(copy[s[0][0]])
            copy.pop(s[0][0])
        f = Fitness(out)
        if f.routeDistance() < mindist:
            mindist = f.routeDistance()
            mindr = f.route
    return mindist,mindr

#dis = greedy(create_city_list())     
#print("The shortest route distance using a greedy algorithm : ",dis)'''