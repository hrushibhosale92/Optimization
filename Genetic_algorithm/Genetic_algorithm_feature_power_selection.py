
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import make_scorer, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston

dataset = load_boston()

X,yy = dataset.data,dataset.target


scaler = MinMaxScaler()
Xx = pd.DataFrame(scaler.fit_transform(X),columns = dataset.feature_names)

col_names = Xx.columns
Xx.columns = range(Xx.shape[1])

X, X_test, y, y_test = train_test_split(Xx, yy, test_size=0.2, random_state=42)



"Define Parameters"
PopulationSize = 10
generations = 10
bitSize = X.shape[1]
"Feature population"
population1 = []
"Power population"
population2 = []
crossoverProbability = 0.7
mutationProbability = 0.01
power_range = (0,1)

 
"Creating population"

population1 = (np.random.randint(low = 0,high = 2,size = (PopulationSize,bitSize)))
population2 = (np.random.uniform(power_range[0],power_range[1], size =  (PopulationSize,bitSize)))

def Transform(X, population1, population2):
    SampleData = X.loc[:,(np.where(population1 == 1)[0])]
    pows = population2[population1 == 1]
    for i in range(SampleData.shape[1]):
        SampleData.iloc[:,i] = SampleData.iloc[:,i].transform(
                lambda x: x**pows[i] if x != 0 else x)
    return SampleData
        
"SVM Cross Validation Function"
def CrossValidation(X, y):
    score = make_scorer(r2_score)
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    model = LinearRegression()
    result = cross_val_score(model, X = X, y = y, scoring = score, cv = kfold)
    return np.mean(result)
    
"Removing population with all zeros i.e. no attributes selected"
def CheckforNullPopulation(population):
    i = 0
    while(i < len(population)):
        if sum(population[i]) == 0:
            population[i][np.random.choice(bitSize)] = 1
        i += 1
    return population

"Tournament Selection function"
def TournamentSelection(fitness, population1, population2):
    fitterSolutions1 = []
    fitterSolutions2 = []
    while True:
        p1 = np.random.randint(PopulationSize)
        p2 = np.random.randint(PopulationSize)
        
        if fitness[p1] >= fitness[p2]:
            fitterSolutions1.append(population1[p1])
            fitterSolutions2.append(population2[p1])
        else:
            fitterSolutions1.append(population1[p2])
            fitterSolutions2.append(population2[p2])
        
        if len(fitterSolutions1) == len(fitness):
            break
    
    return fitterSolutions1, fitterSolutions2


"Crossover function"
def Crossover(newSolution,bitSize):
    CrossOveredExamples = []
    while True:        
        
        p1 = newSolution[np.random.randint(PopulationSize)]
        p2 = newSolution[np.random.randint(PopulationSize)]
        
        if np.random.uniform() < crossoverProbability:
            splitJunction = np.random.randint(bitSize-1)
            CrossOveredExamples.append(np.append(p1[:splitJunction],p2[splitJunction:]))
            CrossOveredExamples.append(np.append(p2[:splitJunction],p1[splitJunction:]))
        else:  
            CrossOveredExamples.append(p1)
            CrossOveredExamples.append(p2)
        
        CrossOveredExamples = CheckforNullPopulation(CrossOveredExamples)
        if len(CrossOveredExamples) >= PopulationSize:
            break
    return CrossOveredExamples


"Flat Crossover"
def FlatCrossover(newSolution,PopulationSize,NumOfVar):
    CrossOveredExamples = []
    while True:
        p1 = newSolution[np.random.randint(PopulationSize)]
        p2 = newSolution[np.random.randint(PopulationSize)] 
        if np.random.uniform() < crossoverProbability:
            splitJunction = np.random.randint(bitSize)
            p1[splitJunction] = np.random.uniform(p1[splitJunction],p2[splitJunction])
            p2[splitJunction] = np.random.uniform(p1[splitJunction],p2[splitJunction])
            CrossOveredExamples.append(p1)
            CrossOveredExamples.append(p2)
        else:
            CrossOveredExamples.append(p1)
            CrossOveredExamples.append(p2)
       
        if len(CrossOveredExamples) == PopulationSize:
            break
    return CrossOveredExamples

"Mutation function"
def Mutation(CrossOveredExamples,bitSize,mutationProbability,newSolution):
    mutatePopulation = []
    for i in range(PopulationSize):
        if np.random.uniform() > mutationProbability:
            mutatePopulation.append(CrossOveredExamples[i])
        else:
            mut_point = np.random.randint(bitSize) 
            a = CrossOveredExamples[i]
            a[mut_point] = abs(a[mut_point] - 1)
            if np.sum(a) == 0:
                mutatePopulation.append(CrossOveredExamples[i])
            else:
                mutatePopulation.append(a)
    
    return mutatePopulation

def UniformMutation(CrossOveredExamples,mutationProbability,k,Range):
    mutatePopulation = []
    for j in range(len(CrossOveredExamples)):
        a = CrossOveredExamples[j]
        if np.random.uniform() > mutationProbability:
            mutatePopulation.append(a)
        else:
            mut_point = np.random.randint(bitSize) 
            a[mut_point] = np.random.uniform(Range[0],Range[1])
            mutatePopulation.append(a)
    return mutatePopulation


def delta(k , y):
    b = 0.5
    return y * (1- (np.random.uniform())**(1-(k / generations)))**b

def NonUniformMutation(CrossOveredExamples,mutationProbability,k,Range):
    mutatePopulation = []
    for j in range(len(CrossOveredExamples)):
        tau = np.random.uniform()
        randomEg = CrossOveredExamples[np.random.randint(len(CrossOveredExamples))]
        if np.random.uniform(0,mutationProbability + 0.01) <= mutationProbability:
            geneToReplace = np.random.randint(len(randomEg))
            if tau >= 0.5:
                randomEg[geneToReplace] = randomEg[geneToReplace] + delta(k , Range[1] - randomEg[geneToReplace])
            else:
                randomEg[geneToReplace] = randomEg[geneToReplace] - delta(k , randomEg[geneToReplace] - Range[0])
        
        mutatePopulation.append(randomEg)
        if len(mutatePopulation) == len(CrossOveredExamples):
            break
    
    return mutatePopulation


population1 = CheckforNullPopulation(population1)

iter = 0
"Creating generations"
for k in range(generations):
    iter += 1
    print(iter)
    "Finding fitness ; here it is cv-accuracy using SVM"
    fitness = []
    for i in range(len(population1)):
        SampleData = Transform(X, population1[i], population2[i])
        fitness.append(CrossValidation(SampleData, y))
    
    "Tournament Selection"    
    newSolution1, newSolution2 = TournamentSelection(fitness,
                                                     population1,
                                                     population2)
    
    "Crossover"
    CrossOveredExamples1 = Crossover(newSolution1, bitSize)
    
    CrossOveredExamples2 = FlatCrossover(newSolution2,PopulationSize, bitSize)
    
    "Mutation"
    mutatePopulation1 = Mutation(CrossOveredExamples1, 
                                bitSize, 
                                mutationProbability, 
                                newSolution1)
    
    mutatePopulation2 = UniformMutation(CrossOveredExamples2,
                                           mutationProbability,
                                           k,
                                           power_range)
        
    population1 = mutatePopulation1
    population2 = mutatePopulation2
    print(max(fitness))

fitness = []
for i in range(len(population1)):
    SampleData = Transform(X, population1[i], population2[i])
    fitness.append(CrossValidation(SampleData, y))
    
index_of_best = np.argmax(fitness)
BestSubsetAccuracy = fitness[index_of_best]
best_power = population2[index_of_best] 
bestSubset = population1[np.where(BestSubsetAccuracy == fitness)[0][0]]


final_data = Transform(Xx, bestSubset,best_power)
final_data.columns = col_names[bestSubset==1]
final_data['Gas holdup in riser'] = yy
final_data.to_csv('transformed_data.csv',index=False,encoding='utf-8')

dff = pd.DataFrame()
dff['Features'] = col_names[bestSubset==1]
dff['Power'] = best_power[bestSubset==1]
dff.to_csv('best_set.csv',index=False,encoding='utf-8')








