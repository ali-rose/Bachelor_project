import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 从Excel文件中读取数据
def read_coordinates1(file_path, sheet_name='Sheet1'):
    df = pd.read_excel(file_path, sheet_name=sheet_name,header=None,index_col=None)

    x_coordinates = df.iloc[:,1].values
    y_coordinates = df.iloc[:,2].values

    # 将坐标合并为一个二维数组
    coordinates = np.column_stack((x_coordinates, y_coordinates))

    return coordinates

def read_coordinates2(file_path, sheet_name='Sheet1'):
    df = pd.read_excel(file_path, sheet_name=sheet_name,header=None,index_col=None)

    return df
file_path = "coordinate.xlsx"
cities = read_coordinates1(file_path)
city=read_coordinates2(file_path)
# 定义遗传算法参数
city_names=city.iloc[:,0].values
population_size = 80
num_generations = 100
crossover_rate = 0.6
mutation_rate = 0.02
num_cities=34


# 计算两点间的距离
def distance(city1, city2):
    return np.linalg.norm(city1 - city2)

# 计算路径长度
def calculate_path_length(path):
    total_length = 0
    for i in range(len(path) - 1):
        total_length += distance(cities[path[i]], cities[path[i + 1]])
    total_length += distance(cities[path[-1]], cities[path[0]])  # 回到起点
    return total_length

# 初始化种群
def initialize_population(population_size, num_cities):
    return [tsp_encode(np.random.permutation(np.arange(1, num_cities + 1))) for _ in range(population_size)]

# 计算适应度
def calculate_fitness(path):
    return 1 / calculate_path_length(path)
'''
# 选择操作 - 锦标赛选择
def tournament_selection(population, fitness_values):
    tournament_size = 5
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitness = [fitness_values[i] for i in selected_indices]
    return selected_indices[np.argmax(tournament_fitness)]
'''
# 选择操作-轮盘赌
def roulette_wheel_selection(population, fitness_values):
    total_fitness = np.sum(fitness_values)
    probabilities = fitness_values / total_fitness
    selected_index = np.random.choice(len(population), p=probabilities)
    return selected_index

# 交叉操作 - 两点交叉
def crossover(parent1, parent2):
    start = np.random.randint(len(parent1))
    end = np.random.randint(start, len(parent1))
    child = np.zeros_like(parent1)
    child[start:end] = parent1[start:end]
    remaining_indices = np.array([i for i in parent2 if i not in child[start:end]])
    child[:start] = remaining_indices[:start]
    child[end:] = remaining_indices[start:]
    return child

# 变异操作 - 交换变异
def mutate(path):
    indices = np.random.choice(len(path), 2, replace=False)
    path[indices[0]], path[indices[1]] = path[indices[1]], path[indices[0]]
    return path

# TSP 编码
def tsp_encode(path):
    return path - 1  # 将城市编号从1到34映射到0到33

# TSP 解码
def tsp_decode(encoded_path):
    return [city_names[i] for i in encoded_path]  # 将城市编号从0到33映射回1到34

# 主遗传算法循环
def genetic_algorithm(population_size, num_generations, crossover_rate, mutation_rate):
    population = initialize_population(population_size, num_cities)
    for generation in range(num_generations):
        # 计算适应度
        fitness_values = [calculate_fitness(path) for path in population]

        # 选择新的种群
        new_population = []
        for _ in range(population_size):
            parent1_index = roulette_wheel_selection(population, fitness_values)
            parent2_index = roulette_wheel_selection(population, fitness_values)
            parent1 = population[parent1_index]
            parent2 = population[parent2_index]

            # 进行交叉操作
            if np.random.rand() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = parent1

            # 进行变异操作
            if np.random.rand() < mutation_rate:
                child = mutate(child)

            new_population.append(child)

        population = new_population

    # 找到最优解
    best_path_index = np.argmax(fitness_values)
    best_path = population[best_path_index]
    best_length = calculate_path_length(best_path)

    return best_path, best_length

# 运行遗传算法
best_path, best_length = genetic_algorithm(population_size, num_generations, crossover_rate, mutation_rate)

decoded_best_path = tsp_decode(best_path)
x_coordinates = cities[:,0]
y_coordinates = cities[:,1]
plt.scatter(x_coordinates, y_coordinates, label='Cities', color='blue')

# 连接最佳路径
for i in range(len(best_path) - 1):
    city1 = best_path[i]
    city2 = best_path[i + 1]
    plt.plot([x_coordinates[city1], x_coordinates[city2]], [y_coordinates[city1], y_coordinates[city2]], color='red')

# 回到起点
city1 = best_path[-1]
city2 = best_path[0]
plt.plot([x_coordinates[city1], x_coordinates[city2]], [y_coordinates[city1], y_coordinates[city2]], color='red')

# 设置标题和标签
plt.title('旅行商问题')
plt.xlabel('横坐标')
plt.ylabel('纵坐标')
plt.legend()

# 显示图形
plt.show()




# 输出最优路径和长度
print(f"最优路径：{decoded_best_path}")
print(f"路径长度：{best_length}")
