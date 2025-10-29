import pygame
import sys
import math
import random

# 初始化Pygame
pygame.init()

# 定义颜色
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (200, 200, 200)

# 窗口尺寸和网格大小
WIDTH, HEIGHT = 600, 600
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE

# 创建窗口
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A*算法路径规划")

# 定义节点类
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.g = 0
        self.h = 0
        self.f = 0
        self.parent = None
        self.walkable = True

# 创建地图
grid = [[Node(x, y) for y in range(GRID_HEIGHT)] for x in range(GRID_WIDTH)]

# 随机设置障碍物
for i in range(130):
    x = random.randint(0, GRID_WIDTH - 1)
    y = random.randint(0, GRID_HEIGHT - 1)
    while (x == 0 and y == 0) or (x == 23 and y == 18):
        x = random.randint(0, GRID_WIDTH - 1)
        y = random.randint(0, GRID_HEIGHT - 1)
    grid[x][y].walkable = False

# 定义启发式函数
def heuristic(node, target):
    return math.sqrt((node.x - target.x) ** 2 + (node.y - target.y) ** 2)

# 实现A*算法
def astar(start, end):
    open_list = [start]
    closed_list = []

    while open_list:
        current_node = open_list[0]
        current_index = 0

        for index, node in enumerate(open_list):
            if node.f < current_node.f:
                current_node = node
                current_index = index

        open_list.pop(current_index)
        closed_list.append(current_node)

        if current_node == end:
            path = []
            current = current_node
            while current:
                path.append((current.x, current.y))
                current = current.parent
            return path[::-1]

        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                x, y = current_node.x + i, current_node.y + j
                if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                    neighbors.append(grid[x][y])

        for neighbor in neighbors:
            if not neighbor.walkable or neighbor in closed_list:
                continue

            neighbor.g = current_node.g + 1
            neighbor.h = heuristic(neighbor, end)
            neighbor.f = neighbor.g + neighbor.h
            neighbor.parent = current_node

            if neighbor not in open_list:
                open_list.append(neighbor)
            else:
                # 更新G值
                if neighbor.g < neighbor.parent.g:
                    neighbor.parent = current_node

# 主循环
start_node = grid[0][0]
end_node = grid[23][18]

path = astar(start_node, end_node)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    # 绘制地图
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            node = grid[x][y]
            color = GREEN if node.walkable else RED
            pygame.draw.rect(screen, color, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    # 绘制路径
    if path:
        for x, y in path:
            pygame.draw.rect(screen, GRAY, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    pygame.display.update()

pygame.quit()
sys.exit()
