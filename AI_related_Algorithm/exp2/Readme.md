# 🤖 A* Pathfinding Algorithm — Visualization with Pygame (Lab 2)

📘 **Project Overview**  
This project implements the **A\*** (*A-Star*) pathfinding algorithm in Python, featuring a **graphical visualization** using the `pygame` library.  
The algorithm dynamically finds the **shortest path** between a start and goal node in a 2D grid with **randomly generated obstacles**.  
This experiment focuses on understanding **heuristic search strategies**, **cost functions**, and the **influence of Euclidean heuristics** in multi-directional movement.

---

## ⚙️ Folder Structure

```plaintext
.
├── main.py                                 # Main program — A* implementation and visualization
└── README.md                               # Project documentation (this file)
```

---

## 🚀 Features

- 🧩 **Heuristic Search Algorithm**
  - Implements the A* pathfinding algorithm on a 30×30 grid.
  - Uses **Euclidean distance** as the heuristic (supports 8-directional movement).
  - Automatically reconstructs the shortest path via parent node backtracking.

- 🧱 **Dynamic Obstacle Generation**
  - Randomly generates **130 obstacles** on the grid (excluding start/end points).
  - Demonstrates different path outcomes across multiple runs.

- 🖥️ **Graphical Visualization**
  - Real-time grid rendering via `pygame`.
  - Colors:
    - 🟩 **Green** – Walkable cells  
    - 🟥 **Red** – Obstacles  
    - 🩶 **Gray** – Final shortest path  
    - ⚪ **White** – Background  

- 📈 **Performance Demonstration**
  - Multiple runs illustrate how random obstacle placement affects path variation.
  - Real-time animation allows observation of search progression and final path.

---

## 🧩 Code Overview

### 🔹 Core Classes & Functions

```python
class Node:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.g = self.h = self.f = 0
        self.parent = None
        self.walkable = True
```

```python
def heuristic(node, target):
    """Euclidean distance heuristic"""
    return math.sqrt((node.x - target.x)**2 + (node.y - target.y)**2)
```

```python
def astar(start, end):
    """Main A* algorithm loop"""
    open_list, closed_list = [start], []
    while open_list:
        current = min(open_list, key=lambda n: n.f)
        open_list.remove(current)
        closed_list.append(current)
        if current == end:
            path = []
            while current:
                path.append((current.x, current.y))
                current = current.parent
            return path[::-1]
        # Generate neighbors (8 directions)
        ...
```

---

## 🧠 Algorithm Flow

```plaintext
1. Initialize open_list and closed_list
2. Select node with minimum f = g + h
3. Move node from open_list → closed_list
4. For each neighbor:
   - Skip if obstacle or already closed
   - Compute g, h, and f
   - Update parent and lists accordingly
5. Repeat until goal found
6. Reconstruct optimal path by backtracking parent pointers
```

---

## 🧪 Experimental Setup

| Parameter | Description |
|------------|-------------|
| Grid Size | 30 × 30 |
| Obstacles | 130 random red blocks |
| Start Node | (0, 0) |
| End Node | (23, 18) |
| Heuristic | Euclidean distance |
| Movement | 8 directions (N, NE, E, SE, S, SW, W, NW) |
| Environment | Python 3.10 (Anaconda virtual environment) |
| Library | `pygame` |

---

## 🧩 Example Visualization

```
🟩  = walkable  🟥  = obstacle  🩶  = final path
```

Each run produces a unique map layout, e.g.:

```
Run 1 → Path length: 45
Run 2 → Path length: 52
Run 3 → Path length: 49
```

Paths differ depending on obstacle locations, demonstrating adaptive heuristic exploration.

---

## 🧩 Key Takeaways

1. **Heuristic search efficiency:**  
   A* effectively narrows the search region by estimating distance costs.

2. **Cost function insight:**  
   `f(n) = g(n) + h(n)` balances *actual* and *estimated* cost, ensuring near-optimal path discovery.

3. **Heuristic choice matters:**  
   Euclidean distance provides smoother results than Manhattan distance in diagonal movement cases.

4. **Obstacle sensitivity:**  
   Randomly generated barriers alter path geometry, testing the robustness of the algorithm.

---

## ⚙️ Run the Code

### ▶ Using Python
```bash
pip install pygame
python main.py
```

### 🧠 Controls
- The program launches a 600×600 grid window.  
- Gray path appears once the optimal route is computed.  
- Close the window to exit.

---

## 📄 References

- *Hart, P. E., Nilsson, N. J., & Raphael, B. (1968).*  
  **A Formal Basis for the Heuristic Determination of Minimum Cost Paths.**  
  *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100–107.  
- Pygame Documentation — [https://www.pygame.org/docs](https://www.pygame.org/docs)

---

## 👨‍💻 Author

**Ailixiaer Ailika**  
Bachelor Project — *Artificial Intelligence Techniques Lab 2*  
📍 University Project Repository (Non-Commercial Use)

---

## 🪪 License

Released under the **MIT License**.  
Free to use, modify, and distribute with attribution.
