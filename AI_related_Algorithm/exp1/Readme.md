# ♟️ N-Queens Problem — Backtracking Solution in C++

📘 **Project Overview**  
This project implements the **classic N-Queens problem** using **recursive backtracking** in C++.  
The program finds and prints **all possible placements** of N queens on an N×N chessboard such that no two queens threaten each other.  
Each valid configuration is printed to the console in a visual format.

---

## ⚙️ Folder Structure

```plaintext
.
├── Nqueen.cpp           # Main C++ implementation file
└── README.md            # Project documentation (this file)
```

---

## 🚀 Features

- 🧩 **Backtracking Algorithm**
  - Recursively explores all queen placements row by row.
  - Validates each placement against column and diagonal constraints.
  - Efficient pruning when conflicts are detected.

- 🏁 **Dynamic Board Size**
  - Accepts user input for arbitrary N (e.g., 4×4, 8×8, 12×12).
  - Automatically allocates board memory using `std::vector`.

- 📊 **Visualization**
  - Prints each valid board configuration to the console.
  - Uses `o` to mark queens and `*` for empty cells.

---

## 🧩 Code Overview

### 🔹 Core Functions

```cpp
bool canboard(int row, int col, const vector<vector<int>> &chess);
```
Checks if a queen can be safely placed at `(row, col)` by verifying:
- Column conflicts  
- Left-upper diagonal  
- Right-upper diagonal  

```cpp
void queen(int row, vector<vector<int>>& chess);
```
Recursive function that:
- Places queens row by row.
- Prints the board upon reaching a valid configuration.
- Backtracks after exploring each placement.

```cpp
void print(const vector<vector<int>> &chess);
```
Formats and displays each solution visually in the console.

---

## 🧠 Algorithm Flow

```plaintext
1. Start from row 0
2. For each column in the current row:
   - Check if placing a queen is valid (no conflicts)
   - If valid, place queen and recurse to next row
   - If all queens placed → print board
   - Else → backtrack (remove queen) and continue
3. Output total number of solutions
```

---

## 🧪 Example Output

**Input**
```
请输入棋盘大小: 4
```

**Output**
```
解法1:
*o**
***o
o***
**o*

解法2:
**o*
o***
***o
*o**

共2种解法
```

---

## ⚡ Compilation & Execution

### 🛠️ Using G++
```bash
g++ Nqueen.cpp -o Nqueen
./Nqueen
```

### 💻 Using Visual Studio
1. Open Visual Studio → Create New Project → C++ Console App  
2. Replace `main.cpp` content with `Nqueen.cpp`  
3. Click ▶ **Start Without Debugging (Ctrl + F5)**  

---

## 🧩 Complexity Analysis

| Aspect | Description |
|--------|--------------|
| **Time Complexity** | O(N!) — explores all permutations with pruning |
| **Space Complexity** | O(N²) — for the chessboard matrix |
| **Optimization** | Early termination through validity checks on each placement |

---

## 📄 References

- *Levitin, A. (2017).* **Introduction to the Design and Analysis of Algorithms.** Pearson.  
- *Backtracking and Constraint Solving* — Stanford CS221 Lecture Notes.  

---

## 🧩 Disclaimer

This code is intended for **educational and research purposes only.**  
Users are encouraged to modify and optimize the implementation for performance experiments.

---

## 👨‍💻 Author

**Ailixiaer Ailika**  
Bachelor Project — *Algorithm Design and Analysis (Backtracking Series)*  
📍 University Project Repository (Non-Commercial Use)

---

## 🪪 License

Released under the **MIT License**.  
You may freely use, modify, and distribute this code with attribution.
