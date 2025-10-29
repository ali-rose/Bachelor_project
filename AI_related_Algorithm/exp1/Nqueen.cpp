#include <vector>
#include <iostream>
using namespace std;

int num = 0;     //计数

/*每找到一次解法打印一次棋盘*/
void print(const vector<vector<int>> &chess) { //动态分配空间
	printf("解法%d: \n", num);
	int n = chess.size();
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (chess[i][j] == 1) {
				cout << "o";
			}
			else
				cout << "*";
		}
		cout << endl;
	}
	cout << endl;
}

/*判断是否可以落子*/
bool canboard(int row, int col,const vector<vector<int>> &chess) {
	int i, j;
	int n = chess.size();
	for (int i = 0; i < n; i++) {  //检查同一列有没有棋子
		if (chess[i][col] == 1) {
			return false;
		}
	}
	for (i = row, j = col; i >= 0 && j >= 0; i--, j--) { //检查左上方有没有棋子
		if (chess[i][j] == 1) {
			return false;
		}
	}
	for (i = row, j = col; i >= 0 && j < n; i--, j++) {  //检查右上方有没有棋子
		if (chess[i][j] == 1) {
			return false;
		}
	}
	return true;
}

/*回溯代码*/
void queen(int row,vector<vector<int>>& chess) {
	int n = chess.size();
	if (row == n) {
		num++;
		print(chess);
		return;
	}
	for (int col = 0; col < n; col++) {
		if (canboard(row, col,chess)) {
			chess[row][col] = 1;
			queen(row + 1,chess);
			chess[row][col] = 0;
		}
	}
}
int main()
{
	int n;
	cout << "请输入棋盘大小:";
	cin >> n;
	vector<vector<int>> chess(n, vector<int>(n, 0));//动态分配数组大小
	queen(0,chess);
	printf("共%d种解法\n", num);
	return 0;
}


// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
