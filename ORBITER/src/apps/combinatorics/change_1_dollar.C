#include <iostream>

using namespace std;


int main(void)
{
	int i, j, k, l, c2, c3, c4, cnt;


	cnt = 0;
	for (l = 0; l <= 4; l++) {
		c4 = l * 25;
		for (k = 0; k <= (100 - c4) / 10; k++) {
			c3 = c4 + k * 10;
			for (j = 0; j <= (100 - c3) / 5; j++) {
				c2 = c3 + j * 5;
				i = 100 - c2;

				cnt++;
				cout << cnt << " : " << i << " + " << j << " * 5 + " 
					<< k << " * 10 + " << l << " * 25 = 100" << endl;
				}
			}
		}
	cout << "There are " << cnt << " ways to give change for exactly one dollar" << endl;
	return 0;
}

