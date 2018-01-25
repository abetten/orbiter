// alphabet.C
// 
// Anton Betten
// Oct 2, 2015

#include <iostream>

using namespace std;


#undef ADDED_CONDITIONS


#define TRUE  1
#define FALSE  0


int q = 7;
int nb_sol = 0;


int taken[27];

void step1();
void step2();
void step3();
void step4();
void step5();
void step6();
void step7();
void step8();
void step9();

int p, h;

int main(int argc, char **argv)
{


	for (int i = 0; i <= 26; i++) {
		taken[i] = FALSE;
		}

	taken[2] = TRUE;
	taken[3] = TRUE;
	taken[4] = TRUE;
	taken[5] = TRUE;
	taken[7] = TRUE;
	taken[9] = TRUE;
	taken[25] = TRUE;

	for (p = 1; p <= 12; p++) {
		if (taken[p]) {
			continue;
			}

		h = 2 * p;
		if (taken[h]) {
			continue;
			}

		taken[p] = TRUE;
		taken[h] = TRUE;


		step1();

		taken[p] = FALSE;
		taken[h] = FALSE;

		}

	cout << "nb_sol = " << nb_sol << endl;
}

int d, r;

void step1()
{
	cout << "step1: p=" << p << " h=" << h << endl;

	for (d = 1; d <= 26; d++) {
		if (taken[d]) {
			continue;
			}
		r = 25 + p - d;
		if (r < 1 || r > 26) {
			continue;
			}
		if (taken[r] || r == d) {
			continue;
			}

#ifdef ADDED_CONDITIONS
		// adding a new condition: d < r
		if (d > r) {
			continue;
			}
#endif

		taken[d] = TRUE;
		taken[r] = TRUE;

		step2();

		taken[d] = FALSE;
		taken[r] = FALSE;
		}
	
}

int x;

void step2()
{
	cout << "step2: p=" << p << " h=" << h << " d=" << d << " r=" << r << endl;

	for (x = 1; x <= 26; x++) {
		if (x != 26) continue;
		
		if (taken[x]) {
			continue;
			}
		taken[x] = TRUE;

		step3();

		taken[x] = FALSE;
		
		}
}

int f, t, l;

void step3()
{
	cout << "step3: p=" << p << " h=" << h << " d=" << d << " r=" << r << " x=" << x << endl;

	for (f = 1; f <= 26; f++) {
		if (taken[f]) {
			continue;
			}
		t = 2 * x - 9 - f;
		if (t < 1 || t > 26) {
			continue;
			}
		if (taken[t] || t == f) {
			continue;
			}

		if (d + f != r + t) {
			continue;
			}

		l = 49 - x;
		if (l < 1 || l > 26) {
			continue;
			}
		if (taken[l] || l == t || l == f) {
			continue;
			}

		//if (l != 23) continue;


		taken[f] = TRUE;
		taken[t] = TRUE;
		taken[l] = TRUE;

		step4();

		taken[f] = FALSE;
		taken[t] = FALSE;
		taken[l] = FALSE;

		}
}

int y, c;


void step4()
{
	cout << "step4: p=" << p << " h=" << h << " d=" << d << " r=" << r << " x=" << x << " f=" << f << " t=" << t << " l=" << l << endl;

	for (y = 1; y <= 26; y++) {
		if (taken[y]) {
			continue;
			}
		c = y - l;
		if (c < 1 || c > 26) {
			continue;
			}
		if (taken[c]) {
			continue;
			}

		if (c != 1) continue;
		
		taken[y] = TRUE;
		taken[c] = TRUE;

		step5();

		taken[y] = FALSE;
		taken[c] = FALSE;
		}
}

int o, v;

void step5()
{
	cout << "step5: p=" << p << " h=" << h << " d=" << d << " r=" << r << " x=" << x << " f=" << f << " t=" << t << " l=" << l << " y=" << y << " c=" << c << endl;

	for (o = 6; o <= 12; o++) {
		if (taken[o]) {
			continue;
			}
		v = 2 * o;
		if (v < 1 || v > 26) {
			continue;
			}
		if (taken[v]) {
			continue;
			}
		taken[v] = TRUE;
		taken[o] = TRUE;

		step6();

		taken[v] = FALSE;
		taken[o] = FALSE;
		}
}

int i, w;

void step6()
{
	cout << "step6: p=" << p << " h=" << h << " d=" << d << " r=" << r << " x=" << x << " f=" << f << " t=" << t << " l=" << l << " y=" << y << " c=" << c << " v=" << v << " o=" << o << endl;

	for (w = 1; w <= 26; w++) {
		if (taken[w]) {
			continue;
			}
		i = w - c;
		if (i < 1 || i > 26) {
			continue;
			}
		
#ifdef ADDED_CONDITIONS
		// added a condition: C is smaller than I
		if (i < c) {
			continue;
			}
#endif

		
		if (taken[i]) {
			continue;
			}
		taken[w] = TRUE;
		taken[i] = TRUE;

		step7();

		taken[w] = FALSE;
		taken[i] = FALSE;
		}
}

int j, e;

void step7()
{
	cout << "step7: p=" << p << " h=" << h << " d=" << d << " r=" << r << " x=" << x << " f=" << f << " t=" << t << " l=" << l << " y=" << y << " c=" << c << " v=" << v << " o=" << o << " w=" << w << " i=" << i << endl;

	for (e = 1; e <= 26; e++) {
		if (taken[e]) {
			continue;
			}
		j = e - q;
		if (j < 1 || j > 26) {
			continue;
			}
		if (taken[j]) {
			continue;
			}
		taken[e] = TRUE;
		taken[j] = TRUE;

		step8();

		taken[e] = FALSE;
		taken[j] = FALSE;
		}
}

int b, u, k;

void step8()
{
	cout << "step8: p=" << p << " h=" << h << " d=" << d << " r=" << r << " x=" << x << " f=" << f << " t=" << t << " l=" << l << " y=" << y << " c=" << c << " v=" << v << " o=" << o << " w=" << w << " i=" << i << " j=" << j << " e=" << e << endl;

#if 0
	for (b = 1; b <= 26; b++) {
		if (!taken[b]) {
			cout << b << endl;
			}
		}
#endif


#if 1
	for (b = 1; b <= 26; b++) {
		if (taken[b]) {
			continue;
			}
		for (u = b + 1; u <= 26; u++) {
			if (taken[u]) {
				continue;
				}
			k = b + u;
			if (k < 1 || k > 26) {
				continue;
				}
			if (taken[k] || k == b || k == u) {
				continue;
				}
			taken[b] = TRUE;
			taken[u] = TRUE;
			taken[k] = TRUE;

			step9();

			taken[b] = FALSE;
			taken[u] = FALSE;
			taken[k] = FALSE;
			}
		}
#endif

}


void step9()
{
	cout << "step9: nb_sol=" << nb_sol << " p=" << p << " h=" << h << " d=" << d << " r=" << r << " x=" << x << " f=" << f << " t=" << t << " l=" << l << " y=" << y << " c=" << c << " v=" << v << " o=" << o << " w=" << w << " i=" << i << " j=" << j << " e=" << e << " b=" << b << " u=" << u << " k=" << k << endl;
	nb_sol++;
#if 0
	for (b = 1; b <= 26; b++) {
		if (!taken[b]) {
			cout << b << endl;
			}
		}
#endif
	//exit(1);
}

