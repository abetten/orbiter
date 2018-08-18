// make_design.C
// 
// Anton Betten
// May 17, 2011
//
//
// 
// Creates the incidence matrix of
// a symmetric 2-(v,k,lambda) design from tables computed from 
// a partial ovoid in O^-(6,q) (or partial spread of H(3,q^2)).
// Here, v = 1/2(q^3 + q + 2), k = q^1+1, lambda = 2q

#include "orbiter.h"
#include "ovoid.h"

// global data:

INT t0; // the system time when the program started

void do_it();


// the first partial ovoid with group of order 11520:

INT data1a[] = {
   25,	30,   31,	59,   68,	88,   93,  121,  123,  143, 
   26,	44,   45,	52,   55,	82,  102,  131,  140,  149, 
   19,	37,   43,	53,   75,	85,   94,  127,  141,  142, 
   20,	32,   50,	58,   60,	84,   92,  113,  122,  148, 
   21,	38,   40,	61,   72,	83,   96,  114,  119,  137, 
   27,	28,   33,	54,   67,	87,  101,  132,  134,  136, 
   17,	23,   35,	47,  125,  139,  145,  146,  151,  157, 
   16,	22,   34,	46,  126,  138,  144,  147,  150,  156, 
   24,	41,   42,	56,   62,	91,   98,  118,  120,  154, 
   39,	51,  106,  107,  108,  109,  110,  111,  128,  152, 
   36,	48,   76,	77,   78,	79,   80,	81,  129,  153, 
   18,	29,   49,	57,   63,	89,   97,  124,  135,  155, 
	8,	 9,   10,	11,   12,	13,   14,	15,   90,	99, 
	0,	 1,    2,	 3,    4,	 5,    6,	 7,   86,	95, 
   64,	65,   66,	69,   70,	71,   73,	74,  130,  158, 
  100, 103,  104,  105,  112,  115,  116,  117,  133,  159, 
};
INT data2a[] = {
  134,  135,  140,  141,  146,  147,  152,  153,  158,  159, 
  119,  120,  122,  123,  125,  126,  128,  129,  130,  133, 
   40,   41,   43,   44,   46,   47,   48,   51,   90,   95, 
   28,   29,   31,   32,   34,   35,   36,   39,   86,   99, 
	0,    8,   16,   17,   18,   19,   20,   21,   74,  117, 
	1,    9,   22,   23,   24,   25,   26,   27,   73,  116, 
	2,   10,   71,   76,  106,  115,  136,  142,  148,  154, 
	3,   11,   70,   77,  107,  112,  137,  143,  149,  155, 
	4,   53,   56,   59,   63,   80,  105,  114,  132,  150, 
	5,   52,   58,   67,   72,   91,   97,  104,  109,  145, 
   12,   42,   49,   69,   83,   85,   93,  101,  110,  151, 
   13,   33,   38,   66,   79,   82,   92,  118,  124,  144, 
   14,   57,   62,   68,   75,   84,  100,  102,  111,  138, 
   15,   54,   55,   60,   61,   78,  103,  121,  127,  157, 
	6,   30,   37,   64,   81,   89,   98,  113,  131,  139, 
	7,   45,   50,   65,   87,   88,   94,   96,  108,  156, 
};


int main(int argc, char **argv)
{
	t0 = os_ticks();
	
	do_it();

	the_end_quietly(t0);
}

void do_it()
{
	INT v = 16;
	INT k = 10;
	INT N = 160;

	INT *M;
	INT *D1;
	INT *D2;
	INT i, j, a, b;
	BYTE fname[1000];
	INT *data1 = data1a;
	INT *data2 = data2a;

	M = NEW_INT(v * v);
	D1 = NEW_INT(N);
	D2 = NEW_INT(N);
	for (i = 0; i < v; i++) {
		for (j = 0; j < k; j++) {
			a = data1[i * k + j];
			D1[a] = i;
			a = data2[i * k + j];
			D2[a] = i;
			}
		}
	for (i = 0; i < v; i++) {
		for (j = 0; j < v; j++) {
			M[i * v + j] = 0;
			}
		}
	for (i = 0; i < N; i++) {
		a = D1[i];
		b = D2[i];
		M[a * v + b] = 1;
		}
	cout << "incidence matrix of the designs:" << endl;
	print_integer_matrix_width(cout, M, v, v, v, 4);

	strcpy(fname, "design.inc");
	write_incidence_matrix_to_file(fname, M, v, v, 1 /* verbose_level */);
}

