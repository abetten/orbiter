// util.cpp
//
// Anton Betten
//
// started:  October 23, 2002




#include "foundations.h"





using namespace std;


namespace orbiter {
namespace foundations {



int str2int(std::string &str)
{
	int i, res, l;

	l = (int) str.length();
	res = 0;
	for (i = 0; i < l; i++) {
		res = (res * 10) + (str[i] - 48);
	}
	return res;
}

#if 1
int my_atoi(char *str)
{
	int a;
	if (strlen(str)) {
		sscanf(str, "%d", &a); // don't use atoi, it fails for large numbers.
		return a;
	}
	return 0;
}

long int my_atol(const char *str)
{
	long int a;
	if (strlen(str)) {
		sscanf(str, "%ld", &a); // don't use atoi, it fails for large numbers.
		return a;
	}
	return 0;
}
#endif




int stringcmp(std::string &str, const char *p)
{
	return strcmp(str.c_str(), p);
}

int strtoi(std::string &str)
{
	int i;

	i = atoi(str.c_str());
	return i;
}

long int strtolint(std::string &str)
{
	long int i;

	i = atol(str.c_str());
	return i;
}

double strtof(std::string &str)
{
	double f;

	f = atof(str.c_str());
	return f;
}

int string_starts_with_a_number(std::string &str)
{
	char c;

	c = str.c_str()[0];
	if (c >= '0' && c <= '9') {
		return TRUE;
	}
	else {
		return FALSE;
	}
}



void itoa(char *p, int len_of_p, int i)
{
	sprintf(p, "%d", i);
#if 0
	ostrstream os(p, len_of_p);
	os << i << ends;
#endif
}




void print_line_of_number_signs()
{
	cout << "###########################################################"
			"#######################################" << endl;
}

void print_repeated_character(ostream &ost, char c, int n)
{
	int i;

	for (i = 0; i < n; i++) {
		ost << c;
	}
}

void print_pointer_hex(ostream &ost, void *p)
{
	void *q = p;
	uchar *pp = (uchar *)&q;
	int i, a, low, high;

	ost << "0x";
	for (i = (int)sizeof(pvoid) - 1; i >= 0; i--) {
		a = (int)pp[i];
		//cout << " a=" << a << " ";
		low = a % 16;
		high = a / 16;
		print_hex_digit(ost, high);
		print_hex_digit(ost, low);
	}
}

void print_hex_digit(ostream &ost, int digit)
{
	if (digit < 10) {
		ost << (char)('0' + digit);
	}
	else if (digit < 16) {
		ost << (char)('a' + (digit - 10));
	}
	else {
		cout << "print_hex_digit illegal digit " << digit << endl;
		exit(1);
	}
}



#if 1
//#define HASH_PRIME ((int) 1 << 30 - 1)
#define HASH_PRIME 174962718

int hashing(int hash0, int a)
{
	int h = hash0; // a1 = a;

	do {
		h <<= 1;
		if (ODD(a)){
			h++;
		}
		h = h % HASH_PRIME;	// h %= HASH_PRIME;
		a >>= 1;
	} while (a);
	//cout << "hashing: " << hash0 << " + " << a1 << " = " << h << endl;
	return h;
}

int hashing_fixed_width(int hash0, int a, int bit_length)
{
	int h = hash0;
	int a1 = a;
	int i;

	for (i = 0; i < bit_length; i++) {
		h <<= 1;
		if (ODD(a)){
			h++;
		}
		h = h % HASH_PRIME;	// h %= HASH_PRIME;
		a >>= 1;
	}
	if (a) {
		cout << "hashing_fixed_width a is not zero" << endl;
		cout << "a=" << a1 << endl;
		cout << "bit_length=" << bit_length << endl;
		exit(1);
	}
	//cout << "hashing: " << hash0 << " + " << a1 << " = " << h << endl;
	return h;
}
#endif


void char_swap(char *p, char *q, int len)
{
	int i;
	char c;

	for (i = 0; i < len; i++) {
		c = *q;
		*q++ = *p;
		*p++ = c;
	}
}

void int_swap(int& x, int& y)
{
	int z;

	z = x;
	x = y;
	y = z;
}

void uchar_print_bitwise(std::ostream &ost, uchar u)
{
	uchar mask;
	int i;

	for (i = 0; i < 8; i++) {
		mask = ((uchar) 1) << i;
		if (u & mask) {
			ost << "1";
		}
		else {
			ost << "0";
		}
	}
}

void uchar_move(uchar *p, uchar *q, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		*q++ = *p++;
	}
}

uint32_t root_of_tree_uint32_t (uint32_t* S, uint32_t i)
{
	while (S[i] != i) {
		i = S[i];
	}
	return i;
}

int util_compare_func(void *a, void *b, void *data)
{
	int *A = (int *)a;
	int *B = (int *)b;
	int *p = (int *) data;
	int n = *p;
	int i;

	for (i = 0; i < n; i++) {
		if (A[i] < B[i]) {
			return 1;
		}
		if (A[i] > B[i]) {
			return -1;
		}
	}
	return 0;
}

int compare_strings(void *a, void *b, void *data)
{
	char *A = (char *) a;
	char *B = (char *) b;
	return strcmp(A, B);
}






}}

