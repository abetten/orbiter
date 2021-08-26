// os.cpp


#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {



#include <math.h>
#include <fcntl.h>
#include <stdlib.h>
#include <time.h>

#include <fcntl.h>
/* #include <malloc.h> */
#include <unistd.h>
	/* for sysconf */
#include <limits.h>
	/* for CLK_TCK */
#include <sys/times.h>
	/* for times() */


int os_ticks()
{
#ifdef SYSTEMMAC
	clock_t t;
	
	t = clock();
	return((int)t);
#endif
#ifdef SYSTEMUNIX
	struct tms tms_buffer;

	if (-1 == times(&tms_buffer))
		return(-1);
	return(tms_buffer.tms_utime);
#endif
	return(0);
}

//static int system_time0 = 0;

int os_ticks_system()
{
#ifdef SYSTEMMAC
	clock_t t;
	
	t = clock();
	return((int)t);
#endif
#ifdef SYSTEMUNIX
#if 0
	struct tms tms_buffer;

	if (-1 == times(&tms_buffer))
		return(-1);
	return(tms_buffer.tms_stime);
#endif
	int t;

	t = time(NULL);
	t *= os_ticks_per_second();
	return t;
#endif
	return(0);
}

int os_ticks_per_second()
{
	int clk_tck = 1;
	
#ifdef SYSTEMUNIX
	clk_tck = sysconf(_SC_CLK_TCK);
	/* printf("clk_tck = %ld\n", clk_tck); */
#endif
#ifdef SYSTEMMAC
	clk_tck = CLOCKS_PER_SEC;
#endif
	return(clk_tck);
}

void os_ticks_to_hms(int ticks,
	int *h, int *m, int *s)
{
	int l1, clk_tck;

	clk_tck = os_ticks_per_second();
	l1 = ticks / clk_tck;
	*s = l1 % 60L;
	l1 -= *s;
	l1 /= 60;
	*m = l1 % 60L;
	l1 -= *m;
	l1 /= 60;
	*h = l1;
}

void print_delta_time(int l, char *str)
{
	int h, m, s;

	os_ticks_to_hms(l, &h, &m, &s);
	sprintf(Eostr(str), 
		"%d:%02d:%02d", h, m, s);
}



char *Eostr(char *s)
{
	return(&s[strlen(s)]);
}


char *eostr(char *s)
{
	return(&s[strlen(s)]);
}

int ij2k(int i, int j, int n)
{
	if (i == j) {
		cout << "ij2k() i == j" << endl;
		exit(1);
	}
	if (i > j) {
		return ij2k(j, i, n);
	}
	return ((n - i) * i + ((i * (i - 1)) >> 1) + j - i - 1);
}


void k2ij(int k, int *i, int *j, int n)
{
	int ii;
	
	for (ii = 0; ii < n; ii++) {
		if (k < n - ii - 1) {
			*i = ii;
			*j = k + ii + 1;
			return;
			}
		k -= (n - ii - 1);
		}
	cout << "k too large" << endl;
	exit(1);
}


}}



