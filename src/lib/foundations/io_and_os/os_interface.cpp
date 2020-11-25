/*
 * os_interface.cpp
 *
 *  Created on: Sep 28, 2019
 *      Author: betten
 */





#include "foundations.h"

using namespace std;

#include <cstdio>
#include <sys/types.h>
#ifdef SYSTEMUNIX
#include <unistd.h>
#endif
#include <fcntl.h>

#ifdef SYSTEMUNIX
#ifndef SYSTEMWINDOWS
#include <sys/times.h>
	/* for times() */
#endif
#endif
#include <time.h>
	/* for time() */
#ifdef SYSTEMWINDOWS
#include <io.h>
#include <process.h>
#endif
#ifdef SYSTEMMAC
#include <console.h>
#include <time.h> // for clock()
#include <unix.h>
#endif
#ifdef MSDOS
#include <time.h> // for clock()
#endif


namespace orbiter {
namespace foundations {




void os_interface::runtime(long *l)
{
	*l = 0;
#ifdef SYSTEMUNIX
#ifndef SYSTEMWINDOWS
	struct tms *buffer = (struct tms *) malloc(sizeof(struct tms));
	times(buffer);
	*l = (long) buffer->tms_utime;
	free(buffer);
#endif
#endif
#ifdef SYSTEMMAC
	*l = 0;
#endif
#ifdef MSDOS
	*l = (long) clock();
#endif /* MSDOS */
}


int os_interface::os_memory_usage()
{
#ifdef SYSTEM_IS_MACINTOSH
	struct task_basic_info t_info;
	mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;

	if (KERN_SUCCESS != task_info(mach_task_self(),
		                      TASK_BASIC_INFO, (task_info_t)&t_info,
		                      &t_info_count))
	{
		cout << "os_memory_usage() error in task_info" << endl;
		exit(1);
	}
	// resident size is in t_info.resident_size;
	// virtual size is in t_info.virtual_size;


	//cout << "resident_size=" << t_info.resident_size << endl;
	//cout << "virtual_size=" << t_info.virtual_size << endl;
	return t_info.resident_size;
#endif
#ifdef SYSTEM_LINUX
	int chars = 128;
		// number of characters to read from the
		//  /proc/self/status file in a given line
	FILE* file = fopen("/proc/self/status", "r");
	char line[chars];
	while (fgets(line, chars, file) != NULL) {
		// read one line at a time
		if (strncmp(line, "VmPeak:", 7) == 0) {
			// compare the first 7 characters of every line
			char* p = line + 7;
			// start reading from the 7th index of the line
			p[strlen(p)-3] = '\0';
			// set the null terminator at the beginning of size units
			fclose(file);
			// close the file stream
			return atoi(p);
			// return the size in KiB
			}
		}
#endif
	return 0;
}

int os_interface::os_ticks()
{
#ifdef SYSTEMUNIX
	struct tms tms_buffer;
	int t;

	if (-1 == (int) times(&tms_buffer))
		return(-1);
	t = tms_buffer.tms_utime;
	//cout << "os_ticks " << t << endl;
	return t;
#endif
#ifdef SYSTEMMAC
	clock_t t;

	t = clock();
	return((int)t);
#endif
#ifdef SYSTEMWINDOWS
	return 0;
#endif
}

static int f_system_time_set = FALSE;
static int system_time0 = 0;

int os_interface::os_ticks_system()
{
	int t;

	t = time(NULL);
	if (!f_system_time_set) {
		f_system_time_set = TRUE;
		system_time0 = t;
		}
	//t -= system_time0;
	//t *= os_ticks_per_second();
	return t;
}

int os_interface::os_ticks_per_second()
{
	static int f_tps_computed = FALSE;
	static int tps = 0;
#ifdef SYSTEMUNIX
	int clk_tck = 1;

	if (f_tps_computed)
		return tps;
	else {
		clk_tck = sysconf(_SC_CLK_TCK);
		tps = clk_tck;
		f_tps_computed = TRUE;
		//cout << endl << "clock ticks per second = " << tps << endl;
		return(clk_tck);
		}
#endif
#ifdef SYSTEMWINDOWS
	return 1;
#endif
}

void os_interface::os_ticks_to_dhms(int ticks,
		int tps, int &d, int &h, int &m, int &s)
{
	int l1;
	int f_v = FALSE;

	if (f_v) {
		cout << "os_ticks_to_dhms ticks = " << ticks << endl;
		}
	l1 = ticks / tps;
	if (f_v) {
		cout << "os_ticks_to_dhms l1 = " << l1 << endl;
		}
	s = l1 % 60;
	if (f_v) {
		cout << "os_ticks_to_dhms s = " << s << endl;
		}
	l1 /= 60;
	m = l1 % 60;
	if (f_v) {
		cout << "os_ticks_to_dhms m = " << m << endl;
		}
	l1 /= 60;
	h = l1;
	if (f_v) {
		cout << "os_ticks_to_dhms h = " << h << endl;
		}
	if (h >= 24) {
		d = h / 24;
		h = h % 24;
		}
	else
		d = 0;
	if (f_v) {
		cout << "os_ticks_to_dhms d = " << d << endl;
		}
}

void os_interface::time_check_delta(ostream &ost, int dt)
{
	int tps, d, h, min, s;

	tps = os_ticks_per_second();
	//cout << "time_check_delta tps=" << tps << endl;
	os_ticks_to_dhms(dt, tps, d, h, min, s);

	if ((dt / tps) >= 1) {
		print_elapsed_time(ost, d, h, min, s);
		}
	else {
		ost << "0:00";
		}
	//cout << endl;
}

void os_interface::print_elapsed_time(ostream &ost, int d, int h, int m, int s)
{
	if (d > 0) {
		ost << d << "-" << h << ":" << m << ":" << s;
		}
	else if (h > 0) {
		ost << h << ":" << m << ":" << s;
		}
	else  {
		ost << m << ":" << s;
		}
}

void os_interface::time_check(ostream &ost, int t0)
{
	int t1, dt;

	t1 = os_ticks();
	dt = t1 - t0;
	//cout << "time_check t0=" << t0 << endl;
	//cout << "time_check t1=" << t1 << endl;
	//cout << "time_check dt=" << dt << endl;
	time_check_delta(ost, dt);
}

int os_interface::delta_time(int t0)
{
	int t1, dt;

	t1 = os_ticks();
	dt = t1 - t0;
	return dt;
}


void os_interface::seed_random_generator_with_system_time()
{
	srand((unsigned int) time(0));
}

void os_interface::seed_random_generator(int seed)
{
	srand((unsigned int) seed);
}

int os_interface::random_integer(int p)
// computes a random integer r with $0 \le r < p.$
{
	int n;

	if (p == 0) {
		cout << "random_integer p = 0" << endl;
		exit(1);
		}
	n = (int)(((double)rand() * (double)p / RAND_MAX)) % p;
	return n;
}

void os_interface::os_date_string(char *str, int sz)
{
	system("date >a");
	{
	ifstream f1("a");
	f1.getline(str, sz);
	}
}

int os_interface::os_seconds_past_1970()
{
	int a;

	{
	ofstream fp("b");
	fp << "#!/bin/bash" << endl;
	fp << "echo $(date +%s)" << endl;
	}
	system("chmod ugo+x b");
	system("./b >a");
	{
	char str[1000];

	ifstream f1("a");
	f1.getline(str, sizeof(str));
	sscanf(str, "%d", &a);
	}
	return a;
}

void os_interface::get_string_from_command_line(std::string &p, int argc, std::string *argv,
		int &i, int verbose_level)
{
	if (stringcmp(argv[i], "-long_string") == 0) {
		i++;
		p.assign("");
		while (TRUE) {
			if (stringcmp(argv[i], "-end_string") == 0) {
				i++;
				break;
			}
			p.append(argv[i]);
			i++;
		}
	}
	else {
		p.assign(argv[i]);
		i++;
	}
}

static const char *ascii_code = "abcdefghijklmnop";

static int f_has_swap_initialized = FALSE;
static int f_has_swap = 0;
	// indicates if char swap is present
	// i.e., little endian / big endian

void os_interface::test_swap()
{
	//unsigned long test_long = 0x11223344L;
	int_4 test = 0x11223344L;
	char *ptr;

	ptr = (char *) &test;
	if (ptr[0] == 0x44) {
		f_has_swap = TRUE;
		cout << "we have a swap" << endl;
	}
	else if (ptr[0] == 0x11) {
		f_has_swap = FALSE;
		cout << "we don't have a swap" << endl;
	}
	else {
		cout << "The test_swap test is inconclusive" << endl;
		exit(1);
	}
	f_has_swap_initialized = TRUE;
}

// block_swap_chars:
// switches the chars in the buffer pointed to by "ptr".
// There are "no" intervals of size "size".
// This routine is due to Roland Grund

void os_interface::block_swap_chars(char *ptr, int size, int no)
{
	char *ptr_end, *ptr_start;
	char chr;
	int i;

	if (!f_has_swap_initialized) {
		test_swap();
	}
	if ((f_has_swap) && (size > 1)) {

		for (; no--; ) {

			ptr_start = ptr;
			ptr_end = ptr_start + (size - 1);
			for (i = size / 2; i--; ) {
				chr = *ptr_start;
				*ptr_start++ = *ptr_end;
				*ptr_end-- = chr;
			}
			ptr += size;
		}
	}
}

void os_interface::code_int4(char *&p, int_4 i)
{
	int_4 ii = i;

	//cout << "code_int4 " << i << endl;
	uchar *q = (uchar *) &ii;
	//block_swap_chars((SCHAR *)&ii, 4, 1);
	code_uchar(p, q[0]);
	code_uchar(p, q[1]);
	code_uchar(p, q[2]);
	code_uchar(p, q[3]);
}

int_4 os_interface::decode_int4(char *&p)
{
	int_4 ii;
	uchar *q = (uchar *) &ii;
	decode_uchar(p, q[0]);
	decode_uchar(p, q[1]);
	decode_uchar(p, q[2]);
	decode_uchar(p, q[3]);
	//block_swap_chars((SCHAR *)&ii, 4, 1);
	//cout << "decode_int4 " << ii << endl;
	return ii;
}

void os_interface::code_uchar(char *&p, uchar a)
{
	//cout << "code_uchar " << (int) a << endl;
	int a_high = a >> 4;
	int a_low = a & 15;
	*p++ = ascii_code[a_high];
	*p++ = ascii_code[a_low];
}

void os_interface::decode_uchar(char *&p, uchar &a)
{
	int a_high = (int)(*p++ - 'a');
	int a_low = (int)(*p++ - 'a');
	int i;
	//cout << "decode_uchar a_high = " << a_high << endl;
	//cout << "decode_uchar a_low = " << a_low << endl;
	i = (a_high << 4) | a_low;
	//cout << "decode_uchar i = " << i << endl;
	//cout << "decode_uchar " << (int) i << endl;
	a = (uchar)i;
}



}}
