/*
 * mem_object_registry.cpp
 *
 *  Created on: Apr 23, 2019
 *      Author: betten
 */






#include "foundations.h"

using namespace std;




namespace orbiter {
namespace layer1_foundations {
namespace orbiter_kernel_system {


static int registry_key_pair_compare_by_size(void *K1v, void *K2v);
static int registry_key_pair_compare_by_type(void *K1v, void *K2v);
static int registry_key_pair_compare_by_location(void *K1v, void *K2v);




mem_object_registry::mem_object_registry()
{
	int verbose_level = 0;

	f_automatic_dump = false;
	automatic_dump_interval = 0;
	automatic_dump_fname_mask[0] = 0;

	entries = NULL;
	nb_allocate_total = 0;
	nb_delete_total = 0;
	cur_time = 0;

	f_ignore_duplicates = false;
	f_accumulate = false;

	init(verbose_level);
}

mem_object_registry::~mem_object_registry()
{
	if (entries) {
		delete [] entries;
		entries = NULL;
	}
}

void mem_object_registry::init(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::init" << endl;
	}

	nb_entries_allocated = REGISTRY_SIZE;
	nb_entries_used = 0;

	nb_allocate_total = 0;
	nb_delete_total = 0;
	cur_time = 0;

	f_ignore_duplicates = false;
	f_accumulate = false;

	if (f_v) {
		cout << "mem_object_registry::init trying to allocate "
				<< nb_entries_allocated << " entries" << endl;
	}

	entries = new mem_object_registry_entry[nb_entries_allocated];

	if (f_v) {
		cout << "mem_object_registry::init allocation successful" << endl;
	}


	if (f_v) {
		cout << "mem_object_registry::init done" << endl;
	}
}

void mem_object_registry::accumulate_and_ignore_duplicates(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::accumulate_and_ignore_duplicates" << endl;
	}
	f_accumulate = true;
	f_ignore_duplicates = true;
}

void mem_object_registry::allocate(int N, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::allocate" << endl;
	}

	nb_entries_allocated = N;
	nb_entries_used = 0;

	if (f_v) {
		cout << "mem_object_registry::allocate "
				"trying to allocate "
				<< nb_entries_allocated << " entries" << endl;
	}

	entries = new mem_object_registry_entry[nb_entries_allocated];

	if (f_v) {
		cout << "mem_object_registry::allocate allocation successful" << endl;
	}


	if (f_v) {
		cout << "mem_object_registry::allocate done" << endl;
	}
}



void mem_object_registry::set_automatic_dump(
		int automatic_dump_interval, const char *fname_mask,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::set_automatic_dump" << endl;
	}
	f_automatic_dump = true;
	mem_object_registry::automatic_dump_interval = automatic_dump_interval;
	strcpy(automatic_dump_fname_mask, fname_mask);
}

void mem_object_registry::automatic_dump()
{
	if (!f_automatic_dump) {
		return;
	}
	if ((cur_time % automatic_dump_interval) != 0) {
		return;
	}
	char fname[1000];
	int a;

	a = cur_time / automatic_dump_interval;

	cout << "automatic memory dump " << a << endl;
	snprintf(fname, sizeof(fname), automatic_dump_fname_mask, a);

	dump_to_csv_file(fname);
}

void mem_object_registry::manual_dump()
{
	if (!f_automatic_dump) {
		return;
	}
	char fname[1000];
	int a;

	a = cur_time / automatic_dump_interval + 1;

	snprintf(fname, sizeof(fname), automatic_dump_fname_mask, a);

	dump_to_csv_file(fname);
}

void mem_object_registry::manual_dump_with_file_name(
		const char *fname)
{
	dump_to_csv_file(fname);
}

void mem_object_registry::dump()
{
	int i, s, sz;

	cout << "memory registry:" << endl;

	sz = 0;
	for (i = 0; i < nb_entries_used; i++) {
		s = entries[i].size_of();
		sz += s;
	}

	cout << "nb_entries_used=" << nb_entries_used << endl;
	cout << "nb_allocate_total=" << nb_allocate_total << endl;
	cout << "nb_delete_total=" << nb_delete_total << endl;
	cout << "cur_time=" << cur_time << endl;
	cout << "total allocation size in char=" << sz << endl;
	cout << "table of all currently active memory allocations in increasing "
			"order of the value of the pointer" << endl;
	for (i = 0; i < nb_entries_used; i++) {
		entries[i].print(i);
	}
}

void mem_object_registry::dump_to_csv_file(const char *fname)
{
	int i, s, sz;


	{
		ofstream fp(fname);


		//cout << "memory registry:" << endl;

		fp << "Line,Pointer,Timestamp,Type,N,Sizeof,"
				"ExtraTypeInfo,File,LineInFile" << endl;
		sz = 0;
		for (i = 0; i < nb_entries_used; i++) {
			s = entries[i].size_of();
			sz += s;
		}

		for (i = 0; i < nb_entries_used; i++) {
			entries[i].print_csv(fp, i);
		}
		fp << "END" << endl;
		fp << "nb_entries_used=" << nb_entries_used << endl;
		fp << "nb_allocate_total=" << nb_allocate_total << endl;
		fp << "nb_delete_total=" << nb_delete_total << endl;
		fp << "cur_time=" << cur_time << endl;
		fp << "total allocation size in char=" << sz << endl;
	}
}


int *mem_object_registry::allocate_int(long int n,
		const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	int *p;
	p = new int[n];

	if (f_v) {
		cout << "mem_object_registry::allocate_int cur_time=" << cur_time << " int[n], "
				"n=" << n << " file=" << file << " line=" << line << " p=" << p << endl;
	}
	if (Orbiter->f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_int, n, sizeof(int),
				"", file, line,
				Orbiter->memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_int(int *p, const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_int int[n], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_int "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (Orbiter->f_memory_debug) {
		delete_from_registry(p, Orbiter->memory_debug_verbose_level - 1);
	}
	delete [] p;
}

long int *mem_object_registry::allocate_lint(long int n,
		const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	long int *p;
	p = new long int[n];
	if (f_v) {
		cout << "mem_object_registry::allocate_lint cur_time=" << cur_time << " int[n], "
				"n=" << n << " file=" << file << " line=" << line << " p=" << p << endl;
	}
	if (Orbiter->f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_lint, n, sizeof(int),
				"", file, line,
				Orbiter->memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_lint(long int *p, const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_lint int[n], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_lint "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (Orbiter->f_memory_debug) {
		delete_from_registry(p, Orbiter->memory_debug_verbose_level - 1);
	}
	delete [] p;
}

int **mem_object_registry::allocate_pint(long int n,
		const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	int **p;
	p = new pint[n];
	if (f_v) {
		cout << "mem_object_registry::allocate_pint cur_time=" << cur_time << " pint[n], "
				"n=" << n << " file=" << file << " line=" << line << " p=" << p << endl;
	}
	if (Orbiter->f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_pint, n, sizeof(int *),
				"", file, line,
				Orbiter->memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_pint(int **p,
		const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_pint pint[n], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_pint "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (Orbiter->f_memory_debug) {
		delete_from_registry(p, Orbiter->memory_debug_verbose_level - 1);
	}
	delete [] p;
}

long int **mem_object_registry::allocate_plint(long int n,
		const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	long int **p;
	p = new plint[n];
	if (f_v) {
		cout << "mem_object_registry::allocate_plint cur_time=" << cur_time << " plint[n], "
				"n=" << n << " file=" << file << " line=" << line << " p=" << p << endl;
	}
	if (Orbiter->f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_plint, n, sizeof(long int *),
				"", file, line,
				Orbiter->memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_plint(long int **p,
		const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_plint pint[n], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_plint "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (Orbiter->f_memory_debug) {
		delete_from_registry(p, Orbiter->memory_debug_verbose_level - 1);
	}
	delete [] p;
}

int ***mem_object_registry::allocate_ppint(long int n,
		const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	int ***p;
	p = new ppint[n];
	if (f_v) {
		cout << "mem_object_registry::allocate_ppint cur_time=" << cur_time << " ppint[n], "
				"n=" << n << " file=" << file << " line=" << line << " p=" << p << endl;
	}
	if (Orbiter->f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_ppint, n, sizeof(int **),
				"", file, line,
				Orbiter->memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_ppint(int ***p, const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_ppint ppint[n], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_ppint "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (Orbiter->f_memory_debug) {
		delete_from_registry(p, Orbiter->memory_debug_verbose_level - 1);
	}
	delete [] p;
}

long int ***mem_object_registry::allocate_pplint(long int n,
		const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	long int ***p;
	p = new pplint[n];
	if (f_v) {
		cout << "mem_object_registry::allocate_ppint cur_time=" << cur_time << " pplint[n], "
				"n=" << n << " file=" << file << " line=" << line << " p=" << p << endl;
	}
	if (Orbiter->f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_pplint, n, sizeof(int **),
				"", file, line,
				Orbiter->memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_pplint(long int ***p, const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_pplint ppint[n], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_pplint "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (Orbiter->f_memory_debug) {
		delete_from_registry(p, Orbiter->memory_debug_verbose_level - 1);
	}
	delete [] p;
}

char *mem_object_registry::allocate_char(long int n,
		const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	char *p;
	p = new char[n];
	if (f_v) {
		cout << "mem_object_registry::allocate_char cur_time=" << cur_time << " char[n], "
				"n=" << n << " file=" << file << " line=" << line << " p=" << (int *) p << endl;
	}
	if (Orbiter->f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_char, n, sizeof(char),
				"", file, line,
				Orbiter->memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_char(char *p, const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_char char[n], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_char "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (Orbiter->f_memory_debug) {
		delete_from_registry(p, Orbiter->memory_debug_verbose_level - 1);
	}
	delete [] p;
}

uchar *mem_object_registry::allocate_uchar(long int n,
		const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	uchar *p;
	p = new uchar[n];
	if (f_v) {
		cout << "mem_object_registry::allocate_uchar cur_time=" << cur_time << " uchar[n], "
				"n=" << n << " file=" << file << " line=" << line << " p=" << (int *) p << endl;
	}
	if (Orbiter->f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_uchar, n, sizeof(uchar),
				"", file, line,
				Orbiter->memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_uchar(uchar *p,
		const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_uchar uchar[n], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_uchar "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (Orbiter->f_memory_debug) {
		delete_from_registry(p, Orbiter->memory_debug_verbose_level - 1);
	}
	delete [] p;
}

char **mem_object_registry::allocate_pchar(long int n,
		const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	char **p;
	p = new pchar[n];
	if (f_v) {
		cout << "mem_object_registry::allocate_pchar cur_time=" << cur_time << " pchar[n], "
				"n=" << n << " file=" << file << " line=" << line << " p=" << (int *) p << endl;
	}
	if (Orbiter->f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_pchar, n, sizeof(char *),
				"", file, line,
				Orbiter->memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_pchar(char **p,
		const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_pchar pchar[n], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_pchar "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (Orbiter->f_memory_debug) {
		delete_from_registry(p, Orbiter->memory_debug_verbose_level - 1);
	}
	delete [] p;
}

uchar **mem_object_registry::allocate_puchar(long int n,
		const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	uchar **p;
	p = new puchar[n];
	if (f_v) {
		cout << "mem_object_registry::allocate_puchar cur_time=" << cur_time << " puchar[n], "
				"n=" << n << " file=" << file << " line=" << line << " p=" << (int *) p << endl;
	}
	if (Orbiter->f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_puchar, n, sizeof(char *),
				"", file, line,
				Orbiter->memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_puchar(uchar **p,
		const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_puchar puchar[n], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_puchar "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (Orbiter->f_memory_debug) {
		delete_from_registry(p, Orbiter->memory_debug_verbose_level - 1);
	}
	delete [] p;
}

void **mem_object_registry::allocate_pvoid(long int n,
		const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	void **p;
	p = new pvoid[n];
	if (f_v) {
		cout << "mem_object_registry::allocate_pvoid cur_time=" << cur_time << " pvoid[n], "
				"n=" << n << " file=" << file << " line=" << line << " p=" << p << endl;
	}
	if (Orbiter->f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_PVOID, n, sizeof(void *),
				"", file, line,
				Orbiter->memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_pvoid(void **p,
		const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_pvoid pvoid[n], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_pvoid "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (Orbiter->f_memory_debug) {
		delete_from_registry(p, Orbiter->memory_debug_verbose_level - 1);
	}
	delete [] p;
}

void *mem_object_registry::allocate_OBJECTS(void *p,
		long int n, std::size_t size_of,
		const char *extra_type_info, const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::allocate_OBJECTS cur_time=" << cur_time << " char[n * size_of], "
				"n=" << n << " file=" << file << " line=" << line << " p=" << p << endl;
	}
	if (Orbiter->f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_OBJECTS, n, (int) size_of,
				extra_type_info, file, line,
				Orbiter->memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_OBJECTS(void *p,
		const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_OBJECTS char[n * size_of], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_OBJECTS "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (Orbiter->f_memory_debug) {
		delete_from_registry(p, Orbiter->memory_debug_verbose_level - 1);
	}
	//delete [] p;
}

void *mem_object_registry::allocate_OBJECT(void *p, std::size_t size_of,
		const char *extra_type_info, const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::allocate_OBJECT cur_time=" << cur_time << " char[size_of], "
				" file=" << file << " line=" << line << " p=" << p << endl;
	}
	if (Orbiter->f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_OBJECT, 1, (int) size_of,
				extra_type_info, file, line,
				Orbiter->memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_OBJECT(void *p,
		const char *file, int line)
{
	int f_v = (Orbiter->memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_OBJECT char[size_of], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_OBJECTS "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (Orbiter->f_memory_debug) {
		delete_from_registry(p, Orbiter->memory_debug_verbose_level - 1);
	}
	//delete [] p;
}





int mem_object_registry::search(void *p, int &idx)
{
	int l, r, m;
	int f_found = false;

	if (nb_entries_used == 0) {
		idx = 0;
		return false;
		}
	l = 0;
	r = nb_entries_used;
	// invariant:
	// v[i] <= a for i < l;
	// v[i] >  a for i >= r;
	// r - l is the length of the area to search in.
	while (l < r) {
		m = (l + r) >> 1;
		// if the length of the search area is even
		// we examine the element above the middle
		//res = registry_pointer[m] - p;
		//cout << "search l=" << l << " m=" << m << " r="
		//	<< r << "a=" << a << " v[m]=" << v[m] << " res=" << res << endl;
		if (p >= entries[m].pointer) {
			l = m + 1;
			if (p == entries[m].pointer)
				f_found = true;
			}
		else
			r = m;
		}
	// now: l == r;
	// and f_found is set accordingly */
	if (f_found) {
		l--;
	}
	idx = l;
	return f_found;
}

void mem_object_registry::insert_at(int idx)
{
	int i;

	if (nb_entries_used == nb_entries_allocated) {
		nb_entries_allocated = 2 * nb_entries_allocated;
		cout << "mem_object_registry::insert_at reallocating table to "
				<< nb_entries_allocated << " elements" << endl;
		mem_object_registry_entry *old_entries;

		old_entries = entries;
		entries = new mem_object_registry_entry[nb_entries_allocated];
		for (i = 0; i < nb_entries_used; i++) {
			entries[i] = old_entries[i];
		}
		delete [] old_entries;
	}
	for (i = nb_entries_used; i > idx; i--) {
		entries[i] = entries[i - 1];
		}
	entries[idx].null();
	nb_entries_used++;
}

void mem_object_registry::add_to_registry(void *pointer,
		int object_type, long int object_n, int object_size_of,
		const char *extra_type_info,
		const char *source_file, int source_line,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx;

	if (f_v) {
		cout << "mem_object_registry::add_to_registry" << endl;
	}
	nb_allocate_total++;
	if (search(pointer, idx)) {
		if (f_ignore_duplicates) {

		}
		else {
			cout << "mem_object_registry::add_to_registry pointer p is "
					"already in the registry, something is wrong" << endl;
			cout << "extra_type_info = " << extra_type_info << endl;
			cout << "source_file = " << source_file << endl;
			cout << "source_line = " << source_line << endl;
			cout << "object_type = " << object_type << endl;
			cout << "object_n = " << object_n << endl;
			cout << "object_size_of = " << object_size_of << endl;
			cout << "the previous object is:" << endl;
			entries[idx].print(idx);
			cout << "ignoring the problem" << endl;
			//exit(1);
		}
	}
	insert_at(idx);
	entries[idx].time_stamp = cur_time;
	entries[idx].pointer = pointer;
	entries[idx].object_type = object_type;
	entries[idx].object_n = object_n;
	entries[idx].object_size_of = object_size_of;
	entries[idx].extra_type_info = extra_type_info;
	entries[idx].source_file = source_file;
	entries[idx].source_line = source_line;



	automatic_dump();
	cur_time++;

	if (f_v) {
		cout << "mem_object_registry::add_to_registry done, there are "
				<< nb_entries_used << " entries in the registry" << endl;
	}
}

void mem_object_registry::delete_from_registry(void *pointer, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx, i;

	if (f_v) {
		cout << "mem_object_registry::delete_from_registry" << endl;
	}
	nb_delete_total++;

	if (f_accumulate) {
		// do not delete entries so we can see all allocations
	}
	else {
		if (!search(pointer, idx)) {
			cout << "mem_object_registry::delete_from_registry pointer is "
					"not in registry, something is wrong; "
					"ignoring, pointer = " << pointer << endl;
			//exit(1);
		}
		for (i = idx + 1; i < nb_entries_used; i++) {
			entries[i - 1] = entries[i];
			}
		entries[nb_entries_used - 1].null();
		nb_entries_used--;
	}
	automatic_dump();
	//cur_time++;
	if (f_v) {
		cout << "mem_object_registry::delete_from_registry done, there are "
				<< nb_entries_used << " entries in the registry" << endl;
	}
}
void mem_object_registry::sort_by_size(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "mem_object_registry::sort_by_size" << endl;
	}

	if (f_v) {
		cout << "mem_object_registry::sort_by_size before Heapsort" << endl;
	}
	Sorting.Heapsort(entries, nb_entries_used,
		sizeof(mem_object_registry_entry),
		registry_key_pair_compare_by_size);
	if (f_v) {
		cout << "mem_object_registry::sort_by_size after Heapsort" << endl;
	}

	if (f_v) {
		cout << "mem_object_registry::sort_by_size done" << endl;
	}

}

void mem_object_registry::sort_by_location_and_get_frequency(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "mem_object_registry::sort_by_location_and_get_frequency" << endl;
	}

	sort_by_location(verbose_level - 1);

	int nb_types;
	int *type_first;
	int *type_len;
	int c, f, l;

	type_first = new int[nb_entries_used]; // use system memory
	type_len = new int[nb_entries_used];


	nb_types = 0;
	type_first[0] = 0;
	type_len[0] = 1;
	for (i = 1; i < nb_entries_used; i++) {
		c = registry_key_pair_compare_by_location(
				entries + i, entries + (i - 1));
		if (c == 0) {
			type_len[nb_types]++;
			}
		else {
			type_first[nb_types + 1] =
					type_first[nb_types] + type_len[nb_types];
			nb_types++;
			type_len[nb_types] = 1;
			}
		}
	nb_types++;
	cout << "we have " << nb_types
			<< " different allocation locations:" << endl;

	int t, j, sz, s;
	int *perm;
	int *perm_inv;
	int *frequency;

	perm = new int[nb_types]; // use system memory
	perm_inv = new int[nb_types];
	frequency = new int[nb_types];

	Int_vec_copy(type_len, frequency, nb_types);

	Sorting.int_vec_sorting_permutation(frequency, nb_types,
			perm, perm_inv, false /* f_increasingly */);

	for (t = nb_types - 1; t >= 0; t--) {
		i = perm_inv[t];

		f = type_first[i];
		l = type_len[i];

		sz = 0;
		for (j = 0; j < l; j++) {
			s = entries[f + j].size_of();
			sz += s;
		}

		//idx = entries[f].user_data;
		cout << l << " times file "
				<< entries[f].source_file << " line "
				<< entries[f].source_line
				<< " object type ";
		entries[f].print_type(cout);
		if (entries[f].object_type == POINTER_TYPE_OBJECT ||
				entries[f].object_type == POINTER_TYPE_OBJECTS) {
			cout << " = " << entries[f].extra_type_info;
		}
		cout << " for a total of " << sz << " char" << endl;
		}

	delete [] type_first;
	delete [] type_len;
	delete [] perm;
	delete [] perm_inv;
	delete [] frequency;

	if (f_v) {
		cout << "mem_object_registry::sort_by_location_and_get_frequency" << endl;
	}
}

void mem_object_registry::sort_by_type(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "mem_object_registry::sort_by_type" << endl;
	}

	if (f_v) {
		cout << "mem_object_registry::sort_by_type "
				"before Heapsort" << endl;
	}

	Sorting.Heapsort(entries, nb_entries_used,
		sizeof(mem_object_registry_entry),
		registry_key_pair_compare_by_type);

	if (f_v) {
		cout << "mem_object_registry::sort_by_type "
				"after Heapsort" << endl;
	}

	if (f_v) {
		cout << "mem_object_registry::sort_by_type "
				"done" << endl;
	}

}

void mem_object_registry::sort_by_location(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "mem_object_registry::sort_by_location" << endl;
	}

	if (f_v) {
		cout << "mem_object_registry::sort_by_location "
				"before Heapsort" << endl;
	}

	Sorting.Heapsort(entries, nb_entries_used,
		sizeof(mem_object_registry_entry),
		registry_key_pair_compare_by_location);

	if (f_v) {
		cout << "mem_object_registry::sort_by_location "
				"after Heapsort" << endl;
	}

	if (f_v) {
		cout << "mem_object_registry::sort_by_location "
				"done" << endl;
	}

}

//##############################################################################
// global functions:
//##############################################################################


static int registry_key_pair_compare_by_size(void *K1v, void *K2v)
{
	int s1, s2, c;
	mem_object_registry_entry *K1, *K2;

	K1 = (mem_object_registry_entry *) K1v;
	K2 = (mem_object_registry_entry *) K2v;
	s1 = K1->size_of();
	s2 = K2->size_of();
	c = s2 - s1;
	return c;
}

static int registry_key_pair_compare_by_type(void *K1v, void *K2v)
{
	int t1, t2, l1, l2, c;
	mem_object_registry_entry *K1, *K2;

	K1 = (mem_object_registry_entry *) K1v;
	K2 = (mem_object_registry_entry *) K2v;
	t1 = K1->object_type;
	t2 = K2->object_type;
	c = t2 - t1;
	if (c) {
		return c;
	}
	//new the two entries have the same type
	if (t1 == POINTER_TYPE_OBJECTS || t1 == POINTER_TYPE_OBJECT) {
		c = strcmp(K1->extra_type_info, K2->extra_type_info);
		if (c) {
			return c;
		}
	}
	c = strcmp(K1->source_file, K2->source_file);
	if (c) {
		return c;
	}
	l1 = K1->source_line;
	l2 = K2->source_line;
	c = l2 - l1;
	return c;
}

static int registry_key_pair_compare_by_location(void *K1v, void *K2v)
{
	int l1, l2, c;
	mem_object_registry_entry *K1, *K2;

	K1 = (mem_object_registry_entry *) K1v;
	K2 = (mem_object_registry_entry *) K2v;
	c = strcmp(K1->source_file, K2->source_file);
	if (c) {
		return c;
	}
	l1 = K1->source_line;
	l2 = K2->source_line;
	c = l2 - l1;
	return c;
}


}}}

