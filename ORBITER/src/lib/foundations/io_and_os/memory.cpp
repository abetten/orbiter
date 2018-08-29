// memory.C
//
// Anton Betten
//
// started:  June 25, 2009




#include "foundations.h"

#define REGISTRY_SIZE 1000
#define POINTER_TYPE_INVALID 0
#define POINTER_TYPE_SMALLINT 1
#define POINTER_TYPE_SMALLPINT 2
#define POINTER_TYPE_INT 3
#define POINTER_TYPE_PINT 4
#define POINTER_TYPE_PPINT 5
#define POINTER_TYPE_BYTE 6
#define POINTER_TYPE_UBYTE 7
#define POINTER_TYPE_PBYTE 8
#define POINTER_TYPE_PUBYTE 9
#define POINTER_TYPE_PVOID 10
#define POINTER_TYPE_OBJECT 11
#define POINTER_TYPE_OBJECTS 12



int f_memory_debug = FALSE;
int memory_debug_verbose_level = 0;
mem_object_registry global_mem_object_registry;

static INT registry_key_pair_compare_by_size(void *K1v, void *K2v);
static INT registry_key_pair_compare_by_type(void *K1v, void *K2v);
static INT registry_key_pair_compare_by_location(void *K1v, void *K2v);

mem_object_registry_entry::mem_object_registry_entry()
{
	null();
}

mem_object_registry_entry::~mem_object_registry_entry()
{

}

void mem_object_registry_entry::null()
{
	time_stamp = 0;
	pointer = NULL;
	object_type = POINTER_TYPE_INVALID;
	object_n = 0;
	object_size_of = 0;
	extra_type_info = NULL;
	source_file = NULL;
	source_line = 0;
}

void mem_object_registry_entry::set_type_from_string(BYTE *str)
{
	if (strcmp(str, "int") == 0) {
		object_type = POINTER_TYPE_SMALLINT;
	} else if (strcmp(str, "pint") == 0) {
		object_type = POINTER_TYPE_SMALLPINT;
	} else if (strcmp(str, "INT") == 0) {
		object_type = POINTER_TYPE_INT;
	} else if (strcmp(str, "PINT") == 0) {
		object_type = POINTER_TYPE_PINT;
	} else if (strcmp(str, "PPINT") == 0) {
		object_type = POINTER_TYPE_PPINT;
	} else if (strcmp(str, "BYTE") == 0) {
		object_type = POINTER_TYPE_BYTE;
	} else if (strcmp(str, "UBYTE") == 0) {
		object_type = POINTER_TYPE_UBYTE;
	} else if (strcmp(str, "PBYTE") == 0) {
		object_type = POINTER_TYPE_PBYTE;
	} else if (strcmp(str, "PUBYTE") == 0) {
		object_type = POINTER_TYPE_PUBYTE;
	} else if (strcmp(str, "pvoid") == 0) {
		object_type = POINTER_TYPE_PVOID;
	} else if (strcmp(str, "OBJECT") == 0) {
		object_type = POINTER_TYPE_OBJECT;
	} else if (strcmp(str, "OBJECTS") == 0) {
		object_type = POINTER_TYPE_OBJECTS;
	} else {
		object_type = POINTER_TYPE_INVALID;
	}
}

void mem_object_registry_entry::print_type(ostream &ost)
{
	if (object_type == POINTER_TYPE_INVALID) {
		ost << "invalid entry";
		}
	if (object_type == POINTER_TYPE_SMALLINT) {
		ost << "int";
		}
	else if (object_type == POINTER_TYPE_SMALLPINT) {
		ost << "pint";
		}
	else if (object_type == POINTER_TYPE_INT) {
		ost << "INT";
		}
	else if (object_type == POINTER_TYPE_PINT) {
		ost << "PINT";
		}
	else if (object_type == POINTER_TYPE_PPINT) {
		ost << "PPINT";
		}
	else if (object_type == POINTER_TYPE_BYTE) {
		ost << "BYTE";
		}
	else if (object_type == POINTER_TYPE_UBYTE) {
		ost << "UBYTE";
		}
	else if (object_type == POINTER_TYPE_PBYTE) {
		ost << "PBYTE";
		}
	else if (object_type == POINTER_TYPE_PUBYTE) {
		ost << "PUBYTE";
		}
	else if (object_type == POINTER_TYPE_PVOID) {
		ost << "pvoid";
		}
	else if (object_type == POINTER_TYPE_OBJECT) {
		ost << "OBJECT";
		}
	else if (object_type == POINTER_TYPE_OBJECTS) {
		ost << "OBJECTS";
		}
	else {
		ost << "unknown" << endl;
		}
}


int mem_object_registry_entry::size_of()
{
	if (object_type == POINTER_TYPE_INVALID) {
		cout << "mem_object_registry_entry::size_of invalid entry" << endl;
		exit(1);
		}
	if (object_type == POINTER_TYPE_SMALLINT) {
		return sizeof(int) * object_n;
		}
	else if (object_type == POINTER_TYPE_SMALLPINT) {
		return sizeof(int *) * object_n;
		}
	else if (object_type == POINTER_TYPE_INT) {
		return sizeof(INT) * object_n;
		}
	else if (object_type == POINTER_TYPE_PINT) {
		return sizeof(INT *) * object_n;
		}
	else if (object_type == POINTER_TYPE_PPINT) {
		return sizeof(INT **) * object_n;
		}
	else if (object_type == POINTER_TYPE_BYTE) {
		return sizeof(BYTE) * object_n;
		}
	else if (object_type == POINTER_TYPE_UBYTE) {
		return sizeof(UBYTE) * object_n;
		}
	else if (object_type == POINTER_TYPE_PBYTE) {
		return sizeof(BYTE *) * object_n;
		}
	else if (object_type == POINTER_TYPE_PUBYTE) {
		return sizeof(UBYTE *) * object_n;
		}
	else if (object_type == POINTER_TYPE_PVOID) {
		return sizeof(pvoid) * object_n;
		}
	else if (object_type == POINTER_TYPE_OBJECT) {
		return object_size_of;
		}
	else if (object_type == POINTER_TYPE_OBJECTS) {
		return object_n * object_size_of;
		}
	else {
		cout << "mem_object_registry_entry::size_of "
				"unknown object type " << object_type << endl;
		exit(1);
		}
}

void mem_object_registry_entry::print(INT line)
{
	cout << line << " : ";
	print_pointer_hex(cout, pointer);
	cout << " : " << time_stamp << " : ";

	print_type(cout);

	cout << " : "
		<< object_n << " : "
		<< object_size_of << " : "
		<< extra_type_info << " : "
		<< source_file << " : "
		<< source_line << endl;

}


void mem_object_registry_entry::print_csv(ostream &ost, INT line)
{
	ost << line << ",";
	print_pointer_hex(ost, pointer);
	ost << "," << time_stamp << ",";

	print_type(ost);

	ost << ","
		<< object_n << ","
		<< object_size_of << ","
		<< extra_type_info << ","
		<< source_file << ","
		<< source_line << endl;

}








mem_object_registry::mem_object_registry()
{
	INT verbose_level = 1;

	f_automatic_dump = FALSE;
	automatic_dump_interval = 0;
	automatic_dump_fname_mask[0] = 0;

	entries = NULL;
	nb_allocate_total = 0;
	nb_delete_total = 0;
	cur_time = 0;

	f_ignore_duplicates = FALSE;
	f_accumulate = FALSE;

	init(verbose_level);
}

mem_object_registry::~mem_object_registry()
{
	if (entries) {
		delete [] entries;
		entries = NULL;
	}
}

void mem_object_registry::init(INT verbose_level)
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

	f_ignore_duplicates = FALSE;
	f_accumulate = FALSE;

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

void mem_object_registry::accumulate_and_ignore_duplicates(INT verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::accumulate_and_ignore_duplicates" << endl;
	}
	f_accumulate = TRUE;
	f_ignore_duplicates = TRUE;
}

void mem_object_registry::allocate(INT N, INT verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::allocate" << endl;
	}

	nb_entries_allocated = N;
	nb_entries_used = 0;

	if (f_v) {
		cout << "mem_object_registry::allocate trying to allocate "
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
		INT automatic_dump_interval, const BYTE *fname_mask,
		INT verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::set_automatic_dump" << endl;
	}
	f_automatic_dump = TRUE;
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
	BYTE fname[1000];
	INT a;

	a = cur_time / automatic_dump_interval;

	cout << "automatic memory dump " << a << endl;
	sprintf(fname, automatic_dump_fname_mask, a);

	dump_to_csv_file(fname);
}

void mem_object_registry::manual_dump()
{
	if (!f_automatic_dump) {
		return;
	}
	BYTE fname[1000];
	INT a;

	a = cur_time / automatic_dump_interval + 1;

	sprintf(fname, automatic_dump_fname_mask, a);

	dump_to_csv_file(fname);
}

void mem_object_registry::manual_dump_with_file_name(const BYTE *fname)
{
	dump_to_csv_file(fname);
}

void mem_object_registry::dump()
{
	INT i, s, sz;

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
	cout << "total allocation size in BYTE=" << sz << endl;
	cout << "table of all currently active memory allocations in increasing "
			"order of the value of the pointer" << endl;
	for (i = 0; i < nb_entries_used; i++) {
		entries[i].print(i);
	}
}

void mem_object_registry::dump_to_csv_file(const BYTE *fname)
{
	INT i, s, sz;


	{
		ofstream fp(fname);


		//cout << "memory registry:" << endl;

		fp << "Line,Pointer,Timestamp,Type,N,Sizeof,ExtraTypeInfo,File,LineInFile" << endl;
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
		fp << "total allocation size in BYTE=" << sz << endl;
	}
}


int *mem_object_registry::allocate_int(INT n, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::allocate_int int[n], "
				"n=" << n << " file=" << file << " line=" << line << endl;
	}
	int *p;
	p = new int[n];
	if (f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_SMALLINT, (int) n, sizeof(int),
				"", file, line,
				memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_int(int *p, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

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
	if (f_memory_debug) {
		delete_from_registry(p, memory_debug_verbose_level - 1);
	}
	delete [] p;
}

int **mem_object_registry::allocate_pint(INT n, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::allocate_pint pint[n], "
				"n=" << n << " file=" << file << " line=" << line << endl;
	}
	int **p;
	p = new pint[n];
	if (f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_SMALLPINT, (int) n, sizeof(int *),
				"", file, line,
				memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_pint(int **p, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

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
	if (f_memory_debug) {
		delete_from_registry(p, memory_debug_verbose_level - 1);
	}
	delete [] p;
}

INT *mem_object_registry::allocate_INT(INT n, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::allocate_INT INT[n], "
				"n=" << n << " file=" << file << " line=" << line << endl;
	}
	INT *p;
	p = new INT[n];
	if (f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_INT, (int) n, sizeof(INT),
				"", file, line,
				memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_INT(INT *p, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_INT INT[n], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_INT "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (f_memory_debug) {
		delete_from_registry(p, memory_debug_verbose_level - 1);
	}
	delete [] p;
}

INT **mem_object_registry::allocate_PINT(INT n, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::allocate_PINT PINT[n], "
				"n=" << n << " file=" << file << " line=" << line << endl;
	}
	INT **p;
	p = new PINT[n];
	if (f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_PINT, (int) n, sizeof(INT *),
				"", file, line,
				memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_PINT(INT **p, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_PINT PINT[n], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_PINT "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (f_memory_debug) {
		delete_from_registry(p, memory_debug_verbose_level - 1);
	}
	delete [] p;
}

INT ***mem_object_registry::allocate_PPINT(INT n, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::allocate_PPINT PPINT[n], "
				"n=" << n << " file=" << file << " line=" << line << endl;
	}
	INT ***p;
	p = new PPINT[n];
	if (f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_PPINT, (int) n, sizeof(INT **),
				"", file, line,
				memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_PPINT(INT ***p, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_PPINT PPINT[n], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_PPINT "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (f_memory_debug) {
		delete_from_registry(p, memory_debug_verbose_level - 1);
	}
	delete [] p;
}

BYTE *mem_object_registry::allocate_BYTE(INT n, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::allocate_BYTE BYTE[n], "
				"n=" << n << " file=" << file << " line=" << line << endl;
	}
	BYTE *p;
	p = new BYTE[n];
	if (f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_BYTE, (int) n, sizeof(BYTE),
				"", file, line,
				memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_BYTE(BYTE *p, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_BYTE BYTE[n], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_BYTE "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (f_memory_debug) {
		delete_from_registry(p, memory_debug_verbose_level - 1);
	}
	delete [] p;
}

UBYTE *mem_object_registry::allocate_UBYTE(INT n, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::allocate_UBYTE UBYTE[n], "
				"n=" << n << " file=" << file << " line=" << line << endl;
	}
	UBYTE *p;
	p = new UBYTE[n];
	if (f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_UBYTE, (int) n, sizeof(UBYTE),
				"", file, line,
				memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_UBYTE(UBYTE *p, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_UBYTE UBYTE[n], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_UBYTE "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (f_memory_debug) {
		delete_from_registry(p, memory_debug_verbose_level - 1);
	}
	delete [] p;
}

BYTE **mem_object_registry::allocate_PBYTE(INT n, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::allocate_PBYTE PBYTE[n], "
				"n=" << n << " file=" << file << " line=" << line << endl;
	}
	BYTE **p;
	p = new PBYTE[n];
	if (f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_PBYTE, (int) n, sizeof(BYTE *),
				"", file, line,
				memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_PBYTE(BYTE **p, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_PBYTE PBYTE[n], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_PBYTE "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (f_memory_debug) {
		delete_from_registry(p, memory_debug_verbose_level - 1);
	}
	delete [] p;
}

UBYTE **mem_object_registry::allocate_PUBYTE(INT n, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::allocate_PUBYTE PUBYTE[n], "
				"n=" << n << " file=" << file << " line=" << line << endl;
	}
	UBYTE **p;
	p = new PUBYTE[n];
	if (f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_PUBYTE, (int) n, sizeof(BYTE *),
				"", file, line,
				memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_PUBYTE(UBYTE **p, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_PUBYTE PUBYTE[n], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_PUBYTE "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (f_memory_debug) {
		delete_from_registry(p, memory_debug_verbose_level - 1);
	}
	delete [] p;
}

void **mem_object_registry::allocate_pvoid(INT n, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::allocate_pvoid pvoid[n], "
				"n=" << n << " file=" << file << " line=" << line << endl;
	}
	void **p;
	p = new pvoid[n];
	if (f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_PVOID, (int) n, sizeof(void *),
				"", file, line,
				memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_pvoid(void **p, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

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
	if (f_memory_debug) {
		delete_from_registry(p, memory_debug_verbose_level - 1);
	}
	delete [] p;
}

void *mem_object_registry::allocate_OBJECTS(void *p, INT n, INT size_of,
		const char *extra_type_info, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::allocate_OBJECTS BYTE[n * size_of], "
				"n=" << n << " file=" << file << " line=" << line << endl;
	}
	if (f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_OBJECTS, (int) n, size_of,
				extra_type_info, file, line,
				memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_OBJECTS(void *p, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_OBJECTS BYTE[n * size_of], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_OBJECTS "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (f_memory_debug) {
		delete_from_registry(p, memory_debug_verbose_level - 1);
	}
	//delete [] p;
}

void *mem_object_registry::allocate_OBJECT(void *p, INT size_of,
		const char *extra_type_info, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::allocate_OBJECT BYTE[size_of], "
				" file=" << file << " line=" << line << endl;
	}
	if (f_memory_debug) {
		add_to_registry(p /* pointer */,
				POINTER_TYPE_OBJECT, (int) 1, size_of,
				extra_type_info, file, line,
				memory_debug_verbose_level - 1);
		}
	return p;
}

void mem_object_registry::free_OBJECT(void *p, const char *file, int line)
{
	int f_v = (memory_debug_verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::free_OBJECT BYTE[size_of], "
				" file=" << file << " line=" << line << endl;
	}
	if (p == NULL) {
		cout << "mem_object_registry::free_OBJECTS "
				"NULL pointer, ignoring" << endl;
		cout << "p=" << p << " file=" << file
				<< " line=" << line << endl;
		return;
		}
	if (f_memory_debug) {
		delete_from_registry(p, memory_debug_verbose_level - 1);
	}
	//delete [] p;
}





int mem_object_registry::search(void *p, int &idx)
{
	int l, r, m;
	int f_found = FALSE;

	if (nb_entries_used == 0) {
		idx = 0;
		return FALSE;
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
				f_found = TRUE;
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
		int object_type, int object_n, int object_size_of,
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

		} else {
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
	} else {
		if (!search(pointer, idx)) {
			cout << "mem_object_registry::delete_from_registry pointer is "
					"not in registry, something is wrong; ignoring" << endl;
			//exit(1);
		}
		for (i = idx + 1; i < nb_entries_used; i++) {
			entries[i - 1] = entries[i];
			}
		entries[nb_entries_used - 1].null();
		nb_entries_used--;
	}
	automatic_dump();
	cur_time++;
	if (f_v) {
		cout << "mem_object_registry::delete_from_registry done, there are "
				<< nb_entries_used << " entries in the registry" << endl;
	}
}
void mem_object_registry::sort_by_size(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::sort_by_size" << endl;
	}

	if (f_v) {
		cout << "mem_object_registry::sort_by_size before Heapsort" << endl;
	}
	Heapsort(entries, nb_entries_used,
		sizeof(mem_object_registry_entry),
		registry_key_pair_compare_by_size);
	if (f_v) {
		cout << "mem_object_registry::sort_by_size after Heapsort" << endl;
	}

	if (f_v) {
		cout << "mem_object_registry::sort_by_size done" << endl;
	}

}

void mem_object_registry::sort_by_location_and_get_frequency(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "mem_object_registry::sort_by_location_and_get_frequency" << endl;
	}

	sort_by_location(verbose_level - 1);

	INT nb_types;
	INT *type_first;
	INT *type_len;
	INT c, f, l;

	type_first = new INT[nb_entries_used]; // use system memory
	type_len = new INT[nb_entries_used];


	nb_types = 0;
	type_first[0] = 0;
	type_len[0] = 1;
	for (i = 1; i < nb_entries_used; i++) {
		c = registry_key_pair_compare_by_location(entries + i, entries + (i - 1));
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

	INT t, j, sz, s;
	INT *perm;
	INT *perm_inv;
	INT *frequency;

	perm = new INT[nb_types]; // use system memory
	perm_inv = new INT[nb_types];
	frequency = new INT[nb_types];

	INT_vec_copy(type_len, frequency, nb_types);

	INT_vec_sorting_permutation(frequency, nb_types,
			perm, perm_inv, FALSE /* f_increasingly */);

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

	if (f_v) {
		cout << "mem_object_registry::sort_by_location_and_get_frequency" << endl;
	}
}

void mem_object_registry::sort_by_type(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::sort_by_type" << endl;
	}

	if (f_v) {
		cout << "mem_object_registry::sort_by_type before Heapsort" << endl;
	}
	Heapsort(entries, nb_entries_used,
		sizeof(mem_object_registry_entry),
		registry_key_pair_compare_by_type);
	if (f_v) {
		cout << "mem_object_registry::sort_by_type after Heapsort" << endl;
	}

	if (f_v) {
		cout << "mem_object_registry::sort_by_type done" << endl;
	}

}

void mem_object_registry::sort_by_location(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "mem_object_registry::sort_by_location" << endl;
	}

	if (f_v) {
		cout << "mem_object_registry::sort_by_location before Heapsort" << endl;
	}
	Heapsort(entries, nb_entries_used,
		sizeof(mem_object_registry_entry),
		registry_key_pair_compare_by_location);
	if (f_v) {
		cout << "mem_object_registry::sort_by_location after Heapsort" << endl;
	}

	if (f_v) {
		cout << "mem_object_registry::sort_by_location done" << endl;
	}

}

void start_memory_debug()
{
	f_memory_debug = TRUE;
	cout << "memory debugging started" << endl;
}

void stop_memory_debug()
{
	f_memory_debug = FALSE;
	cout << "memory debugging stopped" << endl;
}

static INT registry_key_pair_compare_by_size(void *K1v, void *K2v)
{
	INT s1, s2, c;
	mem_object_registry_entry *K1, *K2;

	K1 = (mem_object_registry_entry *) K1v;
	K2 = (mem_object_registry_entry *) K2v;
	s1 = K1->size_of();
	s2 = K2->size_of();
	c = s2 - s1;
	return c;
}

static INT registry_key_pair_compare_by_type(void *K1v, void *K2v)
{
	INT t1, t2, l1, l2, c;
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

static INT registry_key_pair_compare_by_location(void *K1v, void *K2v)
{
	INT l1, l2, c;
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





#if 0
#define MEMORY_WATCH_LIST_LENGTH 1000 

static int memory_watch_list_length = 0;
static void *memory_watch_list[MEMORY_WATCH_LIST_LENGTH];


void memory_watch_list_add_pointer(void *p)
{
	int i, idx;

	if (memory_watch_list_length >= MEMORY_WATCH_LIST_LENGTH) {
		cout << "memory_watch_list overflow" << endl;
		exit(1);
		}
	if (memory_watch_list_search(memory_watch_list_length, p, idx)) {
		cout << "memory_watch_list_add_pointer pointer "
				<< p << " is already in memory watch list" << endl;
		exit(1);
		}
	for (i = memory_watch_list_length; i > idx; i--) {
		memory_watch_list[i] = memory_watch_list[i - 1];
		}
	memory_watch_list[idx] = p;
	memory_watch_list_length++;
}

void memory_watch_list_delete_pointer(INT idx)
{
	INT i;

	for (i = idx; i < memory_watch_list_length; i++) {
		memory_watch_list[i] = memory_watch_list[i + 1];
		}
	memory_watch_list_length--;
}



void memory_watch_list_dump()
{
	int i, idx;
	void *p;
	
	cout << "memory watch list:" << endl;
	for (i = 0; i < memory_watch_list_length; i++) {
		cout << setw(4) << i << " : ";
		p = memory_watch_list[i];
		print_pointer_hex(cout, p);
		cout << " : ";
		if (!registry_search(registry_size, p, idx)) {
			cout << "did not find pointer " << p << " in registry" << endl;
			}
		else {
			registry_print_entry(idx);
			}
		cout << endl;
		}
}

void registry_dump()
{
	int i;
	INT sz = 0, s;
	
	if (!f_memory_debug)
		return;
	cout << "there are currently " << registry_size
			<< " objects in the registry" << endl;
	cout << "(INT)sizeof(pvoid)=" << (INT)sizeof(pvoid) << endl;
	for (i = 0; i < registry_size; i++) {
		registry_print_entry(i);
		s = registry_entry_size(i);
		sz += s;
		}
	cout << "overall number of objects in the registry: "
			<< registry_size << endl;
	cout << "overall allocation in bytes: " << sz << endl;
	memory_watch_list_dump();
}

typedef struct registry_key_pair registry_key_pair;

//! internal class for memory debugging


struct registry_key_pair {
	const BYTE *file;
	INT line;
	INT idx;
	INT sz;
};

static INT registry_key_pair_compare(
		registry_key_pair *K1, registry_key_pair *K2)
{
	INT c;
	
	c = strcmp(K1->file, K2->file);
	if (c)
		return c;
	c = K1->line - K2->line;
	return c;
}

static INT registry_key_pair_compare_void_void(void *K1v, void *K2v)
{
	INT c;
	registry_key_pair *K1, *K2;

	K1 = (registry_key_pair *) K1v;
	K2 = (registry_key_pair *) K2v;
	c = strcmp(K1->file, K2->file);
	if (c)
		return c;
	c = K1->line - K2->line;
	return c;
}

static INT registry_key_pair_compare_size_void_void(void *K1v, void *K2v)
{
	INT s1, s2, c;
	registry_key_pair *K1, *K2;

	K1 = (registry_key_pair *) K1v;
	K2 = (registry_key_pair *) K2v;
	s1 = K1->sz;
	s2 = K2->sz;
	c = s2 - s1;
	return c;
}

void registry_dump_sorted()
{
	registry_key_pair *K;
	INT i, sz;
	
	if (!f_memory_debug)
		return;
	print_line_of_number_signs();
	cout << "registry_dump_sorted" << endl;
	if (registry_size == 0) {
		cout << "the registry is empty" << endl;
		print_line_of_number_signs();
		return;
		}
	cout << "allocating " << registry_size << " key pairs" << endl;
	K = new registry_key_pair [registry_size];
	cout << "done, now filling the array" << endl;
	sz = 0;
	for (i = 0; i < registry_size; i++) {
		K[i].file = registry_file[i];
		K[i].line = registry_line[i];
		K[i].idx = i;
		K[i].sz = registry_entry_size(i);
		sz += K[i].sz;
		}
	cout << "calling Heapsort" << endl;
	Heapsort(K, registry_size, sizeof(registry_key_pair), 
		registry_key_pair_compare_void_void);
	
	cout << "after Heapsort" << endl;
	
	INT nb_types;
	INT *type_first;
	INT *type_len;
	INT c, idx, f, l;
	
	type_first = new INT[registry_size];
	type_len = new INT[registry_size];
	
	
	nb_types = 0;
	type_first[0] = 0;
	type_len[0] = 1;
	for (i = 1; i < registry_size; i++) {
		c = registry_key_pair_compare(K + i, K + (i - 1));
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
			<< " different allocation types:" << endl;
	//cout << "showing only those with multiplicity at least 5" << endl;
	INT j;
	INT *frequency;
	INT *perm;
	INT *perm_inv;
	
	frequency = new INT[nb_types];
	perm = new INT[nb_types];
	perm_inv = new INT[nb_types];
	for (i = 0; i < nb_types; i++) {
		frequency[i] = type_len[i];
		}
	INT_vec_sorting_permutation(frequency, nb_types,
			perm, perm_inv, FALSE /* f_increasingly */);
	
	for (j = nb_types - 1; j >= 0; j--) {
		i = perm_inv[j];
		
		f = type_first[i];
		l = type_len[i];
		/*if (l < 5)
			break;*/
			
		idx = K[f].idx;
		cout << l << " times " << K[f].file << " line "
				<< K[f].line << " : ";
		registry_print_type(registry_type[idx]);
		cout << endl;		
		}
	cout << "overall number of objects in the registry: "
			<< registry_size << endl;
	cout << "overall allocation in bytes: " << sz << endl;
	print_line_of_number_signs();


	delete [] K;
	delete [] type_first;
	delete [] type_len;
	delete [] frequency;
	delete [] perm;
	delete [] perm_inv;
}

void registry_dump_sorted_by_size()
{
	registry_key_pair *K;
	INT i, sz;
	
	if (!f_memory_debug)
		return;
	print_line_of_number_signs();
	cout << "registry_dump_sorted_by_size" << endl;
	if (registry_size == 0) {
		cout << "the registry is empty" << endl;
		print_line_of_number_signs();
		return;
		}
	cout << "allocating " << registry_size << " key pairs" << endl;
	K = new registry_key_pair [registry_size];
	cout << "done, now filling the array" << endl;
	sz = 0;
	for (i = 0; i < registry_size; i++) {
		K[i].file = registry_file[i];
		K[i].line = registry_line[i];
		K[i].idx = i;
		K[i].sz = registry_entry_size(i);
		sz += K[i].sz;
		}
	cout << "calling Heapsort" << endl;
	Heapsort(K, registry_size, sizeof(registry_key_pair), 
		registry_key_pair_compare_size_void_void);
	
	cout << "after Heapsort" << endl;

	cout << "showing objects by size" << endl;
	for (i = 0; i < registry_size; i++) {
		/*if (K[i].sz < 1024)
			continue;*/
		cout << K[i].file << " line " << K[i].line << " : ";
		registry_print_type(registry_type[K[i].idx]);
		cout << " of size " << K[i].sz << endl;		
		}

	cout << "overall number of objects in the registry: "
			<< registry_size << endl;
	cout << "overall allocation in bytes: " << sz << endl;

	delete [] K;
}

INT registry_entry_size(INT i)
{
	INT sz;
	
	if (registry_type[i] == POINTER_TYPE_OBJECT) {
		sz = registry_size_of[i] * registry_n[i];
		}
	else {
		sz = registry_sizeof(registry_type[i]) * registry_n[i];
		}
	return sz;
}

void registry_print_entry(INT i)
{
	cout << i << " : ";
	print_pointer_hex(cout, registry_pointer[i]);
	cout << " : ";
	
	registry_print_type(registry_type[i]);
	
	cout << " : " 
		<< registry_n[i] << " : " 
		<< registry_size_of[i] << " : " 
		<< registry_file[i] << " : " 
		<< registry_line[i] << " : " 
		<< endl;
}

void registry_print_type(INT t)
{
	if (t == POINTER_TYPE_SMALLINT) {
		cout << "int";
		}
	else if (t == POINTER_TYPE_SMALLPINT) {
		cout << "int*";
		}
	else if (t == POINTER_TYPE_INT) {
		cout << "INT";
		}
	else if (t == POINTER_TYPE_PINT) {
		cout << "INT*";
		}
	else if (t == POINTER_TYPE_BYTE) {
		cout << "BYTE";
		}
	else if (t == POINTER_TYPE_PBYTE) {
		cout << "BYTE*";
		}
	else if (t == POINTER_TYPE_PVOID) {
		cout << "void*";
		}
	else if (t == POINTER_TYPE_OBJECT) {
		cout << "OBJECT";
		}
	else if (t == POINTER_TYPE_OBJECTS) {
		cout << "OBJECTS";
		}
}

int memory_watch_list_search(int len, void *p, int &idx)
{
	int l, r, m;
	//void *res;
	int f_found = FALSE;
	
	if (len == 0) {
		idx = 0;
		return FALSE;
		}
	l = 0;
	r = len;
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
		if (p >= memory_watch_list[m]) {
			l = m + 1;
			if (p == memory_watch_list[m])
				f_found = TRUE;
			}
		else
			r = m;
		}
	// now: l == r; 
	// and f_found is set accordingly */
	if (f_found)
		l--;
	idx = l;
	return f_found;
}


#endif


