// memory.C
//
// Anton Betten
//
// started:  June 25, 2009




#include "foundations.h"

#define REGISTRY_SIZE (10 * ONE_MILLION)
#define POINTER_TYPE_SMALLINT 1
#define POINTER_TYPE_SMALLPINT 2
#define POINTER_TYPE_INT 3
#define POINTER_TYPE_PINT 4
#define POINTER_TYPE_PPINT 5
#define POINTER_TYPE_BYTE 6
#define POINTER_TYPE_UBYTE 7
#define POINTER_TYPE_PBYTE 8
#define POINTER_TYPE_PVOID 9
#define POINTER_TYPE_OBJECT 10
#define POINTER_TYPE_OBJECTS 11

int f_memory_debug = FALSE;
int f_memory_debug_verbose = FALSE;
INT memory_count_allocate = 0;
int registry_size = 0;
void *registry_pointer[REGISTRY_SIZE];
int registry_type[REGISTRY_SIZE];
int registry_n[REGISTRY_SIZE];
int registry_size_of[REGISTRY_SIZE]; // needed for objects of type class
const char *registry_file[REGISTRY_SIZE];
int registry_line[REGISTRY_SIZE];

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
		cout << "memory_watch_list_add_pointer pointer " << p << " is already in memory watch list" << endl;
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



void add_to_registry(void *p, int pointer_type, int size, int size_of, const char *file, int line)
{
	int idx, i;
	
	//memory_count_allocate++;
	if (registry_size == REGISTRY_SIZE) {
		cout << "add_to_registry registry is full" << endl;
		return;
		}
	if (registry_search(registry_size, p, idx)) {
		cout << "add_to_registry pointer " << p << " is already in registry" << endl;
		registry_print_entry(idx);
		exit(1);
		}
	for (i = registry_size; i > idx; i--) {
		registry_pointer[i] = registry_pointer[i - 1];
		registry_type[i] = registry_type[i - 1];
		registry_n[i] = registry_n[i - 1];
		registry_size_of[i] = registry_size_of[i - 1];
		registry_file[i] = registry_file[i - 1];
		registry_line[i] = registry_line[i - 1];
		}
	registry_size++;
	registry_pointer[idx] = p;
	registry_type[idx] = pointer_type;
	registry_n[idx] = size;
	registry_size_of[idx] = size_of;
	registry_file[idx] = file;
	registry_line[idx] = line;

	if (f_memory_debug_verbose) {
		cout << "ALLOCATE " << file << " " << line << endl;
		}
}

INT delete_from_registry(void *p)
{
	int idx, i;
	
	if (registry_size == REGISTRY_SIZE) {
		cout << "delete_from_registry registry is full" << endl;
		return TRUE;
		}
	if (!registry_search(registry_size, p, idx)) {
		cout << "did not find pointer " << p << " in registry" << endl;
		return FALSE;
		}
	if (f_memory_debug_verbose) {
		cout << "DELETE " << registry_file[idx] << " " << registry_line[idx] << endl;
		}
	for (i = idx + 1; i < registry_size; i++) {
		registry_pointer[i - 1] = registry_pointer[i];
		registry_type[i - 1] = registry_type[i];
		registry_n[i - 1] = registry_n[i];
		registry_size_of[i - 1] = registry_size_of[i];
		registry_file[i - 1] = registry_file[i];
		registry_line[i - 1] = registry_line[i];
		}
	registry_size--;
	return TRUE;
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
	cout << "there are currently " << registry_size << " objects in the registry" << endl;
	cout << "(INT)sizeof(pvoid)=" << (INT)sizeof(pvoid) << endl;
	for (i = 0; i < registry_size; i++) {
		registry_print_entry(i);
		s = registry_entry_size(i);
		sz += s;
		}
	cout << "overall number of objects in the registry: " << registry_size << endl;
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

static INT registry_key_pair_compare(registry_key_pair *K1, registry_key_pair *K2)
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
			type_first[nb_types + 1] = type_first[nb_types] + type_len[nb_types];
			nb_types++;
			type_len[nb_types] = 1;
			}
		}
	nb_types++;
	cout << "we have " << nb_types << " different allocation types:" << endl;
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
	INT_vec_sorting_permutation(frequency, nb_types, perm, perm_inv, FALSE /* f_increasingly */);
	
	for (j = nb_types - 1; j >= 0; j--) {
		i = perm_inv[j];
		
		f = type_first[i];
		l = type_len[i];
		/*if (l < 5)
			break;*/
			
		idx = K[f].idx;
		cout << l << " times " << K[f].file << " line " << K[f].line << " : ";
		registry_print_type(registry_type[idx]);
		cout << endl;		
		}
	cout << "overall number of objects in the registry: " << registry_size << endl;
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

	cout << "overall number of objects in the registry: " << registry_size << endl;
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

INT registry_sizeof(INT t)
{
	if (t == POINTER_TYPE_SMALLINT) {
		return sizeof(int);
		}
	else if (t == POINTER_TYPE_SMALLPINT) {
		return sizeof(int *);
		}
	else if (t == POINTER_TYPE_INT) {
		return sizeof(INT);
		}
	else if (t == POINTER_TYPE_PINT) {
		return sizeof(INT *);
		}
	else if (t == POINTER_TYPE_PPINT) {
		return sizeof(INT **);
		}
	else if (t == POINTER_TYPE_BYTE) {
		return sizeof(BYTE);
		}
	else if (t == POINTER_TYPE_UBYTE) {
		return sizeof(UBYTE);
		}
	else if (t == POINTER_TYPE_PBYTE) {
		return sizeof(BYTE *);
		}
	else if (t == POINTER_TYPE_PVOID) {
		return sizeof(pvoid);
		}
	else if (t == POINTER_TYPE_OBJECT) {
		cout << "registry_sizeof() sizeof type OBJECT unknown" << endl;
		exit(1);
		}
	else if (t == POINTER_TYPE_OBJECTS) {
		cout << "registry_sizeof() sizeof type OBJECTS unknown" << endl;
		exit(1);
		}
	else {
		cout << "registry_sizeof() unknown type " << t << endl;
		exit(1);
		}
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

int registry_search(int len, void *p, int &idx)
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
		if (p >= registry_pointer[m]) {
			l = m + 1;
			if (p == registry_pointer[m])
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


int *allocate_int(INT n, const char *file, int line)
{
	int *p;

	memory_count_allocate++;


	p = new int[n];
	if (f_memory_debug) {
		//cout << "allocate_int n=" << n << " p=" << p << " file=" << file << " line=" << line << endl;
		add_to_registry(p, POINTER_TYPE_SMALLINT, (int) n, sizeof(int), file, line);
		}
	return p;
}

void free_int(int *p, const char *file, int line)
{	
	int idx;
	
	if (p == NULL) {
		cout << "free_int NULL pointer, ignoring" << endl;
		cout << "free_int p=" << p << " file=" << file << " line=" << line << endl;
		return;
		}
	if (f_memory_debug) {
		if (memory_watch_list_search(memory_watch_list_length, p, idx)) {
			cout << "free_int: watched pointer ";
			print_pointer_hex(cout, p);
			cout << " is being freed" << endl;
			memory_watch_list_delete_pointer(idx);
			}
		//cout << "free_int p=" << p << " file=" << file << " line=" << line << endl;
		if (!delete_from_registry(p)) {
			cout << "free_int p=" << p << " file=" << file << " line=" << line << endl;
			exit(1);
			}
		}
	delete [] p;
}

pint *allocate_pint(INT n, const char *file, int line)
{
	pint *p;


	memory_count_allocate++;


	p = new pint[n];
	if (f_memory_debug) {
		//cout << "allocate_pint n=" << n << " p=" << p << " file=" << file << " line=" << line << endl;
		add_to_registry(p, POINTER_TYPE_SMALLPINT, (int) n, sizeof(pint), file, line);
		}
	return p;
}

void free_pint(pint *p, const char *file, int line)
{
	int idx;
	
	if (p == NULL) {
		cout << "free_pint NULL pointer, ignoring" << endl;
		cout << "free_pint p=" << p << " file=" << file << " line=" << line << endl;
		return;
		}
	if (f_memory_debug) {
		if (memory_watch_list_search(memory_watch_list_length, p, idx)) {
			cout << "free_pint: watched pointer ";
			print_pointer_hex(cout, p);
			cout << " is being freed" << endl;
			memory_watch_list_delete_pointer(idx);
			}
		//cout << "free_pint p=" << p << " file=" << file << " line=" << line << endl;
		if (!delete_from_registry(p)) {
			cout << "free_pint p=" << p << " file=" << file << " line=" << line << endl;
			exit(1);
			}
		}
	delete [] p;
}

INT *allocate_INT(INT n, const char *file, int line)
{
	INT *p;

	memory_count_allocate++;


	p = new INT[n];
	if (f_memory_debug) {
		//cout << "allocate_INT n=" << n << " p=" << p << " file=" << file << " line=" << line << endl;
		add_to_registry(p, POINTER_TYPE_INT, (int) n, sizeof(INT), file, line);
		}
	return p;
}

void free_INT(INT *p, const char *file, int line)
{	
	int idx;
	
	if (p == NULL) {
		cout << "free_INT NULL pointer, ignoring" << endl;
		cout << "free_INT p=" << p << " file=" << file << " line=" << line << endl;
		return;
		}
	if (f_memory_debug) {
		if (memory_watch_list_search(memory_watch_list_length, p, idx)) {
			cout << "free_INT: watched pointer ";
			print_pointer_hex(cout, p);
			cout << " is being freed" << endl;
			memory_watch_list_delete_pointer(idx);
			}
		//cout << "free_INT p=" << p << " file=" << file << " line=" << line << endl;
		if (!delete_from_registry(p)) {
			cout << "free_INT p=" << p << " file=" << file << " line=" << line << endl;
			exit(1);
			}
		}
	delete [] p;
}

INT **allocate_PINT(INT n, const char *file, int line)
{
	INT **p;


	memory_count_allocate++;


	p = new PINT[n];
	if (f_memory_debug) {
		//cout << "allocate_INT n=" << n << " p=" << p << " file=" << file << " line=" << line << endl;
		add_to_registry(p, POINTER_TYPE_PINT, (int) n, sizeof(PINT), file, line);
		}
	return p;
}

void free_PINT(INT **p, const char *file, int line)
{	
	int idx;
	
	if (p == NULL) {
		cout << "free_PINT NULL pointer, ignoring" << endl;
		cout << "free_PINT p=" << p << " file=" << file << " line=" << line << endl;
		return;
		}
	if (f_memory_debug) {
		if (memory_watch_list_search(memory_watch_list_length, p, idx)) {
			cout << "free_PINT: watched pointer ";
			print_pointer_hex(cout, p);
			cout << " is being freed" << endl;
			memory_watch_list_delete_pointer(idx);
			}
		//cout << "free_INT p=" << p << " file=" << file << " line=" << line << endl;
		if (!delete_from_registry(p)) {
			cout << "free_PINT p=" << p << " file=" << file << " line=" << line << endl;
			exit(1);
			}
		}
	delete [] p;
}

INT ***allocate_PPINT(INT n, const char *file, int line)
{
	INT ***p;


	memory_count_allocate++;


	p = new PPINT[n];
	if (f_memory_debug) {
		//cout << "allocate_PPINT n=" << n << " p=" << p << " file=" << file << " line=" << line << endl;
		add_to_registry(p, POINTER_TYPE_PPINT, (int) n, sizeof(PPINT), file, line);
		}
	return p;
}

void free_PPINT(INT ***p, const char *file, int line)
{	
	int idx;
	
	if (p == NULL) {
		cout << "free_PPINT NULL pointer, ignoring" << endl;
		cout << "free_PPINT p=" << p << " file=" << file << " line=" << line << endl;
		return;
		}
	if (f_memory_debug) {
		if (memory_watch_list_search(memory_watch_list_length, p, idx)) {
			cout << "free_PPINT: watched pointer ";
			print_pointer_hex(cout, p);
			cout << " is being freed" << endl;
			memory_watch_list_delete_pointer(idx);
			}
		//cout << "free_PPINT p=" << p << " file=" << file << " line=" << line << endl;
		if (!delete_from_registry(p)) {
			cout << "free_PINT p=" << p << " file=" << file << " line=" << line << endl;
			exit(1);
			}
		}
	delete [] p;
}


BYTE *allocate_BYTE(INT n, const char *file, int line)
{
	BYTE *p;


	memory_count_allocate++;


	p = new BYTE[n];
	if (f_memory_debug) {
		//cout << "allocate_BYTE n=" << n << " p=" << p << " file=" << file << " line=" << line << endl;
		add_to_registry(p, POINTER_TYPE_BYTE, (int) n, sizeof(BYTE), file, line);
		}
	return p;
}

void free_BYTE(BYTE *p, const char *file, int line)
{	
	int idx;
	
	if (p == NULL) {
		cout << "free_BYTE NULL pointer, ignoring" << endl;
		cout << "free_BYTE p=" << p << " file=" << file << " line=" << line << endl;
		return;
		}
	if (f_memory_debug) {
		if (memory_watch_list_search(memory_watch_list_length, p, idx)) {
			cout << "free_BYTE: watched pointer ";
			print_pointer_hex(cout, p);
			cout << " is being freed" << endl;
			memory_watch_list_delete_pointer(idx);
			}
		//cout << "free_BYTE p=" << p << " file=" << file << " line=" << line << endl;
		if (!delete_from_registry(p)) {
			cout << "free_BYTE p=" << p << " file=" << file << " line=" << line << endl;
			exit(1);
			}
		}
	delete [] p;
}

UBYTE *allocate_UBYTE(INT n, const char *file, int line)
{
	UBYTE *p;


	memory_count_allocate++;


	p = new UBYTE[n];
	if (f_memory_debug) {
		//cout << "allocate_UBYTE n=" << n << " p=" << p << " file=" << file << " line=" << line << endl;
		add_to_registry(p, POINTER_TYPE_UBYTE, (int) n, sizeof(UBYTE), file, line);
		}
	return p;
}

void free_UBYTE(UBYTE *p, const char *file, int line)
{	
	int idx;
	
	if (p == NULL) {
		cout << "free_UBYTE NULL pointer, ignoring" << endl;
		cout << "free_UBYTE p=" << p << " file=" << file << " line=" << line << endl;
		return;
		}
	if (f_memory_debug) {
		if (memory_watch_list_search(memory_watch_list_length, p, idx)) {
			cout << "free_BYTE: watched pointer ";
			print_pointer_hex(cout, p);
			cout << " is being freed" << endl;
			memory_watch_list_delete_pointer(idx);
			}
		//cout << "free_UBYTE p=" << p << " file=" << file << " line=" << line << endl;
		if (!delete_from_registry(p)) {
			cout << "free_UBYTE p=" << p << " file=" << file << " line=" << line << endl;
			exit(1);
			}
		}
	delete [] p;
}

BYTE **allocate_PBYTE(INT n, const char *file, int line)
{
	BYTE **p;


	memory_count_allocate++;


	p = new PBYTE[n];
	if (f_memory_debug) {
		//cout << "allocate_BYTE n=" << n << " p=" << p << " file=" << file << " line=" << line << endl;
		add_to_registry(p, POINTER_TYPE_PBYTE, (int) n, sizeof(PBYTE), file, line);
		}
	return p;
}

void free_PBYTE(BYTE **p, const char *file, int line)
{	
	int idx;
	
	if (p == NULL) {
		cout << "free_PBYTE NULL pointer, ignoring" << endl;
		cout << "free_PBYTE p=" << p << " file=" << file << " line=" << line << endl;
		return;
		}
	if (f_memory_debug) {
		if (memory_watch_list_search(memory_watch_list_length, p, idx)) {
			cout << "free_PBYTE: watched pointer ";
			print_pointer_hex(cout, p);
			cout << " is being freed" << endl;
			memory_watch_list_delete_pointer(idx);
			}
		//cout << "free_PBYTE p=" << p << " file=" << file << " line=" << line << endl;
		if (!delete_from_registry(p)) {
			cout << "free_PBYTE p=" << p << " file=" << file << " line=" << line << endl;
			exit(1);
			}
		}
	delete [] p;
}

void **allocate_pvoid(INT n, const char *file, int line)
{
	void **p;


	memory_count_allocate++;


	p = new pvoid[n];
	if (f_memory_debug) {
		//cout << "allocate_BYTE n=" << n << " p=" << p << " file=" << file << " line=" << line << endl;
		add_to_registry(p, POINTER_TYPE_PVOID, (int) n, sizeof(pvoid), file, line);
		}
	return p;
}

void free_pvoid(void **p, const char *file, int line)
{	
	int idx;
	
	if (p == NULL) {
		cout << "free_pvoid NULL pointer, ignoring" << endl;
		cout << "free_pvoid p=" << p << " file=" << file << " line=" << line << endl;
		return;
		}
	if (f_memory_debug) {
		if (memory_watch_list_search(memory_watch_list_length, p, idx)) {
			cout << "free_pvoid: watched pointer ";
			print_pointer_hex(cout, p);
			cout << " is being freed" << endl;
			memory_watch_list_delete_pointer(idx);
			}
		//cout << "free_PBYTE p=" << p << " file=" << file << " line=" << line << endl;
		if (!delete_from_registry(p)) {
			cout << "free_pvoid p=" << p << " file=" << file << " line=" << line << endl;
			exit(1);
			}
		}
	delete [] p;
}

void *allocate_OBJECT(void *p, int size_of, const char *file, int line)
{
	memory_count_allocate++;


	if (f_memory_debug) {
		//cout << "allocate_CLASS n=" << n << " p=" << p << " file=" << file << " line=" << line << endl;
		add_to_registry(p, POINTER_TYPE_OBJECT, 1, size_of, file, line);
		}
	return p;
}

void free_OBJECT(void *p, const char *file, int line)
{	
	int idx;
	
	if (p == NULL) {
		cout << "free_OBJECT NULL pointer, ignoring" << endl;
		cout << "free_OBJECT p=" << p << " file=" << file << " line=" << line << endl;
		return;
		}
	if (f_memory_debug) {
		if (memory_watch_list_search(memory_watch_list_length, p, idx)) {
			cout << "free_OBJECT: watched pointer ";
			print_pointer_hex(cout, p);
			cout << " is being freed" << endl;
			memory_watch_list_delete_pointer(idx);
			}
		//cout << "free_OBJECT p=" << p << " file=" << file << " line=" << line << endl;
		if (!delete_from_registry(p)) {
			cout << "free_OBJECT p=" << p << " file=" << file << " line=" << line << endl;
			//exit(1);
			}
		}
}

void *allocate_OBJECTS(void *p, INT n, int size_of, const char *file, int line)
{
	memory_count_allocate++;


	if (f_memory_debug) {
		//cout << "allocate_CLASS n=" << n << " p=" << p << " file=" << file << " line=" << line << endl;
		add_to_registry(p, POINTER_TYPE_OBJECTS, (int) n, size_of, file, line);
		}
	return p;
}

void free_OBJECTS(void *p, const char *file, int line)
{
	int idx;
	
	if (p == NULL) {
		cout << "free_OBJECTS NULL pointer, ignoring" << endl;
		cout << "free_OBJECTS p=" << p << " file=" << file << " line=" << line << endl;
		return;
		}
	if (f_memory_debug) {
		if (memory_watch_list_search(memory_watch_list_length, p, idx)) {
			cout << "free_OBJECTS: watched pointer ";
			print_pointer_hex(cout, p);
			cout << " is being freed" << endl;
			memory_watch_list_delete_pointer(idx);
			}
		//cout << "free_OBJECTS p=" << p << " file=" << file << " line=" << line << endl;
		if (!delete_from_registry(p)) {
			cout << "free_OBJECTS p=" << p << " file=" << file << " line=" << line << endl;
			//exit(1);
			}
		}
}

void free_PINT_all(INT **p, INT nb)
{
	INT i;
	
	for (i = 0; i < nb; i++) {
		FREE_INT(p[i]);
		}
	FREE_PINT(p);
}


