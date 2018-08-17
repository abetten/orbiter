// page_table.C
//
// Anton Betten
// 7/31/09

#include "orbiter.h"


#include <stdlib.h> // for system


#define MAX_BTREE_PAGE_TABLE 100


static page_table **btree_page_table = NULL;
static INT *btree_page_table_f_used = NULL;





static INT btree_page_registry_key_pair_compare_void_void(void *K1v, void *K2v);

// ##########################################################################################################
// global function
// ##########################################################################################################

void page_table_init(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "page_table_init" << endl;
		}
	btree_page_table = new ppage_table[MAX_BTREE_PAGE_TABLE];
	btree_page_table_f_used = new INT[MAX_BTREE_PAGE_TABLE];
	
	for (i = 0; i < MAX_BTREE_PAGE_TABLE; i++) {
		btree_page_table[i] = NULL;
		btree_page_table_f_used[i] = FALSE;
		}
	
}

void page_table_exit(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "page_table_exit" << endl;
		}
	if (btree_page_table) {
		delete [] btree_page_table;
		btree_page_table = NULL;
		}
	if (btree_page_table_f_used) {
		delete [] btree_page_table_f_used;
		btree_page_table_f_used = NULL;
		}
	
}

INT page_table_alloc(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i;
	
	if (f_v) {
		cout << "page_table_alloc, verbose_level=" << verbose_level << endl;
		}
	for (i = 0; i < MAX_BTREE_PAGE_TABLE; i++) {
		if (!btree_page_table_f_used[i]) {
			if (f_vv) {
				cout << "page_table_alloc allocating slot " << i << endl;
				}
			btree_page_table[i] = new page_table;
			
			btree_page_table[i]->init(verbose_level - 1);
			
			btree_page_table_f_used[i] = TRUE;
			if (f_vv) {
				cout << " slot " << i << " is " << btree_page_table[i] << endl;
				}
			return i;
			}
		}
	cout << "page_table_alloc() no free btree_data_table" << endl;
	exit(1);
}

void page_table_free(INT idx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "page_table_free freeing slot " << idx << endl;
		}
	if (!btree_page_table_f_used[idx]) {
		cout << "page_table_free slot " << idx << " is not used" << endl;
		exit(1);
		}
	if (btree_page_table[idx] == NULL) {
		cout << "page_table_free btree_data_table[idx] == NULL" << endl;
		exit(1);
		}
	
	if (f_v) {
		cout << "before delete btree_page_table[idx]" << endl;
		}
	delete btree_page_table[idx];
	if (f_v) {
		cout << "after delete btree_page_table[idx]" << endl;
		}
	
	btree_page_table[idx] = NULL;
	btree_page_table_f_used[idx] = FALSE;
	if (f_v) {
		cout << "page_table_free done" << endl;
		}
}

page_table *page_table_pointer(INT slot)
{
	page_table *T;
	
	if (slot < 0 || slot >= MAX_BTREE_PAGE_TABLE) {
		cout << "page_table_pointer illegal slot" << endl;
		cout << "slot=" << slot << endl;
		exit(1);
		}
	T = btree_page_table[slot];
	if (T == NULL) {
		cout << "page_table_pointer T == NULL" << endl;
		exit(1);
		}
	return T;
}


static INT btree_page_registry_key_pair_compare_void_void(void *K1v, void *K2v)
{
	btree_page_registry_key_pair *K1, *K2;

	K1 = (btree_page_registry_key_pair *) K1v;
	K2 = (btree_page_registry_key_pair *) K2v;
	if (K1->x < K2->x)
		return 1;
	if (K1->x > K2->x)
		return -1;
	if (K1->idx < K2->idx)
		return 1;
	if (K1->idx > K2->idx)
		return -1;
	return 0;
}


// ##########################################################################################################
// class page_table
// ##########################################################################################################

page_table::page_table()
{
	btree_pages = NULL;
	btree_page_registry_length = 0;
	btree_page_registry_allocated_length = 0;
	btree_table = NULL;
}

page_table::~page_table()
{
	//cout << "page_table::~page_table" << endl;
	//cout << "this=" << this << endl;
	if (btree_pages) {
		//cout << "calling delete btree_pages" << endl;
		delete btree_pages;
		//cout << "after delete btree_pages" << endl;
		btree_pages = NULL;
		}
	if (btree_table) {
		delete [] btree_table;
		btree_table = NULL;
		}
	btree_page_registry_length = 0;
	btree_page_registry_allocated_length = 0;
	//cout << "page_table::~page_table done" << endl;
}

void page_table::init(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 1);
	
	if (f_v) {
		cout << "page_table::init, verbose_level=" << verbose_level << endl;
		}
	btree_pages = new page_storage;
	
	INT page_length_log = BTREE_PAGE_LENGTH_LOG;
	
	btree_pages->init(
		sizeof(PageTyp), 
		page_length_log, 
		verbose_level - 2);
	
	btree_page_registry_allocated_length = 1000;
	btree_page_registry_length = 0;
	if (f_vv) {
		cout << "btree_page_registry_allocated_length = " << btree_page_registry_allocated_length << endl;
		}
	
	btree_table = new btree_page_registry_key_pair[btree_page_registry_allocated_length];
}

void page_table::reallocate_table(INT verbose_level)
{
	INT new_length, i;
	
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 1);
	
	if (f_vv) {
		cout << "page_table::reallocate_table" << endl;
		}

	btree_page_registry_key_pair *new_btree_table;
	
	new_length = btree_page_registry_allocated_length + 1000;


	new_btree_table = new btree_page_registry_key_pair[new_length];
	for (i = 0; i < btree_page_registry_length; i++) {
		new_btree_table[i] = btree_table[i];
		}
	delete [] btree_table;
	btree_table = new_btree_table;
	btree_page_registry_allocated_length = new_length;
	if (f_v) {
		cout << "page_table::reallocate_table: btree_page_registry_allocated_length = " << btree_page_registry_allocated_length << endl;
		}
	
}

void page_table::print()
{
	INT i;
	
	cout << "page registry of length " << btree_page_registry_length << endl;
	cout << "i : x : idx : ref" << endl;
	for (i = 0; i < btree_page_registry_length; i++) {
		cout << i << " " 
			<< btree_table[i].x << " " 
			<< btree_table[i].idx << " " 
			<< btree_table[i].ref << endl;
		}
}

INT page_table::search(INT len, INT btree_idx, INT btree_x, INT &idx)
{
	btree_page_registry_key_pair K;
	
	K.idx = btree_idx;
	K.x = btree_x;
	return page_table::search_key_pair(len, &K, idx);
}

INT page_table::search_key_pair(INT len, btree_page_registry_key_pair *K, INT &idx)
{
	INT l, r, m, c;
	//void *res;
	INT f_found = FALSE;
	
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
		c = btree_page_registry_key_pair_compare_void_void(K, btree_table + m);
		if (c <= 0) {
			l = m + 1;
			if (c == 0) {
				f_found = TRUE;
				}
			}
		else {
			r = m;
			}
		}
	// now: l == r; 
	// and f_found is set accordingly */
	if (f_found) {
		l--;
		}
	idx = l;
	return f_found;
}

void page_table::save_page(Buffer *BF, INT buf_idx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT x, idx, ref;
	PageTyp *P;
	
	if (f_v) {
		cout << "page_table::save_page" << endl;
		}
	x = BF->PageNum;
	if (!search(btree_page_registry_length, 
		buf_idx, x, idx)) {
		if (f_v) {
			cout << "making a new table entry" << endl;
			}
		//print();
		//exit(1);
		
		Buffer *empty_buf;
		empty_buf = new Buffer;
		
		allocate_rec(empty_buf, buf_idx, x, verbose_level - 1);
		
		delete empty_buf;
		
		if (!search(btree_page_registry_length, 
			buf_idx, x, idx)) {
			cout << "page_table::save_page cannot find the page in the table (2nd time)" << endl;
			cout << "x=" << x << endl; 
			cout << "buf_idx=" << buf_idx << endl;
			print();
			exit(1);
			}
		}
	ref = btree_table[idx].ref;
	P = (PageTyp *) btree_pages->s_i(ref);
	*P = BF->Page;
}

INT page_table::load_page(Buffer *BF, INT x, INT buf_idx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT idx, ref;
	PageTyp *P;
	
	if (f_v) {
		cout << "page_table::load_page x=" << x << endl;
		}
	if (!search(btree_page_registry_length, 
		buf_idx, x, idx)) {
		if (f_v) {
			cout << "page_table::load_page cannot find the page in the table" << endl;
			cout << "x=" << x << endl; 
			cout << "buf_idx=" << buf_idx << endl;
			}
		//print();
		//exit(1);
		return FALSE;
		}
	ref = btree_table[idx].ref;
	P = (PageTyp *) btree_pages->s_i(ref);
	BF->Page = *P;
	return TRUE;
}

void page_table::allocate_rec(Buffer *BF, INT buf_idx, INT x, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT ref;
	INT idx, i;

	if (f_v) {
		cout << "page_table::allocate_rec x=" << x << endl;
		}
	if (btree_page_registry_length == btree_page_registry_allocated_length) {
		reallocate_table(verbose_level - 1);
		}

	fill_char((BYTE *)BF, sizeof(Buffer), 0);
	ref = btree_pages->store((UBYTE *) &BF->Page);

	if (search(btree_page_registry_length, buf_idx, x, idx)) {
		cout << "page_table::allocate_rec error, found entry in the table" << endl;
		cout << "page_table::allocate_rec btree_table: at " << idx << " x=" << x << " buf_idx = " << buf_idx << " ref=" << ref << endl;
		print();
		exit(1);
		}
	for (i = btree_page_registry_length; i > idx; i--) {
		btree_table[i] = btree_table[i - 1];
		}
	btree_table[idx].x = x;
	btree_table[idx].idx = buf_idx;
	btree_table[idx].ref = ref;
	//cout << "btree::AllocateRec() btree_table: at " << idx << " x=" << x << " idx = " << buf_idx() << " ref=" << ref << endl;
	btree_page_registry_length++;
	/*if ((btree_page_registry_length % 10) == 0) {
		btree_page_registry_print();
		}*/
}

void page_table::write_pages_to_file(btree *B, INT buf_idx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, ref, x;
	PageTyp *P;

	if (f_v) {
		cout << "page_table::write_pages_to_file" << endl;
		}
	
	for (i = 0; i < btree_page_registry_length; i++) {
		if (btree_table[i].idx == buf_idx) {
			ref = btree_table[i].ref;
			x = btree_table[i].x;
			if (f_v) {
				cout << "writing page " << x << " (ref=" << ref << ")" << endl;
				}
			P = (PageTyp *) btree_pages->s_i(ref);
			B->file_seek(x);
			B->file_write(P, "page_table::write_pages_to_file");
			}
		}
}
