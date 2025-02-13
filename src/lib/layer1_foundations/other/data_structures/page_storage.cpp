// page_storage.cpp
//
// Anton Betten
// October 23, 2002

#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace data_structures {


page_storage::page_storage()
{
	Record_birth();
	overall_length = 0;
	entry_size = 0;
	page_length_log = 0;
	page_length = 0;
	page_size = 0;
	allocation_table_length = 0;

	page_ptr_used = 0;
	page_ptr_allocated = 0;
	page_ptr_oversize = 0;

	pages = NULL;
	allocation_tables = NULL;

	next_free_entry = 0;
	nb_free_entries = 0;

	f_elt_print_function = false;
	elt_print = NULL;
	elt_print_data = NULL;
}

page_storage::~page_storage()
{
	Record_death();
	long int i;
	
	//cout << "page_storage::~page_storage" << endl;
	if (pages) {
		//cout << "page_ptr_used=" << page_ptr_used << endl;
		for (i = 0; i < page_ptr_used; i++) {
			//cout << "deleting page" << i << endl;
			FREE_uchar(pages[i]);
			FREE_uchar(allocation_tables[i]);
			//cout << "deleting page" << i << " done" << endl;
			}
		//cout << "deleting pages" << endl;
		FREE_puchar(pages);
		//cout << "deleting allocation_tables" << endl;
		FREE_puchar(allocation_tables);
		pages = NULL;
		allocation_tables = NULL;
		}
	//cout << "page_storage::~page_storage done" << endl;
}

void page_storage::init(
		int entry_size,
		int page_length_log, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i;
	
	if (f_v) {
		cout << "page_storage::init "
				"verbose_level=" << verbose_level << endl;
		}
	f_elt_print_function = false;
	
	overall_length = 0;
	next_free_entry = -1;
	nb_free_entries = 0;
	
	//f_v = true;
	
	page_storage::entry_size = entry_size;
	if (entry_size < (int)sizeof(int)) {
		page_storage::entry_size = sizeof(int);
		if (f_v) {
			cout << "warning: raising entry_size to sizeof(int) = "
					<< page_storage::entry_size << endl;
			}
		}

	page_storage::page_length_log = page_length_log;
	while (true) {
		page_length = 1L << page_storage::page_length_log;
		page_size = page_length * page_storage::entry_size;

		if (f_v) {
			cout << "page_storage::entry_size="
					<< page_storage::entry_size << endl;
			cout << "(int)sizeof(int)="
					<< (int)sizeof(int) << endl;
			cout << "page_length_log = "
					<< page_storage::page_length_log << endl;
			cout << "page_length = "
					<< page_length << endl;
			cout << "page_size = "
					<< page_size << endl;
			}
		if ((page_size / page_length) != page_storage::entry_size) {
			if (f_v) {
				cout << "page_storage::init page_size "
						"too big (arithmetic overflow)" << endl;
				}
			page_storage::page_length_log--;
			continue;
			}
		if (page_size > PAGE_STORAGE_MAX_PAGE_SIZE) {
			if (f_v) {
				cout << "page_storage::init page_size too big" << endl;
				cout << "the maximum page size in char is "
						<< PAGE_STORAGE_MAX_PAGE_SIZE << endl;
				}
			page_storage::page_length_log--;
			continue;
			}
		if (f_v) {
			cout << "page_size is OK" << endl;
			}
		break;
		}
		
		
	page_ptr_oversize = 2;
	allocation_table_length = (page_length >> 3) + 1;
	overall_length = 0;
	if (f_v) {
		cout << "allocation_table_length="
				<< allocation_table_length << endl;
		cout << "allocating pages / allocation_tables" << endl;
		}
	pages = NEW_puchar(page_ptr_oversize);
	allocation_tables = NEW_puchar(page_ptr_oversize);
	
	page_ptr_allocated = page_ptr_oversize;
	page_ptr_used = 1;
	if (f_v) {
		cout << "allocating page[0] of size " << page_size << endl;
		}
	pages[0] = NEW_uchar(page_size);
	if (f_v) {
		cout << "allocating allocation_tables[0] "
				"of size " << allocation_table_length << endl;
		}
	allocation_tables[0] = NEW_uchar(allocation_table_length);
	for (i = 0; i < allocation_table_length; i++) {
		allocation_tables[0][i] = 0;
		}
	if (f_v) {
		cout << "pages[0]/allocation_tables[0] allocated" << endl;
		//print();
		}
}

void page_storage::add_elt_print_function(
		void (* elt_print)(void *p, void *data, std::ostream &ost),
		void *elt_print_data)
{
	f_elt_print_function = true;
	page_storage::elt_print = elt_print;
	page_storage::elt_print_data = elt_print_data;
}

void page_storage::print()
{
	cout << "page_storage:" << endl;
	cout << "entry_size=" << entry_size << endl;
	cout << "page_length_log=" << page_length_log << endl;
	cout << "page_length=" << page_length << endl;
	cout << "page_size (in char )=" << page_size << endl;
	cout << "page_ptr_oversize=" << page_ptr_oversize << endl;
	cout << "allocation_table_length (in char)="
			<< allocation_table_length << endl;
	cout << "overall_length=" << overall_length << endl;
	cout << "page_ptr_used=" << page_ptr_used << endl;
	cout << "nb_free_entries = " << nb_free_entries << endl;
	cout << "next_free_entry = " << next_free_entry << endl;
	print_storage_used();
}

uchar *page_storage::s_i_and_allocate(
		long int i)
{
	uchar *p, *q;
	int j;
	algorithms Algo;
	
	long int page_idx = i & (page_length - 1);
	long int page = i >> page_length_log;
	if (page >= page_ptr_used) {
		cout << "page_storage::s_i_and_allocate "
				"page >= page_ptr_used" << endl;
		cout << "i=" << i << endl;
		print();
		exit(1);
		}
	//cout << "s_i(" << i << ") page=" << page
	// << " page_idx=" << page_idx << endl;
	if (page_idx * entry_size >= page_size) {
		cout << "page_idx * entry_size >= page_size" << endl;
		exit(1);
		}
	p = pages[page];
	p += page_idx * entry_size;
	q = allocation_tables[page];
	int word = page_idx >> 3;
	int bit = page_idx & 7;
	uchar mask = ((uchar) 1) << bit;
	//cout << "p=" << ((int) p) << " q=" << ((int)(q + word)) << endl;
	
	if (word >= allocation_table_length) {
		cout << "page_storage::s_i_and_allocate "
				"word >= allocation_table_length" << endl;
		cout << "word=" << word << endl;
		cout << "allocation_table_length="
				<< allocation_table_length << endl;
		print();
		exit(1);
		}
#if 0
	cout << "page_idx=" << page_idx << " word=" << word
			<< " bit=" << bit << " ";
	if (word)
		j = - 1;
	else 
		j = 0;
	for (; j < 5; j++) {
		uchar_print_bitwise(cout, q[word + j]);
		cout << " ";
		}
	cout << endl;
#endif
	if (q[word] & mask) {
		cout << "page_storage::s_i_and_allocate "
				"allocating entry which is currently in use" << endl;
		cout << "i=" << i << endl;
		cout << "page=" << page << endl;
		cout << "word=" << word << endl;
		cout << "bit=" << bit << endl;
		cout << "mask=" << (int)mask << endl;
		for (j = 0; j < 10; j++) {
			cout << (int)q[word + j] << " ";
			}
		cout << endl;
		for (j = 0; j < 10; j++) {
			Algo.uchar_print_bitwise(cout, q[word + j]);
			cout << " ";
			}
		cout << endl;
		print();
		exit(1);
		}
	q[word] |= mask;
#if 0
	cout << "page_idx=" << page_idx << " word="
			<< word << " bit=" << bit << " ";
	if (word)
		j = - 1;
	else 
		j = 0;
	for (; j < 5; j++) {
		uchar_print_bitwise(cout, q[word + j]);
		cout << " ";
		}
	cout << endl;
#endif
	return p;
}

uchar *page_storage::s_i_and_deallocate(
		long int i)
{
	uchar *p;
	
	if (i >= overall_length) {
		cout << "page_storage::s_i_and_deallocate "
				"i >= overall_length" << endl;
		cout << "i=" << i << endl;
		print();
		exit(1);
		}
	long int page_idx = i & (page_length - 1);
	long int page = i >> page_length_log;
	if (page >= page_ptr_used) {
		cout << "page_storage::s_i_and_deallocate "
				"page >= page_ptr_used" << endl;
		cout << "i=" << i << endl;
		print();
		exit(1);
		}
	//cout << "s_i(" << i << ") page=" << page
	// << " page_idx=" << page_idx << endl;
	p = pages[page] + page_idx * entry_size;
	int word = page_idx >> 3;
	int bit = page_idx & 7;
	uchar mask = ((uchar) 1) << bit;
	uchar not_mask = ~mask;
	if ((allocation_tables[page][word] & mask) == 0) {
		cout << "page_storage::s_i_and_deallocate "
				"deallocating entry which is currently not in use" << endl;
		cout << "i=" << i << endl;
		print();
		exit(1);
		}
	allocation_tables[page][word] &= not_mask;
	return p;
}

uchar *page_storage::s_i(
		long int i)
{
	if (i >= overall_length) {
		cout << "page_storage::s_i "
				"i >= overall_length" << endl;
		cout << "i=" << i << endl;
		print();
		exit(1);
		}
	
	long int page_idx = i & (page_length - 1);
	long int page = i >> page_length_log;
	if (page >= page_ptr_used) {
		cout << "page_storage::s_i "
				"page >= page_ptr_used" << endl;
		cout << "i=" << i << endl;
		print();
		exit(1);
		}
	int word = page_idx >> 3;
	int bit = page_idx & 7;
	uchar mask = ((uchar) 1) << bit;
	if ((allocation_tables[page][word] & mask) == 0) {
		cout << "page_storage::s_i "
				"access to entry which is currently not used" << endl;
		cout << "i=" << i << endl;
		print();
		exit(1);
		}
	//cout << "s_i(" << i << ") page=" << page
	// << " page_idx=" << page_idx << endl;
	return pages[page] + page_idx * entry_size;
}

uchar *page_storage::s_i_and_allocation_bit(
		long int i, int &f_allocated)
{
	if (i >= overall_length) {
		cout << "page_storage::s_i "
				"i >= overall_length" << endl;
		cout << "i=" << i << endl;
		print();
		exit(1);
		}
	
	long int page_idx = i & (page_length - 1);
	long int page = i >> page_length_log;
	if (page >= page_ptr_used) {
		cout << "page_storage::s_i "
				"page >= page_ptr_used" << endl;
		cout << "i=" << i << endl;
		print();
		exit(1);
		}
	int word = page_idx >> 3;
	int bit = page_idx & 7;
	uchar mask = ((uchar) 1) << bit;
	if (allocation_tables[page][word] & mask) {
		f_allocated = true;
		}
	else
		f_allocated = false;
	//cout << "s_i(" << i << ") page=" << page
	// << " page_idx=" << page_idx << endl;
	return pages[page] + page_idx * entry_size;
}

void page_storage::check_allocation_table()
{
	long int page_idx = overall_length & (page_length - 1);
	long int page = overall_length >> page_length_log;
	algorithms Algo;

	if (page >= page_ptr_used) {
		cout << "check_allocation_table::s_i "
				"page >= page_ptr_used" << endl;
		print();
		exit(1);
		}
	int word = page_idx >> 3;
	//int bit = page_idx & 7;
	//uchar mask = ((uchar) 1) << bit;
	for (word++; word < allocation_table_length; word++) {
		if (allocation_tables[page][word]) {
			int j;
			
			cout << "page_storage::check_allocation_table "
					"allocation_tables[page][word]" << endl;
			cout << "page_idx >> 3 = " << (page_idx >> 3) << endl;
			cout << "word = " << word << endl;
			for (j = 0; j < 10; j++) {
				cout << (int)allocation_tables[page][word + j] << " ";
				}
			cout << endl;
			for (j = 0; j < 10; j++) {
				Algo.uchar_print_bitwise(cout,
						allocation_tables[page][word + j]);
				cout << " ";
				}
			cout << endl;
			print();
			exit(1);
			}
		}
}

long int page_storage::store(
		uchar *elt)
{
	long int i, hdl;
	uchar *p, *q;
	algorithms Algo;
	
	if (nb_free_entries) {
		long int nfe = next_free_entry;
		
		p = s_i_and_allocate(nfe);
		long int next_next_free_entry;
		
		Algo.uchar_move(p, (uchar *) &next_next_free_entry, sizeof(long int));
		if (nb_free_entries > 1) {
			if (next_next_free_entry < 0 ||
					next_next_free_entry >= overall_length) {
				cout << "page_storage::store "
						"next_next_free_entry illegal" << endl;
				exit(1);
				}
			}
		else {
			if (next_next_free_entry != -1) {
				cout << "page_storage::store "
						"next_next_free_entry should be -1" << endl;
				exit(1);
				}
			}
		next_free_entry = next_next_free_entry;

		Algo.uchar_move(elt, p, entry_size);
		nb_free_entries--;
		hdl = nfe;
#ifdef DEBUG_PAGE_STORAGE
		cout << "page_storage: " << hdl << " reused" << endl;
		if (f_elt_print_function) {
			(*elt_print)(p, elt_print_data, cout);
			cout << endl;
			}
#endif
		}
	else {
		if (overall_length > (long int) page_ptr_used * page_length) {
			cout << "page_storage::store "
					"overall_length > page_ptr_used * page_length" << endl;
			cout << "overall_length=" << overall_length << endl;
			cout << "page_ptr_used=" << page_ptr_used << endl;
			cout << "page_length=" << page_length << endl;
			exit(1);
			}
		if (overall_length == page_ptr_used * page_length) {
			cout << "\npage_storage::store "
					"allocating page # "
					<< (overall_length >> page_length_log) << endl;
			if (page_ptr_used == page_ptr_allocated) {
				uchar **pages1 =
						NEW_puchar(page_ptr_used + page_ptr_oversize);
				uchar **allocation_tables1 =
						NEW_puchar(page_ptr_used + page_ptr_oversize);
				for (i = 0; i < page_ptr_used; i++) {
					pages1[i] = pages[i];
					allocation_tables1[i] = allocation_tables[i];
					}
				FREE_puchar(pages);
				FREE_puchar(allocation_tables);
				pages = pages1;
				allocation_tables = allocation_tables1;
				page_ptr_allocated += page_ptr_oversize;
				}
			p = NEW_uchar(page_size);
			q = NEW_uchar(allocation_table_length);
			for (i = 0; i < page_size; i++) {
				p[i] = 0;
				}
			for (i = 0; i < allocation_table_length; i++) {
				q[i] = 0;
				}
			pages[page_ptr_used] = p;
			allocation_tables[page_ptr_used] = q;
			page_ptr_used++;
			}
			
#ifdef DEBUG_PAGE_STORAGE
		cout << "calling s_i_and_allocate" << endl;
		cout << "calling s_i_and_allocate "
				"overall_length = " << overall_length << endl;
#endif		
		p = s_i_and_allocate(overall_length);
		Algo.uchar_move(elt, p, entry_size);
		//cout << "storing at " << p << endl;
		//check_allocation_table();
		hdl = overall_length++;
#ifdef DEBUG_PAGE_STORAGE
		cout << "page_storage: " << hdl << " allocated" << endl;
		if (f_elt_print_function) {
			(*elt_print)(p, elt_print_data, cout);
			cout << endl;
			}
#endif
		}
	//check_free_list();
	return hdl;
}

void page_storage::dispose(
		long int hdl)
{
#if 1
	uchar *p = s_i_and_deallocate(hdl);
	algorithms Algo;

	long int next_next_free_entry = next_free_entry;
	Algo.uchar_move((uchar *) &next_next_free_entry, p, sizeof(long int));
	next_free_entry = hdl;
	nb_free_entries++;
	//check_free_list();
#ifdef DEBUG_PAGE_STORAGE
	cout << "page_storage: " << hdl << " deallocated" << endl;
#endif
#endif
}

void page_storage::check_free_list()
{
	long int nb = 0, nfe;
	int f_allocated;
	uchar *p;
	algorithms Algo;
	
	if (nb_free_entries == 0)
		return;
	nfe = next_free_entry;
	while (true) {
		nb++;
		p = s_i_and_allocation_bit(nfe, f_allocated);
		if (f_allocated) {
			cout << "page_storage::check_free_list "
					"inconsistency: empty entry marked "
					"as allocated" << endl;
			cout << "nb_free_entries = " << nb_free_entries;
			cout << "entry = " << nfe << endl;
			print();
			exit(1);
			}
		Algo.uchar_move(p, (uchar *) &nfe, sizeof(long int));
		if (nfe == -1)
			break;
		}
	if (nb != nb_free_entries) {
		cout << "page_storage::check_free_list "
				"inconsistency" << endl;
		cout << "nb_free_entries = " << nb_free_entries;
		cout << "nb = " << nb << endl;
		print();
		exit(1);
		}
}

void page_storage::print_storage_used()
{
	cout << overall_length << " group elements stored overall on " 
		<< page_ptr_used << " pages a "
		<< " 2^{" << page_length_log << "} entries\n"
		<< nb_free_entries << " entries currently empty" << endl;
}

#if 0
void test_page_storage(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "test_page_storage" << endl;
	}
	{
		page_storage *Elts;
		Elts = NEW_OBJECT(page_storage);

		int char_per_elt = 20;
		int page_length_log = PAGE_LENGTH_LOG;

		Elts->init(char_per_elt /* entry_size */, page_length_log, verbose_level - 1);
		cout << "destroying Elts" << endl;
		FREE_OBJECT(Elts);
	}
	if (f_v) {
		cout << "test_page_storage done" << endl;
	}
}
#endif

}}}}


