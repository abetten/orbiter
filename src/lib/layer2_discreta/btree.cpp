// btree.cpp
//
// Anton Betten
// 28.11.2000
// moved from D2 to ORBI Nov 15, 2007

#include "../layer2_discreta/discreta.h"
#include "layer1_foundations/foundations.h"

using namespace std;



namespace orbiter {
namespace layer2_discreta {
namespace typed_objects {


//int bt_debug = FALSE;

#undef USE_TABLE
#define WRITE_INFO_ONLY_AT_END

#define MAX_ROOT_BUF 20








int f_RootBF_free[MAX_ROOT_BUF];
Buffer *RootBF = NULL;
Buffer *tmpBF = NULL;
	// used in: bt_open(), WriteInfo(), AllocateRec(), ReleaseRec()



int fstream_table_used[MAX_FSTREAM_TABLE];
fstream *fstream_table[MAX_FSTREAM_TABLE];





static void bt_item_copy(ItemTyp *a, ItemTyp *b);

// #############################################################################
// global stuff
// #############################################################################



void database_init(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	
	if (f_v) {
		cout << "database_init()" << endl;
		}
	
	if (f_vv) {
		cout << "BTREEMAXKEYLEN=" << BTREEMAXKEYLEN << endl;
		cout << "BTREEHALFPAGESIZE=" << BTREEHALFPAGESIZE << endl;
		cout << "BTREE_PAGE_LENGTH_LOG=" << BTREE_PAGE_LENGTH_LOG << endl;
		cout << "database_init() sizeof(ItemTyp)=" << sizeof(ItemTyp) << endl;
		cout << "database_init() sizeof(PageTyp)=" << sizeof(PageTyp) << endl;
		cout << "database_init() sizeof(Buffer)=" << sizeof(Buffer) << endl;
		cout << "database_init() sizeof(int_4)=" << sizeof(int_4) << endl;
		}
	
	RootBF = (Buffer *) new Buffer[MAX_ROOT_BUF];
	if (RootBF == NULL) {
		cout << "database_init() no memory for RootBF" << endl;
		exit(1);
		}
	for (i = 0; i < MAX_ROOT_BUF; i++) {
		f_RootBF_free[i] = TRUE;
		}
	f_RootBF_free[0] = FALSE;
	
	
	
	tmpBF = RootBF;
	
	for (i = 0; i < MAX_FSTREAM_TABLE; i++) {
		fstream_table_used[i] = FALSE;
		}

	page_table_init(verbose_level - 1);

	if (f_v) {
		cout << "database_init() done" << endl;
		}
}

void database_exit(void)
{
	int verbose_level = 0;
	
	if (RootBF) {
		delete [] RootBF;
		RootBF = NULL;
		}
	page_table_exit(verbose_level - 1);
}

int fstream_table_get_free_entry()
// never gives out handle 0, as it stands for file not open
// that is, the zeroth-entry is never used
{
	for (int i = 1; i < MAX_FSTREAM_TABLE; i++) {
		if (!fstream_table_used[i])
			return i;
		}
	cout << "fstream_table_get_free_entry() table full, "
			"too many open files" << endl;
	exit(1);
}


int root_buf_alloc(void)
// error if there is no free buffer 
{
	int i;
	
	for (i = 0; i < MAX_ROOT_BUF; i++) {
		if (f_RootBF_free[i]) {
			f_RootBF_free[i] = FALSE;
			return i;
			}
		}
	cout << "root_buf_alloc() no free root buffer" << endl;
	exit(1);
}

void root_buf_free(int i)
{
	if (i < 0 || i >= MAX_ROOT_BUF) {
		cout << "root_buf_free()|i illegal" << endl;
		exit(1);
		}
	f_RootBF_free[i] = TRUE;
}



// #############################################################################
// class btree
// #############################################################################

btree::btree() : Vector()
{
	k = BTREE;
}

btree::btree(const discreta_base &x)
	// copy constructor:    this := x
{
	cout << "btree::copy constructor for object: "
			<< const_cast<discreta_base &>(x) << "\n";
	const_cast<discreta_base &>(x).copyobject_to(*this);
}

btree& btree::operator = (const discreta_base &x)
	// copy assignment
{
	cout << "btree::operator = (copy assignment)" << endl;
	copyobject(const_cast<discreta_base &>(x));
	return *this;
}

void btree::settype_btree()
{
	OBJECTSELF s;
	
	s = self;
	new(this) btree;
	self = s;
	k = BTREE;
}

btree::~btree()
{
	freeself_btree();
}

void btree::freeself_btree()
{
	// cout << "group_selection::freeself_btree()\n";
	freeself_vector();
}

kind btree::s_virtual_kind()
{
	return BTREE;
}

void btree::copyobject_to(discreta_base &x)
{
#ifdef COPY_VERBOSE
	cout << "btree::copyobject_to()\n";
	print_as_vector(cout);
#endif
	Vector::copyobject_to(x);
	x.as_btree().settype_btree();
#ifdef COPY_VERBOSE
	x.as_btree().print_as_vector(cout);
#endif
}

ostream& btree::print(ostream& ost)
{
	
	ost << "BTREE: Root = " << Root() << " FreeRec = " << FreeRec() 
		<< " AllocRec = " << AllocRec() << endl; 
	return ost;
}

void btree::init(
		const char *file_name, int f_duplicatekeys,
	int btree_idx)
{
	m_l_n(11);
	c_kind(BTREE);
	btree::f_duplicatekeys() = f_duplicatekeys;
	key().change_to_vector();
	key().m_l(0);
	filename().change_to_hollerith();
	filename().init(file_name);
	f_open() = FALSE;
	stream() = 0;
	buf_idx() = 0;
	Root() = 0;
	FreeRec() = 0;
	AllocRec() = 0;
	btree::btree_idx() = btree_idx;
	page_table_idx() = -1;
}

void btree::add_key_int4(int field1, int field2)
{
	bt_key bk;
	
	bk.init_int4(field1, field2);
	key().append(bk);
}

void btree::add_key_int2(int field1, int field2)
{
	bt_key bk;
	
	bk.init_int2(field1, field2);
	key().append(bk);
}

void btree::add_key_string(int output_size, int field1, int field2)
{
	bt_key bk;
	
	bk.init_string(output_size, field1, field2);
	key().append(bk);
}

void btree::key_fill_in(char *the_key, Vector& the_object)
{
	bt_key_fill_in(the_key, key(), the_object);
}

void btree::key_print(char *the_key, ostream& ost)
{
	bt_key_print(the_key, key(), ost);
}

void btree::create(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	Buffer *Root_BF;
	
	if (f_v) {
		cout << "btree::create" << endl;
		}
	if (f_open() == TRUE) {
		cout << "btree::create() already open" << endl;
		exit(1);
		}
	file_create();
	
	buf_idx() = root_buf_alloc();
	Root_BF = RootBF + buf_idx();
	fill_char(Root_BF, (int)sizeof(Buffer), 0);

	FreeRec() = 0;
	AllocRec() = 0;
	Root() = 0;
	
	f_open() = TRUE;
	
	WriteInfo(FALSE);
	
	if (f_vv) {
		cout << "before page_table_alloc" << endl;
		}
	page_table_idx() = page_table_alloc(verbose_level - 2);
	if (f_vv) {
		cout << "page_table_idx = " << page_table_idx() << endl;
		}
}

void btree::open(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	Buffer *Root_BF;
	
	if (f_v) {
		cout << "btree::open verbose_level=" << verbose_level << endl;
		}
	if (f_open() == TRUE) {
		cout << "btree::open() already open" << endl;
		exit(1);
		}
	file_open();
	
	buf_idx() = root_buf_alloc();
	Root_BF = RootBF + buf_idx();
	fill_char(Root_BF, (int)sizeof(Buffer), 0);
	


	if (f_vv) {
		cout << "before page_table_alloc" << endl;
		}
	page_table_idx() = page_table_alloc(verbose_level - 2);
	if (f_vv) {
		cout << "page_table_idx = " << page_table_idx() << endl;
		}

	
	ReadInfo(verbose_level);
	
	if (Root() != 0) {
		if (f_vv) {
			cout << "reading root page " << Root() << endl;
			}
		file_seek(Root());
		file_read(&Root_BF->Page, "btree::open");
		Root_BF->PageNum = (int_4) Root();
		}
	else {
		Root_BF->PageNum = 0;
		}
}

void btree::close(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "btree::close" << endl;
		}

#if 0
	// write root:
	Buffer *BF;
	BF = RootBF + buf_idx();

	file_seek(Root());
	file_write(&BF->Page, "SavePage");
#endif

#ifdef USE_TABLE
	page_table *T;
	
	T = page_table_pointer(page_table_idx());
	T->write_pages_to_file(this, buf_idx(), verbose_level);
#endif
#ifdef WRITE_INFO_ONLY_AT_END
	WriteInfo(verbose_level);
#endif
	file_close();
	root_buf_free(buf_idx());
	page_table_free(page_table_idx(), verbose_level - 2);
	page_table_idx() = -1;
}

void btree::ReadInfo(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "btree::ReadInfo" << endl;
		}
	
	if (f_vv) {
		cout << "reading page 0" << endl;
		}
	Buffer *BF;
	BF = new Buffer;
	file_seek(0);
	file_read(&BF->Page, "btree::open");
	
	FreeRec() = BF->Page.NextFreeRec;
	AllocRec() = BF->Page.AllocRec;
	Root() = BF->Page.RootRec;
	
	if (f_vv) {
		cout << "FreeRec()" << FreeRec() << endl;
		cout << "AllocRec()" << AllocRec() << endl;
		cout << "Root()" << Root() << endl;
		}
	delete BF;
}

void btree::WriteInfo(int verbose_level)
/* Schreibt die Variablen 'AllocRec', 'FreeRec' und 'Root' 
 * als 'AllocRec', 'NextFreeRec' und 'RootRec' 
 * in die 0-te Seite der Datenbank. */
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int size;
	Buffer *BF = tmpBF;
	
	if (f_v) {
		cout << "btree::WriteInfo" << endl;
		}
	if (!f_open()) {
		cout << "btree::WriteInfo() file not open" << endl;
		exit(1);
		}
	size = sizeof(Buffer);
	fill_char((char *)BF, size, 0);
	BF->Page.AllocRec = (int_4) AllocRec();
	BF->Page.NextFreeRec = (int_4) FreeRec();
	BF->Page.RootRec = (int_4) Root();
	
	if (f_vv) {
		cout << "btree::WriteInfo writing page 0" << endl;
		cout << "FreeRec()" << FreeRec() << endl;
		cout << "AllocRec()" << AllocRec() << endl;
		cout << "Root()" << Root() << endl;
		}
	
	file_seek(0);
	file_write(&BF->Page, "WriteInfo");
}

int btree::AllocateRec(int verbose_level)
/* Ein freier Record der Datanbank wird ermittelt
 * --
 * rueckgabe: Gibt Nummer eines freien Records an */
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	Buffer *BF = tmpBF;
	int x;
	
	if (f_vv) {
		cout << "btree::AllocateRec" << endl;
		}
	if (!f_open()) {
		cout << "btree::AllocateRec() not open\n";
		exit(1);
		}
	if (FreeRec() == 0) {
		AllocRec()++;
#ifdef WRITE_INFO_ONLY_AT_END
#else
		WriteInfo(FALSE);
#endif
		x = AllocRec();
#ifdef USE_TABLE
		page_table *T;
		
		T = page_table_pointer(page_table_idx());
		T->allocate_rec(BF, buf_idx(), x, verbose_level - 1);
#else
#endif
		}
	else {
#ifdef USE_TABLE
		cout << "btree::AllocateRec FreeRec() > 0 not yet implemented" << endl;
		exit(1);
#endif
		fill_char((char *)BF, sizeof(Buffer), 0);
		x = FreeRec();
		file_seek(x);
		file_read(&BF->Page, "AllocateRec");
		FreeRec() = BF->Page.NextFreeRec;
#ifdef WRITE_INFO_ONLY_AT_END
#else
		WriteInfo(f_v);
#endif
		}
	if (f_v) {
		cout << "btree::AllocateRec() x = " << x << endl;
		}
	return x;
}

void btree::ReleaseRec(int x)
/* Gibt einen Record wieder frei
 * Der Block wird an den Anfang der Frei-Liste eingefuegt.
 * --
 * int x - Nummer des freizugebenden Records */
{
	int size;
	Buffer *BF = tmpBF;
	
	if (!f_open()) {
		cout << "btree::ReleaseRec() not open\n";
		exit(1);
		}
	size = (int)sizeof(Buffer);
	fill_char((char *)BF, size, 0);
	BF->Page.NextFreeRec = (int_4) FreeRec();
	file_seek(x);
	file_write(&BF->Page, "ReleaseRec");
	FreeRec() = x;
#ifdef WRITE_INFO_ONLY_AT_END
#else
	WriteInfo(FALSE);
#endif
#ifdef USE_TABLE
	cout << "btree::ReleaseRec not yet implemented" << endl;
	exit(1);
#endif
}

void btree::LoadPage(Buffer *BF, int x, int verbose_level)
/* Laedt eine Seite in den Speicher. Soll die Wurzel ge-
 * laden werden, so wird nur der Puffer kopiert
 * --
 * Buffer *BF - Puffer enthaelt nach Aufruf die Seite
 * int x      - zu ladende Seite */
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "btree::LoadPage x=" << x << endl;
		}
	if (!f_open()) {
		cout << "btree::LoadPage() not open" << endl;
		exit(1);
		}
	if (FALSE /* x == Root() */) {
		Buffer *Root_BF;
		Root_BF = RootBF + buf_idx();
		*BF = *Root_BF;
		}
	else {
		BF->PageNum = (int_4) x;
#ifdef USE_TABLE
		page_table *T;
	
		T = page_table_pointer(page_table_idx());
		if (!T->load_page(BF, x, buf_idx(), verbose_level)) {
			if (f_vv) {
				cout << "btree::LoadPage loading page " << x <<" from file" <<  endl;
				}
			file_seek(x);
			BF->PageNum = x;
			file_read(&BF->Page, "LoadPage");
			if (f_vvv) {
				page_print(BF, cout);
				}
			if (f_vv) {
				cout << "saving page " << x << " to page tables" << endl;
				}
			T->save_page(BF, buf_idx(), verbose_level);
			//page_print(BF, cout);
			}
#else
		file_seek(x);
		file_read(&BF->Page, "LoadPage");
#endif
		}
}

void btree::SavePage(Buffer *BF, int verbose_level)
/* Eine Seite wird auf den Hintergrundspeicher geschrieben
 * --
 * Buffer *BF - Zu speichernde Seite */
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "btree::SavePage" << endl;
		}
	if (!f_open()) {
		cout << "btree::SavePage() not open" << endl;
		exit(1);
		}
	if (FALSE /*BF->PageNum == Root()*/) {
		Buffer *Root_BF;
		Root_BF = RootBF + buf_idx();
		*Root_BF = *BF;
		file_seek(Root_BF->PageNum);
		file_write(&Root_BF->Page, "SavePage, root");
		}
	else {
#ifdef USE_TABLE
		page_table *T;
	
		T = page_table_pointer(page_table_idx());
		T->save_page(BF, buf_idx(), verbose_level);
#else
		file_seek(BF->PageNum);
		file_write(&BF->Page, "SavePage");
#endif

		}
}

int btree::search_string(discreta_base& key_op, int& pos, int verbose_level)
{
	KEYTYPE the_key;
	char *p_key = the_key.c;
	
	bt_key &the_bt_key = key()[0].as_bt_key();
	
	if (the_bt_key.type() != bt_key_string) {
		cout << "btree::search_string() bt_key not of type string" << endl;
		exit(1);
		}
	
	bt_key_fill_in_string(&p_key, the_bt_key.output_size(), key_op);
	
	return search(&the_key, NULL, &pos, 1, verbose_level);
}

void btree::search_interval_int8(long int i_min, long int i_max,
	int& first, int &len, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "btree::search_interval_int8" << endl;
	}

	KEYTYPE key_min, key_max;

	char *p_key_min = key_min.c;
	char *p_key_max = key_max.c;

	integer I_min, I_max;

	I_min.m_i(i_min - 1);
	I_max.m_i(i_max);

	int idx_min, idx_max;
	int f_found_min, f_found_max;
	
	bt_key_fill_in_int8(&p_key_min, I_min);
	bt_key_fill_in_int8(&p_key_max, I_max);
	if (f_v) {
		cout << "btree::search_interval_int8 I_min=" << I_min << " I_max=" << I_max << endl;
	}
	
	f_found_min = search(&key_min, NULL, &idx_min, 1, verbose_level);
	f_found_max = search(&key_max, NULL, &idx_max, 1, verbose_level);
	if (f_v) {
		cout << "btree::search_interval_int8 f_found_min=" << f_found_min << " idx_min=" << idx_min << endl;
		cout << "btree::search_interval_int8 f_found_max=" << f_found_max << " idx_max=" << idx_max << endl;
	}
	first = idx_min + 1;
	len = idx_max - idx_min;
}

void btree::search_interval_int8_int8(long int l0, long int u0,
	long int l1, long int u1,
	int& first, int &len, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "btree::search_interval_int8_int8" << endl;
	}

	if (f_v) {
		cout << "btree::search_interval_int8_int8 low=(" << l0 << "," << l1 << ") high=(" << u0 << "," << u1 << ")" << endl;
	}

	KEYTYPE key_min, key_max;

	char *p_key_min = key_min.c;
	char *p_key_max = key_max.c;

	integer I_min, I_max;

	int idx_min, idx_max;
	int f_found_min, f_found_max;
	
	I_min.m_i(l0);
	I_max.m_i(u0);
	bt_key_fill_in_int8(&p_key_min, I_min);
	bt_key_fill_in_int8(&p_key_max, I_max);

	I_min.m_i(l1 - 1);
	I_max.m_i(u1);
	bt_key_fill_in_int8(&p_key_min, I_min);
	bt_key_fill_in_int8(&p_key_max, I_max);


	f_found_min = search(&key_min, NULL, &idx_min, 2, verbose_level);
	f_found_max = search(&key_max, NULL, &idx_max, 2, verbose_level);
	if (f_v) {
		cout << "search_interval_int8_int8() f_found_min=" << f_found_min << " idx_min=" << idx_min << endl;
		cout << "search_interval_int8_int8() f_found_max=" << f_found_max << " idx_max=" << idx_max << endl;
	}
	first = idx_min + 1;
	len = idx_max - idx_min;
}

void btree::search_interval_int8_int8_int8(long int l0, long int u0,
	long int l1, long int u1, long int l2, long int u2,
	int& first, int &len, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "btree::search_interval_int8_int8_int8" << endl;
	}

	KEYTYPE key_min, key_max;
	char *p_key_min = key_min.c;
	char *p_key_max = key_max.c;
	integer I_min, I_max;
	int idx_min, idx_max;
	int f_found_min, f_found_max;
	
	I_min.m_i(l0);
	I_max.m_i(u0);
	bt_key_fill_in_int8(&p_key_min, I_min);
	bt_key_fill_in_int8(&p_key_max, I_max);

	I_min.m_i(l1);
	I_max.m_i(u1);
	bt_key_fill_in_int8(&p_key_min, I_min);
	bt_key_fill_in_int8(&p_key_max, I_max);

	I_min.m_i(l2 - 1);
	I_max.m_i(u2);
	bt_key_fill_in_int8(&p_key_min, I_min);
	bt_key_fill_in_int8(&p_key_max, I_max);


	f_found_min = search(&key_min, NULL, &idx_min, 3, verbose_level);
	f_found_max = search(&key_max, NULL, &idx_max, 3, verbose_level);
	if (f_v) {
		cout << "search_interval_int4_int4_int4() f_found_min=" << f_found_min << " idx_min=" << idx_min << endl;
		cout << "search_interval_int4_int4_int4() f_found_max=" << f_found_max << " idx_max=" << idx_max << endl;
		}
	first = idx_min + 1;
	len = idx_max - idx_min;
}

void btree::search_interval_int8_int8_int8_int8(
		long int l0, long int u0,
		long int l1, long int u1,
		long int l2, long int u2,
		long int l3, long int u3,
		int& first, int &len, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "btree::search_interval_int8_int8_int8_int8" << endl;
	}


	KEYTYPE key_min, key_max;
	char *p_key_min = key_min.c;
	char *p_key_max = key_max.c;
	integer I_min, I_max;
	int idx_min, idx_max;
	int f_found_min, f_found_max;
	
	I_min.m_i(l0);
	I_max.m_i(u0);
	bt_key_fill_in_int8(&p_key_min, I_min);
	bt_key_fill_in_int8(&p_key_max, I_max);

	I_min.m_i(l1);
	I_max.m_i(u1);
	bt_key_fill_in_int8(&p_key_min, I_min);
	bt_key_fill_in_int8(&p_key_max, I_max);

	I_min.m_i(l2);
	I_max.m_i(u2);
	bt_key_fill_in_int8(&p_key_min, I_min);
	bt_key_fill_in_int8(&p_key_max, I_max);

	I_min.m_i(l3 - 1);
	I_max.m_i(u3);
	bt_key_fill_in_int8(&p_key_min, I_min);
	bt_key_fill_in_int8(&p_key_max, I_max);


	f_found_min = search(&key_min, NULL, &idx_min, 4, verbose_level);
	f_found_max = search(&key_max, NULL, &idx_max, 4, verbose_level);
	if (f_v) {
		cout << "search_interval_int8_int8_int8_int8() f_found_min=" << f_found_min << " idx_min=" << idx_min << endl;
		cout << "search_interval_int8_int8_int8_int8() f_found_max=" << f_found_max << " idx_max=" << idx_max << endl;
		}
	first = idx_min + 1;
	len = idx_max - idx_min;
}

int btree::search_int8_int8(
		long int data1, long int data2, int& idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "btree::search_int8_int8 data=(" << data1 << "," << data2 << ")" << endl;
	}

	KEYTYPE key;
	char *p_key = key.c;
	integer I1, I2;
	int f_found;
	
	I1.m_i(data1);
	I2.m_i(data2);
	bt_key_fill_in_int8(&p_key, I1);
	bt_key_fill_in_int8(&p_key, I2);



	f_found = search(&key, NULL, &idx, 2, verbose_level);
	if (f_v) {
		cout << "search_int4_int4() f_found=" << f_found << " idx=" << idx << endl;
		}
	return f_found;
}

int btree::search_unique_int8(long int i, int verbose_level)
// returns -1 if the element could not be found or is not unique.
// otherwise, the idx of the element is returned
{
	int first, len;
	
	search_interval_int8(i, i, first, len, verbose_level);
	if (len > 1) {
		cout << "btree::search_unique_int8() WARNING: element is not unique, returning -1" << endl;
		return -1;
		}
	if (len == 0) {
		return -1;
		}
	return first;
}

int btree::search_unique_int8_int8_int8_int8(long int i0, long int i1,
	long int i2, long int i3, int verbose_level)
// returns -1 if an element whose key starts with [i0,i1,i2,i3] could not be found or is not unique.
// otherwise, the idx of that element is returned
{
	int f_v = (verbose_level >= 1);
	int first, len;
	
	search_interval_int8_int8_int8_int8(
		i0, i0, 
		i1, i1, 
		i2, i2, 
		i3, i3, 
		first, len, verbose_level);
	
	if (f_v) {
		print_range(first - 10, len + 20, cout);
		cout << "key=[" << i0 << ", " << i1 << ", " << i2 << ", " << i3 << "]" << endl;
		cout << "first=" << first << " len=" << len << endl;
		}

	if (len > 1) {
		print_all(cout);
		cout << "btree::search_unique_int4_int4_int4_int4() WARNING: element is not unique, returning -1" << endl;
		cout << "key=[" << i0 << ", " << i1 << ", " << i2 << ", " << i3 << "]" << endl;
		cout << "first=" << first << " len=" << len << endl;
		exit(1);
		}
	if (len == 0) {
		return -1;
		}
	return first;
}

int btree::search_datref_of_unique_int8(long int i, int verbose_level)
{
	int no = search_unique_int8(i, verbose_level);
	if (no == -1) {
		cout << "btree::search_unique_int8() no == -1, cannot determine element" << endl;
		exit(1);
		}
	KEYTYPE key;
	DATATYPE data;
	
	ith(no, &key, &data, verbose_level - 1);
	return data.datref;
}

int btree::search_datref_of_unique_int8_if_there(long int i, int verbose_level)
{
	int no = search_unique_int8(i, verbose_level);
	if (no == -1) {
		return -1;
		}
	KEYTYPE key;
	DATATYPE data;
	
	ith(no, &key, &data, verbose_level - 1);
	return data.datref;
}

long int btree::get_highest_int8()
// returns -1 if the btree is empty,
// otherwise the int4 value of the key 
// of the last (highest) element in the btree.
{
	int len;
	KEYTYPE key;
	DATATYPE data;
	int verbose_level = 0;
	
	len = length(verbose_level);
	if (len == 0) {
		return -1;
	}
	ith(len - 1, &key, &data, verbose_level);
	// cout << i << " : ";
	// key_print(key.c, ost);
	// cout << endl;
	char *p_key = key.c;
	int_8 i;
	bt_key_get_int8(&p_key, i);
	return i;
}

void btree::get_datrefs(int first, int len, Vector& datrefs)
{
	KEYTYPE key;
	DATATYPE data;
	int verbose_level = 0;

	datrefs.m_l_n(len);
	for (int i = 0; i < len; i++) {
		ith(first + i, &key, &data, verbose_level);
		datrefs.m_ii(i, data.datref);
	}
}

int btree::search(void *pSearchKey, 
	DATATYPE *pData, int *idx, 
	int key_depth, 
	int verbose_level)
// returns true iff searchKey has been found.

/* void pointer pSearchKey hier.
 * pSearchKey wird nicht benutzt, nur an (*cmp_func)() 
 * durchgeschleift. Insbesondere kann pSearchKey 
 * auf laengere Daten als KEYTYPE zeigen. 
 * Anwendung: ein evtl. langer Suchstring, 
 * der nicht in einen KEYTYPE passen wuerde. 
 * Es wird zusatzlich mit pData->datref 
 * gesucht, sofern pData != NIL.
 * idx enthaelt Nummer des gefundenen bzw des naechst 
 * kleineren Datensatzes. 
 * idx kann also auch -1 werden (nicht gefunden - 
 * vor dem 0-ten Datensatz) */
{
	int f_v = (verbose_level >= 1);
	Buffer Buf;
	int f_found;
	
	if (f_v) {
		cout << "btree::search Root()=" << Root() << endl;
		}
	if (!f_open()) {
		cout << "btree::search() not open" << endl;
		exit(1);
		}
	*idx = -1;
	f_found = SearchBtree(Root(), 
		pSearchKey, 
		pData, 
		&Buf, 
		idx, 
		key_depth, 
		verbose_level);
	if (f_v) {
		cout << "btree::search done, f_found=" << f_found << endl;
		}
	return f_found;
}

int btree::SearchBtree(int page, 
	void *pSearchKey, DATATYPE *pData, 
	Buffer *Buf, int *idx, int key_depth, 
	int verbose_level)
// returns true iff searchKey has been found.

/* Sucht einen Schluessel in der Datenbank
 * --
 * KEYTYPE  pSearchKey - Zu suchender Schluessel
 * DATATYPE *pData     - Zum Schluessel gehoerende Daten
 * int      *idx         - Enthaelt Nummer des gefundenen bzw des naechst 
 *                         kleineren Datensatzes
 * int      *f_found     - Gibt an, ob Daten gefunden wurden */
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int x, f_found, f_found1, idx1;
	
	if (f_v) {
		cout << "btree::SearchBtree page=" << page << endl;
		}
	if (page == 0) {
		f_found = FALSE;
		}
	else {
		LoadPage(Buf, page, verbose_level - 1);
		if (f_vv) {
			cout << "calling SearchPage" << endl;
			}
		f_found = SearchPage(Buf, 
			pSearchKey, NULL, 
			idx, &x, 
			key_depth, 
			verbose_level);
		if (f_v) {
			cout << "SearchPage returns " << f_found << " with x=" << x << endl;
			}
		idx1 = *idx;
		f_found1 = f_found;
		if (f_found) {
			if (pData) {
				*pData = Buf->Page.Item[x].Data;
				}
			}
		f_found = SearchBtree(Buf->Page.Item[x].Ref, 
			pSearchKey, pData, 
			Buf, idx, key_depth, 
			verbose_level);
		if (!f_found && f_found1) {
			if (pData) {
				LoadPage(Buf, page, verbose_level - 1);
				*pData = Buf->Page.Item[x].Data;
				}
			f_found = TRUE;
			*idx = idx1;
			}
#if 0
		if (*f_found) {
			if (FKpData) {
				*FKpData = FKBF->Page.Item[x].Data;
				}
			}
		else {
			if (SearchBtree(p, 
				FKBF->Page.Item[x].Ref, 
				idx, f_found) != OK) {
				return error("SearchBtree() error in SearchBtree()");
				}
			}
#endif
		}
	if (f_v) {
		cout << "btree::SearchBtree page=" << page << " done, f_found=" << f_found << endl;
		}
	return f_found;
}

int btree::SearchPage(Buffer *buffer, 
	void *pSearchKey, DATATYPE *pSearchData, 
	int *cur, int *x, int key_depth, 
	int verbose_level)
// binary search within one page.
// returns true if the key searched for has been found, false otherwise
// in case that the key has been found, 
// x contains the position of that element.
// Otherwise, x contains the position of the next smaller element



/* Fuehrt binaere Suche innerhalb einer Seite aus.
 * --
 * Buffer *BF            - zu untersuchende Seite.
 * KEYTYPE  *SearchKey   - Zu suchender Schluessel.
 * DATATYPE *pSearchData - optional, nur datref verwendet. 
 *                         Bei gleichen Schluesseln werden 
 *                         zusaetzlich die datref's verglichen.
 * int *x                - Gibt Position an, falls Suche 
 *                         erfolgreich. Ansonsten naechst 
 *                         kleinerers Element.
 *                         x kann also auch null werden.
 * int *Found            - Gibt an, ob Schluessel gefunden.
 * 
 * Es wird aufgerufen:
 *   WORD (*cmp_func)(void *key1, void *key2, int *res);
 * Key1 stammt aus der Datenbank, key2 ist der durchgeschleifte, 
 * zu suchende Schluessel. 
 * Ergebnis:
 *   res < 0:  key1 < key2
 *   res == 0: key1 == key2
 *   res > 0:  key1 > key2
 * Bei gleichen Schluesseln werden intern noch die datref's 
 * verglichen, sofern pSearchData gesetzt ist. 
 * Bei gleichen Schluesseln (und evtl. gleichen datref's) 
 * wird der letzte Eintrag gesucht und zurueckgegeben. 
 * *cur wird erhoeht um die Childs Zaehler auf dieser Seite 
 * (incl. 0-tem Eintrag) bis unmittelbar vor dem gefundenen 
 * Datensatz, und dann + 1, so dass cur die aktuelle 
 * Datensatznummer enthaelt, wenn es vorher im Suchbaum 
 * schon mitgefuehrt wurde. */
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	ItemTyp *item = NULL;
	uint_4 searchdatref = 0;
	int childs;
	int r, l, i;
	int res;
	int f_found = FALSE;
	
	if (f_v) {
		cout << "btree::SearchPage" << endl;
		}
	if (f_vv) {
		cout << "page:" << endl;
		page_print(buffer, cout);
		}
	if (pSearchData != NULL) {
		searchdatref = pSearchData->datref;
		}
	l = 0;
	r = buffer->Page.NumItems + 1;
	while (l + 1 < r) {
		*x = ((l + r) >> 1);
		item = &buffer->Page.Item[*x];
		
		if (f_vv) {
			cout << "btree::SearchPage()|l = " << l << " *x = " << *x << " r = " << r << endl;
			item_print(item, *x, cout);
			}
		res = bt_key_compare(
			item->Key.c, 
			(char *) pSearchKey, 
			key(), 
			key_depth);
		if (res == 0) {
			/* wenn pSearchData gesetzt, 
			 * dann bei gleichen Schluesseln 
			 * Suche nach datref */
			if (pSearchData != NULL) {
				if (item->Data.datref > searchdatref) {
					res = 1;
					}
				else if (item->Data.datref < searchdatref) {
					res = -1;
					}
				}
			}
		if (res == 0) {
			f_found = TRUE;
			l = *x;
			}
		else {
			if (res > 0) { /* Page.Item[].Key > *pSearchKey */
				r = *x;
				}
			else {
				if (f_found) {
					cout << "SearchPage() not ascending" << endl;
					exit(1);
					}
				l = *x;
				}
			}
		}
	*x = l;
	item = buffer->Page.Item;
	for (i = 0; i < *x; i++) {
		childs = item[i].Childs;
		*cur += childs;
		*cur += 1;
		}
	if (f_v) {
		cout << "btree::SearchPage done, f_found=" << f_found << endl;
		}
	return f_found;
}

int btree::length(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int l;
	int j, pagelen;
	//Buffer *Root_BF;
	Buffer *BF;
	
	if (f_v) {
		cout << "btree::length" << endl;
		}
	if (!f_open()) {
		cout << "btree::length() not open" << endl;
		exit(1);
		}
#if 0
	Root_BF = RootBF + buf_idx();
	l = 0;
	pagelen = Root_BF->Page.NumItems;
	// cout << "root page length = " << pagelen << endl;
	for (j = 0; j <= pagelen; j++) { /* mit nulltem Index ! */
		l += Root_BF->Page.Item[j].Childs;
		}
	l += pagelen;
#else
	BF = new Buffer;
	
	if (f_v) {
		cout << "loading root page " << Root() << endl;
		}
	LoadPage(BF, Root(), verbose_level - 1);
	l = 0;
	pagelen = BF->Page.NumItems;
	// cout << "root page length = " << pagelen << endl;
	for (j = 0; j <= pagelen; j++) { /* mit nulltem Index ! */
		l += BF->Page.Item[j].Childs;
		}
	l += pagelen;
	
	delete BF;
#endif
	return l;
}

void btree::ith(int l, 
	KEYTYPE *key, DATATYPE *data, int verbose_level)
/* key muss auf einen ganzen KEYTYPE zeigen, 
 * da der gesamte keycarrier mittels struct-
 * zuweisung kopiert wird. */
{
	int f_v = (verbose_level >= 1);
	int cur, page, ref;
	int i, f_found;
	Buffer *buffer;
	
	if (f_v) {
		cout << "btree::ith searching for entry " << l << endl;
		}
	if (!f_open()) {
		cout << "btree::ith() not open" << endl;
		exit(1);
		}
	
	buffer = new Buffer;
	
	page = Root();
	cur = 0;
	while (TRUE) {
		LoadPage(buffer, page, verbose_level - 1);
		if (f_v) {
			cout << "btree::ith loaded page = " << page << endl;
			page_print(buffer, cout);
			}
			
		f_found = page_i_th(l, buffer, &cur, &i, verbose_level);
		
		if (f_found) {
			if (f_v) {
				cout << "found in " << i << endl;
				}
			*key = buffer->Page.Item[i].Key;
			*data = buffer->Page.Item[i].Data;
			break;
			}
		if (f_v) {
			cout << "not found" << endl;
			}
		ref = buffer->Page.Item[i].Ref;
		if (ref == 0) {
			cout << "btree::ith ref == 0" << endl;
			exit(1);
			}
		page = ref;
		}
	delete buffer;
	if (f_v) {
		cout << "btree::ith() done" << endl;
		}
}

int btree::page_i_th(int l, 
	Buffer *buffer, 
	int *cur, int *i, 
	int verbose_level)
// returns true iff element could be found
{
	int f_v = (verbose_level >= 1);
	int childs;
	int page_len, j;
	
	if (f_v) {
		cout << "btree::page_i_th looking for entry " << l << endl;
		}
	page_len = buffer->Page.NumItems;
	for (j = 0; j <= page_len; j++) {
		childs = buffer->Page.Item[j].Childs;
		if (*cur + childs > l) {
			*i = j;
			return FALSE;
			}
		if (*cur + childs == l) {
			if (j == page_len) {
				cout << "btree::page_i_th() j == page_len" << endl;
				page_print(buffer, cout);
				exit(1);
				}
			/* gefunden: (in j + 1) */
			*i = j + 1;
			return TRUE;
			}
		/* naechster Zweig: */
		*cur += childs + 1;
		}
	cout << "btree::page_i_th() not found" << endl;
	exit(1);
}


static KEYTYPE *IKpKey;
static DATATYPE *IKpData;
static Buffer *IKBF;
static int IKFound;
static int IKRisen;
static int f_keyadded;

void btree::insert_key(KEYTYPE *pKey, 
	DATATYPE *pData, int verbose_level)
/* Fuegt einen Schluessel in die Datenbank ein. Wenn
 * der Schluessel schon existiert, werden nur die Daten
 * ('Data') aktualisiert
 * --
 * KEYTYPE Key   - Einzufuegender Schluessel
 * DATATYPE Data - Die zum Schluessel gehoerenden Daten */
{
	int f_v = (verbose_level >= 1);
	int RootSplit;
	ItemTyp RootItem;
	Buffer *NewRoot = NULL;
	Buffer *BF1 = NULL;
	int NewNeighbourChilds, new_page_num;
	
	if (f_v) {
		cout << "btree::insert_key key=";
		key_print(pKey->c, cout);
		cout << endl;
		}
	if (!f_open()) {
		cout << "btree::insert_key() file not open" << endl;
		exit(1);
		}
	NewRoot = new Buffer;
	BF1 = new Buffer;
	fill_char((char *)NewRoot, sizeof(Buffer), 0);
	fill_char((char *)BF1, sizeof(Buffer), 0);
	f_keyadded = FALSE;
	IKpKey = pKey;
	IKpData = pData;
	RootSplit = FALSE;
	IKBF = BF1;
	if (f_v) {
		cout << "btree::insert_key: calling Update() Root=" << Root() << endl;
		}
		
	Update(Root(), 
		&RootSplit, 
		&RootItem, 
		&NewNeighbourChilds, 
		verbose_level);
	
	if (f_v) {
		if (RootSplit) {
			cout << "btree::insert_key RootSplit" << endl;
			}
		else {
			cout << "btree::insert_key not RootSplit" << endl;
			}
		}
	if (RootSplit) {
		if (f_v) {
			cout << "btree::insert_key RootSplit Item[1]=";
			key_print(RootItem.Key.c, cout);
			cout << endl;
			}
		new_page_num = AllocateRec(verbose_level);
		NewRoot->PageNum = (int_4) new_page_num;
		if (f_v) {
			cout << "btree::insert_key: RootSplit, NewRoot->PageNum = " << NewRoot->PageNum << endl;
			}

		NewRoot->Page.NumItems = 1;
		NewRoot->Page.Item[0].Ref = (int_4) Root();
		NewRoot->Page.Item[0].Childs = (int_4) NewNeighbourChilds;
		NewRoot->Page.Item[1] = RootItem;
		
		Root() = NewRoot->PageNum;
		if (f_v) {
			cout << "btree::insert_key Root=" << Root() << endl;
			page_print(NewRoot, cout);
			}
		

		if (f_v) {
			cout << "saving the new root" << endl;
			}
		SavePage(NewRoot, verbose_level - 1);
		//Root_BF = RootBF + buf_idx();
		//*Root_BF = *NewRoot;
#ifdef WRITE_INFO_ONLY_AT_END
#else
		WriteInfo(verbose_level);
#endif
		}
	delete NewRoot;
	delete BF1;
	IKBF = NULL;
	if (f_v) {
		cout << "btree::insert_key done" << endl;
		}
}

void btree::Update(int Node, int *Rise, 
	ItemTyp *RisenItem, int *RisenNeighbourChilds, 
	int verbose_level)
/* Einfuegen in den Zweig Node.
 * RisenNeighbourChilds nur gesetzt, wenn Rise TRUE ist. */
{
	int f_v = (verbose_level >= 1);
	int x, idx, z;

	if (f_v) {
		cout << "Update() in page " << Node << endl;
		}
	if (Node == 0) {
		/* Auf Blattebene angekommen wird von update()
		 * eingefuegt als wenn ein RisenItem eingefuegt 
		 * werden muesste. Deswegen wird hier diese
		 * Situation vorbereitet. */
		*Rise = TRUE;
		*RisenNeighbourChilds = 0;
		/* Nachbar darf ebenfalls keine Nachfolger haben */
		RisenItem->Key = *IKpKey;
		RisenItem->Data = *IKpData;
		RisenItem->Ref = 0;
		RisenItem->Childs = 0;
		f_keyadded = TRUE;
		return;
		}
	if (f_v) {
		cout << "Update() loading page " << Node << endl;
		}
	LoadPage(IKBF, Node, verbose_level);

	if (f_v) {
		cout << "Update() searching in page " << Node << endl;
		}
	IKFound = SearchPage(IKBF, 
		(void *)IKpKey, IKpData, 
		&idx, &x, 0, 
		verbose_level - 1);
	if (f_v) {
		if (IKFound) {
			cout << "key found at x=" << x << endl;
			}
		else {
			cout << "key not found, x=" << x << endl;
			}
		}
	if (IKFound && !f_duplicatekeys()) {
		if (f_v) {
			cout << "duplicate keys not allowed, we simply overwrite the Data" << endl;
			}
		IKBF->Page.Item[x].Data = *IKpData;
		SavePage(IKBF, verbose_level - 1);
		/* keine doppelten Schluessel erlaubt */
		/* Rise bleibt unveraendert FALSE */
		return;
		}
	/* Einfuegen in den Zweig von x: */
	IKRisen = FALSE;
	if (f_v) {
		cout << "Update() in page " << Node << " calling update in branch x=" << x << endl;
		}
	Update(IKBF->Page.Item[x].Ref, 
		&IKRisen, 
		RisenItem, RisenNeighbourChilds, 
		verbose_level);
	if (f_v) {
		cout << "Update() loading page " << Node << endl;
		}
	/* Neuladen der Seite, da der Buffer in der Rekursion
	 * benutzt wird. */
	LoadPage(IKBF, Node, verbose_level - 1);
	if (f_v) {
		page_print(IKBF, cout);
		}
	if (IKRisen) {
		if (f_v) {
			cout << "Update() in page " << Node << " IKRisen Item[" << x << "]=";
			key_print(RisenItem->Key.c, cout);
			cout << endl;
			}
		/* RisenItem muss nach x eingefuegt werden: */
		IKBF->Page.Item[x].Childs = (int_4) *RisenNeighbourChilds;
		/* Nach Seiten-Split hat der linke Nachbar weniger 
		 * Nachfolger */
		if (IKBF->Page.NumItems < BTREEMAXPAGESIZE) {
			// insert on this page:
			IKBF->Page.NumItems++;
			for (z = IKBF->Page.NumItems - 1; z >= x + 1; z--) {
				/* IKBF->Page.Item[z + 1] = IKBF->Page.Item[z]; */
				bt_item_copy(&IKBF->Page.Item[z], &IKBF->Page.Item[z + 1]);
				}
			/* IKBF->Page.Item[x + 1] = *RisenItem; */
			bt_item_copy(RisenItem, &IKBF->Page.Item[x + 1]);
			*Rise = FALSE;
			}
		else {
			// page full, split:
			if (f_v) {
				cout << "btree::Update() page is full, calling Split(), x = " << x << endl;
				}
			*RisenNeighbourChilds = 0; /* redundant */
			
			Split(IKBF, 
				RisenItem, x, 
				RisenNeighbourChilds, 
				verbose_level);
			
			*Rise = TRUE;
			}
		}
	else { /* IKRisen == FALSE */
		if (f_keyadded) {
			IKBF->Page.Item[x].Childs++;
			}
		}
	if (f_v) {
		cout << "btree::Update() saving old page " << IKBF->PageNum << endl;
		}
	SavePage(IKBF, verbose_level - 1);
}


void btree::Split(Buffer *BF, ItemTyp *Item, 
	int x, int *RisenNeighbourChilds, 
	int verbose_level)
/* Fuegt Item in volle Seite BF nach position x ein.
 * Die uebervolle Seite wird zerlegt, die 2. Haelfte in 
 * eine neue Seite kopiert. Das mittlere Element wird 
 * angehoben und bekommt die neue Seite als Nachfolger.
 * Der alte Nachfolger des mittleren Elements wird in die 
 * neue Seite an 0ter Position eingehaengt.
 * RisenNeighbourChilds wird als Summe der Datensatze der 
 * verkleinerten Seite berechnet und muss von der auf-
 * rufenden Funktion links neben Item eingetragen werden.
 * Die neu generierte Seite wird abgespeichert; die ver-
 * kleinerte muss von der rufenden Funktion gesichert 
 * werden. */
{
	int f_v = (verbose_level >= 1);
	ItemTyp SplitItem;
	Buffer *SplitBF = NULL;
	int sum1, sum2;
	int z;
	int new_page_num;
	
	if (f_v) {
		cout << "btree::Split page=" << BF->PageNum << " at x=" << x << endl;
		}
	if (f_v) {
		cout << "btree::Split original page:" << endl;
		page_print(BF, cout);
		}
	SplitBF = new Buffer;
	fill_char((char *)SplitBF, sizeof(Buffer), 0);
	new_page_num = AllocateRec(f_v);
	if (f_v) {
		cout << "btree::Split new page=" << new_page_num << endl;
		}
	SplitBF->PageNum = (int_4) new_page_num;
	if (x < BTREEHALFPAGESIZE) {
		SplitItem = BF->Page.Item[BTREEHALFPAGESIZE];
		for (z = BTREEHALFPAGESIZE - 1; z >= x + 1; z--) {
			BF->Page.Item[z + 1] = BF->Page.Item[z];
			}
		BF->Page.Item[x + 1] = *Item;
		}
	else {
		if (x > BTREEHALFPAGESIZE) {
			SplitItem = BF->Page.Item[BTREEHALFPAGESIZE + 1];
			for (z = BTREEHALFPAGESIZE + 2; z <= x; z++) {
				BF->Page.Item[z - 1] = BF->Page.Item[z];
				}
			BF->Page.Item[x] = *Item;
			}
		else {
			SplitItem = *Item;
			}
		}
	SplitBF->Page.Item[0].Ref = SplitItem.Ref;
	SplitBF->Page.Item[0].Childs = sum2 = SplitItem.Childs;
	sum1 = BF->Page.Item[0].Childs;
	for (z = 1; z <= BTREEHALFPAGESIZE; z++) {
		SplitBF->Page.Item[z] = BF->Page.Item[BTREEHALFPAGESIZE + z];
		sum2 += SplitBF->Page.Item[z].Childs;
		sum1 += BF->Page.Item[z].Childs;
		}
	sum1 += BTREEHALFPAGESIZE;
	sum2 += BTREEHALFPAGESIZE;
	BF->Page.NumItems = BTREEHALFPAGESIZE;
	SplitBF->Page.NumItems = BTREEHALFPAGESIZE;
	SplitItem.Ref = SplitBF->PageNum;
	SplitItem.Childs = (int_4) sum2;
	*Item = SplitItem;
	
	if (f_v) {
		cout << "btree::Split after split:" << endl;
		cout << "original page: " << BF->PageNum << endl;
		page_print(BF, cout);
		cout << "new page: " << SplitBF->PageNum << endl;
		page_print(SplitBF, cout);
		}
	
	if (f_v) {
		cout << "btree::Split saving new page:" << endl;
		}
	SavePage(SplitBF, verbose_level - 1);
	
	*RisenNeighbourChilds = sum1;
	if (f_v) {
		cout << "SplitItem:";
		key_print(Item->Key.c, cout);
		cout << endl;
		}
	
	delete SplitBF;
}

static int DKFound;
static int DKidx, DKcur;
/*static Buffer *DKBF;*/
static int f_keydeleted;

void btree::delete_ith(int idx, int verbose_level)
/* Loescht einen Datensatz in der Datenbank
 * --
 * DelKey - Zu loeschender Schluessel */
{
	int f_v = (verbose_level >= 1);
	int z, z2;
	int Underflow;
	/* Buffer BF; */
	Buffer *Root_BF;
	
	if (f_v) {
		cout << "btree::delete_ith idx=" << idx << endl;
		}
	if (!f_open()) {
		cout << "btree::delete_ith file not open" << endl;
		exit(1);
		}
	
	
	
	DKidx = idx;
	DKcur = 0;
	/* DKBF = &BF; */
	f_keydeleted = FALSE;
	Delete(Root(), Underflow, verbose_level);
#if 0
	Root_BF = RootBF + buf_idx();
#else
	Root_BF = new Buffer;
#endif
	if (f_v) {
		cout << "loading root page " << Root() << endl;
		}
	LoadPage(Root_BF, Root(), verbose_level - 1);
	
	if (Underflow && Root_BF->Page.NumItems == 0) {
		z = Root();
		z2 = Root_BF->Page.Item[0].Ref;
		if (z2 == 0) { /* leere Datenbank */
#ifdef WRITE_INFO_ONLY_AT_END
#else
			WriteInfo(FALSE /* f_v */);
#endif
			return;
			}
		LoadPage(Root_BF, z2, verbose_level - 1);
		
		ReleaseRec(z);
		Root() = Root_BF->PageNum; // !!! 
		
#ifdef WRITE_INFO_ONLY_AT_END
#else
		WriteInfo(FALSE /* f_v */);
#endif
		}
	if (f_v) {
		cout << "btree::delete_ith done" << endl;
		}
}

void btree::Delete(
		int Node, int& Underflow, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int x, y, z;
	Buffer *DKBF = NULL;

	if (f_v) {
		cout << "btree::Delete Node=" << Node << endl;
		}
	if (Node == 0) {
		Underflow = FALSE;
		return;
		}
	DKBF = new Buffer;
	fill_char((char *) DKBF, sizeof(Buffer), 0);
	LoadPage(DKBF, Node, verbose_level - 1);
	
	/*	SearchPage(p, DKBF, DKpDelKey, NIL, &idx, &x, &DKFound);*/
	
	DKFound = page_i_th(DKidx, DKBF, &DKcur, &x, verbose_level);
	if (DKFound) {
		y = DKBF->Page.Item[x - 1].Ref;
		if (y == 0) {
			DKBF->Page.NumItems--;
			Underflow = DKBF->Page.NumItems < BTREEHALFPAGESIZE;
			for (z = x; z <= DKBF->Page.NumItems; z++) {
				DKBF->Page.Item[z] = DKBF->Page.Item[z + 1];
				}
			SavePage(DKBF, verbose_level - 1);
			f_keydeleted = TRUE;
			}
		else {
			FindGreatest(y, 
				Underflow, DKBF, x, verbose_level);
			
			if (f_keydeleted) {
				DKBF->Page.Item[x - 1].Childs--;
				SavePage(DKBF, verbose_level - 1);
				}
			if (Underflow) {
				Compensate(Node, 
					y, 
					x - 1, 
					Underflow, 
					verbose_level);
				}
			}
		}
	else {
		y = DKBF->Page.Item[x].Ref;
		
		Delete(y, Underflow, verbose_level);
		
		if (f_keydeleted) {
			LoadPage(DKBF, Node, verbose_level - 1);
			DKBF->Page.Item[x].Childs--;
			SavePage(DKBF, verbose_level - 1);
			}
		if (Underflow) {
			Compensate(Node, 
				y, 
				x, 
				Underflow, 
				verbose_level);
			}
		}
	delete DKBF;
	if (f_v) {
		cout << "btree::Delete done" << endl;
		}
}

void btree::FindGreatest(int Node1, 
	int& Underflow, Buffer *DKBF, int x, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Node2;
	int NumBF;
	Buffer *buf = NULL;
	Buffer *Root_BF;

	if (f_v) {
		cout << "btree::FindGreatest Node1=" << Node1 << " x=" << x << endl;
		}
	buf = new Buffer;
	fill_char((char *)buf, sizeof(Buffer), 0);
	LoadPage(buf, Node1, verbose_level - 1);
	NumBF = buf->Page.NumItems;
	Node2 = buf->Page.Item[NumBF].Ref;
	if (Node2 != 0) {
		FindGreatest(Node2, Underflow, DKBF, x, verbose_level);
		if (f_keydeleted) {
			LoadPage(buf, Node1, verbose_level - 1);
			buf->Page.Item[NumBF].Childs--;
			SavePage(buf, verbose_level - 1);
			}
		if (Underflow) {
			Compensate(Node1, Node2, 
				NumBF, Underflow, 
				verbose_level);
			}
		}
	else {
		DKBF->Page.Item[x].Key = buf->Page.Item[NumBF].Key;
		DKBF->Page.Item[x].Data = buf->Page.Item[NumBF].Data;
		f_keydeleted = TRUE;
		if (DKBF->PageNum == Root()) {
			Root_BF = RootBF + buf_idx();
			*Root_BF = *DKBF;
			}
		NumBF--;
		Underflow = (NumBF < BTREEHALFPAGESIZE);
		buf->Page.NumItems = (int_4) NumBF;
		SavePage(buf, verbose_level - 1);
		}
	delete buf;
	if (f_v) {
		cout << "btree::FindGreatest done" << endl;
		}
}

void btree::Compensate(int Precedent, 
	int Node, int Path, int& Underflow, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Neighbour;
	int NumBF2, NumBF3;
	int x, z;
	int sum;
	Buffer *BF1 = NULL;
	Buffer *BF2 = NULL;
	Buffer *BF3 = NULL;
	
	if (f_v) {
		cout << "btree::Compensate Precedent=" << Precedent << " Node=" << Node << endl;
		}
	BF1 = new Buffer;
	BF2 = new Buffer;
	BF3 = new Buffer;
	fill_char((char *)BF1, sizeof(Buffer), 0);
	fill_char((char *)BF2, sizeof(Buffer), 0);
	fill_char((char *)BF3, sizeof(Buffer), 0);
	LoadPage(BF1, Node, verbose_level - 1);
	LoadPage(BF3, Precedent, verbose_level - 1);
	NumBF3 = BF3->Page.NumItems;
	if (f_v) {
		cout << "Path=" << Path << endl;
		cout << "NumBF3=" << NumBF3 << endl;
		}
	if (Path < NumBF3) {
		if (f_v) {
			cout << "rightmost leave with neighbor to the right" << endl;
			cout << "Path=" << Path << endl;
			cout << "NumBF3=" << NumBF3 << endl;
			}
		/* Blatt nicht rechts aussen, dh. ein rechter
		 * Nachbar existiert. */
		Neighbour = BF3->Page.Item[Path + 1].Ref;
		LoadPage(BF2, Neighbour, verbose_level - 1);
		NumBF2 = BF2->Page.NumItems;
		x = (NumBF2 + 1 - BTREEHALFPAGESIZE) / 2;
		BF1->Page.Item[BTREEHALFPAGESIZE].Key = BF3->Page.Item[Path + 1].Key;
		BF1->Page.Item[BTREEHALFPAGESIZE].Data = BF3->Page.Item[Path + 1].Data;
		BF1->Page.Item[BTREEHALFPAGESIZE].Ref = BF2->Page.Item[0].Ref;
		BF1->Page.Item[BTREEHALFPAGESIZE].Childs = sum = BF2->Page.Item[0].Childs;
		if (x > 0) {
			if (f_v) {
				cout << "case I with x=" << x << endl;
				}
			/*printf("Fall I x = %d\n", x);*/
			for (z = 1; z <= x - 1; z++) {
				BF1->Page.Item[BTREEHALFPAGESIZE + z] = BF2->Page.Item[z];
				sum += BF2->Page.Item[z].Childs;
				}
			sum += x;
			BF3->Page.Item[Path].Childs += sum;
			BF3->Page.Item[Path + 1].Childs -= sum;
			BF3->Page.Item[Path + 1].Key = BF2->Page.Item[x].Key;
			BF3->Page.Item[Path + 1].Data = BF2->Page.Item[x].Data;
			BF2->Page.Item[0].Ref = BF2->Page.Item[x].Ref;
			BF2->Page.Item[0].Childs = BF2->Page.Item[x].Childs;
			NumBF2 = NumBF2 - x;
			for (z = 1; z <= NumBF2; z++) {
				BF2->Page.Item[z] = BF2->Page.Item[x + z];
				}
			BF2->Page.NumItems = (int_4) NumBF2;
			BF1->Page.NumItems = (int_4) (BTREEHALFPAGESIZE + x - 1);
			SavePage(BF1, verbose_level - 1);
			SavePage(BF2, verbose_level - 1);
			SavePage(BF3, verbose_level - 1);
			Underflow = FALSE;
			}
		else {
			if (f_v) {
				cout << "case II with x=" << x << endl;
				}
			/*printf("Fall II x = %d\n", x);*/
			BF3->Page.Item[Path].Childs += BF3->Page.Item[Path + 1].Childs + 1;
			for (z = 1; z <= BTREEHALFPAGESIZE; z++) {
				BF1->Page.Item[BTREEHALFPAGESIZE + z] = BF2->Page.Item[z];
				}
			for (z = Path + 1; z <= NumBF3 - 1; z++) {
				BF3->Page.Item[z] = BF3->Page.Item[z + 1];
				}
			BF1->Page.NumItems = BTREEMAXPAGESIZE;
			BF3->Page.NumItems = (int_4) NumBF3 - 1L;
			Underflow = (NumBF3 <= BTREEHALFPAGESIZE);
			SavePage(BF1, verbose_level - 1);
			SavePage(BF3, verbose_level - 1);
			ReleaseRec(Neighbour);
			}
		}
	else {
		if (f_v) {
			cout << "rightmost leave with neighbor to the left" << endl;
			}
		/* Blatt rechts aussen; Nachbar links */
		Neighbour = BF3->Page.Item[Path - 1].Ref;
		LoadPage(BF2, Neighbour, verbose_level - 1);
		NumBF2 = BF2->Page.NumItems;
		x = (NumBF2 + 1 - BTREEHALFPAGESIZE) / 2;
		if (x > 0) {
			if (f_v) {
				cout << "case III with x=" << x << endl;
				}
			/*printf("Fall III x = %d\n", x);*/
			for (z = BTREEHALFPAGESIZE - 1L; z >= 1; z--) {
				BF1->Page.Item[z + x] = BF1->Page.Item[z];
				}
			BF1->Page.Item[x].Key = BF3->Page.Item[Path].Key;
			BF1->Page.Item[x].Data = BF3->Page.Item[Path].Data;
			BF1->Page.Item[x].Ref = BF1->Page.Item[0].Ref;
			BF1->Page.Item[x].Childs = BF1->Page.Item[0].Childs;
			NumBF2 = NumBF2 - x;
			BF1->Page.Item[0].Ref = BF2->Page.Item[NumBF2 + 1].Ref;
			BF1->Page.Item[0].Childs = sum =
				BF2->Page.Item[NumBF2 + 1].Childs;
			for (z = x - 1; z >= 1; z--) {
				BF1->Page.Item[z] = BF2->Page.Item[NumBF2 + 1 + z];
				sum += BF2->Page.Item[NumBF2 + 1 + z].Childs;
				}
			sum += x;
			BF3->Page.Item[Path].Key = BF2->Page.Item[NumBF2 + 1].Key;
			BF3->Page.Item[Path].Data = BF2->Page.Item[NumBF2 + 1].Data;
			BF3->Page.Item[Path].Childs += sum;
			BF3->Page.Item[Path - 1].Childs -= sum;
			BF2->Page.NumItems = (int_4) NumBF2;
			BF1->Page.NumItems = (int_4) (x + BTREEHALFPAGESIZE - 1L);
			SavePage(BF1, verbose_level - 1);
			SavePage(BF2, verbose_level - 1);
			SavePage(BF3, verbose_level - 1);
			Underflow = FALSE;
			}
		else {
			if (f_v) {
				cout << "case IV with x=" << x << endl;
				}
			/*printf("Fall IV x = %d\n", x);*/
			BF2->Page.Item[NumBF2 + 1].Key = BF3->Page.Item[Path].Key;
			BF2->Page.Item[NumBF2 + 1].Data = BF3->Page.Item[Path].Data;
			BF2->Page.Item[NumBF2 + 1].Ref = BF1->Page.Item[0].Ref;
			BF2->Page.Item[NumBF2 + 1].Childs = BF1->Page.Item[0].Childs;
			for (z = 1; z <= BTREEHALFPAGESIZE - 1; z++) {
				BF2->Page.Item[NumBF2 + 1 + z] = BF1->Page.Item[z];
				}
			BF2->Page.NumItems = BTREEMAXPAGESIZE;
			BF3->Page.NumItems = (int_4) NumBF3 - 1;
			BF3->Page.Item[Path - 1].Childs += 
				BF3->Page.Item[Path].Childs + 1;
			Underflow = (NumBF3 <= BTREEHALFPAGESIZE);
			SavePage(BF2, verbose_level - 1);
			SavePage(BF3, verbose_level - 1);
			ReleaseRec(Node);
			}
		}
	delete BF1;
	delete BF2;
	delete BF3;
	if (f_v) {
		cout << "btree::Compensate done" << endl;
		}
}

void btree::print_all(ostream& ost)
{
	int verbose_level = 0;
	
	int bt_len = length(verbose_level);
	
	print_range(0, bt_len, ost);
}

void btree::print_range(int first, int len, std::ostream& ost)
{
	int i, j, bt_len;
	KEYTYPE key;
	DATATYPE data;
	int verbose_level = 0;
	
	bt_len = length(verbose_level);
	for (i = 0; i < len; i++) {
		j = first + i;
		if (j < 0) {
			continue;
			}
		if (j >= bt_len) {
			continue;
			}
		ith(j, &key, &data, verbose_level);
		cout << j << " : ";
		key_print(key.c, ost);
		cout << endl;
		}
}

void btree::print_page(int x, std::ostream& ost)
{
	int y;
	Buffer BF;
	int verbose_level = 0;

	if (x == 0) {
		return;
		}
	LoadPage(&BF, x, verbose_level);
	ost << "page " << x << ":\n";
	page_print(&BF, ost);
	for (y = 0; y <= BF.Page.NumItems; y++) {
		print_page(BF.Page.Item[y].Ref, ost);
		}
}

void btree::page_print(Buffer *BF, std::ostream& ost)
{
	int i, len; //, childs, ref, datref, data_size;
	ItemTyp *item = NULL;
	
	ost << "PageNum = " << BF->PageNum << endl;
	len = BF->Page.NumItems;
	ost << "BF->Page.NumItems = " << len << endl;
	for (i = 0; i <= len; i++) {
		// ostrstream s;
		
		item = &BF->Page.Item[i];
		
		item_print(item, i, ost);
		
#if 0
		childs = item->Childs;
		ref = item->Ref;
		ost << "item " << i << ": Childs=" << childs << " Ref=" << ref; 
		if (i != 0) {
			datref = item->Data.datref;
			data_size = item->Data.data_size;
			ost << " (" << datref << "/" << data_size << "): ";
			key_print(item->Key.c, ost);
			// bt_key_print(item->Key.c, key(), s);
			// s << ends;
			// ost << s;
			}
		ost << endl;
#endif
		}
	ost << endl;
}

void btree::item_print(ItemTyp *item, int i, std::ostream& ost)
{
	int childs, ref, datref, data_size;
	
	childs = item->Childs;
	ref = item->Ref;

	ost << setw(3) << i << ": ";
	
	if (i != 0) {
		key_print(item->Key.c, ost);

		datref = item->Data.datref;
		data_size = item->Data.data_size;
		ost << " (" << setw(10) << datref 
			<< "/" << setw(6) << data_size << "): ";
		// bt_key_print(item->Key.c, key(), s);
		// s << ends;
		// ost << s;
		}

	ost << "_{" << setw(8) << childs 
		<< "," << setw(8) << ref << "}" << endl; }


// #############################################################################
// global functions and low level functions
// #############################################################################


void btree::file_open()
{
	int idx = fstream_table_get_free_entry();
	fstream *f = new fstream(filename().s(), ios::in | ios::out | ios::binary);
	fstream_table[idx] = f;
	fstream_table_used[idx] = TRUE;
	stream() = idx;
	f_open() = TRUE;
	// cout << "btree::file_open() file " << filename().s() << " opened" << endl;
}

void btree::file_create()
{
	hollerith cmd;
	
	cmd.init("rm ");
	cmd.append(filename().s());
	system(cmd.s());
	
	int idx = fstream_table_get_free_entry();
	{
	fstream *f = new fstream(filename().s(), ios::out | ios::binary);
	if (!*f) {
		cout << "btree::file_create() file " << filename().s() << " could not be created" << endl;
		exit(1);
		}
	f->close();
	delete f;
	}
	fstream *f = new fstream(filename().s(), ios::in | ios::out | ios::binary);
	if (!*f) {
		cout << "btree::file_create() file " << filename().s() << " could not be opened" << endl;
		exit(1);
		}
	fstream_table[idx] = f;
	fstream_table_used[idx] = TRUE;
	stream() = idx;
	f_open() = TRUE;
	// cout << "btree::file_create() file " << filename().s() << " created" << endl;
}

void btree::file_close()
{
	int idx = (int) stream();
	if (!fstream_table_used[idx]) {
		cout << "btree::file_close() !fstream_table_used[idx]" << endl;
		cout << "idx=" << idx << endl;
		exit(1);
		}
	delete fstream_table[idx];
	fstream_table_used[idx] = FALSE;
	stream() = 0;
	f_open() = FALSE;
	// cout << "btree::file_close() file " << filename().s() << " closed" << endl;
}

void btree::file_write(PageTyp *page, const char *message)
{
	if (!f_open()) {
		cout << "btree::file_write() file not open" << endl;
		exit(1);
		}
	int idx = (int) stream();
	//cout << "file_write idx=" << idx << " " << message << endl;
	if (!fstream_table_used[idx]) {
		cout << "btree::file_write() !fstream_table_used[idx]" << endl;
		cout << "idx=" << idx << endl;
		exit(1);
		}
	fstream_table[idx]->write((char *) page, sizeof(PageTyp));
}

void btree::file_read(PageTyp *page, const char *message)
{
	if (!f_open()) {
		cout << "btree::file_read() file not open" << endl;
		exit(1);
		}
	int idx = (int) stream();
	//cout << "file_read idx=" << idx << " " << message << endl;
	if (!fstream_table_used[idx]) {
		cout << "btree::file_read() !fstream_table_used[idx]" << endl;
		cout << "idx=" << idx << endl;
		exit(1);
		}
	fstream_table[idx]->read((char *) page, sizeof(PageTyp));
}

void btree::file_seek(int page_no)
{
	int offset;
	
	//cout << "file_seek page " << page_no << endl;
	if (!f_open()) {
		cout << "btree::file_seek() file not open" << endl;
		exit(1);
		}
	offset = page_no * (int)sizeof(PageTyp);
	int idx = (int) stream();
	if (!fstream_table_used[idx]) {
		cout << "btree::file_seek() !fstream_table_used[idx]" << endl;
		cout << "idx=" << idx << endl;
		exit(1);
		}
	fstream_table[idx]->seekg(offset);
}



static void bt_item_copy(ItemTyp *a, ItemTyp *b)
{
	int i, len;
	char *pca = (char *)a;
	char *pcb = (char *)b;
	
	len = sizeof(ItemTyp);
	for (i = 0; i < len; i++)
		pcb[i] = pca[i];
}

}}}


