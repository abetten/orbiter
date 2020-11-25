// database.cpp
//
// Anton Betten
// 27.11.2000
// moved from D2 to ORBI Nov 15, 2007

#include "orbiter.h"

#include <stdlib.h> // for system


using namespace std;


namespace orbiter {
namespace discreta {


database::database() : Vector()
{
	k = DATABASE;
}

database::database(const discreta_base &x)
	// copy constructor:    this := x
{
	cout << "database::copy constructor for object: "
			<< const_cast<discreta_base &>(x) << "\n";
	const_cast<discreta_base &>(x).copyobject_to(*this);
}

database& database::operator = (const discreta_base &x)
	// copy assignment
{
	cout << "database::operator = (copy assignment)" << endl;
	copyobject(const_cast<discreta_base &>(x));
	return *this;
}

void database::settype_database()
{
	OBJECTSELF s;
	
	s = self;
	new(this) database;
	self = s;
	k = DATABASE;
}

database::~database()
{
	freeself_database();
}

void database::freeself_database()
{
	// cout << "group_selection::freeself_database()\n";
	freeself_vector();
}

kind database::s_virtual_kind()
{
	return DATABASE;
}

void database::copyobject_to(discreta_base &x)
{
#ifdef COPY_VERBOSE
	cout << "database::copyobject_to()\n";
	print_as_vector(cout);
#endif
	Vector::copyobject_to(x);
	x.as_database().settype_database();
#ifdef COPY_VERBOSE
	x.as_database().print_as_vector(cout);
#endif
}

ostream& database::print(ostream& ost)
{
	
	return ost;
}

void database::init(const char *filename, int objectkind, int f_compress)
{
	init_with_file_type(filename, objectkind, f_compress, DB_FILE_TYPE_STANDARD);
}

void database::init_with_file_type(const char *filename, 
	int objectkind, int f_compress, int file_type)
{
	m_l(8);
	c_kind(DATABASE);
	s_i(0).change_to_vector();
	s_i(1).change_to_hollerith();
	btree_access().m_l(0);
	database::filename().init(filename);
	database::objectkind() = objectkind;
	database::f_compress() = f_compress;
	f_open() = FALSE;
	database::file_type() = file_type;
	file_size() = 0;
}

void database::create(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	char *buf;
	
	if (f_v) {
		cout << "database::create" << endl;
		}
	buf = NEW_char(size_of_header());
	file_create(0 /*verbose_level - 1*/);
	file_size() = size_of_header();
	for (i = 0; i < size_of_header(); i++)
		buf[i] = 0;
	file_seek(0);
	file_write(buf, size_of_header(), 1);
	put_file_size();
	for (i = 0; i < btree_access().s_l(); i++) {
		btree_access_i(i).create(verbose_level - 1);
		}
	FREE_char(buf);
}

void database::open(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "database::open, verbose_level=" << verbose_level << endl;
		}
	file_open(0 /*verbose_level - 1*/);
	for (i = 0; i < btree_access().s_l(); i++) {
		btree_access_i(i).open(verbose_level - 1);
		}
}

void database::close(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "database::close" << endl;
		}
	put_file_size();
	file_close(0 /*verbose_level - 1*/);
	for (i = 0; i < btree_access().s_l(); i++) {
		btree_access_i(i).close(verbose_level);
		}
}

void database::delete_files()
{
	hollerith cmd;
	int i;
	
	cmd.init("rm ");
	cmd.append(filename().s());
	system(cmd.s());
	for (i = 0; i < btree_access().s_l(); i++) {
		cmd.init("rm ");
		cmd.append(btree_access_i(i).filename().s());
		system(cmd.s());
		}
}

void database::put_file_size()
{
	int_4 l, l1;
	int f_v = FALSE;
	os_interface Os;

	if (f_v) {
		cout << "database::put_file_size" << endl;
		}
	file_seek(DB_POS_FILESIZE);
	l = l1 = (int_4) file_size();
	Os.block_swap_chars((char *)&l, sizeof(int_4), 1);
	file_write(&l, 4, 1);
}

void database::get_file_size()
{
	int_4 l;
	int f_v = FALSE;
	os_interface Os;

	if (f_v) {
		cout << "database::get_file_size" << endl;
		}
	file_seek(DB_POS_FILESIZE);
	file_read(&l, 4, 1);
	Os.block_swap_chars((char *)&l, sizeof(int_4), 1);
	file_size() = l;
}

void database::user2total(int user, int &total, int &pad)
{
	int r, r1, sz;
	
	sz = size_of_header();
	if (file_type() == DB_FILE_TYPE_STANDARD) {
		r = user % sz;
		if (r != 0) {
			r1 = sz - r;
			}
		else {
			r1 = 0;
			}
		pad = r1 + sz;
		total = sz + sz + user + pad;
		}
	else if (file_type() == DB_FILE_TYPE_COMPACT) {
		r = user % sz;
		pad = sz - r;
		total = sizeof(int_4) * 2 + user + pad;
		}
}

int database::size_of_header()
{
	if (file_type() == DB_FILE_TYPE_STANDARD) {
		return 16;
		}
	else if (file_type() == DB_FILE_TYPE_COMPACT) {
		return 8;
		}
	else {
		cout << "database::size_of_header() unknown file_type" << endl;
		cout << "file_type()=" << file_type() << endl;
		exit(1);
		}
}

int database::size_of_header_log()
{
	if (file_type() == DB_FILE_TYPE_STANDARD) {
		return 4;
		}
	else if (file_type() == DB_FILE_TYPE_COMPACT) {
		return 3;
		}
	else {
		cout << "database::size_of_header_log() unknown file_type" << endl;
		cout << "file_type()=" << file_type() << endl;
		exit(1);
		}
	
}

void database::add_object_return_datref(Vector &the_object, uint_4 &datref, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	KEYTYPE *key_type = NULL;
	DATATYPE data_type;

	if (f_v) {
		cout << "database::add_object_return_datref" << endl;
		}
	if (!f_open()) {
		cout << "database::add_object_return_datref() database not open" << endl;
		exit(1);
		}
	key_type = new KEYTYPE;
	if (the_object.s_kind() != objectkind()) {
		cout << "database::add_object_return_datref() wrong kind of object" << endl;
		exit(1);
		}
	
	memory M;

	if (FALSE) {
		cout << "database::add_object_return_datref(): packing object" << endl;
		}
	the_object.pack(M, FALSE, 0/*debug_depth*/);

	if (f_compress()) {
		if (FALSE) {
			cout << "database::add_object_return_datref(): compressing object" << endl;
			}
		M.compress(FALSE);
		}
	int i, size;
	//uint4 datref;
	char *pc;
	size = M.used_length();
	pc = (char *) M.self.char_pointer;
	if (FALSE) {
		cout << "database::add_object_return_datref(): saving data via add_data_DB()" << endl;
		}
	add_data_DB((void *)pc, size, &datref, verbose_level - 4);
	if (FALSE) {
		cout << "finished with add_data_DB()" << endl;
		}
	data_type.datref = datref;
	data_type.data_size = (uint_4) size;

	for (i = 0; i < btree_access().s_l(); i++) {
		btree & bt = btree_access_i(i);
		bt.key_fill_in(key_type->c, the_object);
		
		if (f_vv) {
			cout << "database::add_object_return_datref(): calling insert_key for btree #" << i << ": ";
			bt.key_print(key_type->c, cout);
			cout << endl;
			}
		bt.insert_key(key_type, &data_type, verbose_level - 2);
		if (f_vv) {
			cout << "database::add_object_return_datref(): after insert_key for btree #" << i << endl;
			}
		}

	delete key_type;
	if (f_v) {
		cout << "database::add_object_return_datref done" << endl;
		}
}

void database::add_object(Vector &the_object, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	uint_4 datref;

	if (f_v) {
		cout << "database::add_object" << endl;
		}
	add_object_return_datref(the_object, datref, verbose_level);
	if (f_v) {
		cout << "database::add_object done" << endl;
		}
}

void database::delete_object(Vector& the_object, 
	uint_4 datref, int verbose_level)
{
	int i, j, len;
	int idx;
	KEYTYPE key_type;
	DATATYPE data_type;
	
	if (!f_open()) {
		cout << "database::delete_object() database not open" << endl;
		exit(1);
		}
	if (the_object.s_kind() != objectkind()) {
		cout << "database::delete_object() wrong kind of object" << endl;
		exit(1);
		}
	int size = get_size_from_datref(datref, verbose_level - 1);
	data_type.datref = datref;
	data_type.data_size = (uint_4) size;
	len = btree_access().s_l();
	for (i = 0; i < len; i++) {
		for (j = 0; j < BTREEMAXKEYLEN; j++) {
			key_type.c[j] = 0;
			}
		btree & bt = btree_access_i(i);
		bt.key_fill_in(key_type.c, the_object);
		if (!bt.search(key_type.c, &data_type, &idx, 0, verbose_level)) {
			cout << "database::delete_object() WARNING: btree entry not found" << endl;
			continue;
			}
		bt.delete_ith(idx, verbose_level);
		}
	free_data_DB(datref, size, verbose_level - 2);
}

void database::get_object(uint_4 datref,
	Vector &the_object, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int size;
	DATATYPE data_type;
	
	if (f_v) {
		cout << "database::get_object" << endl;
		}
	size = get_size_from_datref(datref, verbose_level - 1);
	data_type.datref = datref;
	data_type.data_size = (uint_4) size;
	get_object(&data_type, the_object, verbose_level - 1);
}

void database::get_object(DATATYPE *data_type, Vector &the_object, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int size, total = 0, pad, i;
	os_interface Os;
	
	if (f_v) {
		cout << "database::get_object" << endl;
		}
	if (!f_open()) {
		cout << "database::get_object(data_type) database not open" << endl;
		exit(1);
		}
	size = data_type->data_size;
	user2total(size, total, pad);
	memory M;
	
	M.alloc(size);
	char *pc = M.self.char_pointer;
	char *d = NULL;
	char *pc1;
	
	d = new char[total];
	
	file_seek(((uint)data_type->datref) << size_of_header_log());
	file_read(d, 1, total);
	
	if (file_type() == DB_FILE_TYPE_STANDARD) {
		int_4 *header = (int_4 *) d;
		int_4 *header2 = header + 4;
		
		pc1 = d + 8 * 4;
		Os.block_swap_chars((char *)header, sizeof(int_4), 4);
		Os.block_swap_chars((char *)header2, sizeof(int_4), 4);
		if (header[0] != MAGIC_SYNC) {
			cout << "database::get_object()|header: no MAGIC_SYNC" << endl;
			cout << "data_type->datref=" << data_type->datref << endl;
			exit(1);
			}
		if (!header[1]) {
			cout << "database::get_object()|header: data is not used" << endl;
			exit(1);
			}
		if (header[2] != size) {
			cout << "database::get_object()|header: header[2] != size" << endl;
			exit(1);
			}
		if (header[3] != total) {
			cout << "database::get_object()|header: header[3] != total" << endl;
			exit(1);
			}
		if (header2[0] != MAGIC_SYNC) {
			cout << "database::get_object()|header2: no MAGIC_SYNC" << endl;
			exit(1);
			}
		if (!header2[1]) {
			cout << "database::get_object()|header2: data is not used" << endl;
			exit(1);
			}
		if (header2[2] != size) {
			cout << "database::get_object()|header2: header[2] != size" << endl;
			exit(1);
			}
		if (header2[3] != total) {
			cout << "database::get_object()|header2: header[3] != total" << endl;
			exit(1);
			}
		}
	else if (file_type() == DB_FILE_TYPE_COMPACT) {
		int_4 *header = (int_4 *) d;
		
		pc1 = d + 4 * 2;
		Os.block_swap_chars((char *)header, sizeof(int_4), 2);
		if (header[0] != MAGIC_SYNC) {
			cout << "database::get_object()|header: no MAGIC_SYNC" << endl;
			cout << "data_type->datref=" << data_type->datref << endl;
			exit(1);
			}
		if (header[1] != size) {
			cout << "database::get_object()|header: header[1] != size" << endl;
			exit(1);
			}
		}
	else {
		cout << "database::get_object() unknown file_type" << endl;
		cout << "file_type()=" << file_type() << endl;
		exit(1);
		}
	for (i = 0; i < size; i++)
		pc[i] = pc1[i];
	// M.alloc_length();
	M.used_length() = size;
	if (f_compress()) {
		if (f_vv) {
			cout << "database::get_object(): decompressing object" << endl;
			}
		M.decompress(f_vv);
		}
	M.cur_pointer() = 0;
	
	the_object.freeself();
	the_object.unpack(M, FALSE, 0/*debug_depth*/);
	
	delete [] d;
}

void database::get_object_by_unique_int4(int btree_idx, int id, 
	Vector& the_object, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	btree& B = btree_access_i(btree_idx);
	int datref;
	
	if (f_v) {
		cout << "database::get_object_by_unique_int4 calling search_datref_of_unique_int4" << endl;
		}
	datref = B.search_datref_of_unique_int4(id, verbose_level - 1);
	if (f_v) {
		cout << "datref=" << datref << " calling get_object" << endl;
		}
	get_object((uint_4) datref, the_object, verbose_level - 1);
}

int database::get_object_by_unique_int4_if_there(int btree_idx, int id, 
	Vector& the_object, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	btree& B = btree_access_i(btree_idx);
	int datref;
	
	if (f_v) {
		cout << "database::get_object_by_unique_int4 calling search_datref_of_unique_int4" << endl;
		}
	datref = B.search_datref_of_unique_int4_if_there(id, verbose_level - 1);
	if (f_v) {
		cout << "datref=" << datref << endl;
		}
	if (datref == -1)
		return FALSE;
	if (f_v) {
		cout << "calling get_object" << endl;
		}
	get_object((uint_4) datref, the_object, verbose_level - 1);
	return TRUE;
}

int database::get_highest_int4(int btree_idx)
{
	btree & B = btree_access_i(btree_idx);
	
	return B.get_highest_int4();
}

void database::ith_object(int i, int btree_idx, 
	Vector& the_object, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	KEYTYPE key_type;
	DATATYPE data_type;
	
	if (f_v) {
		cout << "database::ith_object i=" << i << endl;
		}
	ith(i, btree_idx, &key_type, &data_type, verbose_level);
	get_object(&data_type, the_object, verbose_level - 1);
}

void database::ith(int i, int btree_idx, 
	KEYTYPE *key_type, DATATYPE *data_type, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "database::ith i=" << i << endl;
		}
	btree & bt = btree_access_i(btree_idx);
	bt.ith(i, key_type, data_type, verbose_level);
}

void database::print_by_btree(int btree_idx, ostream& ost)
{
	btree&B = btree_access_i(btree_idx);
	int i, len;
	Vector the_object;
	int verbose_level = 0;
	
	open(verbose_level);
	len = B.length(verbose_level);
	cout << "database " << filename().s() << ", btree " << B.filename().s() << " of length " << len << endl;
	for (i = 0; i < len; i++) {
		ith_object(i, btree_idx, the_object, verbose_level);
		cout << i << " : " << the_object << endl;
		}
	close(verbose_level);
}

void database::print_by_btree_with_datref(int btree_idx, ostream& ost)
{
	btree &B = btree_access_i(btree_idx);
	int i, len;
	Vector the_object;
	int verbose_level = 0;
	KEYTYPE key_type;
	DATATYPE data_type;
	
	open(verbose_level);
	len = B.length(verbose_level);
	cout << "database " << filename().s() << ", btree " << B.filename().s() << " of length " << len << endl;
	for (i = 0; i < len; i++) {
		B.ith(i, &key_type, &data_type, verbose_level);
		get_object(&data_type, the_object, verbose_level - 1);
		cout << i << " : " 
			<< data_type.datref << " " 
			<< data_type.data_size << " " 
			<< the_object << endl;
		}
	close(verbose_level);
}

void database::print_subset(Vector& datrefs, ostream& ost)
{
	int i, len;
	Vector the_object;
	int verbose_level = 0;
	
	len = datrefs.s_l();
	for (i = 0; i < len; i++) {
		get_object((uint_4) datrefs.s_ii(i), the_object, verbose_level);
		cout << i << " : " << the_object << endl;
		}
}

void database::extract_subset(Vector& datrefs, char *out_path, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 1);
	int i, len;
	Vector the_object;
	
	if (f_v) {
		cout << "database::extract_subset()" << endl;
		}
	database D;
	
	design_parameter p;

	p.init_database(D, out_path);
	D.create(verbose_level - 1);


	len = datrefs.s_l();
	if (f_v) {
		cout << "copying " << len << " datasets into database " << out_path << endl;
		}
	for (i = 0; i < len; i++) {
		if (f_v && !f_vv) {
			cout << i << " ";
			if ((i % 10) == 0)
				cout << endl;
			}
		get_object((uint_4) datrefs.s_ii(i), the_object, verbose_level - 2);
		if (f_vv) {
			cout << i << " : " << the_object << endl;
			}
		D.add_object(the_object, verbose_level - 2);
		}
	D.close(verbose_level - 1);
}

void database::search_int4(int btree_idx, int imin, int imax, 
	Vector &datrefs, int verbose_level)
{
	Vector Btree_idx, Imin, Imax;
	
	Btree_idx.m_l(1);
	Imin.m_l(1);
	Imax.m_l(1);
	Btree_idx.m_ii(0, btree_idx);
	Imin.m_ii(0, imin);
	Imax.m_ii(0, imax);
	search_int4_multi_dimensional(Btree_idx, Imin, Imax, datrefs, verbose_level);
}

void database::search_int4_2dimensional(int btree_idx0, int imin0, int imax0, 
	int btree_idx1, int imin1, int imax1, 
	Vector &datrefs, int verbose_level)
{
	Vector Btree_idx, Imin, Imax;
	
	Btree_idx.m_l(2);
	Imin.m_l(2);
	Imax.m_l(2);
	Btree_idx.m_ii(0, btree_idx0);
	Imin.m_ii(0, imin0);
	Imax.m_ii(0, imax0);
	Btree_idx.m_ii(1, btree_idx1);
	Imin.m_ii(1, imin1);
	Imax.m_ii(1, imax1);
	search_int4_multi_dimensional(Btree_idx, Imin, Imax, datrefs, verbose_level);
}

void database::search_int4_multi_dimensional(Vector& btree_idx, 
	Vector& i_min, Vector &i_max, Vector& datrefs, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, l, bi, imin, imax, first, len, db_length;
	Vector v1, v2, First, Len, Len_sorted;
	
	if (!f_open()) {
		cout << "database::search_int4_multi_dimensional() database not open" << endl;
		exit(1);
		}
	datrefs.m_l(0);
	l = btree_idx.s_l();
	First.m_l_n(l);
	Len.m_l_n(l);
	for (i = 0; i < l; i++) {
		bi = btree_idx.s_ii(i);
		imin = i_min.s_ii(i);
		imax = i_max.s_ii(i);
		if (f_v) {
			cout << "database::search_int4_multi_dimensional() i=" << i 
				<< " bi=" << bi << " imin=" << imin << " imax=" << imax << endl;
			}
		btree &B = btree_access_i(bi);
		B.search_interval_int4(imin, imax, first, len, verbose_level - 1);
		if (f_v) {
			cout << "after search_interval_int4() first = " << first << " len=" << len << endl;
			}
		if (len == 0)
			return;
		First.m_ii(i, first);
		Len.m_ii(i, len);
		}
	if (f_v) {
		cout << "First = " << First << endl;
		cout << "Len = " << Len << endl;
		}
	permutation p;
	
	Len_sorted = Len;
	Len_sorted.sort_with_logging(p);
	if (f_v) {
		cout << "Len (sorted) = " << Len_sorted << endl;
		}
	int j, h, l2;
	j = p[0];
	bi = btree_idx.s_ii(j);
	btree &B = btree_access_i(bi);
	db_length = B.length(verbose_level);
	B.get_datrefs(First.s_ii(j), Len.s_ii(j), v1);
	v1.sort();
	if (f_v) {
		cout << "db_length = " << db_length << endl;
		cout << "datrefs after 0-th btree " << j << " : " << v1 << " (length=" << v1.s_l() << ")" << endl;
		}
	if (f_vv) {
		print_subset(v1, cout);
		}
	for (i = 1; i < l; i++) {
		KEYTYPE key;
		DATATYPE data;
		int datref, idx;
		integer datref_object;
		
		v2.m_l_n(v1.s_l());
		l2 = 0;
		j = p[i];
		bi = btree_idx.s_ii(j);
		first = First.s_ii(j);
		len = Len.s_ii(j);
		if (len == db_length) {
			if (f_v) {
				cout << i << "th-btree selects all datasets, no restriction here." << endl;
				}
			continue;
			}
		btree &B = btree_access_i(bi);
		for (h = 0; h < len; h++) {
			B.ith(first + h, &key, &data, verbose_level - 1);
			datref = data.datref;
			datref_object.m_i(datref);
			if (v1.search(datref_object, &idx)) {
				v2.m_ii(l2++, datref);
				}
			}
		v2.realloc(l2);
		v2.sort();
		v1.swap(v2);
		if (f_v) {
			cout << "datrefs after " << i << "th-btree " << j << " : " << v1 << " (length=" << v1.s_l() << ")" << endl;
			}
		if (f_vv) {
			print_subset(v1, cout);
			}
		}
	v1.swap(datrefs);
	if (f_v) {
		print_subset(datrefs, cout);
		}
}


int database::get_size_from_datref(uint_4 datref, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int size;
	int_4 *header = NULL;
	os_interface Os;
	
	if (f_v) {
		cout << "database::get_size_from_datref" << endl;
		}
	if (!f_open()) {
		cout << "database::get_size_from_datref() database not open" << endl;
		exit(1);
		}
	if (file_type() == DB_FILE_TYPE_STANDARD) {
		header = new int_4[8];
		file_seek(((uint)datref) << size_of_header_log());
		file_read((char *)header, 1, 8 * 4);
		Os.block_swap_chars((char *)header, sizeof(int_4), 8);
		if (header[0] != MAGIC_SYNC) {
			cout << "database::get_size_from_datref()|header: no MAGIC_SYNC, probably the datref is wrong" << endl;
			cout << "datref=" << datref << endl;
			exit(1);
			}
		if (!header[1]) {
			cout << "database::get_size_from_datref()|header: data is not used" << endl;
			exit(1);
			}
		size = header[2];
		delete [] header;
		}
	else if (file_type() == DB_FILE_TYPE_COMPACT) {
		header = new int_4[2];
		file_seek(((uint)datref) << size_of_header_log());
		file_read((char *)header, 1, 4 * 2);
		Os.block_swap_chars((char *)header, sizeof(int_4), 2);
		if (header[0] != MAGIC_SYNC) {
			cout << "database::get_size_from_datref()|header: no MAGIC_SYNC, probably the datref is wrong" << endl;
			cout << "datref=" << datref << endl;
			exit(1);
			}
		size = header[1];
		delete [] header;
		}
	else {
		cout << "database::size_of_header() unknown file_type" << endl;
		cout << "file_type()=" << file_type() << endl;
		exit(1);
		}
	if (f_v) {
		cout << "database::get_size_from_datref size = " << size << endl;
		}
	
	return size;
}

void database::add_data_DB(void *d, 
	int size, uint_4 *datref, int verbose_level)
{
	if (file_type() == DB_FILE_TYPE_STANDARD) {
		add_data_DB_standard(d, size, datref, verbose_level);
		}
	else if (file_type() == DB_FILE_TYPE_COMPACT) {
		add_data_DB_compact(d, size, datref, verbose_level);
		}
}

void database::add_data_DB_standard(void *d, 
	int size, uint_4 *datref, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int total = 0, pad = 0;
	char *data2 = NULL;
	char *pc, *pc0;
	int i;
	int_4 *pi;
	int_4 header[4];
	int_4 new_header[4];
		/* 0: SYNC
		 * 1: f_used
		 * 2: length user data
		 * 3: total length (header inclusive), 
		 *    a multiple of 16, 
		 *    one unused full 16 char block guaranteed.
		 */
	int old_file_size;
	os_interface Os;
	
	if (f_v) {
		cout << "database::add_data_DB_standard()" << endl;
		}
	user2total(size, total, pad);
	data2 = (char *) new char[total];
	header[0] = MAGIC_SYNC;
	header[1] = TRUE;
	header[2] = (int_4) size;
	header[3] = (int_4) total;
	Os.block_swap_chars((char *)header, sizeof(int_4), 4);
	pi = (int_4 *)data2;
	pi[0] = header[0];
	pi[1] = header[1];
	pi[2] = header[2];
	pi[3] = header[3];
	pi[4] = header[0];
	pi[5] = header[1];
	pi[6] = header[2];
	pi[7] = header[3];
	Os.block_swap_chars((char *)header, sizeof(int_4), 4);
		// swap header back, there will be another test 
	pc = (char *)(pi + 8);
	pc0 = (char *)d;
	if (f_vv) {
		cout << "size = " << size << " pad = " << pad
				<< " total = " << total << endl;
		}
	for (i = 0; i < size; i++)
		pc[i] = pc0[i];
	for (i = 0; i < pad; i++)
		pc[size + i] = 0;
	old_file_size = file_size();
	if (old_file_size <= 0) {
		cout << "database::add_data_DB_standard old_file_size <= 0" << endl;
		exit(1);
		}
	file_seek(old_file_size);
	file_write(data2, 1, total);
	*datref = (uint_4)(old_file_size >> size_of_header_log());
	if (((int)((uint)*datref << size_of_header_log())) != old_file_size) {
		cout << "database::add_data_DB_standard ((uint)*datref << size_of_header_log()) != old_file_size" << endl;
		cout << "old_file_size=" << old_file_size << endl;
		cout << "*datref=" << *datref << endl;
		cout << "size_of_header_log()=" << size_of_header_log() << endl;
		exit(1);
		}
	file_size() += total;
	//put_file_size();
	
	file_seek(old_file_size);
	file_read(new_header, 4, 4);
	Os.block_swap_chars((char *)new_header, sizeof(int_4), 4);
	if (header[0] != new_header[0]) {
		cout << "header[0] != new_header[0]\n";
		}
	if (header[1] != new_header[1]) {
		cout << "header[1] != new_header[1]\n";
		}
	if (header[2] != new_header[2]) {
		cout << "header[2] != new_header[2]\n";
		}
	if (header[3] != new_header[3]) {
		cout << "header[3] != new_header[3]\n";
		}
	delete [] data2;
}

void database::add_data_DB_compact(void *d, 
	int size, uint_4 *datref, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int total = 0, pad = 0;
	char *data2 = NULL;
	char *pc, *pc0;
	int i;
	int_4 *pi;
	int_4 header[2];
	int_4 new_header[2];
		// 0: SYNC
		// 1: size of user data are
	int old_file_size;
	os_interface Os;
	
	if (f_v) {
		cout << "database::add_data_DB_compact()" << endl;
		}
	user2total(size, total, pad);
	data2 = (char *) new char[total];
	header[0] = MAGIC_SYNC;
	header[1] = (int_4) size;
	Os.block_swap_chars((char *)header, sizeof(int_4), 2);
	pi = (int_4 *)data2;
	pi[0] = header[0];
	pi[1] = header[1];
	Os.block_swap_chars((char *)header, sizeof(int_4), 2);
		// swap header back, there will be another test 
	pc = (char *)(pi + 2);
	pc0 = (char *)d;
	if (f_vv) {
		cout << "size = " << size << " pad = " << pad << " total = " << total << endl;
		}
	for (i = 0; i < size; i++)
		pc[i] = pc0[i];
	for (i = 0; i < pad; i++)
		pc[size + i] = 0;
	old_file_size = file_size();
	file_seek(old_file_size);
	file_write(data2, 1, total);
	*datref = (uint_4)(old_file_size >> size_of_header_log());
	if (((int)((unsigned int)*datref << size_of_header_log())) != old_file_size) {
		cout << "database::add_data_DB_compact ((unsigned int)*datref << size_of_header_log()) != old_file_size" << endl;
		cout << "old_file_size=" << old_file_size << endl;
		cout << "*datref=" << *datref << endl;
		cout << "size_of_header_log()=" << size_of_header_log() << endl;
		exit(1);
		}
	file_size() += total;
	//put_file_size();
	
	file_seek(old_file_size);
	file_read(new_header, 4, 2);
	Os.block_swap_chars((char *)new_header, sizeof(int_4), 2);
	if (header[0] != new_header[0]) {
		cout << "header[0] != new_header[0]\n";
		}
	if (header[1] != new_header[1]) {
		cout << "header[1] != new_header[1]\n";
		}
	delete [] data2;
}

void database::free_data_DB(uint_4 datref, int size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int total;
	int_4 header[8];
	os_interface Os;
	
	if (f_v) {
		cout << "database::free_data_DB()" << endl;
		}
	if (file_type() == DB_FILE_TYPE_COMPACT)
		return;
	file_seek(((unsigned int)datref) << size_of_header_log());
	total = 8 * 4;
	file_read(header, 1, total);
	Os.block_swap_chars((char *)header, 4, 8);
	if (header[0] != MAGIC_SYNC) {
		cout << "database::free_data_DB()|header: no MAGIC_SYNC\n";
		exit(1);
		}
	if (!header[1]) {
		cout << "database::free_data_DB()|header: data is not used\n";
		exit(1);
		}
	if (header[2] != size) {
		cout << "database::free_data_DB()|header: header[2] != size\n";
		exit(1);
		}
	if (header[4] != MAGIC_SYNC) {
		cout << "database::free_data_DB()|header2: no MAGIC_SYNC\n";
		exit(1);
		}
	if (!header[5]) {
		cout << "database::free_data_DB()|header2: data is not used\n";
		exit(1);
		}
	if (header[6] != size) {
		cout << "database::free_data_DB()|header2: header[6] != size\n";
		exit(1);
		}
	if (header[7] != header[3]) {
		cout << "database::free_data_DB()|header2: header[7] != header[3]\n";
		exit(1);
		}
	header[1] = FALSE;
	header[5] = FALSE;
	Os.block_swap_chars((char *)header, 4, 8);
	file_seek(((unsigned int)datref) << size_of_header_log());
	file_write(header, 1, total);
}

void database::file_open(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx = fstream_table_get_free_entry();
	fstream *f = new fstream(filename().s(), ios::in | ios::out | ios::binary);
	fstream_table[idx] = f;
	fstream_table_used[idx] = TRUE;
	stream() = idx;
	f_open() = TRUE;
	get_file_size();
	if (f_v) {
		cout << "database::file_open() file " << filename().s() << " opened" << endl;
		}
}

void database::file_create(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	hollerith cmd;
	
	cmd.init("rm ");
	cmd.append(filename().s());
	system(cmd.s());
	
	int idx = fstream_table_get_free_entry();
	{
	fstream *f = new fstream(filename().s(), ios::out | ios::binary);
	if (!*f) {
		cout << "database::file_create() file " << filename().s() << " could not be created" << endl;
		exit(1);
		}
	f->close();
	delete f;
	}
	fstream *f = new fstream(filename().s(), ios::in | ios::out | ios::binary);
	if (!*f) {
		cout << "database::file_create() file " << filename().s() << " could not be opened" << endl;
		exit(1);
		}
	fstream_table[idx] = f;
	fstream_table_used[idx] = TRUE;
	stream() = idx;
	f_open() = TRUE;
	if (f_v) {
		cout << "database::file_create() file " << filename().s() << " created" << endl;
		}
}

void database::file_close(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx = (int) stream();
	if (!fstream_table_used[idx]) {
		cout << "database::file_close() !fstream_table_used[idx]" << endl;
		cout << "idx=" << idx << endl;
		exit(1);
		}
	delete fstream_table[idx];
	fstream_table_used[idx] = FALSE;
	stream() = 0;
	f_open() = FALSE;
	if (f_v) {
		cout << "database::file_close() file " << filename().s() << " closed" << endl;
		}
}

void database::file_seek(int offset)
{
	if (!f_open()) {
		cout << "database::file_seek() file not open" << endl;
		exit(1);
		}
	int idx = (int) stream();
	if (!fstream_table_used[idx]) {
		cout << "database::file_seek() !fstream_table_used[idx]" << endl;
		cout << "idx=" << idx << endl;
		exit(1);
		}
	fstream_table[idx]->seekg(offset);
}

void database::file_write(void *p, int size, int nb)
{
	if (!f_open()) {
		cout << "database::file_write() file not open" << endl;
		exit(1);
		}
	int idx = (int) stream();
	if (!fstream_table_used[idx]) {
		cout << "database::file_write() !fstream_table_used[idx]" << endl;
		cout << "idx=" << idx << endl;
		exit(1);
		}
	fstream_table[idx]->write((char *) p, size * nb);
}

void database::file_read(void *p, int size, int nb)
{
	if (!f_open()) {
		cout << "database::file_read() file not open" << endl;
		exit(1);
		}
	int idx = (int) stream();
	if (!fstream_table_used[idx]) {
		cout << "database::file_read() !fstream_table_used[idx]" << endl;
		cout << "idx=" << idx << endl;
		exit(1);
		}
	fstream_table[idx]->read((char *) p, size * nb);
}

}}
