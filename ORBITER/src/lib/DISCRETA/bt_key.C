// bt_key.C
//
// Anton Betten
// 27.11.2000
// moved from D2 to ORBI Nov 15, 2007

#include "orbiter.h"

#include <string.h> // strncmp


bt_key::bt_key() : Vector()
{
	k = BT_KEY;
}

bt_key::bt_key(const discreta_base &x)
	// copy constructor:    this := x
{
	cout << "bt_key::copy constructor for object: " << const_cast<discreta_base &>(x) << "\n";
	const_cast<discreta_base &>(x).copyobject_to(*this);
}

bt_key& bt_key::operator = (const discreta_base &x)
	// copy assignment
{
	cout << "bt_key::operator = (copy assignment)" << endl;
	copyobject(const_cast<discreta_base &>(x));
	return *this;
}

void bt_key::settype_bt_key()
{
	OBJECTSELF s;
	
	s = self;
	new(this) bt_key;
	self = s;
	k = BT_KEY;
}

bt_key::~bt_key()
{
	freeself_bt_key();
}

void bt_key::freeself_bt_key()
{
	// cout << "group_selection::freeself_bt_key()\n";
	freeself_vector();
}

kind bt_key::s_virtual_kind()
{
	return BT_KEY;
}

void bt_key::copyobject_to(discreta_base &x)
{
#ifdef COPY_VERBOSE
	cout << "bt_key::copyobject_to()\n";
	print_as_vector(cout);
#endif
	Vector::copyobject_to(x);
	x.as_bt_key().settype_bt_key();
#ifdef COPY_VERBOSE
	x.as_bt_key().print_as_vector(cout);
#endif
}

ostream& bt_key::print(ostream& ost)
{
	
	return ost;
}

void bt_key::init(enum bt_key_kind type, INT output_size, INT field1, INT field2)
{
	m_l_n(7);
	c_kind(BT_KEY);
	
	bt_key::type() = type;
	bt_key::output_size() = output_size;
	bt_key::field1() = field1;
	bt_key::field2() = field2;
	int_vec_first() = 0;
	int_vec_len() = 0;
	f_ascending() = TRUE;
	
}

void bt_key::init_INT4(INT field1, INT field2)
{
	init(bt_key_int, 4, field1, field2);
}

void bt_key::init_INT2(INT field1, INT field2)
{
	init(bt_key_int, 2, field1, field2);
}

void bt_key::init_string(INT output_size, INT field1, INT field2)
{
	init(bt_key_string, output_size, field1, field2);
}

void bt_key::init_int4_vec(INT field1, INT field2, INT vec_fst, INT vec_len)
{
	init(bt_key_int_vec, 4, field1, field2);
	bt_key::int_vec_first() = vec_fst;
	bt_key::int_vec_len() = vec_len;
}

void bt_key::init_int2_vec(INT field1, INT field2, INT vec_fst, INT vec_len)
{
	init(bt_key_int_vec, 2, field1, field2);
	bt_key::int_vec_first() = vec_fst;
	bt_key::int_vec_len() = vec_len;
}

INT bt_lexicographic_cmp(BYTE *p1, BYTE *p2)
{
	return strcmp(p1, p2);
}

INT bt_key_int_cmp(BYTE *p1, BYTE *p2)
{
	INT4 *p_l1, *p_l2;
	
	p_l1 = (INT4 *) p1;
	p_l2 = (INT4 *) p2;
	if (*p_l1 < *p_l2) {
		return -1;
		}
	if (*p_l1 > *p_l2) {
		return 1;
		}
	return 0;
}

INT bt_key_int2_cmp(BYTE *p1, BYTE *p2)
{
	INT4 *p_l1, *p_l2;
	
	p_l1 = (INT4 *) p1;
	p_l2 = (INT4 *) p2;
	if (*p_l1 < *p_l2) {
		return -1;
		}
	if (*p_l1 > *p_l2) {
		return 1;
		}
	if (p_l1[1] < p_l2[1]) {
		return -1;
		}
	if (p_l1[1] > p_l2[1]) {
		return 1;
		}
	return 0;
}

void bt_key_print_INT4(BYTE **key, ostream& ost)
{
	INT4 i;
	bt_key_get_INT4(key, i);
	ost << i;
}

void bt_key_print_INT2(BYTE **key, ostream& ost)
{
	INT2 i;
	bt_key_get_INT2(key, i);
	ost << i;
}

void bt_key_print(BYTE *key, Vector& V, ostream& ost)
{
	BYTE *the_key = key;
	BYTE c;
	INT i, j, l1, output_size;
	enum bt_key_kind k;
	
	ost << "[";
	for (i = 0; i < V.s_l(); i++) {
		bt_key& Key = V[i].as_bt_key();
		k = Key.type();
		output_size = Key.output_size();
		if (k == bt_key_int) {
			if (output_size == 4) {
				bt_key_print_INT4(&the_key, ost);
				}
			else if (output_size == 2) {
				bt_key_print_INT2(&the_key, ost);
				}
			else {
				cout << "bt_key_print() output_size not 2 or 4" << endl;
				exit(1);
				}
			}
		else if (k == bt_key_string) {
			for (j = 0; j < output_size; j++) {
				if (the_key[j] == 0) {
					break;
					}
				}
			l1 = j;
			for (j = 0; j < output_size; j++) {
				if (j < l1)
					c = *the_key;
				else
					c = ' ';
				ost << c;
				the_key++;
				}
			// ost << ends;
			}
		else if (k == bt_key_int_vec) {
			ost << "(";
			for (j = 0; j < Key.int_vec_len(); j++) {
				if (output_size == 4) {
					bt_key_print_INT4(&the_key, ost);
					}
				else if (output_size == 2) {
					bt_key_print_INT2(&the_key, ost);
					}
				else {
					cout << "bt_key_print() output_size not 2 or 4" << endl;
					exit(1);
					}
				if (j < Key.int_vec_len())
					ost << ", ";
				}
			ost << ")";
			}
		else {
			cout << "bt_key_print() unknown bt_key_kind" << endl;
			exit(1);
			}
		if (i < V.s_l() - 1)
			ost << " ";
		}
	ost << "]";
}

INT bt_key_compare_INT4(BYTE **p_key1, BYTE **p_key2)
{
	INT4 int1, int2;
	INT i;
	BYTE c;
	BYTE *pc_1 = (BYTE *) &int1;
	BYTE *pc_2 = (BYTE *) &int2;
	
	for (i = 0; i < 4; i++) {
		c = **p_key1;
		(*p_key1)++;
		pc_1[i] = c;
		c = **p_key2;
		(*p_key2)++;
		pc_2[i] = c;
		}
	if (int1 < int2) {
		return -1;
		}
	if (int1 > int2) {
		return 1;
		}
	return 0;
}

INT bt_key_compare_INT2(BYTE **p_key1, BYTE **p_key2)
{
	INT2 int1, int2;
	INT i;
	BYTE c;
	BYTE *pc_1 = (BYTE *) &int1;
	BYTE *pc_2 = (BYTE *) &int2;
	
	for (i = 0; i < 2; i++) {
		c = **p_key1;
		(*p_key1)++;
		pc_1[i] = c;
		c = **p_key2;
		(*p_key2)++;
		pc_2[i] = c;
		}
	if (int1 < int2) {
		return -1;
		}
	if (int1 > int2) {
		return 1;
		}
	return 0;
}

INT bt_key_compare(BYTE *key1, BYTE *key2, Vector& V, INT depth)
{
	BYTE *the_key1 = key1;
	BYTE *the_key2 = key2;
	INT i, j, output_size, res;
	enum bt_key_kind k;
	
	if (depth == 0)
		depth = V.s_l();
	for (i = 0; i < depth; i++) {
		bt_key& Key = V[i].as_bt_key();
		k = Key.type();
		output_size = Key.output_size();
		if (k == bt_key_int) {
			if (output_size == 4) {
				res = bt_key_compare_INT4(&the_key1, &the_key2);
				if (res)
					return res;
				}
			else if (output_size == 2) {
				res = bt_key_compare_INT2(&the_key1, &the_key2);
				if (res)
					return res;
				}
			else {
				cout << "bt_key_compare() output_size not 2 or 4" << endl;
				exit(1);
				}
			}
		else if (k == bt_key_string) {
			res = strncmp(the_key1, the_key2, output_size);
			if (res)
				return res;
			the_key1 += output_size;
			the_key2 += output_size;
			}
		else if (k == bt_key_int_vec) {
			for (j = 0; j < Key.int_vec_len(); j++) {
				if (output_size == 4) {
					res = bt_key_compare_INT4(&the_key1, &the_key2);
					if (res)
						return res;
					}
				else if (output_size == 2) {
					res = bt_key_compare_INT2(&the_key1, &the_key2);
					if (res)
						return res;
					}
				else {
					cout << "bt_key_compare() output_size not 2 or 4" << endl;
					exit(1);
					}
				}
			}
		else {
			cout << "bt_key_compare() unknown bt_key_kind" << endl;
			exit(1);
			}
		}
	return 0;
}

void bt_key_fill_in_INT4(BYTE **p_key, discreta_base& key_op)
{
	if (key_op.s_kind() != INTEGER) {
		cout << "bt_key_fill_in_INT4() object not an INTEGER" << endl;
		exit(1);
		}
	integer& key_op_int = key_op.as_integer();
	INT4 a = (INT4) key_op_int.s_i();
	INT i;
	BYTE *pc = (BYTE *) &a;
	BYTE c;
	
	for (i = 0; i < 4; i++) {
		c = pc[i];
		**p_key = c;
		(*p_key)++;
		}
}

void bt_key_fill_in_INT2(BYTE **p_key, discreta_base& key_op)
{
	if (key_op.s_kind() != INTEGER) {
		cout << "bt_key_fill_in_INT2() object not an INTEGER" << endl;
		exit(1);
		}
	integer& key_op_int = key_op.as_integer();
	INT2 a = key_op_int.s_i();
	INT i;
	BYTE *pc = (BYTE *) &a;
	BYTE c;
	
	for (i = 0; i < 2; i++) {
		c = pc[i];
		**p_key = c;
		(*p_key)++;
		}
}

void bt_key_fill_in_string(BYTE **p_key, INT output_size, discreta_base& key_op)
{
	if (key_op.s_kind() != HOLLERITH) {
		cout << "bt_key_fill_in_string() object not an HOLLERITH" << endl;
		exit(1);
		}
	hollerith& h = key_op.as_hollerith();
	strncpy(*p_key, h.s(), output_size);
	*p_key += output_size;
}

void bt_key_fill_in(BYTE *key, Vector& V, Vector& the_object)
{
	BYTE *the_key = key;
	INT i, j, output_size;
	enum bt_key_kind k;
	
	for (i = 0; i < V.s_l(); i++) {

		if (the_key - key > BTREEMAXKEYLEN) {
			cout << "bt_key_fill_in the_key - key > BTREEMAXKEYLEN" << endl;
			cout << "BTREEMAXKEYLEN=" << BTREEMAXKEYLEN << endl;
			cout << "the_key - key=" << the_key - key << endl;
			exit(1);
			}

		bt_key& Key = V[i].as_bt_key();
		k = Key.type();
		output_size = Key.output_size();
		discreta_base& key_object = the_object.s_i(Key.field1());
		
		if (k == bt_key_int) {
			if (output_size == 4) {
				bt_key_fill_in_INT4(&the_key, key_object);
				}
			else if (output_size == 2) {
				bt_key_fill_in_INT2(&the_key, key_object);
				}
			else {
				cout << "bt_key_fill_in() output_size not 2 or 4" << endl;
				exit(1);
				}
			}
		else if (k == bt_key_string) {
			bt_key_fill_in_string(&the_key, output_size, key_object);
			}
		else if (k == bt_key_int_vec) {
			INT fst = Key.int_vec_first();
			Vector& key_vec = key_object.as_vector();
			discreta_base *key_object1;
			integer null_ob;
			null_ob.m_i(0);
			
			for (j = 0; j < Key.int_vec_len(); j++) {
				if (fst + j < key_vec.s_l())
					key_object1 = &key_vec[fst + j];
				else 
					key_object1 = &null_ob;
				if (output_size == 4) {
					bt_key_fill_in_INT4(&the_key, *key_object1);
					}
				else if (output_size == 2) {
					bt_key_fill_in_INT2(&the_key, *key_object1);
					}
				else {
					cout << "bt_key_fill_in() output_size not 2 or 4" << endl;
					exit(1);
					}
				}
			}
		else {
			cout << "bt_key_fill_in() unknown bt_key_kind" << endl;
			exit(1);
			}
		}
}

void bt_key_get_INT4(BYTE **key, INT4 &i)
{
	BYTE *pc = (BYTE *)&i;
	
	pc[0] = **key; (*key)++;
	pc[1] = **key; (*key)++;
	pc[2] = **key; (*key)++;
	pc[3] = **key; (*key)++;
}

void bt_key_get_INT2(BYTE **key, INT2 &i)
{
	BYTE *pc = (BYTE *)&i;
	
	pc[0] = **key; (*key)++;
	pc[1] = **key; (*key)++;
}


