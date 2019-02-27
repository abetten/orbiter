// bt_key.C
//
// Anton Betten
// 27.11.2000
// moved from D2 to ORBI Nov 15, 2007

#include "orbiter.h"

#include <string.h> // strncmp

using namespace std;


namespace orbiter {
namespace discreta {

bt_key::bt_key() : Vector()
{
	k = BT_KEY;
}

bt_key::bt_key(const discreta_base &x)
	// copy constructor:    this := x
{
	cout << "bt_key::copy constructor for object: "
			<< const_cast<discreta_base &>(x) << "\n";
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

void bt_key::init(enum bt_key_kind type, int output_size, int field1, int field2)
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

void bt_key::init_int4(int field1, int field2)
{
	init(bt_key_int, 4, field1, field2);
}

void bt_key::init_int2(int field1, int field2)
{
	init(bt_key_int, 2, field1, field2);
}

void bt_key::init_string(int output_size, int field1, int field2)
{
	init(bt_key_string, output_size, field1, field2);
}

void bt_key::init_int4_vec(int field1, int field2, int vec_fst, int vec_len)
{
	init(bt_key_int_vec, 4, field1, field2);
	bt_key::int_vec_first() = vec_fst;
	bt_key::int_vec_len() = vec_len;
}

void bt_key::init_int2_vec(int field1, int field2, int vec_fst, int vec_len)
{
	init(bt_key_int_vec, 2, field1, field2);
	bt_key::int_vec_first() = vec_fst;
	bt_key::int_vec_len() = vec_len;
}

int bt_lexicographic_cmp(char *p1, char *p2)
{
	return strcmp(p1, p2);
}

int bt_key_int_cmp(char *p1, char *p2)
{
	int_4 *p_l1, *p_l2;
	
	p_l1 = (int_4 *) p1;
	p_l2 = (int_4 *) p2;
	if (*p_l1 < *p_l2) {
		return -1;
		}
	if (*p_l1 > *p_l2) {
		return 1;
		}
	return 0;
}

int bt_key_int2_cmp(char *p1, char *p2)
{
	int_4 *p_l1, *p_l2;
	
	p_l1 = (int_4 *) p1;
	p_l2 = (int_4 *) p2;
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

void bt_key_print_int4(char **key, ostream& ost)
{
	int_4 i;
	bt_key_get_int4(key, i);
	ost << i;
}

void bt_key_print_int2(char **key, ostream& ost)
{
	int_2 i;
	bt_key_get_int2(key, i);
	ost << i;
}

void bt_key_print(char *key, Vector& V, ostream& ost)
{
	char *the_key = key;
	char c;
	int i, j, l1, output_size;
	enum bt_key_kind k;
	
	ost << "[";
	for (i = 0; i < V.s_l(); i++) {
		bt_key& Key = V[i].as_bt_key();
		k = Key.type();
		output_size = Key.output_size();
		if (k == bt_key_int) {
			if (output_size == 4) {
				bt_key_print_int4(&the_key, ost);
				}
			else if (output_size == 2) {
				bt_key_print_int2(&the_key, ost);
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
					bt_key_print_int4(&the_key, ost);
					}
				else if (output_size == 2) {
					bt_key_print_int2(&the_key, ost);
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

int bt_key_compare_int4(char **p_key1, char **p_key2)
{
	int_4 int1, int2;
	int i;
	char c;
	char *pc_1 = (char *) &int1;
	char *pc_2 = (char *) &int2;
	
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

int bt_key_compare_int2(char **p_key1, char **p_key2)
{
	int_2 int1, int2;
	int i;
	char c;
	char *pc_1 = (char *) &int1;
	char *pc_2 = (char *) &int2;
	
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

int bt_key_compare(char *key1, char *key2, Vector& V, int depth)
{
	char *the_key1 = key1;
	char *the_key2 = key2;
	int i, j, output_size, res;
	enum bt_key_kind k;
	
	if (depth == 0)
		depth = V.s_l();
	for (i = 0; i < depth; i++) {
		bt_key& Key = V[i].as_bt_key();
		k = Key.type();
		output_size = Key.output_size();
		if (k == bt_key_int) {
			if (output_size == 4) {
				res = bt_key_compare_int4(&the_key1, &the_key2);
				if (res)
					return res;
				}
			else if (output_size == 2) {
				res = bt_key_compare_int2(&the_key1, &the_key2);
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
					res = bt_key_compare_int4(&the_key1, &the_key2);
					if (res)
						return res;
					}
				else if (output_size == 2) {
					res = bt_key_compare_int2(&the_key1, &the_key2);
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

void bt_key_fill_in_int4(char **p_key, discreta_base& key_op)
{
	if (key_op.s_kind() != INTEGER) {
		cout << "bt_key_fill_in_int4 "
				"object not an INTEGER" << endl;
		exit(1);
		}
	integer& key_op_int = key_op.as_integer();
	int_4 a = (int_4) key_op_int.s_i();
	int i;
	char *pc = (char *) &a;
	char c;
	
	for (i = 0; i < 4; i++) {
		c = pc[i];
		**p_key = c;
		(*p_key)++;
		}
}

void bt_key_fill_in_int2(char **p_key, discreta_base& key_op)
{
	if (key_op.s_kind() != INTEGER) {
		cout << "bt_key_fill_in_int2 "
				"object not an INTEGER" << endl;
		exit(1);
		}
	integer& key_op_int = key_op.as_integer();
	int_2 a = key_op_int.s_i();
	int i;
	char *pc = (char *) &a;
	char c;
	
	for (i = 0; i < 2; i++) {
		c = pc[i];
		**p_key = c;
		(*p_key)++;
		}
}

void bt_key_fill_in_string(char **p_key, int output_size, discreta_base& key_op)
{
	if (key_op.s_kind() != HOLLERITH) {
		cout << "bt_key_fill_in_string "
				"object not an HOLLERITH" << endl;
		exit(1);
		}
	hollerith& h = key_op.as_hollerith();
	strncpy(*p_key, h.s(), output_size);
	*p_key += output_size;
}

void bt_key_fill_in(char *key, Vector& V, Vector& the_object)
{
	char *the_key = key;
	int i, j, output_size;
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
				bt_key_fill_in_int4(&the_key, key_object);
				}
			else if (output_size == 2) {
				bt_key_fill_in_int2(&the_key, key_object);
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
			int fst = Key.int_vec_first();
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
					bt_key_fill_in_int4(&the_key, *key_object1);
					}
				else if (output_size == 2) {
					bt_key_fill_in_int2(&the_key, *key_object1);
					}
				else {
					cout << "bt_key_fill_in output_size not 2 or 4" << endl;
					exit(1);
					}
				}
			}
		else {
			cout << "bt_key_fill_in unknown bt_key_kind" << endl;
			exit(1);
			}
		}
}

void bt_key_get_int4(char **key, int_4 &i)
{
	char *pc = (char *)&i;
	
	pc[0] = **key; (*key)++;
	pc[1] = **key; (*key)++;
	pc[2] = **key; (*key)++;
	pc[3] = **key; (*key)++;
}

void bt_key_get_int2(char **key, int_2 &i)
{
	char *pc = (char *)&i;
	
	pc[0] = **key; (*key)++;
	pc[1] = **key; (*key)++;
}

}}

