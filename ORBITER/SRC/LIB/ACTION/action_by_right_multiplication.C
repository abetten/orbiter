// action_by_right_multiplication.C
//
// Anton Betten
// January 10, 2009

#include "galois.h"
#include "action.h"

INT action_by_right_multiplication::cntr_new = 0;
INT action_by_right_multiplication::cntr_objects = 0;
INT action_by_right_multiplication::f_debug_memory = FALSE;

void *action_by_right_multiplication::operator new(size_t bytes)
{
	cntr_new++;
	cntr_objects++;
	if (f_debug_memory) {
		cout << "action_by_right_multiplication::operator new bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void *action_by_right_multiplication::operator new[](size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(action_by_right_multiplication);
	cntr_new++;
	cntr_objects += n;
	if (f_debug_memory) {
		cout << "action_by_right_multiplication::operator new[] n=" << n 
			<< " bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void action_by_right_multiplication::operator delete(void *ptr, size_t bytes)
{
	if (f_debug_memory) {
		cout << "action_by_right_multiplication::operator delete bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects--;
	return ::free(ptr);
}

void action_by_right_multiplication::operator delete[](void *ptr, size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(action_by_right_multiplication);
	if (f_debug_memory) {
		cout << "action_by_right_multiplication::operator delete[] n=" << n 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects -= n;
	return ::free(ptr);
}

action_by_right_multiplication::action_by_right_multiplication()
{
	null();
}

action_by_right_multiplication::~action_by_right_multiplication()
{
	free();
}

void action_by_right_multiplication::null()
{
	Base_group = NULL;
	Elt1 = NULL;
	Elt2 = NULL;
}

void action_by_right_multiplication::free()
{
	
	if (Base_group && f_ownership) {
		delete Base_group;
		}
	if (Elt1) {
		FREE_INT(Elt1);
		}
	if (Elt2) {
		FREE_INT(Elt2);
		}
	null();
}


void action_by_right_multiplication::init(sims *Base_group, INT f_ownership, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	longinteger_object go;
	action *A;
	
	if (f_v) {
		cout << "action_by_right_multiplication::init" << endl;
		}
	action_by_right_multiplication::Base_group = Base_group;
	action_by_right_multiplication::f_ownership = f_ownership;
	A = Base_group->A;
	Base_group->group_order(go);
	goi = go.as_INT();
	if (f_v) {
		cout << "action_by_right_multiplication::init we are acting on a group of order " << goi << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
}

void action_by_right_multiplication::compute_image(action *A, INT *Elt, INT i, INT &j, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_by_right_multiplication::compute_image i = " << i << endl;
		}
	if (i < 0 || i >= goi) {
		cout << "action_by_right_multiplication::compute_image i = " << i << " out of range" << endl;
		exit(1);
		}
	Base_group->element_unrank_INT(i, Elt1);
	A->mult(Elt1, Elt, Elt2);
	j = Base_group->element_rank_INT(Elt2);
	if (f_v) {
		cout << "action_by_right_multiplication::compute_image image is " << j << endl;
		}
}

