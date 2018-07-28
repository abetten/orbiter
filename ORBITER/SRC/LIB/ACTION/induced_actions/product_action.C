// product_action.C
//
// Anton Betten
// December 19, 2007

#include "GALOIS/galois.h"
#include "action.h"

INT product_action::cntr_new = 0;
INT product_action::cntr_objects = 0;
INT product_action::f_debug_memory = FALSE;

void *product_action::operator new(size_t bytes)
{
	cntr_new++;
	cntr_objects++;
	if (f_debug_memory) {
		cout << "product_action::operator new bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void *product_action::operator new[](size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(product_action);
	cntr_new++;
	cntr_objects += n;
	if (f_debug_memory) {
		cout << "product_action::operator new[] n=" << n 
			<< " bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void product_action::operator delete(void *ptr, size_t bytes)
{
	if (f_debug_memory) {
		cout << "product_action::operator delete bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects--;
	return ::free(ptr);
}

void product_action::operator delete[](void *ptr, size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(product_action);
	if (f_debug_memory) {
		cout << "product_action::operator delete[] n=" << n 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects -= n;
	return ::free(ptr);
}

product_action::product_action()
{
	null();
}

product_action::~product_action()
{
	free();
}

void product_action::null()
{
	A1 = NULL;
	A2 = NULL;
	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
	elt1 = NULL;
	elt2 = NULL;
	elt3 = NULL;
	offset = 0;
}

void product_action::free()
{
	if (A1) {
		delete [] A1;
		}
	if (A2) {
		delete [] A2;
		}
	if (Elt1)
		FREE_INT(Elt1);
	if (Elt2)
		FREE_INT(Elt2);
	if (Elt3)
		FREE_INT(Elt3);
	if (elt1)
		delete [] elt1;
	if (elt2)
		delete [] elt2;
	if (elt3)
		delete [] elt3;
	null();
}


void product_action::init(action *A1, action *A2, INT f_use_projections, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "product_action::init" << endl;
		}
	product_action::f_use_projections = f_use_projections;
	product_action::A1 = A1;
	product_action::A2 = A2;
	if (f_use_projections)
		offset = A1->degree + A2->degree;
	else	
		offset = 0;
	degree = offset + A1->degree * A2->degree;
	elt_size_in_INT = A1->elt_size_in_INT + A2->elt_size_in_INT;
	coded_elt_size_in_char = A1->coded_elt_size_in_char + A2->coded_elt_size_in_char;

	Elt1 = NEW_INT(elt_size_in_INT);
	Elt2 = NEW_INT(elt_size_in_INT);
	Elt3 = NEW_INT(elt_size_in_INT);
	elt1 = new UBYTE[coded_elt_size_in_char];
	elt2 = new UBYTE[coded_elt_size_in_char];
	elt3 = new UBYTE[coded_elt_size_in_char];


	Elts = new page_storage;
	if (f_vv) {
		cout << "matrix_group::init_linear() calling Elts->init()" << endl;
		}
	Elts->init(coded_elt_size_in_char /* entry_size */, PAGE_LENGTH_LOG, f_vv);
	//Elts->add_elt_print_function(elt_print, (void *) this);
}

INT product_action::compute_image(action *A, INT *Elt, INT i, INT verbose_level)
{
	//verbose_level = 1;
	INT x, y, xx, yy, j;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);

	if (f_vv) {
		cout << "product_action::compute_image i = " << i << endl;
		}
	if (f_use_projections) {
		if (i < offset) {
			if (i < A1->degree) {
				j = A1->element_image_of(i, Elt, FALSE);
				}
			else {
				i -= A1->degree;
				j = A2->element_image_of(i, Elt + A1->elt_size_in_INT, FALSE);
				j += A1->degree;
				}
			}
		else {
			i -= offset;
			x = i / A2->degree;
			y = i % A2->degree;
			xx = A1->element_image_of(x, Elt, FALSE);
			yy = A2->element_image_of(y, Elt + A1->elt_size_in_INT, FALSE);
			j = xx * A2->degree + yy;
			j += offset;
			}
		}
	else {
		x = i / A2->degree;
		y = i % A2->degree;
		xx = A1->element_image_of(x, Elt, FALSE);
		yy = A2->element_image_of(y, Elt + A1->elt_size_in_INT, FALSE);
		j = xx * A2->degree + yy;
		}
	if (f_v) {
		cout << "product_action::compute_image image of " << i << " is " << j << endl;
		}
	return j;
}

void product_action::element_one(action *A, INT *Elt, INT verbose_level)
{
	A1->element_one(Elt, verbose_level);
	A2->element_one(Elt + A1->elt_size_in_INT, verbose_level);
}

INT product_action::element_is_one(action *A, INT *Elt, INT verbose_level)
{
	if (!A1->element_is_one(Elt, verbose_level)) {
		return FALSE;
		}
	if (!A2->element_is_one(Elt + A1->elt_size_in_INT, verbose_level)) {
		return FALSE;
		}
	return TRUE;
}

void product_action::element_unpack(UBYTE *elt, INT *Elt, INT verbose_level)
{
	A1->element_unpack(elt, Elt, verbose_level);
	A2->element_unpack(elt + A1->coded_elt_size_in_char, Elt + A1->elt_size_in_INT, verbose_level);
}

void product_action::element_pack(INT *Elt, UBYTE *elt, INT verbose_level)
{
	A1->element_pack(Elt, elt, verbose_level);
	A2->element_pack(Elt + A1->elt_size_in_INT, elt + A1->coded_elt_size_in_char, verbose_level);
}

void product_action::element_retrieve(action *A, INT hdl, INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	UBYTE *p_elt;
	
	if (f_v) {
		cout << "product_action::element_retrieve() hdl = " << hdl << endl;
		}
	p_elt = Elts->s_i(hdl);
	A1->element_unpack(p_elt, Elt, verbose_level);
	A2->element_unpack(p_elt + A1->coded_elt_size_in_char, Elt + A1->elt_size_in_INT, verbose_level);
}

INT product_action::element_store(action *A, INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT hdl;
	
	A1->element_pack(Elt, elt1, verbose_level);
	A2->element_pack(Elt + A1->elt_size_in_INT, elt1 + A1->coded_elt_size_in_char, verbose_level);
	hdl = Elts->store(elt1);
	if (f_v) {
		cout << "product_action::element_store() hdl = " << hdl << endl;
		}
	return hdl;
}

void product_action::element_mult(INT *A, INT *B, INT *AB, INT verbose_level)
{
	A1->element_mult(A, B, AB, verbose_level);
	A2->element_mult(A + A1->elt_size_in_INT, B + A1->elt_size_in_INT, AB + A1->elt_size_in_INT, verbose_level);
}

void product_action::element_invert(INT *A, INT *Av, INT verbose_level)
{
	A1->element_invert(A, Av, verbose_level);
	A2->element_invert(A + A1->elt_size_in_INT, Av + A1->elt_size_in_INT, verbose_level);
}

void product_action::element_transpose(INT *A, INT *At, INT verbose_level)
{
	A1->element_transpose(A, At, verbose_level);
	A2->element_transpose(A + A1->elt_size_in_INT, At + A1->elt_size_in_INT, verbose_level);
}

void product_action::element_move(INT *A, INT *B, INT verbose_level)
{
	A1->element_move(A, B, verbose_level);
	A2->element_move(A + A1->elt_size_in_INT, B + A1->elt_size_in_INT, verbose_level);
}

void product_action::element_print(INT *A, ostream &ost)
{
	ost << "(" << endl;
	A1->element_print(A, ost);
	ost << ", " << endl;
	A2->element_print(A + A1->elt_size_in_INT, ost);
	ost << ")" << endl;
}

void product_action::element_print_latex(INT *A, ostream &ost)
{
	ost << "\\left(" << endl;
	A1->element_print_latex(A, ost);
	ost << ", " << endl;
	A2->element_print_latex(A + A1->elt_size_in_INT, ost);
	ost << "\\\right)" << endl;
}

void product_action::make_element(INT *Elt, INT *data, INT verbose_level)
{
	A1->make_element(Elt, data, verbose_level);
	A2->make_element(Elt + A1->elt_size_in_INT, 
		data + A1->make_element_size, verbose_level);
}



