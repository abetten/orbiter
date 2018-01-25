// dynamic_memory.C
//
// Anton Betten
// 
// separated out from partition_backtrack.C: 1/3/10

#include "galois.h"
#include "incidence.h"

// #####################################################################################
// dynamic_memory
// #####################################################################################

dynamic_memory::dynamic_memory()
{
	ptr = NULL;
	size_allocated = 0;
	size_used = 0;
}

dynamic_memory::~dynamic_memory()
{
	if (size_allocated) {
		FREE_INT(ptr);
		}
}

void dynamic_memory::allocate()
{
	if (size_allocated) {
		FREE_INT(ptr);
		}
	ptr = NEW_INT(1024);
	size_allocated = 1024;
	size_used = 0;
}

void dynamic_memory::reallocate(INT required_size, INT f_copy_over)
{
	if (required_size > size_allocated) {
		INT *ptr2 = NEW_INT(required_size);
		if (f_copy_over) {
			INT i;
			for (i = 0; i < size_used; i++) 
				ptr2[i] = ptr[i];
			}
		if (size_allocated) {
			FREE_INT(ptr);
			}
		ptr = ptr2;
		size_allocated = required_size;
		}
}


