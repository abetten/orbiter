// partitionstack.C
//
// Anton Betten
//
// started in D2/partition_stack.C: November 22, 2000
// included into GALOIS: July 3, 2007
// added TDO for orthogonal: July 10, 2007




#include "galois.h"


// ####################################################################################
// now comes partitionstack
// ####################################################################################

INT partitionstack::cntr_new = 0;
INT partitionstack::cntr_objects = 0;
INT partitionstack::f_debug_memory = FALSE;

void *partitionstack::operator new(size_t bytes)
{
	cntr_new++;
	cntr_objects++;
	if (f_debug_memory) {
		cout << "partitionstack::operator new bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void *partitionstack::operator new[](size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(partitionstack);
	cntr_new++;
	cntr_objects += n;
	if (f_debug_memory) {
		cout << "partitionstack::operator new[] n=" << n 
			<< " bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void partitionstack::operator delete(void *ptr, size_t bytes)
{
	if (f_debug_memory) {
		cout << "partitionstack::operator delete bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects--;
	return ::free(ptr);
}

void partitionstack::operator delete[](void *ptr, size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(partitionstack);
	if (f_debug_memory) {
		cout << "partitionstack::operator delete[] n=" << n 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects -= n;
	return ::free(ptr);
}

ostream& operator<<(ostream& ost, partitionstack& p)
{
	// cout << "partitionstack::operator<< starting" << endl;
	p.print(ost);
	// cout << "partitionstack::operator<< finished" << endl";
	return ost;
};

partitionstack::partitionstack()
{
	pointList = NULL;
	invPointList = NULL;
	cellNumber = NULL;

	startCell = NULL;
	cellSize = NULL;
	parent = NULL;


	subset = NULL;
	subset_first = NULL;
	subset_length = NULL;
	subsets = NULL;
}

partitionstack::~partitionstack()
{
	free();
}

void partitionstack::allocate(INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;

	if (f_v) {
		cout << "allocating partition stack n=" << n << endl;
		}

	partitionstack::n = n;
	ht = 1;

	//cout << "partitionstack::partitionstack() 1" << endl;
	pointList = NEW_INT(n);
	invPointList = NEW_INT(n);

	//cout << "partitionstack::partitionstack() 4" << endl;
	cellNumber = NEW_INT(n);

	startCell = NEW_INT(n + 1);
	cellSize = NEW_INT(n + 1);
	parent = NEW_INT(n + 1);

	// used if SPLIT_MULTIPLY is not defined:
	subset = NEW_INT(n + 1);

	//cout << "partitionstack::partitionstack() 7" << endl;
	// used if SPLIT_MULTIPLY is defined:
	nb_subsets = 0;
	subset_first = NEW_INT(n + 1);
	subset_length = NEW_INT(n + 1);
	subsets = NEW_INT(n + 1);

	//cout << "partitionstack::partitionstack() 8" << endl;
	for (i = 0; i < n; i++) {
		pointList[i] = i;
		invPointList[i] = i;
		cellNumber[i] = 0;
		}
	startCell[0] = 0;
	cellSize[0] = n;
	parent[0] = 0;

}

void partitionstack::free()
{
	if (pointList)
		FREE_INT(pointList);
	if (invPointList)
		FREE_INT(invPointList);
	if (cellNumber)
		FREE_INT(cellNumber);

	if (startCell)
		FREE_INT(startCell);
	if (cellSize)
		FREE_INT(cellSize);
	if (parent)
		FREE_INT(parent);

	if (subset)
		FREE_INT(subset);
	if (subset_first)
		FREE_INT(subset_first);
	if (subset_length)
		FREE_INT(subset_length);
	if (subsets)
		FREE_INT(subsets);

}

INT partitionstack::parent_at_height(INT h, INT cell)
{
	if (cell < h)
		return cell;
	else
		return parent_at_height(h, parent[cell]);
}

INT partitionstack::is_discrete()
{
	if (ht > n) {
		cout << "partitionstack::is_discrete() ht > n" << endl;
		exit(1);
		}
	if (ht == n)
		return TRUE;
	else
		return FALSE;
}

INT partitionstack::smallest_non_discrete_cell()
{
	INT min_size = 0, cell, i;

	cell = -1;
	for (i = 0; i < ht; i++) {
		if (cellSize[i] == 1)
			continue;
		if (cell == -1 || cellSize[i] < min_size) {
			cell = i;
			min_size = cellSize[i];
			}
		}
	if (cell == -1) {
		cout << "partitionstack::smallest_non_discrete_cell() partition is discrete" << endl;
		}
	return cell;
}

INT partitionstack::biggest_non_discrete_cell()
{
	INT max_size = 0, cell, i;

	cell = -1;
	for (i = 0; i < ht; i++) {
		if (cellSize[i] == 1)
			continue;
		if (cell == -1 || cellSize[i] > max_size) {
			cell = i;
			max_size = cellSize[i];
			}
		}
	if (cell == -1) {
		cout << "partitionstack::biggest_non_discrete_cell() partition is discrete" << endl;
		}
	return cell;
}

INT partitionstack::smallest_non_discrete_cell_rows_preferred()
{
	INT min_size = 0, cell, i;
	INT first_column_element = startCell[1];

	cell = -1;
	for (i = 0; i < ht; i++) {
		if (cellSize[i] == 1)
			continue;
		if (startCell[i] >= first_column_element)
			continue;
		if (cell == -1 || cellSize[i] < min_size) {
			cell = i;
			min_size = cellSize[i];
			}
		}
	if (cell == -1) {
		cell = smallest_non_discrete_cell();
		}
	return cell;
}

INT partitionstack::biggest_non_discrete_cell_rows_preferred()
{
	INT max_size = 0, cell, i;
	INT first_column_element = startCell[1];

	cell = -1;
	for (i = 0; i < ht; i++) {
		if (cellSize[i] == 1)
			continue;
		if (startCell[i] >= first_column_element)
			continue;
		if (cell == -1 || cellSize[i] > max_size) {
			cell = i;
			max_size = cellSize[i];
			}
		}
	if (cell == -1) {
		cell = biggest_non_discrete_cell();
		}
	return cell;
}

INT partitionstack::nb_partition_classes(INT from, INT len)
{
	INT i, c, l, n;

	n = 0;
	i = from;
	while (i < from + len) {
		c = cellNumber[i];
		l = cellSize[c];
		n++;
		i += l;
		}
	return n;
}

INT partitionstack::is_subset_of_cell(INT *set, INT size, INT &cell_idx)
{
	INT i, a, idx, c;

	for (i = 0; i < size; i++) {
		a = set[i];
		idx = invPointList[a];
		c = cellNumber[idx];
		if (i == 0) {
			cell_idx = c;
			}
		else {
			if (cell_idx != c) {
				return FALSE;
				}
			}
		}
	return TRUE;
}

void partitionstack::sort_cells()
{
	INT i;
	
	for (i = 0; i < ht; i++) {
		sort_cell(i);
		}
	check();
}

void partitionstack::sort_cell(INT cell)
{
	INT i, first, len, a;

	first = startCell[cell];
	len = cellSize[cell];
	
#if 0
	cout << "before sort, cell " << cell << " : " << endl;
	for (i = 0; i < len; i++) {
		cout << pointList[first + i] << " ";
		}
	cout << endl;
#endif
	INT_vec_quicksort_increasingly(pointList + first, len);
#if 0
	cout << "after sort, cell " << cell << " : " << endl;
	for (i = 0; i < len; i++) {
		cout << pointList[first + i] << " ";
		}
	cout << endl;
#endif
	
#if 0
	for (i = 0; i < len; i++) {
		for (j = i + 1; j < len; j++) {
			a = pointList[first + i];
			b = pointList[first + j];
			if (a < b)
				continue;
			pointList[first + i] = b;
			pointList[first + j] = a;
			}
		}
#endif
	for (i = 0; i < len; i++) {
		a = pointList[first + i];
		invPointList[a] = first + i;
		}
}

void partitionstack::reverse_cell(INT cell)
{
	INT i, j, first, len, half_len, a, b;

	first = startCell[cell];
	len = cellSize[cell];
	half_len = len >> 1;
	for (i = 0; i < half_len; i++) {
		j = len - 1 - i;
		a = pointList[first + i];
		b = pointList[first + j];
		pointList[first + i] = b;
		pointList[first + j] = a;
		}
	for (i = 0; i < len; i++) {
		a = pointList[first + i];
		invPointList[a] = first + i;
		}
}

void partitionstack::check()
{
	INT i, a;

	for (i = 0; i < n; i++) {
		a = pointList[i];
		if (invPointList[a] != i) {
			cout << "partitionstack::check() invPointList corrupt" << endl;
			cout << "i=" << i << " pointList[i]=a=" << a << " invPointList[a]=" << invPointList[a] << endl;
			print_raw();
			print(cout);
			exit(1);
			}
		}
}

void partitionstack::print_raw()
{
	INT i, first, len;
	
	cout << "ht = " << ht << endl;
	cout << "i : first : len " << endl;
	for (i = 0; i < ht; i++) {
		first = startCell[i];
		len = cellSize[i];
		cout << setw(5) << i << " : " << setw(5) << first << " : " << setw(5) << len << endl;
		}
	cout << "i : pointList : invPointList : cellNumber" << endl;
	for (i = 0; i < n; i++) {
		cout << setw(5) << i << " : " << setw(5) << pointList[i] << " : " << setw(5) << invPointList[i] << " : " << setw(5) << cellNumber[i] << endl;
		}
}

void partitionstack::print_class(ostream& ost, INT idx)
{
	INT first, len, j;
	INT *S;
	
	S = NEW_INT(n);
	first = startCell[idx];
	len = cellSize[idx];
	ost << "C_{" << idx << "} of size " << len << " descendant of " << parent[idx] << " is ";
	for (j = 0; j < len; j++) {
		S[j] = pointList[first + j];
		}
	INT_vec_heapsort(S, len);
	INT_set_print(ost, S, len);
	ost << "_{" << len << "}" << endl;
	FREE_INT(S);
}

void partitionstack::print_classes_tex(ostream& ost)
{
	INT i;
	
	for (i = 0; i < ht; i++) {
		ost << "$";
		print_class_tex(ost, i);
		ost << "$\\\\" << endl;
		}	
}

void partitionstack::print_class_tex(ostream& ost, INT idx)
{
	INT first_column_element = startCell[1];
	INT first, len, j;
	INT *S;
	
	S = NEW_INT(n);
	first = startCell[idx];
	len = cellSize[idx];
	ost << "C_{" << idx << "} = ";
	for (j = 0; j < len; j++) {
		S[j] = pointList[first + j];
		}
	if (is_col_class(idx)) {
		for (j = 0; j < len; j++) {
			S[j] -= first_column_element;
			}
		}
	INT_vec_heapsort(S, len);
	ost << "\\{ ";
	for (j = 0; j < len; j++) {
		ost << S[j];
		if (j < len - 1)
			ost << ", ";
		}
	ost << " \\}";
	ost << "_{" << len << "}" << endl;
	FREE_INT(S);
}

void partitionstack::print_class_point_or_line(ostream& ost, INT idx)
{
	INT first_column_element = startCell[1];
	INT first, len, j;
	INT *S;
	
	S = NEW_INT(n);
	first = startCell[idx];
	len = cellSize[idx];
	ost << "C_{" << idx << "} of size " << len << " descendant of C_{" << parent[idx] << "} is ";
	for (j = 0; j < len; j++) {
		S[j] = pointList[first + j];
		}
	if (is_col_class(idx)) {
		for (j = 0; j < len; j++) {
			S[j] -= first_column_element;
			}
		}
	INT_vec_heapsort(S, len);
	//INT_set_print(ost, S, len);
	ost << "{ ";
	for (j = 0; j < len; j++) {
		if (is_col_class(idx)) {
			ost << "\\ell_{";
			}
		else {
			ost << "P_{";
			}
		ost << S[j] << "}";
		if (j < len - 1)
			ost << ", ";
		}
	ost << " }";
	ost << "_{" << len << "}" << endl;
	FREE_INT(S);
}

void partitionstack::print_classes(ostream& ost)
{
	INT i;
	
	for (i = 0; i < ht; i++) {
		print_class(ost, i);
		}
}

void partitionstack::print_classes_points_and_lines(ostream& ost)
{
	INT i;
	
	for (i = 0; i < ht; i++) {
		print_class_point_or_line(ost, i);
		}
}

ostream& partitionstack::print(ostream& ost)
{
	INT i, j, first, len, a, pt, prev_pt, j0;
	INT f_erroneous2 = FALSE;

	//check();
	//ost << "partitionstack of height " << ht << " : ";
	ost << "( ";
	for (i = 0; i < ht; i++) {
		first = startCell[i];
		len = cellSize[i];
		//ost << "C_{" << i << "} of size " << len << " descendant of " << parent[i] << " is ";
		//ost << "{ ";
		j0 = 0;
		for (j = 1; j <= len; j++) {
			prev_pt = pointList[first + j - 1];
			if (j == len || pointList[first + j] != prev_pt + 1) {
				pt = pointList[first + j];
				if (j0 == j - 1) {
					cout << prev_pt;
					if (j < len)
						cout << ", ";
					}
				else {
					cout << pointList[first + j0] << "-" << prev_pt;
					if (j < len)
						cout << ", ";
					}
				j0 = j;
				}
			if (j < len) {
				a = pointList[first + j];
				if (invPointList[a] != first + j)
					f_erroneous2 = TRUE;
				}
			}
		//ost << " }_{" << len << "}";
		if (i < ht - 1)
			ost << "| ";
		}
	ost << " ) height " << ht << " class sizes: (";
	for (i = 0; i < ht; i++) {
		len = cellSize[i];
		ost << len;
		if (i < ht - 1)
			ost << ",";
		}
	ost << ") parent: ";
	for (i = 0; i < ht; i++) {
		ost << parent[i] << " ";
		}
	ost << endl;
	if (f_erroneous2) {
		cout << "erroneous partition stack: invPointList corrupt"
			<< endl;
		exit(1);
		}
	return ost;
}

void partitionstack::print_cell(INT i)
{
	INT j, first, len;

	first = startCell[i];
	len = cellSize[i];
	cout << "{ ";
	for (j = 0; j < len; j++) {
		cout << pointList[first + j];
		if (j < len - 1)
			cout << ", ";
		}
	cout << " }";
}

void partitionstack::print_cell_latex(ostream &ost, INT i)
{
	INT j, first, len;

	first = startCell[i];
	len = cellSize[i];
	ost << "\\{ ";
	for (j = 0; j < len; j++) {
		ost << pointList[first + j];
		if (j < len - 1)
			ost << ", ";
		}
	ost << " \\}";
}

void partitionstack::write_cell_to_file(INT i, BYTE *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT j, first, len;
	INT *set;

	if (f_v) {
		cout << "partitionstack::write_cell_to_file writing cell " << i << " to file " << fname << endl;
		}
	first = startCell[i];
	len = cellSize[i];
	set = NEW_INT(len);
	for (j = 0; j < len; j++) {
		set[j] = pointList[first + j];
		}
	write_set_to_file(fname, set, len, verbose_level - 1);
	FREE_INT(set);
}

void partitionstack::write_cell_to_file_points_or_lines(INT i, BYTE *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT j, first, len, m = 0;
	INT *set;

	if (f_v) {
		cout << "partitionstack::write_cell_to_file_points_or_lines writing cell " << i << " to file " << fname << endl;
		}
	if (is_col_class(i)) {
		m = startCell[1];
		}
	first = startCell[i];
	len = cellSize[i];
	set = NEW_INT(len);
	for (j = 0; j < len; j++) {
		set[j] = pointList[first + j] - m;
		}
	write_set_to_file(fname, set, len, verbose_level - 1);
	FREE_INT(set);
}

void partitionstack::print_subset()
{
#ifdef SPLIT_MULTIPLY
	INT i;

	for (i = 0; i < nb_subsets; i++) {
		INT_set_print(subsets + subset_first[i],
			subset_length[i]);
		if (i < nb_subsets - 1)
			cout << ", ";
		}
#else
	INT_set_print(subset, subset_size);
#endif
}

void partitionstack::refine_arbitrary_set(INT size, INT *set, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *set2;
	INT i, sz, sz2, a, c, d;
	
	if (f_v) {
		cout << "partitionstack::refine_arbitrary_set" << endl;
		if (f_vv) {
			cout << "set: ";
			INT_vec_print(cout, set, size);
			cout << endl;
			}
		}
	set2 = NEW_INT(size);
	for (i = 0; i < size; i++) {
		set2[i] = set[i];
		}
	sz = size;
	while (sz) {
		a = set2[0];
		c = cellNumber[invPointList[a]];
		subset[0] = a;
		subset_size = 1;
		sz2 = 0;
		for (i = 1; i < sz; i++) {
			a = set2[i];
			d = cellNumber[invPointList[a]];
			if (c == d) {
				subset[subset_size++] = a;
				}
			else {
				set2[sz2++] = a;
				}
			}
		if (subset_size < cellSize[c]) {
			split_cell(FALSE);
			}
		sz = sz2;
		}
	
	FREE_INT(set2);
	if (f_v) {
		cout << "partitionstack::refine_arbitrary_set finished" << endl;
		}
}


void partitionstack::split_cell(INT verbose_level)
{
#ifdef SPLIT_MULTIPLY
	INT i;

	for (i = 0; i < nb_subsets; i++) {
		split_cell(subsets + subset_first[i], subset_length[i], verbose_level);
		}
#else
	split_cell(subset, subset_size, verbose_level);
#endif
}

void partitionstack::split_multiple_cells(INT *set, INT set_size, INT f_front, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *f_done;
	INT *cell_nb;
	INT *set2;
	INT set2_sz;
	INT nb_done;
	INT i, a, pos_a, c;

	if (f_v) {
		cout << "split_multiple_cells() for subset { ";
		for (i = 0; i < set_size; i++) {
			cout << set[i] << " ";
			}
		cout << "}" << endl;
		}
	f_done = NEW_INT(set_size);
	cell_nb = NEW_INT(set_size);
	set2 = NEW_INT(set_size);
	
	for (i = 0; i < set_size; i++) {
		f_done[i] = FALSE;
		}
	for (i = 0; i < set_size; i++) {
		a = set[i];
		pos_a = invPointList[a];
		c = cellNumber[pos_a];
		cell_nb[i] = c;
		}
	if (f_v) {
		cout << "cell_nb : ";
		INT_vec_print(cout, cell_nb, set_size);
		cout << endl;
		}
	nb_done = 0;
	while (nb_done < set_size) {
		for (i = 0; i < set_size; i++) {
			if (!f_done[i]) {
				break;
				}
			}
		// now we split the set containing set[i]
		c = cell_nb[i];
		set2_sz = 0;
		for (; i < set_size; i++) {
			if (!f_done[i] && cell_nb[i] == c) {
				set2[set2_sz++] = set[i];
				nb_done++;
				f_done[i] = TRUE;
				}
			}
		if (f_vv) {
			cout << "splitting set of size " << set2_sz << " which is ";
			INT_vec_print(cout, set2, set2_sz);
			cout << " from class " << c << endl;
			}
		split_cell_front_or_back(set2, set2_sz, f_front, verbose_level - 2);
		if (f_vv) {
			cout << "after split:" << endl;
			print_classes_points_and_lines(cout);
			}
		}
	FREE_INT(f_done);
	FREE_INT(cell_nb);
	FREE_INT(set2);
}

void partitionstack::split_line_cell_front_or_back(INT *set, INT set_size, INT f_front, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT first_column_element = startCell[1];
	INT *set2, i;

	if (f_v) {
		cout << "partitionstack::split_line_cell_front_or_back" << endl;
		}
	set2 = NEW_INT(set_size);
	for (i = 0; i < set_size; i++) {
		set2[i] = set[i] + first_column_element;
		}
	split_cell_front_or_back(set2, set_size, f_front, verbose_level);
	FREE_INT(set2);
}

void partitionstack::split_cell_front_or_back(INT *set, INT set_size, INT f_front, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT a, b, c, f, l, j, pos_a, new_pos, i;

	if (f_v) {
		cout << "split_cell_front_or_back() for subset { ";
		for (i = 0; i < set_size; i++) {
			cout << set[i] << " ";
			}
		cout << "}" << endl;
		}
	check();
	if (set_size <= 0) {
		cout << "partitionstack::split_cell_front_or_back() set_size <= 0" << endl;
		exit(1);
		}
	a = set[0];
	pos_a = invPointList[a];
	c = cellNumber[pos_a];
	f = startCell[c];
	l = cellSize[c];
	if (f_vv) {
		cout << "split_cell_front_or_back() c=" << c << " f=" << f
			<< " l=" << l << endl;
		}
	if (set_size == l) {
		// nothing to do
		return;
		}
	else if (set_size > l) {
		cout << "split_cell_front_or_back() subset_size > cellSize" << endl;
		cout << "split_cell_front_or_back() for subset { ";
		for (i = 0; i < set_size; i++) {
			cout << set[i] << " ";
			}
		cout << "}" << endl;
		cout << *this << endl;
		exit(1);
		}
	for (j = 0; j < set_size; j++) {
		a = set[set_size - 1 - j];
		pos_a = invPointList[a];
		if (f_front) {
			new_pos = f + j;
			}
		else {
			new_pos = f + l - 1 - j;
			}
		if (FALSE /*f_vv*/) {
			cout << "split_cell_front_or_back: a=" << a
				<< " pos_a=" << pos_a << " new_pos="
				<< new_pos << endl;
			}
		if (pos_a != new_pos) {
			b = pointList[new_pos];
			pointList[new_pos] = a;
			invPointList[a] = new_pos;
			pointList[pos_a] = b;
			invPointList[b] = pos_a;
			}
		if (f_front) {
			//cellNumber[pos_a] = ht;
			}
		else {
			cellNumber[new_pos] = ht;
			}
		}

	if (f_front) {
		cellSize[c] = set_size;
		startCell[ht] = f + set_size;
		cellSize[ht] = l - set_size;
		for (j = 0; j < l - set_size; j++) {
			cellNumber[f + set_size + j] = ht;
			}
		}
	else {
		cellSize[c] = l - set_size;
		// cout << "cellSize[c]=" << cellSize[c] << endl;

		parent[ht] = c;
		startCell[ht] = f + l - set_size;
		cellSize[ht] = set_size;
		}
	parent[ht] = c;
	ht++;
	if (f_v) {
		cout << "split_cell_front_or_back() done" << endl;
		}
}

void partitionstack::split_cell(INT *set, INT set_size, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "partitionstack::split_cell" << endl;
		}
	split_cell_front_or_back(set, set_size, FALSE, verbose_level);
}

void partitionstack::join_cell()
{
	INT i, f1, f2, l1, l2, p;

	ht--;
	f2 = startCell[ht];
	l2 = cellSize[ht];
	p = parent[ht];
	f1 = startCell[p];
	l1 = cellSize[p];
	if (f1 + l1 != f2) {
		cout << "partitionstack::join_cell() f1 + l1 != f2" << endl;
		cout << "cell = " << p << endl;
		cout << *this << endl;
		exit(1);
		}
	for (i = 0; i < l2; i++) {
		cellNumber[f2 + i] = p;
		}
	cellSize[p] += l2;
}

void partitionstack::reduce_height(INT ht0)
{
	while (ht > ht0) {
		join_cell();
		}
}

void partitionstack::isolate_point(INT pt)
{
#ifdef SPLIT_MULTIPLY
	nb_subsets = 1;
	subset_first[0] = 0;
	subset_first[1] = 1;
	subset_length[0] = 1;
	subsets[0] = pt;
#else
	subset_size = 1;
	subset[0] = pt;
#endif
}

void partitionstack::subset_continguous(INT from, INT len)
{
#ifdef SPLIT_MULTIPLY
	INT i;
	nb_subsets = 1;
	subset_first[0] = 0;
	subset_first[1] = len;
	subset_length[0] = len;
	for (i = 0; i < len; i++)
		subsets[i] = from + i;
#else
	for (subset_size = 0; subset_size < len; subset_size++)
		subset[subset_size] = from + subset_size;
#endif
}

INT partitionstack::is_row_class(INT c)
{
	INT first_column_element = startCell[1];

	if (c >= ht) {
		cout << "partitionstack::is_row_class c >= ht, fatal" << endl;
		exit(1);
		}
	if (pointList[startCell[c]] >= first_column_element)
		return FALSE;
	else
		return TRUE;
}

INT partitionstack::is_col_class(INT c)
{
	if (c >= ht) {
		cout << "partitionstack::is_col_class c >= ht, fatal" << endl;
		exit(1);
		}
	if (is_row_class(c)) {
		return FALSE;
		}
	else {
		return TRUE;
		}
}

void partitionstack::allocate_and_get_decomposition(
	INT *&row_classes, INT *&row_class_inv, INT &nb_row_classes,
	INT *&col_classes, INT *&col_class_inv, INT &nb_col_classes, 
	INT verbose_level)
{
	INT i, c;
	
	row_classes = NEW_INT(ht);
	col_classes = NEW_INT(ht);
	row_class_inv = NEW_INT(ht);
	col_class_inv = NEW_INT(ht);
	get_row_and_col_classes(row_classes, nb_row_classes, col_classes, nb_col_classes, verbose_level - 1);
	for (i = 0; i < ht; i++) {
		row_class_inv[i] = col_class_inv[i] = -1;
		}
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		row_class_inv[c] = i;
		}
	for (i = 0; i < nb_col_classes; i++) {
		c = col_classes[i];
		col_class_inv[c] = i;
		}
}

void partitionstack::get_row_and_col_permutation(
	INT *row_classes, INT nb_row_classes,
	INT *col_classes, INT nb_col_classes, 
	INT *row_perm, INT *row_perm_inv, 
	INT *col_perm, INT *col_perm_inv)
{
	INT i, j, c, a, f, l, pos;
	INT first_column_element = startCell[1];

	pos = 0;
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		f = startCell[c];
		l = cellSize[c];
		for (j = 0; j < l; j++) {
			a = pointList[f + j];
			row_perm_inv[pos] = a;
			row_perm[a] = pos;
			pos++;
			}
		}
	pos = 0;
	for (i = 0; i < nb_col_classes; i++) {
		c = col_classes[i];
		f = startCell[c];
		l = cellSize[c];
		for (j = 0; j < l; j++) {
			a = pointList[f + j] - first_column_element;
			col_perm_inv[pos] = a;
			col_perm[a] = pos;
			pos++;
			}
		}
}

void partitionstack::get_row_and_col_classes(
	INT *row_classes, INT &nb_row_classes,
	INT *col_classes, INT &nb_col_classes, INT verbose_level)
{
	INT i, c, l;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "partitionstack::get_row_and_col_classes n = " << n << endl;
		}
	nb_row_classes = 0;
	nb_col_classes = 0;
#if 0
	for (c = 0; c < ht; c++) {
		if (is_row_class(c)) {
			row_classes[nb_row_classes++] = c;
			}
		else {
			col_classes[nb_col_classes++] = c;
			}
		}
#endif
	i = 0;
	while (i < n) {
		c = cellNumber[i];
		if (f_vv) {
			cout << i << " : " << c << endl;
			}
		if (is_row_class(c)) {
			row_classes[nb_row_classes++] = c;
			}
		else {
			col_classes[nb_col_classes++] = c;
			}
		l = cellSize[c];
		i += l;
		}
}

void partitionstack::initial_matrix_decomposition(INT nbrows, INT nbcols,
	INT *V, INT nb_V, INT *B, INT nb_B, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, l;

	if (f_v) {
		cout << "partitionstack::initial_matrix_decomposition before: " << endl << *this << endl;
		}

	// split rows and columns
	subset_continguous(nbrows, nbcols);
	split_cell(FALSE);

	l = V[0];
	for (i = 1; i < nb_V; i++) {
		subset_continguous(l, nbrows - l);
		split_cell(FALSE);
		l += V[i];
		}

	l = B[0];
	for (i = 1; i < nb_B; i++) {
		subset_continguous(nbrows + l, nbcols - l);
		split_cell(FALSE);
		l += B[i];
		}

	if (f_v) {
		cout << "partitionstack::initial_matrix_decomposition after" << endl << *this << endl;
		}
}

INT partitionstack::is_descendant_of(INT cell, INT ancestor_cell, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT c;

	if (f_v) {
		cout << "is_descendant_of: cell=" << cell << endl;
		}
	c = cell;
	if (cell == ancestor_cell) {
		if (f_v) {
			cout << "is_descendant_of: cell == ancestor_cell, so yes" << endl;
			}
		return TRUE;
		}
	while (parent[c] != c) {
		c = parent[c];
		if (f_v) {
			cout << "is_descendant_of: c=" << c << endl;
			}
		if (c == ancestor_cell) {
			if (f_v) {
				cout << "is_descendant_of: c == ancestor_cell, so yes" << endl;
				}
			return TRUE;
			}
		}
	if (f_v) {
		cout << "is_descendant_of: parent[c] == c, so no" << endl;
		}
	return FALSE;
}

INT partitionstack::is_descendant_of_at_level(INT cell, INT ancestor_cell, INT level, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT c;

	if (f_v) {
		cout << "is_descendant_of_at_level: cell=" << cell << " ancestor_cell = " << ancestor_cell << " level = " << level << endl;
		}
	if (cell == ancestor_cell) {
		if (f_v) {
			cout << "is_descendant_of_at_level: cell == ancestor_cell, so yes" << endl;
			}
		return TRUE;
		}
	c = cell;
	if (c < level) {
		if (f_v) {
			cout << "is_descendant_of_at_level: c < level, so no" << endl;
			}
		return FALSE;
		}
	while (parent[c] != c) {
		c = parent[c];
		if (f_v) {
			cout << "is_descendant_of_at_level: c=" << c << endl;
			}
		if (c == ancestor_cell) {
			if (f_v) {
				cout << "is_descendant_of_at_level: c == ancestor_cell, so yes" << endl;
				}
			return TRUE;
			}
		if (c < level) {
			if (f_v) {
				cout << "is_descendant_of_at_level: c < level, so no" << endl;
				}
			return FALSE;
			}
		}
	if (f_v) {
		cout << "is_descendant_of_at_level: parent[c] == c, so no" << endl;
		}
	return FALSE;
}

INT partitionstack::cellSizeAtLevel(INT cell, INT level)
{
	INT i, s, S = 0;

	for (i = level; i < ht; i++) {
		if (is_descendant_of_at_level(i, cell, level, FALSE)) {
			//cout << "cell " << i << " of size " << cellSize[i] << " is a descendant of cell " << cell << endl;
			s = cellSize[i];
			S += s;
			}
		}
	if (cell < level)
		S += cellSize[cell];
	return S;
}

// TDO for orthogonal:

INT partitionstack::compute_TDO(orthogonal &O, INT ht0, 
	INT marker1, INT marker2, INT depth, INT verbose_level)
{
	INT h1, h2, ht1, remaining_depth = depth;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	
	ht1 = ht;
	h2 = 0;
	
	if (f_v) {
		cout << "compute_TDO depth=" << depth << " ht=" << ht << endl;
		}
	if (remaining_depth) {
		if (f_vv) {
			cout << "refine_column_partition ht0=" << ht0 << " ht1=" << ht1 << endl;
			}
		h1 = refine_column_partition(O, ht0, verbose_level - 3);
		//cout << "h1=" << h1 << endl;
		h2 = hashing(h1, h2);
		if (f_v) {
			cout << "after refine_column_partition ht=" << ht << endl;
			}
		if (f_vv) {
			get_and_print_col_decomposition_scheme(O, marker1, marker2);
			}	
		if (f_vvv) {
			print_classes(cout);
			}
		//cout << "h2=" << h2 << endl;

		//cout << "after refinement:" << endl;
		//cout << *this << endl;
		remaining_depth--;
		}

	while (remaining_depth) {

		if (remaining_depth) {
			ht0 = ht1;
			ht1 = ht;
		
			if (f_vv) {
				cout << "refine_column_partition ht0=" << ht0 << " ht1=" << ht1 << endl;
				}
			h1 = refine_row_partition(O, ht0, verbose_level - 3);
			//cout << "h1=" << h1 << endl;
			h2 = hashing(h1, h2);
			//cout << "h2=" << h2 << endl;
			if (f_v) {
				cout << "after refine_row_partition ht=" << ht << endl;
				}
			if (f_vv) {
				get_and_print_row_decomposition_scheme(O, marker1, marker2);
				}
			if (f_vvv) {
				print_classes(cout);
				}
			remaining_depth--;
			}
		
		if (ht == ht1 || remaining_depth == 0)
			break;

		if (remaining_depth) {
			ht0 = ht1;
			ht1 = ht;
		
			if (f_vv) {
				cout << "refine_column_partition ht0=" << ht0 << " ht1=" << ht1 << endl;
				}
			h1 = refine_column_partition(O, ht0, verbose_level - 3);
			//cout << "h1=" << h1 << endl;
			h2 = hashing(h1, h2);
			//cout << "h2=" << h2 << endl;
			if (f_v) {
				cout << "after refine_column_partition ht=" << ht << endl;
				}
			if (f_vv) {
				get_and_print_col_decomposition_scheme(O, marker1, marker2);
				}
			if (f_vvv) {
				print_classes(cout);
				}
			remaining_depth--;
		
			if (ht == ht1)
				break;
			}
		}

	//cout << "hash = " << h2 << endl;
	
	if (f_v) {
		cout << "compute_TDO finished, hash=" << h2 << endl;
		//get_and_print_decomposition_schemes(O, marker1, marker2);
		}	
	return h2;
}

void partitionstack::get_and_print_row_decomposition_scheme(orthogonal &O, 
	INT marker1, INT marker2)
{
	INT *row_classes, *row_class_inv, nb_row_classes;
	INT *col_classes, *col_class_inv, nb_col_classes;
	INT *row_scheme;
	INT f_v = FALSE;
	
	cout << "computing row scheme" << endl;
	allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		f_v);
	
	row_scheme = NEW_INT(nb_row_classes * nb_col_classes);

	get_row_decomposition_scheme(O, 
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		row_scheme, f_v);

	//cout << *this << endl;
	
	cout << "row_scheme:" << endl;
	print_decomposition_scheme(cout, 
		row_classes, nb_row_classes,
		col_classes, nb_col_classes, 
		row_scheme, marker1, marker2);

	FREE_INT(row_classes);
	FREE_INT(row_class_inv);
	FREE_INT(col_classes);
	FREE_INT(col_class_inv);
	FREE_INT(row_scheme);
}

void partitionstack::get_and_print_col_decomposition_scheme(orthogonal &O, 
	INT marker1, INT marker2)
{
	INT *row_classes, *row_class_inv, nb_row_classes;
	INT *col_classes, *col_class_inv, nb_col_classes;
	INT *col_scheme;
	INT f_v = FALSE;
	
	cout << "computing col scheme" << endl;
	allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		f_v);
	
	col_scheme = NEW_INT(nb_row_classes * nb_col_classes);

	get_col_decomposition_scheme(O, 
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		col_scheme, f_v);

	//cout << *this << endl;
	
	cout << "col_scheme:" << endl;
	print_decomposition_scheme(cout, 
		row_classes, nb_row_classes,
		col_classes, nb_col_classes, 
		col_scheme, marker1, marker2);

	FREE_INT(row_classes);
	FREE_INT(row_class_inv);
	FREE_INT(col_classes);
	FREE_INT(col_class_inv);
	FREE_INT(col_scheme);
}

void partitionstack::get_and_print_decomposition_schemes(orthogonal &O, 
	INT marker1, INT marker2)
{
	INT *row_classes, *row_class_inv, nb_row_classes;
	INT *col_classes, *col_class_inv, nb_col_classes;
	INT *row_scheme, *col_scheme;
	INT f_v = FALSE;
	
	cout << "computing both schemes" << endl;
	allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		f_v);
	
	row_scheme = NEW_INT(nb_row_classes * nb_col_classes);
	col_scheme = NEW_INT(nb_row_classes * nb_col_classes);

	get_row_decomposition_scheme(O, 
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		row_scheme, f_v);

	row_scheme_to_col_scheme(O, 
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		row_scheme, col_scheme, f_v);
	
	//cout << *this << endl;
	
	cout << "row_scheme:" << endl;
	print_decomposition_scheme(cout, 
		row_classes, nb_row_classes,
		col_classes, nb_col_classes, 
		row_scheme, marker1, marker2);

	cout << "col_scheme:" << endl;
	print_decomposition_scheme(cout, 
		row_classes, nb_row_classes,
		col_classes, nb_col_classes, 
		col_scheme, marker1, marker2);

	FREE_INT(row_classes);
	FREE_INT(row_class_inv);
	FREE_INT(col_classes);
	FREE_INT(col_class_inv);
	FREE_INT(row_scheme);
	FREE_INT(col_scheme);
}

void partitionstack::print_decomposition_tex(ostream &ost, 
	INT *row_classes, INT nb_row_classes,
	INT *col_classes, INT nb_col_classes)
{
	INT i, j, c, f, l, a;
	INT first_column_element = startCell[1];
	
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		f = startCell[c];
		l = cellSize[c];
		ost << "${\\cal V}_" << i << " = \\{";
		for (j = 0; j < l; j++) {
			a = pointList[f + j];
			ost << a;
			if (j < l - 1) {
				ost << ", ";
				}
			if ((j + 1) % 25 == 0) {
				ost << "\\\\" << endl;
				}
			}
		ost << "\\}$ of size " << l << "\\\\" << endl;
		}
	for (i = 0; i < nb_col_classes; i++) {
		c = col_classes[i];
		f = startCell[c];
		l = cellSize[c];
		ost << "${\\cal B}_" << i << " = \\{";
		for (j = 0; j < l; j++) {
			a = pointList[f + j] - first_column_element;
			ost << a;
			if (j < l - 1) {
				ost << ", ";
				}
			if ((j + 1) % 25 == 0) {
				ost << "\\\\" << endl;
				}
			}
		ost << "\\}$ of size " << l << "\\\\" << endl;
		}
}

void partitionstack::print_decomposition_scheme(ostream &ost, 
	INT *row_classes, INT nb_row_classes,
	INT *col_classes, INT nb_col_classes, 
	INT *scheme, INT marker1, INT marker2)
{
	INT c, i, j;
	
	if (marker1 >= 0) ost << "  ";
	if (marker2 >= 0) ost << "  ";

	ost << "             | ";
	for (j = 0; j < nb_col_classes; j++) {
		c = col_classes[j];
		ost << setw(6) << cellSize[c] << "_{" << setw(3) << c << "}";
		}
	ost << endl;

	if (marker1 >= 0) ost << "--";
	if (marker2 >= 0) ost << "--";
	ost << "---------------";
	for (i = 0; i < nb_col_classes; i++) {
		ost << "------------";
		}
	ost << endl;
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		ost << setw(6) << cellSize[c] << "_{" << setw(3) << c << "}";
		if (marker1 >= 0) {
			if (is_descendant_of(c, marker1, 0)) {
				ost << " *";
				}
			else
				ost << "  ";
			}
		if (marker2 >= 0) {
			if (is_descendant_of(c, marker2, 0)) {
				ost << " *";
				}
			else
				ost << "  ";
			}
		ost << " | ";
		//f = P.startCell[c];
		for (j = 0; j < nb_col_classes; j++) {
			ost << setw(12) << scheme[i * nb_col_classes + j];
			}
		ost << endl;
		}
	ost << endl;
	ost << endl;
}

void partitionstack::print_decomposition_scheme_tex(ostream &ost, 
	INT *row_classes, INT nb_row_classes,
	INT *col_classes, INT nb_col_classes, 
	INT *scheme)
{
	INT c, i, j;
	
	ost << "\\begin{align*}" << endl;
	ost << "\\begin{array}{r|*{" << nb_col_classes << "}{r}}" << endl;
	ost << " ";
	for (j = 0; j < nb_col_classes; j++) {
		ost << " & ";
		c = col_classes[j];
		ost << setw(6) << cellSize[c] << "_{" << setw(3) << c << "}";
		}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		ost << setw(6) << cellSize[c] << "_{" << setw(3) << c << "}";
		//f = P.startCell[c];
		for (j = 0; j < nb_col_classes; j++) {
			ost << " & " << setw(12) << scheme[i * nb_col_classes + j];
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
	ost << "\\end{align*}" << endl;
}

void partitionstack::print_tactical_decomposition_scheme_tex(ostream &ost, 
	INT *row_classes, INT nb_row_classes,
	INT *col_classes, INT nb_col_classes, 
	INT *row_scheme, INT *col_scheme, INT f_print_subscripts)
{
	print_tactical_decomposition_scheme_tex_internal(ost, TRUE, 
		row_classes, nb_row_classes,
		col_classes, nb_col_classes,
		row_scheme, col_scheme, f_print_subscripts);
}

void partitionstack::print_tactical_decomposition_scheme_tex_internal(
	ostream &ost, INT f_enter_math_mode, 
	INT *row_classes, INT nb_row_classes,
	INT *col_classes, INT nb_col_classes, 
	INT *row_scheme, INT *col_scheme, INT f_print_subscripts)
{
	INT c, i, j;
	
	if (f_enter_math_mode) {
		ost << "\\begin{align*}" << endl;
		}
	ost << "\\begin{array}{r|*{" << nb_col_classes << "}{r}}" << endl;
	ost << " ";
	for (j = 0; j < nb_col_classes; j++) {
		ost << " & ";
		c = col_classes[j];
		ost << setw(6) << cellSize[c];
		if (f_print_subscripts) {
			ost << "_{" << setw(3) << c << "}";
			}
		}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		ost << setw(6) << cellSize[c];
			if (f_print_subscripts) {
				ost << "_{" << setw(3) << c << "}";
				}
		//f = P.startCell[c];
		for (j = 0; j < nb_col_classes; j++) {
			ost << " & " << setw(12) << row_scheme[i * nb_col_classes + j] 
				<< "\\backslash " << col_scheme[i * nb_col_classes + j];
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
	if (f_enter_math_mode) {
		ost << "\\end{align*}" << endl;
		}
}

void partitionstack::print_row_tactical_decomposition_scheme_tex(
	ostream &ost, INT f_enter_math_mode, 
	INT *row_classes, INT nb_row_classes,
	INT *col_classes, INT nb_col_classes, 
	INT *row_scheme, INT f_print_subscripts)
{
	INT c, i, j;
	
	if (f_enter_math_mode) {
		ost << "\\begin{align*}" << endl;
		}
	ost << "\\begin{array}{r|*{" << nb_col_classes << "}{r}}" << endl;
	ost << "\\rightarrow ";
	for (j = 0; j < nb_col_classes; j++) {
		ost << " & ";
		c = col_classes[j];
		ost << setw(6) << cellSize[c];
		if (f_print_subscripts) {
			ost << "_{" << setw(3) << c << "}";
			}
		}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		ost << setw(6) << cellSize[c];
			if (f_print_subscripts) {
				ost << "_{" << setw(3) << c << "}";
				}
		//f = P.startCell[c];
		for (j = 0; j < nb_col_classes; j++) {
			ost << " & " << setw(12) << row_scheme[i * nb_col_classes + j];
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
	if (f_enter_math_mode) {
		ost << "\\end{align*}" << endl;
		}
}

void partitionstack::print_column_tactical_decomposition_scheme_tex(
	ostream &ost, INT f_enter_math_mode, 
	INT *row_classes, INT nb_row_classes,
	INT *col_classes, INT nb_col_classes, 
	INT *col_scheme, INT f_print_subscripts)
{
	INT c, i, j;
	
	if (f_enter_math_mode) {
		ost << "\\begin{align*}" << endl;
		}
	ost << "\\begin{array}{r|*{" << nb_col_classes << "}{r}}" << endl;
	ost << "\\downarrow ";
	for (j = 0; j < nb_col_classes; j++) {
		ost << " & ";
		c = col_classes[j];
		ost << setw(6) << cellSize[c];
		if (f_print_subscripts) {
			ost << "_{" << setw(3) << c << "}";
			}
		}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		ost << setw(6) << cellSize[c];
			if (f_print_subscripts) {
				ost << "_{" << setw(3) << c << "}";
				}
		//f = P.startCell[c];
		for (j = 0; j < nb_col_classes; j++) {
			ost << " & " << setw(12) << col_scheme[i * nb_col_classes + j];
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
	if (f_enter_math_mode) {
		ost << "\\end{align*}" << endl;
		}
}

void partitionstack::print_non_tactical_decomposition_scheme_tex(
	ostream &ost, INT f_enter_math_mode, 
	INT *row_classes, INT nb_row_classes,
	INT *col_classes, INT nb_col_classes, 
	INT f_print_subscripts)
{
	INT c, i, j;
	
	if (f_enter_math_mode) {
		ost << "\\begin{align*}" << endl;
		}
	ost << "\\begin{array}{r|*{" << nb_col_classes << "}{r}}" << endl;
	ost << " ";
	for (j = 0; j < nb_col_classes; j++) {
		ost << " & ";
		c = col_classes[j];
		ost << setw(6) << cellSize[c];
		if (f_print_subscripts) {
			ost << "_{" << setw(3) << c << "}";
			}
		}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		ost << setw(6) << cellSize[c];
			if (f_print_subscripts) {
				ost << "_{" << setw(3) << c << "}";
				}
		//f = P.startCell[c];
		for (j = 0; j < nb_col_classes; j++) {
			ost << " & ";
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
	if (f_enter_math_mode) {
		ost << "\\end{align*}" << endl;
		}
}

void partitionstack::row_scheme_to_col_scheme(orthogonal &O, 
	INT *row_classes, INT *row_class_inv, INT nb_row_classes,
	INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
	INT *row_scheme, INT *col_scheme, INT verbose_level)
{
	INT I, J, c1, l1, c2, l2, a, b, c;
	
	for (I = 0; I < nb_row_classes; I++) {
		c1 = row_classes[I];
		l1 = cellSize[c1];
		for (J = 0; J < nb_col_classes; J++) {
			c2 = col_classes[J];
			l2 = cellSize[c2];
			a = row_scheme[I * nb_col_classes + J];
			b = a * l1;
			if (b % l2) {
				cout << "row_scheme_to_col_scheme: cannot be tactical" << endl;
				exit(1);
				}
			c = b / l2;
			col_scheme[I * nb_col_classes + J] = c;
			}
		}
}

void partitionstack::get_row_decomposition_scheme(orthogonal &O, 
	INT *row_classes, INT *row_class_inv, INT nb_row_classes,
	INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
	INT *row_scheme, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT I, J, i, j, c1, f1, l1, x, y, u, c;
	INT *neighbors;
	INT *data0;
	INT *data1;
	
	if (f_v) {
		cout << "get_row_decomposition_scheme" << endl;
		}
	neighbors = NEW_INT(O.alpha);
	data0 = NEW_INT(nb_col_classes);
	data1 = NEW_INT(nb_col_classes);
	for (i = 0; i < nb_row_classes * nb_col_classes; i++) {
		row_scheme[i] = 0;
		}
	for (I = 0; I < nb_row_classes; I++) {
		c1 = row_classes[I];
		f1 = startCell[c1];
		l1 = cellSize[c1];
		for (j = 0; j < nb_col_classes; j++) 
			data0[j] = 0;
		for (i = 0; i < l1; i++) {
			x = pointList[f1 + i];
			for (J = 0; J < nb_col_classes; J++) 
				data1[J] = 0;
			O.lines_on_point_by_line_rank(x, neighbors, verbose_level - 2);
			for (u = 0; u < O.alpha; u++) {
				y = neighbors[u];
				j = O.nb_points + y;
				c = cellNumber[invPointList[j]];
				J = col_class_inv[c];
				data1[J]++;
				}
			if (i == 0) {
				for (J = 0; J < nb_col_classes; J++) {
					data0[J] = data1[J];
					}
				}
			else {
				for (J = 0; J < nb_col_classes; J++) {
					if (data0[J] != data1[J]) {
						cout << "not tactical I=" << I << " i=" << i << " J=" << J << endl;
						}
					}
				}
			} // next i
		for (J = 0; J < nb_col_classes; J++) {
			row_scheme[I * nb_col_classes + J] = data0[J];
			}
		}
	FREE_INT(neighbors);
	FREE_INT(data0);
	FREE_INT(data1);
}

void partitionstack::get_col_decomposition_scheme(orthogonal &O, 
	INT *row_classes, INT *row_class_inv, INT nb_row_classes,
	INT *col_classes, INT *col_class_inv, INT nb_col_classes, 
	INT *col_scheme, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT I, J, i, j, c1, f1, l1, x, y, u, c;
	INT *neighbors;
	INT *data0;
	INT *data1;
	
	if (f_v) {
		cout << "get_col_decomposition_scheme" << endl;
		}
	neighbors = NEW_INT(O.q + 1);
	data0 = NEW_INT(nb_row_classes);
	data1 = NEW_INT(nb_row_classes);
	for (i = 0; i < nb_row_classes * nb_col_classes; i++) {
		col_scheme[i] = 0;
		}
	for (J = 0; J < nb_col_classes; J++) {
		c1 = col_classes[J];
		f1 = startCell[c1];
		l1 = cellSize[c1];
		for (i = 0; i < nb_row_classes; i++) 
			data0[i] = 0;
		for (j = 0; j < l1; j++) {
			y = pointList[f1 + j] - O.nb_points;
			for (I = 0; I < nb_row_classes; I++) 
				data1[I] = 0;
			
			O.points_on_line_by_line_rank(y, neighbors, verbose_level - 2);
			
			for (u = 0; u < O.q + 1; u++) {
				x = neighbors[u];
				c = cellNumber[invPointList[x]];
				I = row_class_inv[c];
				data1[I]++;
				}
			if (j == 0) {
				for (I = 0; I < nb_row_classes; I++) {
					data0[I] = data1[I];
					}
				}
			else {
				for (I = 0; I < nb_row_classes; I++) {
					if (data0[I] != data1[I]) {
						cout << "not tactical J=" << J << " j=" << j << " I=" << I << endl;
						}
					}
				}
			} // next j
		for (I = 0; I < nb_row_classes; I++) {
			col_scheme[I * nb_col_classes + J] = data0[I];
			}
		}
	FREE_INT(neighbors);
	FREE_INT(data0);
	FREE_INT(data1);
}

INT partitionstack::refine_column_partition(orthogonal &O, INT ht0, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT row_cell, f, l, i, j, x, y, u, c, cell, N, first, next, ht1, depth, idx;
	INT *data;
	INT *neighbors, h;
	
	N = O.nb_points + O.nb_lines;
	ht1 = ht;
	depth = ht1 - ht0 + 1;
	if (f_v) {
		cout << "refine_column_partition ht0=" << ht0 << " ht=" << ht << " depth=" << depth << endl;
		}
	data = NEW_INT(N * depth);
	for (i = 0; i < N * depth; i++) 
		data[i] = 0;
	
	neighbors = NEW_INT(O.alpha);
	for (y = 0; y < O.nb_lines; y++) {
		j = O.nb_points + y;
		c = cellNumber[invPointList[j]];
		data[j * depth + 0] = c;
		}

	for (row_cell = ht0; row_cell < ht1; row_cell++) {
		idx = row_cell - ht0;
		f = startCell[row_cell];
		l = cellSize[row_cell];
		if (f_vvv) {
			cout << "refine_column_partition idx=" << idx 
				<< " row_cell=" << row_cell 
				<< " f=" << f << " l=" << l << endl;
			}
		if (!is_row_class(row_cell)) {
			cout << "row_cell is not a row cell" << endl;
			cout << "ht0=" << ht0 << endl;
			cout << "ht1=" << ht1 << endl;
			cout << *this << endl;
			exit(1);
			}
	
	
		for (i = 0; i < l; i++) {
			x = pointList[f + i];
			//if (f_v) {cout << i << " : " << x << " : ";}
			O.lines_on_point_by_line_rank(x, neighbors, 0/*verbose_level - 2*/);
			//if (f_v) {INT_vec_print(cout, neighbors, O.alpha);cout << endl;}
			for (u = 0; u < O.alpha; u++) {
				y = neighbors[u];
				j = O.nb_points + y;
				data[j * depth + 1 + idx]++;
				}
			}
		}
#if 0
	if (f_vvv) {
		cout << "data:" << endl;
		for (y = 0; y < O.nb_lines; y++) {
			j = O.nb_points + y;
			cout << y << " : " << j << " : ";
			INT_vec_print(cout, data + j * depth, depth);
			cout << endl;
			}
		cout << endl;
		}
#endif
		
	ht0 = ht;
	for (cell = 0; cell < ht0; cell++) {
		if (is_row_class(cell))
			continue;
			
		if (cellSize[cell] == 1)
			continue;
		first = startCell[cell];
		next = first + cellSize[cell];

		radix_sort(first /* left */,
				   next - 1 /* right */,
				   data, depth, 0 /*radix*/, FALSE);
		}
	if (f_vv) {
		cout << "after sorting, with " << ht - ht0 << " new classes" << endl;
		cout << *this << endl;
		}

	if (f_vvv) {
		print_column_refinement_info(ht0, data, depth);
		}
		
	h = hash_column_refinement_info(ht0, data, depth, 0);
	
	FREE_INT(data);
	FREE_INT(neighbors);
	return h;
}

INT partitionstack::refine_row_partition(orthogonal &O, INT ht0, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT col_cell, f, l, i, j, x, y, u, c, cell, N, first, next, ht1, depth, idx;
	INT *data;
	INT *neighbors, nb_neighbors, h;
	
	N = O.nb_points + O.nb_lines;
	ht1 = ht;
	depth = ht1 - ht0 + 1;
	if (f_v) {
		cout << "refine_row_partition ht0=" << ht0 << " ht=" << ht << " depth=" << depth << endl;
		}
	data = NEW_INT(N * depth);
	for (i = 0; i < N * depth; i++) 
		data[i] = 0;
	
	nb_neighbors = O.F->q + 1;
	neighbors = NEW_INT(nb_neighbors);
	for (x = 0; x < O.nb_points; x++) {
		i = x;
		c = cellNumber[invPointList[i]];
		data[i * depth + 0] = c;
		}

	for (col_cell = ht0; col_cell < ht1; col_cell++) {
		idx = col_cell - ht0;
		f = startCell[col_cell];
		l = cellSize[col_cell];
		if (f_vvv) {
			cout << "refine_row_partition idx=" << idx 
				<< " col_cell=" << col_cell 
				<< " f=" << f 
				<< " l=" << l << endl;
			}

		if (!is_col_class(col_cell)) {
			cout << "col_cell is not a col cell" << endl;
			cout << "ht0=" << ht0 << endl;
			cout << "ht1=" << ht1 << endl;
			cout << *this << endl;
			exit(1);
			}
	
	
	
		for (j = 0; j < l; j++) {
			y = pointList[f + j];
			//if (f_v) {cout << j << " : " << y << " : ";}
			
			O.points_on_line_by_line_rank(y - O.nb_points, neighbors, 0/* verbose_level - 2*/);
			
			//if (f_v) {INT_vec_print(cout, neighbors, O.alpha);cout << endl;}
			for (u = 0; u < nb_neighbors; u++) {
				x = neighbors[u];
				i = x;
				data[i * depth + 1 + idx]++;
				}
			}
		}
#if 0
	if (f_vvv) {
		cout << "data:" << endl;
		for (i = 0; i < O.nb_lines; i++) {
			cout << i << " : ";
			INT_vec_print(cout, data + i * depth, depth);
			cout << endl;
			}
		cout << endl;
		}
#endif
		
	ht0 = ht;
	for (cell = 0; cell < ht0; cell++) {
		if (is_col_class(cell))
			continue;
			
		if (cellSize[cell] == 1)
			continue;
		first = startCell[cell];
		next = first + cellSize[cell];

		radix_sort(first /* left */,
				   next - 1 /* right */,
				   data, depth, 0 /*radix*/, FALSE);
		}
	if (f_vv) {
		cout << "after sorting, with " << ht - ht0 << " new classes" << endl;
		cout << *this << endl;
		}

	if (f_vv) {
		print_row_refinement_info(ht0, data, depth);
		}

	h = hash_row_refinement_info(ht0, data, depth, 0);

	
	FREE_INT(data);
	FREE_INT(neighbors);
	return h;
}

INT partitionstack::hash_column_refinement_info(INT ht0, INT *data, INT depth, INT hash0)
{
	INT cell, i, j, first, len, ancestor;
	INT h;
	
	if (ht0 == ht) {
		h = hashing(hash0, 1);
		}
	else {
		h = hashing(hash0, 0);
		}
	
	for (cell = 0; cell < ht0; cell++) {
		if (is_row_class(cell))
			continue;
		first = startCell[cell];
		len = cellSize[cell];
		
		h = hashing(h, len);
		
		j = pointList[first];
		for (i = 0; i < depth; i++) {
			h = hashing(h, data[j * depth + i]);
			}
		}
	for (cell = ht0; cell < ht; cell++) {
		ancestor = parent_at_height(ht0, cell);
		h = hashing(h, ancestor);
		
		first = startCell[cell];
		len = cellSize[cell];
		
		h = hashing(h, len);

		j = pointList[first];
		for (i = 0; i < depth; i++) {
			h = hashing(h, data[j * depth + i]);
			}
		}
	return h;
}

INT partitionstack::hash_row_refinement_info(INT ht0, INT *data, INT depth, INT hash0)
{
	INT cell, i, j, first, len, ancestor;
	INT h;
	
	if (ht0 == ht) {
		h = hashing(hash0, 1);
		}
	else {
		h = hashing(hash0, 0);
		}
	for (cell = 0; cell < ht0; cell++) {
		if (is_col_class(cell))
			continue;
		first = startCell[cell];
		len = cellSize[cell];
		
		h = hashing(h, len);
		
		j = pointList[first];
		for (i = 0; i < depth; i++) {
			h = hashing(h, data[j * depth + i]);
			}
		}
	for (cell = ht0; cell < ht; cell++) {
		ancestor = parent_at_height(ht0, cell);
		h = hashing(h, ancestor);
		
		first = startCell[cell];
		len = cellSize[cell];
		
		h = hashing(h, len);

		j = pointList[first];
		for (i = 0; i < depth; i++) {
			h = hashing(h, data[j * depth + i]);
			}
		}
	return h;
}

void partitionstack::print_column_refinement_info(INT ht0, INT *data, INT depth)
{
	INT cell, j, first, ancestor;
	
	cout << "the old col parts:" << endl;
	for (cell = 0; cell < ht0; cell++) {
		if (is_row_class(cell))
			continue;
		first = startCell[cell];
		j = pointList[first];
		cout << "cell " << cell << " of size " << cellSize[cell] << " : ";
		INT_vec_print(cout, data + j * depth, depth);
		cout << " : ";
		INT_vec_print(cout, pointList + first, cellSize[cell]);
		cout << endl;
		}
	if (ht0 == ht) {
		cout << "no splitting" << endl;
		}
	else {
		cout << "the " << ht - ht0 << " new col parts that were split off are:" << endl;
		for (cell = ht0; cell < ht; cell++) {
			ancestor = parent_at_height(ht0, cell);
			first = startCell[cell];
			j = pointList[first];
			cout << "cell " << cell << " of size " << cellSize[cell] 
				<< " ancestor cell is " << ancestor << " : ";
			INT_vec_print(cout, data + j * depth, depth);
			cout << " : ";
			INT_vec_print(cout, pointList + first, cellSize[cell]);
			cout << endl;
			}
		}
}

void partitionstack::print_row_refinement_info(INT ht0, INT *data, INT depth)
{
	INT cell, j, first, ancestor;
	
	cout << "the old row parts:" << endl;
	for (cell = 0; cell < ht0; cell++) {
		if (is_col_class(cell))
			continue;
		first = startCell[cell];
		j = pointList[first];
		cout << "cell " << cell << " of size " << cellSize[cell] << " : ";
		INT_vec_print(cout, data + j * depth, depth);
		cout << " : ";
		INT_vec_print(cout, pointList + first, cellSize[cell]);
		cout << endl;
		}
	if (ht0 == ht) {
		cout << "no splitting" << endl;
		}
	else {
		cout << "the " << ht - ht0 << " new row parts that were split off are:" << endl;
		for (cell = ht0; cell < ht; cell++) {
			ancestor = parent_at_height(ht0, cell);
			first = startCell[cell];
			j = pointList[first];
			cout << "cell " << cell << " of size " << cellSize[cell] 
				<< " ancestor cell is " << ancestor << " : ";
			INT_vec_print(cout, data + j * depth, depth);
			cout << " : ";
			INT_vec_print(cout, pointList + first, cellSize[cell]);
			cout << endl;
			}
		}
}


void partitionstack::radix_sort(INT left, INT right, INT *C, 
	INT length, INT radix, INT verbose_level)
{
	INT ma, mi, i, lo, mask;
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "radix sort radix = " << radix << ", left = " << left << ", right = " << right << endl;
		}
	if (radix == length)
		return;
	if (left == right)
		return;
	ma = mi = C[pointList[left] * length + radix];
	for (i = left + 1; i <= right; i++)  {
		ma = MAXIMUM(ma, C[pointList[i] * length + radix]);
		mi = MINIMUM(mi, C[pointList[i] * length + radix]);
		}
	if (f_v) {
		cout << "radix sort radix=" << radix << ", minimum is " << mi << " maximum is " << ma << endl;
		}
	if (mi == ma) {
		radix_sort(left, right, C, length, radix + 1, verbose_level);
		return;
		}
	lo = my_log2(ma);
	if (f_v) {
		cout << "log2 = " << lo << endl;
		}
	mask = (1 << (lo - 1));
	if (f_v) {
		cout << "mask = " << mask << endl;
		}
	radix_sort_bits(left, right, C, length, radix, mask, verbose_level);
}

void partitionstack::radix_sort_bits(INT left, INT right, 
	INT *C, INT length, INT radix, INT mask, INT verbose_level)
{
	INT l, r, i, len;
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "partitionstack::radix_sort_bits() mask = " << mask
				<< " left=" << left << " right=" << right << endl;
		}
	if (left >= right)
		return;
	if (mask == 0) {
		radix_sort(left, right, C, length, radix + 1, verbose_level);
		return;
		}
	l = left;
	r = right;
	while (l < r) {
		while (l <= right) {
			if (!(C[pointList[l] * length + radix] & mask))
				break;
			l++;
			}
		while (r >= left) {
			if ((C[pointList[r] * length + radix] & mask))
				break;
			r--;
			}
		// now: everything in [left .. l -1] has the bit = 1
		// everything in [r+1,...,right] has the bit = 0
		if (l < r) {
			swap_ij(pointList, invPointList, l, r);
			}
		if (f_v) {
			cout << "l = " << l << " r = " << r << endl;
			}
		}

	// now l = r+1
	// if r = left - 1 then all elements had that bit equal to 0
	// if l = right + 1 then all elements had that bit equal to 1
	mask >>= 1;
	if (r == left - 1) {
		if (f_v) {
			cout << "radix_sort_bits no splitting, all bits 0" << endl;
			}
		radix_sort_bits(left, right, C, length, radix, mask, verbose_level);
		}
	else if (l == right + 1) {
		if (f_v) {
			cout << "radix_sort_bits no splitting, all bits 1" << endl;
			}
		radix_sort_bits(left, right, C, length, radix, mask, verbose_level);
		}
	else {
		if (f_v) {
			cout << "radix_sort_bits splitting l=" << l << " r=" << r << endl;
			}
		// we are splitting off the points in the interval [l..right]
		len = right - l + 1;
		for (i = 0; i < len; i++)
			subset[i] = pointList[l + i];
		//cout << "radix_sort split partition, len = " << len << endl;
		subset_size = len;
		split_cell(FALSE);

		radix_sort_bits(left, r, C, length, radix, mask, verbose_level);
		radix_sort_bits(l, right, C, length, radix, mask, verbose_level);
		}
}

void partitionstack::swap_ij(INT *perm, INT *perm_inv, INT i, INT j)
{
	INT tmp;

	tmp = perm[j];
	perm[j] = perm[i];
	perm[i] = tmp;
	perm_inv[perm[i]] = i;
	perm_inv[perm[j]] = j;
}

INT partitionstack::my_log2(INT m)
{
	INT i = 0;

	while (m) {
		i++;
		m >>= 1;
	}
	return i;
}

void partitionstack::split_by_orbit_partition(INT nb_orbits, 
	INT *orbit_first, INT *orbit_len, INT *orbit,
	INT offset, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, f, l, cell_idx, cell_size;
	INT *Set;

	if (f_v) {
		cout << "partitionstack::split_by_orbit_partition" << endl;
		}
	Set = NEW_INT(n);
	
	for (i = 0; i < nb_orbits; i++) {
		f = orbit_first[i];
		l = orbit_len[i];
		if (f_vv) {
			cout << "partitionstack::split_by_orbit_partition orbit " << i << " first=" << f << " length=" << l << endl;
			}
		for (j = 0; j < l; j++) {
			Set[j] = orbit[f + j] + offset;
			}
		if (f_vv) {
			cout << "orbit: ";
			INT_vec_print(cout, Set, l);
			cout << endl;
			}
		if (!is_subset_of_cell(Set, l, cell_idx)) {
			cout << "partitionstack::split_by_orbit_partition the subset is not subset of a cell of the partition, error" << endl;
			exit(1);
			}
		cell_size = cellSize[cell_idx];
		if (l < cell_size) {
			// we need to split the cell:
			if (f_v) {
				cout << "orbit " << i << " of length=" << l << " is split off from cell " << cell_idx 
					<< " to form a new cell C_{" << ht << "}, so " 
					<< cell_size << " = " << cell_size - l << " + " << l << endl;
				}
			split_cell(Set, l, 0 /*verbose_level*/);
			}
		}
	FREE_INT(Set);
	if (f_v) {
		cout << "partitionstack::split_by_orbit_partition done" << endl;
		}
}



