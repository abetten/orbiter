# Orbiter Reference Manual for AI-Assisted Makefile Writing

Orbiter Build 3344, May 27, 2026. C++ computer algebra system for classification of combinatorial objects.

## Executable and Paths

```makefile
ORBITER=orbiter/src/apps/orbiter/orbiter.out
ORBITER_EXE_PATH=$(shell pwd)/orbiter/src/apps/orbiter
PDFLATEX=pdflatex
OPEN=open
```

The work directory MUST be OUTSIDE the orbiter source tree. The `-orbiter_path $(ORBITER_EXE_PATH)` flag is required for any command that generates poset classification reports (it tells Orbiter where to find its helper scripts).

---

## Core Language: The Dash Code

All Orbiter commands are prefixed with `-`. The language is parsed as a flat list of arguments. No semicolons. No line continuation in the language itself — use makefile `\` for multi-line. Every keyword is a "dash word": `-define`, `-with`, `-do`, `-end`, `-graph`, etc.

### The define/with/do Pattern

**Creating an object:**
```
-define <Name> -<object_type> [construction_args] -end
```

**Acting on an object:**
```
-with <Name> -do -<activity_type> [activity_args] -end
```

Objects are stored in a global symbol table by their `<Name>` (a literal string, no spaces). Names can reference other named objects in the session. All `-define` blocks must appear before `-with` blocks that use them, within a single `orbiter.out` invocation.

**Example — full pipeline:**
```makefile
example:
	$(ORBITER) -v 2 \
	  -define F -finite_field -q 2 -end \
	  -define G -linear_group -PGL 4 F -end \
	  -with G -do \
	    -group_theoretic_activity \
	      -report Do \
	  -end
```

---

## Verbosity

`-v <level>` is always the first argument after `$(ORBITER)`:

| Level | Effect |
|-------|--------|
| 0 | Silent (no output) |
| 1 | Minimal |
| 2 | Normal (most makefile targets use this) |
| 3 | Verbose debug |
| 4-8 | Increasingly detailed internal tracing |

---

## Makefile Header Variables (Blue Box Data)

Makefile variables (uppercase) store large data like generator matrices, codeword lists, etc. They are passed on the command line via `$(VAR_NAME)`.

```makefile
GEN_C13="1,2,3,4,5,6,7,8,9,10,11,12,0"
# (0,1,2,3,4,5,6,7,8,9,10,11,12)

GOLAY_24_CODE_AUT_GENS= ...   # 12 matrices stored as a string

HIRSCHFELD_SURFACE_EQUATION_ALGEBRAIC_FORM= ...  # polynomial coefficients
```

---

## Control Flow

### Loops over numeric range
```
-loop i <start> <end> <step>
  ... commands using i ...
-end_loop i
```

### Loops over vector elements
```
-loop_over i <vector_label>
  ... commands using i ...
-end_loop_over i
```

### Capture output into symbol table
```
-assign <Name> -from
-with R -do -ring_theoretic_activity -parse_equation_wo_parameters ... -end
```
The object produced by the activity is captured as `<Name>`.

### Utility commands (top-level, no define/with needed)
```
-print_symbols          # list all objects in symbol table
-list_arguments         # print all command-line arguments
```

---

## Object Types

### 1. `-finite_field`

```
-define F -finite_field -q <q> -end
-define F -finite_field -q <q> -compute_related_fields -end
```

`q` must be a prime power. Elements are integers 0 to q-1. The primitive polynomial is chosen automatically. `-compute_related_fields` also creates subfields.

**Examples:**
```makefile
-define F -finite_field -q 2 -end
-define F -finite_field -q 4 -end
-define F -finite_field -q 27 -compute_related_fields -end
```

**Activities:** (none needed — field is used as argument to other objects)

---

### 2. `-projective_space`

```
-define P -projective_space -n <n> -field <F_label> -v 0 -end
```

Creates PG(n, q). Points are indexed 0 to θ_n(q)-1 where θ_n(q) = (q^(n+1)-1)/(q-1). The rightmost nonzero coordinate of the homogeneous vector equals 1 (normalization convention).

**Examples:**
```makefile
-define F -finite_field -q 2 -end
-define P -projective_space -n 3 -field F -v 0 -end
```

---

### 3. `-linear_group` (Matrix Groups)

```
-define G -linear_group -<group_type> <n> <F_label> [subgroup_option] -end
```

**Group type flags:**
| Flag | Group | Notes |
|------|-------|-------|
| `-PGL <n> F` | PGL(n, q) | acts on points of PG(n-1, q) |
| `-PSL <n> F` | PSL(n, q) | |
| `-PGGL <n> F` | PΓL(n, q) = PGL extended by Frobenius | |
| `-PGGL_3 <n> F` | PΓL(3, q) variant | |
| `-PGO <n> F` | PGO(n, q) | orthogonal group |
| `-PGOp <n> F` | PGO+(n, q) | |
| `-PGOm <n> F` | PGO-(n, q) | |
| `-PGGO <n> F` | PΓO(n, q) | |
| `-AGL <n> F` | AGL(n, q) | affine |
| `-AGGL <n> F` | AΓL(n, q) | affine semilinear |
| `-GL <n> F` | GL(n, q) | |

**Subgroup modification flags** (appended after group type):
| Flag | Effect |
|------|--------|
| `-symplectic_group` | symplectic subgroup |
| `-null_polarity_group` | null polarity group |
| `-diagonal` | diagonal matrices subgroup |
| `-monomial` | monomial matrices subgroup |
| `-Janko1` | first Janko group (needs PGL(7,11)) |
| `-singer <k>` | subgroup of index k in Singer cycle |
| `-singer_and_frobenius <k>` | Singer cycle extended by Frobenius |
| `-borel_upper` | Borel subgroup of upper triangular matrices |
| `-borel_lower` | Borel subgroup of lower triangular matrices |
| `-identity_group` | identity subgroup |
| `-orthogonal <epsilon>` | O^epsilon(n,q), epsilon in {+1,-1} |
| `-subgroup_by_generators <label> <label_tex> <order> <n_gens> <gen1> ...` | subgroup from generators |
| `-subgroup_from_file <f> <l>` | read subgroup from file f, label l |

**Examples:**
```makefile
# PGL(4,2) acts on 15 points of PG(3,2)
-define G -linear_group -PGL 4 F -end

# PSL(5,3) acts on 121 points of PG(4,3)
-define G -linear_group -PSL 5 F -end

# PGL(4,5) acts on 156 points of PG(3,5)
-define G -linear_group -PGL 4 F -end

# PΓL(3,4)
-define G -linear_group -PGGL 3 4 -end

# Symplectic group Sp(4,2)
-define G -linear_group -GL 4 F -symplectic_group -end

# PGO(5,2) acting on Q(4,2)
-define G -linear_group -PGO 5 F -end

# PGO+(6,2) acting on Q+(5,2)
-define G -linear_group -PGOp 6 F -end

# PGO-(6,2) acting on Q-(5,2)
-define G -linear_group -PGOm 6 F -end

# AGL(1,27)
-define G -linear_group -AGL 1 F -end

# Janko group J1
-define G -linear_group -PGL 7 11 -Janko1 -end

# Null polarity group of PG(6,2) acting on PG(5,2)
-define G -linear_group -PGL 6 F -null_polarity_group -end

# M24 as subgroup of PGL(12,2) via generators
-define G -linear_group -PGL 12 2 \
    -subgroup_by_generators "M24" "244823040" 12 \
    $(GOLAY_24_CODE_AUT_GENS) \
  -end
```

**Group theoretic activities:**
```
-with G -do -group_theoretic_activity
  -report Do                # full LaTeX report (stabilizer chain, basic orbits, Schreier trees)
  -sylow                    # compute p-Sylow subgroups (one per prime divisor)
  -export_orbiter           # export in Orbiter makefile format → <label>.makefile
  -export_group_table       # export group table (Cayley table)
  -save_elements_csv "<fname.csv>"  # save all elements as permutations in CSV
-end
```

**Report pattern:**
```makefile
PGL_4_2_export:
	$(ORBITER) -v 2 \
	  -define Do -draw_options -radius 200 -line_width 0.3 -end \
	  -define F -finite_field -q 2 -end \
	  -define G -linear_group -PGL 4 F -end \
	  -with G -do \
	    -group_theoretic_activity \
	      -report Do \
	  -end \
	  -with G -do \
	    -group_theoretic_activity \
	      -export_orbiter \
	  -end
	$(PDFLATEX) PGL_4_2_report.tex
	$(OPEN) PGL_4_2_report.pdf
```

**Recreating group from exported file:**
```makefile
PGL_4_2_generated:
	$(ORBITER) -v 2 \
	  -define gens -vector -file PGL_4_2_gens.csv -end \
	  -define G -permutation_group \
	    -bsgs PGL_4_2 "{\rm PGL}(4,2)" 15 20160 "0,1,2,3" 6 gens \
	  -end
```

---

### 4. `-permutation_group`

```
-define G -permutation_group [construction_args] -end
```

**Construction options:**
```
-bsgs <label> <label_tex> <degree> <order> "<base_pts>" <nb_gens> <gens_label>
-symmetric_group <n>          # symmetric group S_n
-subgroup_by_generators <label> <label_tex> <order> <n_gens> <gen1> ...
```

**Example (cyclic group C13):**
```makefile
GEN_C13="1,2,3,4,5,6,7,8,9,10,11,12,0"

C13:
	$(ORBITER) -v 2 \
	  -define gens -vector -dense $(GEN_C13) -end \
	  -define G -permutation_group \
	    -bsgs C13 C_{13} 13 13 "0" 1 gens \
	  -end \
	  -with G -do -group_theoretic_activity -export_orbiter -end \
	  -with G -do -group_theoretic_activity -report Do -end \
	  -with G -do -group_theoretic_activity -save_elements_csv "C13_elts.csv" -end
	$(PDFLATEX) C13_report.tex
```

**C13 as subgroup of Sym(13):**
```makefile
C13_as_subgroup:
	$(ORBITER) -v 2 \
	  -define Do -draw_options -radius 200 -line_width 0.3 -end \
	  -define G -permutation_group -symmetric_group 13 \
	    -subgroup_by_generators C13 C13 1 $(GEN_C13) \
	  -end \
	  -with G -do -group_theoretic_activity -export_orbiter -end \
	  -with G -do -group_theoretic_activity -report Do -end \
	  -with G -do -group_theoretic_activity -save_elements_csv "C13_elts.csv" -end
```

---

### 5. `-vector`

```
-define V -vector -dense "<comma_separated_values>" -end
-define V -vector -file <filename.csv> -end
-define V -vector -zeros <n> -end
```

Vectors are stored as arrays of integers. They are used as generators for groups, as block lists for designs, as incidence matrices, etc.

---

### 6. `-draw_options`

```
-define Do -draw_options [options] -end
```

Draw options are passed to `-report` and other visualization activities. Common options:

| Option | Value | Effect |
|--------|-------|--------|
| `-radius <r>` | integer | drawing radius in pixels |
| `-line_width <w>` | float | line width |
| `-nodes_empty` | (none) | draw nodes as empty circles |
| `-nodes` | (none) | draw nodes filled |
| `-scale <s>` | float | scale factor |
| `-x_stretch <f>` | float | horizontal stretch |
| `-embedded` | (none) | use embedding coordinates |

**Example:**
```makefile
-define Do -draw_options -radius 300 -nodes_empty -line_width 1.5 -scale 0.1 -end
-define Do -draw_options -radius 200 -line_width 0.3 -end
```

---

### 7. `-graph`

```
-define G -graph -<graph_type> [args] -end
```

**Graph construction commands:**

| Command | Arguments | Graph produced |
|---------|-----------|----------------|
| `-Hamming <n> <q>` | n, q | Hamming graph H(n,q): n*q vertices |
| `-Johnson <n> <k>` | n, k | Johnson graph J(n,k): C(n,k) vertices |
| `-Petersen` | (none) | Petersen graph (10 vertices) |
| `-complete <n>` | n | complete graph K_n |
| `-complete_bipartite <m> <n>` | m, n | K_{m,n} |
| `-cycle <n>` | n | cycle graph C_n |
| `-path <n>` | n | path graph P_n |
| `-grid <m> <n>` | m, n | m x n grid graph |
| `-collinearity_graph <Inc_label>` | vector label | graph from incidence matrix (row = point, col = block; two points adjacent iff they share a block) |
| `-Cayley <G_label> <gens>` | group label, generators | Cayley graph |
| `-double_cover <G_label>` | graph label | **FORK-SPECIFIC** (see below) |
| `-from_file <filename>` | filename | load graph from file |

**`-double_cover` (fork-specific addition, not in standard manual):**
```
-define DC -graph -double_cover <existing_graph_label> -end
```
- `<existing_graph_label>` must be the name of an already-defined graph object in the symbol table
- If the source graph has n vertices, the double cover has **2*n + 1** vertices
- Construction: two copies of the source graph (vertices 0..n-1 and n..2n-1) plus one extra vertex z = 2n adjacent to ALL vertices in the second copy {n, ..., 2n-1}; adjacency within each copy mirrors the source graph
- Output label stored as: `"double_cover_" + graph_label`
- Output label_tex: `"{\rm double\_cover\_{" + graph_label + "}}"`

**Example:**
```makefile
double_cover_example:
	$(ORBITER) -v 2 \
	  -define G -graph -Hamming 3 2 -end \
	  -define DC -graph -double_cover G -end \
	  -with DC -do -graph_theoretic_activity -properties -end
```

**Graph theoretic activities:**
```
-with G -do -graph_theoretic_activity
  -properties                          # basic graph properties (order, degree, etc.)
  -create_distance_poset <i>           # create distance poset from vertex i
  -adjacency_matrix                    # print/export adjacency matrix
  -export_csv "<filename>"             # export graph to CSV
  -report Do                           # LaTeX report with drawing
  -draw Do                             # draw graph using draw_options Do
-end
```

**Collinearity graph from incidence matrix:**
```makefile
PGO_5_2_collinearity_graph:
	$(ORBITER) -v 2 \
	  -define F -finite_field -q 2 -end \
	  -define O -orthogonal_space 0 5 F -without_group -end \
	  -with O -do -orthogonal_space_activity \
	    -export_point_line_incidence_matrix \
	  -end
	$(ORBITER) -v 3 \
	  -define Inc -vector -file O_5_2_incidence_matrix.csv -end \
	  -define Gamma -graph -collinearity_graph Inc -end \
	  -with Gamma -do -graph_theoretic_activity -properties -end
```

**Distance poset drawing:**
```makefile
	-with Gamma -do -graph_theoretic_activity -create_distance_poset 0 -end
	$(ORBITER) -v 3 -draw_layered_graph \
	  Johnson_5_2_0_distance_poset.layered_graph \
	  -place -radius 250 -embedded -nodes \
	  -line_width 1.1 -x_stretch 1.4 -scale 0.25 \
	-end
	$(PDFLATEX) Johnson_5_2_0_distance_poset_draw.tex
```

---

### 8. `-graph_classification`

Classifies all graphs on n vertices up to isomorphism using poset classification.

```
-define GC -graph_classification
  -n <n>
  -poset_classification_control
    -problem_label <label>
    -depth <d>
    -draw_options <Do_label>
  -end
-end
```

**Activities:**
```
-with GC -do -graph_classification_activity
  -list_graphs_at_level <min> <max>    # list all graphs at given level(s)
  -draw_graphs_at_level <level>        # draw graphs at level
-end
```

**Example:**
```makefile
graph_classify_5:
	$(ORBITER) -v 2 \
	  -orbiter_path $(ORBITER_EXE_PATH) \
	  -define Do -draw_options -radius 300 -nodes_empty -line_width 1.5 -scale 0.1 -end \
	  -define GC -graph_classification \
	    -n 5 \
	    -poset_classification_control \
	      -problem_label graphs_v5 -depth 10 -draw_options Do \
	    -end \
	  -end \
	  -with GC -do -graph_classification_activity \
	    -list_graphs_at_level 5 5 \
	  -end \
	  -with GC -do -graph_classification_activity -draw_options Do \
	    -draw_graphs_at_level 5 \
	  -end
```

---

### 9. `-design`

```
-define D -design [construction] -end
```

**Construction options:**

| Construction | Arguments | Description |
|-------------|-----------|-------------|
| `-field F -family PG_2_q` | field label | Projective plane PG(2,q) as a 2-design |
| `-field F -family AG_2_q` | field label | Affine plane AG(2,q) |
| `-list_of_blocks_coded <v> <k> "<block_list>"` | v, k, list | t-(v,k,lambda) from coded block list |
| `-from_incidence_matrix <filename>` | filename | load from incidence matrix CSV |

**Design activities:**
```
-with D -do -design_activity
  -export_flags               # export flag CSV
  -tactical_decomposition     # compute tactical decomposition (TDO/TDA)
  -report                     # LaTeX report
  -print_blocks               # print all blocks
  -automorphism_group         # compute automorphism group
-end
```

**Examples:**
```makefile
design_PG_2_3:
	$(ORBITER) -v 8 \
	  -define F -finite_field -q 3 -end \
	  -define D -design -field F -family PG_2_q -end \
	  -with D -do -design_activity -export_flags -end \
	  -with D -do -design_activity -tactical_decomposition -end

design_from_blocks:
	$(ORBITER) -v 3 \
	  -define D -design -list_of_blocks_coded 15 3 \
	    "61, 67, 72, 76, 129, 147, 152, 156, 197, 204, 215, 224, 249, 261, 267, 276, 296, 303, 309, 319" \
	  -end \
	  -with D -do -design_activity -report -end
```

---

### 10. `-combinatorial_object`

Used to classify sets of points, geometric objects, or other combinatorial structures up to the action of a group (typically a projective group).

```
-define C -combinatorial_object
  -label <label> <label_tex>
  -set_of_points "<pt1,pt2,...>"
  -set_of_points "<pt1,pt2,...>"
  ...
-end
```

Or from a geometric object:
```
-define C -combinatorial_object
  -label <label> <label_tex>
  -geometric_object <Geo_label>
-end
```

**Combinatorial object activities:**
```
-with C -do -combinatorial_object_activity
  -canonical_form_PG <P_label>         # classify under projective group of P
    [-save_ago]                          # save automorphism group orbits
    [-max_TDO_depth <d>]
    [-nauty_control
      -nauty_log "<filename>"
      [-save_nauty_input_graphs <prefix>]
    -end]
  -post_processing                      # run post-processing on classification
  -report                               # LaTeX report
    [-export_flag_orbits]
    [-show_TDO]
    [-show_TDA]
    [-dont_show_incidence_matrices]
    [-incidence_draw_options <Do_label>]
-end
```

**Example — classify quadrangles in PG(2,2):**
```makefile
quadrangles_PG_2_2:
	$(ORBITER) -v 3 \
	  -define C -combinatorial_object \
	    -label quadrangle quadrangle \
	    -set_of_points "0,1,2,3" \
	    -set_of_points "1,3,4,6" \
	  -end \
	  -define F -finite_field -q 2 -end \
	  -define P -projective_space -n 2 -field F -v 0 -end \
	  -with C -do -combinatorial_object_activity \
	    -canonical_form_PG P \
	    -nauty_control -nauty_log "quadrangles_PG_2_2_nauty_log.txt" -end \
	  -end \
	  -with C -do -combinatorial_object_activity -post_processing -end \
	  -with C -do -combinatorial_object_activity \
	    -report \
	    -export_flag_orbits -show_TDO -show_TDA \
	    -dont_show_incidence_matrices \
	    -incidence_draw_options Doinc \
	  -end
```

---

### 11. `-polynomial_ring`

```
-define R -polynomial_ring -field <F_label>
  -number_of_variables <n>
  -homogeneous_of_degree <d>
-end
```

**Ring theoretic activities:**
```
-with R -do -ring_theoretic_activity
  -parse_equation_wo_parameters "<label>" "<label_tex>" <coefficients>
-end
```

The result can be captured with `-assign`:
```makefile
-assign Eqn -from
-with R -do -ring_theoretic_activity
  -parse_equation_wo_parameters "label" "label\_tex" $(EQUATION_VAR)
-end
```

---

### 12. `-geometric_object`

```
-define Geo -geometric_object <P_label>
  -projective_variety <R_label>
    "<label>" "<label_tex>"
    <Eqn_label>
-end
```

Used to define a projective variety (e.g., a cubic or quartic surface) that can then be turned into a combinatorial object for classification.

**Example — classify Hirschfeld surface over F_4:**
```makefile
Hirschfeld_q4_c:
	$(ORBITER) -v 2 \
	  -define F -finite_field -q 4 -end \
	  -define R -polynomial_ring -field F \
	    -number_of_variables 4 -homogeneous_of_degree 3 \
	  -end \
	  -define P -projective_space -n 3 -field F -v 0 -end \
	  -assign Eqn -from \
	  -with R -do -ring_theoretic_activity \
	    -parse_equation_wo_parameters \
	      "Hirschfeld_surface_q4" "Hirschfeld\_surface\_q4" \
	      $(HIRSCHFELD_SURFACE_EQUATION_ALGEBRAIC_FORM) \
	  -end \
	  -define Geo -geometric_object P \
	    -projective_variety R \
	      "Hirschfeld_surface_q4" "Hirschfeld\_surface\_q4" \
	      Eqn \
	  -end \
	  -define C -combinatorial_object \
	    -label Hirschfeld_surface_q4 Hirschfeld\_surface\_q4 \
	    -geometric_object Geo \
	  -end \
	  -with C -do -combinatorial_object_activity \
	    -canonical_form_PG P \
	    -save_ago -max_TDO_depth 10 \
	    -nauty_control -nauty_log "Hirschfeld_q4_nauty_log.txt" \
	      -save_nauty_input_graphs Hirschfeld_q4 \
	    -end \
	  -end
```

---

### 13. `-spread`

A spread of PG(2k-1, q) is a partition of its points into lines (k=1) or k-subspaces.

```
-define S -spread -kernel_field <F_label>
  -group <G_label>
  -k <k>
  -catalogue <idx>
-end
```

**Spread activities:**
```
-with S -do -spread_activity
  -report              # LaTeX report on the spread
  -export_incma        # export incidence matrix
-end
```

**Example:**
```makefile
create_spread_9a:
	$(ORBITER) -v 3 \
	  -define F -finite_field -q 3 -end \
	  -define G -linear_group -PGL 4 F -end \
	  -define S -spread -kernel_field F \
	    -group G -k 2 -catalogue 0 \
	  -end \
	  -with S -do -spread_activity -report -end
```

---

### 14. `-translation_plane`

```
-define T -translation_plane <S_label> <G_label> <G1_label> -end
```

`S` = spread, `G` = group in GL(2k, q), `G1` = group in GL(2k+1, q).

**Translation plane activities:**
```
-with T -do -translation_plane_activity
  -export_incma        # export incidence matrix
  -report              # LaTeX report
-end
```

**Example:**
```makefile
create_translation_plane_9b:
	$(ORBITER) -v 3 \
	  -define F -finite_field -q 3 -end \
	  -define G -linear_group -PGL 4 F -end \
	  -define G1 -linear_group -PGL 5 F -end \
	  -define S -spread -kernel_field F -group G -k 2 -catalogue 1 -end \
	  -define T -translation_plane S G G1 -end \
	  -with T -do -translation_plane_activity -export_incma -end \
	  -with T -do -translation_plane_activity -report -end
```

---

### 15. `-spread_table`

Builds a table of all spreads for a given projective space.

```
-define T -spread_table
  -space <P_label>
  -rk <k>
  -iso_types "<list>"
  -path "<directory_path>"
  -control <Control_label>
-end
```

**Example:**
```makefile
PG_3_2_spread_table:
	mkdir SPREAD_TABLES_2
	$(ORBITER) -v 5 \
	  -define Control -poset_classification_control -problem_label spreads_PG_3_2 -W -end \
	  -define F -finite_field -q 2 -end \
	  -define P3 -projective_space -n 3 -field F -v 0 -end \
	  -define T -spread_table -space P3 -rk 2 -iso_types "0" -path "SPREAD_TABLES_2/" -control Control -end
```

---

### 16. `-spread_classifier`

Classifies all spreads of a projective space.

```
-define C -spread_classifier
  -projective_space <P_label>
  -poset_classification_control <Control_label>
  -k <k>
  -starter_size <s>
  -output_prefix "<dir>"
-end
```

**Spread classify activities:**
```
-with C -do -spread_classify_activity
  -compute_starter
-end
```

**Example:**
```makefile
classify_spreads_16_4:
	$(ORBITER) -v 4 \
	  -define Control -poset_classification_control -problem_label spreads_16_4 -draw_options Do -end \
	  -define F -finite_field -q 4 -end \
	  -define P -projective_space -n 3 -field F -v 0 -end \
	  -define C -spread_classifier -projective_space P \
	    -poset_classification_control Control -k 2 -starter_size 17 -output_prefix "." \
	  -end \
	  -with C -do -spread_classify_activity -compute_starter -end
```

---

### 17. `-orthogonal_space`

```
-define O -orthogonal_space <epsilon> <n> <F_label> [-without_group] -end
```

`epsilon` in {-1, 0, 1}: type of quadric. `-without_group` skips computing automorphism group (faster).

**Orthogonal space activities:**
```
-with O -do -orthogonal_space_activity
  -export_point_line_incidence_matrix   # write incidence matrix to CSV
  -report                               # LaTeX report
-end
```

---

### 18. `-poset_classification_control`

Configuration object for poset-based classification algorithms (Schmalz algorithm).

```
-define Control -poset_classification_control
  -problem_label <label>
  [-depth <d>]
  [-draw_options <Do_label>]
  [-W]          # write results
-end
```

Passed as argument to `-graph_classification`, `-spread_table`, `-spread_classifier`, etc.

---

## File Naming Conventions

Output files follow predictable patterns based on object labels:

| File | Contents |
|------|----------|
| `<label>_report.tex` | LaTeX report for object labeled `<label>` |
| `<label>_report.pdf` | PDF compiled from above |
| `<label>_gens.csv` | Generator CSV for group named `<label>` |
| `<label>.makefile` | Exported Orbiter makefile |
| `<label>_elts.csv` | All group elements in permutation form |
| `<label>_incidence_matrix.csv` | Incidence matrix export |
| `<problem_label>_*.layered_graph` | Poset/layered graph files |
| `<problem_label>_draw.tex` | Drawing LaTeX for layered graph |

---

## CSV Tools (Top-Level Commands)

These are standalone commands that do not use define/with/do:

```makefile
# Filter rows where column "ColName" equals "value"
$(ORBITER) -v 3 -csv_file_filter <input.csv> "<ColName>" "<value>"

# Tally values in a column
$(ORBITER) -v 3 -tally_column <input.csv> <ColName>

# Concatenate multiple CSV files matching a printf mask
$(ORBITER) -v 3 -csv_file_concatenate_from_mask 0 <nb_files> <mask_fmt> <output.csv>
# Example: -csv_file_concatenate_from_mask 0 $(NB_FILES) surface_catalogue_q13_iso%ld_quartics.csv quartics_q13.csv

# Select specific rows and columns from CSV
$(ORBITER) -v 4 -csv_file_select_rows_and_cols <fname> "<row_indices>" "<col_indices>"
# Example: -csv_file_select_rows_and_cols data.csv "0,2,4" "0,2,4"
```

**CSV format (group elements example):**
```
Row,C0,C1,C2,...,C14
0,0,1,2,9,14,5,6,7,8,3,11,10,13,12,4
1,1,0,2,10,13,5,6,7,8,11,3,9,14,4,12
...
```

---

## Layered Graph Drawing (Top-Level Command)

```makefile
$(ORBITER) -v 3 -draw_layered_graph \
  <filename.layered_graph> \
  -place -radius <r> [-embedded] [-nodes | -nodes_empty] \
  -line_width <w> [-x_stretch <f>] -scale <s> \
-end
$(PDFLATEX) <filename>_draw.tex
```

---

## Gnuplot Interface (Top-Level Command)

```makefile
# Plot data from a CSV file
$(ORBITER) -v 3 -gnuplot <data.csv> "<Title>" "<X label>" "<Y label>"
$(OPEN) <data>_gnuplot.png

# Euler function (auto-generates CSV then plot)
$(ORBITER) -v 1 -eulerfunction_interval 1 50
$(ORBITER) -v 3 -gnuplot table_eulerfunction_1_50.csv "Euler function" "Input" "Output"
```

---

## Poset Classification Reports

Any command that runs poset classification (graph_classification, spread_classifier, etc.) requires:
```
-orbiter_path $(ORBITER_EXE_PATH)
```
as a top-level argument (not inside define/with). This is how Orbiter locates its auxiliary scripts for generating LaTeX poset diagrams.

---

## The `-without_group` Flag

When creating geometric objects like projective spaces or orthogonal spaces, adding `-without_group` skips computing the automorphism group. This is useful when only incidence matrix export is needed and the group computation would be expensive:
```makefile
-define O -orthogonal_space 0 5 F -without_group -end
```

---

## Common Makefile Patterns

### Pattern 1: Create group, generate report
```makefile
GROUP_target:
	$(ORBITER) -v 2 \
	  -define Do -draw_options -radius 200 -line_width 0.3 -end \
	  -define F -finite_field -q <q> -end \
	  -define G -linear_group -<TYPE> <n> F -end \
	  -with G -do -group_theoretic_activity -report Do -end
	$(PDFLATEX) <label>_report.tex
	$(OPEN) <label>_report.pdf
```

### Pattern 2: Classify objects under group action
```makefile
classify_target:
	$(ORBITER) -v 3 \
	  -define F -finite_field -q <q> -end \
	  -define P -projective_space -n <n> -field F -v 0 -end \
	  -define C -combinatorial_object \
	    -label <lbl> <lbl_tex> \
	    -set_of_points "<pts>" \
	    ... \
	  -end \
	  -with C -do -combinatorial_object_activity \
	    -canonical_form_PG P \
	    -nauty_control -nauty_log "<lbl>_nauty_log.txt" -end \
	  -end \
	  -with C -do -combinatorial_object_activity -post_processing -end \
	  -with C -do -combinatorial_object_activity -report -end
```

### Pattern 3: Graph construction and analysis
```makefile
graph_target:
	$(ORBITER) -v 2 \
	  -define G -graph -<TYPE> <args> -end \
	  -with G -do -graph_theoretic_activity -properties -end \
	  -with G -do -graph_theoretic_activity -create_distance_poset 0 -end
	$(ORBITER) -v 3 -draw_layered_graph \
	  <label>_0_distance_poset.layered_graph \
	  -place -radius 250 -embedded -nodes \
	  -line_width 1.1 -x_stretch 1.4 -scale 0.25 \
	-end
	$(PDFLATEX) <label>_0_distance_poset_draw.tex
```

### Pattern 4: Double cover of existing graph
```makefile
double_cover_target:
	$(ORBITER) -v 2 \
	  -define G -graph -<TYPE> <args> -end \
	  -define DC -graph -double_cover G -end \
	  -with DC -do -graph_theoretic_activity -properties -end
```

### Pattern 5: Multi-step pipeline (two invocations)
```makefile
pipeline_target:
	# Step 1: export data
	$(ORBITER) -v 2 \
	  -define F -finite_field -q 2 -end \
	  -define O -orthogonal_space 0 5 F -without_group -end \
	  -with O -do -orthogonal_space_activity -export_point_line_incidence_matrix -end
	# Step 2: use exported data
	$(ORBITER) -v 3 \
	  -define Inc -vector -file O_5_2_incidence_matrix.csv -end \
	  -define Gamma -graph -collinearity_graph Inc -end \
	  -with Gamma -do -graph_theoretic_activity -properties -end
```

---

## Common Pitfalls

1. **`-end` is mandatory** to close every `-define`, `-with ... -do`, and nested blocks. Missing `-end` causes a parse error or silently wrong behavior.

2. **Symbol table is session-scoped.** Each invocation of `orbiter.out` starts with an empty symbol table. You cannot reference objects from a previous invocation; export to CSV/file and reload instead.

3. **`-orbiter_path` must appear at top level**, not inside a define/with block. It is required for any classification report.

4. **The `-v` level must be the very first argument** after `$(ORBITER)`.

5. **No spaces in label names.** Labels must be single tokens. Use underscores: `my_graph`, not `my graph`.

6. **Makefile variable expansion.** When a makefile variable expands to multiple tokens (e.g., `$(GEN_C13)`), it is passed as-is to the command line. Quote if passing as a single CSV string: `"$(GEN_C13)"`.

7. **`-double_cover` is fork-specific** — it is NOT in the standard Orbiter distribution and will not be documented in the users_guide.pdf. The source is in `orbiter/src/lib/layer5_top_level/apps_graph_theory/create_graph_description.cpp` (argument parsing) and `create_graph.cpp` (implementation, `create_double_cover` at line 2123).

8. **Point indexing starts at 0.** PG(n,q) has points indexed 0 to θ_n(q)-1. When specifying sets of points, use 0-based integer indices.

9. **`-nauty_control ... -end` closes with its own `-end`** inside the parent `-canonical_form_PG` activity. The nesting is: activity args → nauty_control block → `-end` closes nauty_control → then the outer activity block `-end` closes it.

10. **Report generation** always produces two steps: first run orbiter to generate `.tex`, then run pdflatex:
    ```makefile
    $(PDFLATEX) <label>_report.tex
    $(OPEN) <label>_report.pdf
    ```

11. **`-compute_related_fields`** on a finite field computes all subfields. Use when AΓL or PΓL groups are needed (they require subfield data).

---

## Quick Reference: Object Types and Their Activities

| Object type | Create flag | Activity flag |
|------------|-------------|---------------|
| Finite field | `-finite_field` | (none; used as argument) |
| Projective space | `-projective_space` | (none; used as argument) |
| Linear group (matrix) | `-linear_group` | `-group_theoretic_activity` |
| Permutation group | `-permutation_group` | `-group_theoretic_activity` |
| Graph | `-graph` | `-graph_theoretic_activity` |
| Graph classification | `-graph_classification` | `-graph_classification_activity` |
| Design | `-design` | `-design_activity` |
| Combinatorial object | `-combinatorial_object` | `-combinatorial_object_activity` |
| Polynomial ring | `-polynomial_ring` | `-ring_theoretic_activity` |
| Geometric object | `-geometric_object` | (used as argument to combinatorial_object) |
| Spread | `-spread` | `-spread_activity` |
| Translation plane | `-translation_plane` | `-translation_plane_activity` |
| Spread table | `-spread_table` | (none; computation done during define) |
| Spread classifier | `-spread_classifier` | `-spread_classify_activity` |
| Orthogonal space | `-orthogonal_space` | `-orthogonal_space_activity` |
| Vector | `-vector` | (none; used as argument) |
| Draw options | `-draw_options` | (none; used as argument) |
| Poset classification control | `-poset_classification_control` | (none; used as argument) |

---

## Group Theoretic Activity Summary

All of these are sub-commands of `-group_theoretic_activity`:

| Sub-command | Arguments | Effect |
|-------------|-----------|--------|
| `-report <Do_label>` | draw options label | Full LaTeX report (stabilizer chain, Schreier trees, basic orbits, GAP/Magma export) |
| `-sylow` | (none) | Compute one p-Sylow subgroup per prime p dividing |G| |
| `-export_orbiter` | (none) | Export group in Orbiter makefile format (`<label>.makefile`, `<label>_gens.csv`) |
| `-export_group_table` | (none) | Export Cayley multiplication table |
| `-save_elements_csv "<fname>"` | filename | Save all elements as permutations in CSV |

---

## Coding Theory (`-linear_code`)

```
-define C -linear_code -field F -generator_matrix <rows> <cols> <data> -end
```

**Linear code activities:**
```
-with C -do -linear_code_activity
  -codewords                  # enumerate all codewords
  -automorphism_group         # compute automorphism group
  -report                     # LaTeX report
  -weight_distribution        # weight distribution
  -minimum_distance           # compute minimum distance
-end
```

---

## Interfaces

### GAP export
Group reports automatically include GAP format:
```
G := Group([...generators as cycle notation...]);
```

### Magma export
Group reports automatically include Magma format:
```
G := GeneralLinearGroup(n, GF(q));
H := sub< G | [matrix entries...] >;
```

### Nauty integration
Used via `-canonical_form_PG` with `-nauty_control`:
- Input graphs saved if `-save_nauty_input_graphs <prefix>` specified
- Log written to file specified by `-nauty_log "<filename>"`

### Gnuplot integration
```makefile
$(ORBITER) -v 3 -gnuplot <data.csv> "<Title>" "<X label>" "<Y label>"
```
Output: `<data>_gnuplot.png`

---

## Mathematical Constants

- θ_n(q) = (q^(n+1) - 1)/(q-1) = number of points of PG(n, q)
- AG(n, q) has q^n points
- Hamming graph H(n, q): n*q vertices, each codeword a length-n string over alphabet of size q
- Johnson graph J(n, k): C(n, k) vertices = k-subsets of {0,...,n-1}
