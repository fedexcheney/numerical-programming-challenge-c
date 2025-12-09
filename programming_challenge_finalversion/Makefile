# Makefile for Matrix Multiplication Performance Challenge
# Optimized for Apple Silicon (ARM64)

CC = clang
CFLAGS = -Wall -Wextra
MAIN = main.c
MATMULT_VARIANTS = matmult_ijp.c matmult_jpi.c matmult_pji.c matmult_blocked.c matmult_neon.c matmult_omp.c matmult_1.c matmult_2.c
TARGETS = $(MATMULT_VARIANTS:.c=)

LIBOMP = $(shell brew --prefix libomp)

# Debug build for all variants
debug: CFLAGS += -g -O0 -fsanitize=address
debug: LDFLAGS += -fsanitize=address
debug: $(addsuffix -debug, $(TARGETS))

# Optimized build for all variants (We are using -mcpu=apple-m1 Use the right "m" (m1, m2, m3, etc.) version of the CPU)
opt: CFLAGS += -O3 -mcpu=apple-m1 -ffast-math -funroll-loops
opt: CPPFLAGS += -I$(LIBOMP)/include
opt: LDFLAGS += -L$(LIBOMP)/lib -Wl,-rpath,$(LIBOMP)/lib -framework Accelerate
opt: $(addsuffix -opt, $(TARGETS))

# Individual variant targets
%-debug: $(MAIN) %.c
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

%-opt: $(MAIN) %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $(MAIN) -o main.o
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $*.c -o $*.o
	$(CC) main.o $*.o $(LDFLAGS) -lomp -o $@

# Performance test for all variants
perf: opt
	@first_variant=$$(echo $(TARGETS) | cut -d' ' -f1); \
	./$$first_variant-opt 100 100 100 perf_$$first_variant.csv; \
	head -n 1 perf_$$first_variant.csv > perf.csv; \
	for variant in $(TARGETS); do \
		for size in 100 500 1000 1500 2000; do \
			echo "Running $$variant with size $$size..."; \
			./$$variant-opt $$size $$size $$size perf_$$variant.csv; \
			tail -n 1 perf_$$variant.csv >> perf.csv; \
		done; \
	done
	@rm -f perf_*.csv

# Test individual variant
test-%: %-opt
	@echo "Testing $*..."
	@./$*-opt 1000 1000 1000 test_$*.csv
	@cat test_$*.csv
	@rm -f test_$*.csv

# Clean
clean:
	rm -f *-debug *-opt perf.csv test_*.csv

.PHONY: debug opt perf clean test-%
