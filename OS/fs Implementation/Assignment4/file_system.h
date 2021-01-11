#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;

#define G_WRITE 1
#define G_READ 0
#define LS_D 0 // Modified time
#define LS_S 1 // Size

#define RM 2

#define set_bit(x,y)  (x|=(1<<y))
#define clr_bit(x,y)  (x&=~(1<<y))


struct FileSystem {
	uchar *volume;
	int SUPERBLOCK_SIZE;
	int FCB_SIZE;
	int FCB_ENTRIES;
	int STORAGE_SIZE;
	int STORAGE_BLOCK_SIZE;
	int MAX_FILENAME_SIZE;
	int MAX_FILE_NUM;
	int MAX_FILE_SIZE;
	int FILE_BASE_ADDRESS;

	uchar *temp;
	int SUPERBLOCK_SENTINEL;
	int FCB_SENTINEL;
};

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
	int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
	int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
	int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS, uchar *temp);

__device__ void sblock_clr_blocks(FileSystem * fs, u32 clr_block_num);
__device__ void sblock_set_blocks(FileSystem *fs, u32 set_block_num);

__device__ u32 fcbs_add(FileSystem *fs, uchar * file_name, u32 file_addr, u32 file_size, u32 file_modi_time);
__device__ u32 fcbs_search(FileSystem *fs, char * file_name);
__device__ u32 fcb_get_name(FileSystem * fs, u32 fcb_index_num);
__device__ u32 fcb_get_addr(FileSystem * fs, u32 fcb_index_num);
__device__ void fcb_set_addr(FileSystem * fs, u32 fcb_index_num, u32 file_addr);
__device__ u32 fcb_get_size(FileSystem * fs, u32 fcb_index_num);
__device__ void fcb_set_size(FileSystem *fs, u32 fcb_index_num, u32 file_size);
__device__ u32 fcb_get_modi_time(FileSystem * fs, u32 fcb_index_num);
__device__ void fcb_set_modi_time(FileSystem *fs, u32 fcb_index_num, u32 file_modi_time);

__device__ u32 fs_open(FileSystem *fs, char *s, int op);
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp);
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp);
__device__ void fs_clr(FileSystem *fs, u32 fp);
__device__ void fs_gsys(FileSystem *fs, int op);
__device__ void fs_gsys(FileSystem *fs, int op, char *s);

__device__ void choice_sort(FileSystem * fs, int op);
__device__ bool fcbs_compare(FileSystem * fs, u32 fcb_index_lhs, u32 fcb_index_rhs, int op);
__device__ void fcbs_swap(FileSystem *fs, u32 fcb_index_a, u32 fcb_index_b);

__device__ void fs_display(FileSystem *fs, u32 steps);



/*
	1. Super Block:
	Super Block takes memory of 4KB(4096 bytes, 32768 bits).
	Contents of files take memory of 1024KB. Each block size takes 32B.
	1024KB/32B = 2^(15) = 32768.

	Thus, we are available to use Super Block bits as a bit-map to represent whether
	one block is available.

	TODO:
	We may require some further method to manipulate the super block, including:
	a. get
	b. set
	c. clear
	1 of 32768 bits.

	2. FCB (File control block)
	FCB size is 32bytes, with 1024 entry. Each entry is given 32 bytes.
	Our mission is to use 32 bytes to store one files information, including:
	a. file name(20 bytes)
	b. file address (10 bits or 2 bytes)
	c. file modification time (dynamic, if store as second, then may be more than 4 bytes)

	3. fp (file pointer)
	fp will point to the address of the file on the global storage. (A block number)
	fp will also define whether current operation can be write/read. (One bit)


*/

#endif
