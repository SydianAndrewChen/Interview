#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS, uchar *temp)
{
  // init variables
	fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;
  fs->temp = temp;
  fs->FCB_SENTINEL = -1;
  fs->SUPERBLOCK_SENTINEL = -1;

}

__device__ void sblock_clr_blocks(FileSystem * fs, u32 clr_block_num) {
	u32 sentinel_index = fs->SUPERBLOCK_SENTINEL / 8; // Denote the last non-empty super blocks
	u32 sentinel_offset = fs->SUPERBLOCK_SENTINEL % 8; // Denote the last super blick
	if (clr_block_num <= sentinel_offset) {
		fs->volume[sentinel_index] <<= clr_block_num;
	}
	else {
		clr_block_num -= sentinel_offset;
		while (clr_block_num > 7) {
			fs->volume[--sentinel_index] = 0;
			clr_block_num -= 8;
		}
		fs->volume[--sentinel_index] <<= clr_block_num;
	}
	fs->SUPERBLOCK_SENTINEL -= clr_block_num;
	//fs_display(fs, -2);
}

__device__ void sblock_set_blocks(FileSystem *fs, u32 set_block_num) {
	int sentinel_index = fs->SUPERBLOCK_SENTINEL / 8; // Denote the last non-empty super blocks
	int sentinel_offset = fs->SUPERBLOCK_SENTINEL % 8; // Denote the last super block
	if (fs->SUPERBLOCK_SENTINEL < 0) {
		sentinel_index -= 1;
		sentinel_offset += 8;
	}
	if ((7 - sentinel_offset) >= set_block_num) {
		for (int i = 0; i < set_block_num; i++)
		{
			set_bit(fs->volume[sentinel_index], 7-sentinel_offset-(i+1));
		}
	}
	else {
		set_block_num -= (7 - sentinel_offset);
		while (set_block_num > 7) {
			fs->volume[++sentinel_index] = 0xff;
			set_block_num -= 8;
		}
		if (set_block_num > 0) 
			fs->volume[++sentinel_index] = ( ~(0xff >> (set_block_num)));
	}
	fs->SUPERBLOCK_SENTINEL += set_block_num;
}

__device__ u32 fcbs_add(FileSystem *fs, uchar * file_name, u32 file_addr, u32 file_size, u32 file_modi_time) {
	if (file_size > fs->MAX_FILE_SIZE) printf("FILE_SIZE ERR\n");
	if (fs->FCB_SENTINEL == fs->MAX_FILE_NUM) printf("FILE_NUM ERR FCB Already full!\n");
	u32 fcb_addr, fcb_index;
	for (int i = 0; i < fs->FCB_ENTRIES; i++)
	{
		fcb_addr = (i*fs->FCB_SIZE) + fs->SUPERBLOCK_SIZE;
		if ((int)fs->volume[fcb_addr] == 0) {
			fcb_set_size(fs, i, file_size);
			fcb_set_modi_time(fs, i, file_modi_time);
			fcb_set_addr(fs, i, file_addr);
			fcb_index = i;
			break;
		}
	}
	if ((fcb_addr - fs->SUPERBLOCK_SIZE) / fs->FCB_SIZE >= fs->FCB_ENTRIES) {
		printf("FILE_CONTROL_BLOCK ERR!\n");
		return;
	}
	/* Storing file_name char by char */
	for (int i = 0; file_name[i] != '\0'; i++) {
		fs->volume[fcb_addr + i] = file_name[i]; // Stored from 0 to 20
		if (i > fs->MAX_FILENAME_SIZE) {
			printf("FILE_NAME ERR!\n");
			for (int j = 0; j < fs->FCB_SIZE; j++)
				fs->volume[fcb_addr + j] = 0;
			return;
		}
	}
	fs->FCB_SENTINEL++;

	return fcb_index;
}

__device__ u32 fcbs_search(FileSystem *fs, char * file_name) {
	for (int i = 0; i < fs->FCB_ENTRIES; i++){
		if (fs->volume[i*fs->FCB_SIZE + fs->SUPERBLOCK_SIZE] != 0) { // Non-empty
			int j = 0;
			bool flag = true;
			while (file_name[j] != '\0') {
				if (fs->volume[i*fs->FCB_SIZE + fs->SUPERBLOCK_SIZE + j] != file_name[j++]){

					flag = false;
					break;
				}
			}
			if (fs->volume[i*fs->FCB_SIZE + fs->SUPERBLOCK_SIZE + j] != 0) continue;
			if (flag) return i;
		}
	}
	return -1; // Not found;
}


__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	int i = 0;
	u32 fcb_index = fcbs_search(fs, s);
	if (fcb_index == -1) {
		fcb_index = fcbs_add(fs, (uchar*)s, -1, 0, gtime++); // 10 bits, initially no space is allocated to storage
	}
	if (op == G_WRITE) {
		fcb_index |= 0x40000000; // Add a bit representing for read/write only
	}
	return fcb_index;
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	if (fp == -1) {
		printf("ERROR!\n");
	}
	u32 faddr = fcb_get_addr(fs, (fp & 0x0fffffff));
	if (faddr == -1) {
		printf("The file is empty!\n");
		return;
	}
	for (int i = 0; i < size; i++) {
		output[i] = fs->volume[faddr + i];
	}
}

__device__ void fs_clr(FileSystem *fs, u32 fp) {
	u32 size = fcb_get_size(fs, fp);
	u32 faddr = fcb_get_addr(fs, fp);

	int block_num = (bool)((size - (size / (fs->STORAGE_BLOCK_SIZE))*(fs->STORAGE_BLOCK_SIZE)) > 0) +
		size / (fs->STORAGE_BLOCK_SIZE);
	if (faddr == u32(-1)) { // File already empty
		return;
	}
	for (int i = 0; i < size; i++) { // Clear the file content
		fs->volume[faddr + i] = 0;
	}
	//fs_display(fs, -1);
	sblock_clr_blocks(fs, block_num);
	for (int i = fp + 1; i <= fs->FCB_SENTINEL; i++) {
		u32 inital_addr = fcb_get_addr(fs, i);

		for (int j = 0; j < fcb_get_size(fs, i); j++) {
			fs->volume[faddr++] = fs->volume[inital_addr + j];
		}
		fcb_set_addr(fs, i, inital_addr - block_num * fs->STORAGE_BLOCK_SIZE);
	}
	fcb_set_addr(fs, fp, -1);
	fcb_set_size(fs, fp, 0);
	//fs_display(fs, 0);
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	if (((fp & 0xf0000000) >> 30) != G_WRITE) {
		printf("ACCESS ERR: No writings allowed!\n");
		return;
	}
	fp &= 0x0fffffff;
	if (fp == -1) {
		printf("ERROR!\n");
	}
	fs_clr(fs, fp);
	int block_num = (bool)((size - (size / (fs->STORAGE_BLOCK_SIZE))*(fs->STORAGE_BLOCK_SIZE)) > 0) + 
						size / (fs->STORAGE_BLOCK_SIZE);
	u32 faddr = (fs->SUPERBLOCK_SENTINEL+1) * fs->STORAGE_BLOCK_SIZE + fs->FILE_BASE_ADDRESS;
	for (int i = 0; i < size; i++) {
		fs->volume[faddr + i] = input[i];
	}
	sblock_set_blocks(fs, block_num);
	fcb_set_size(fs, fp, size);
	fcb_set_modi_time(fs, fp, gtime++);
	fcb_set_addr(fs, fp, faddr);
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
	if (fs->FCB_SENTINEL == -1) {
		return;
	}
	switch (op)
	{
		case LS_D:
			printf("=== sort by modified time ===\n");
			break;
		case LS_S:
			printf("=== sort by size ===\n");
			break;
		default:
			printf("WRONG OP, NOT SUPPORTED!\n");
			return;
	}
	choice_sort(fs, op);
	for (int i = 0; i < fs->FCB_ENTRIES; i++)
	{
		int fcb_addr = i * fs->FCB_SIZE + fs->SUPERBLOCK_SIZE;
		if (fs->volume[fcb_addr] != 0) {
			for (int j = 0; j < 20; j++)
			{
				printf("%c", fs->volume[fcb_addr + j]);
			}
			if (op == LS_S) printf("%d", fcb_get_size(fs, i));
			printf("\n");
		}
	}
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	u32 removed_index = fcbs_search(fs, s);
	if (removed_index == -1) {
		printf("No such file found!\n");
		return;
	}
	fs_clr(fs, removed_index);// Super Block and File-content area cleaned
	for (int i = 0; i < fs->FCB_SIZE; i++)
	{
		fs->volume[removed_index*fs->FCB_SIZE + fs->SUPERBLOCK_SIZE + i] = 0;
	}
	fcbs_swap(fs, removed_index, fs->FCB_SENTINEL);
	// modify the fcbs
	fs->FCB_SENTINEL--;
}

__device__ void choice_sort(FileSystem * fs, int op) {
	for (int i = 0; i <= fs->FCB_SENTINEL; i++){
		int max = i;
		for (int j = i+1; j <= fs->FCB_SENTINEL; j++) {
			max = (fcbs_compare(fs, max, j, op) ? max : j);
		}
		fcbs_swap(fs, i, max);
	}
}

__device__ bool fcbs_compare(FileSystem * fs, u32 fcb_index_lhs, u32 fcb_index_rhs, int op) {
	if (op == LS_D) {
		return (fcb_get_modi_time(fs, fcb_index_lhs) > fcb_get_modi_time(fs, fcb_index_rhs));
	}
	else if (op == LS_S) {
		return (fcb_get_size(fs, fcb_index_lhs) > fcb_get_size(fs, fcb_index_rhs));
	}
	printf("???\n");
}

__device__ void fcbs_swap(FileSystem *fs, u32 fcb_index_a, u32 fcb_index_b) {
	if (fcb_index_a == fcb_index_b) return;
	u32 fcb_addr_a = fcb_index_a * fs->FCB_SIZE + fs->SUPERBLOCK_SIZE;
	u32 fcb_addr_b = fcb_index_b * fs->FCB_SIZE + fs->SUPERBLOCK_SIZE;
	for (int i = 0; i < fs->FCB_SIZE; i++)
	{
		fs->temp[i] = fs->volume[fcb_addr_a + i];
		fs->volume[fcb_addr_a + i] = fs->volume[fcb_addr_b + i];
		fs->volume[fcb_addr_b + i] = fs->temp[i];
	}
}

__device__ u32 fcb_get_name(FileSystem * fs, u32 fcb_index_num) { // Get the addr of file-name
	return (fcb_index_num * 32 + fs->SUPERBLOCK_SIZE);
}

__device__ u32 fcb_get_addr(FileSystem * fs, u32 fcb_index_num) {
	// printf("fcb_get_addr invoked...\n");
	int fcb_addr = fcb_index_num * fs->FCB_SIZE + fs->SUPERBLOCK_SIZE;
	u32 file_block_addr = ((u32)fs->volume[fcb_addr + 20]) +
		((u32)fs->volume[fcb_addr + 21])*(1 << 8) +
		((u32)fs->volume[fcb_addr + 22])*(1 << 16) +
		((u32)fs->volume[fcb_addr + 23])*(1 << 24);

	return file_block_addr;
}

__device__ void fcb_set_addr(FileSystem * fs, u32 fcb_index_num, u32 file_addr) {
	int fcb_addr = fcb_index_num * fs->FCB_SIZE + fs->SUPERBLOCK_SIZE;
	fs->volume[fcb_addr + 20] = (file_addr & 0x000000ff);
	fs->volume[fcb_addr + 21] = ((file_addr >> 8) & 0x000000ff);
	fs->volume[fcb_addr + 22] = ((file_addr >> 16) & 0x000000ff);
	fs->volume[fcb_addr + 23] = ((file_addr >> 24) & 0x000000ff);

}

__device__ u32 fcb_get_size(FileSystem * fs, u32 fcb_index_num) {
	int fcb_addr = fcb_index_num * fs->FCB_SIZE + fs->SUPERBLOCK_SIZE;
	u32 file_size = ((int)fs->volume[fcb_addr + 24]) +
		((int)fs->volume[fcb_addr + 25]) * (1 << 8) +
		((int)fs->volume[fcb_addr + 26]) * (1 << 16);
	return file_size;
}

__device__ void fcb_set_size(FileSystem *fs, u32 fcb_index_num, u32 file_size) {
	int fcb_addr = fcb_index_num * fs->FCB_SIZE + fs->SUPERBLOCK_SIZE;
	fs->volume[fcb_addr + 24] = (file_size & 0x000000ff);
	fs->volume[fcb_addr + 25] = ((file_size >> 8) & 0x000000ff);
	fs->volume[fcb_addr + 26] = ((file_size >> 16)& 0x000000ff);
}

__device__ u32 fcb_get_modi_time(FileSystem * fs, u32 fcb_index_num) {
	int fcb_addr = fcb_index_num * fs->FCB_SIZE + fs->SUPERBLOCK_SIZE;
	u32 file_modi_time = ((int)fs->volume[fcb_addr + 28]) +
		((int)fs->volume[fcb_addr + 29]) * (1 << 8) +
		((int)fs->volume[fcb_addr + 30]) * (1 << 16);
	return file_modi_time;
}

__device__ void fcb_set_modi_time(FileSystem *fs, u32 fcb_index_num, u32 file_modi_time) {
	int fcb_addr = fcb_index_num * fs->FCB_SIZE + fs->SUPERBLOCK_SIZE;
	fs->volume[fcb_addr + 28] = (file_modi_time & 0x000000ff);
	fs->volume[fcb_addr + 29] = ((file_modi_time >> 8) & 0x000000ff);
	fs->volume[fcb_addr + 30] = ((file_modi_time >> 16) & 0x000000ff);
}

__device__ void fs_display(FileSystem *fs, u32 steps) {
	printf("\n\n--------------------------\n");
	printf("Displaying the File System...\n");
	printf("This is: Step %d\n", steps);
	printf("FCB_SENTINEL: %d\n", fs->FCB_SENTINEL);
	printf("SB_SENTINEL: %d\n", fs->SUPERBLOCK_SENTINEL);
	printf("Showing super block:\n");
	for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++)
	{
		if (fs->volume[i] != 0) printf("%d   %d\n", i, (int)fs->volume[i]);
	}
	printf("\nShowing FCBs:\n");

	for (int i = 0; i < fs->FCB_ENTRIES; i++)
	{
		int fcb_addr = i * fs->FCB_SIZE + fs->SUPERBLOCK_SIZE;
		if (fs->volume[fcb_addr] != 0) {
			printf("FCB index: %d\n", i);
			printf("File name: ");
			for (int j = 0; j < 20; j++)
			{
				printf("%c", fs->volume[fcb_addr + j]);
			}
			printf("\n");
			printf("File size: %d\n", fcb_get_size(fs, i));
			printf("File modification time: %d\n", fcb_get_modi_time(fs, i));
			printf("File addr: %d\n\n", fcb_get_addr(fs, i));
		}
	}
}