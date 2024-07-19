#ifndef _H_BPCG
#define _H_BPCG

float** bpcg_optimizer(float**, int, int, int, float, int*, int*);

float** file_read(int, char*, int*, int*);
int     file_write(char*, int, int, int, float**, int*);

#endif