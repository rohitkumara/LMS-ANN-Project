#include <stdio.h>
#include <stdlib.h>

void main(){
	
	int h_t_1[32] = {0};
	int x_t[32] = {0};
	int W_z[32] = {0};
	int U_z[32] = {0};
	int W_r[32] = {0};
	int U_r[32] = {0};
	int W[32] = {0};
	int U[32] = {0};
	gru_single_cell(h_t_1, x_t, W_z, U_z, W_r, U_r, W, U);

}

void gru_single_cell(int h_t_1[], int x_t[], int W_z[], int U_z[], int W_r[], int U_r[], int W[], int U[]){
	// int z_t = sum_funct(hadamard(W_z, x_t), hadamard(U_z, h_t_1));

}

int* hadamard(int first_array[], int second_array[]){
	int* return_array = (int *)malloc(32*sizeof(int));
	for(int i=0; i<32; i++){
		return_array[i] = first_array[i]*second_array[i];
	}
	return return_array;
}

int* sum_funct(int first_array[], int second_array[]){
	int* return_array = (int *)malloc(32*sizeof(int));
	for(int i=0; i<32; i++){
		return_array[i] = first_array[i]+second_array[i];
	}
	return return_array;
}


