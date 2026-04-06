

#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
// 此头文件包含 __syncthreads ()函数
#include "device_functions.h"

#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include<stdio.h>
//#include<thrust/reduce.h>
//#include<thrust/sequence.h>
//#include<thrust/host_vector.h>
//#include<thrust/device_vector.h>
#include <string>
#include <algorithm>
#include<cstdlib>
#include<ctime>
#include <fstream>
#include<string.h>
#include<random>
#include<algorithm>
#include<set>
using namespace std;

int MAX_ITER = 20;

clock_t start1, end1;
//一些参数
time_t t;
default_random_engine e(time(0));        // 生成无符号随机整数
 //0 到 1 （包含）的均匀分布
uniform_real_distribution<double > u(0, 1);

double endtime = 0;
//double maxMemory = 0; // the maximum memory usage
const int pop_size = 200;//the size of population
const int item_length = 41270;//item的数量5 75 41270
int database_length = 990002;//数据库行数7 3196 990002
int chrom_len = 0; //TWU大于minsup的item数量
const int block = 64;
//string input = "test.txt";
char input[100] = "kosarak_utility_spmf.txt";//T10I6D1000K
string output = "result.txt";
const int minUtility = 1000000;//40 60000 625430 614647 603864 593080 582297 2156658

//几个hash表
int hash1[100000] = { 0 };
int hash_item[100000] = { 0 };
int hash_eachitem_twu[100000] = { 0 };
int hash_twu[100000] = { 0 };
int hash_support[100000] = { 0 };
int *hash_sort_cpu;
int *hash_sort_gpu;
int *eachitem_twu;
int *eachitem_twu_gpu;
int *one_twu;
int *support;
int *item;
int *item1;
int *item2;
int *item_gpu;
int *item1_gpu;
int *item2_gpu;

int *database;
int *database_utility;

int *database_gpu;
int *database_utility_gpu;

int *eachline_length;
int *start_position;
int *eachline_length_gpu;
int *start_position_gpu;

int max_database_size = 0;
int chrom_len_num = 0;//满足1-twu的最大数，方便建立散列表

int *population_gpu;//记录种群的hash表
int *population_gpu1;//多种群搜索1
int *population_gpu2;//多种群搜索2
int *population_gpu3;//多种群搜索3
int *fitness_gpu;//记录适应度
int *fitness_gpu1;//记录适应度
int *fitness_gpu2;//记录适应度
int *fitness_gpu3;//记录适应度
int *fitness_cpu1;//记录适应度
int *fitness_cpu2;//记录适应度
int *fitness_cpu3;//记录适应度
int *population_count_one_gpu;//记录为1的个数，方便适应度计算
int *population_count_one_gpu1;//记录为1的个数，方便适应度计算
int *population_count_one_gpu2;//记录为1的个数，方便适应度计算
int *population_count_one_gpu3;//记录为1的个数，方便适应度计算
int *result_gpu;
int *result_length;

int *frequent_item_gpu;
int *frequent_length_gpu;
void read_txt()
{
	FILE *fp;
	char str[20000];

	/* 打开用于读取的文件 */
	fp = fopen(input, "r");
	int i = 0;
	while (fgets(str, 20000, fp) != NULL) {

		int tmp_line[20000];
		int k = 0;
		int n = 0;
		int tmp_num = 0;
		while (str[k] != ':')
		{
			if (str[k] == ' ')
			{
				tmp_line[n] = tmp_num;
				tmp_num = 0;
				n++;
			}
			else
			{
				tmp_num = tmp_num * 10 + (str[k] - '0');
			}
			k++;
		}
		tmp_line[n] = tmp_num;
		if (i == 0)
		{
			start_position[0] = 0;
		}
		else
		{
			start_position[i] = start_position[i - 1] + eachline_length[i - 1];
		}
		eachline_length[i] = n + 1;//每行的长度
		k++;
		tmp_num = 0;
		while (str[k] != ':')
		{
			tmp_num = tmp_num * 10 + (str[k] - '0');
			k++;
		}
		int eachline_twu = tmp_num;
		for (int i = 0; i <= n; i++)
		{
			if (hash1[tmp_line[i]] + eachline_twu < 0)
			{
				hash1[tmp_line[i]] = INT_MAX;
			}
			else
			{
				hash1[tmp_line[i]] += eachline_twu;
			}
		}

		i++;
	}
	for (int i = 0; i < 100000; i++)
	{
		//cout << hash1[i] << endl;
		if (hash1[i] >= minUtility)
		{
			hash_item[i] = 1;
			hash_twu[i] = hash1[i];
			chrom_len++;
		}
	}
	fclose(fp);

}
void read_txt2()
{
	///////////////////////////////////////////////二次读取文件
	FILE *fp;
	char str[20000];

	/* 打开用于读取的文件 */
	/* 打开用于读取的文件 */
	fp = fopen(input, "r");
	int i = 0;
	int ii = 0;
	while (fgets(str, 20000, fp) != NULL) {

		int tmp_line[20000];
		int tmp_utility[20000];
		int k = 0;
		int n = 0;
		int len = 0;
		int tmp_num = 0;
		while (str[k] != ':')
		{
			if (str[k] == ' ')
			{
				tmp_line[n] = tmp_num;
				tmp_num = 0;
				n++;
			}
			else
			{
				tmp_num = tmp_num * 10 + (str[k] - '0');
			}
			k++;
		}
		tmp_line[n] = tmp_num;
		len = n + 1;
		k++;
		tmp_num = 0;
		while (str[k] != ':')
		{
			k++;
		}
		tmp_num = 0;
		int m = 0;
		k++;
		while (k < strlen(str))
		{
			if (str[k] == ' ' || str[k] == '\n')
			{
				tmp_utility[m] = tmp_num;
				tmp_num = 0;
				m++;
			}
			else
			{
				tmp_num = tmp_num * 10 + (str[k] - '0');
			}
			k++;
		}
		k = 0;
		for (int id = 0; id < len; id++)
		{
			int tmp = tmp_line[id];
			if (hash_item[tmp] == 1)
			{
				if (hash_twu[tmp] >= minUtility)
				{
					hash_support[tmp] += 1;
					hash_eachitem_twu[tmp] += tmp_utility[id];
					database[ii] = tmp;
					database_utility[ii] = tmp_utility[id];
					//cout << database_utility[ttid + k] << endl;
					ii++;
					k++;
				}
			}

		}
		if (k != 0)
		{
			if (i == 0)
			{
				start_position[0] = 0;
			}
			else
			{
				start_position[i] = start_position[i - 1] + eachline_length[i - 1];
			}
			//cout << k << endl;
			eachline_length[i] = k;
			i++;
		}
	}
	database_length = i - 1;
	cout << "database_length = " << database_length << endl;
	fclose(fp);
}
void quick_sort(int left, int right, int *&item, int *&utility)
{
	int i, j, c, temp, temp1;
	if (left > right)
		return;

	i = left;
	j = right;
	temp = utility[i];
	temp1 = item[i];
	while (i != j)
	{
		while (utility[j] <= temp && i < j)
		{
			j--;
		}

		while (utility[i] >= temp && i < j)
		{
			i++;
		}

		if (i < j)
		{
			c = utility[i];
			utility[i] = utility[j];
			utility[j] = c;
			c = item[i];
			item[i] = item[j];
			item[j] = c;
		}
	}


	//left为起始值（参照值）此时的I为第一次排序结束的最后值，与参照值交换位置
	utility[left] = utility[i];
	utility[i] = temp;
	item[left] = item[i];
	item[i] = temp1;

	//继续递归直到排序完成
	quick_sort(left, i - 1, item, utility);
	quick_sort(i + 1, right, item, utility);
}

__global__ void init_population(int *population_gpu, int *item_gpu, int chrom_len, int *population_count_one_gpu, long rand)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curandState state;
	curand_init(rand, id, 0, &state);
	int x = curand_uniform(&state) * 100;
	//printf("id = %d,%d\n",id, item_gpu[x]);
	population_gpu[id * chrom_len + x] = 1;
	population_count_one_gpu[id] = 1;

}
__global__ void printf_kernel(int *database_gpu, int *eachline_length_gpu, int *start_position_gpu)
{
	for (int i = 0; i < 10; i++)
	{
		printf("%d,%d::", i, start_position_gpu[i]);
		for (int j = 0; j < eachline_length_gpu[i]; j++)
		{
			printf("%d ", database_gpu[start_position_gpu[i] + j]);
		}
		printf("\n");
	}
}
__global__ void printf_kernel1(int *frequent_length_gpu)
{
	printf("%d,", frequent_length_gpu[0]);
}
__global__ void printf_kernel2(int *frequent_length_gpu)
{
	for (int i = 0; i < 10; i++)
	{
		printf("%d\n", frequent_length_gpu[i]);
	}

}
__global__ void printf_kernel3(int *result_gpu, int *result_length)
{
	printf("result = :\n");
	for (int i = 0; i < result_length[0]; i++)
	{
		printf("%d, ", result_gpu[i]);
	}
}
__global__ void printf_kernel4(int *population_gpu, int chrom_len)
{
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < chrom_len; j++)
		{
			if (population_gpu[i * chrom_len + j] == 1)
			{
				printf("(%d) ", j, population_gpu[i * chrom_len + j]);
			}

		}
		printf("\n");
	}
}
__global__ void calculate_fitness(int *population_gpu, int *database_gpu, int *database_utility_gpu, int *population_count_one_gpu, int *fitness_gpu, int database_length, int *eachline_length_gpu, int *start_position_gpu, int chrom_len)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	int bid = blockIdx.y;
	int ttid = tid + blockDim.x * blockIdx.x;
	int num1 = 3;
	//printf("bid = %d\n", bid);
	int iter1 = 0;//计算迭代次数
	if (ttid * num1 + num1 - 1 < database_length)
	{
		iter1 = num1;
	}
	else
	{
		iter1 = database_length - ttid * num1 + 1;
	}

	int tmp_fitness = 0;
	for (int ii = 0; ii < iter1; ii++)
	{
		int k = 0;
		int tmp = 0;
		for (int i = 0; i < eachline_length_gpu[ttid * num1 + ii]; i++)
		{
			int num = start_position_gpu[ttid * num1 + ii] + i;//数据库第几行第几个元素
			if (population_gpu[bid * chrom_len + database_gpu[num]] == 1)//如果存在这个元素
			{
				//printf("k = %d\n", k);
				k++;
				tmp += database_utility_gpu[num];
			}
		}
		if (k == population_count_one_gpu[bid])//如果结果符合，保存到共享内存中
		{
			tmp_fitness += tmp;
		}
	}
	atomicAdd(&fitness_gpu[bid], tmp_fitness);

}
__global__ void calculate_result(int *fitness_gpu, int *result_gpu, int *result_length, int *frequent_item_gpu, int *frequent_length_gpu, int * population_gpu, int chrom_len, int MAX_ITER)
{
	for (int i = 0; i < pop_size; i++)
	{
		int fitness = fitness_gpu[i];
		if (fitness >= minUtility)
		{
			int flag = 0;
			for (int j = 0; j < result_length[0]; j++)
			{
				if (result_gpu[j] == fitness)
				{
					flag = 1;
					break;
				}
			}
			if (!flag)
			{
				for (int k = 0; k < chrom_len; k++)
				{
					if (population_gpu[i * chrom_len + k] == 1)
					{
						int flag1 = 0;
						for (int m = 0; m < frequent_length_gpu[0]; m++)
						{
							if (frequent_item_gpu[m] == k)
							{
								flag1 = 1;
								break;
							}
						}
						if (!flag1)
						{
							//printf("k = -------%d\n", k);
							frequent_item_gpu[frequent_length_gpu[0]++] = k;
						}

					}
				}
				result_gpu[result_length[0]] = fitness;
				//printf("--result_length = %d,frequent_length_gpu = %d,MAX_ITER = %d\n", result_length[0], frequent_length_gpu[0], MAX_ITER);
				//printf("%d,", 500 - MAX_ITER);
				result_length[0] += 1;
			}
		}
	}

}
__global__ void mutation(int *population_gpu1, int *item_gpu, int chrom_len, int *population_count_one_gpu1, long rand, int MAX_ITER, int *frequent_item_gpu, int *frequent_length_gpu)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curandState state;
	curand_init(rand, id, 0, &state);

	while (true)
	{
		int x;
		int ttid;
		if (MAX_ITER < 10)
		{
			if (id < 0)
			{
				x = curand_uniform(&state) * frequent_length_gpu[0];
				ttid = id * chrom_len + frequent_item_gpu[x];
			}
			else
			{
				x = curand_uniform(&state) * 100;
				int x1 = curand_uniform(&state) * 100;
				x = x < x1 ? x : x1;
				//printf("id = %d,%d\n",id, item_gpu[x]);
				ttid = id * chrom_len + x;
			}

		}
		else
		{
			x = curand_uniform(&state) * 100;
			int x1 = curand_uniform(&state) * 100;
			x = x < x1 ? x : x1;
			//printf("id = %d,%d\n",id, item_gpu[x]);
			ttid = id * chrom_len + x;
		}

		if (population_gpu1[ttid] == 1)
		{
			population_gpu1[ttid] = 0;
			population_count_one_gpu1[id] -= 1;
			break;
		}
		else
		{
			if (population_count_one_gpu1[id] < 5)
			{
				population_gpu1[ttid] = 1;
				population_count_one_gpu1[id] += 1;
				break;
			}
		}
	}
}

__global__ void kernel1(int *population_gpu, int * population_gpu1, int * population_gpu2, int * population_gpu3, int * population_count_one_gpu, int *population_count_one_gpu1, int *population_count_one_gpu2, int *population_count_one_gpu3, int chrom_len, int *fitness_gpu, int *fitness_gpu1, int *fitness_gpu2, int *fitness_gpu3, int rand, int MAX_ITER)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	int bid = blockIdx.y;
	int ttid = tid + blockDim.x * blockIdx.x;
	if (ttid >= chrom_len)
	{
		return;
	}
	//判断那个最大，记录位置
	int tmp_num = 1;//第一大
	int tmp_num1 = 2;//第二大
	int big = fitness_gpu[bid];
	int small = fitness_gpu1[bid];
	if (big < small)
	{
		int tmp = big;
		big = small;
		small = tmp;
		tmp_num1 = 1;
		tmp_num = 2;
	}
	if (fitness_gpu2[bid] > big)
	{
		big = fitness_gpu2[bid];
		tmp_num1 = tmp_num;
		tmp_num = 3;
		small = big;
	}
	else
	{
		if (fitness_gpu2[bid] > small)
		{
			small = fitness_gpu2[bid];
			small = 3;
		}
	}
	if (fitness_gpu3[bid] > big)
	{
		big = fitness_gpu3[bid];
		small = big;
		tmp_num1 = tmp_num;
		tmp_num = 4;
	}
	else
	{
		if (fitness_gpu3[bid] > small)
		{
			small = fitness_gpu3[bid];
			small = 4;
		}
	}
	if (MAX_ITER < 40 && rand >= 8)
	{
		//tmp_num = tmp_num1;
	}
	int index = bid * chrom_len + ttid;
	if (tmp_num == 2)
	{
		population_gpu[index] = population_gpu1[index];
		if (ttid == 0)
		{
			fitness_gpu[bid] = fitness_gpu1[bid];
			population_count_one_gpu[bid] = population_count_one_gpu1[bid];
		}
	}
	else if (tmp_num == 3)
	{
		population_gpu[index] = population_gpu2[index];
		if (ttid == 0)
		{
			fitness_gpu[bid] = fitness_gpu2[bid];
			population_count_one_gpu[bid] = population_count_one_gpu2[bid];
		}
	}
	else if (tmp_num == 4)
	{
		population_gpu[index] = population_gpu3[index];
		if (ttid == 0)
		{
			fitness_gpu[bid] = fitness_gpu3[bid];
			population_count_one_gpu[bid] = population_count_one_gpu3[bid];
		}
	}
}

__global__ void calculate_result_one(int *eachitem_twu_gpu, int *result_gpu, int *result_length, int chrom_len)
{
	for (int i = 0; i < chrom_len; i++)
	{
		if (eachitem_twu_gpu[i] >= minUtility)
		{
			result_gpu[result_length[0]] = eachitem_twu_gpu[i];
			result_length[0] += 1;
		}
	}
	printf("%d,", result_length[0]);
}

__global__ void parrallel_map_process(int *database_gpu, int *hash_sort_gpu, int *eachline_length_gpu, int *start_position_gpu, int database_length)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if (id >= database_length)
	{
		return;
	}
	//printf("%d\n", id);
	int start = start_position_gpu[id];
	for (int i = 0; i < eachline_length_gpu[id]; i++)
	{
		database_gpu[start + i] = hash_sort_gpu[database_gpu[start + i]];
	}
}

__device__ void quick_sort_device(int *database_gpu, int *database_utility_gpu, int left, int right)
{
	int i, j, c, temp, temp1;
	if (left > right)
		return;

	i = left;
	j = right;
	temp = database_gpu[i];
	temp1 = database_utility_gpu[i];
	while (i != j)
	{
		while (database_gpu[j] <= temp && i < j)
		{
			j--;
		}

		while (database_gpu[i] >= temp && i < j)
		{
			i++;
		}

		if (i < j)
		{
			c = database_gpu[i];
			database_gpu[i] = database_gpu[j];
			database_gpu[j] = c;
			c = database_utility_gpu[i];
			database_utility_gpu[i] = database_utility_gpu[j];
			database_utility_gpu[j] = c;
		}
	}
	//left为起始值（参照值）此时的I为第一次排序结束的最后值，与参照值交换位置
	database_gpu[left] = database_gpu[i];
	database_gpu[i] = temp;
	database_utility_gpu[left] = database_utility_gpu[i];
	database_utility_gpu[i] = temp1;

	//继续递归直到排序完成
	quick_sort_device(database_gpu, database_utility_gpu, left, i - 1);
	quick_sort_device(database_gpu, database_utility_gpu, i + 1, right);
}
__global__ void parrallel_sort_process(int *database_gpu, int *eachline_length_gpu, int *start_position_gpu, int database_length, int *database_utility_gpu)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if (id >= database_length)
	{
		return;
	}
	int start = start_position_gpu[id];
	quick_sort_device(database_gpu, database_utility_gpu, start, start + eachline_length_gpu[id] - 1);
}
int main()
{
	cout << "begin!!!!!!!" << endl;

	eachline_length = (int *)malloc(database_length * sizeof(int));
	start_position = (int *)malloc(database_length * sizeof(int));
	srand((unsigned)time(NULL));
	cudaMalloc((void**)&frequent_item_gpu, 1000 * sizeof(int));
	cudaMalloc((void**)&frequent_length_gpu, sizeof(int));
	//开始计时
	start1 = clock();
	//////////////////////////////////////////////////////////////////////////////////////读文件
	read_txt();//第一次读文件计算i-twu剪枝

	max_database_size = start_position[database_length - 1] + eachline_length[database_length - 1];
	//cout << "max_database_size = " << max_database_size << endl;
	database = (int *)malloc(max_database_size * sizeof(int));
	database_utility = (int *)malloc(max_database_size * sizeof(int));

	read_txt2();//第二次读文件保存压缩成线性的数据库，保存每行初始点，长度

	max_database_size = start_position[database_length - 1] + eachline_length[database_length - 1];
	cout << "max_database_size = " << max_database_size << endl;
	cout << "chrom_len = " << chrom_len << endl;
	eachitem_twu = (int *)malloc(chrom_len * sizeof(int));
	cudaMalloc((void**)&eachitem_twu_gpu, chrom_len * sizeof(int));
	one_twu = (int *)malloc(chrom_len * sizeof(int));
	support = (int *)malloc(chrom_len * sizeof(int));
	item = (int *)malloc(chrom_len * sizeof(int));
	item1 = (int *)malloc(chrom_len * sizeof(int));
	item2 = (int *)malloc(chrom_len * sizeof(int));

	int k = 0;
	for (int i = 0; i < 100000; i++)
	{
		//cout << hash_item[i] << endl;
		if (hash_item[i] == 1)
		{
			eachitem_twu[k] = hash_eachitem_twu[i];
			one_twu[k] = hash_twu[i];
			if (k < chrom_len)
			{
				item[k] = i;
			}
			support[k] = hash_support[i];
			k++;
		}
	}
	memcpy(item1, item, chrom_len * sizeof(int));
	memcpy(item2, item, chrom_len * sizeof(int));

	//chrom_len++;
	cout << "chrom_len = " << chrom_len << endl;
	//对item,eachitem_twu,one_twu,support排序,先对one_twu一个排序
	quick_sort(0, k - 1, item, one_twu);
	quick_sort(0, k - 1, item1, eachitem_twu);
	quick_sort(0, k - 1, item2, support);


	/*for (int i = 0; i <k; i++)
	{
		cout << eachitem_twu[i] << endl;
	}*/
	///////////////////////////////////////////////////////////////////////////////////初始化蜂群
	//种群初始化，使用hash表初始化而不是bit
	//拷贝到gpu

	cudaMalloc((void**)&fitness_gpu, pop_size * sizeof(int));
	cudaMalloc((void**)&fitness_gpu1, pop_size * sizeof(int));
	cudaMalloc((void**)&fitness_gpu2, pop_size * sizeof(int));
	cudaMalloc((void**)&fitness_gpu3, pop_size * sizeof(int));
	fitness_cpu1 = (int *)malloc(pop_size * sizeof(int));
	fitness_cpu2 = (int *)malloc(pop_size * sizeof(int));
	fitness_cpu3 = (int *)malloc(pop_size * sizeof(int));
	cudaMalloc((void**)&result_gpu, pop_size * sizeof(int));
	cudaMalloc((void**)&result_length, 1 * sizeof(int));
	cudaMalloc((void**)&population_count_one_gpu, pop_size * sizeof(int));
	cudaMalloc((void**)&population_count_one_gpu1, pop_size * sizeof(int));
	cudaMalloc((void**)&population_count_one_gpu2, pop_size * sizeof(int));
	cudaMalloc((void**)&population_count_one_gpu3, pop_size * sizeof(int));
	cudaMalloc((void**)&database_gpu, max_database_size * sizeof(int));
	cudaMalloc((void**)&database_utility_gpu, max_database_size * sizeof(int));
	cudaMalloc((void**)&eachline_length_gpu, database_length * sizeof(int));
	cudaMalloc((void**)&start_position_gpu, database_length * sizeof(int));
	cudaMemcpy(database_gpu, database, max_database_size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(database_utility_gpu, database_utility, max_database_size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(eachline_length_gpu, eachline_length, database_length * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(start_position_gpu, start_position, database_length * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&item_gpu, chrom_len * sizeof(int));
	cudaMalloc((void**)&item1_gpu, chrom_len * sizeof(int));
	cudaMalloc((void**)&item2_gpu, chrom_len * sizeof(int));


	cudaMemset(population_count_one_gpu, 0, pop_size * sizeof(int));
	cudaMemset(result_gpu, 0, 1 * sizeof(int));
	cudaMemcpy(item_gpu, item, chrom_len * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(item1_gpu, item1, chrom_len * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(item2_gpu, item2, chrom_len * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(eachitem_twu_gpu, eachitem_twu, chrom_len * sizeof(int), cudaMemcpyHostToDevice);

	hash_sort_cpu = (int *)malloc(100000 * sizeof(int));
	memset(hash_sort_cpu, 0, 100000 * sizeof(int));
	cudaMalloc((void**)&hash_sort_gpu, 100000 * sizeof(int));
	for (int i = 0; i < chrom_len; i++)//生成cpu_hash
	{
		hash_sort_cpu[item1[i]] = i;
	}
	cudaMemcpy(hash_sort_gpu, hash_sort_cpu, 100000 * sizeof(int), cudaMemcpyHostToDevice);//生成gpu_hash
	int split_database_num = database_length / block + 1;
	//并行映射数据库每行
	parrallel_map_process << <split_database_num, block >> > (database_gpu, hash_sort_gpu, eachline_length_gpu, start_position_gpu, database_length);
	//并行排序数据库每行
	//parrallel_sort_process << <split_database_num, block >> > (database_gpu, eachline_length_gpu, start_position_gpu, database_length, database_utility_gpu);
	printf_kernel << <1, 1 >> > (database_gpu, eachline_length_gpu, start_position_gpu);

	cudaMalloc((void**)&population_gpu, pop_size * chrom_len * sizeof(int));
	cudaMalloc((void**)&population_gpu1, pop_size * chrom_len * sizeof(int));
	cudaMalloc((void**)&population_gpu2, pop_size * chrom_len * sizeof(int));
	cudaMalloc((void**)&population_gpu3, pop_size * chrom_len * sizeof(int));
	cudaMemset(population_gpu, 0, pop_size * chrom_len * sizeof(int));

	//计算1-item的结果
	calculate_result_one << < 1, 1 >> > (eachitem_twu_gpu, result_gpu, result_length, chrom_len);
	//初始化种群
	init_population << <1, pop_size >> > (population_gpu, item1_gpu, chrom_len, population_count_one_gpu, rand());
	int grid1 = database_length / (block * 3) + 1;
	//cout << "grid1 = " << grid1 << endl;
	dim3 grid(grid1, pop_size);
	//适应度计算,并计算结果
	printf_kernel4 << <1, 1 >> > (population_gpu, chrom_len);
	calculate_fitness << <grid, block >> > (population_gpu, database_gpu, database_utility_gpu, population_count_one_gpu, fitness_gpu, database_length, eachline_length_gpu, start_position_gpu, chrom_len);
	printf_kernel2 << <1, 1 >> > (fitness_gpu);
	//calculate_result << <1, pop_size >> > (fitness_gpu, result_gpu, result_length);
	int grid2 = chrom_len / block + 1;
	dim3 grid11(grid2, pop_size);

	//进入循环

	calculate_result << <1, 1 >> > (fitness_gpu, result_gpu, result_length, frequent_item_gpu, frequent_length_gpu, population_gpu, chrom_len, MAX_ITER);
	//////////////////////////////////////
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//////////////////////////////////////
	while (MAX_ITER--)
	{
		cudaMemcpy(population_gpu1, population_gpu, pop_size * chrom_len * sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(population_gpu2, population_gpu, pop_size * chrom_len * sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(population_gpu3, population_gpu, pop_size * chrom_len * sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(population_count_one_gpu1, population_count_one_gpu, pop_size * sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(population_count_one_gpu2, population_count_one_gpu, pop_size * sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(population_count_one_gpu3, population_count_one_gpu, pop_size * sizeof(int), cudaMemcpyDeviceToDevice);
		//重置参数
		cudaMemset(fitness_gpu1, 0, pop_size * sizeof(int));
		cudaMemset(fitness_gpu2, 0, pop_size * sizeof(int));
		cudaMemset(fitness_gpu3, 0, pop_size * sizeof(int));
		//变异
		mutation << <1, pop_size >> > (population_gpu1, item_gpu, chrom_len, population_count_one_gpu1, rand(), MAX_ITER, frequent_item_gpu, frequent_length_gpu);
		mutation << <1, pop_size >> > (population_gpu2, item1_gpu, chrom_len, population_count_one_gpu2, rand(), MAX_ITER, frequent_item_gpu, frequent_length_gpu);
		mutation << <1, pop_size >> > (population_gpu3, item2_gpu, chrom_len, population_count_one_gpu3, rand(), MAX_ITER, frequent_item_gpu, frequent_length_gpu);
		//计算适应度

		calculate_fitness << <grid, block >> > (population_gpu1, database_gpu, database_utility_gpu, population_count_one_gpu1, fitness_gpu1, database_length, eachline_length_gpu, start_position_gpu, chrom_len);
		calculate_fitness << <grid, block >> > (population_gpu2, database_gpu, database_utility_gpu, population_count_one_gpu2, fitness_gpu2, database_length, eachline_length_gpu, start_position_gpu, chrom_len);
		calculate_fitness << <grid, block >> > (population_gpu3, database_gpu, database_utility_gpu, population_count_one_gpu3, fitness_gpu3, database_length, eachline_length_gpu, start_position_gpu, chrom_len);

		//保存结果
		calculate_result << < 1, 1 >> > (fitness_gpu1, result_gpu, result_length, frequent_item_gpu, frequent_length_gpu, population_gpu1, chrom_len, MAX_ITER);//1,1的
		calculate_result << <1, 1 >> > (fitness_gpu2, result_gpu, result_length, frequent_item_gpu, frequent_length_gpu, population_gpu2, chrom_len, MAX_ITER);
		calculate_result << <1, 1 >> > (fitness_gpu3, result_gpu, result_length, frequent_item_gpu, frequent_length_gpu, population_gpu3, chrom_len, MAX_ITER);
		printf_kernel1 << <1, 1 >> > (result_length);
		//选择最优的种群

		kernel1 << <grid11, block >> > (population_gpu, population_gpu1, population_gpu2, population_gpu3, population_count_one_gpu, population_count_one_gpu1, population_count_one_gpu2, population_count_one_gpu3, chrom_len, fitness_gpu, fitness_gpu1, fitness_gpu2, fitness_gpu3, rand() % 10, MAX_ITER);

	}
	//////////////////////////////////////
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "elapsedTime = " << elapsedTime << endl;
	//////////////////////////////////////
	/*for (set<int>::iterator it = se.begin(); it != se.end(); it++)
	{
		cout << *it << " occurs " << endl;
	}*/
	//calculate_fitness;
	//printf_kernel << <1, 1 >> > (population_gpu);
	cudaMemset(fitness_gpu, 0, pop_size * sizeof(int));
	printf_kernel4 << <1, 1 >> > (population_gpu, chrom_len);
	//printf_kernel3 << <1, 1 >> > (result_gpu, result_length);
	calculate_fitness << <grid, block >> > (population_gpu, database_gpu, database_utility_gpu, population_count_one_gpu, fitness_gpu, database_length, eachline_length_gpu, start_position_gpu, chrom_len);
	printf_kernel2 << <1, 1 >> > (fitness_gpu);
	cout << "计时结束" << endl;
	//cout << endl;
	end1 = clock();		//程序结束用时
	endtime = (double)(end1 - start1) / CLOCKS_PER_SEC;
	cout << "Total time:" << endtime * 1000 << "ms" << endl;	//ms为单位

	cudaFree(database_utility_gpu);
	cudaFree(eachline_length_gpu);
	cudaFree(start_position_gpu);
	cudaFree(population_gpu);
	cudaFree(population_gpu1);
	cudaFree(population_gpu2);
	cudaFree(population_gpu3);
	cudaFree(fitness_gpu);
	cudaFree(population_count_one_gpu);
	cudaFree(population_count_one_gpu1);
	cudaFree(population_count_one_gpu2);
	cudaFree(population_count_one_gpu3);

	//释放cpu内存
	free(database);
	free(database_utility);
	free(eachline_length);
	free(start_position);
	free(item);
	free(item1);
	free(item2);
	free(support);
	free(eachitem_twu);
	free(one_twu);
}
