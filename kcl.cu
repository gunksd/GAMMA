//目的是在给定的图中查找 k-clique（k 完全子图），也就是由 k 个节点组成的相互连接的子图。
#include <stdlib.h>
#include <iostream>
#include "utils.h"
#include "accessMode.cuh"
#include "expand.cuh"
#include <cuda_runtime.h>

using namespace std;
void expand(int i) {
	return ;
}
__global__ void set_validation(OffsetT *row_start, uint8_t *valid_candi, uint32_t nnodes, uint32_t min_deg) {
	uint32_t tid = threadIdx.x + blockDim.x*blockIdx.x;
	for (uint32_t i = tid; i < nnodes; i += (blockDim.x*gridDim.x)) {
		if (row_start[i+1] - row_start[i] >= min_deg)
			valid_candi[i] = 1;
	}
	return ;
}
//主要的参数有文件名、k 值（要找的 clique 大小）、图内存类型（存储方式）和是否为调试模式。
int main(int argc, char *argv[]) {
	if (argc < 4) {
		printf("usage: ./kcl ($filename) ($clique size) graph_mem_type debug\n");
		return 0;
	}
	if (string(argv[argc-1]) != "debug") {
		log_set_quiet(true);
	}
	Clock start("Start");
	uint32_t k = std::atoi(argv[2]);
	assert(k <= embedding_max_length);
	std::string file_name = argv[1];
	CSRGraph data_graph;
	mem_type mt_emb = (mem_type)1;//0 GPU 1 Unified 2 Zero 3 Combine
	mem_type mt_graph = (mem_type)atoi(argv[3]);
	if (mt_graph > 1)
		check_cuda_error(cudaSetDeviceFlags(cudaDeviceMapHost));
		//图读取
	data_graph.read(file_name, false, mt_graph);//no label for k-clique
	log_info(start.start());
	log_info(start.count("nedges %lu, nnodes %d", data_graph.get_nedges(), data_graph.get_nnodes()));
	EmbeddingList emb_list;
	uint32_t nnodes = data_graph.get_nnodes();
	uint64_t nedges = data_graph.get_nedges();
	log_info(start.count("embedding initialization done!"));
	//check_cuda_error(cudaDeviceSynchronize());
	//TODO: here we plan to add a optimizer to determine expand order, expand constraint, and so on.
	//set the first level
	KeyT *seq, *results; 
	cudaMalloc((void **)&seq, sizeof(KeyT)*nnodes);
	check_cuda_error(cudaMalloc((void **)&results, sizeof(KeyT)*nnodes));
	check_cuda_error(cudaMemset(results, -1, sizeof(KeyT)*nnodes));
	uint8_t *valid_candi;
	//首先通过 set_validation 内核，筛选出度数大于等于 k-1 的节点，作为有效候选节点。
	//这样可以减少初始的候选节点数目，从而加快后续扩展的效率
	check_cuda_error(cudaMalloc((void **)&valid_candi, sizeof(uint8_t)*nnodes));
	check_cuda_error(cudaMemset(valid_candi, 0, sizeof(uint8_t)*nnodes));
	set_validation<<<10000, 256>>>(data_graph.row_start, valid_candi, nnodes, k-1);
	thrust::sequence(thrust::device, seq, seq + nnodes);
	uint32_t valid_node_num = thrust::copy_if(thrust::device, seq, seq + nnodes, valid_candi, results, is_valid())- results;
	check_cuda_error(cudaDeviceSynchronize());
	emb_list.init(valid_node_num, k, mt_emb, false);//初始化创建列表，valid_node_num创建第一层
	emb_list.copy_to_level(0, results, 0, valid_node_num);
	check_cuda_error(cudaFree(seq));
	check_cuda_error(cudaFree(results));

	//set the second level
	//emb_list.add_level(nedges);
	//expand for every vertex in the query graph
	access_mode_controller access_controller;//设置图的访问模式，基于当前嵌入列表和扩展约束计算最优的访问方式。
	access_controller.set_vertex_page_border(data_graph);
	log_info(start.count("access controller initalization done!"));
	Clock Expand("Expand");
	log_info(Expand.start());
	for (int i = 1; i < k; i ++) {
		//construct the expand constraint
		uint64_t _nbrs = 0, _order_nbr = 0;
		//int8_t *_order_nbr_cmp = new int8_t [i];
		for (uint8_t j = 0; j < i; j ++) {
			_nbrs = _nbrs | (j << (j*8));
			//_order_nbr_cmp[j] = 1;
			_order_nbr = _order_nbr | (j << (j*8));
		}
		expand_constraint ec((node_data_type)0xff, (uint8_t)k-1, _nbrs, (uint8_t)i, 
							 (emb_order)1, _order_nbr, (uint8_t)i);
		//expand
		log_info(Expand.count("for the %dth iteration, start expand... ...",i));
		bool write_back = i == k-1 ? false : true;
		expand_dynamic(data_graph, emb_list, i, ec, write_back);
		//expand_in_batch(data_graph, emb_list, i, ec);
		log_info(Expand.count("for the %dth iteration, end expand",i));
		Expand.pause();
		//emb_off_type results = emb_list.check_valid_num(i);
		Expand.goon();
		//set access mode
		if (mt_graph == 3) {
			Expand.pause();
			access_controller.cal_access_mode_by_EL(data_graph, ec, emb_list);
			Expand.goon();
		}
		log_info(Expand.count("for the %dth iteration, end set access mode",i));
		//delete ec;
	}
	log_info(Expand.count("end expand"));
	log_info(start.count("k-clique count ends."));
	//#TODO copy the results back to CPU and check the results;
	//CSRGraph data_graph_h;
	//data_graph.copy_to_cpu(data_graph_h);
	//#show the results in data_graph_h
	emb_list.clean();
	access_controller.clean();
	data_graph.clean();//清理

	return 0;
}
//实现了在图中查找 k-clique 的功能，使用 CUDA 来加速查找过程。
//代码的核心思路是通过嵌入扩展的方法逐步找到符合条件的 k 个节点的完全子图，
//使用csr数据结构和 set_validation() 函数进行节点筛选，结合嵌入扩展，最终实现了有效的 k-clique 搜索。
