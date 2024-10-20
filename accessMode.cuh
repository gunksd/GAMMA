//对大规模图数据进行访问模式分析和管理。
//它利用 CUDA 的 GPU 并行能力实现了多种操作，包括内存页面拆分、访问模式设置、累积访问时间统计、以及图中邻接点访问计数。

#include "graph.cuh"
#include "utils.h"
#include "embedding.cuh"
#include "expand.cuh"

//识别内存页面的边界，存储到 memory_page_split 中
//KeyT *memory_page_split用于存储页面分割位置
//OffsetT *vertexList包含所有顶点的偏移量。
//KeyT vertex_count：顶点总数
//uint32_t total_memory_pages：总的内存页面数量。
//使用 blockDim 和 gridDim 来并行计算每个顶点所在的页面。
//页面大小设定为 4000 字节，每个顶点按照内存大小被分配到不同的页面中。
//当顶点跨越到新的页面时，记录下这个位置以标记页面的开始。

__global__ void memory_page_split_identify(KeyT *memory_page_split, OffsetT *vertexList, 
								KeyT vertex_count, uint32_t total_memory_pages) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	for (uint32_t i = idx; i < vertex_count; i += blockDim.x * gridDim.x) {
		long int before_pid = (i == 0) ? 10000 : (vertexList[i-1]*sizeof(KeyT)/4000);
		long int  now_pid = vertexList[i]*sizeof(KeyT)/4000;
		if (before_pid != now_pid)
			memory_page_split[now_pid] = i;
	}	
	return;
}

//功能：根据访问频率设置每个页面的访问模式。
//CSRGraph g：存储图的 CSR 表示。
//KeyT *memory_page_split：表示页面划分的边界。
//uint32_t total_memory_pages：页面总数。
//uint8_t *access_mode：用于存储每个页面的访问模式。
//ATT* vertex_access_times：存储每个顶点的访问次数。
//uint64_t avg_access_per_page：每个页面的平均访问数。
//逻辑：使用 warp 的并行计算模型（32个线程）来计算每个页面的总访问数。
//使用 __shfl_down_sync 实现 warp 内的访问次数归约。
//如果页面的访问次数超过一定的阈值（1.1 倍于平均值），则设置页面为统一访问模式。
__global__ void set_access_mode(CSRGraph g, KeyT *memory_page_split, uint32_t total_memory_pages, 
					 uint8_t *access_mode, ATT* vertex_access_times,
					 uint64_t avg_access_per_page) {
	uint32_t warp_id = (threadIdx.x + blockIdx.x*blockDim.x)/32;
	uint32_t total_warp_num = blockDim.x * gridDim.x /32;
	uint32_t lane_id = threadIdx.x % 32;
	for (uint32_t w = warp_id; w < total_memory_pages; w += total_warp_num) {
		uint32_t cur_page_start = memory_page_split[w];
		uint32_t cur_page_end = memory_page_split[w+1];
		uint64_t valid_vertex_num = 0;
		//check page access mode
		for (uint32_t i = cur_page_start + lane_id; i < cur_page_end; i += 32) {
			uint32_t time = vertex_access_times[i];
			uint32_t deg = g.getDegree(i);
			//uint32_t threshold = (cur_page_end - cur_page_start)/20;
			//time = (time > threshold) ? threshold : time;
			//threshold = threshold > 3 ? 3 : threshold;
			//valid_vertex_num += (time > threshold) ? threshold : time;
			valid_vertex_num += (uint64_t)time*deg;
		}
		valid_vertex_num += __shfl_down_sync(0xffffffff, valid_vertex_num, 16);
		valid_vertex_num += __shfl_down_sync(0xffffffff, valid_vertex_num, 8);
		valid_vertex_num += __shfl_down_sync(0xffffffff, valid_vertex_num, 4);
		valid_vertex_num += __shfl_down_sync(0xffffffff, valid_vertex_num, 2);
		valid_vertex_num += __shfl_down_sync(0xffffffff, valid_vertex_num, 1);
		valid_vertex_num = __shfl_sync(0xffffffff, valid_vertex_num, 0);
		uint32_t unified = 0;
		//the judgement of two access mode is to be optimized
		if (valid_vertex_num > 1.1*avg_access_per_page)
			unified = 1;
		if (!unified)
			continue;
		//set access mode bitmaps
		for (uint32_t i = cur_page_start + lane_id; i < cur_page_end; i += 32) {
			access_mode[i] = unified;
		}
	}
	return ;
}
//TODO this is an interesting problem to use GPU for elements occurance counting
//here we use atomic counting 
//instead SORT and REDUCE_BY_KEY in thrust is worthy a try
//可以使用GPU来进行元素计数功能，这里是使用的原子计数法，更好的操作是使用 Thrust 库中的 SORT 和 REDUCE_BY_KEY 来实现这一任务

//功能：累积每个顶点的访问次数。
//ATT *access_times：访问次数数组。
//KeyT *EL_frontier：前沿（即访问的顶点列表）。
//uint32_t frontier_size：前沿大小。
//逻辑：使用 atomicAdd 来避免竞争条件，累积前沿中各顶点的访问次数。

// atomicAdd 是 CUDA 中用于 原子加法操作 的函数。它的作用是：
//原子操作：在并行编程中，多线程可能会同时对某个共享变量进行写操作，这可能导致竞争条件，进而导致结果不正确。
//原子操作确保某个操作在执行期间不会被其他线程打断，从而保证结果的正确性。
__global__ void accumulate_vertex_access_time(ATT *access_times,
											  KeyT *EL_frontier,
											  uint32_t frontier_size) {
	uint32_t idx = threadIdx.x + blockIdx.x*blockDim.x;
	for (uint32_t i = idx; i < frontier_size; i += (blockDim.x * gridDim.x)) {
		atomicAdd(access_times + EL_frontier[i], 1);
	}
	return ;
}

//功能：统计邻接顶点的总访问数。
//CSRGraph g：图的 CSR 表示。
//EmbeddingList emblist：嵌入列表，存储访问的节点。
//uint64_t *warp_reduced_nbr_access：用于存储每个 warp 的访问次数归约结果。
//emb_off_type cur_nbr_size：当前邻接点大小。
//uint8_t level：嵌入层级。
//逻辑：对每个节点的邻接点进行累加。
//使用 warp-level 的 shuffle 操作进行并行归约，得到总访问数。
__global__ void count_total_access_nbr(CSRGraph g, EmbeddingList emblist, uint64_t *warp_reduced_nbr_access, 
									   emb_off_type cur_nbr_size, uint8_t level) {
	uint32_t idx = threadIdx.x + blockIdx.x* blockDim.x;
	uint32_t total_threads = blockDim.x * gridDim.x;
	uint64_t access_nbr_num = 0;
	for (emb_off_type i = idx; i < cur_nbr_size; i += total_threads) {
		KeyT cur_nbr = emblist.get_vid(level, i);
		access_nbr_num += g.getDegree(cur_nbr);
	}
	__syncthreads();
	access_nbr_num += __shfl_down_sync(0xffffffff, access_nbr_num, 16);
	access_nbr_num += __shfl_down_sync(0xffffffff, access_nbr_num, 8);
	access_nbr_num += __shfl_down_sync(0xffffffff, access_nbr_num, 4);
	access_nbr_num += __shfl_down_sync(0xffffffff, access_nbr_num, 2);
	access_nbr_num += __shfl_down_sync(0xffffffff, access_nbr_num, 1);
	if (threadIdx.x % 32 == 0)
		warp_reduced_nbr_access[idx/32] = access_nbr_num;
	return ;
}
//TODO: the vertex of each layer in the embedding list determines the access mode,
//which can be saved and use later to save the recalculation time. But for now,
//We only implement the naive version.
//TODO: we should also consider the time locality of the embeddinglist -- vertices
//in the same layer of embedding list not necessarily are accessed in the same kernel
//We should also consider that later.

//顶点的访问模式可以保存起来，以便以后使用，从而减少重新计算的时间。
//通过保存访问模式（如缓存这些信息），在后续的计算中可以直接复用，避免重复的计算过程，进而提高程序的效率。
//而现在是没有缓存的
//此外，同一层的嵌入列表中的顶点不一定在同一个 CUDA 内核中被访问。
//也就是说，这些顶点可能会被不同的内核（kernels）在不同的时间点访问，这样可能会导致时间局部性无法完全被利用。
//需要考虑到这一点，或许在未来的设计中，利用更多的优化策略来安排内核的执行和数据的存取，以便提高访问效率和局部性。

//主要功能：控制和管理图的访问模式，确定每个内存页面的访问方式（例如统一访问或随机访问）。
//输入：KeyT graph_total_page：存储图数据的总页面数。
//KeyT* mem_page_vertex_border：存储页面的边界。
//uint64_t *frontier_access_neighbors：存储前沿顶点的邻接点访问数
//1.set_vertex_page_border：
//设置顶点的页面边界，确定图中每个页面包含哪些顶点。
//使用 memory_page_split_identify 核函数完成页面划分。
//2.cal_access_mode_by_EL：
//通过前沿列表 EmbeddingList 计算每个页面的访问模式。
//主要步骤：
//重置所有页面的访问模式。
//统计嵌入列表中每个前沿的邻接点总数。
//调用 accumulate_vertex_access_time 核函数统计每个顶点的访问次数。
//调用 set_access_mode 核函数，设置每个页面的访问模式。
//3.clean
//释放所有使用过的 GPU 内存和主机内存。
class access_mode_controller{
private:
	//TODO this is used for LFU-based access mode determination method
	//uint32_t avaiable_pages;
	//uint32_t page_timestamp;
	KeyT graph_total_page;
	KeyT* mem_page_vertex_border;
	uint64_t *frontier_access_neighbors;
public:
	access_mode_controller() {}
	~access_mode_controller() {}
	void set_vertex_page_border(CSRGraph g) {
		KeyT nnodes = g.get_nnodes(); // 获取图中顶点数量
		OffsetT nedges = g.get_nedges(); // 获取图中边数量
		graph_total_page = (sizeof(KeyT)*nedges + 3999)/4000;// 计算总页面数，按照4KB页面大小划分
		// 分配 GPU 内存以存储页面边界信息
		check_cuda_error(cudaMalloc((void **)&mem_page_vertex_border, sizeof(KeyT)*(graph_total_page+1)));
		// 在 GPU 上调用内核函数来标识页面边界
		memory_page_split_identify<<<20000, BLOCK_SIZE>>>(mem_page_vertex_border,
								   						 g.get_row_start_by_mem_controller(),
								   						 nnodes, graph_total_page);
		// 将最后一个页面的顶点边界设为图的节点数
		check_cuda_error(cudaMemcpy(mem_page_vertex_border+graph_total_page,
									&nnodes, sizeof(KeyT), cudaMemcpyHostToDevice));
		// 分配 CPU 内存用于前沿邻居访问计数
		frontier_access_neighbors = (uint64_t *)malloc(sizeof(uint64_t)*embedding_max_length);
		memset(frontier_access_neighbors, 0, sizeof(uint64_t)*embedding_max_length);
		return ;
	}
	void cal_access_mode_by_EL(CSRGraph g, expand_constraint ec, EmbeddingList emblist){
		Clock set_mem_access("access control");
		set_mem_access.start();
		uint8_t* access_mode = g.get_access_mode_by_mem_controller();
		ATT* vertex_access_times;
		KeyT nnodes = g.get_nnodes();
		OffsetT nedges = g.get_nedges();
		 // 初始化 GPU 上的访问模式数组
		check_cuda_error(cudaMemset(access_mode, 0, sizeof(uint8_t)*nnodes)); 
		check_cuda_error(cudaMalloc((void **)&vertex_access_times, sizeof(ATT)*nnodes));
		check_cuda_error(cudaMemset(vertex_access_times, 0, sizeof(ATT)*nnodes));
		
		//get total access neighbors
		//若存在直接加一，不存在则调用库函数
		uint32_t nblocks = 16000;
		uint32_t total_warps = nblocks * BLOCK_SIZE/32;
		uint64_t *warp_reduced_nbr_access;
		check_cuda_error(cudaMalloc((void **)&warp_reduced_nbr_access, sizeof(uint64_t)*total_warps));
		uint64_t total_access_neighbors = 0;
		for (uint32_t i = 0; i < ec.nbr_size; i ++) {
			uint8_t cur_nbr = (ec.nbrs>>(8*i))&0xff;
			if (frontier_access_neighbors[cur_nbr] != 0)
				total_access_neighbors += frontier_access_neighbors[cur_nbr];
			else {
				emb_off_type cur_nbr_size = emblist.size(cur_nbr);
				check_cuda_error(cudaMemset(warp_reduced_nbr_access, 0, sizeof(uint64_t)*total_warps));
				count_total_access_nbr<<<nblocks, BLOCK_SIZE>>>(g, emblist, warp_reduced_nbr_access, cur_nbr_size, cur_nbr);
				check_cuda_error(cudaDeviceSynchronize());
				frontier_access_neighbors[cur_nbr] = thrust::reduce(thrust::device, warp_reduced_nbr_access, warp_reduced_nbr_access + total_warps);
				total_access_neighbors += frontier_access_neighbors[cur_nbr];
				//printf("cur_nbr_size %d\n", cur_nbr_size);
				//printf("total access neighbors %lu\n", total_access_neighbors);
			}
		}
		//计算访问模式：遍历所有的前沿节点，根据邻居关系确定每个顶点的访问次数。
        //调用内核函数 accumulate_vertex_access_time 来计算每个顶点的访问时间，并累加到 vertex_access_times 中
		check_cuda_error(cudaFree(warp_reduced_nbr_access));
		emb_off_type total_emb_size = 0;
		for (uint32_t i = 0; i < ec.nbr_size; i ++) {
			uint8_t query_vertex = (ec.nbrs>>(8*i))&0xff;
			KeyT *cur_EL_frontier = *(emblist.get_vid_list_by_mem_controller()+query_vertex);
			unsigned frontier_size = emblist.size(query_vertex);
			accumulate_vertex_access_time<<<20000, BLOCK_SIZE>>>(vertex_access_times,
																cur_EL_frontier,
																frontier_size);
			check_cuda_error(cudaDeviceSynchronize());	
			total_emb_size += frontier_size;
		}
		//设置访问模式（根据频率和访问次数决定采用哪个模式）
		//log_info(set_mem_access.count("total access time %lu, total ftr size %d, and %d nbr access per page", total_access_neighbors, total_emb_size, total_access_neighbors/graph_total_page));
		set_access_mode<<<20000, BLOCK_SIZE>>>(g, mem_page_vertex_border,graph_total_page,
											  access_mode,vertex_access_times,
											  total_access_neighbors/graph_total_page);
		check_cuda_error(cudaDeviceSynchronize());
		log_info(set_mem_access.count("end access control"));
		//check the propotion of unified memory access;
		/*KeyT *mem_vertex_border = new KeyT [graph_total_page+1];
		check_cuda_error(cudaMemcpy(mem_vertex_border, mem_page_vertex_border, sizeof(KeyT)*(graph_total_page+1), cudaMemcpyDeviceToHost));
		uint8_t *access_mode_h = new uint8_t [nnodes];
		check_cuda_error(cudaMemcpy(access_mode_h, access_mode, sizeof(uint8_t)*nnodes, cudaMemcpyDeviceToHost));
		uint32_t unified_page_num = 0;
		for (uint32_t i = 0; i < graph_total_page; i++)
			if (access_mode_h[mem_vertex_border[i]] == 1)
				unified_page_num ++;
		log_info("the %d of %d pages use unified memory access", unified_page_num, graph_total_page);
		ofstream file;
		char buffer [10];
		sprintf(buffer, "%d", emblist.level());
		file.open(buffer);
		file << "this is the "<< emblist.level() << "th level\n";
		ATT *vertex_access_times_h = new ATT [nnodes];
		check_cuda_error(cudaMemcpy(vertex_access_times_h, vertex_access_times, sizeof(ATT)*nnodes, cudaMemcpyDeviceToHost)); 
		for (int i = 0; i < graph_total_page; i ++) {
			file << "page " << i << "'s access mode " << (uint32_t)access_mode_h[mem_vertex_border[i]] << "\n";
			uint32_t total_time = 0;
			for (int j = mem_vertex_border[i]; j < mem_vertex_border[i+1]; j ++) {
					total_time += vertex_access_times_h[j];
					file << (uint32_t)vertex_access_times_h[j] << "\n";
			}
			file << "total page " << mem_vertex_border[i+1] - mem_vertex_border[i] <<
					" total access time " << total_time << "\n";
		}
		file.close();
		log_info(set_mem_access.count("end access mem control check"));
		delete [] mem_vertex_border;
		delete [] access_mode_h;
		delete [] vertex_access_times_h;*/
		//用于检查统一内存（Unified Memory）访问的比例
		check_cuda_error(cudaFree(vertex_access_times));
	}
	void clean() {
		check_cuda_error(cudaFree(mem_page_vertex_border));
		free(frontier_access_neighbors);
		return ;
	}
};
