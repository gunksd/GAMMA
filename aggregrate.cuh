#ifndef AGGREGRATE_H
#define AGGREGRATE_H
#include "graph.cuh"
#include "embedding.cuh"

//功能：将嵌入 (emb) 转换为一个唯一的模式 ID (patternID)
//计算嵌入中各个顶点的编码，并最终形成一个用于识别模式的 ID：
//顶点编码：为每个嵌入中的顶点创建一个唯一编码。编码包含多个信息：
//1.顶点的顺序。
//2.顶点的度（即与之相连的边的数量）。
//3.顶点的标签。
//迭代编码：通过迭代的方式，结合顶点的连接关系逐步优化编码。
//生成模式 ID：根据顶点编码，生成一个模式 ID，其中包含模式的邻接关系和顶点标签

__device__ patternID emb2pattern_id(KeyT *emb, label_type *einfo, uint32_t *vlabel, int len, CSRGraph g) {
	//Here is how we use vlabel to keep all charactors we use
	//for dulicated vids, their 32-bit wide data is used as : 0xf(real data pos)fffffff(all set to 1)
	//for real data, 32-bit data is used as: 0xf(final vertex order) ffff (vertex encoding) f (degree) ff(label)
	//"emb" is only used for vertex duplicate check, after that, it is used for vertex encoding
	//memset(vlabel, 0, sizeof(uint32_t)*len);
	//TODO bit usage has not been changed after we enlarge the bit number of patternID

	//vlabel的使用说明：用于保存我们需要的所有字符或者特征。
	//是数据结构 emb 在不同阶段有不同的功能：一开始用于验证数据的唯一性，后来用于对顶点进行标识和编码。
	//如何为重复的顶点编写ID：32 位整数来存储顶点的数据，前四位设置成0xf，表示实际数据位置，后面的 28 位全设置为 1（即 0xfffffff）
	//vlabel会重新置为零，以此来防止垃圾数据
	//可能会根据需要重新调优
	vlabel[0] = 0;
	//count the number of distinct vertex
	int distinct_v = 1;
	//int e1 = emb[0], e2 = emb[1];
	for (int i = 1; i < len; i ++) {
		vlabel[i] = 0;
		for (int j = 0; j < i; j ++) {//find the first same vid
			if (emb[i] == emb[j]) {
				vlabel[i] |= 0xfffffff;
				vlabel[i] |= (j<<28);
				break;
			} 
		}
		if ((vlabel[i] &0xfffffff) != 0xfffffff)
			distinct_v ++;
	}
	//NOTE: the valid number of all vertex is not consecutive
	//encoding vertices
		//label collection and degree counting (label_num < 256)
	for (int i = 0; i < len; i ++) {
		if ((vlabel[i]&0xfffffff) == 0xfffffff)
			continue;
		vlabel[i] = g.getData(emb[i]);
	}
	for (int i = 1; i < len; i ++) {
		int real_dst_pos = (vlabel[i]&0xfffffff) == 0xfffffff ? (vlabel[i] >> 28) : i;
		int real_src_pos = (vlabel[einfo[i]]&0xfffffff) == 0xfffffff ? (vlabel[einfo[i]] >> 28): einfo[i];
		vlabel[real_src_pos] += (1<<8);
		vlabel[real_dst_pos] += (1<<8);
	}
		//set initial encoding
	for (int i = 0; i < len; i ++) {
		if ((vlabel[i] &0xfffffff) == 0xfffffff)
			continue;
		uint32_t value =((vlabel[i]&0xff)+1)*(((vlabel[i]>>8)&0xf)+1);
		vlabel[i] += (value<<12);
	}
		//iterativly encoding
	uint32_t buffer[embedding_max_length];
	uint32_t max_it = len-1 > 3 ? 3 : len - 1;
	for (int it = 0; it < max_it; it ++) {
		memcpy(buffer, vlabel, sizeof(uint32_t)*len);
		for (int i = 1; i < len; i ++) {
			int real_dst_pos = (vlabel[i]&0xfffffff) == 0xfffffff ? (vlabel[i] >> 28) : i;
			int real_src_pos = (vlabel[einfo[i]]&0xfffffff) == 0xfffffff ? (vlabel[einfo[i]] >> 28): einfo[i];

			int encoding = (buffer[real_dst_pos] >> 12) &0xffff;
			int deg = (buffer[real_dst_pos] >> 8)&0xf;
			vlabel[real_src_pos] += ((encoding/deg)<<12)/(it+2);
			encoding = (buffer[real_src_pos] >> 12) &0xffff;
			deg = (buffer[real_src_pos] >> 8)&0xf;
			vlabel[real_dst_pos] += ((encoding/deg)<<12)/(it+2);
		}
	}
	//calculate vertex order by vertex encoding
	int vertex_order = 0xf;
	for (int i = 0; i < len; i ++) {
		if ((vlabel[i]&0xfffffff) == 0xfffffff)
			continue;
		vlabel[i] += (vertex_order << 28);
	}
	for (uint32_t i = 0; i < distinct_v; i ++) {
		uint32_t max_encoding = 0;
		uint32_t max_encoding_pos = 0;
		for (int j = 0; j < len; j ++) {
			if ((vlabel[j]&0xfffffff) == 0xfffffff)
				continue;
			if (((vlabel[j]>>28)&0xf) != 0xf)
				continue;
			uint32_t my_encoding = (vlabel[j]&0xfffffff);
			if (my_encoding > max_encoding) {
				max_encoding_pos = j;
				max_encoding = my_encoding;
			}
		}
		vlabel[max_encoding_pos] = (i << 28) | (vlabel[max_encoding_pos] &0xfffffff);
	}
	/*int f1 = (vlabel[0]>>28)&0xf > (vlabel[1]>>28)&0xf ? 1: -1;
	int f2 = (g.getData(emb[0]) < g.getData(emb[1])) ? 1 : -1;
	if (f1*f2 == -1) {
		printf("%x %x %x %x %x %x\n",(vlabel[0]>>28)&0xf, (vlabel[1]>>28)&0xf, g.getData(emb[0]), g.getData(emb[1]), vlabel[0]&0xfffffff, vlabel[1]&0xfffffff) ;
	}*/
	//generate the pattern_id
	//64bit pattern id : 7+6+5+4+3+2+1 + 4X8, so the maximum embedding is 7, and max label num is 15 by this
	patternID pattern_id(0, 0);
	//pattern_id.lab = (uint64_t)(g.getData(emb[0]) + (g.getData(emb[1])<<8));
	//return pattern_id;
	for (int i = 0; i < len; i ++) {
		int real_dst_pos = (vlabel[i]&0xfffffff) == 0xfffffff ? ((vlabel[i] >> 28)&0xf): i;
		//set label
		if (real_dst_pos == i) {
			int label = g.getData(emb[i]), _off = ((vlabel[i] >>28)&0xf)*8;
			pattern_id.lab += (label << _off);
		}
		if (i == 0)
			continue;
		int real_src_pos = (vlabel[einfo[i]]&0xfffffff) == 0xfffffff ? vlabel[einfo[i]] >> 28: einfo[i];
		int src_v_order = (vlabel[real_src_pos] >> 28)& 0xf;
		int dst_v_order = (vlabel[real_dst_pos] >> 28)& 0xf;
		if (src_v_order > dst_v_order) {
			int t = src_v_order; src_v_order = dst_v_order; dst_v_order = t;
		}
		int off = (7+8-src_v_order)*src_v_order/2 + (dst_v_order-src_v_order-1);
		pattern_id.nbr = pattern_id.nbr | (1 << off);
		//if (g.getData(emb[0]) == 0 && g.getData(emb[1]) == 1) 
		//	printf("the labels is %lx\n", pattern_id.lab);
		//if (len == 2 && g.getData(emb[0]) == 0)
		//	printf("%d %d label : 0 %d\n", emb[0], emb[1], g.getData(emb[1]));
	}
	return pattern_id;
}

//功能：对嵌入列表中的每个嵌入调用 emb2pattern_id，将嵌入映射到唯一的模式 ID。
//实现细节：
//使用共享内存存储嵌入和相关信息 (local_emb, local_einfo, vlabel)，减少访存延迟。
//通过 for 循环并行处理每个嵌入列表中的嵌入。
//根据嵌入的有效性生成模式 ID，并存储在 pids 数组中。
__global__ void map_emb2pid(CSRGraph g, EmbeddingList emblist, patternID *pids, 
				            int level, emb_off_type base_off, uint32_t emb_size) {
		__shared__ KeyT local_emb[BLOCK_SIZE][embedding_max_length];
		__shared__ label_type local_einfo[BLOCK_SIZE][embedding_max_length];
		__shared__ uint32_t vlabel[BLOCK_SIZE][embedding_max_length];
		int idx = threadIdx.x + blockDim.x*blockIdx.x;
		int tid = threadIdx.x;
		for (int i = idx; i < emb_size; i += (blockDim.x*gridDim.x)) {
			emblist.get_edge_embedding(level, base_off+i, local_emb[tid], local_einfo[tid]);
			//if (i < 30) 
			//	printf("%d th emb, %d %d %d %d\n", i, local_emb[tid][0], local_emb[tid][1], g.getData(local_emb[tid][0]), g.getData(local_emb[tid][1]));
			bool valid_emb = true;
			for (uint32_t j = 0; j <= level; j ++) {
				if(local_emb[tid][j] == 0xffffffff) {
					valid_emb = false;
					break;
				}
			}
			patternID pattern_id(-1, (uint64_t)-1);
			if (valid_emb) {
				pattern_id = emb2pattern_id(local_emb[tid], local_einfo[tid], vlabel[tid], level+1, g);
			}
			pids[base_off+i] = pattern_id;
		}
		return ;
}

//功能：控制嵌入到模式 ID 的映射过程，使用 CUDA 核函数 map_emb2pid 来完成该任务。
//将嵌入列表按批次处理，减少 GPU 资源的占用。
//调用 map_emb2pid 核函数，将所有嵌入映射为模式 ID

void map_embeddings_to_pids(CSRGraph g, EmbeddingList emb_list, patternID *pattern_ids, int level) {
		uint64_t emb_nums = emb_list.size(level);
		uint32_t batch_num = (emb_nums+expand_batch_size-1)/expand_batch_size;
	check_cuda_error(cudaDeviceSynchronize());
	//printf("the batch num of map embedding to pids is %d\n", batch_num);
	for (uint32_t i = 0; i < batch_num; i++) {
		emb_off_type base_off = (uint64_t)i*expand_batch_size;
		uint32_t cur_size = emb_nums - base_off;
		cur_size = cur_size > expand_batch_size ? expand_batch_size : cur_size;
		uint32_t num_blocks = 10000;
		map_emb2pid<<<num_blocks, BLOCK_SIZE>>>(g, emb_list, pattern_ids, level, base_off, cur_size);
		check_cuda_error(cudaDeviceSynchronize());
		//if (i%10 == 0) log_info("the %dth batch of embedding is done", i+1);
	}
	log_info("map_embedding_to_pids is done");
	//emb_list.display(level, emb_nums);	
	//code validation
	/*patternID *h_pids = new patternID [emb_nums];
	check_cuda_error(cudaMemcpy(h_pids, pattern_ids, sizeof(patternID)*emb_nums, cudaMemcpyDeviceToHost));
	for (int p = 0; p < emb_nums; p ++)
		printf("%d %d\n", h_pids[p].lab&0xff, (h_pids[p].lab>>8)&0xff);
	delete [] h_pids;*/
	//emb_list.check_all(level, emb_nums, pattern_ids, g);
	//sample mapped pattern
	/*int sample_size = 40;
	uint64_t *patterns_h = new uint64_t [sample_size];
	for (int i = 0; i < sample_size; i ++) {
		check_cuda_error(cudaMemcpy(patterns_h+i, pattern_ids+emb_nums/sample_size*i, sizeof(uint64_t), cudaMemcpyDeviceToHost));
		printf("%lx\n", patterns_h[i]);
	}
	delete [] patterns_h;*/
	return ;
}

//统计频繁出现的模式，并将其标记为频繁模式。
//使用共享内存和 __shfl_down_sync 来并行归约每个线程的计数结果。
//如果某个模式的出现次数达到指定阈值 (threshold)，将其标记为频繁模式。

__global__ void count_frequent_pattern(patternID *pattern_ids, emb_off_type pid_size, int threshold, 
									   uint32_t *fre_pattern_num, uint8_t *stencil) {
	uint32_t tid = threadIdx.x + blockDim.x*blockIdx.x;
	int local_fre_pattern = 0;
	for (emb_off_type i = tid; i <= pid_size-threshold; i += (blockDim.x*gridDim.x)) {
		if (i == 0 || !(pattern_ids[i] == pattern_ids[i-1])) {
			if (pattern_ids[i] == pattern_ids[i+threshold-1]) {
				stencil[i] = 1;
				local_fre_pattern ++;
			}
		}
	}
	local_fre_pattern += __shfl_down_sync(0xffffffff, local_fre_pattern, 16);
	local_fre_pattern += __shfl_down_sync(0xffffffff, local_fre_pattern, 8);
	local_fre_pattern += __shfl_down_sync(0xffffffff, local_fre_pattern, 4);
	local_fre_pattern += __shfl_down_sync(0xffffffff, local_fre_pattern, 2);
	local_fre_pattern += __shfl_down_sync(0xffffffff, local_fre_pattern, 1);
	if (threadIdx.x%32 == 0) {
		fre_pattern_num[tid/32] = local_fre_pattern;
	}
	return ;
}

//验证嵌入是否属于频繁模式，并标记有效嵌入。
//通过调用 emb2pattern_id 为嵌入生成模式 ID。
//使用二分查找 (binarySearch) 判断生成的模式 ID 是否是频繁模式。
//对符合条件的嵌入标记为有效嵌入。

__global__ void emb_validation_check(EmbeddingList emb_list, emb_off_type emb_size, patternID *fre_patterns,
									 uint32_t fre_pattern_num, uint8_t *valid_embs, int level, 
									 emb_off_type base_off, uint32_t *counter, CSRGraph g) {
	__shared__ KeyT local_emb[BLOCK_SIZE][embedding_max_length];
	__shared__ label_type local_einfo[BLOCK_SIZE][embedding_max_length];
	__shared__ uint32_t vlabel[BLOCK_SIZE][embedding_max_length];
	int thread_id = threadIdx.x+blockIdx.x*blockDim.x;
	int idx = threadIdx.x;
	uint32_t local_count = 0;
	for (uint32_t _i = thread_id; _i < emb_size; _i += (blockDim.x*gridDim.x)) {
		emb_list.get_edge_embedding(level, base_off+_i, local_emb[idx], local_einfo[idx]);
		bool valid_emb = true;
		for (uint32_t j = 0; j <= level; j ++) {
			if(local_emb[idx][j] == 0xffffffff) {
				valid_emb = false;
				break;
			}
		}
		if (valid_emb) {
			patternID pattern_id = emb2pattern_id(local_emb[idx], local_einfo[idx], vlabel[idx], level+1, g);
			if (binarySearch<patternID>(fre_patterns, fre_pattern_num, pattern_id) != -1) {
				valid_embs[ _i + base_off] = 1;
				local_count ++;
			}
		}
	}
	local_count += __shfl_down_sync(0xffffffff, local_count, 16);
	local_count += __shfl_down_sync(0xffffffff, local_count, 8);
	local_count += __shfl_down_sync(0xffffffff, local_count, 4);
	local_count += __shfl_down_sync(0xffffffff, local_count, 2);
	local_count += __shfl_down_sync(0xffffffff, local_count, 1);
	if (threadIdx.x%32 == 0)
		counter[thread_id/32] = local_count;
	return ;

}
struct cmp_pid {
	__device__ __host__ bool operator() (const patternID& p1, const patternID& p2) {
		return p1.nbr < p2.nbr || (p1.nbr == p2.nbr && p1.lab < p2.lab);
	}
};

//设置边模式的频繁模式标志，用于快速判断哪些边模式是频繁的。
//遍历嵌入列表中的每对顶点，并将其标签组合标记为频繁模式。
//使用原子操作 (atomicOr) 设置频繁边模式标志，以避免数据竞争。

__global__ void set_freq_edge_pattern(EmbeddingList emb_list, emb_off_type emb_size, uint32_t l, uint32_t *freq_edge_patterns, CSRGraph g) {
	__shared__ KeyT sh_emb[BLOCK_SIZE][embedding_max_length];
	int thread_id = threadIdx.x+ blockDim.x*blockIdx.x;
	KeyT *local_emb = sh_emb[threadIdx.x];
	for (emb_off_type i = thread_id; i < emb_size; i += (blockDim.x*gridDim.x)) {
		emb_list.get_embedding(l, i, local_emb);
		if (local_emb[0] == 0xffffffff || local_emb[1] == 0xffffffff)
			continue;
		uint32_t src_label = g.getData(local_emb[0]), dst_label = g.getData(local_emb[1]);
		int multiple = (src_label * max_label) + dst_label;
		atomicOr(freq_edge_patterns + multiple/32, 1<<(multiple%32));
		multiple = (dst_label * max_label) + src_label;
		atomicOr(freq_edge_patterns + multiple/32, 1<<(multiple%32));
	}
	return ;
}

//关键函数：
//功能：执行嵌入和模式的过滤和聚合
//包括排序嵌入模式 ID、统计频繁模式、过滤出有效嵌入等操作。

void aggregrate_and_filter(CSRGraph g, EmbeddingList emb_list, patternID *pattern_ids, int level, int threshold, uint32_t *freq_edge_patterns) {
	//sort all pattern_ids
	//WARNING: this may cause all embedding list thrash bettween cpu and gpu, but that's affordable
	//使用 thrust::sort 对嵌入的模式 ID 进行排序，便于后续的频繁模式统计。
	emb_off_type pid_size = emb_list.size(level);
	log_info("start sort embedding ids... ...");
	thrust::sort(thrust::device, pattern_ids, pattern_ids + pid_size, cmp_pid());//TODO: out of memory?
	log_info("sort all embedding ids done");	
	//filter out all pattern ids whose support satisfy the threshold
	uint32_t block_num = 10000;
	uint32_t *fre_pattern_num;
	uint8_t *stencil;
	check_cuda_error(cudaMalloc((void **)&stencil, pid_size*sizeof(uint8_t)));
	check_cuda_error(cudaMemset(stencil, 0, sizeof(uint8_t)*pid_size));
	check_cuda_error(cudaMalloc((void **)&fre_pattern_num, BLOCK_SIZE/32*sizeof(uint32_t)*block_num));
	check_cuda_error(cudaMemset(fre_pattern_num, 0, BLOCK_SIZE/32*sizeof(uint32_t)*block_num));
	//TODO here we assume all pattern_ids and stential can be put on the device, and no batch process
	//使用 count_frequent_pattern 核函数统计满足阈值的频繁模式。
	//使用 thrust::reduce 计算频繁模式的数量。
	count_frequent_pattern<<<block_num, BLOCK_SIZE>>>(pattern_ids, pid_size, threshold, fre_pattern_num, stencil);
	check_cuda_error(cudaDeviceSynchronize());
	uint32_t total_fre_pattern = thrust::reduce(thrust::device, fre_pattern_num, fre_pattern_num + BLOCK_SIZE/32*block_num);//the number of valid patterns
	log_info("count frequent patterns done, total frequent pattern num is %d", total_fre_pattern);
	check_cuda_error(cudaFree(fre_pattern_num));
	patternID *fre_patterns;//frequent patterns
	check_cuda_error(cudaMalloc((void **)&fre_patterns, sizeof(patternID)*total_fre_pattern));
	thrust::copy_if(thrust::device, pattern_ids, pattern_ids + pid_size, stencil, fre_patterns, is_valid());
	check_cuda_error(cudaFree(stencil));
	check_cuda_error(cudaFree(pattern_ids));
	log_info("frequent pattern collection done");
	//filter out all embeddings whose pattern ids satisfy the threshold
	uint8_t *valid_emb;
	check_cuda_error(cudaMalloc((void **)&valid_emb, sizeof(uint8_t)*pid_size));
	check_cuda_error(cudaMemset(valid_emb, 0, sizeof(uint8_t)*pid_size));
	uint32_t batch_num = (pid_size + expand_batch_size -1)/expand_batch_size;
	uint32_t valid_emb_num = 0;
	uint32_t *d_counter;
	check_cuda_error(cudaMalloc((void **)&d_counter, BLOCK_SIZE/32*block_num*sizeof(uint32_t)));
	for (int i = 0; i < batch_num; i ++) {
		check_cuda_error(cudaMemset(d_counter, 0, BLOCK_SIZE/32*block_num*sizeof(uint32_t)));
		emb_off_type base_off = (emb_off_type)i*expand_batch_size;
		uint32_t cur_size = pid_size - base_off;
		cur_size = cur_size > expand_batch_size ? expand_batch_size : cur_size;
		//使用 emb_validation_check 核函数验证每个嵌入是否属于频繁模式。
		emb_validation_check<<<block_num, BLOCK_SIZE>>>(emb_list, cur_size, fre_patterns, total_fre_pattern, valid_emb, level, base_off, d_counter, g);
		check_cuda_error(cudaDeviceSynchronize());
		valid_emb_num += thrust::reduce(thrust::device, d_counter, d_counter+BLOCK_SIZE/32*block_num);
	}
	log_info("embedding validation check done, and valid emb num for now is %d",valid_emb_num);
	//将验证后的有效嵌入保留，丢弃无效嵌入，从而压缩嵌入列表。
	check_cuda_error(cudaFree(d_counter));
	//embedding list compaction
	emb_list.compaction(level, valid_emb, valid_emb_num);
	check_cuda_error(cudaFree(valid_emb));	
	if (level == 1) {//here we set frequent edge pattern flags
		check_cuda_error(cudaMemset(freq_edge_patterns, 0, sizeof(uint32_t)*max_label*max_label/32));
		set_freq_edge_pattern<<<block_num, BLOCK_SIZE>>>(emb_list, emb_list.size(), level, freq_edge_patterns, g);
		check_cuda_error(cudaDeviceSynchronize());
	}
	log_info("embedding list compaction done");
}

#endif

//1.用编码的方式将嵌入转化为唯一的模式ID
//2.统计模式ID的频繁程度
//3.对嵌入列表进行过滤和压缩，仅保留符合频繁模式的嵌入。
//4.通过 CUDA 的块（block）和线程（thread）并行机制，加速了模式 ID 计算、频繁模式统计等过程。
//例如，__shfl_down_sync 用于线程间的高效数据交换和归约，减少了线程间的同步开销。