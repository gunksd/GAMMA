#include<iostream>
#include<stdlib.h>
#include<vector>
#include "utils.h"
#include<fstream>
#include<string.h>
using namespace std;
class queryGraph{
public:
	uint32_t vertex_num;//顶点数量
	vector<uint8_t> vertex_label;//标签
	vector<vector<uint32_t>> adj_list;//邻接列表，存储每个顶点的邻居。
	vector<vector<uint32_t>> order_list;//顺序列表，存储顶点的有序关系
	vector<uint32_t> core;//核心顶点的索引列表（度数大于 1 的顶点）
	vector<uint32_t> satellite;//卫星顶点的索引列表（度数为 1 的顶点）。
	void readFromFile(const char filename []) {
		std::ifstream file_read;
		file_read.open(filename, std::ios::in);
		file_read >> vertex_num;
		//cout << vertex_num << endl;
		for (uint32_t i = 0; i < vertex_num; i ++) {
			uint32_t label_now;
			file_read >> label_now;
			//cout << label_now << " ";
			vertex_label.push_back((uint8_t)label_now);
		}
		//cout << endl;
		for (uint32_t i = 0; i < vertex_num; i ++) {
			vector<uint32_t> v;
			uint32_t nbr_size;
			file_read >> nbr_size;
			//cout << nbr_size << " ";
			uint32_t nbr_now;
			for (uint32_t j = 0; j < nbr_size; j ++) {
				file_read >> nbr_now;
				//cout << nbr_now << " ";
				v.push_back(nbr_now);
			}
			adj_list.push_back(v);
			//cout << endl;
		}
		for (uint32_t i = 0; i < vertex_num; i ++) {
			vector<uint32_t> v;
			uint32_t nbr_size;
			file_read >> nbr_size;
			//cout << nbr_size << " ";
			uint32_t order_nbr_now;
			for (uint32_t j = 0; j < nbr_size; j ++) {
				file_read >> order_nbr_now;
				//cout << order_nbr_now << " ";
				v.push_back(order_nbr_now);
			}
			order_list.push_back(v);
			//cout << endl;
		}
		file_read.close();
		for (uint32_t i = 0; i < vertex_num; i ++) {
			if (adj_list[i].size() > 1) {
				core.push_back(i);
			}
			else {
				satellite.push_back(i);
			}
		}
		return ;
	}
	void clear() {
		for (uint32_t i = 0; i < vertex_num; i ++) {
			adj_list[i].clear();
			order_list[i].clear();
		}
		adj_list.clear();
		order_list.clear();
		core.clear();
		satellite.clear();
		return ;
	}
};
//定义了一个 queryGraph 类
//用于读取图数据并进行一些处理操作，主要是从文件读取图的顶点信息和邻接关系
//然后将顶点分为 "核心" 和 "卫星" 两类。
//将每个顶点按照其邻居的数量分为 "核心顶点" 和 "卫星顶点"。
//核心顶点的度数（邻居数）大于 1，而卫星顶点的度数为 1。
//核心顶点存在 core 中，卫星顶点保存在 satellite 中。