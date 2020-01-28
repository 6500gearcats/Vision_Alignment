#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "optimize_trained_net.hpp"
#include "layer.hpp"

Network :: Network(std::vector<uint32_t> lSize_arr, std::vector<uint32_t> act_func_arr)
{
    this->lSize_arr = lSize_arr;
    this->act_func_arr = act_func_arr;
}

void Network :: get_data(std::string file)
{
    std::ifstream inFile;
    inFile.open(file);
    if (!inFile) {
        std::cout << "Unable to open file datafile.txt";
        exit(1);   // call system to stop
    }

    this->weight_matrix = (Matrix **)calloc(this->lSize_arr.size() - 1, sizeof(Matrix *));
    float push_value;
    std::string float_disposable = "";
    std::string iteration;
    for(int n = 0; n < this->lSize_arr.size() - 1; ++n)
    {
        this->weight_matrix[n] = new Matrix(lSize_arr[n], lSize_arr[n+1]);
        for(int i = 0; i < this->lSize_arr[n]; ++i)
        {
            for(int j = 0; j < this->lSize_arr[n+1]; ++n)
            {
                for(int k = 0; k < 6; ++k){
                    inFile >> iteration;

                    if(iteration != " "){
                        float_disposable += iteration;
                        iteration = "";
                    }else{
                        iteration = "";
                        break;
                    }
                }
                push_value = std::stof(float_disposable);
                this->weight_matrix[n]->push(push_value);
            }
        }
    }

    for(int n = 0; n < this->lSize_arr.size() - 1; ++n)
    {
        for(int i = 0; i < this->lSize_arr[n]; ++i)
        {
            for(int j = 0; j < this->lSize_arr[n+1]; ++n)
            {
                std::cout << this->weight_matrix[n]->get_value(i, j);
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void Network :: run()
{

}