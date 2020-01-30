#include <vector>
//#include <string>
#include <fstream>
#include <iostream>
#include "optimize_trained_net.hpp"
#include "layer.hpp"
#include "activation_data.hpp"

Network :: Network(std::vector<uint32_t> lSize_arr, std::vector<std::string> act_func_arr, std::vector<bool> bias_arr)
{
    this->lSize_arr = lSize_arr;
    this->act_func_arr = act_func_arr;
    this->bias_arr = bias_arr;
}

void Network :: get_data(std::string file)
{
    std::ifstream inFile;
    inFile.open(file);
    if (!inFile) {
        std::cout << "Unable to open file datafile.txt";
        exit(1);   // call system to stop
    }

    this->weight_matrix = (float ***)calloc(this->lSize_arr.size() - 1, sizeof(float **));
    if(bias_arr.size() != 0)
        this->bias_matrix = (float **)calloc(this->lSize_arr.size() - 1, sizeof(float *));
    std::string iterater;

    for(int i = 0; i < this->lSize_arr.size() - 1; ++i){
        this->weight_matrix[i] = new float*[lSize_arr[i]];
        if(bias_arr.size() != 0)
            this->bias_matrix[i] = new float[lSize_arr[i+1]];
        for(int j = 0; j < this->lSize_arr[i]; ++j){
            this->weight_matrix[i][j] = new float[lSize_arr[i+1]];
        }
    }

    for(int n = 0; n < lSize_arr.size() - 1; ++n)
    {
        for(int i = 0; i < lSize_arr[n]; ++i)
        {
            for(int j = 0; j < lSize_arr[n+1]; ++j)
            {
                inFile >> iterater;
                weight_matrix[n][i][j] = stof(iterater);
            }
        }
    }
    if(bias_arr.size() != 0){
        for(int n = 0; n < lSize_arr.size() - 1; ++n)
        {
            for(int i = 0; i < lSize_arr[n+1]; ++i)
            {
                inFile >> iterater;
                bias_matrix[n][i] = stof(iterater);
            }
        }
    }
    
}

void Network :: toString()
{
    for(int n = 0; n < this->lSize_arr.size() - 1; ++n)
    {
        for(int i = 0; i < this->lSize_arr[n]; ++i)
        {
            for(int j = 0; j < this->lSize_arr[n+1]; ++j)
            {
                std::cout << this->weight_matrix[n][i][j];
            }
            std::cout << std::endl;
        }
    }
    std::cout << "\n\n";
}

void Network :: init_input_data(float **input_data, uint32_t num_sets, uint32_t set_size)
{
    this->input_data = new float*[num_sets];
    for(int i = 0; i < num_sets; ++i)
        this->input_data[i] = new float[set_size];
    
    this->input_data = input_data;
}

void Network :: init_net()
{
    this->pseudo_net = (float **)calloc(this->lSize_arr.size(), sizeof(float *));
    for(int i = 0; i < this->lSize_arr.size(); ++i){
        this->pseudo_net[i] = (float *)calloc(this->lSize_arr[i], sizeof(float));
        for(int j = 0; j < lSize_arr[i]; ++j){
            this->pseudo_net[i][j] = 0.0;
        }
    }
}

uint32_t Network :: get_output(uint32_t data_idx)
{
    uint32_t idx = 0;
    float disposable = 0;
    //this->toString();
    for(int i = 0; i < this->lSize_arr[0]; ++i){
        this->pseudo_net[0][i] = this->input_data[data_idx][i];
    }
    for(int n = 0; n < this->lSize_arr.size() - 1; ++n)
    {
        //Activates the neurons in n layer
        for(int i = 0; i < lSize_arr[n]; ++i){
            this->pseudo_net[n][i] = __activate_value__(this->pseudo_net[n][i], this->act_func_arr[n]);
        }

        for(int i = 0; i < lSize_arr[n+1]; ++i)
        {
            for(int j = 0; j < lSize_arr[n]; ++j)
            {
                disposable += (this->pseudo_net[n][j] * this->weight_matrix[n][j][i]);
            }
            if(this->bias_arr[n+1])
                this->pseudo_net[n+1][i] = disposable + this->bias_matrix[n][i];
            else
                this->pseudo_net[n+1][i] = disposable + this->bias_matrix[n][i];
            disposable=0;
        }
    }
    for(int n = 0; n < lSize_arr[lSize_arr.size() - 1]; ++n){
        this->pseudo_net[lSize_arr.size() - 1][n] = __activate_value__(this->pseudo_net[lSize_arr.size() - 1][n], this->act_func_arr[lSize_arr.size() - 1]);
    }
    
    for(int i = 0; i < lSize_arr[lSize_arr.size() - 1] - 1; ++i){
        if(this->pseudo_net[lSize_arr.size() - 1][i] < this->pseudo_net[lSize_arr.size() - 1][i+1])
            idx = i + 1;
    }

    std::cout << this->pseudo_net[4][0] << "  " << this->pseudo_net[4][1] << "\n";
    return idx;
}
