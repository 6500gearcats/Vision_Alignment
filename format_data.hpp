//
//  model.hpp
//  AI_Backbone
//
//  Created by maxwell on 1/5/2020.
//  Copyright © 2019 organized-organization. All rights reserved.
//

#include <iostream>
#include <vector>
#include <cstdlib>
#include <fstream>

#ifndef format_data_hpp
#define format_data_hpp

#define NUM_COORIDINATES 8
#define COORIDINATES_DIMENTIONALITY 2
#define MAX_PIXEL_LEN 155

// network tables ntcore library

/**
 * Data should be normalized before inputed into the neural network
 *      i.e. [0, 1] or [-1, 1]
 * The neural network will run twice(like a batch size of 2) once with x values, once with y values, that is epoch 1
 * @param location of file
 * @return vector of type vector of type T
 * */

// inline std::vector<float> read_file(std::string file_name, uint32_t X_or_Y_data)
// {
//     std::ifstream in;
//     in.open(file_name);
//     std::vector<float> rVal;
    
//     uint32_t i = 0;
//     if(X_or_Y_data == 1)
//     {
//         while(i < COORIDINATES_DIMENTIONALITY * NUM_COORIDINATES)
//         {
//             rVal.push_back((in << i) / MAX_PIXEL_LEN);
//             i += 2;
//         }
//         return rVal;
//     }else{
//         while(i < COORIDINATES_DIMENTIONALITY * NUM_COORIDINATES)
//         {
//             rVal.push_back((in << (i+1)) / MAX_PIXEL_LEN);
//             i += 2;
//         }
//         return rVal;
//     }
// }

#endif