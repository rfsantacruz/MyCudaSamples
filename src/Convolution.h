/*
 * Convolution.h
 *
 *  Created on: Mar 30, 2015
 *      Author: rfsantacruz
 */

#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_

#define MASK_WIDTH 5

void conv1DHost(const float* input, const int input_width, const float* mask, float* output);

void conv2DHost(const float* input, const int input_width, const float* mask, float* output);

#endif /* CONVOLUTION_H_ */
