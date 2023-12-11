#include "cuda_runtime.h"
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>
#include <iostream>

__global__ void blurImg(unsigned char *d_in, unsigned char *d_out, int w, int h, int ch, int blur, double sum){
    for(unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; j < w; j += gridDim.x * blockDim.x){
        for(unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; i < h; i += gridDim.y * blockDim.y){
            double sumr = 0, sumg = 0, sumb = 0;
            for(int ki = -blur; ki <= blur; ki++){
                for(int kj = -blur; kj <= blur; kj++){
                    int idx = (i + ki) * w * ch + (j + kj) * ch;
                    if(idx >= 0 && idx < w * h * ch){
                        double factor = pow(2, blur * 2);
                        int diff = abs(ki)+abs(kj);
                        factor /= pow(2, diff);
                        factor /= sum;
                        sumr += (double)d_in[idx] * factor;
                        sumg += (double)d_in[idx + 1] * factor;
                        sumb += (double)d_in[idx + 2] * factor;
                    }
                }
            }
            unsigned int idx = i * w * ch + j * ch;
            d_out[idx] = ceil(sumr);
            d_out[idx + 1] = ceil(sumg);
            d_out[idx + 2] = ceil(sumb);
        }
    }
}

int main(int argc, char** argv){
    if(argc != 3){
        std::cout << "Please, provide the correct arguments\n./cudaBlur *blur level 1~9* *image-path*" << std::endl;
        return -1;
    }
    int blursize = argv[1][0] - 48;
    const char* path = argv[2];
    int w, h, ch;
    unsigned char* img = stbi_load(path, &w, &h, &ch, 0);
    unsigned char* res = new unsigned char[w*h*ch];
    unsigned char* d_in, *d_out;
    double V = pow(2, blursize * 2);
    double sum = V;
    double marginal = 1;
    double qtd = 0;
    for(int i = 1; i <= blursize; i++){
        qtd += 4;
        sum += V * qtd / pow(2, i);
        sum += marginal * qtd;
        marginal *= 2;
    }
    dim3 bNum(16, 16);
    dim3 tNum(16, 16);
    cudaMalloc((void**) &d_in, sizeof(unsigned char) * w * h * ch);
    cudaMalloc((void**) &d_out, sizeof(unsigned char) * w * h * ch);
    cudaMemcpy(d_in, img, sizeof(unsigned char) * w * h * ch, cudaMemcpyHostToDevice);
    blurImg<<<bNum, tNum>>>(d_in, d_out, w, h, ch, blursize, sum);
    cudaDeviceSynchronize();
    cudaMemcpy(res, d_out, sizeof(unsigned char) * w * h * ch, cudaMemcpyDeviceToHost);
    stbi_write_jpg("../images/output.jpg", w, h, ch, res, 100);
    cudaFree(d_in); cudaFree(d_out); delete res; stbi_image_free(img);
}