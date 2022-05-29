//
// Created by Ta3ik on 29.05.2022.
//

#ifndef ALEXNET_ADDITIONAL_METHODS_H
#define ALEXNET_ADDITIONAL_METHODS_H

#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cassert>
#include "stb_image.h"

typedef struct{
    int w;
    int h;
    int c;
    float *data;
} image;

static float get_pixel(image *img, unsigned int x, unsigned int y, unsigned int c)
{
    assert(x<(img->w) && y<(img->h) && c<(img->c));
    return img->data[c*img->w*img->h+y*img->w+x];
}

static void set_pixel(image *img, float value, unsigned int x, unsigned int y, unsigned int c)
{
    if (x < 0 || y < 0 || c < 0 || x >= (img->w) || y >= (img->h) || c >= (img->c)) return;
    assert(x<(img->w) && y<(img->h) && c<(img->c));
    img->data[c*(img->w)*(img->h)+y*(img->w)+x] = value;
}

static void add_pixel(image *img, float value, unsigned int x, unsigned int y, unsigned int c)
{
    assert(x<(img->w) && y<(img->h) && c<(img->c));
    img->data[c*(img->w)*(img->h)+y*(img->w)+x] += value;
}

void make_image(image *img, int w, int h, int c)
{
    /**
     * Make image
     *
     * Input:
     *      w, h, c
     * Output:
     *      img
     * */
    img->w = w;
    img->h = h;
    img->c = c;
    img->data = (float *)malloc(h*w*c * sizeof(float));
}

void free_image(image *img)
{
    if (img->data)
        free(img->data);
}

void resize_image(image *im, int w, int h)
{
    image resized, part;
    make_image(&resized, w, h, im->c);
    make_image(&part,  w, im->h, im->c);
    float   w_scale = (im->w-1) * 1.0 / (w-1),
            h_scale = (im->h-1) * 1.0 / (h-1);
     float val;
     unsigned int r, c, k;
    for (k = 0; k < im->c; ++k){
        for (r = 0; r < im->h; ++r){
            for (c = 0; c < w; ++c){
                val = 0;
                if (c == w-1 || im->w == 1) {
                    val = get_pixel(im, im->w-1, r, k);
                }else {
                    float sx = c * w_scale;
                    int   ix = (int)sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                }
                set_pixel(&part, val, c, r, k);
            }
        }
    }
    for (k = 0; k < im->c; ++k){
        for (r = 0; r < h; ++r){
            float sy = r * h_scale;
            int   iy = (int) sy;
            float dy = sy - iy;
            for (c = 0; c < w; ++c){
                val = (1-dy) * get_pixel(&part, c, iy, k);
                set_pixel(&resized, val, c, r, k);
            }
            if (r == h-1 || im->h == 1) continue;
            for (c = 0; c < w; ++c){
                val = dy * get_pixel(&part, c, iy+1, k);
                add_pixel(&resized, val, c, r, k);
            }
        }
    }

    memcpy(im, &resized, sizeof(image));
    free_image(&part);
    free_image(&resized);
}

void horizontal_flip(image *im)
{
    /**
     * Data argumention : horizontal flip
     * */
    float *tmp = (float *)malloc(im->w * im->h * im->c * sizeof(float));
    for (short z = 0; z < im->c; z++)
    {
        for (short y = 0; y < im->h; y++)
        {
            int st_idx = y * im->w + z * im->w * im->h;
            for (short x = 0; x < im->w; x++)
                tmp[st_idx+x] = im->data[st_idx+im->w-x];
        }
    }
    memcpy(im->data, tmp, im->w * im->h * im->c * sizeof(float));
    free(tmp);
}

image load_image(char *filename, int W, int H, int channels, int is_h_flip)
{
    /**
     * load image from file
     *
     * Input:
     *      filename
     *      channels
     *      is_h_filp   whether to apply horizontal flip
     * Return:
     *      image
     * */
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data)
    {
        printf("Error! Can't load image %s! \n", filename);
        exit(0);
    }
    if (channels)
    {
        c=channels;
    }
    image img;
    make_image(&img, w, h, c);
     int dst_idx, src_idx;
    for (int k = 0; k < c; k++)
    {
        for (int j = 0; j < h; j++)
        {
            for (int i = 0; i < w; i++)
            {
                dst_idx = i + w*j + w*h*k;
                src_idx = k + c*i + c*w*j;
                img.data[dst_idx] = (float)data[src_idx] / 127.5 - 1;
            }
        }
    }
    free(data);

    if ((h&&w) && (H!=img.h || W!=img.w))
        resize_image(&img, H, W);

    if (is_h_flip)
        horizontal_flip(&img);

    return img;
}

std::vector<float> getBrightness(int n, float *X, int *Y, int w, int h, int c, int CLASSES, FILE *fp){
    image img;
    make_image(&img, w, h, c);
    int label,idx;
    char imgpath[256];
    int imagesize = w*h*c;
    for(int i=0; i<n; i++)
    {
        if(feof(fp))
            rewind(fp);
        fscanf(fp, "%d %s", &label, imgpath);

        Y[i] = label;
        img = load_image(imgpath, w, h, c, 0);
        memcpy(X+i*imagesize, img.data, imagesize*sizeof(float));
    }
}


#endif //ALEXNET_ADDITIONAL_METHODS_H