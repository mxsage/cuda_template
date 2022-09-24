// Addition include dirs: C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.6\Common;C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.6\bin\win64\Release;%(AdditionalIncludeDirectories)
// CUDA device options: compute_80,sm_80

// Windowing utilities
#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <SDL2/SDL_opengl.h>

// Image utilities
#include "tinyexr.h"

// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// CUDA utilities
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>

#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <time.h>

#include <utility>
#include <functional>
#include <queue>
#include <thread>
#include <future>
#include <list> 
#include <chrono>

#include <glm/gtc/random.hpp>

#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

float* image;

int width = 800;
int height = 800;

extern "C" void render(float* img, int w, int h);

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource, *cuda_vbo_resource; // CUDA Graphics Resource (to transfer PBO)

void initPixelBuffer() {
    if (pbo) {
        // unregister this buffer object from CUDA C
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

        // delete old buffer
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }

    // create pixel buffer object for display
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(float) * 4,
        0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(
        &cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // create texture for display
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA,
        GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

StopWatchInterface *timer = 0;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
char fps[256];
void computeFPS()
{
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "%3.1f fps", ifps);

        fpsCount = 0;

        fpsLimit = (int)std::max(1.f, ifps);
        sdkResetTimer(&timer);
    }
}

glm::vec2 mousePos{ -1000., -1000. };

void drawDensity()
{
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // copy from pbo to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glPushMatrix();
    //glScalef(.5f, .5f, 1.f);
    // draw textured quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(0, 0);
    glTexCoord2f(1, 0);
    glVertex2f(1, 0);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(0, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    glPopMatrix();
}


void display()
{
    sdkStartTimer(&timer);

    float *d_output;

    size_t num_bytes;
    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes, cuda_pbo_resource));
    render(d_output, width, height);
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
    drawDensity();
    

    sdkStopTimer(&timer);

    computeFPS();
}

bool saveExr(const char *file, float *data, unsigned int w, unsigned int h) {
    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 1;

    image.width = w;
    image.height = h;

    std::vector<float> images[1];
    images[0].resize(w * h);

    for (int i = 0; i < w * h; i++) {
        images[0][i] = data[i * 4];
    }

    float* image_ptr[1];
    image_ptr[0] = &(images[0].at(0));
    image.images = (unsigned char**)image_ptr;

    header.num_channels = 1;
    header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
    // Must be BGR(A) order, since most of EXR viewers expect this channel order.
    strncpy(header.channels[0].name, "Y", 255); header.channels[0].name[strlen("Y")] = '\0';

    header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
    header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
    for (int i = 0; i < header.num_channels; i++) {
        header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
    }

    const char* err;
    int ret = SaveEXRImageToFile(&image, &header, file, &err);
    if (ret != TINYEXR_SUCCESS) {
        fprintf(stderr, "Save EXR err: %s\n", err);
        return ret;
    }
    printf("Saved exr file. [ %s ] \n", file);

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);
}

void saveImage()
{
    size_t num_bytes;


    glDisable(GL_BLEND);
    // map PBO to get CUDA device pointer
    float *d_output;

    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes, cuda_pbo_resource));

    float *h_output = (float *)malloc(width * height * 4 * sizeof(float));
    checkCudaErrors(cudaMemcpy(h_output, d_output, width * height * 4 * sizeof(float),
        cudaMemcpyDeviceToHost));

    saveExr("images/hq_render.exr", h_output, width, height);

    free(h_output);
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    if (pbo)
    {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }

    checkCudaErrors(cudaProfilerStop());
}

int main(int argc, char* argv[])
{
    int32_t WindowFlags = SDL_WINDOW_OPENGL;

    SDL_Window* window = SDL_CreateWindow("CUDA MPM",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        width, height, WindowFlags);
    //SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN);
    SDL_GLContext Context = SDL_GL_CreateContext(window);
    glewInit();

    //initGL(&argc, argv);
    sdkCreateTimer(&timer);

    cudaSetDevice(1);

    //glutDisplayFunc(display);
    //glutKeyboardFunc(keyboard);
    //glutIdleFunc(idle);
    //glutMouseFunc(mouse);
    //glutMotionFunc(motion);
    //glutReshapeFunc(reshape);
    //glutCloseFunc(cleanup);

    initPixelBuffer();

    glViewport(0, 0, width, height);
    SDL_GetError();
    //glutMainLoop();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, (float)height / width, 0.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, width, height);

    bool running = true;
    bool fullscreen = false;
    bool mouseDown = false;
    while (running)
    {
        SDL_Event Event;
        while (SDL_PollEvent(&Event))
        {
            if (Event.type == SDL_KEYDOWN)
            {
                switch (Event.key.keysym.sym)
                {
                case SDLK_ESCAPE:
                    running = 0;
                    break;
                case 'f':
                    fullscreen = !fullscreen;
                    if (fullscreen)
                    {
                        SDL_SetWindowFullscreen(window, WindowFlags | SDL_WINDOW_FULLSCREEN);
                    }
                    else
                    {
                        SDL_SetWindowFullscreen(window, WindowFlags);
                    }
                    break;
                case 's':
                    saveImage();
                    break;
                default:
                    break;
                }
            }
            else if (Event.type == SDL_MOUSEBUTTONDOWN)
            {
                mouseDown = true;
            }
            else if (Event.type == SDL_MOUSEBUTTONUP)
            {
                mouseDown = false;
            }

            if (mouseDown)
            {
                mousePos.x = (float)Event.button.x / width;
                mousePos.y = 1. - (float)Event.button.y / height;
            }
            else
            {
                mousePos.x = -9999.;
                mousePos.y = -9999.;
            }

            if (Event.type == SDL_QUIT)
            {
                running = 0;
            }
        }

        display();

        SDL_SetWindowTitle(window, fps);

        SDL_GL_SwapWindow(window);
    }
    cleanup();
}
