// Bruno Armond Braga Nusp: 12542331
// Matheus Sanches Jurgensen Nusp: 12542199
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <png.h>

// Funcao que aplica a matriz de transformacao A
// ao pixel px = (r, g, b)
// (new_r, new_g, new_b)' = A * (r, g, b)'
__host__ __device__ void modify_pixel(png_bytep px, double *A) {
    double r = px[0] / 255.0;
    double g = px[1] / 255.0;
    double b = px[2] / 255.0;

    double new_r = A[0] * r + A[1] * g + A[2] * b;
    double new_g = A[3] * r + A[4] * g + A[5] * b;
    double new_b = A[6] * r + A[7] * g + A[8] * b;

    new_r = fmin(fmax(new_r, 0.0), 1.0);
    new_g = fmin(fmax(new_g, 0.0), 1.0);
    new_b = fmin(fmax(new_b, 0.0), 1.0);

    px[0] = (png_byte) round(new_r * 255.0);
    px[1] = (png_byte) round(new_g * 255.0);
    px[2] = (png_byte) round(new_b * 255.0);
}

// Altera a matiz (hue) de uma imagem sequencialmente
void modify_hue_seq(png_bytep image, int width, int height, double hue_diff) {
    double c = cos(2 * M_PI * hue_diff);
    double s = sin(2 * M_PI * hue_diff);
    double one_third = 1.0 / 3.0;
    double sqrt_third = sqrt(one_third);

    // Matriz A compoe as operacoes de
    // conversao de RGB para HSV, mudanca de hue,
    // e conversao de HSV de volta para RGB
    // (new_r, new_g, new_b)' = A * (r, g, b)'
    // https://stackoverflow.com/questions/8507885/shift-hue-of-an-rgb-color

    double a11 = c + one_third * (1.0 - c);
    double a12 = one_third * (1.0 - c) - sqrt_third * s;
    double a13 = one_third * (1.0 - c) + sqrt_third * s;
    double a21 = a13; double a22 = a11; double a23 = a12;
    double a31 = a12; double a32 = a13; double a33 = a11;

    double A[9] = {a11, a12, a13, a21, a22, a23, a31, a32, a33};

    for (int i = 0; i < height; i++) {
        png_bytep row = &(image[i * width * 3]);
        for (int j = 0; j < width; j++) {
            png_bytep px = &(row[j * 3]);
            modify_pixel(px, A);
        }
    }
}

// Funcao auxiliar para identificar erros CUDA
void checkErrors(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s [Erro CUDA: %s]\n",
                msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void calculates_A(double hue_diff, double *A) {
    double c = cos(2 * M_PI * hue_diff);
    double s = sin(2 * M_PI * hue_diff);
    double one_third = 1.0 / 3.0;
    double sqrt_third = sqrt(one_third);

    double a11 = c + one_third * (1.0 - c);
    double a12 = one_third * (1.0 - c) - sqrt_third * s;
    double a13 = one_third * (1.0 - c) + sqrt_third * s;

    A[0] = a11;
    A[1] = a12;
    A[2] = a13;
    A[4] = A[8] = A[0];
    A[5] = A[6] = A[1];
    A[3] = A[7] = A[2];
}

// Kernel CUDA para alteracao do hue
// Voce deve modificar essa funcao no EP3
__global__ void modify_hue_kernel(png_bytep d_image,
                                  int width,
                                  int height,
                                  double *A) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = (y * width + x) * 3;
        png_bytep px = &(d_image[index]);
        modify_pixel(px, A);
    }
}

// Altera a matiz (hue) de uma imagem em paralelo
void modify_hue(png_bytep h_image,
                int width,
                int height,
                size_t image_size,
                double hue_diff) {

    int size_A = 9 * sizeof(double); // matriz 3x3
    double *h_A = (double *) malloc(size_A); // alocando matriz A no host
    calculates_A(hue_diff, h_A);

    double *d_A;
    cudaMalloc(&d_A, size_A);
    checkErrors(cudaGetLastError(), "Alocacao da matriz A no device");

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    checkErrors(cudaGetLastError(), "Copia da matriz A para o device");

    png_bytep d_image;
    cudaMalloc((void**)&d_image, image_size);
    checkErrors(cudaGetLastError(), "Alocacao da imagem no device");

    cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
    checkErrors(cudaGetLastError(), "Copia da imagem para o device");

    // blocos por grid e threads por bloco
    dim3 dim_block(32, 32);
    dim3 dim_grid((width + dim_block.x - 1) / dim_block.x, (height + dim_block.y - 1) / dim_block.y);


    modify_hue_kernel<<<dim_grid, dim_block>>>(d_image, width, height, d_A);
    checkErrors(cudaGetLastError(), "Lançamento do kernel");

    cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);
    checkErrors(cudaGetLastError(), "Copia da imagem para o host");

    cudaFree(d_image);
    cudaFree(d_A);
    free(h_A);
}

// Le imagem png de um arquivo de entrada para a memoria
void read_png_image(const char *filename,
                    png_bytep *image,
                    int *width,
                    int *height,
                    size_t *image_size) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Erro ao ler o arquivo de entrada %s\n", filename);
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fprintf(stderr, "Erro ao criar PNG read struct \n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        fprintf(stderr, "Erro ao criar PNG info struct \n");
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Em caso de erro nas funcoes da libpng,
    // programa "pula" para este ponto de execucao
    if (setjmp(png_jmpbuf(png))) {
        fprintf(stderr, "Erro ao ler imagem PNG \n");
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    *width = png_get_image_width(png, info);
    *height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    // Verifica se imagem png possui o formato apropriado
    if ((color_type != PNG_COLOR_TYPE_RGB && color_type != PNG_COLOR_TYPE_GRAY)
        || bit_depth != 8) {
        printf("Formato PNG nao suportado, deve ser 8-bit RGB ou grayscale\n");
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_read_update_info(png, info);

    // Alocacao de memoria para imagem e ponteiros para as linhas
    *image_size = png_get_rowbytes(png, info) * (*height);
    *image = (png_bytep) malloc(*image_size);

    png_bytep *row_pointers = (png_bytep *) malloc(sizeof(png_bytep) * (*height));
    for (int i = 0; i < *height; i++) {
        row_pointers[i] = *image + i * png_get_rowbytes(png, info);
    }

    // Leitura da imagem para a memoria
    png_read_image(png, row_pointers);

    // Finalizacao da leitura
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    free(row_pointers);
}

// Escreve imagem png da memoria para um arquivo de saida
void write_png_image(const char *filename,
                     png_bytep image,
                     int width,
                     int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Erro ao criar o arquivo de saida %s\n", filename);
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fprintf(stderr, "Erro ao criar PNG write struct \n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        fprintf(stderr, "Erro ao criar PNG info struct.\n");
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Em caso de erro nas funcoes da libpng,
    // programa "pula" para este ponto de execucao
    if (setjmp(png_jmpbuf(png))) {
        printf("Erro ao escrever imagem PNG \n");
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return;
    }

    png_init_io(png, fp);

    // Configura o formato da imagem a ser criada
    png_set_IHDR(
        png, info, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT
    );

    png_write_info(png, info);

    // Criacao de ponteiros para as linhas
    png_bytep row_pointers[height];
    for (int i = 0; i < height; i++) {
        row_pointers[i] = &(image[i * width * 3]);
    }

    // Escrita da imagem a partir da memoria
    png_write_image(png, row_pointers);
    png_write_end(png, NULL);

    // Finalizacao da escrita
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

int main(int argc, char *argv[]) {
    png_bytep image;
    int width, height;
    size_t image_size;

    // Leitura e validacao dos parametros de entrada
    if (argc != 4) {
        printf("Uso: ./hue_modify <input_file> <output_file> <hue_diff>\n");
        printf("0.0 <= hue_diff <= 1.0\n");
        exit(EXIT_FAILURE);
    }

    double hue_diff;
    int ret = sscanf(argv[3], "%lf", &hue_diff);
    if (ret == 0 || ret == EOF) {
        fprintf(stderr, "Erro ao ler hue_diff\n");
        exit(EXIT_FAILURE);
    }

    if (hue_diff < 0.0 || hue_diff > 1.0) {
        fprintf(stderr, "hue_diff deve ser entre 0.0 e 1.0\n");
        exit(EXIT_FAILURE);
    }

    // Leitura da imagem para memoria
    read_png_image(argv[1], &image, &width, &height, &image_size);

    // Processamento da imagem (alteracao do hue)

    // Versao sequencial:
    //modify_hue_seq(image, width, height, hue_diff);

    // // Versao paralela
    modify_hue(image, width, height, image_size, hue_diff);

    // Escrita da imagem para arquivo
    write_png_image(argv[2], image, width, height);

    // Liberacao de memoria
    free(image);
    return 0;
}

