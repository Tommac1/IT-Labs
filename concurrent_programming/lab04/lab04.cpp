#include <iostream>
#include <fstream>
#include <thread>

#include <cmath>
#include <cstring>

// PPM OBJECT DELCARATION =============
class PPM {
public:
    std::string header;
    int width; 
    int height;
    int colors;
    char **data;

    PPM(int _w, int _h, int _c) { 
        width = _w;
        height = _h;
        colors = _c;

        data = new char*[height];

        for (int i = 0; i < height; ++i) {
            data[i] = new char[width * 3];
            std::memset(data[i], 0, sizeof(data[i]));
        }
    };

    ~PPM() { 

        for (int i = 0; i < height; ++i)
            delete[] data[i];

        delete[] data;
    };

    void draw_pixel(int x, int y) {
        this->data[x][y * 3] = 127;
        this->data[x][y * 3 + 1] = 127;
        this->data[x][y * 3 + 2] = 127;
    };

    friend std::ofstream& operator <<(std::ofstream &os, const PPM &other);
};

std::ofstream& operator <<(std::ofstream &outputStream, const PPM &other)
{
    outputStream << "P6"    << "\n"
        << other.width      << " "
        << other.height     << "\n"
        << other.colors     << "\n";

    for (int i = 0; i < other.height; ++i)
        outputStream.write(other.data[i], other.width * 3);

    return outputStream;
}

// GLOBAL DATA ==========================
const int MAX_LEVEL = 5; 
PPM ppm_out(513, 513, 3);
const int N = 512;

// FUNCTIONS ==========================
void draw_line(int x1, int y1, int x2, int y2)
{
    // Bresenham's line algorithm
    const bool steep = (abs(y2 - y1) > abs(x2 - x1));
    if (steep)
    {
        std::swap(x1, y1);
        std::swap(x2, y2);
    }

    if (x1 > x2)
    {
        std::swap(x1, x2);
        std::swap(y1, y2);
    }

    const int dx = x2 - x1;
    const int dy = abs(y2 - y1);

    double error = dx / 2.0f;
    const int y_step = (y1 < y2) ? 1 : -1;
    int y = y1;

    const int max_x = x2;

    for (int x = x1; x < max_x; x++)
    {
        if (steep)
            ppm_out.draw_pixel(x, y);
        else
            ppm_out.draw_pixel(y, x);

        error -= dy;
        if(error < 0)
        {
            y += y_step;
            error += dx;
        }
    }
}

void print_triangle(int x, int y, int size)
{
    draw_line(x, y, x, y + size);
    draw_line(x, y + size, x + size, y + size);
    draw_line(x, y, x + size, y + size);
}

void worker_job(int x, int y, int level)
{
    int size = N / pow(2, level);

    print_triangle(x, y, size);

    size >>= 1;
    if (level < MAX_LEVEL && size > 0) {
        std::thread t1(worker_job, x, y, level + 1);
        std::thread t2(worker_job, x, y + size, level + 1);
        std::thread t3(worker_job, x + size, y + size, level + 1);

        if (t1.joinable()) t1.join();
        if (t2.joinable()) t2.join();
        if (t3.joinable()) t3.join();
    }
}

int main()
{
    std::ofstream file_out("sierpinski_triangle.ppm", std::fstream::binary);

    std::thread t1(worker_job, 0, 0, 0);

    if (t1.joinable())
        t1.join();

    file_out << ppm_out;

    return 0;

}
