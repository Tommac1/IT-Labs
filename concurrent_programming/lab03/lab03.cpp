#include <iostream>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>

// PPM OBJECT DELCARATION =============
class PPM {
public:
    std::string header;
    int width; 
    int height;
    int colors;
    char **data;

    PPM() { };
    ~PPM() { };

    PPM& operator =(PPM other) {
        width = other.width;
        height = other.height;
        colors = other.colors;

        data = new char*[height];

        for (int i = 0; i < height; ++i)
            data[i] = new char[width * 3];

        for (int i = 0; i < height; ++i)
            for (int j = 0; j < width * 3; ++j)
                data[i][j] = other.data[i][j];

        return *this;
    }

    friend std::ifstream& operator >>(std::ifstream &is, const PPM &other);
    friend std::ofstream& operator <<(std::ofstream &os, const PPM &other);
};

std::ifstream& operator >>(std::ifstream &inputStream, PPM &other)
{
    inputStream >> other.header;
    inputStream >> other.width >> other.height >> other.colors;
    inputStream.get(); // skip the trailing white space

    other.data = new char*[other.height];

    for (int i = 0; i < other.height; ++i)
        other.data[i] = new char[other.width * 3];

    for (int i = 0; i < other.height; ++i)
        inputStream.read(other.data[i], other.width * 3);

    return inputStream;
}

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

const int masks[4][3][3] = {{{0, 1, 0}, {1, -4, 1}, {0, 1, 0}},
    {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
    {{0, -1, 0}, {-1, 4, -1}, {0, -1, 0}},
    {{-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}}};
std::mutex mut;
int tile = 0;
PPM ppm_in;
PPM ppm_out;
int x_step = 0;
int y_step = 0;

// FUNCTIONS ==========================

void init_env()
{
    x_step = ppm_in.height / 4;
    y_step = ppm_in.width / 4 * 3;
    tile = 0;
}
 
int multiply(int h, int w, int id)
{
    int ret = 0;

    for (int i = h - 1; i <= h + 1; ++i) {
        for (int j = w - 3; j <= w + 3; j += 3) {
            ret += (ppm_in.data[i][j] * masks[id][i % 3][(j / 3) % 3]);
        }
    }

    // adjust to uint8
    ret = (ret < 0) ? (0) : ((ret > 255) ? (255) : (ret));

    return ret;
}

void mask_area(int id, int x, int y)
{
    // RED
    for (int i = x + 1; i < x + x_step - 1; ++i) {
        for (int j = y + 3; j < y + y_step - 3; j += 3) {
            ppm_out.data[i][j] = multiply(i, j, id);
        }
    }
    
    // GREEN
    for (int i = x + 1; i < x + x_step - 1; ++i) {
        for (int j = y + 4; j < y + y_step - 3; j += 3) {
            ppm_out.data[i][j] = multiply(i, j, id);
        }
    }

    // BLUE
    for (int i = x + 1; i < x + x_step - 1; ++i) {
        for (int j = y + 5; j < y + y_step - 3; j += 3) {
            ppm_out.data[i][j] = multiply(i, j, id);
        }
    }
}

void worker_job(int id)
{
    int x_area = 0;
    int y_area = 0;

//    std::this_thread::sleep_for(std::chrono::nanoseconds(id));

    while (tile < 16) {
        mut.lock();
        if (tile < 16) {
            x_area = tile / 4;
            y_area = tile % 4;

            std::cout   << "id: " << id 
                        << ", tile: " << tile 
                        << ", x = " << x_area * x_step 
                        << ", y = " << y_area * y_step 
                        << std::endl; 

            mask_area(id, (x_area * x_step), (y_area * y_step));

            tile++;
        }
        mut.unlock();

        // delay to prevent one thread unlock() 
        // and intantly lock() afterwards
        for (x_area = 0; x_area < 9000; ++x_area) ;
    }
}

void spawn_workers()
{
    std::vector <std::thread> vt;

    for (int i = 0; i < 4; ++i) {
        vt.push_back(std::thread(worker_job, i));
    }

    for (auto &i : vt)
        if (i.joinable())
            i.join();
}

int main()
{
    int ret = 0;
    std::ifstream file_in("vladi.ppm", std::fstream::in | std::fstream::binary);
    std::ofstream file_out("vladi_masked.ppm", std::fstream::binary);


    if (file_in.is_open()) {
        file_in >> ppm_in;

        ppm_out = ppm_in;

        init_env();
        spawn_workers();

        file_out << ppm_out;

        file_in.close();
    }
    else {
        std::cerr << "Error opening file_in." << std::endl;
        ret = 1;
    }

    return ret;

}
