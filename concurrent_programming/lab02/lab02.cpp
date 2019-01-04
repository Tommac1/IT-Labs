/* 
c program:
--------------------------------
1. draws Mandelbrot set for Fc(z)=z*z +c
using Mandelbrot algorithm ( boolean escape time )
-------------------------------         
2. technique of creating ppm file is  based on the code of Claudio Rocchini
http://en.wikipedia.org/wiki/Image:Color_complex_plot.jpg
create 24 bit color graphic file ,  portable pixmap file = PPM 
see http://en.wikipedia.org/wiki/Portable_pixmap
to see the file use external application ( graphic viewer)
*/

#include <stdio.h>
#include <math.h>
#include <vector>
#include <thread>
#include <mutex>

const int iXmax = 800; 
const int iYmax = 800;


const double CxMin=-2.5;
const double CxMax=1.5;
const double CyMin=-2.0;
const double CyMax=2.0;
const double EscapeRadius=2;
const int IterationMax=200;

double PixelWidth=(CxMax-CxMin)/iXmax;
double PixelHeight=(CyMax-CyMin)/iYmax;

const int MaxColorComponentValue=255; 

static unsigned char color[iYmax][iXmax][3];

std::mutex color_mutex;

int mask[4][3][3] = {{{0, -1, 0}, {-1, 4, -1}, {0, -1, 0}}, 
    {{0, 1, 0}, {1, -4, 1}, {0, 1, 0}},
    {{1, 2, 1}, {2, -8, 2}, {1, 2, 1}},
    {{-1, 2, -1}, {2, -4, 2}, {-1, 2, -1}}};

void write_line(int ymin, int ymax)
{
    /* screen ( integer) coordinate */
    int iX,iY;
    /* world ( double) coordinate = parameter plane*/
    double Cx,Cy;


    /* bail-out value , radius of circle ;  */
    double ER2=EscapeRadius*EscapeRadius;


    /* Z=Zx+Zy*i  ;   Z0 = 0 */
    double Zx, Zy;
    double Zx2, Zy2; /* Zx2=Zx*Zx;  Zy2=Zy*Zy  */
    /*  */
    int Iteration;

    for (iY=ymin; iY<=ymax; iY++)
    {
        Cy=CyMin + iY*PixelHeight;
        if (fabs(Cy)< PixelHeight/2) Cy=0.0; /* Main antenna */
        for (iX=0;iX<iXmax;iX++)
        {         
                Cx=CxMin + iX*PixelWidth;
                /* initial value of orbit = critical point Z= 0 */
                Zx=0.0;
                Zy=0.0;
                Zx2=Zx*Zx;
                Zy2=Zy*Zy;

                for (Iteration=0;Iteration<IterationMax && ((Zx2+Zy2)<ER2);Iteration++)
                {
                    Zy=2*Zx*Zy + Cy;
                    Zx=Zx2-Zy2 +Cx;
                    Zx2=Zx*Zx;
                    Zy2=Zy*Zy;
                };

                /* compute  pixel color (24 bit = 3 bytes) */
                if (Iteration==IterationMax)
                { /*  interior of Mandelbrot set = black */
                     color[iY][iX][0]=0;
                     color[iY][iX][1]=0;
                     color[iY][iX][2]=0;                           
                }
                else 
                { /* exterior of Mandelbrot set = white */
                     color[iY][iX][0]=255; /* Red*/
                     color[iY][iX][1]=255; /* Green */ 
                     color[iY][iX][2]=255; /* Blue */
                };
            }
    }

}

void apply_mask(int x, int y, int mask[3][3])
{
    color_mutex.lock();
    for (int i = y; i < iXmax / 4; ++i) {
        for (int j = x; j < iYmax / 4; ++j) {
            color[i][j][0] *= mask[i % 3][j % 3];
            color[i][j][1] *= mask[i % 3][j % 3];
            color[i][j][2] *= mask[i % 3][j % 3];
       }
    }
    color_mutex.unlock();
}

void thread_filter()
{
    for (int i = 0; i < iXmax; i += (iXmax / 4)) {
        for (int j = 0; j < iYmax; j += (iYmax / 4)) {
            apply_mask(i, j, mask[i]);
        }
    }
}

void filter()
{

    std::vector <std::thread> vt;

    for (int i = 0; i < 4; ++i) {
        vt.push_back(std::thread(thread_filter));
    }

    for (auto &thread : vt)
        if (thread.joinable())
            thread.join();

}

int main()
{
    /* color component ( R or G or B) is coded from 0 to 255 */
    /* it is 24 bit color RGB file */
    FILE * fp;
    char *filename = "new1.ppm";
    char *comment = "# ";/* comment should start with # */

    /*create new file,give it a name and open it in binary mode  */
    fp = fopen(filename,"wb"); /* b -  binary mode */

    /*write ASCII header to the file*/
    fprintf(fp,"P6\n %s\n %d\n %d\n %d\n",comment,iXmax,iYmax,MaxColorComponentValue);

    std::vector <std::thread> vt;

    for (int i = 0; i < 8; ++i) {
        vt.push_back(std::thread(write_line, i * 100, (i + 1) * 100 - 1));
    }

    for (auto &thread : vt)
        if (thread.joinable())
            thread.join();

    filter();

    /* write color to the file */
    fwrite(color, (800 * 800 * 3), 1, fp);

    fclose(fp);
    return 0;
}